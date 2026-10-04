#!/usr/bin/env python3
"""Calibrate DMD line position -> wavelength.

    python dmd_calibration.py               # reads dmd_calibration.json
    python dmd_calibration.py --simulate    # rehearse, no hardware

    source -> prism -> spectrum on the DMD -> prism (recombine) -> spectrometer

One line (line_width_rows mirror rows wide) is stepped across the DMD, from
offset_start to offset_stop in offset_step rows. At every position the line
sends a narrow band of the spectrum to the spectrometer; the script finds the
centre wavelength of that peak (peak_method) and its FWHM. A polynomial
wavelength(offset) of fit_order is fitted through the centres -- a prism's
dispersion is smooth but not linear -- and positions off it by more than
max_residual_nm are dropped.

Before the scan:
  * the DMD passes everything and you type an exposure: this is the brightest
    a single line can get at any pixel, so nothing clips during the scan;
  * pass-everything is compared with block-everything: if blocking is the
    brighter one, line_on is the wrong way round and the script stops.

A background is taken before and after the scan (the DMD blocking everything,
or the spectrometer shutter: `background`) and interpolated in time.

Output, in <data_dir>/calibrations/<date>_<time>/:
  dmd_calibration.csv     offset_rows -> wavelength_nm, used by dmd_measure.py
                          (it takes the newest calibration automatically)
  calibration_spectra.csv every background-subtracted spectrum, one column per offset
  calibration_run.json    settings, fit, summary
  calibration.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from dmd_common import (CAL_NAME, CAL_RUN, HERE, ask_exposure, clipped, data_root,
                        fill_report, find_peak, finish_figure, float_or_none, get_plt,
                        hardware_args, int_or_none, interp_background, make_rig,
                        new_folder, offset_limits, parse_config, stamp, str2bool, write_csv)


def check_exposure(r, a, band):
    """The DMD passes everything; you type an exposure and see how full the
    brightest pixel gets."""
    r.show_all(True)
    print("\nEXPOSURE: the DMD now passes the whole spectrum -- the brightest any "
          "single line can be.")
    ms = ask_exposure(a.exposure_ms)
    while True:
        r.set_exposure(ms)
        raw = r.snap()
        txt, clip = fill_report(raw, a, band)
        print(f"  {r.exposure_ms:g} ms x {r.hw_average} frames: {txt}")
        new = input("  Enter = keep this exposure, or type another (ms): ").strip()
        if new:
            try:
                ms = float(new)
                if not 0.01 <= ms <= 30000:
                    raise ValueError
            except ValueError:
                ms = ask_exposure()
            continue
        if clip and input("  it CLIPS: peaks may clip too. Keep it anyway? [y/N]: "
                          ).strip().lower() != "y":
            ms = ask_exposure()
            continue
        return


def check_direction(r, a, passing, band):
    """Stop if blocking everything gives more light than passing everything
    (at the starting exposure, before you are asked for one)."""
    r.show_all(False)
    blocked = r.snap()
    light = float((passing - blocked)[band].sum())
    ref = float(np.abs(blocked[band]).sum()) or 1.0
    if light < 0:
        raise SystemExit(f"\nBlocking everything gives MORE light than passing everything: "
                         f"line_on = {str(a.line_on).lower()} is the wrong way round. Set "
                         f"\"line_on\": {str(not a.line_on).lower()} in dmd_calibration.json.")
    if light < 0.01 * ref:
        if input("  hardly any light passes the DMD -- is the source on and aligned? "
                 "Continue anyway? [y/N]: ").strip().lower() != "y":
            raise SystemExit("stopped")


def scan(r, a, offsets):
    t_b0 = r.now()
    bg0 = r.background()
    raw, t = [], np.empty(len(offsets))
    print(f"\nScanning {len(offsets)} positions ({a.line_width_rows}-row line):")
    for k, off in enumerate(offsets):
        r.show_lines([off], a.line_width_rows)
        t[k] = r.now()
        raw.append(r.snap())
        done = int(30 * (k + 1) / len(offsets))
        print(f"\r  [{'#' * done}{'.' * (30 - done)}] {k + 1}/{len(offsets)}  "
              f"offset {off:+5d}", end="", flush=True)
    print()
    t_b1 = r.now()
    bg1 = r.background()
    raw = np.array(raw)
    net = raw - interp_background(t, t_b0, t_b1, bg0, bg1)
    return raw, net, t, (bg0, bg1, t_b0, t_b1)


def fit_mapping(off, wl, ok, order, max_resid):
    """Polynomial wavelength(offset) through the good centres; the worst point
    beyond max_resid is dropped and the fit redone, until none is left.
    Returns (polynomial or None, used mask)."""
    used = ok.copy()
    if order is None:
        return None, used
    while True:
        if used.sum() < order + 2:
            raise SystemExit(f"only {used.sum()} good positions -- too few for fit_order "
                             f"{order}. Check the light level, min_peak_counts and the "
                             f"offset range.")
        P = np.polynomial.Polynomial.fit(off[used], wl[used], order)
        res = np.where(used, np.abs(wl - P(off)), -1.0)
        if res.max() <= max_resid:
            return P, used
        used[int(np.argmax(res))] = False


def plot(a, wl_px, net, off, cen, fwhm, used, ok, P, lo_off, hi_off, run):
    plt = get_plt(a.show_plots)
    if plt is None:
        return
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    band = (wl_px >= a.wl_min_nm) & (wl_px <= a.wl_max_nm)
    cmap = plt.get_cmap("viridis")
    for k in range(len(off)):
        ax[0, 0].plot(wl_px[band], net[k, band], lw=0.8,
                      color=cmap(k / max(len(off) - 1, 1)))
    ax[0, 0].set(xlabel="wavelength (nm)", ylabel="counts - background",
                 title=f"every line position ({a.line_width_rows} rows wide)")

    ax[0, 1].plot(off[used], cen[used], "o", ms=4, label="used")
    if (ok & ~used).any():
        ax[0, 1].plot(off[ok & ~used], cen[ok & ~used], "x", color="C3", label="outlier")
    if P is not None:
        xx = np.linspace(lo_off, hi_off, 400)
        ax[0, 1].plot(xx, P(xx), "k-", lw=1, label=f"fit, order {a.fit_order}")
    ax[0, 1].set(xlabel="line offset (mirror rows)", ylabel="peak centre (nm)",
                 title="calibration")
    ax[0, 1].legend()

    if P is not None:
        ax[1, 0].axhline(0, color="k", lw=0.8)
        ax[1, 0].plot(off[used], cen[used] - P(off[used]), "o", ms=4)
        if (ok & ~used).any():
            ax[1, 0].plot(off[ok & ~used], cen[ok & ~used] - P(off[ok & ~used]), "x",
                          color="C3")
        ax[1, 0].set(xlabel="line offset (mirror rows)", ylabel="centre - fit (nm)",
                     title="fit residuals")
    else:
        ax[1, 0].set_axis_off()

    ax[1, 1].plot(cen[used], fwhm[used], "o", ms=4)
    ax[1, 1].set(xlabel="wavelength (nm)", ylabel="FWHM (nm)", title="width of the line's peak")
    fig.tight_layout()
    finish_figure(plt, fig, run / "calibration.png", a.show_plots)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    hardware_args(ap)
    g = ap.add_argument_group("calibration")
    g.add_argument("--offset-start", type=int, default=-900, dest="offset_start",
                   help="first line offset, mirror rows from the chip centre")
    g.add_argument("--offset-stop", type=int, default=900, dest="offset_stop")
    g.add_argument("--offset-step", type=int, default=20, dest="offset_step")
    g.add_argument("--line-width-rows", type=int, default=20, dest="line_width_rows")
    g.add_argument("--exposure-ms", type=float_or_none, default=None, dest="exposure_ms",
                   help="offered at the exposure prompt (Enter takes it)")
    g.add_argument("--wl-min-nm", type=float, default=400.0, dest="wl_min_nm",
                   help="peaks are looked for in wl_min_nm-wl_max_nm")
    g.add_argument("--wl-max-nm", type=float, default=900.0, dest="wl_max_nm")
    g.add_argument("--min-peak-counts", type=float, default=300.0, dest="min_peak_counts",
                   help="positions with a dimmer peak are not used")
    g.add_argument("--peak-method", choices=("centroid", "gaussian", "max"),
                   default="centroid", dest="peak_method")
    g.add_argument("--peak-threshold-rel", type=float, default=0.5,
                   dest="peak_threshold_rel",
                   help="pixels above this fraction of the peak give its centre")
    g.add_argument("--fit-order", type=int_or_none, default=3, dest="fit_order",
                   help="polynomial order of wavelength(offset); null = no fit, "
                        "straight lines between the measured centres")
    g.add_argument("--max-residual-nm", type=float, default=1.0, dest="max_residual_nm",
                   help="positions further than this from the fit are dropped")
    g.add_argument("--show-plots", type=str2bool, default=True, dest="show_plots")
    g.add_argument("--notes", default="")
    a = parse_config(ap, argv, HERE / "dmd_calibration.json")

    step = abs(a.offset_step) or 1
    sign = 1 if a.offset_stop >= a.offset_start else -1
    offsets = np.arange(a.offset_start, a.offset_stop + sign, sign * step)
    lo_lim, hi_lim = offset_limits(a, a.line_width_rows)
    on_chip = (offsets >= lo_lim) & (offsets <= hi_lim)
    if not on_chip.all():
        print(f"note: {np.sum(~on_chip)} offsets dropped -- a {a.line_width_rows}-row line "
              f"fits on the chip for offsets {lo_lim} to {hi_lim}")
    offsets = offsets[on_chip]
    if offsets.size < 3:
        raise SystemExit("fewer than 3 line positions -- check offset_start/stop/step")

    run_name = stamp()
    print(f"\nsaving to  : {data_root(a) / 'calibrations' / run_name}")
    print(f"scan       : offsets {offsets[0]} to {offsets[-1]} in {step}-row steps "
          f"({offsets.size} positions), line {a.line_width_rows} rows, "
          f"orientation {a.orientation:g} deg, line_on {str(a.line_on).lower()}")

    rig = make_rig(a)
    with rig as r:
        r.set_exposure(a.exposure_ms or 10.0)
        r.show_all(True)
        passing = r.snap()
        band = (r.wl >= a.wl_min_nm) & (r.wl <= a.wl_max_nm)
        check_direction(r, a, passing, band)
        check_exposure(r, a, band)
        raw, net, t, (bg0, bg1, t_b0, t_b1) = scan(r, a, offsets)
        wl_px = np.asarray(r.wl)
        hardware = r.describe()
        truth = ([r.true_centre(o, a.line_width_rows) for o in offsets]
                 if r.simulated else None)

    # ---- peaks ----------------------------------------------------------- #
    n = offsets.size
    cen, fwhm, peak = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    status = np.array(["ok"] * n, dtype=object)
    clip = clipped(raw[:, band], a.full_scale_counts)
    for k in range(n):
        p = find_peak(wl_px, net[k], a.wl_min_nm, a.wl_max_nm, a.peak_method,
                      a.peak_threshold_rel)
        if p is None:
            status[k] = "no pixels"
            continue
        cen[k], fwhm[k], peak[k] = p["centre_nm"], p["fwhm_nm"], p["peak"]
        if clip[k]:
            status[k] = "clipped"
        elif p["peak"] < a.min_peak_counts:
            status[k] = "dim"
        elif p["lo_nm"] <= wl_px[band][0] or p["hi_nm"] >= wl_px[band][-1]:
            status[k] = "at band edge"
    ok = status == "ok"

    P, used = fit_mapping(offsets.astype(float), cen, ok, a.fit_order, a.max_residual_nm)
    status[ok & ~used] = "outlier"
    if used.sum() < 2:
        raise SystemExit("fewer than 2 good positions -- nothing to calibrate")
    lo_off, hi_off = offsets[used].min(), offsets[used].max()
    inside = (offsets >= lo_off) & (offsets <= hi_off)
    if P is not None:
        wl_fit = np.where(inside, P(offsets), np.nan)
        resid = np.where(ok, cen - P(offsets), np.nan)
        dense = np.linspace(lo_off, hi_off, 2000)
        slope = P.deriv()(dense)
        monotonic = bool((slope > 0).all() or (slope < 0).all())
    else:
        wl_fit = np.where(used, cen, np.nan)
        resid = np.full(n, np.nan)
        d = np.diff(cen[used])
        monotonic = bool((d > 0).all() or (d < 0).all())

    # ---- report ---------------------------------------------------------- #
    print("\n offset   centre    FWHM     peak   status")
    for k in range(n):
        c = f"{cen[k]:8.2f}" if np.isfinite(cen[k]) else "       -"
        f = f"{fwhm[k]:6.2f}" if np.isfinite(fwhm[k]) else "     -"
        pk = f"{peak[k]:8.0f}" if np.isfinite(peak[k]) else "       -"
        print(f" {offsets[k]:+6d} {c} nm {f} nm {pk}   {status[k]}")
    used_wl = wl_fit[used]
    print(f"\n{used.sum()} of {n} positions used: {np.nanmin(used_wl):.1f}-"
          f"{np.nanmax(used_wl):.1f} nm, offsets {lo_off} to {hi_off}")
    summary = {"positions": int(n), "used": int(used.sum()),
               "wl_range_nm": [float(np.nanmin(used_wl)), float(np.nanmax(used_wl))],
               "offset_range_rows": [int(lo_off), int(hi_off)],
               "fwhm_median_nm": float(np.nanmedian(fwhm[used])), "monotonic": monotonic}
    if P is not None:
        r_used = resid[used]
        summary.update(fit_rms_nm=float(np.sqrt(np.mean(r_used ** 2))),
                       fit_max_nm=float(np.max(np.abs(r_used))))
        disp = [abs(float(P.deriv()(o))) for o in (lo_off, 0.5 * (lo_off + hi_off), hi_off)]
        summary["nm_per_row"] = disp
        print(f"  fit order {a.fit_order}: rms residual {summary['fit_rms_nm']:.3f} nm, "
              f"max {summary['fit_max_nm']:.3f} nm")
        print(f"  dispersion: {disp[0]:.3f} / {disp[1]:.3f} / {disp[2]:.3f} nm per row "
              f"at offsets {lo_off} / {0.5 * (lo_off + hi_off):.0f} / {hi_off}")
    print(f"  line FWHM: median {summary['fwhm_median_nm']:.2f} nm "
          f"({np.nanmin(fwhm[used]):.2f}-{np.nanmax(fwhm[used]):.2f})")
    if not monotonic:
        print("  WARNING: wavelength is NOT monotonic in the offset -- lower fit_order or "
              "narrow the offsets; dmd_measure.py will refuse this calibration")
    if truth is not None:
        err = (wl_fit - np.array(truth))[used]
        summary["sim_max_error_nm"] = float(np.max(np.abs(err)))
        print(f"  [sim] calibration vs true line centres: max {np.max(np.abs(err)):.3f} nm")
    for s in ("clipped", "dim", "outlier", "at band edge"):
        if (status == s).any():
            print(f"  {np.sum(status == s)} positions {s}")

    # ---- save ------------------------------------------------------------ #
    run = new_folder(data_root(a) / "calibrations", run_name)
    write_csv(run / CAL_NAME, {
        "offset_rows": offsets, "wavelength_nm": wl_fit, "measured_nm": cen,
        "fwhm_nm": fwhm, "peak_counts": peak, "residual_nm": resid,
        "used": used.astype(int), "status": status.astype(str)})
    cols = {"wavelength_nm": wl_px}
    cols.update({f"offset_{o:+d}": net[k] for k, o in enumerate(offsets)})
    write_csv(run / "calibration_spectra.csv", cols)
    write_csv(run / "background.csv", {"wavelength_nm": wl_px, "before": bg0, "after": bg1})
    meta = {"kind": "dmd_calibration", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "settings": vars(a), "hardware": hardware, "summary": summary,
            "fit": None if P is None else {
                "order": a.fit_order,
                "coef_low_to_high": [float(c) for c in P.convert().coef],
                "note": "wavelength_nm = sum(coef[i] * offset**i)"},
            "background_times_s": [t_b0, t_b1], "notes": a.notes}
    (run / CAL_RUN).write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nsaved: {run / CAL_NAME}")
    print("dmd_measure.py takes the newest calibration automatically "
          "(or set calibration_file in dmd_measure.json).")
    plot(a, wl_px, net, offsets.astype(float), cen, fwhm, used, ok, P, lo_off, hi_off, run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
