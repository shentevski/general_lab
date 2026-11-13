#!/usr/bin/env python3
"""Calibrate the HB source: stage angle -> wavelength, LED current -> peak amplitude.

    python calibration.py               # reads calibration.json
    python calibration.py --simulate    # rehearse, no hardware

    DC40 LED -> element on the PRMTZ8 rotation stage -> CCT10 spectrometer

The stage steps through the angles (angle_start_deg to angle_stop_deg in
angle_step_deg, or the list angles_deg). At every angle the LED steps through
the currents (current_count of them from current_start_ma to current_stop_ma,
or the list currents_ma) and a spectrum is recorded at each. Every spectrum
gives the centre of the band (peak_method), its FWHM, its peak amplitude
(counts above background) and the light in the band (counts x nm within
band_halfwidth_nm of the centre).

  wavelength map  the centres at reference_current_ma, fitted with a polynomial
                  wavelength(angle) of fit_order; angles further than
                  max_residual_nm from the fit are dropped
  amplitude map   peak amplitude against LED current at every angle

set_wavelength.py and constant_power_scan.py use both maps backwards.

Before the scan:
  * the LED warms up at the highest current (led_warmup_s);
  * prescan: the stage visits every angle once at the highest current, to find
    the brightest spectrum the calibration will see;
  * at that angle you type an exposure (Enter takes the one offered: the
    brightest pixel at target_fill of full scale). The whole calibration uses
    this one exposure, so all amplitudes are in the same counts --
    set_wavelength.py measures at it too.

A background (LED off, or the spectrometer shutter closed: `background`) is
taken before and after the scan and interpolated in time. At the end the first
angle is measured again (drift_check): how much the light drifted meanwhile.
Ctrl-C stops the scan; the angles finished so far are analysed and saved.

Output, in <data_dir>/calibrations/<date>_<time>/:
  hb_calibration.csv    one row per (angle, current): centre, FWHM, peak, band light
  wavelength_map.csv    one row per angle: measured centre, fit, residual
  spectra.npz           every raw spectrum, the backgrounds, wavelengths and times
  calibration_run.json  settings, exposure, fit, summary
  calibration.png
set_wavelength.py and constant_power_scan.py take the newest calibration
automatically (or set calibration_file in their JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from hb_common import (CAL_MAP, CAL_RUN, CAL_TABLE, HERE, ask_exposure, ask_yes,
                       band_counts, clipped, data_root, fill_report, find_peak,
                       finish_figure, float_list, float_or_none, get_plt, hardware_args,
                       int_or_none, interp_background, make_rig, new_folder, parse_config,
                       progress, stamp, steps, str2bool, unique_keep_order, write_csv)


# --------------------------------------------------------------------------- #
# what to measure
# --------------------------------------------------------------------------- #


def angle_list(a):
    if a.angles_deg:
        ang = unique_keep_order(a.angles_deg)
    else:
        ang = steps(a.angle_start_deg, a.angle_stop_deg, a.angle_step_deg, "angle_step_deg")
    if len(ang) < 3:
        raise SystemExit("fewer than 3 angles -- check angle_start/stop/step_deg or angles_deg")
    return np.asarray(ang, float)


def current_list(a, top):
    """The LED currents, mA, ascending; none above the LED limit `top`."""
    if a.currents_ma:
        cur = np.unique(np.round(np.asarray(a.currents_ma, float), 3))
    else:
        stop = top if a.current_stop_ma is None else float(a.current_stop_ma)
        if stop > top + 1e-9:
            print(f"note: current_stop_ma = {stop:g} is above the LED limit -- {top:g} mA used")
            stop = top
        if int(a.current_count) < 1:
            raise SystemExit("current_count must be at least 1")
        cur = np.unique(np.round(np.linspace(float(a.current_start_ma), stop,
                                             int(a.current_count)), 3))
    bad = (cur <= 0) | (cur > top + 1e-9)
    if bad.any():
        print(f"note: {cur[bad].tolist()} mA dropped -- currents must be above 0 and at most "
              f"the LED limit of {top:g} mA")
        cur = cur[~bad]
    if cur.size == 0:
        raise SystemExit("no LED current left to measure at")
    return cur


def reference_index(a, currents):
    """Index of the current the wavelength map is made at (null = the highest)."""
    if a.reference_current_ma is None:
        return currents.size - 1
    j = int(np.argmin(np.abs(currents - float(a.reference_current_ma))))
    if abs(currents[j] - float(a.reference_current_ma)) > 1e-6:
        print(f"note: reference_current_ma = {a.reference_current_ma:g} is not one of the "
              f"currents -- {currents[j]:g} mA used")
    return j


# --------------------------------------------------------------------------- #
# measuring
# --------------------------------------------------------------------------- #


def prescan(r, a, angles, band):
    """Every angle once at the highest current. Returns the light at each angle
    as counts per ms above the dark level (the median pixel), so angles compare
    even where the exposure had to be shortened to stop clipping."""
    print(f"\nPRESCAN: {angles.size} angles at the highest current")
    rate = np.zeros(angles.size)
    t0 = r.now()
    for k, ang in enumerate(angles):
        r.move_to(ang)
        while True:
            raw = r.snap()
            if clipped(raw[band], a.full_scale_counts).any() and r.exposure_ms > 0.011:
                r.set_exposure(max(r.exposure_ms / 4, 0.01))
                continue
            break
        rate[k] = (raw[band].max() - np.median(raw)) / r.exposure_ms
        progress(k + 1, angles.size, f"{ang:7.2f} deg", r.now() - t0)
    return rate


def check_exposure(r, a, band, offered):
    """At the brightest angle and current: you type an exposure and see how
    full the brightest pixel gets."""
    print("\nEXPOSURE: the LED is at the highest current, the stage at the brightest angle --")
    print("the brightest spectrum of the calibration. One exposure is used for all of it.")
    ms = ask_exposure(offered)
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
        if clip and not ask_yes("  it CLIPS: the brightest peaks would be lost. Keep it anyway?"):
            ms = ask_exposure()
            continue
        return raw


def scan(r, a, angles, currents):
    """Every current at every angle (current_settle after each change). Ctrl-C
    stops after the angle being measured; only finished angles are returned."""
    n_a, n_c = angles.size, currents.size
    raw = np.zeros((n_a, n_c, r.wl.size), np.float32)
    t = np.zeros((n_a, n_c))
    read = np.zeros(n_a)
    done, stopped = 0, False
    print(f"\nSCAN: {n_a} angles x {n_c} currents = {n_a * n_c} spectra (Ctrl-C stops)")
    t0 = r.now()
    try:
        for i, ang in enumerate(angles):
            r.set_current(currents[0])        # settles while the stage moves
            r.move_to(ang)
            read[i] = r.angle()
            for j, cur in enumerate(currents):
                if j:
                    r.set_current(cur)
                r.sleep(a.led_settle_s)
                t[i, j] = r.now()
                raw[i, j] = r.snap()
            done = i + 1
            progress(done, n_a, f"{ang:7.2f} deg", r.now() - t0)
    except KeyboardInterrupt:
        stopped = True
        print(f"\n  stopped (Ctrl-C) after {done} of {n_a} angles")
    return raw[:done].astype(float), t[:done], read[:done], stopped


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def peaks(a, wl, net, raw, band):
    """centre, FWHM, peak (counts) and status of every spectrum: arrays (angles, currents)."""
    n_a, n_c = net.shape[:2]
    cen, fwhm, peak = (np.full((n_a, n_c), np.nan) for _ in range(3))
    status = np.full((n_a, n_c), "ok", dtype=object)
    edge_lo, edge_hi = wl[band][0], wl[band][-1]
    for i in range(n_a):
        clip = clipped(raw[i][:, band], a.full_scale_counts)
        for j in range(n_c):
            p = find_peak(wl, net[i, j], a.wl_min_nm, a.wl_max_nm, a.peak_method,
                          a.peak_threshold_rel)
            if p is None:
                status[i, j] = "no pixels"
                continue
            cen[i, j], fwhm[i, j], peak[i, j] = p["centre_nm"], p["fwhm_nm"], p["peak"]
            if clip[j]:
                status[i, j] = "clipped"
            elif p["peak"] < a.min_peak_counts:
                status[i, j] = "dim"
            elif p["lo_nm"] <= edge_lo or p["hi_nm"] >= edge_hi:
                status[i, j] = "at band edge"
    return cen, fwhm, peak, status


def fit_map(ang, wl, ok, order, max_resid):
    """Polynomial wavelength(angle) through the good centres; the worst point
    beyond max_resid is dropped and the fit redone until none is left.
    Returns (polynomial or None, used mask)."""
    used = ok.copy()
    if order is None:
        return None, used
    while True:
        if used.sum() < order + 2:
            raise SystemExit(f"only {used.sum()} good angles -- too few for fit_order {order}. "
                             f"Check the light level (exposure, min_peak_counts), wl_min/max_nm "
                             f"and the angles.")
        P = np.polynomial.Polynomial.fit(ang[used], wl[used], order)
        res = np.where(used, np.abs(wl - P(ang)), -1.0)
        if res.max() <= max_resid:
            return P, used
        used[int(np.argmax(res))] = False


def plot(a, wl_px, net, angles, currents, ref_j, cen, peak, status, used, P, lo, hi, run):
    plt = get_plt(a.show_plots)
    if plt is None:
        return
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    band = (wl_px >= a.wl_min_nm) & (wl_px <= a.wl_max_nm)
    cmap = plt.get_cmap("viridis")
    n_a = angles.size
    for i in range(n_a):
        ax[0, 0].plot(wl_px[band], net[i, ref_j, band], lw=0.8,
                      color=cmap(i / max(n_a - 1, 1)))
    ax[0, 0].set(xlabel="wavelength (nm)", ylabel="counts - background",
                 title=f"every angle at {currents[ref_j]:g} mA")
    c_ok = cen[:, ref_j][np.isfinite(cen[:, ref_j])]
    if c_ok.size:
        pad = 40.0
        ax[0, 0].set_xlim(max(a.wl_min_nm, c_ok.min() - pad), min(a.wl_max_nm, c_ok.max() + pad))

    ok_ref = status[:, ref_j] == "ok"
    c_ref = cen[:, ref_j]
    ax[0, 1].plot(angles[used], c_ref[used], "o", ms=4, label="used")
    if (ok_ref & ~used).any():
        ax[0, 1].plot(angles[ok_ref & ~used], c_ref[ok_ref & ~used], "x", color="C3",
                      label="outlier")
    if P is not None:
        xx = np.linspace(lo, hi, 400)
        ax[0, 1].plot(xx, P(xx), "k-", lw=1, label=f"fit, order {a.fit_order}")
    ax[0, 1].set(xlabel="stage angle (deg)", ylabel="band centre (nm)", title="wavelength map")
    ax[0, 1].legend()

    if P is not None:
        ax[1, 0].axhline(0, color="k", lw=0.8)
        good = status == "ok"
        ii, jj = np.nonzero(good)
        others = jj != ref_j
        ax[1, 0].plot(angles[ii[others]], (cen[good] - P(angles[ii]))[others], ".",
                      color="0.6", ms=3, label="other currents")
        ax[1, 0].plot(angles[used], c_ref[used] - P(angles[used]), "o", ms=4,
                      label=f"{currents[ref_j]:g} mA (fitted)")
        ax[1, 0].set(xlabel="stage angle (deg)", ylabel="centre - fit (nm)",
                     title="fit residuals")
        ax[1, 0].legend()
    else:
        ax[1, 0].set_axis_off()

    wl_map = P(angles) if P is not None else c_ref
    order = np.argsort(wl_map)
    cm2 = plt.get_cmap("plasma")
    n_c = currents.size
    label_every = max(1, int(np.ceil(n_c / 6)))
    for j in range(n_c):
        y = np.where(np.isin(status[:, j], ["ok", "dim"]), peak[:, j], np.nan)
        ax[1, 1].plot(wl_map[order], y[order], "-o", ms=3, lw=1,
                      color=cm2(0.85 * j / max(n_c - 1, 1)),
                      label=f"{currents[j]:g} mA" if (j % label_every == 0 or j == n_c - 1)
                      else None)
    ax[1, 1].set(xlabel="wavelength (nm)", ylabel="peak (counts)",
                 title="peak amplitude at each LED current")
    ax[1, 1].legend(fontsize=9)
    fig.tight_layout()
    finish_figure(plt, fig, run / "calibration.png", a.show_plots)


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    hardware_args(ap)
    g = ap.add_argument_group("calibration")
    g.add_argument("--angle-start-deg", type=float, default=0.0, dest="angle_start_deg")
    g.add_argument("--angle-stop-deg", type=float, default=60.0, dest="angle_stop_deg")
    g.add_argument("--angle-step-deg", type=float, default=2.0, dest="angle_step_deg")
    g.add_argument("--angles-deg", type=float_list, default=None, dest="angles_deg",
                   help="explicit angles, e.g. 0,5,10 (overrides start/stop/step)")
    g.add_argument("--current-start-ma", type=float, default=100.0, dest="current_start_ma")
    g.add_argument("--current-stop-ma", type=float_or_none, default=None,
                   dest="current_stop_ma", help="null = the LED limit")
    g.add_argument("--current-count", type=int, default=10, dest="current_count",
                   help="currents, evenly spaced from start to stop")
    g.add_argument("--currents-ma", type=float_list, default=None, dest="currents_ma",
                   help="explicit currents, e.g. 100,200,400 (overrides start/stop/count)")
    g.add_argument("--reference-current-ma", type=float_or_none, default=None,
                   dest="reference_current_ma",
                   help="the wavelength map is made at this current; null = the highest")
    g.add_argument("--led-warmup-s", type=float, default=30.0, dest="led_warmup_s",
                   help="LED on at the highest current this long before the prescan")
    g.add_argument("--prescan", type=str2bool, default=True,
                   help="visit every angle once to find the brightest (else: the first angle)")
    g.add_argument("--target-fill", type=float, default=0.7, dest="target_fill",
                   help="offered exposure fills the brightest pixel to this fraction of full scale")
    g.add_argument("--exposure-ms", type=float_or_none, default=None, dest="exposure_ms",
                   help="offered at the exposure prompt instead (Enter takes it)")
    g.add_argument("--drift-check", type=str2bool, default=True, dest="drift_check",
                   help="measure the first angle again at the end")
    g.add_argument("--wl-min-nm", type=float, default=400.0, dest="wl_min_nm",
                   help="peaks are looked for in wl_min_nm-wl_max_nm")
    g.add_argument("--wl-max-nm", type=float, default=900.0, dest="wl_max_nm")
    g.add_argument("--min-peak-counts", type=float, default=300.0, dest="min_peak_counts",
                   help="dimmer spectra are not used for the wavelength map")
    g.add_argument("--peak-method", choices=("centroid", "gaussian", "max"),
                   default="centroid", dest="peak_method")
    g.add_argument("--peak-threshold-rel", type=float, default=0.5,
                   dest="peak_threshold_rel",
                   help="pixels above this fraction of the peak give its centre")
    g.add_argument("--band-halfwidth-nm", type=float_or_none, default=None,
                   dest="band_halfwidth_nm",
                   help="band light = counts x nm within this of the centre; "
                        "null = 1.5 x the median FWHM")
    g.add_argument("--fit-order", type=int_or_none, default=5, dest="fit_order",
                   help="polynomial order of wavelength(angle); null = no fit, "
                        "straight lines between the measured centres")
    g.add_argument("--max-residual-nm", type=float, default=0.5, dest="max_residual_nm",
                   help="angles further than this from the fit are dropped")
    g.add_argument("--show-plots", type=str2bool, default=True, dest="show_plots")
    g.add_argument("--notes", default="")
    a = parse_config(ap, argv, HERE / "calibration.json")
    a.angles_deg = float_list(a.angles_deg)
    a.currents_ma = float_list(a.currents_ma)
    if not 0.05 <= a.target_fill <= 0.98:
        raise SystemExit("target_fill must be between 0.05 and 0.98")

    angles = angle_list(a)
    run_name = stamp()
    print(f"\nsaving to  : {data_root(a) / 'calibrations' / run_name}")
    print(f"angles     : {angles[0]:g} to {angles[-1]:g} deg ({angles.size} angles)")

    rig = make_rig(a, spectrometer=True, power_meter=False)
    with rig as r:
        currents = current_list(a, r.led_max_ma)
        ref_j = reference_index(a, currents)
        print(f"currents   : {', '.join(f'{c:g}' for c in currents)} mA "
              f"(wavelength map at {currents[ref_j]:g} mA)")
        r.set_exposure(a.exposure_ms or 10.0)
        r.snap()                                       # learn the wavelength axis
        band = (r.wl >= a.wl_min_nm) & (r.wl <= a.wl_max_nm)
        if band.sum() < 10:
            raise SystemExit(f"wl_min_nm-wl_max_nm ({a.wl_min_nm:g}-{a.wl_max_nm:g}) is outside "
                             f"the spectrometer's {r.wl[0]:.0f}-{r.wl[-1]:.0f} nm")

        r.set_current(currents[-1])
        r.led_on()
        if a.led_warmup_s > 0:
            print(f"\nLED on at {currents[-1]:g} mA, warming up for {a.led_warmup_s:g} s")
            r.sleep(a.led_warmup_s)
        if a.prescan:
            rate = prescan(r, a, angles, band)
            k = int(np.argmax(rate))
            dark = float(np.median(r.snap()))
            offered = (a.target_fill * a.full_scale_counts - dark) / max(rate[k], 1e-9)
            offered = float(np.clip(float(f"{offered:.2g}"), 0.01, 30000))
            print(f"  brightest at {angles[k]:g} deg; {offered:g} ms puts its brightest pixel at "
                  f"~{100 * a.target_fill:.0f}% of full scale")
        else:
            k, offered = 0, 10.0
        r.move_to(angles[k])
        r.sleep(a.led_settle_s)
        check_exposure(r, a, band, a.exposure_ms or offered)

        t_b0 = r.now()
        bg0 = r.background()
        r.sleep(a.led_settle_s)
        brightest = r.snap() - bg0
        if brightest[band].max() < a.min_peak_counts:
            print(f"  the brightest spectrum peaks at only {brightest[band].max():.0f} counts above "
                  f"background (min_peak_counts = {a.min_peak_counts:g})")
            if not ask_yes("  hardly any light reaches the spectrometer -- is the LED on and "
                           "the beam aligned? Continue anyway?"):
                raise SystemExit("stopped")

        raw, t, read, stopped = scan(r, a, angles, currents)
        angles = angles[:raw.shape[0]]
        drift = None
        if a.drift_check and not stopped and angles.size >= 2:
            r.set_current(currents[ref_j])
            r.move_to(angles[0])
            r.sleep(a.led_settle_s)
            drift = (r.now(), r.snap())
        t_b1 = r.now()
        bg1 = r.background()
        wl_px = np.asarray(r.wl, float)
        hardware = r.describe()
        exposure_ms, hw_average = r.exposure_ms, r.hw_average
        truth = (np.array([r.true_centre(x, currents[ref_j], a) for x in angles])
                 if r.simulated else None)

    n_a, n_c = raw.shape[:2]
    if n_a < 3:
        raise SystemExit(f"only {n_a} angles finished -- too few to calibrate; nothing saved")

    # ---- save the raw data first (the analysis may still refuse) ----------- #
    run = new_folder(data_root(a) / "calibrations", run_name)
    np.savez_compressed(run / "spectra.npz", wavelength_nm=wl_px, angles_deg=angles,
                        currents_ma=currents, raw_counts=raw.astype(np.float32), time_s=t,
                        background_before=bg0, background_after=bg1,
                        background_time_s=np.array([t_b0, t_b1]),
                        exposure_ms=exposure_ms, hw_average=hw_average)

    # ---- peaks ------------------------------------------------------------- #
    bg = interp_background(t.ravel(), t_b0, t_b1, bg0, bg1).reshape(raw.shape)
    net = raw - bg
    cen, fwhm, peak, status = peaks(a, wl_px, net, raw, band)
    c_ref = cen[:, ref_j]
    ok_ref = status[:, ref_j] == "ok"

    P, used = fit_map(angles, c_ref, ok_ref, a.fit_order, a.max_residual_nm)
    if used.sum() < 2:
        raise SystemExit("fewer than 2 good angles -- nothing to calibrate")
    lo, hi = float(angles[used].min()), float(angles[used].max())
    inside = (angles >= lo) & (angles <= hi)
    if P is not None:
        wl_fit = np.where(inside, P(angles), np.nan)
        resid = np.where(ok_ref, c_ref - P(angles), np.nan)
        dense = np.linspace(lo, hi, 2000)
        slope = P.deriv()(dense)
        monotonic = bool((slope > 0).all() or (slope < 0).all())
    else:
        o = np.argsort(angles[used])
        wl_fit = np.where(inside, np.interp(angles, angles[used][o], c_ref[used][o]), np.nan)
        resid = np.full(n_a, np.nan)
        d = np.diff(c_ref[used][o])
        monotonic = bool((d > 0).all() or (d < 0).all())
    map_status = status[:, ref_j].copy()
    map_status[ok_ref & ~used] = "outlier"

    # light in the band, around the band centre at the reference current
    halfwidth = (float(a.band_halfwidth_nm) if a.band_halfwidth_nm
                 else 1.5 * float(np.nanmedian(fwhm[used, ref_j])))
    bandlight = np.full((n_a, n_c), np.nan)
    for i in range(n_a):
        for j in range(n_c):
            if status[i, j] in ("ok", "dim"):
                c0 = c_ref[i] if np.isfinite(c_ref[i]) else cen[i, j]
                bandlight[i, j] = band_counts(wl_px, net[i, j], c0, halfwidth)

    # how far the centre moves with the LED current, per angle
    shift = np.full(n_a, np.nan)
    for i in range(n_a):
        c = cen[i][status[i] == "ok"]
        if c.size >= 2:
            shift[i] = float(c.max() - c.min())

    # ---- report -------------------------------------------------------------- #
    print(f"\n   angle    centre  residual   FWHM   peak at {currents[ref_j]:g} mA   "
          f"shift with current   status")
    for i in range(n_a):
        cc = f"{c_ref[i]:8.2f}" if np.isfinite(c_ref[i]) else "       -"
        rr = f"{resid[i]:+7.3f}" if np.isfinite(resid[i]) else "      -"
        ff = f"{fwhm[i, ref_j]:6.2f}" if np.isfinite(fwhm[i, ref_j]) else "     -"
        pk = f"{peak[i, ref_j]:8.0f}" if np.isfinite(peak[i, ref_j]) else "       -"
        sh = f"{shift[i]:6.3f}" if np.isfinite(shift[i]) else "     -"
        print(f" {angles[i]:7.2f} {cc} nm {rr} nm {ff} nm {pk}           {sh} nm"
              f"          {map_status[i]}")
    used_wl = wl_fit[used]
    summary = {"angles": int(n_a), "currents": int(n_c), "used": int(used.sum()),
               "wl_range_nm": [float(np.nanmin(used_wl)), float(np.nanmax(used_wl))],
               "angle_range_deg": [lo, hi], "monotonic": monotonic,
               "fwhm_median_nm": float(np.nanmedian(fwhm[used, ref_j])),
               "centre_shift_with_current_max_nm": (float(np.nanmax(shift))
                                                    if np.isfinite(shift).any() else None),
               "peak_counts_at_top_current": [float(np.nanmin(peak[used, -1])),
                                              float(np.nanmax(peak[used, -1]))],
               "stopped_early": stopped}
    print(f"\n{used.sum()} of {n_a} angles used: {summary['wl_range_nm'][0]:.1f}-"
          f"{summary['wl_range_nm'][1]:.1f} nm at {lo:g} to {hi:g} deg")
    if P is not None:
        r_used = resid[used]
        summary.update(fit_rms_nm=float(np.sqrt(np.mean(r_used ** 2))),
                       fit_max_nm=float(np.max(np.abs(r_used))))
        mid = 0.5 * (lo + hi)
        disp = [abs(float(P.deriv()(x))) for x in (lo, mid, hi)]
        summary["nm_per_deg"] = disp
        print(f"  fit order {a.fit_order}: rms residual {summary['fit_rms_nm']:.3f} nm, "
              f"max {summary['fit_max_nm']:.3f} nm")
        print(f"  tuning: {disp[0]:.3f} / {disp[1]:.3f} / {disp[2]:.3f} nm per degree at "
              f"{lo:g} / {mid:g} / {hi:g} deg")
    print(f"  band FWHM: median {summary['fwhm_median_nm']:.2f} nm; band light summed over "
          f"+-{halfwidth:.1f} nm")
    if summary["centre_shift_with_current_max_nm"] is not None:
        print(f"  the centre moves by up to {summary['centre_shift_with_current_max_nm']:.3f} nm "
              f"between the lowest and highest current")
    print(f"  peak amplitude at {currents[-1]:g} mA: {summary['peak_counts_at_top_current'][0]:.0f}"
          f"-{summary['peak_counts_at_top_current'][1]:.0f} counts ({exposure_ms:g} ms x "
          f"{hw_average})")
    if not monotonic:
        print("  WARNING: the wavelength is NOT monotonic in the angle (normal incidence inside "
              "the range?) -- narrow the angles or lower fit_order; set_wavelength.py will "
              "refuse this calibration")
    if drift is not None:
        t_d, raw_d = drift
        net_d = raw_d - interp_background(t_d, t_b0, t_b1, bg0, bg1)[0]
        p_d = find_peak(wl_px, net_d, a.wl_min_nm, a.wl_max_nm, a.peak_method,
                        a.peak_threshold_rel)
        if p_d is not None and np.isfinite(peak[0, ref_j]) and peak[0, ref_j] > 0:
            minutes = (t_d - t[0, ref_j]) / 60.0
            drift = {"angle_deg": float(angles[0]), "current_ma": float(currents[ref_j]),
                     "minutes": minutes,
                     "peak_change_rel": p_d["peak"] / peak[0, ref_j] - 1,
                     "centre_change_nm": p_d["centre_nm"] - c_ref[0]}
            print(f"  drift check ({angles[0]:g} deg again, {minutes:.1f} min later): peak "
                  f"{100 * drift['peak_change_rel']:+.2f}%, centre "
                  f"{drift['centre_change_nm']:+.3f} nm")
        else:
            drift = None
    if truth is not None:
        err = (wl_fit - truth)[used]
        summary["sim_max_error_nm"] = float(np.max(np.abs(err)))
        print(f"  [sim] wavelength map vs true band centres: max {np.max(np.abs(err)):.3f} nm")
    for s in ("clipped", "dim", "outlier", "at band edge", "no pixels"):
        n_s = int(np.sum(status == s)) if s != "outlier" else int(np.sum(map_status == s))
        if n_s:
            print(f"  {n_s} {'angles' if s == 'outlier' else 'spectra'} {s}")

    # ---- save ------------------------------------------------------------------ #
    write_csv(run / CAL_TABLE, {
        "angle_deg": np.repeat(angles, n_c), "current_ma": np.tile(currents, n_a),
        "centre_nm": cen.ravel(), "fwhm_nm": fwhm.ravel(), "peak_counts": peak.ravel(),
        "band_counts": bandlight.ravel(), "time_s": t.ravel(),
        "status": status.ravel().astype(str)})
    write_csv(run / CAL_MAP, {
        "angle_deg": angles, "angle_read_deg": read, "wavelength_nm": wl_fit,
        "measured_nm": c_ref, "residual_nm": resid, "fwhm_nm": fwhm[:, ref_j],
        "peak_counts": peak[:, ref_j], "centre_shift_nm": shift,
        "used": used.astype(int), "status": map_status.astype(str)})
    o = np.argsort(angles[used])
    meta = {"kind": "hb_calibration", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "settings": vars(a), "hardware": hardware,
            "exposure_ms": exposure_ms, "hw_average": hw_average,
            "currents_ma": currents.tolist(), "reference_current_ma": float(currents[ref_j]),
            "band_halfwidth_nm": halfwidth,
            "map": {"order": a.fit_order if P is not None else None,
                    "coef_low_to_high": (None if P is None
                                         else [float(c) for c in P.convert().coef]),
                    "angle_range_deg": [lo, hi],
                    "measured_angles_deg": angles[used][o].tolist(),
                    "measured_nm": c_ref[used][o].tolist(),
                    "note": "wavelength_nm = sum(coef[i] * angle_deg**i); without a fit, "
                            "straight lines between the measured centres"},
            "background_time_s": [t_b0, t_b1], "drift_check": drift,
            "summary": summary, "notes": a.notes}
    (run / CAL_RUN).write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nsaved: {run}")
    print("set_wavelength.py and constant_power_scan.py take the newest calibration "
          "automatically (or set calibration_file in their JSON).")
    plot(a, wl_px, net, angles, currents, ref_j, cen, peak, status, used, P, lo, hi, run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped (Ctrl-C) -- the LED is switched off")
        sys.exit(130)
