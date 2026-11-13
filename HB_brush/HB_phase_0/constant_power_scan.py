#!/usr/bin/env python3
"""Scan the wavelength with the output power held constant (power meter).

    python constant_power_scan.py               # reads constant_power_scan.json
    python constant_power_scan.py --simulate    # rehearse, no hardware

    DC40 LED -> element on the PRMTZ8 rotation stage -> power meter (in the output beam)

The wavelengths (wl_start_nm to wl_stop_nm in wl_step_nm, or the list
wavelengths_nm; start/stop null = the whole calibrated range) are visited in
order. At each, the stage turns to the calibrated angle, the power meter is
told the wavelength (its sensor's response correction), and the LED current is
adjusted until the power is within power_tol_rel of the target:

  * the calibration knows how the light grows with the current at every angle
    (the light in the band, from the spectra), so only its scale in uW is
    unknown;
  * the first current uses the scale from the probe (or from the previous
    wavelength); every reading updates the scale here (max_iterations).

Before the scan:
  * the power meter is zeroed with the LED off (zero_pm);
  * the LED warms up at the highest current (led_warmup_s);
  * probe: every wavelength once at the highest current -- the most power each
    can give. The weakest limits a constant power: probe_margin of it is
    offered as the target (target_power_uw in the JSON is offered instead if
    set); you type the target, Enter takes the one offered.

The highest current is the calibration's highest (or the LED limit if lower):
the calibration is not extrapolated. A wavelength where even that falls short
is marked "unreachable". dwell_s: stay this long at each wavelength once it is
set (the power is read again at the end of the dwell). At the end the LED is
switched off and the power meter read once more (what is left with the LED
off: the zero's drift and room light).

Ctrl-C stops the scan; what was measured is saved.

Output, in <data_dir>/constant_power/<date>_<time>/:
  constant_power_scan.csv  wavelength, angle, current, power, error, status
  probe.csv                the power at the highest current, every wavelength
  scan_run.json            settings, target, calibration used, summary
  constant_power_scan.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from hb_common import (HERE, Calibration, ask_float, ask_yes, data_root, find_calibration,
                       finish_figure, float_list, float_or_none, get_plt, hardware_args,
                       make_rig, new_folder, parse_config, progress, stamp, steps, str2bool,
                       unique_keep_order, write_csv)


def wavelength_list(a, cal):
    if a.wavelengths_nm:
        wls = np.asarray(unique_keep_order(a.wavelengths_nm), float)
    else:
        step = abs(float(a.wl_step_nm))
        if step <= 0:
            raise SystemExit("wl_step_nm must be above 0")
        start = (float(a.wl_start_nm) if a.wl_start_nm is not None
                 else math.ceil(cal.wl_lo / step - 1e-9) * step)
        stop = (float(a.wl_stop_nm) if a.wl_stop_nm is not None
                else math.floor(cal.wl_hi / step + 1e-9) * step)
        wls = steps(start, stop, step, "wl_step_nm")
    inside = (wls >= cal.wl_lo - 1e-9) & (wls <= cal.wl_hi + 1e-9)
    if not inside.all():
        print(f"note: {wls[~inside].tolist()} nm dropped -- outside the calibrated "
              f"{cal.wl_lo:.1f}-{cal.wl_hi:.1f} nm")
    wls = wls[inside]
    if wls.size == 0:
        raise SystemExit("no wavelength left to scan")
    return wls


class Holder:
    """Sets the LED current that gives the target power at an angle.

    power = scale x band(angle, current), with band() from the calibration: its
    shape in the current is measured, only the scale (uW per unit of band
    light) is not. Each reading gives the scale at this angle and the next
    current from it."""

    def __init__(self, r, a, cal, top):
        self.r, self.a, self.cal, self.top = r, a, cal, top

    def band(self, angle, current):
        return self.cal.value_at(angle, current, "band")

    def current_for(self, angle, value):
        """The current giving band light `value` here (the top if beyond)."""
        if value >= self.band(angle, self.top):
            return self.top
        return min(self.cal.current_for(angle, value, "band"), self.top)

    def read_uw(self, n):
        p, sd = self.r.read_power_w(n)
        return 1e6 * p, 1e6 * sd

    def hold(self, angle, target, scale):
        a, r = self.a, self.r
        cur = self.current_for(angle, target / scale) if scale else 0.5 * self.top
        status, it = "not converged", 0
        p = sd = err = np.nan
        for it in range(1, a.max_iterations + 1):
            r.set_current(cur)
            r.sleep(a.led_settle_s)
            p, sd = self.read_uw(a.pm_readings)
            err = p / target - 1
            if abs(err) <= a.power_tol_rel:
                status = "ok"
                break
            if cur >= self.top - 1e-6 and p < target:
                status = "unreachable"
                break
            v = self.band(angle, cur)
            if p <= 1e-3 * target or v <= 0:
                status = "no light"
                break
            scale = p / v
            new = self.current_for(angle, target / scale)
            if abs(new - cur) < 1e-3:
                break                                   # cannot do better here
            cur = new
        return {"current_ma": cur, "power_uw": p, "power_std_uw": sd, "error_rel": err,
                "iterations": it, "status": status, "scale": scale}


def probe(r, a, h, wls, angles, pm_wl):
    print(f"\nPROBE: every wavelength at {h.top:g} mA -- the most power each can give")
    r.set_current(h.top)
    pmax = np.full(wls.size, np.nan)
    t0 = r.now()
    for k in range(wls.size):
        r.move_to(angles[k])
        r.pm_wavelength(pm_wl[k])
        pmax[k] = h.read_uw(a.pm_readings)[0]
        progress(k + 1, wls.size, f"{wls[k]:7.1f} nm  {pmax[k]:9.4g} uW", r.now() - t0)
    return pmax


def plot(a, res, target, pmax, wls_probe, run):
    plt = get_plt(a.show_plots)
    if plt is None:
        return
    fig, ax = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 2]})
    wl = np.asarray(res["wavelength_nm"])
    p = np.asarray(res["power_uw"])
    sd = np.asarray(res["power_std_uw"])
    st = np.asarray(res["status"])
    good = st == "ok"
    ax[0].axhline(target, color="C2", lw=1, label="target")
    if pmax is not None:
        ax[0].plot(wls_probe, pmax, "--", color="0.5", lw=1, label="highest current (probe)")
    ax[0].errorbar(wl[good], p[good], yerr=sd[good], fmt="o", ms=4, capsize=2, color="C0",
                   label="held")
    if (~good).any():
        ax[0].plot(wl[~good], p[~good], "x", color="C3", ms=7, label="not held")
    ax[0].set(ylabel="power (uW)", title=f"constant power scan, target {target:.4g} uW")
    ax[0].legend(fontsize=10)

    tol = 100 * a.power_tol_rel
    ax[1].axhspan(-tol, tol, color="C2", alpha=0.15, lw=0, label=f"+-{tol:g}%")
    ax[1].axhline(0, color="C2", lw=1)
    dev = 100 * (p / target - 1)
    ax[1].errorbar(wl[good], dev[good], yerr=100 * sd[good] / target, fmt="o", ms=4,
                   capsize=2, color="C0")
    if (~good).any():
        ax[1].plot(wl[~good], dev[~good], "x", color="C3", ms=7)
    lim = max(3 * tol, 1.2 * np.nanmax(np.abs(dev[good]))) if good.any() else 3 * tol
    ax[1].set_ylim(-lim, lim)
    ax[1].set(ylabel="power - target (%)")
    ax[1].legend(fontsize=10, loc="upper right")

    ax[2].plot(wl, res["current_ma"], "o-", ms=4, lw=1)
    ax[2].set(xlabel="wavelength (nm)", ylabel="LED current (mA)")
    fig.tight_layout()
    finish_figure(plt, fig, run / "constant_power_scan.png", a.show_plots)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    hardware_args(ap)
    g = ap.add_argument_group("constant power scan")
    g.add_argument("--calibration-file", default=None, dest="calibration_file",
                   help="a calibration folder (or a file in it); null = the newest")
    g.add_argument("--pm-serial", default=None, dest="pm_serial",
                   help="power meter serial; null = the first found")
    g.add_argument("--pm-average-count", type=int, default=100, dest="pm_average_count",
                   help="samples the power meter averages per reading")
    g.add_argument("--pm-readings", type=int, default=5, dest="pm_readings",
                   help="readings averaged per power value")
    g.add_argument("--zero-pm", type=str2bool, default=True, dest="zero_pm",
                   help="zero the power meter with the LED off before the scan")
    g.add_argument("--wl-start-nm", type=float_or_none, default=None, dest="wl_start_nm")
    g.add_argument("--wl-stop-nm", type=float_or_none, default=None, dest="wl_stop_nm")
    g.add_argument("--wl-step-nm", type=float, default=5.0, dest="wl_step_nm")
    g.add_argument("--wavelengths-nm", type=float_list, default=None, dest="wavelengths_nm",
                   help="explicit wavelengths, e.g. 610,620,650 (overrides start/stop/step)")
    g.add_argument("--target-power-uw", type=float_or_none, default=None,
                   dest="target_power_uw", help="offered at the target prompt")
    g.add_argument("--probe", type=str2bool, default=True,
                   help="every wavelength at the highest current first")
    g.add_argument("--probe-margin", type=float, default=0.9, dest="probe_margin",
                   help="offered target = this x the weakest wavelength's highest power")
    g.add_argument("--power-tol-rel", type=float, default=0.01, dest="power_tol_rel")
    g.add_argument("--max-iterations", type=int, default=6, dest="max_iterations")
    g.add_argument("--led-warmup-s", type=float, default=30.0, dest="led_warmup_s")
    g.add_argument("--dwell-s", type=float, default=0.0, dest="dwell_s",
                   help="stay this long at each wavelength once it is set")
    g.add_argument("--show-plots", type=str2bool, default=True, dest="show_plots")
    g.add_argument("--notes", default="")
    a = parse_config(ap, argv, HERE / "constant_power_scan.json", HERE / "calibration.json")
    a.wavelengths_nm = float_list(a.wavelengths_nm)

    cal = Calibration(find_calibration(a, a.calibration_file))
    cal.check(a)
    print(f"calibration: {cal.describe()}")
    wls = wavelength_list(a, cal)
    angles = np.array([cal.angle_for(x) for x in wls])
    print(f"scan       : {wls.size} wavelengths, {wls[0]:g} to {wls[-1]:g} nm")
    run_name = stamp()

    rig = make_rig(a, spectrometer=False, power_meter=True)
    with rig as r:
        top = min(float(cal.currents.max()), float(r.led_max_ma))
        h = Holder(r, a, cal, top)
        lo_pm, hi_pm = r.pm_wavelength_range()
        pm_wl = np.clip(wls, lo_pm, hi_pm)
        if (pm_wl != wls).any():
            print(f"  WARNING: the power meter's sensor covers {lo_pm:g}-{hi_pm:g} nm -- "
                  f"{wls[pm_wl != wls].tolist()} nm are read with its correction at the edge")
        if a.zero_pm:
            r.led_off()
            r.sleep(0.5)
            print("\nzeroing the power meter (LED off) ...")
            r.zero_pm()
            print(f"  LED off now reads {h.read_uw(a.pm_readings)[0]:.4g} uW")
        r.set_current(top)
        r.led_on()
        if a.led_warmup_s > 0:
            print(f"\nLED on at {top:g} mA, warming up for {a.led_warmup_s:g} s")
            r.sleep(a.led_warmup_s)

        pmax = scales = None
        offered = a.target_power_uw
        if a.probe:
            pmax = probe(r, a, h, wls, angles, pm_wl)
            k = int(np.nanargmin(pmax))
            j = int(np.nanargmax(pmax))
            print(f"  weakest {pmax[k]:.4g} uW at {wls[k]:g} nm, strongest {pmax[j]:.4g} uW "
                  f"at {wls[j]:g} nm")
            band_top = np.array([h.band(x, top) for x in angles])
            scales = np.where(band_top > 0, pmax / band_top, np.nan)
            if offered is None:
                offered = float(f"{a.probe_margin * pmax[k]:.3g}")
        print()
        target = ask_float("target power (uW)", offered, 1e-6, None)
        if pmax is not None:
            short = pmax < target * (1 - a.power_tol_rel)
            if short.any():
                print(f"  {int(short.sum())} wavelengths cannot reach {target:g} uW even at "
                      f"{top:g} mA: {wls[short].tolist()} nm")
                if not ask_yes("  scan anyway (they are marked unreachable)?", True):
                    raise SystemExit("stopped")

        run = new_folder(data_root(a) / "constant_power", run_name)
        print(f"\nsaving to  : {run}")
        if pmax is not None:
            write_csv(run / "probe.csv", {"wavelength_nm": wls, "angle_deg": angles,
                                          "current_ma": np.full(wls.size, top),
                                          "power_uw": pmax})
        cols = ("wavelength_nm", "angle_deg", "current_ma", "led_measured_ma", "power_uw",
                "power_std_uw", "error_rel", "power_end_uw", "iterations", "status", "time_s")
        res = {c: [] for c in cols}
        print(f"\nSCAN at {target:g} uW, within {100 * a.power_tol_rel:g}% (Ctrl-C stops)")
        print("  wavelength    angle   current      power     error  it  status")
        scale = None
        stopped = False
        t0 = r.now()
        try:
            for k in range(wls.size):
                r.move_to(angles[k])
                r.pm_wavelength(pm_wl[k])
                if scales is not None and np.isfinite(scales[k]):
                    scale = scales[k]
                out = h.hold(angles[k], target, scale)
                if out["status"] in ("ok", "not converged"):
                    scale = out["scale"]
                p_end = np.nan
                if a.dwell_s > 0:
                    r.sleep(a.dwell_s)
                    p_end = h.read_uw(a.pm_readings)[0]
                row = {"wavelength_nm": wls[k], "angle_deg": angles[k],
                       "current_ma": out["current_ma"],
                       "led_measured_ma": r.measure_current_ma(),
                       "power_uw": out["power_uw"], "power_std_uw": out["power_std_uw"],
                       "error_rel": out["error_rel"], "power_end_uw": p_end,
                       "iterations": out["iterations"], "status": out["status"],
                       "time_s": r.now() - t0}
                for c in cols:
                    res[c].append(row[c])
                write_csv(run / "constant_power_scan.csv", res)      # after every point
                print(f"  {wls[k]:8.2f} nm {angles[k]:8.3f} {out['current_ma']:7.1f} mA "
                      f"{out['power_uw']:9.4g} uW {100 * out['error_rel']:+6.2f}% "
                      f"{out['iterations']:3d}  {out['status']}")
        except KeyboardInterrupt:
            stopped = True
            print(f"\n  stopped (Ctrl-C) after {len(res['status'])} of {wls.size} wavelengths")
        r.led_off()
        r.sleep(0.5)
        dark = h.read_uw(a.pm_readings)[0]
        hardware = r.describe()

    # ---- summary ---------------------------------------------------------------- #
    st = np.asarray(res["status"])
    p = np.asarray(res["power_uw"], float)
    ok = st == "ok"
    summary = {"target_uw": target, "wavelengths": int(wls.size), "measured": int(st.size),
               "stopped_early": stopped, "top_current_ma": top, "led_off_end_uw": dark}
    for s in ("ok", "unreachable", "not converged", "no light"):
        summary[s.replace(" ", "_")] = int(np.sum(st == s))
    print(f"\n{int(ok.sum())} of {st.size} wavelengths held at {target:g} uW "
          f"+-{100 * a.power_tol_rel:g}%")
    if ok.any():
        summary.update(power_mean_uw=float(p[ok].mean()),
                       power_spread_rel=float(p[ok].std() / target),
                       error_max_rel=float(np.max(np.abs(p[ok] / target - 1))),
                       iterations_mean=float(np.mean(np.asarray(res["iterations"])[ok])))
        print(f"  held points: mean {summary['power_mean_uw']:.4g} uW, spread "
              f"{100 * summary['power_spread_rel']:.2f}%, worst {100 * summary['error_max_rel']:.2f}%"
              f", {summary['iterations_mean']:.1f} readings per wavelength")
    for s in ("unreachable", "not converged", "no light"):
        if (st == s).any():
            print(f"  {s}: {np.asarray(res['wavelength_nm'])[st == s].tolist()} nm")
    print(f"  LED off at the end: {dark:.4g} uW ({100 * dark / target:+.2f}% of the target)")
    if a.dwell_s > 0 and ok.any():
        pe = np.asarray(res["power_end_uw"], float)[ok]
        summary["dwell_change_max_rel"] = float(np.nanmax(np.abs(pe / p[ok] - 1)))
        print(f"  after {a.dwell_s:g} s dwell: power changed by up to "
              f"{100 * summary['dwell_change_max_rel']:.2f}%")

    meta = {"kind": "hb_constant_power_scan", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "settings": vars(a), "hardware": hardware, "calibration": str(cal.folder),
            "target_uw": target, "summary": summary, "notes": a.notes}
    (run / "scan_run.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nsaved: {run}")
    if st.size:
        plot(a, res, target, pmax, wls, run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped (Ctrl-C) -- the LED is switched off")
        sys.exit(130)
