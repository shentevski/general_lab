#!/usr/bin/env python3
"""Set the HB source to a wavelength and a peak amplitude, from the calibration.

    python set_wavelength.py                     # asks, again and again
    python set_wavelength.py --wavelength-nm 650 --amplitude-counts 20000   # once
    python set_wavelength.py --simulate          # rehearse, no hardware

    DC40 LED -> element on the PRMTZ8 rotation stage -> CCT10 spectrometer

You type a wavelength: the stage turns to the angle the calibration gives for
it. You type a peak amplitude -- counts above background at the calibration's
exposure, the same counts as peak_counts in hb_calibration.csv: the LED current
is set to what the calibration gives for that amplitude at this angle
(interpolated between the two nearest calibrated angles). The amplitude must
be within what was calibrated there (no extrapolation); 0 switches the LED off.

A spectrum at the calibration's exposure then checks the result (verify). With
refine_iterations > 0 the angle is nudged by the wavelength error (using the
local nm per degree) and the current by the amplitude error, until both are
within wl_tolerance_nm and amplitude_tolerance_rel. The background for these
spectra is taken at the start and again before a new setting once it is older
than background_refresh_s (with the LED off, or the shutter closed: as in
calibration.json).

The shared hardware settings come from calibration.json. Every setting is
logged in <data_dir>/set_wavelength/<date>_<time>/log.csv, its spectrum in
spectra.npz. When you quit, the LED stays on at the last setting if you say so
(leave_led_on is the answer offered; a one-shot run uses it directly).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from hb_common import (HERE, Calibration, append_csv, ask_yes, data_root, find_calibration,
                       find_peak, float_or_none, hardware_args, new_folder, parse_config,
                       make_rig, stamp, str2bool)


def ask(prompt):
    """A line typed by the user ('' on Enter, None when the input has ended)."""
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return None


class Session:
    """The log of one run: log.csv (one row per setting) and spectra.npz.
    The folder is made at the first setting."""

    def __init__(self, a, cal):
        self.a, self.cal = a, cal
        self.name = stamp()
        self.folder = None
        self.spectra, self.wl = [], None

    def _open(self):
        if self.folder is None:
            self.folder = new_folder(data_root(self.a) / "set_wavelength", self.name)
            meta = {"kind": "hb_set_wavelength", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "settings": vars(self.a), "calibration": str(self.cal.folder),
                    "exposure_ms": self.cal.exposure_ms, "hw_average": self.cal.hw_average}
            (self.folder / "session.json").write_text(json.dumps(meta, indent=2, default=str))
            print(f"  (logging to {self.folder})")
        return self.folder

    def log(self, row, wl=None, net=None):
        folder = self._open()
        append_csv(folder / "log.csv", row)
        if net is not None:
            self.wl = wl
            self.spectra.append(np.asarray(net, np.float32))
            np.savez_compressed(folder / "spectra.npz", wavelength_nm=self.wl,
                                net_counts=np.array(self.spectra),
                                note="one row per verified setting (log.csv rows with a "
                                     "measured centre), counts minus background")


class Source:
    """The stage + LED, set from the calibration and checked with the spectrometer."""

    def __init__(self, r, a, cal):
        self.r, self.a, self.cal = r, a, cal
        s = cal.settings
        self.win = (float(s.get("wl_min_nm", 200.0)), float(s.get("wl_max_nm", 1000.0)))
        self.method = s.get("peak_method", "centroid")
        self.rel = float(s.get("peak_threshold_rel", 0.5))
        self.fwhm = float(cal.run.get("summary", {}).get("fwhm_median_nm") or 10.0)
        self.top = min(float(cal.currents.max()), float(r.led_max_ma))
        self.bg, self.t_bg = None, None
        self.angle = self.current = None

    # -- calibration ------------------------------------------------------------
    def max_amplitude(self, angle):
        """The highest peak amplitude the calibration reaches here (at self.top)."""
        return min(self.cal.value_range(angle)[1], self.cal.value_at(angle, self.top))

    def plan(self, wl, amp):
        """(angle, current) for a wavelength and amplitude; ValueError if outside
        the calibration."""
        angle = self.cal.angle_for(wl)
        if amp <= 0:
            return angle, 0.0
        top_amp = self.max_amplitude(angle)
        if amp > top_amp * (1 + 1e-9):
            raise ValueError(f"{amp:.0f} counts is above the {top_amp:.0f} the calibration "
                             f"reaches at {wl:g} nm")
        return angle, min(self.cal.current_for(angle, amp), self.top)

    # -- hardware -----------------------------------------------------------------
    def background(self):
        r = self.r
        print("  background ...", end="", flush=True)
        was_on = r.led_is_on
        self.bg = r.background()
        self.t_bg = r.now()
        if was_on:
            r.sleep(self.a.led_settle_s)
        print(" done")

    def go(self, angle, current):
        r = self.r
        if current <= 0:
            r.led_off()
            r.move_to(angle)
        else:
            r.set_current(current)
            r.move_to(angle)
            if not r.led_is_on:
                r.led_on()
            r.sleep(self.a.led_settle_s)
        self.angle, self.current = angle, current

    def measure(self):
        """(peak dict or None, net spectrum)."""
        raw = self.r.snap()
        net = raw - self.bg
        return find_peak(self.r.wl, net, self.win[0], self.win[1], self.method, self.rel), net

    def slope(self, angle):
        """d wavelength / d angle (nm per degree, signed) from the calibration."""
        lo, hi = self.cal.angle_lo, self.cal.angle_hi
        x0, x1 = max(angle - 0.05, lo), min(angle + 0.05, hi)
        return float(self.cal.wavelength_at(x1) - self.cal.wavelength_at(x0)) / (x1 - x0)


def apply(src, a, wl, amp, session):
    """Set one wavelength + amplitude, check it, refine it, log it."""
    r, cal = src.r, src.cal
    angle, cur = src.plan(wl, amp)
    angle_cal, cur_cal = angle, cur
    if a.verify and (src.bg is None or r.now() - src.t_bg > a.background_refresh_s):
        src.background()
    src.go(angle, cur)
    if cur <= 0:
        print(f"  -> {angle:.3f} deg, LED off")
        session.log(_row(wl, amp, angle, cur, angle_cal, cur_cal, r))
        return
    print(f"  -> {angle:.3f} deg, {cur:.1f} mA")
    if not a.verify:
        session.log(_row(wl, amp, angle, cur, angle_cal, cur_cal, r))
        return

    p = net = None
    within, n_ref = None, 0
    for it in range(a.refine_iterations + 1):
        p, net = src.measure()
        if p is None:
            print("  measured: no pixels in the calibration's wavelength window")
            break
        dl = p["centre_nm"] - wl
        da = p["peak"] / amp - 1
        print(f"  measured: {p['centre_nm']:.2f} nm ({dl:+.2f}), peak {p['peak']:.0f} counts "
              f"({100 * da:+.1f}%), FWHM {p['fwhm_nm']:.1f} nm")
        within = abs(dl) <= a.wl_tolerance_nm and abs(da) <= a.amplitude_tolerance_rel
        if within or it == a.refine_iterations:
            break
        # plausible enough to steer by?
        expected = cal.value_at(src.angle, src.current)
        slope = src.slope(src.angle)
        if (abs(dl) > max(2 * src.fwhm, 5.0) or not 0.3 <= p["peak"] / expected <= 3
                or abs(slope) < 1e-3):
            print("  the reading is far from what the calibration expects -- not refining "
                  "(spectrometer in the beam? stage homed? light path changed?)")
            break
        new_angle = src.angle
        if abs(dl) > a.wl_tolerance_nm:
            new_angle = float(np.clip(src.angle - dl / slope, cal.angle_lo, cal.angle_hi))
        new_cur = src.current
        if abs(da) > a.amplitude_tolerance_rel:
            scale = p["peak"] / expected                  # brighter / dimmer than calibrated
            want = amp / scale
            top_amp = src.max_amplitude(new_angle)
            if want >= top_amp:
                new_cur = src.top
                print(f"  (the source is {100 * (1 - scale):.0f}% dimmer than at calibration: "
                      f"{amp:.0f} counts needs more than {src.top:g} mA)")
            else:
                new_cur = cal.current_for(new_angle, want)
        if abs(new_angle - src.angle) < 1e-4 and abs(new_cur - src.current) < 1e-3:
            break                                         # nothing left to change
        n_ref += 1
        print(f"  refine {n_ref}: {new_angle:.3f} deg, {new_cur:.1f} mA")
        src.go(new_angle, new_cur)
    if within is False:
        print(f"  NOT within {a.wl_tolerance_nm:g} nm / {100 * a.amplitude_tolerance_rel:g}%")
    session.log(_row(wl, amp, src.angle, src.current, angle_cal, cur_cal, r, p, n_ref, within),
                r.wl, net)


def _row(wl, amp, angle, cur, angle_cal, cur_cal, r, p=None, n_ref=0, within=None):
    led = r.measure_current_ma() if cur > 0 else 0.0
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "wavelength_nm": wl,
            "amplitude_counts": amp, "angle_deg": angle, "current_ma": cur,
            "led_measured_ma": led, "angle_from_calibration_deg": angle_cal,
            "current_from_calibration_ma": cur_cal,
            "measured_nm": p["centre_nm"] if p else np.nan,
            "measured_peak_counts": p["peak"] if p else np.nan,
            "measured_fwhm_nm": p["fwhm_nm"] if p else np.nan,
            "refinements": n_ref,
            "within_tolerance": "" if within is None else ("yes" if within else "no")}


def interactive(src, a, session):
    cal = src.cal
    print(f"\ncalibrated: {cal.wl_lo:.1f}-{cal.wl_hi:.1f} nm. "
          f"Enter (nothing) or q at the wavelength prompt quits.")
    last_amp = None
    while True:
        txt = ask("\nwavelength (nm): ")
        if not txt or txt.lower() in ("q", "quit", "exit"):
            return
        try:
            wl = float(txt)
            angle = cal.angle_for(wl)
        except ValueError as e:
            print(f"  {e}" if "outside" in str(e) else "  a number, please")
            continue
        top_amp = src.max_amplitude(angle)
        offer = last_amp if last_amp is not None and last_amp <= top_amp else None
        while True:
            hint = f" [Enter = {offer:.0f}]" if offer is not None else ""
            txt = ask(f"peak amplitude at {wl:g} nm (counts, up to {top_amp:.0f}; "
                      f"0 = LED off){hint}: ")
            if txt is None:
                return
            if not txt and offer is not None:
                amp = offer
                break
            try:
                amp = float(txt)
            except ValueError:
                print("  a number, please")
                continue
            if 0 <= amp <= top_amp:
                break
            print(f"  between 0 and {top_amp:.0f}, please")
        apply(src, a, wl, amp, session)
        if amp > 0:
            last_amp = amp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    hardware_args(ap)
    g = ap.add_argument_group("set_wavelength")
    g.add_argument("--calibration-file", default=None, dest="calibration_file",
                   help="a calibration folder (or a file in it); null = the newest")
    g.add_argument("--wavelength-nm", type=float_or_none, default=None, dest="wavelength_nm",
                   help="set this once and exit (with --amplitude-counts)")
    g.add_argument("--amplitude-counts", type=float_or_none, default=None,
                   dest="amplitude_counts")
    g.add_argument("--verify", type=str2bool, default=True,
                   help="check every setting with a spectrum")
    g.add_argument("--refine-iterations", type=int, default=2, dest="refine_iterations",
                   help="corrections after the check (0 = only check)")
    g.add_argument("--wl-tolerance-nm", type=float, default=0.2, dest="wl_tolerance_nm")
    g.add_argument("--amplitude-tolerance-rel", type=float, default=0.02,
                   dest="amplitude_tolerance_rel")
    g.add_argument("--background-refresh-s", type=float, default=600.0,
                   dest="background_refresh_s",
                   help="retake the background before a setting once it is this old")
    g.add_argument("--leave-led-on", type=str2bool, default=True, dest="leave_led_on",
                   help="keep the LED on at the last setting when the script ends")
    g.add_argument("--notes", default="")
    a = parse_config(ap, argv, HERE / "set_wavelength.json", HERE / "calibration.json")

    cal = Calibration(find_calibration(a, a.calibration_file))
    cal.check(a)
    print(f"calibration: {cal.describe()}")
    one_shot = a.wavelength_nm is not None
    if one_shot:
        if a.amplitude_counts is None:
            raise SystemExit("--wavelength-nm needs --amplitude-counts too")
        try:                                   # refuse before touching the hardware
            angle = cal.angle_for(a.wavelength_nm)
            if a.amplitude_counts > 0:
                cal.current_for(angle, a.amplitude_counts)
        except ValueError as e:
            raise SystemExit(str(e))

    rig = make_rig(a, spectrometer=a.verify, power_meter=False)
    with rig as r:
        src = Source(r, a, cal)
        if src.top < cal.currents.max():
            print(f"  note: the LED limit ({r.led_max_ma:g} mA) is below the calibration's "
                  f"highest current ({cal.currents.max():g} mA) -- amplitudes are limited")
        if a.verify:
            r.set_exposure(cal.exposure_ms, cal.hw_average)
        session = Session(a, cal)
        try:
            if one_shot:
                apply(src, a, a.wavelength_nm, a.amplitude_counts, session)
            else:
                interactive(src, a, session)
        except KeyboardInterrupt:
            print("\n(Ctrl-C)")
        except ValueError as e:
            print(f"  {e}")
        if r.led_is_on:
            if one_shot:
                leave = a.leave_led_on
            else:
                try:
                    leave = ask_yes(f"leave the LED on at {src.current:.1f} mA "
                                    f"({src.angle:.3f} deg)?", a.leave_led_on)
                except (EOFError, KeyboardInterrupt):
                    leave = a.leave_led_on
            r.leave_led_on = leave
            print("LED left ON" if leave else "LED switched off")
    if session.folder is not None:
        print(f"log: {session.folder / 'log.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped (Ctrl-C)")
        sys.exit(130)
