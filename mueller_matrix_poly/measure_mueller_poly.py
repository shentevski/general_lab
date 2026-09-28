#!/usr/bin/env python3
"""Measure the Mueller matrix of a sample at every wavelength: LED + spectrometer.

    python measure_mueller_poly.py                # reads measure_mueller_poly.json
    python measure_mueller_poly.py --simulate     # rehearse, no hardware

The spectral version of ../mueller_matrix/measure_mueller.py: the same layout
and procedure, with the laser replaced by a broadband LED (MBB1F1, 470-850 nm)
and the signal power meter by a CCT10 spectrometer.

    LED -> collimator -> polarizer -> beamsplitter -> HWP / QWP -> [sample] -> QWP (rotating) -> analyzer -> fibre -> CCT10
                                           |                                    '------------ polarimeter ------------'
                                           '-> reference power meter (optional LED monitor)

At every QWP step a whole spectrum is recorded, so one run gives M at every
wavelength. For each input state, as before:

    1. set the state by hand and type a label for it
    2. REMOVE the sample      -> the input state is measured
    3. PUT THE SAMPLE BACK    -> the state is measured through the sample

Enter with no label to finish. After every state it reports how well the
states pin down M -- now at every wavelength, and the worst one counts.

Why this layout suits a spectrometer
  * the analyzer is fixed, so the light entering the spectrometer always has
    the same polarization: the grating's and the fibre's polarization
    dependence do not matter;
  * every state is measured, not assumed: the HWP / QWP that make the states
    need no calibration at any wavelength (a 520-nm HWP makes an elliptical
    "D" at 800 nm -- that is fine, it is measured). They only need to spread
    the states over the sphere at every wavelength, which the coverage check
    after each state tells you;
  * the spectral response cancels: every number is a ratio at the same pixel.

What the polarimeter DOES need at every wavelength is the QWP retardance
delta(lambda): set calibration_file to the Thorlabs retardance curve (CSV:
wavelength nm, retardance waves), or better to the qwp_calibration.csv written
by QWP_analyzer_characterization_poly.py. The file is copied into the run.

Spectrometer
  * ONE exposure for the whole run: the Stokes fit assumes a linear detector,
    so the counts must stay well below full scale. At the start you type the
    exposure; the script shows how full the brightest pixel gets on a bright
    state (H or R, sample out) and you keep it or type another (aim for
    ~50-70 % of full scale). Every spectrum is checked for clipping;
  * a dark (shutter closed) before and after every sweep, interpolated in
    time: a leftover offset looks exactly like lost polarization;
  * raw counts are saved (amplitude correction off, no SDK dark), so the
    analysis can redo everything.

LED monitor: set ref_pm_serial and a power meter behind the beamsplitter is
read just before and just after every spectrum; dividing by it removes LED
drift (it corrects the total power, not a change of spectral shape: let the
LED warm up for 30-60 min at constant current).

Every state is saved the moment it is measured.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from poly_common import (angles_of, bin_edges, bin_matrix, calibration_at,
                         centres, clipped, coverage, extract_stokes,
                         float_or_none, has_reference, parse_with_config,
                         psa_blind, reference_factor, resolve_path,
                         signal_counts, str2bool, write_sweep)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


class Rig:
    """QWP rotation stage + CCT10 (+ reference PM100D).
    set_state / set_sample / set_led are no-ops: on the bench YOU do those."""

    def __init__(self, a):
        self.a = a
        self.spec = self.ref = self.qwp = None
        self.t0 = time.time()
        self.wl = None

    def __enter__(self):
        import kcube
        from thorlabs_spectrometer import Spectrometer

        self.spec = Spectrometer(self.a.spectrometer_id or None)
        print(f"spectrometer    : {self.spec.device_id}")
        self.spec.amplitude_correction = False     # raw counts
        try:
            self.spec.clear_dark()                  # we take our own darks
        except Exception:
            pass
        self.spec.open_shutter()
        self.set_exposure(self.a.exposure_ms or 10.0, self.a.hw_average)

        if self.a.ref_pm_serial:
            from thorlabs_powermeter import PowerMeter, list_power_meters
            found = list_power_meters()
            hits = [d for d in found if d.serial_number == self.a.ref_pm_serial
                    or str(self.a.ref_pm_serial) in d.resource_name]
            if not hits:
                print("Power meters found:")
                for d in found:
                    print(f"    {d.serial_number:12s} {d.model:10s} {d.resource_name}")
                raise SystemExit(f"reference power meter '{self.a.ref_pm_serial}' not found")
            self.ref = PowerMeter(hits[0].resource_name)
            self.ref.power_unit = "W"
            self.ref.wavelength_nm = self.a.ref_wavelength_nm
            self.ref.average_count = self.a.pm_average_count
            if self.a.ref_power_range_w is None:
                self.ref.auto_range = True
            else:
                self.ref.auto_range = False
                self.ref.power_range_w = self.a.ref_power_range_w
            print(f"reference meter : {self.ref.identity.serial_number} (LED monitor)")
        else:
            print("reference meter : none (ref_pm_serial is null) -- no LED correction")

        stage = None if str(self.a.qwp_stage).lower() in ("", "none") else self.a.qwp_stage
        self.qwp = kcube.KCube(self.a.qwp_motor, stage_name=stage)
        self.qwp.connect()
        if self.a.home:
            self.qwp.home()
        return self

    def __exit__(self, *exc):
        for obj in (self.spec, self.ref, self.qwp):
            try:
                obj and obj.close()
            except Exception:
                pass
        return False

    @property
    def has_ref(self):
        return self.ref is not None

    def set_exposure(self, ms, hw_average):
        self.spec.exposure_ms = float(ms)
        self.spec.hw_average = int(hw_average)
        self.exposure_ms, self.hw_average = self.spec.exposure_ms, self.spec.hw_average

    def _snap(self):
        s = self.spec.snap()
        if self.wl is None:
            self.wl = s.wavelengths
        return s.intensities

    def read(self):
        """(raw counts (P,), reference W). The reference is read just before
        and just after the spectrum and averaged: it brackets the exposure."""
        if self.ref is None:
            return self._snap(), float("nan")
        r1 = self.ref.read_power()
        c = self._snap()
        r2 = self.ref.read_power()
        return c, 0.5 * (float(r1) + float(r2))

    def dark(self, frames):
        self.spec.close_shutter()
        try:
            return np.mean([self._snap() for _ in range(int(frames))], axis=0)
        finally:
            self.spec.open_shutter()

    def zero_ref(self):
        if self.ref is not None:
            self.ref.zero()

    def now(self):               return time.time() - self.t0
    def move_to(self, deg):      self.qwp.move_to(float(deg) % 360.0)
    def move_relative(self, d):  self.qwp.move_relative(float(d))
    def position(self):          return float(self.qwp.get_position())
    def set_state(self, label):  pass
    def set_sample(self, inside): pass
    def set_led(self, on):       pass

    def describe(self):
        return {"backend": "hardware", "qwp_motor": self.a.qwp_motor,
                "spectrometer": self.spec.device_id,
                "exposure_ms": self.exposure_ms, "hw_average": self.hw_average,
                "reference_meter": str(self.ref.identity) if self.ref else None,
                "reference_read": "before + after each spectrum, averaged" if self.ref else None}


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #


def _rot(phi):
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.]])


def _retarder(delta, axis):
    """(P, 4, 4) linear retarders, retardance delta (P,) rad, fast axis `axis`
    rad -- the toolkit's retarder_mueller, one per wavelength."""
    cd, sd = np.cos(delta), np.sin(delta)
    m = np.zeros(np.shape(delta) + (4, 4))
    m[..., 0, 0] = m[..., 1, 1] = 1
    m[..., 2, 2] = m[..., 3, 3] = cd
    m[..., 2, 3], m[..., 3, 2] = -sd, sd
    return _rot(axis) @ m @ _rot(-axis)


def _diattenuator(D, axis):
    m = np.zeros(np.shape(D) + (4, 4))
    sq = np.sqrt(1 - D ** 2)
    m[..., 0, 0] = m[..., 1, 1] = 1
    m[..., 0, 1] = m[..., 1, 0] = D
    m[..., 2, 2] = m[..., 3, 3] = sq
    return _rot(axis) @ m @ _rot(-axis)


def quartz_waves(wl_nm, waves, design_nm):
    """Retardance (waves) of a zero-order quartz plate that has `waves` at
    design_nm: it scales as dn(lambda) / lambda."""
    dn = lambda w: 0.00867 + 1.47e-4 / (np.asarray(w, float) / 1000) ** 2
    return waves * (design_nm / np.asarray(wl_nm, float)) * dn(wl_nm) / dn(design_nm)


def psa_rows(theta, delta, s3_sign=1):
    """(4, P) first row of analyzer(0) @ QWP(theta, delta(lambda)): the toolkit's
    analyzer_row written out so it runs on every pixel at once."""
    cd, sd = np.cos(delta), np.sin(delta)
    c4, s4, s2 = np.cos(4 * theta), np.sin(4 * theta), np.sin(2 * theta)
    return 0.5 * np.array([np.ones_like(cd), (1 + cd) / 2 + (1 - cd) / 2 * c4,
                           (1 - cd) / 2 * s4, s3_sign * sd * s2])


class SimRig:
    """Rehearsal stand-in: LED, spectrometer, a dispersive sample and a
    dispersive polarimeter, so a run can be checked against the truth.

    * the LED: MBB1F1-like spectrum (FWHM 500-780 nm, -10 dB at 470 / 850),
      drifting and fluctuating (sim_led_*), seen by the reference meter too;
    * the input polarizer: extinction sim_polarizer_er in 510-800 nm, falling
      to a tenth of it at 470 and 850 (LPVISC-like), so the states are NOT
      fully polarized -- an unpolarized LED only gets DOP = 1 - 2/ER;
    * the states: H = polarizer only; V D A with a zero-order HWP, R L with a
      zero-order QWP, both designed for 520 nm -- elliptical elsewhere, set
      by hand with sim_state_error_deg errors;
    * the polarimeter QWP: the calibration in the config (file or constant)
      IS the truth, unless the characterization changes it on purpose;
    * the sample: a partial diattenuator (D ~ 0.3 at 15 deg), then a quartz
      retarder (60 deg at 550 nm, fast axis 20 deg), then a weak depolarizer,
      80 % transmission -- every part of it dispersive;
    * the spectrometer: 2048 pixels 200-1000 nm, 16-bit, read noise 12
      counts, shot noise (SNR 330 at full scale), a dark offset that drifts,
      dark current, stray light (sim_stray_rel), clipping at full scale.
    Time runs on a virtual clock.
    """

    simulated = True             # sweeps do not really wait for the stage
    READ_S = 0.03                # USB readout of one spectrum
    SHUTTER_S = 0.08
    SWAP_S = 20.0                # you taking the sample out / putting it back
    STATE_S = 60.0               # you setting a new state
    DRIFT_PERIOD_S = 900.0

    PLATES = {"H": [], "V": [("hwp", 45)], "D": [("hwp", 22.5)], "A": [("hwp", -22.5)],
              "R": [("qwp", -45)], "RCP": [("qwp", -45)],
              "L": [("qwp", 45)], "LCP": [("qwp", 45)]}

    def __init__(self, a, retardance_scale=1.0):
        self.a = a
        self.rng = np.random.default_rng(1)
        self.wl = np.linspace(200.0, 1000.0, 2048)
        wl = self.wl
        self.led = np.exp(-np.log(2) * ((wl - 640) / 140) ** 4)
        cal = calibration_at(vars(a), wl, HERE, strict=False)
        self.psa_delta = np.deg2rad(cal["retardance_deg"]) * retardance_scale
        self.psa_zero = cal["zero_deg"]
        er = a.sim_polarizer_er * 10 ** (-np.clip(np.maximum(510 - wl, wl - 800) / 40, 0, 1))
        self.p_in = (er - 1) / (er + 1)
        d = np.deg2rad
        D = 0.3 * np.exp(-((wl - 650) / 300) ** 2)
        ret = _retarder(d(60) * quartz_waves(wl, 1, 550), d(20))
        self.sample = (0.8 * (1 - 0.1 * (wl - 650) / 200))[:, None, None] * (
            np.diag([1, .95, .95, .95]) @ ret @ _diattenuator(D, d(15)))
        self.hwp = 2 * np.pi * quartz_waves(wl, 0.5, 520)
        self.qwp = 2 * np.pi * quartz_waves(wl, 0.25, 520)
        self.pattern = self.rng.normal(0, 20, wl.size)
        self.state = self._make_state("H", err=0.0)
        self.inside = False
        self.led_on = True
        self.pos = 0.0
        self.clock = 0.0
        self.has_ref = bool(a.ref_pm_serial)
        self.exposure_ms, self.hw_average = (a.exposure_ms or 10.0), a.hw_average

    def __enter__(self):
        print("spectrometer    : simulated (2048 px, 200-1000 nm)")
        print("reference meter : " + ("simulated (LED monitor)" if self.has_ref else
                                      "none (ref_pm_serial is null) -- no LED correction"))
        return self

    def __exit__(self, *exc):
        return False

    def _make_state(self, label, err=None):
        e = np.deg2rad(self.a.sim_state_error_deg if err is None else err)
        plates = self.PLATES.get(label.strip().upper())
        if plates is None:
            plates = [("hwp", self.rng.uniform(0, 180)), ("qwp", self.rng.uniform(0, 180))]
            print(f"  [sim] '{label}' is not H/V/D/A/R/L -- using random plate angles")
        pol = self.rng.normal(0, e)
        S = np.stack([np.ones_like(self.wl), self.p_in * np.cos(2 * pol),
                      self.p_in * np.sin(2 * pol), 0 * self.wl], axis=-1)
        for plate, ang in plates:
            M = _retarder(self.hwp if plate == "hwp" else self.qwp,
                          np.deg2rad(ang) + self.rng.normal(0, e))
            S = np.einsum("pij,pj->pi", M, S)
        return S

    def truth(self):
        return {"wavelengths": self.wl, "M": self.sample, "led": self.led,
                "psa_retardance_waves": self.psa_delta / (2 * np.pi),
                "psa_zero_deg": np.broadcast_to(self.psa_zero, self.wl.shape),
                "input_dop": self.p_in}

    def set_state(self, label):
        self.clock += self.STATE_S
        self.state = self._make_state(label)

    def set_sample(self, inside):
        self.clock += self.SWAP_S
        self.inside = bool(inside)

    def set_led(self, on):
        self.led_on = bool(on)

    def set_exposure(self, ms, hw_average):
        self.exposure_ms = float(np.clip(ms, 0.01, 30000))
        self.hw_average = int(hw_average)

    def now(self):                 return self.clock
    def move_to(self, deg):        self.pos = float(deg)
    def move_relative(self, d):    self.pos += float(d)
    def position(self):            return self.pos % 360.0
    def zero_ref(self):            pass

    def _dark_level(self):
        return (1000 + self.pattern + 15 * np.sin(2 * np.pi * self.clock / 1800)
                + 0.5 * self.exposure_ms)

    def _frame(self, light):
        n = max(self.hw_average, 1)
        sig = self.a.sim_counts_per_ms * self.exposure_ms * light
        sig = sig + self.a.sim_stray_rel * sig.mean()
        noise = np.sqrt(12.0 ** 2 + 0.56 * np.clip(sig, 0, None)) / np.sqrt(n)
        c = self._dark_level() + sig + self.rng.normal(0, 1, sig.size) * noise
        return np.clip(c, 0, self.a.full_scale_counts)

    def _led_power(self):
        laser = 1 + self.a.sim_led_drift_rel * np.sin(
            2 * np.pi * self.clock / self.DRIFT_PERIOD_S + 0.7)
        return laser * (1 + self.rng.normal(0, self.a.sim_led_noise_rel))

    def read(self):
        self.clock += self.a.qwp_settle + self.exposure_ms * self.hw_average / 1e3 + self.READ_S
        P = self._led_power() if self.led_on else 0.0
        S = np.einsum("pij,pj->pi", self.sample, self.state) if self.inside else self.state
        row = psa_rows(np.deg2rad(self.pos - self.psa_zero), self.psa_delta, self.a.s3_sign)
        light = P * self.led * np.einsum("ip,pi->p", row, S)
        ref = (0.5e-3 * P * (1 + self.rng.normal(0, 1e-4))
               if self.has_ref else float("nan"))
        return self._frame(light), ref

    def dark(self, frames):
        self.clock += 2 * self.SHUTTER_S + frames * self.exposure_ms * self.hw_average / 1e3
        return np.mean([self._frame(0 * self.wl) for _ in range(int(frames))], axis=0)

    def describe(self):
        return {"backend": "simulated", "exposure_ms": self.exposure_ms,
                "hw_average": self.hw_average,
                "reference_read": "before + after each spectrum" if self.has_ref else None,
                "sim_state_error_deg": self.a.sim_state_error_deg,
                "sim_polarizer_er": self.a.sim_polarizer_er,
                "sim_led_drift_rel": self.a.sim_led_drift_rel,
                "sim_led_noise_rel": self.a.sim_led_noise_rel,
                "sim_stray_rel": self.a.sim_stray_rel,
                "truth": "sim_truth.npz"}


# --------------------------------------------------------------------------- #
# measurement steps (shared with the characterization)
# --------------------------------------------------------------------------- #


def sweep(rig, a, tag, span_deg=360.0):
    """One QWP sweep of qwp_steps positions over span_deg, a spectrum at each,
    with a shutter dark just before and just after -> dict with SWEEP_KEYS."""
    n = int(a.qwp_steps)
    step = span_deg / n
    com = a.qwp_zero_deg + np.arange(n) * step
    rig.move_to(com[0])
    t_db = rig.now()
    dark_b = rig.dark(a.dark_frames)
    counts, ref, t, meas = [], np.empty(n), np.empty(n), np.empty(n)
    t_start = rig.now()
    for k in range(n):
        if k:
            rig.move_relative(step)
        settle(rig, a)
        t[k] = rig.now()
        c, ref[k] = rig.read()
        counts.append(c)
        meas[k] = rig.position()
        done = int(30 * (k + 1) / n)
        print(f"\r  measuring {tag:6s} [{'#' * done}{'.' * (30 - done)}] {k+1}/{n}",
              end="", flush=True)
    t_da = rig.now()
    dark_a = rig.dark(a.dark_frames)
    print(f"   {rig.now() - t_start:.0f} s")
    return {"commanded_deg": com, "measured_deg": meas, "counts": np.array(counts),
            "ref_W": ref, "t_s": t, "wavelengths": np.asarray(rig.wl),
            "dark_before": dark_b, "dark_after": dark_a, "t_dark": np.array([t_db, t_da]),
            "exposure_ms": rig.exposure_ms, "hw_average": rig.hw_average}


def settle(rig, a):
    if not getattr(rig, "simulated", False):
        time.sleep(a.qwp_settle)


def band(wl, a):
    return (wl >= a.wl_min_nm) & (wl <= a.wl_max_nm)


def parse_ms(txt):
    """An exposure typed in ms, or None (with the reason printed)."""
    try:
        ms = float(txt)
    except ValueError:
        print("  a number of milliseconds, please")
        return None
    if not 0.01 <= ms <= 30000:
        print("  the CCT10 takes 0.01 - 30000 ms")
        return None
    return ms


def ask_exposure(suggested=None):
    """Ask until a valid exposure is typed; Enter takes the suggestion, if any."""
    while True:
        txt = input("  exposure in ms"
                    + (f" [Enter = {suggested:g}]" if suggested else "") + ": ").strip()
        if not txt and suggested:
            return float(suggested)
        ms = parse_ms(txt) if txt else None
        if ms is not None:
            return ms


def brightest(rig, a):
    """Brightest pixel (raw counts) over a coarse half turn of the QWP -- the
    current state, sample out -- and whether any spectrum clipped."""
    raw_max, clip = 0.0, False
    for k in range(12):                            # 15-deg steps over 180 deg
        rig.move_to(a.qwp_zero_deg + 15 * k)
        settle(rig, a)
        c, _ = rig.read()
        m = band(rig.wl, a)
        raw_max = max(raw_max, float(c[m].max()))
        clip |= bool(clipped(c[m], a.full_scale_counts)[0])
    return raw_max, clip


def check_exposure(rig, a):
    """You enter the exposure. The script shows how bright the brightest pixel
    gets over a coarse half turn (the current state, sample out) and whether it
    clips; keep it, or type another. It then stays fixed for the whole run.
    exposure_ms in the JSON is offered as the suggestion (null = none).

    Aim for the brightest pixel at ~50-70 % of full scale: a clipped spectrum
    breaks the fit's linear-detector assumption, too little light only adds
    noise."""
    fs = a.full_scale_counts
    ms = ask_exposure(a.exposure_ms)
    while True:
        rig.set_exposure(ms, a.hw_average)
        raw_max, clip = brightest(rig, a)
        fill = raw_max / fs
        verdict = ("CLIPPED -- lower it" if clip else
                   "near full scale -- lower it" if fill > 0.85 else
                   "dim -- a longer exposure gives less noise" if fill < 0.2 else "good")
        print(f"  exposure {rig.exposure_ms:g} ms x {rig.hw_average} frames: brightest "
              f"pixel {raw_max:.0f} counts = {100 * fill:.0f}% of full scale ({verdict})")
        txt = input("  Enter = keep this exposure for the whole run, or type another "
                    "(ms): ").strip()
        if txt:
            new = parse_ms(txt)
            ms = new if new is not None else ask_exposure()
            continue
        if clip and input("  it CLIPS: clipped spectra are not linear. Keep it anyway? "
                          "[y/N]: ").strip().lower() != "y":
            ms = ask_exposure()
            continue
        return


def quick_stokes(sw, a, cal, base, bin_nm, zero_deg=None):
    """Stokes vectors in coarse bins, for live feedback -- LED-corrected when
    possible, at the run's calibration (cal, file paths relative to base)."""
    edges = bin_edges(a.wl_min_nm, a.wl_max_nm, bin_nm)
    W = bin_matrix(sw["wavelengths"], edges)
    wl = centres(edges)
    P = (signal_counts(sw) @ W) * reference_factor(sw)[:, None]
    c = calibration_at(cal, wl, base, strict=False)
    f = extract_stokes(P, angles_of(sw),
                       zero_deg=c["zero_deg"] if zero_deg is None else zero_deg,
                       retardance_deg=c["retardance_deg"], s3_sign=cal["s3_sign"])
    f["wl"], f["rms_rel"] = wl, f["resid_rms"] / np.mean(P, axis=0)
    return f


def show_stokes(f, show_wl_nm, name):
    lines = []
    for w in show_wl_nm:
        if not (f["wl"][0] - 1e-9 <= w <= f["wl"][-1] + 1e-9):
            continue
        k = int(np.argmin(abs(f["wl"] - w)))
        S = f["S"][:, k] / f["S"][0, k]
        lines.append(f"    {f['wl'][k]:5.0f} nm  [{S[1]:+.3f} {S[2]:+.3f} {S[3]:+.3f}]  "
                     f"DOP {np.linalg.norm(S[1:]):.3f}  fit rms {100 * f['rms_rel'][k]:.2f}%")
    print(f"  {name}\n" + "\n".join(lines))


def sweep_health(sw, a):
    """One line: clipping, light level, dark drift, LED drift."""
    m = band(sw["wavelengths"], a)
    clip = int(clipped(sw["counts"][:, m], a.full_scale_counts).sum())
    fill = sw["counts"][:, m].max() / a.full_scale_counts
    ddark = float(np.median((sw["dark_after"] - sw["dark_before"])[m]))
    txt = (f"peak {100 * fill:.0f}% of full scale, dark moved {ddark:+.1f} counts"
           + (f", LED {100 * (sw['ref_W'][-1] / sw['ref_W'][0] - 1):+.2f}% over the sweep"
              if has_reference(sw) else ""))
    if clip:
        txt += f"\n  WARNING: {clip} spectra CLIPPED -- lower the exposure and redo this state"
    return txt


def copy_calibration(a, run: Path):
    """Put the calibration file used into the run, so the run is complete."""
    cal = {"qwp_zero_deg": a.qwp_zero_deg, "qwp_retardance_waves": a.qwp_retardance_waves,
           "calibration_file": None, "zero_source": a.zero_source,
           "s3_sign": a.s3_sign, "analyzer_deg": 0.0}
    src = resolve_path(a.calibration_file, HERE)
    if src is not None:
        if not src.is_file():
            raise SystemExit(f"calibration_file not found: {src}")
        shutil.copy(src, run / "qwp_calibration_used.csv")
        cal["calibration_file"] = "qwp_calibration_used.csv"
        cal["calibration_file_original"] = str(src)
    return cal


def start(r, a):
    """Block the LED: zero the reference meter, measure the ambient light that
    reaches the spectrometer. Then set / check the exposure."""
    if a.block_at_start:
        input("\nBLOCK the LED (before the beamsplitter), then press Enter...")
        r.set_led(False)
        r.zero_ref()
        dark = r.dark(a.dark_frames)
        amb, _ = r.read()
        amb = (amb - dark) / r.exposure_ms              # counts per ms
        r.set_led(True)
        input("UNBLOCK the LED, then press Enter...")
        return amb
    return None


def ambient_report(amb, r, a):
    if amb is None:
        return
    m = band(r.wl, a)
    amb_max = float(np.max(amb[m])) * r.exposure_ms
    print(f"  ambient light reaching the spectrometer: up to {amb_max:.0f} counts at "
          f"{r.exposure_ms:g} ms"
          + ("  -- shield the fibre / switch the room light off: unmodulated light "
             "looks like lost polarization" if amb_max > 0.001 * a.full_scale_counts
             else " (negligible)"))


def hardware_args(ap):
    """Hardware, calibration and simulator settings -- shared with the
    characterization, which reads them from measure_mueller_poly.json."""
    ap.add_argument("--qwp-motor", default="28000005", dest="qwp_motor")
    ap.add_argument("--qwp-stage", default="none", dest="qwp_stage")
    ap.add_argument("--home", type=str2bool, default=True)
    ap.add_argument("--qwp-steps", type=int, default=100, dest="qwp_steps",
                    help="QWP positions per sweep, spread over qwp_sweep_deg")
    ap.add_argument("--qwp-sweep-deg", type=float, default=360.0, dest="qwp_sweep_deg",
                    help="360 (a full turn, beam walk separated) or 180")
    ap.add_argument("--qwp-settle", type=float, default=0.3, dest="qwp_settle")
    ap.add_argument("--qwp-zero-deg", type=float, default=93.6, dest="qwp_zero_deg",
                    help="stage angle at which the QWP fast axis is along the analyzer")
    ap.add_argument("--qwp-retardance-waves", type=float, default=0.25,
                    dest="qwp_retardance_waves",
                    help="QWP retardance used at EVERY wavelength when there is no "
                         "calibration_file")
    ap.add_argument("--calibration-file", default=None, dest="calibration_file",
                    help="CSV of QWP retardance vs wavelength (Thorlabs data, or "
                         "qwp_calibration.csv from the characterization); null = the "
                         "constant qwp_retardance_waves")
    ap.add_argument("--zero-source", default="constant", choices=("constant", "file"),
                    dest="zero_source",
                    help="constant: qwp_zero_deg at every wavelength (zero-order and "
                         "quartz/MgF2 achromatic plates); file: the zero_deg column")
    ap.add_argument("--s3-sign", type=int, default=1, choices=(1, -1), dest="s3_sign")
    ap.add_argument("--spectrometer-id", default=None, dest="spectrometer_id",
                    help="CCT device ID; null = the first one found")
    ap.add_argument("--exposure-ms", type=float_or_none, default=None, dest="exposure_ms",
                    help="exposure offered at the start (Enter takes it); null = you "
                         "type one")
    ap.add_argument("--hw-average", type=int, default=10, dest="hw_average",
                    help="frames the spectrometer averages per QWP step")
    ap.add_argument("--full-scale-counts", type=float, default=65535,
                    dest="full_scale_counts", help="raw counts at saturation")
    ap.add_argument("--dark-frames", type=int, default=3, dest="dark_frames",
                    help="spectra averaged for each shutter dark")
    ap.add_argument("--wl-min-nm", type=float, default=470.0, dest="wl_min_nm")
    ap.add_argument("--wl-max-nm", type=float, default=850.0, dest="wl_max_nm")
    ap.add_argument("--ref-pm-serial", default=None, dest="ref_pm_serial",
                    help="reference meter (LED monitor after the beamsplitter); "
                         "null = no reference")
    ap.add_argument("--ref-wavelength-nm", type=float, default=620.0,
                    dest="ref_wavelength_nm",
                    help="wavelength set on the reference meter (only scales its reading)")
    ap.add_argument("--pm-average-count", type=int, default=100, dest="pm_average_count")
    ap.add_argument("--ref-power-range-w", type=float_or_none, default=None,
                    dest="ref_power_range_w")
    ap.add_argument("--block-at-start", type=str2bool, default=True, dest="block_at_start",
                    help="block the LED at the start: zero the reference meter and "
                         "measure the ambient light")
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-state-error-deg", type=float, default=3.0,
                    dest="sim_state_error_deg")
    ap.add_argument("--sim-counts-per-ms", type=float, default=2000.0,
                    dest="sim_counts_per_ms",
                    help="counts per ms at the LED peak with all light through")
    ap.add_argument("--sim-led-drift-rel", type=float, default=0.005,
                    dest="sim_led_drift_rel")
    ap.add_argument("--sim-led-noise-rel", type=float, default=0.001,
                    dest="sim_led_noise_rel")
    ap.add_argument("--sim-polarizer-er", type=float, default=1000.0,
                    dest="sim_polarizer_er")
    ap.add_argument("--sim-stray-rel", type=float, default=0.002, dest="sim_stray_rel")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    hardware_args(ap)
    ap.add_argument("--feedback-bin-nm", type=float, default=20.0, dest="feedback_bin_nm",
                    help="bin width of the live feedback (the analysis sets its own)")
    ap.add_argument("--show-wl-nm", type=float, nargs="*",
                    default=[500, 550, 600, 650, 700, 750, 800], dest="show_wl_nm")
    a = parse_with_config(ap, argv, HERE / "measure_mueller_poly.json")

    root = Path(a.out) if a.out else Path.home() / "Desktop"
    run = root / time.strftime("mueller_poly_%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    cal = copy_calibration(a, run)
    check = calibration_at(cal, np.array([a.wl_min_nm, a.wl_max_nm]), run, strict=False)
    print(f"\nrun folder : {run}")
    print(f"PSA QWP    : {check['source']}")
    if cal["calibration_file"] is None:
        print("             no calibration_file: one retardance for the whole band is "
              "only right for an achromatic plate -- live Stokes vectors are indicative")
    print(f"sweep      : {a.qwp_steps} steps over {a.qwp_sweep_deg:g} deg, "
          f"{a.wl_min_nm:g}-{a.wl_max_nm:g} nm\n")

    meta = {"kind": "mueller_spectral",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_file": str(Path(a.config).resolve()),
            "settings": vars(a), "calibration": cal,
            "qwp_sweep_deg": a.qwp_sweep_deg, "notes": a.notes, "states": []}

    rig = SimRig(a) if a.simulate else Rig(a)
    with rig as r:
        def save():
            meta["hardware"] = r.describe()
            (run / "run.json").write_text(json.dumps(meta, indent=2, default=str))

        if a.simulate:
            np.savez_compressed(run / "sim_truth.npz", **r.truth())
        amb = start(r, a)
        print("\nEXPOSURE: set a BRIGHT state (H or R), sample OUT, then type an "
              "exposure -- the script shows how full the brightest pixel gets.")
        check_exposure(r, a)
        ambient_report(amb, r, a)
        print(f"  exposure for the whole run: {r.exposure_ms:g} ms x {r.hw_average} frames")
        save()

        S_in, idx = [], 0
        while True:
            idx += 1
            print(f"\n=== state {idx} ===")
            label = input("Set the input state by hand, then type a label for it"
                          "\n  (e.g. H, D, R -- approximate is fine; Enter = finish): ").strip()
            if not label:
                rank = coverage(np.stack(S_in, -1))[0].min() if S_in else 0
                if rank < 4:
                    ans = input(f"Only rank {rank} of 4 at some wavelengths -- the "
                                f"analysis cannot reconstruct M there. Finish anyway? "
                                f"[y/N]: ").strip().lower()
                    if ans != "y":
                        idx -= 1
                        continue
                break
            r.set_state(label)

            fits = {}
            for side, prompt, tag in (
                    ("in", "REMOVE the sample, then press Enter to measure the input state...",
                     "input"),
                    ("out", "PUT THE SAMPLE BACK, then press Enter to measure through it...",
                     "output")):
                input(prompt)
                r.set_sample(side == "out")
                sw = sweep(r, a, tag, a.qwp_sweep_deg)
                write_sweep(run / f"state_{idx:02d}_{side}.npz", sw)
                fits[side] = quick_stokes(sw, a, cal, run, a.feedback_bin_nm)
                print(f"  {sweep_health(sw, a)}")
                show_stokes(fits[side], a.show_wl_nm, "S_in  (sample out)" if side == "in"
                            else "S_out (through sample)")

            meta["states"].append({"index": idx, "label": label,
                                   "input": f"state_{idx:02d}_in.npz",
                                   "output": f"state_{idx:02d}_out.npz"})
            save()
            S_in.append(fits["in"]["S"].T)                  # (N, 4)
            wl = fits["in"]["wl"]
            if psa_blind(calibration_at(cal, wl, run, strict=False)["retardance_deg"]).any():
                print("  NOTE: the polarimeter QWP is close to 0 or 180 deg retardance "
                      "somewhere in the band -- it is nearly blind there")

            rank, cond = coverage(np.stack(S_in, -1))
            if rank.min() < 4:
                bad = wl[rank < 4]
                circ = np.abs(np.stack(S_in, -1)[:, 3] / np.stack(S_in, -1)[:, 0]).max(1)
                hint = ("add a state with a circular component (e.g. R or L)"
                        if len(S_in) >= 3 and (circ[rank < 4] < 0.2).all()
                        else "need at least 4 states")
                where = (f"{bad[0]:.0f} nm" if bad.size == 1 else
                         f"{bad.size} wavelengths in {bad[0]:.0f}-{bad[-1]:.0f} nm")
                print(f"  saved. states {len(S_in)} | rank < 4 at {where} -- {hint}")
            else:
                worst = int(np.argmax(cond))
                verdict = ("good" if cond[worst] < 4 else
                           "usable -- spread the states more" if cond[worst] < 10 else
                           "POOR -- states too similar there")
                print(f"  saved. states {len(S_in)} | rank 4 everywhere, condition number "
                      f"median {np.median(cond):.2f}, worst {cond[worst]:.2f} at "
                      f"{wl[worst]:.0f} nm ({verdict}; 1.73 is the best possible)")

    print(f"\n{len(meta['states'])} states saved in {run}")
    print("Analyse with: python analyze_mueller_poly.py   "
          "(set run_dir in analyze_mueller_poly.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
