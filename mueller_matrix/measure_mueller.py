#!/usr/bin/env python3
"""Measure the Mueller matrix of a sample, one hand-made input state at a time.

    python measure_mueller.py                  # reads measure_mueller.json
    python measure_mueller.py --simulate       # rehearse, no hardware

Optical layout

    laser -> polarizer -> beamsplitter -> HWP / QWP -> [sample] -> QWP (rotating) -> analyzer -> power meter
                               |                                   '------------ polarimeter ------------'
                               '-> reference power meter (laser monitor)

For each input state:

    1. you set the state by hand and type a label for it
    2. you REMOVE the sample      -> the script measures the input state
    3. you PUT THE SAMPLE BACK    -> the script measures through the sample

and repeat as many times as you like. After every state it reports how well
the states you have so far pin down M (rank and condition number); press
Enter with no label to finish.

The states do NOT need to be exact. Each one is measured, not assumed, so an
approximate H or R is as good as a perfect one. What matters is that

  * the state does not change between its "input" and "through sample"
    measurements -- don't touch the polarizer optics in between,
  * together the states span everything: at least 4, and at least one with
    a real circular component (linear states alone give rank 3),
  * they are well spread on the Poincare sphere (H V D A R L is ideal).

Laser monitor: set ref_pm_serial in the JSON and the second power meter is
read at every QWP step, at the same moment and over the same averaging
window as the signal meter. Dividing by it removes laser power drift and
noise -- within a sweep, and between the input and through-sample sweeps
(which is what keeps M00 an honest transmittance). The beamsplitter sits
after the polarizer, so the light it splits has a fixed polarization and
its split ratio does not change with the state you prepare.

Every state is saved the moment it is measured, so an interrupted run keeps
what it has.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mueller_common import (corrected_power, extract_stokes, has_reference,
                            parse_with_config, waves_to_deg, write_sweep)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


def _setup_meter(pm, a, range_w):
    pm.power_unit = "W"
    pm.wavelength_nm = a.wavelength_nm
    pm.average_count = a.pm_average_count      # same window on both meters
    if range_w is None:
        pm.auto_range = True
    else:
        pm.auto_range = False
        pm.power_range_w = range_w


def _pick_meters(a):
    """Resource names of the signal and reference meters, found by serial."""
    from thorlabs_powermeter import list_power_meters

    found = list_power_meters()
    print("Power meters found:")
    for d in found:
        print(f"    {d.serial_number:12s} {d.model:10s} {d.resource_name}")

    def by_serial(serial, role):
        hits = [d for d in found
                if d.serial_number == serial or str(serial) in d.resource_name]
        if not hits:
            raise SystemExit(f"{role} power meter '{serial}' not found -- "
                             f"check the serial in measure_mueller.json")
        return hits[0].resource_name

    ref = by_serial(a.ref_pm_serial, "reference") if a.ref_pm_serial else None
    if a.pm_serial:
        sig = by_serial(a.pm_serial, "signal")
    else:
        others = [d.resource_name for d in found if d.resource_name != ref]
        if len(others) != 1:
            raise SystemExit(
                f"{len(others)} candidate signal meters -- set pm_serial (signal, "
                f"after the analyzer) and ref_pm_serial (laser monitor) in "
                f"measure_mueller.json, using the serials listed above")
        sig = others[0]
    if sig == ref:
        raise SystemExit("pm_serial and ref_pm_serial are the same meter")
    return sig, ref


class Rig:
    """QWP rotation stage + signal PM100D (+ reference PM100D).
    set_state / set_sample are no-ops: on the bench YOU do those by hand."""

    def __init__(self, a):
        self.a = a
        self.pm = self.ref = self._pool = None
        self.parallel = True
        self.t0 = time.time()

    def __enter__(self):
        import kcube
        from thorlabs_powermeter import PowerMeter

        sig_res, ref_res = _pick_meters(self.a)
        self.pm = PowerMeter(sig_res)
        _setup_meter(self.pm, self.a, self.a.power_range_w)
        print(f"signal meter    : {self.pm.identity.serial_number}")
        if ref_res:
            self.ref = PowerMeter(ref_res)
            _setup_meter(self.ref, self.a, self.a.ref_power_range_w)
            self._pool = ThreadPoolExecutor(max_workers=1)
            print(f"reference meter : {self.ref.identity.serial_number} (laser monitor)")
        else:
            print("reference meter : none (ref_pm_serial is null) -- no laser correction")

        stage = None if str(self.a.qwp_stage).lower() in ("", "none") else self.a.qwp_stage
        self.qwp = kcube.KCube(self.a.qwp_motor, stage_name=stage)
        self.qwp.connect()
        if self.a.home:
            self.qwp.home()
        return self

    def __exit__(self, *exc):
        if self._pool:
            self._pool.shutdown(wait=True)
        for obj in (self.pm, self.ref, getattr(self, "qwp", None)):
            try:
                obj and obj.close()
            except Exception:
                pass
        return False

    @property
    def has_ref(self):
        return self.ref is not None

    def read(self):
        """(signal W, reference W). Both meters integrate at the same time;
        if the driver refuses parallel reads, fall back to reading the
        reference just before and just after the signal and averaging."""
        if self.ref is None:
            return float(self.pm.read_power()), float("nan")
        if self.parallel:
            try:
                fut = self._pool.submit(self.ref.read_power)
                sig = self.pm.read_power()
                return float(sig), float(fut.result())
            except Exception as e:
                print(f"\n  parallel meter reading failed ({e}); reading them "
                      f"one after the other from now on")
                self.parallel = False
        r1 = self.ref.read_power()
        sig = self.pm.read_power()
        r2 = self.ref.read_power()
        return float(sig), 0.5 * (r1 + r2)

    def zero_meter(self):
        for pm in (self.pm, self.ref):
            if pm is not None:
                pm.zero()

    def now(self):               return time.time() - self.t0
    def move_to(self, deg):      self.qwp.move_to(float(deg) % 360.0)
    def move_relative(self, d):  self.qwp.move_relative(float(d))
    def position(self):          return float(self.qwp.get_position())
    def set_state(self, label):  pass
    def set_sample(self, inside): pass

    def describe(self):
        return {"backend": "hardware", "qwp_motor": self.a.qwp_motor,
                "qwp_model": getattr(self.qwp, "model", None),
                "signal_meter": str(self.pm.identity),
                "reference_meter": str(self.ref.identity) if self.ref else None,
                "reference_read": (None if not self.ref else "parallel" if self.parallel
                                   else "sequential (ref before + after signal, averaged)"),
                "pm_average_count": self.a.pm_average_count,
                "power_range_w": self.a.power_range_w,
                "ref_power_range_w": self.a.ref_power_range_w}


class SimRig:
    """Rehearsal stand-in with a known sample, so a run can be checked.

    Hand-made states are imperfect on purpose (sim_state_error_deg), to show
    that approximate states still give the right M. The laser drifts and
    fluctuates (sim_laser_*), seen by both meters, to show what the
    reference removes. Time runs on a virtual clock, so drift is realistic
    even when the rehearsal itself runs in seconds.
    """

    READ_S = 0.3             # one meter reading
    SWAP_S = 20.0            # you taking the sample out / putting it back
    STATE_S = 60.0           # you setting a new state
    DRIFT_PERIOD_S = 900.0

    NOMINAL = {"H": (1, 1, 0, 0), "V": (1, -1, 0, 0), "D": (1, 0, 1, 0),
               "A": (1, 0, -1, 0), "R": (1, 0, 0, 1), "RCP": (1, 0, 0, 1),
               "L": (1, 0, 0, -1), "LCP": (1, 0, 0, -1)}

    def __init__(self, a):
        from polarization_toolkit.hardware.simulated import (
            retarder_mueller, rotator_mueller, analyzer_row)
        self._ret, self._rot, self._row = retarder_mueller, rotator_mueller, analyzer_row
        self.a = a
        self.rng = np.random.default_rng(1)
        d = np.deg2rad
        # a partial diattenuator (D = 0.3 at 15 deg) ...
        D, ax = 0.3, np.array([np.cos(d(30)), np.sin(d(30)), 0.0])
        MD = np.eye(4); MD[0, 1:] = MD[1:, 0] = D * ax
        sq = np.sqrt(1 - D * D)
        MD[1:, 1:] = sq * np.eye(3) + (1 - sq) * np.outer(ax, ax)
        # ... then a 60 deg retarder at 20 deg, then a weak depolarizer
        self.sample = np.diag([1, .95, .95, .95]) @ retarder_mueller(d(60), d(20)) @ MD
        self.sample *= 0.8                                   # 80% transmission
        self.state = np.array([1., 1, 0, 0])
        self.inside = False
        self.pos = 0.0
        self.P0 = 1e-3                                       # 1 mW
        self.clock = 0.0
        self.has_ref = bool(a.ref_pm_serial)
        self.parallel = True

    def __enter__(self):
        print("signal meter    : simulated")
        print("reference meter : " + ("simulated (laser monitor)" if self.has_ref else
                                      "none (ref_pm_serial is null) -- no laser correction"))
        return self

    def __exit__(self, *exc):
        return False

    def set_state(self, label):
        self.clock += self.STATE_S
        nom = self.NOMINAL.get(label.strip().upper())
        if nom is None:
            v = self.rng.normal(size=3); v /= np.linalg.norm(v)
            nom = (1, *v)
            print(f"  [sim] '{label}' is not H/V/D/A/R/L -- using a random state")
        e = np.deg2rad(self.a.sim_state_error_deg)
        axis = self.rng.uniform(0, np.pi)
        wobble = self._rot(self.rng.normal(0, e)) @ self._ret(self.rng.normal(0, 2 * e), axis)
        self.state = wobble @ np.array(nom, float)          # imperfect, like by hand

    def set_sample(self, inside):
        self.clock += self.SWAP_S
        self.inside = bool(inside)

    def now(self):                 return self.clock
    def move_to(self, deg):        self.pos = float(deg)
    def move_relative(self, d):    self.pos += float(d)
    def position(self):            return self.pos % 360.0
    def zero_meter(self):          pass

    def read(self):
        self.clock += self.a.qwp_settle + self.READ_S
        laser = self.P0 * (1 + self.a.sim_laser_drift_rel
                           * np.sin(2 * np.pi * self.clock / self.DRIFT_PERIOD_S + 0.7))
        laser *= 1 + self.rng.normal(0, self.a.sim_laser_noise_rel)   # both meters see it
        S = (self.sample @ self.state) if self.inside else self.state
        row = self._row(np.deg2rad(self.pos - self.a.qwp_zero_deg), 0.0,
                        np.deg2rad(waves_to_deg(self.a.qwp_retardance_waves)))
        sig = laser * float(row @ S) * (1 + self.rng.normal(0, self.a.sim_noise_rel))
        ref = (0.5 * laser * (1 + self.rng.normal(0, self.a.sim_noise_rel))
               if self.has_ref else float("nan"))
        return sig, ref

    def describe(self):
        return {"backend": "simulated", "true_sample_mueller": self.sample.tolist(),
                "reference_read": "parallel" if self.has_ref else None,
                "sim_state_error_deg": self.a.sim_state_error_deg,
                "sim_noise_rel": self.a.sim_noise_rel,
                "sim_laser_drift_rel": self.a.sim_laser_drift_rel,
                "sim_laser_noise_rel": self.a.sim_laser_noise_rel}


# --------------------------------------------------------------------------- #


def sweep(rig, a, tag, span_deg=360.0):
    """One QWP sweep of qwp_steps positions over span_deg -> dict with the
    sweep CSV columns. 360 deg also separates beam walk (360-deg period)."""
    n = int(a.qwp_steps)
    step = span_deg / n
    sw = {"commanded_deg": a.qwp_zero_deg + np.arange(n) * step,
          "measured_deg": np.empty(n), "power_W": np.empty(n),
          "ref_W": np.empty(n), "t_s": np.empty(n)}
    rig.move_to(sw["commanded_deg"][0])
    t_start = rig.now()
    for k in range(n):
        if k:
            rig.move_relative(step)
        time.sleep(a.qwp_settle)
        sw["t_s"][k] = rig.now()
        sw["power_W"][k], sw["ref_W"][k] = rig.read()
        sw["measured_deg"][k] = rig.position()
        done = int(30 * (k + 1) / n)
        print(f"\r  measuring {tag:6s} [{'#' * done}{'.' * (30 - done)}] {k+1}/{n}",
              end="", flush=True)
    print(f"   {rig.now() - t_start:.0f} s")
    return sw


def analyse_sweep(sw, a):
    """Stokes vector for live feedback (laser-corrected when possible)."""
    kw = dict(qwp_zero_deg=a.qwp_zero_deg,
              retardance_deg=waves_to_deg(a.qwp_retardance_waves), s3_sign=a.s3_sign)
    fit = extract_stokes(corrected_power(sw), sw["measured_deg"], **kw)
    if has_reference(sw):
        raw = extract_stokes(sw["power_W"], sw["measured_deg"], **kw)
        fit["rms_raw"] = raw["residual_rms"] / np.mean(sw["power_W"])
    fit["rms_rel"] = fit["residual_rms"] / np.mean(corrected_power(sw))
    return fit


def laser_line(sw):
    """Laser noise and drift during one sweep, from the reference meter."""
    r = sw["ref_W"]
    t = sw["t_s"] - sw["t_s"][0]
    slope, icpt = np.polyfit(t, r, 1)
    noise = np.std(r - (slope * t + icpt)) / np.mean(r)
    drift = slope * t[-1] / np.mean(r)
    return f"laser: noise {100 * noise:.2f}%, drift {100 * drift:+.2f}% over the sweep"


def fmt(S):
    n = S / S[0]
    return (f"[{n[0]:.3f} {n[1]:+.3f} {n[2]:+.3f} {n[3]:+.3f}]  "
            f"DOP {np.linalg.norm(n[1:]):.3f}")


def coverage(S_list):
    """Rank and condition number of the input states measured so far."""
    S = np.column_stack(S_list)
    sv = np.linalg.svd(S / S[0], compute_uv=False)
    rank = int((sv > 1e-2 * sv[0]).sum())
    return rank, (sv[0] / sv[-1] if len(S_list) >= 4 else float("inf"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwp-motor", default="28000005", dest="qwp_motor")
    ap.add_argument("--qwp-stage", default="none", dest="qwp_stage")
    ap.add_argument("--home", type=lambda s: str(s).lower() in ("1", "true", "yes"),
                    default=True)
    ap.add_argument("--qwp-steps", type=int, default=100, dest="qwp_steps",
                    help="QWP positions per sweep (a count, spread over qwp_sweep_deg)")
    ap.add_argument("--qwp-sweep-deg", type=float, default=360.0, dest="qwp_sweep_deg",
                    help="QWP sweep span: 360 (a full turn, beam walk separated) "
                         "or 180")
    ap.add_argument("--qwp-settle", type=float, default=0.3, dest="qwp_settle")
    ap.add_argument("--qwp-zero-deg", type=float, default=93.6, dest="qwp_zero_deg",
                    help="stage angle at which the QWP is aligned with the analyzer")
    ap.add_argument("--qwp-retardance-waves", type=float, default=0.25,
                    dest="qwp_retardance_waves",
                    help="QWP retardance at the working wavelength, in WAVES "
                         "as the manufacturer quotes it (0.25 = ideal)")
    ap.add_argument("--s3-sign", type=int, default=1, choices=(1, -1), dest="s3_sign")
    ap.add_argument("--wavelength-nm", type=float, default=520.0, dest="wavelength_nm")
    ap.add_argument("--pm-serial", default=None, dest="pm_serial",
                    help="signal meter (after the analyzer); null = the only "
                         "meter that is not the reference")
    ap.add_argument("--pm-average-count", type=int, default=300, dest="pm_average_count",
                    help="averages per reading, on both meters")
    ap.add_argument("--power-range-w", type=float, default=None, dest="power_range_w",
                    help="signal meter range in W; null = auto range")
    ap.add_argument("--ref-pm-serial", default=None, dest="ref_pm_serial",
                    help="reference meter (laser monitor after the polarizer); "
                         "null = no reference")
    ap.add_argument("--ref-power-range-w", type=float, default=None,
                    dest="ref_power_range_w",
                    help="reference meter range in W; null = auto range")
    ap.add_argument("--zero-meter", type=lambda s: str(s).lower() in ("1", "true", "yes"),
                    default=True, dest="zero_meter")
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-state-error-deg", type=float, default=3.0,
                    dest="sim_state_error_deg")
    ap.add_argument("--sim-noise-rel", type=float, default=0.001, dest="sim_noise_rel",
                    help="independent noise of each meter (relative)")
    ap.add_argument("--sim-laser-drift-rel", type=float, default=0.01,
                    dest="sim_laser_drift_rel", help="laser drift amplitude (relative)")
    ap.add_argument("--sim-laser-noise-rel", type=float, default=0.003,
                    dest="sim_laser_noise_rel", help="laser noise (relative)")
    a = parse_with_config(ap, argv, HERE / "measure_mueller.json")

    delta = waves_to_deg(a.qwp_retardance_waves)
    root = Path(a.out) if a.out else Path.home() / "Desktop"
    run = root / time.strftime("mueller_%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    print(f"\nrun folder : {run}")
    print(f"PSA QWP    : zero at {a.qwp_zero_deg:g} deg, retardance "
          f"{a.qwp_retardance_waves:g} waves = {delta:.2f} deg")
    print(f"sweep      : {a.qwp_steps} steps over {a.qwp_sweep_deg:g} deg "
          f"({a.qwp_sweep_deg / a.qwp_steps:.2f} deg each), {a.wavelength_nm:g} nm\n")

    meta = {"kind": "mueller_powermeter",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_file": str(Path(a.config).resolve()),
            "settings": vars(a),
            "calibration": {"qwp_zero_deg": a.qwp_zero_deg,
                            "qwp_retardance_waves": a.qwp_retardance_waves,
                            "s3_sign": a.s3_sign, "analyzer_deg": 0.0,
                            "wavelength_nm": a.wavelength_nm},
            "qwp_sweep_deg": a.qwp_sweep_deg, "notes": a.notes, "states": []}

    rig = SimRig(a) if a.simulate else Rig(a)
    with rig as r:
        def save():
            meta["hardware"] = r.describe()
            (run / "run.json").write_text(json.dumps(meta, indent=2, default=str))

        if a.zero_meter:
            what = ("both power meters (block the laser BEFORE the beamsplitter)"
                    if r.has_ref else "the power meter")
            input(f"\nBLOCK the beam to zero {what}, then press Enter...")
            r.zero_meter()
            input("UNBLOCK the beam, then press Enter...")
        save()

        S_in_list = []
        idx = 0
        while True:
            idx += 1
            print(f"\n=== state {idx} ===")
            label = input("Set the input state by hand, then type a label for it"
                          "\n  (e.g. H, D, R -- approximate is fine; Enter = finish): ").strip()
            if not label:
                rank = coverage(S_in_list)[0] if S_in_list else 0
                if rank < 4:
                    ans = input(f"Only rank {rank} of 4 -- the analysis cannot "
                                f"reconstruct M. Finish anyway? [y/N]: ").strip().lower()
                    if ans != "y":
                        idx -= 1
                        continue
                break
            r.set_state(label)

            sweeps, fits = {}, {}
            for side, prompt, tag in (
                    ("in", "REMOVE the sample, then press Enter to measure the input state...",
                     "input"),
                    ("out", "PUT THE SAMPLE BACK, then press Enter to measure through it...",
                     "output")):
                input(prompt)
                r.set_sample(side == "out")
                sw = sweep(r, a, tag, a.qwp_sweep_deg)
                write_sweep(run / f"state_{idx:02d}_{side}.csv", sw["commanded_deg"],
                            sw["measured_deg"], sw["power_W"], sw["ref_W"], sw["t_s"])
                f = analyse_sweep(sw, a)
                sweeps[side], fits[side] = sw, f
                name = "S_in " if side == "in" else "S_out"
                fit_txt = (f"fit rms {100 * f['rms_raw']:.2f}% raw -> "
                           f"{100 * f['rms_rel']:.2f}% laser-corrected"
                           if "rms_raw" in f else f"fit rms {100 * f['rms_rel']:.2f}%")
                print(f"  {name} = {fmt(f['S'])}   {fit_txt}")
                if has_reference(sw):
                    print(f"          {laser_line(sw)}")
            if has_reference(sweeps["in"]):
                change = np.mean(sweeps["out"]["ref_W"]) / np.mean(sweeps["in"]["ref_W"]) - 1
                print(f"  laser changed {100 * change:+.2f}% between the input and "
                      f"through-sample sweeps (corrected)")

            meta["states"].append({"index": idx, "label": label,
                                   "input": f"state_{idx:02d}_in.csv",
                                   "output": f"state_{idx:02d}_out.csv"})
            save()
            S_in_list.append(fits["in"]["S"])

            rank, cond = coverage(S_in_list)
            if rank < 4:
                hint = ("add a state with a circular component (e.g. R or L)"
                        if len(S_in_list) >= 3 and abs(np.column_stack(S_in_list)[3]).max() < 0.2
                        else "need at least 4 states")
                print(f"  saved. states {len(S_in_list)} | rank {rank} of 4 -- {hint}")
            else:
                verdict = ("good" if cond < 4 else "usable -- spread the states more"
                           if cond < 10 else "POOR -- states too similar")
                print(f"  saved. states {len(S_in_list)} | rank 4, condition "
                      f"number {cond:.2f} ({verdict}; 1.73 is the best possible)")

    print(f"\n{len(meta['states'])} states saved in {run}")
    print("Analyse with: python analyze_mueller.py   (set run_dir in analyze_mueller.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
