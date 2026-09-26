#!/usr/bin/env python3
"""Measure the Mueller matrix of a sample, one hand-made input state at a time.

    python measure_mueller.py                  # reads measure_mueller.json
    python measure_mueller.py --simulate       # rehearse, no hardware

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

Every state is saved the moment it is measured, so an interrupted run keeps
what it has.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from mueller_common import (extract_stokes, parse_with_config, waves_to_deg,
                            write_sweep)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


class Rig:
    """QWP rotation stage + PM100D. set_state / set_sample are no-ops here:
    on the real bench YOU do those by hand."""

    def __init__(self, a):
        self.a = a

    def __enter__(self):
        import kcube
        from thorlabs_powermeter import PowerMeter

        stage = None if str(self.a.qwp_stage).lower() in ("", "none") else self.a.qwp_stage
        self.qwp = kcube.KCube(self.a.qwp_motor, stage_name=stage)
        self.qwp.connect()
        if self.a.home:
            self.qwp.home()

        self.pm = PowerMeter(self.a.pm_resource) if self.a.pm_resource else PowerMeter()
        self.pm.__enter__()
        self.pm.power_unit = "W"
        self.pm.wavelength_nm = self.a.wavelength_nm
        self.pm.average_count = self.a.pm_average_count
        if self.a.power_range_w is None:
            self.pm.auto_range = True
        else:
            self.pm.auto_range = False
            self.pm.power_range_w = self.a.power_range_w
        return self

    def __exit__(self, *exc):
        for obj in (getattr(self, "pm", None), getattr(self, "qwp", None)):
            try:
                obj and obj.close()
            except Exception:
                pass
        return False

    # motion / reading
    def move_to(self, deg):      self.qwp.move_to(float(deg) % 360.0)
    def move_relative(self, d):  self.qwp.move_relative(float(d))
    def position(self):          return float(self.qwp.get_position())
    def read_power(self):        return float(self.pm.read_power())
    def zero_meter(self):        self.pm.zero()
    def set_state(self, label):  pass
    def set_sample(self, inside): pass
    def describe(self):
        return {"backend": "hardware", "qwp_motor": self.a.qwp_motor,
                "qwp_model": getattr(self.qwp, "model", None),
                "power_meter": str(getattr(self.pm, "identity", "")),
                "pm_average_count": self.a.pm_average_count,
                "power_range_w": self.a.power_range_w}


class SimRig:
    """Rehearsal stand-in with a known sample, so a run can be checked.

    Hand-made states are imperfect on purpose (sim_state_error_deg), to show
    that approximate states still give the right M.
    """

    SAMPLE = None      # set in __init__

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

    NOMINAL = {"H": (1, 1, 0, 0), "V": (1, -1, 0, 0), "D": (1, 0, 1, 0),
               "A": (1, 0, -1, 0), "R": (1, 0, 0, 1), "RCP": (1, 0, 0, 1),
               "L": (1, 0, 0, -1), "LCP": (1, 0, 0, -1)}

    def __enter__(self):  return self
    def __exit__(self, *exc): return False

    def set_state(self, label):
        nom = self.NOMINAL.get(label.strip().upper())
        if nom is None:
            v = self.rng.normal(size=3); v /= np.linalg.norm(v)
            nom = (1, *v)
            print(f"  [sim] '{label}' is not H/V/D/A/R/L -- using a random state")
        e = np.deg2rad(self.a.sim_state_error_deg)
        axis = self.rng.uniform(0, np.pi)
        wobble = self._rot(self.rng.normal(0, e)) @ self._ret(self.rng.normal(0, 2 * e), axis)
        self.state = wobble @ np.array(nom, float)          # imperfect, like by hand

    def set_sample(self, inside):  self.inside = bool(inside)
    def move_to(self, deg):        self.pos = float(deg)
    def move_relative(self, d):    self.pos += float(d)
    def position(self):            return self.pos % 360.0
    def zero_meter(self):          pass

    def read_power(self):
        S = (self.sample @ self.state) if self.inside else self.state
        row = self._row(np.deg2rad(self.pos - self.a.qwp_zero_deg), 0.0,
                        np.deg2rad(waves_to_deg(self.a.qwp_retardance_waves)))
        p = self.P0 * float(row @ S)
        return p * (1 + self.rng.normal(0, self.a.sim_noise_rel))

    def describe(self):
        return {"backend": "simulated", "true_sample_mueller": self.sample.tolist(),
                "sim_state_error_deg": self.a.sim_state_error_deg,
                "sim_noise_rel": self.a.sim_noise_rel}


# --------------------------------------------------------------------------- #


def sweep(rig, a, tag):
    """One full QWP sweep; returns (commanded, measured, power, t)."""
    n = int(a.qwp_steps)
    step = 180.0 / n
    commanded = a.qwp_zero_deg + np.arange(n) * step
    measured, power, t = np.empty(n), np.empty(n), np.empty(n)
    rig.move_to(commanded[0])
    t0 = time.time()
    for k in range(n):
        if k:
            rig.move_relative(step)
        time.sleep(a.qwp_settle)
        power[k] = rig.read_power()
        measured[k] = rig.position()
        t[k] = time.time() - t0
        done = int(30 * (k + 1) / n)
        print(f"\r  measuring {tag:6s} [{'#' * done}{'.' * (30 - done)}] {k+1}/{n}",
              end="", flush=True)
    print(f"   {t[-1]:.0f} s")
    return commanded, measured, power, t


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
                    help="QWP positions per sweep (a count, over 180 deg)")
    ap.add_argument("--qwp-settle", type=float, default=0.3, dest="qwp_settle")
    ap.add_argument("--qwp-zero-deg", type=float, default=93.6, dest="qwp_zero_deg",
                    help="stage angle at which the QWP is aligned with the analyzer")
    ap.add_argument("--qwp-retardance-waves", type=float, default=0.25,
                    dest="qwp_retardance_waves",
                    help="QWP retardance at the working wavelength, in WAVES "
                         "as the manufacturer quotes it (0.25 = ideal)")
    ap.add_argument("--s3-sign", type=int, default=1, choices=(1, -1), dest="s3_sign")
    ap.add_argument("--wavelength-nm", type=float, default=520.0, dest="wavelength_nm")
    ap.add_argument("--pm-resource", default=None, dest="pm_resource")
    ap.add_argument("--pm-average-count", type=int, default=300, dest="pm_average_count")
    ap.add_argument("--power-range-w", type=float, default=None, dest="power_range_w",
                    help="fixed meter range in W; null = auto range")
    ap.add_argument("--zero-meter", type=lambda s: str(s).lower() in ("1", "true", "yes"),
                    default=True, dest="zero_meter")
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-state-error-deg", type=float, default=3.0,
                    dest="sim_state_error_deg")
    ap.add_argument("--sim-noise-rel", type=float, default=0.001, dest="sim_noise_rel")
    a = parse_with_config(ap, argv, HERE / "measure_mueller.json")

    delta = waves_to_deg(a.qwp_retardance_waves)
    root = Path(a.out) if a.out else Path.home() / "Desktop"
    run = root / time.strftime("mueller_%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    print(f"\nrun folder : {run}")
    print(f"PSA QWP    : zero at {a.qwp_zero_deg:g} deg, retardance "
          f"{a.qwp_retardance_waves:g} waves = {delta:.2f} deg")
    print(f"sweep      : {a.qwp_steps} steps over 180 deg, {a.wavelength_nm:g} nm")

    meta = {"kind": "mueller_powermeter",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_file": str(Path(a.config).resolve()),
            "settings": vars(a),
            "calibration": {"qwp_zero_deg": a.qwp_zero_deg,
                            "qwp_retardance_waves": a.qwp_retardance_waves,
                            "s3_sign": a.s3_sign, "analyzer_deg": 0.0,
                            "wavelength_nm": a.wavelength_nm},
            "notes": a.notes, "states": []}

    def save():
        (run / "run.json").write_text(json.dumps(meta, indent=2, default=str))

    rig = SimRig(a) if a.simulate else Rig(a)
    with rig as r:
        meta["hardware"] = r.describe()
        if a.zero_meter:
            input("\nBLOCK the beam to zero the power meter, then press Enter...")
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

            input("REMOVE the sample, then press Enter to measure the input state...")
            r.set_sample(False)
            c, m, p, t = sweep(r, a, "input")
            f_in = f"state_{idx:02d}_in.csv"
            write_sweep(run / f_in, c, m, p, t)
            S_in = extract_stokes(p, m, qwp_zero_deg=a.qwp_zero_deg,
                                  retardance_deg=delta, s3_sign=a.s3_sign)["S"]
            print(f"  S_in  = {fmt(S_in)}")

            input("PUT THE SAMPLE BACK, then press Enter to measure through it...")
            r.set_sample(True)
            c, m, p, t = sweep(r, a, "output")
            f_out = f"state_{idx:02d}_out.csv"
            write_sweep(run / f_out, c, m, p, t)
            S_out = extract_stokes(p, m, qwp_zero_deg=a.qwp_zero_deg,
                                   retardance_deg=delta, s3_sign=a.s3_sign)["S"]
            print(f"  S_out = {fmt(S_out)}")

            meta["states"].append({"index": idx, "label": label,
                                   "input": f_in, "output": f_out})
            save()
            S_in_list.append(S_in)

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
