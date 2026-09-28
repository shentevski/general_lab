#!/usr/bin/env python3
"""Transmittance of a sample, measured directly with the two power meters.

    python measure_transmission.py                 # reads measure_transmission.json
    python measure_transmission.py --simulate      # rehearse, no hardware

An independent check of the Mueller M00 that uses no polarimeter at all: the
power through the sample divided by the power without it, each divided by the
reference meter so laser drift cancels.

Optical layout for this measurement

    laser -> polarizer -> beamsplitter -> [sample] -> signal meter
                               '-> reference meter (laser monitor)

The signal meter must collect ALL the light behind the sample: take the
polarimeter's QWP and analyzer out, or put the sensor directly after the
sample. A wave plate changes the polarization, and anything polarizing after
it would turn that into a fake change of transmission. Keep the sample near
normal incidence, the beam centred and well inside the sensor, and the sensor
close to the sample (the plate deviates the beam slightly).

The sequence is  out, in, out, in, ..., out:  every "in" is bracketed by two
"out" readings, so a slow drift of the laser, of the split ratio or of the
meters cancels to first order. Each reading is `readings` samples of both
meters at the same moment; the dark offset (beam blocked) is subtracted.

Output in Desktop/transmission_<date>_<time>/: run.json (settings, hardware,
results), readings.csv (every sample) and transmission.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mueller_common import parse_with_config

HERE = Path(__file__).resolve().parent


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y", "on")


def float_or_none(s):
    return None if str(s).lower() in ("", "none", "null") else float(s)


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


class Meters:
    """Signal + reference PM100D, read at the same moment (the Mueller rig's
    meter handling, without the rotation stage)."""

    def __init__(self, a):
        self.a = a
        self.pm = self.ref = self._pool = None
        self.parallel = True
        self.t0 = time.time()

    def __enter__(self):
        from thorlabs_powermeter import PowerMeter
        from measure_mueller import _pick_meters, _setup_meter

        sig_res, ref_res = _pick_meters(self.a)
        if not ref_res:
            raise SystemExit("set ref_pm_serial in measure_mueller.json: without the "
                             "laser monitor this measurement is only as good as the "
                             "laser's drift")
        self.pm = PowerMeter(sig_res)
        _setup_meter(self.pm, self.a, self.a.power_range_w)
        self.ref = PowerMeter(ref_res)
        _setup_meter(self.ref, self.a, self.a.ref_power_range_w)
        self._pool = ThreadPoolExecutor(max_workers=1)
        print(f"signal meter    : {self.pm.identity.serial_number}")
        print(f"reference meter : {self.ref.identity.serial_number}")
        return self

    def __exit__(self, *exc):
        if self._pool:
            self._pool.shutdown(wait=True)
        for obj in (self.pm, self.ref):
            try:
                obj and obj.close()
            except Exception:
                pass
        return False

    def read(self):
        if self.parallel:
            try:
                fut = self._pool.submit(self.ref.read_power)
                sig = self.pm.read_power()
                return float(sig), float(fut.result())
            except Exception as e:
                print(f"\n  parallel reading failed ({e}); reading one after the other")
                self.parallel = False
        r1 = self.ref.read_power()
        sig = self.pm.read_power()
        r2 = self.ref.read_power()
        return float(sig), 0.5 * (r1 + r2)

    def zero(self):
        self.pm.zero()
        self.ref.zero()

    def now(self):
        return time.time() - self.t0

    def set_state(self, what):  pass          # you do it by hand

    def describe(self):
        return {"backend": "hardware", "signal_meter": str(self.pm.identity),
                "reference_meter": str(self.ref.identity),
                "reference_read": "parallel" if self.parallel else "sequential",
                "pm_average_count": self.a.pm_average_count,
                "power_range_w": self.a.power_range_w,
                "ref_power_range_w": self.a.ref_power_range_w}


class SimMeters:
    """Rehearsal stand-in: a sample of known transmittance, a drifting laser, a
    small dark offset on both meters, and a virtual clock."""

    READ_S = 0.3
    SWAP_S = 15.0
    DRIFT_PERIOD_S = 900.0

    def __init__(self, a):
        self.a = a
        self.rng = np.random.default_rng(2)
        self.clock = 0.0
        self.state = "dark"
        self.dark = (2e-7, 1e-7)                         # W, signal / reference
        self.parallel = True

    def __enter__(self):
        print(f"signal meter    : simulated (true transmittance {self.a.sim_transmission:g})")
        print("reference meter : simulated")
        return self

    def __exit__(self, *exc):
        return False

    def set_state(self, what):
        self.clock += self.SWAP_S
        self.state = what

    def read(self):
        self.clock += self.READ_S
        laser = 2.2e-4 * (1 + self.a.sim_laser_drift_rel
                          * np.sin(2 * np.pi * self.clock / self.DRIFT_PERIOD_S))
        laser *= 1 + self.rng.normal(0, self.a.sim_laser_noise_rel)
        if self.state == "dark":
            return (self.dark[0] + self.rng.normal(0, 2e-9),
                    self.dark[1] + self.rng.normal(0, 2e-9))
        t = self.a.sim_transmission if self.state == "in" else 1.0
        noise = lambda: 1 + self.rng.normal(0, self.a.sim_noise_rel)
        return (laser * t * noise() + self.dark[0], 0.95 * laser * noise() + self.dark[1])

    def zero(self):
        pass

    def now(self):
        return self.clock

    def describe(self):
        return {"backend": "simulated", "true_transmittance": self.a.sim_transmission,
                "sim_laser_drift_rel": self.a.sim_laser_drift_rel,
                "sim_laser_noise_rel": self.a.sim_laser_noise_rel,
                "sim_noise_rel": self.a.sim_noise_rel}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def reading(m, a, what, rows, block):
    """`a.readings` samples of both meters; returns (signal, reference) arrays."""
    m.set_state(what)
    if a.settle_s:
        time.sleep(a.settle_s if not a.simulate else 0)
    sig, ref = np.empty(a.readings), np.empty(a.readings)
    for k in range(a.readings):
        sig[k], ref[k] = m.read()
        rows.append({"block": block, "what": what, "k": k, "t_s": m.now(),
                     "power_W": sig[k], "ref_W": ref[k]})
    return sig, ref


def block_ratio(b, dark):
    """Dark-subtracted signal / reference of one reading."""
    return float(np.mean(b["sig"] - dark[0]) / np.mean(b["ref"] - dark[1]))


def analyse(blocks, dark):
    """Transmittance per 'in', each against the mean of the 'out' before and after."""
    ratio = [block_ratio(b, dark) for b in blocks]
    raw = [np.mean(b["sig"] - dark[0]) for b in blocks]
    T, Traw, drift = [], [], []
    for i in range(1, len(blocks) - 1, 2):           # blocks: out, in, out, in, ..., out
        before, now, after = ratio[i - 1], ratio[i], ratio[i + 1]
        T.append(now / (0.5 * (before + after)))
        Traw.append(raw[i] / (0.5 * (raw[i - 1] + raw[i + 1])))
        drift.append(after / before - 1)
    T, Traw = np.array(T), np.array(Traw)
    sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return {"T_each": T.tolist(), "T": float(T.mean()), "T_sem": sem(T),
            "T_spread": float(np.std(T, ddof=1)) if len(T) > 1 else float("nan"),
            "T_raw_each": Traw.tolist(), "T_raw": float(Traw.mean()), "T_raw_sem": sem(Traw),
            "out_ratio_drift": drift, "ratios": ratio,
            "laser_change": float(np.ptp([np.mean(b["ref"]) for b in blocks])
                                  / np.mean([np.mean(b["ref"]) for b in blocks]))}


def figure(blocks, dark, res, out, a):
    import matplotlib
    if not a.show_figures:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds, dr = dark
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, b in enumerate(blocks):
        y = (b["sig"] - ds) / (b["ref"] - dr)
        x = np.arange(len(y)) + i * (len(y) + 3)
        ax.plot(x, y, "o", ms=4, color="#2a78d6" if b["what"] == "out" else "#eb6834",
                label={"out": "sample out", "in": "sample in"}[b["what"]] if i < 2 else None)
    ax.set_xticks([])
    ax.set_ylabel("signal / reference")
    ax.set_title(f"T = {res['T']:.4f} ± {res['T_sem']:.4f}  (without the reference: "
                 f"{res['T_raw']:.4f} ± {res['T_raw_sem']:.4f})", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    if a.save_figures:
        fig.savefig(out / "transmission.png", dpi=130)
    if a.show_figures:
        plt.show()
    plt.close(fig)


def compare_with_mueller(run_dir):
    """M00 of a Mueller run, from its analysis (any analysis* folder, newest first)."""
    run = Path(run_dir)
    for res in sorted(run.glob("analysis*/results.json"), key=lambda p: p.stat().st_mtime,
                      reverse=True):
        r = json.loads(res.read_text())
        p = r["parameters"]["M00"]
        return {"file": str(res), "M00": p["value"], "stat": p["stat"], "sys": p["sys"]}
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=5,
                    help="number of 'in' readings; each is bracketed by 'out' readings")
    ap.add_argument("--readings", type=int, default=20,
                    help="samples of both meters per reading")
    ap.add_argument("--settle-s", type=float, default=1.0, dest="settle_s",
                    help="wait after each Enter before reading")
    ap.add_argument("--measure-dark", type=str2bool, default=True, dest="measure_dark",
                    help="read both meters with the beam blocked and subtract it")
    ap.add_argument("--compare-run", default=None, dest="compare_run",
                    help="a Mueller run folder: compare with its M00")
    ap.add_argument("--show-figures", type=str2bool, default=True, dest="show_figures")
    ap.add_argument("--save-figures", type=str2bool, default=True, dest="save_figures")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-transmission", type=float, default=0.973, dest="sim_transmission")
    ap.add_argument("--sim-laser-drift-rel", type=float, default=0.02, dest="sim_laser_drift_rel")
    ap.add_argument("--sim-laser-noise-rel", type=float, default=0.002, dest="sim_laser_noise_rel")
    ap.add_argument("--sim-noise-rel", type=float, default=0.0005, dest="sim_noise_rel")
    # hardware: same keys as measure_mueller.py
    ap.add_argument("--wavelength-nm", type=float, default=520.0, dest="wavelength_nm")
    ap.add_argument("--pm-serial", default=None, dest="pm_serial")
    ap.add_argument("--pm-average-count", type=int, default=300, dest="pm_average_count")
    ap.add_argument("--power-range-w", type=float_or_none, default=None, dest="power_range_w")
    ap.add_argument("--ref-pm-serial", default=None, dest="ref_pm_serial")
    ap.add_argument("--ref-power-range-w", type=float_or_none, default=None,
                    dest="ref_power_range_w")
    ap.add_argument("--zero-meter", type=str2bool, default=True, dest="zero_meter")
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--notes", default="")

    # measure_mueller.json first (meter serials, wavelength, ranges), then this
    # script's own JSON, then the command line
    shared = HERE / "measure_mueller.json"
    if shared.is_file():
        known = {x.dest for x in ap._actions}
        data = json.loads(shared.read_text())
        ap.set_defaults(**{k: v for k, v in data.items() if k in known})
        print(f"meters + wavelength: {shared}")
    a = parse_with_config(ap, argv, HERE / "measure_transmission.json")
    if a.cycles < 1 or a.readings < 2:
        raise SystemExit("cycles must be >= 1 and readings >= 2")

    root = Path(a.out) if a.out else Path.home() / "Desktop"
    out = root / time.strftime("transmission_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    print(f"\nrun folder : {out}")
    print(f"sequence   : out, in x{a.cycles} (each bracketed by out), {a.readings} samples "
          f"per reading, {a.wavelength_nm:g} nm")
    print("\nThe signal meter must see ALL the light behind the sample: polarimeter QWP "
          "and analyzer OUT of the beam.")

    rows, blocks = [], []
    dark = (0.0, 0.0)
    m = SimMeters(a) if a.simulate else Meters(a)
    with m:
        if a.zero_meter or a.measure_dark:
            input("\nBLOCK the laser BEFORE the beamsplitter, then press Enter...")
            if a.zero_meter:
                m.zero()
            if a.measure_dark:
                s, r = reading(m, a, "dark", rows, 0)
                dark = (float(np.mean(s)), float(np.mean(r)))
                print(f"  dark: signal {dark[0] * 1e9:+.1f} nW, reference {dark[1] * 1e9:+.1f} nW")
            input("UNBLOCK the beam, then press Enter...")

        order = ["out"] + ["in", "out"] * a.cycles
        for i, what in enumerate(order, 1):
            prompt = ("take the sample OUT" if what == "out" else "put the sample IN")
            input(f"\n[{i}/{len(order)}] {prompt}, then press Enter...")
            s, r = reading(m, a, what, rows, i)
            blocks.append({"what": what, "sig": s, "ref": r})
            print(f"  {what:3s}  signal {np.mean(s) * 1e6:8.3f} uW ± {100 * np.std(s) / np.mean(s):.2f}%  "
                  f"reference {np.mean(r) * 1e6:8.3f} uW   ratio {block_ratio(blocks[-1], dark):.5f}")
            if what == "out" and i > 1:                # an 'in' is now bracketed
                before, now, after = (block_ratio(b, dark) for b in blocks[-3:])
                print(f"       -> transmittance of this cycle {now / (0.5 * (before + after)):.4f}")
        hardware = m.describe()

    res = analyse(blocks, dark)
    print("\nRESULT")
    for k, (t, tr) in enumerate(zip(res["T_each"], res["T_raw_each"]), 1):
        print(f"  cycle {k}:  T = {t:.4f}   (without reference {tr:.4f})   "
              f"out-ratio drift {100 * res['out_ratio_drift'][k - 1]:+.2f}%")
    print(f"\n  transmittance      T = {res['T']:.4f} ± {_n(res['T_sem'])} (stat)   "
          f"spread of cycles {_n(res['T_spread'])}")
    print(f"  without reference  T = {res['T_raw']:.4f} ± {_n(res['T_raw_sem'])}   "
          f"(laser changed {100 * res['laser_change']:.2f}% during the run)")
    if a.simulate:
        print(f"  simulated true value {a.sim_transmission:.4f}")
    cmp = compare_with_mueller(a.compare_run) if a.compare_run else None
    if cmp:
        diff = res["T"] - cmp["M00"]
        err = np.hypot(res["T_sem"] if np.isfinite(res["T_sem"]) else 0, cmp["stat"])
        print(f"\n  Mueller M00 = {cmp['M00']:.4f} ± {cmp['stat']:.4f}  ({cmp['file']})")
        print(f"  direct - M00 = {diff:+.4f}  ({abs(diff) / err:.1f} x the combined stat error)")
    elif a.compare_run:
        print(f"\n  no analysis*/results.json found in {a.compare_run}")

    with open(out / "readings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "what", "k", "t_s", "power_W", "ref_W"])
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in row.items()})
    meta = {"kind": "transmission", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_file": str(Path(a.config).resolve()), "settings": vars(a),
            "hardware": hardware, "dark_W": {"signal": dark[0], "reference": dark[1]},
            "results": res, "compare": cmp, "notes": a.notes}
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    if a.save_figures or a.show_figures:
        figure(blocks, dark, res, out, a)
    print(f"\nsaved in {out}")
    return 0


def _n(x):
    return f"{x:.4f}" if np.isfinite(x) else "-"


if __name__ == "__main__":
    sys.exit(main())
