#!/usr/bin/env python3
"""Transmittance of a sample with one power meter (the reference PM).

    python measure_transmission.py            # reads measure_transmission.json

Put the power meter directly behind the sample, with no polarizing optics in
between. You are asked for the sample out, in, out, in, ..., out: every "in"
is compared with the average of the "out" just before and just after it, so a
slow drift of the laser cancels.

Output in Desktop/transmission_<date>_<time>/: readings.csv and run.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from mueller_common import parse_with_config

HERE = Path(__file__).resolve().parent


def open_meter(a):
    from thorlabs_powermeter import PowerMeter, list_power_meters
    from measure_mueller import _setup_meter

    serial = a.meter_serial or a.ref_pm_serial
    if not serial:
        raise SystemExit("no meter: set meter_serial (or ref_pm_serial in measure_mueller.json)")
    hits = [d for d in list_power_meters()
            if d.serial_number == serial or str(serial) in d.resource_name]
    if not hits:
        raise SystemExit(f"power meter '{serial}' not found")
    pm = PowerMeter(hits[0].resource_name)
    _setup_meter(pm, a, a.power_range_w)
    print(f"power meter: {pm.identity.serial_number}")
    return pm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=5, help="number of 'sample in' readings")
    ap.add_argument("--readings", type=int, default=20, help="meter readings averaged per step")
    ap.add_argument("--meter-serial", default=None, dest="meter_serial",
                    help="null = the reference meter (ref_pm_serial in measure_mueller.json)")
    ap.add_argument("--power-range-w", type=float, default=None, dest="power_range_w",
                    help="null = auto range")
    ap.add_argument("--ref-pm-serial", default=None, dest="ref_pm_serial")
    ap.add_argument("--wavelength-nm", type=float, default=520.0, dest="wavelength_nm")
    ap.add_argument("--pm-average-count", type=int, default=300, dest="pm_average_count")
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--notes", default="")

    # meter serial, wavelength and averaging from measure_mueller.json
    shared = HERE / "measure_mueller.json"
    if shared.is_file():
        data = json.loads(shared.read_text())
        ap.set_defaults(**{k: data[k] for k in ("ref_pm_serial", "wavelength_nm",
                                                "pm_average_count") if k in data})
    a = parse_with_config(ap, argv, HERE / "measure_transmission.json")

    pm = open_meter(a)
    out = (Path(a.out) if a.out else Path.home() / "Desktop") / time.strftime(
        "transmission_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    print(f"run folder: {out}")
    rows, means = [], []
    try:
        input("\nBLOCK the beam, then press Enter (zeroing the meter)...")
        pm.zero()
        input("UNBLOCK the beam, then press Enter...")

        steps = ["out"] + ["in", "out"] * a.cycles
        for i, what in enumerate(steps, 1):
            input(f"\n[{i}/{len(steps)}] sample {what.upper()}, then press Enter...")
            p = np.array([pm.read_power() for _ in range(a.readings)], float)
            rows += [{"step": i, "sample": what, "power_W": f"{v:.9g}"} for v in p]
            means.append(p.mean())
            print(f"  {what:3s}  {p.mean() * 1e6:9.3f} uW  ± {100 * p.std() / p.mean():.2f}%")
            if what == "out" and i > 1:
                t = means[-2] / (0.5 * (means[-3] + means[-1]))
                print(f"  -> transmittance {t:.4f}")
    finally:
        pm.close()

    T = np.array([means[i] / (0.5 * (means[i - 1] + means[i + 1]))
                  for i in range(1, len(means) - 1, 2)])
    sem = T.std(ddof=1) / np.sqrt(len(T)) if len(T) > 1 else float("nan")
    print(f"\ntransmittance T = {T.mean():.4f} ± {sem:.4f}   "
          f"(cycles: {', '.join(f'{t:.4f}' for t in T)})")

    with open(out / "readings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "sample", "power_W"])
        w.writeheader()
        w.writerows(rows)
    (out / "run.json").write_text(json.dumps(
        {"kind": "transmission", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "settings": vars(a), "step_means_W": means, "T_each": T.tolist(),
         "T": float(T.mean()), "T_sem": float(sem), "notes": a.notes}, indent=2, default=str))
    print(f"saved in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
