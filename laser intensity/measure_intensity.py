#!/usr/bin/env python3
"""
measure_intensity.py
====================
Record laser power against time with a Thorlabs PM-series meter and save the
run to the Desktop.

    python measure_intensity.py --wavelength_nm 520 --duration_s 3600 --interval_s 0.02

Writes into ``~/Desktop/power_meter_data/``:

    power_<stamp>_<nm>.csv        elapsed_s, power_W
    power_<stamp>_<nm>_dark.csv   the beam-blocked segment
    power_<stamp>_<nm>_meta.json  instrument state

then::

    python analyze_stability.py

Two things you set, one the script sets:

* **The wavelength**, here. The console keeps whatever was set last, and a
  reading at the wrong wavelength is silently wrong, so ``--wavelength_nm``
  is required.
* **The bandwidth, on the console's front panel: Meas Config -> BW -> LO.**
  The driver cannot read or set it. LO is ~15 Hz, which is what Thorlabs
  recommends for photodiode heads and what the analysis assumes.
* **Auto-range**, which the script turns on, exactly as Thorlabs' own example
  does.

The script deliberately does not touch the averaging count: Thorlabs' example
never sets it, and a large value makes one reading outlast the driver's read
timeout. Whatever the console is set to is recorded in the metadata.

The beam-blocked segment is not optional politeness. It is the only thing that
separates *the laser is noisy* from *the meter is noisy*.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from thorlabs_powermeter import PowerMeter, PowerMeterError, list_power_meters


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Log laser power against time and save it to the Desktop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wavelength_nm", "--wavelength-nm", type=float,
                        required=True, dest="wavelength_nm", metavar="NM",
                        help="operating wavelength; sets the responsivity (required)")
    parser.add_argument("--duration_s", type=float, default=600.0, metavar="S",
                        help="how long to log for")
    parser.add_argument("--interval_s", type=float, default=0.02, metavar="S",
                        help="gap between readings; 0.02 covers everything a "
                             "console on LO can show you")
    parser.add_argument("--dark_s", type=float, default=10.0, metavar="S",
                        help="length of the beam-blocked segment; 0 skips it")
    parser.add_argument("--out_dir", type=Path,
                        default=Path.home() / "Desktop" / "power_meter_data",
                        metavar="DIR", help="where the run is saved")
    parser.add_argument("--label", default="", metavar="TEXT",
                        help="tag appended to the filenames")
    parser.add_argument("--note", default="", metavar="TEXT",
                        help="free text stored in the metadata")
    parser.add_argument("--list", action="store_true",
                        help="list attached meters and exit")
    return parser.parse_args(argv)


def run_stem(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = ["power", stamp, f"{args.wavelength_nm:g}nm"]
    if args.label:
        parts.append("".join(c if c.isalnum() or c in "-_" else "_"
                             for c in args.label))
    return "_".join(parts)


def sample_series(pm, duration_s, interval_s, csv_path, header_lines, label):
    """Log power to ``csv_path`` as it is measured and return the arrays.

    Rows are written and flushed as they arrive, so a Ctrl-C or a dropped USB
    cable three hours into a run still leaves usable data on disk.
    """
    times: list[float] = []
    powers: list[float] = []
    overwrite = sys.stdout.isatty()
    last_flush = last_print = 0.0
    interrupted = False

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(f"# {line}\n")
        handle.write("elapsed_s,power_W\n")

        start = time.perf_counter()
        next_sample = start
        try:
            while True:
                elapsed = time.perf_counter() - start
                if elapsed > duration_s:
                    break

                power = pm.read_power()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                powers.append(power)
                handle.write(f"{elapsed:.4f},{power:.6e}\n")

                if elapsed - last_flush > 2.0:
                    handle.flush()
                    last_flush = elapsed
                if elapsed - last_print > (0.25 if overwrite else 2.0):
                    line = (f"  {label} {elapsed:7.1f}/{duration_s:.0f} s"
                            f"   now {power * 1e6:10.4f} uW"
                            f"   mean {np.mean(powers) * 1e6:10.4f} uW"
                            f"   [{len(powers)} samples]")
                    print(f"\r{line}" if overwrite else line,
                          end="" if overwrite else "\n", flush=True)
                    last_print = elapsed

                if interval_s > 0:
                    next_sample += interval_s
                    # Wake at the end of the segment if that comes first, so a
                    # long interval cannot overrun a short segment.
                    remaining = min(next_sample, start + duration_s) - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            interrupted = True
            print("\n  Interrupted -- keeping what was measured so far.")

    print()
    return np.array(times), np.array(powers), interrupted


def describe(pm):
    """Instrument state worth freezing into the metadata.

    Every optional query is guarded: a console that refuses one should cost you
    a field in the JSON, not the run you just spent an hour on.
    """
    def maybe(read, default=None):
        try:
            return read()
        except Exception:
            return default

    identity, sensor = pm.identity, pm.sensor
    return {
        "resource_name": pm.resource_name,
        "model": identity.model,
        "serial_number": identity.serial_number,
        "firmware": identity.firmware,
        "sensor_name": sensor.name,
        "sensor_serial": sensor.serial_number,
        "sensor_type": sensor.type_name,
        "calibration_message": sensor.calibration_message,
        "wavelength_nm": maybe(lambda: pm.wavelength_nm),
        "power_unit": maybe(lambda: pm.power_unit, "W"),
        "average_count": maybe(lambda: pm.average_count),
        "auto_range": maybe(lambda: pm.auto_range),
        "power_range_w": maybe(lambda: pm.power_range_w),
        # The raw photocurrent behind the reading; the analysis turns it into a
        # shot-noise floor.
        "photocurrent_a": maybe(lambda: pm.read_current()),
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        found = list_power_meters()
        print("Attached power meters:")
        for device in found:
            print("  ", device, f"[{device.resource_name}]")
        if not found:
            print("   (none)")
        return 0

    if args.interval_s >= args.duration_s:
        print(f"--interval_s is {args.interval_s:g} s but --duration_s is only "
              f"{args.duration_s:g} s, so the run would be a single sample. "
              f"Did you mean --duration_s {args.interval_s:g}?", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_stem(args)
    power_csv = args.out_dir / f"{stem}.csv"
    dark_csv = args.out_dir / f"{stem}_dark.csv"
    meta_json = args.out_dir / f"{stem}_meta.json"

    with PowerMeter() as pm:
        # Thorlabs' own example sleeps 2 s between opening the device and
        # configuring it: the console reads the sensor's non-volatile memory on
        # connect and is not ready straight away.
        time.sleep(2.0)

        sensor = pm.sensor
        print(f"Connected to {pm.identity.model} (S/N {pm.identity.serial_number})")
        print(f"Sensor:      {sensor}")
        print(f"Calibration: {sensor.calibration_message}")

        pm.wavelength_nm = args.wavelength_nm
        pm.auto_range = True
        pm.power_unit = "W"

        low, high = pm.wavelength_range_nm
        if not low <= args.wavelength_nm <= high:
            print(f"\n!! {args.wavelength_nm:g} nm is outside this sensor's "
                  f"calibrated range ({low:.0f}-{high:.0f} nm). The console will "
                  f"extrapolate and the reading is not traceable.\n")
        print(f"Wavelength:  {pm.wavelength_nm:.1f} nm")
        print(f"Reading:     {pm.read_power() * 1e6:.4f} uW")

        dark_mean = dark_std = None
        if args.dark_s > 0:
            input("\nBlock the beam -- light-tight, not just a hand -- then press Enter: ")
            print("  Letting auto-range settle on the dark level...", flush=True)
            for _ in range(10):
                settled = pm.read_power()
            print(f"  Blocked reading: {settled * 1e9:+.4f} nW", flush=True)

            print(f"  Measuring the dark level for {args.dark_s:.0f} s...", flush=True)
            _, dark_powers, _ = sample_series(
                pm, args.dark_s, args.interval_s, dark_csv,
                ["segment: dark (beam blocked)",
                 f"wavelength_nm: {args.wavelength_nm:g}"], "dark")
            if dark_powers.size > 1:
                dark_mean = float(np.mean(dark_powers))
                dark_std = float(np.std(dark_powers, ddof=1))
                print(f"  Dark offset {dark_mean * 1e9:+.3f} nW, "
                      f"noise {dark_std * 1e9:.3f} nW rms")
            input("Unblock the beam and press Enter to start: ")

        meta = describe(pm)
        meta.update({
            "script": "measure_intensity.py",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "requested_duration_s": args.duration_s,
            "requested_interval_s": args.interval_s,
            "note": args.note,
            "dark_file": dark_csv.name if dark_mean is not None else None,
            "dark_mean_w": dark_mean,
            "dark_std_w": dark_std,
        })

        print(f"\nLogging for {args.duration_s:.0f} s "
              f"(Ctrl-C stops early and keeps the data)...")
        header = [f"{key}: {meta[key]}" for key in
                  ("started_at", "model", "sensor_name", "calibration_message",
                   "wavelength_nm", "average_count", "power_range_w", "note")
                  if meta.get(key) not in (None, "")]
        times, powers, interrupted = sample_series(
            pm, args.duration_s, args.interval_s, power_csv, header, "run")

    if powers.size == 0:
        print("No samples recorded.")
        return 1

    meta.update({
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "interrupted": interrupted,
        "n_samples": int(powers.size),
        "actual_duration_s": float(times[-1]),
        "mean_w": float(np.mean(powers)),
        "power_file": power_csv.name,
    })
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mean = meta["mean_w"]
    print(f"\n{powers.size} samples over {times[-1]:.1f} s")
    print(f"Mean  {mean * 1e6:.4f} uW    rms spread "
          f"{np.std(powers, ddof=1) / mean * 100:.3f} % (includes drift)")
    if dark_std:
        print(f"Detector noise floor {dark_std / mean * 100:.4f} % of this level")
    print(f"\nSaved {power_csv}")
    if dark_mean is not None:
        print(f"      {dark_csv}")
    print(f"      {meta_json}")
    print(f"\nAnalyse it with:\n  python analyze_stability.py \"{power_csv}\"")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PowerMeterError as error:
        print(f"\nPower meter error: {error}", file=sys.stderr)
        sys.exit(1)
