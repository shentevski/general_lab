#!/usr/bin/env python3
"""
measure_intensity.py
====================
Record laser power against time with a Thorlabs PM-series meter and save the
run to the Desktop, along with everything ``analyze_stability.py`` needs to put
an uncertainty on the number.

    python measure_intensity.py --wavelength_nm 450 --duration_s 600

Writes into ``~/Desktop/power_meter_data/``:

    power_<stamp>_<label>.csv        elapsed_s, power_W, head_temp_C
    power_<stamp>_<label>_dark.csv   the beam-blocked segment (if taken)
    power_<stamp>_<label>_meta.json  instrument state, sensor, settings

then::

    python analyze_stability.py

Three things decide whether the data is worth analysing:

* **The wavelength.** The console keeps whatever was set last -- by the GUI, by
  a previous script -- and a reading at the wrong wavelength is silently wrong,
  not an error. That is why ``--wavelength_nm`` is required.
* **The range.** Auto-range is on by default, exactly as in
  ``examples/log_power.py`` -- the setup here makes no console call that the
  working example scripts do not make. The cost is that a range switch mid-run
  puts a step in the data that reads back as "drift"; ``--lock_range`` freezes
  the range to avoid that, and ``--range_w`` pins one outright. Try those only
  if the plain run works first.
* **The averaging.** This script does not set it. Thorlabs' own PMxxx ctypes
  example never does either -- it opens the device, waits, sets wavelength,
  auto-range and unit, and measures. Whatever the console's front panel is set
  to is what you get, and it is recorded in the metadata. ``--average_count``
  will override it, but a large value makes one reading outlast the driver's
  read timeout, and every call after that fails with "a previous response is
  still pending".
* **The console's own bandwidth filter.** The PM100D has a separate HI/LO
  analogue bandwidth setting on the front panel (Meas Config), and Thorlabs
  recommends LO for photodiode heads. The driver cannot read or set it, so it
  is invisible here -- check the panel and record it with ``--note``.

The beam-blocked segment is not optional politeness: it is the only thing that
separates *the laser is noisy* from *the meter is noisy*, and the analysis
script uses it as the detector noise floor.
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

SCHEMA_VERSION = 1

# PM100D manual, utility software section: "a rate of 3000 averages the
# incoming measurement values for approx. 1 second" -- 3000 hardware samples
# per second, so average_count / 3000 is the integration time.
HARDWARE_SAMPLES_PER_S = 3000.0

# ------------------------------------------------------------------ plumbing


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Log laser power against time and save it to the Desktop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--wavelength_nm", "--wavelength-nm", type=float, required=True,
        dest="wavelength_nm", metavar="NM",
        help="operating wavelength; sets the sensor responsivity (required)",
    )
    parser.add_argument(
        "--duration_s", type=float, default=300.0, metavar="S",
        help="how long to log for",
    )
    parser.add_argument(
        "--interval_s", type=float, default=0.1, metavar="S",
        help="target spacing between samples; 0 means as fast as the meter goes",
    )
    parser.add_argument(
        "--average_count", type=int, default=None, metavar="N",
        help="override the console's averaging. Left alone by default: "
             "Thorlabs' own example never sets it, and a large value makes one "
             "read outlast the driver timeout",
    )

    parser.add_argument(
        "--out_dir", type=Path, default=Path.home() / "Desktop" / "power_meter_data",
        metavar="DIR", help="where the run is saved",
    )
    parser.add_argument(
        "--label", default="", metavar="TEXT",
        help="tag appended to the filenames, e.g. 'after_realign'",
    )
    parser.add_argument(
        "--note", default="", metavar="TEXT",
        help="free-text note stored in the metadata (what was on the bench)",
    )

    parser.add_argument(
        "--dark_s", type=float, default=10.0, metavar="S",
        help="length of the beam-blocked segment; 0 skips it",
    )
    parser.add_argument(
        "--zero", action="store_true",
        help="also run the console's dark-offset adjustment. Off by default: the "
             "driver's routine reports success it cannot verify and can hang the "
             "console. Zero from the front panel if you want it, once",
    )
    parser.add_argument(
        "--no_prompt", action="store_true",
        help="never wait for the keyboard; skips the whole blocked-beam phase",
    )
    parser.add_argument(
        "--warmup_s", type=float, default=0.0, metavar="S",
        help="settle for this long after unblocking, before logging starts",
    )

    parser.add_argument(
        "--lock_range", action="store_true",
        help="freeze the range where auto-range lands, which avoids range-switch "
             "steps on long runs. Off by default: it needs a driver call the "
             "working example scripts never make",
    )
    parser.add_argument(
        "--range_w", type=float, default=None, metavar="W",
        help="pin the range ceiling instead of locking whatever auto-range picks",
    )
    parser.add_argument(
        "--attenuation_db", type=float, default=None, metavar="DB",
        help="correction for external optics ahead of the head",
    )
    parser.add_argument(
        "--beam_diameter_mm", type=float, default=None, metavar="MM",
        help="beam diameter, only used for the power-density figure",
    )

    parser.add_argument(
        "--resource", default=None, metavar="NAME",
        help="VISA resource name; default is the first meter found",
    )
    parser.add_argument("--channel", type=int, default=1, help="console input channel")
    parser.add_argument(
        "--temperature_every_s", type=float, default=2.0, metavar="S",
        help="how often to read head temperature; 0 disables it",
    )
    parser.add_argument(
        "--list", action="store_true", help="list attached meters and exit",
    )

    return parser.parse_args(argv)


def run_stem(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = ["power", stamp, f"{args.wavelength_nm:g}nm"]
    if args.label:
        # Keep the filename shell- and Finder-friendly.
        clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.label)
        parts.append(clean)
    return "_".join(parts)


def ask(prompt: str, enabled: bool) -> None:
    if enabled:
        input(prompt)


# --------------------------------------------------------------- acquisition


def sample_series(pm, duration_s, interval_s, temperature_every_s, csv_path,
                  header_lines, progress_label, read_temperature=False):
    """Log power to ``csv_path`` as it is measured and return the arrays.

    Rows are written and flushed as they arrive, so a Ctrl-C or a dropped USB
    cable three hours into a drift run still leaves usable data on disk.
    """
    times: list[float] = []
    powers: list[float] = []
    temps: list[float] = []

    overwrite = sys.stdout.isatty()
    next_temperature = 0.0
    last_flush = 0.0
    last_print = 0.0
    interrupted = False

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(f"# {line}\n")
        handle.write("elapsed_s,power_W,head_temp_C\n")

        start = time.perf_counter()
        next_sample = start
        try:
            while True:
                now = time.perf_counter()
                elapsed = now - start
                if elapsed > duration_s:
                    break

                power = pm.read_power()
                elapsed = time.perf_counter() - start

                temperature = float("nan")
                if read_temperature and elapsed >= next_temperature:
                    temperature = pm.read_head_temperature()
                    next_temperature = elapsed + temperature_every_s

                times.append(elapsed)
                powers.append(power)
                temps.append(temperature)

                temp_text = "" if np.isnan(temperature) else f"{temperature:.3f}"
                handle.write(f"{elapsed:.4f},{power:.6e},{temp_text}\n")
                if elapsed - last_flush > 2.0:
                    handle.flush()
                    last_flush = elapsed

                if elapsed - last_print > (0.25 if overwrite else 2.0):
                    mean = float(np.mean(powers))
                    line = (f"  {progress_label} {elapsed:7.1f}/{duration_s:.0f} s"
                            f"   now {power * 1e6:10.4f} uW"
                            f"   mean {mean * 1e6:10.4f} uW"
                            f"   [{len(powers)} samples]")
                    print(f"\r{line}" if overwrite else line,
                          end="" if overwrite else "\n", flush=True)
                    last_print = elapsed

                if interval_s > 0:
                    next_sample += interval_s
                    # Wake at the end of the segment if that comes first.
                    wake = min(next_sample, start + duration_s)
                    remaining = wake - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            interrupted = True
            print("\n  Interrupted -- keeping what was measured so far.")

    print()
    return np.array(times), np.array(powers), np.array(temps), interrupted


def describe(pm, args):
    """Instrument state worth freezing into the metadata.

    Metadata is nice to have; the measurement is not. Every optional query is
    guarded, so a console that refuses one of them costs you a field in the
    JSON rather than the run you just spent an hour on.
    """
    identity = pm.identity
    sensor = pm.sensor

    def maybe(read, default=None):
        try:
            return read()
        except Exception:
            return default

    low, high = maybe(lambda: pm.wavelength_range_nm, (None, None))

    info = {
        "resource_name": pm.resource_name,
        "channel": pm.channel,
        "manufacturer": identity.manufacturer,
        "model": identity.model,
        "serial_number": identity.serial_number,
        "firmware": identity.firmware,
        "sensor_name": sensor.name,
        "sensor_serial": sensor.serial_number,
        "sensor_type": sensor.type_name,
        "sensor_subtype": sensor.subtype_name,
        "sensor_is_power": sensor.is_power_sensor,
        "sensor_has_temperature": sensor.has_temperature_sensor,
        # Already part of the sensor read above -- do not query the console again.
        "calibration_message": sensor.calibration_message,
        "wavelength_nm": maybe(lambda: pm.wavelength_nm),
        "wavelength_range_nm": [low, high],
        "average_count": maybe(lambda: pm.average_count, 1),
        "power_unit": maybe(lambda: pm.power_unit, "W"),
        "auto_range": maybe(lambda: pm.auto_range),
        "power_range_w": maybe(lambda: pm.power_range_w),
        "attenuation_db": maybe(lambda: pm.attenuation_db),
        "beam_diameter_mm": maybe(lambda: pm.beam_diameter_mm),
        # The raw photocurrent behind the power reading; the analysis turns it
        # into a shot-noise floor.
        "photocurrent_a": maybe(lambda: pm.read_current()),
    }
    info["integration_time_s"] = info["average_count"] / HARDWARE_SAMPLES_PER_S
    return info


def header_lines_from(meta: dict) -> list[str]:
    """Mirror the key metadata into CSV comments, so the CSV stands alone."""
    keys = [
        "started_at", "model", "serial_number", "sensor_name", "sensor_serial",
        "calibration_message", "wavelength_nm", "power_unit", "average_count",
        "auto_range", "power_range_w", "attenuation_db", "note",
    ]
    lines = [f"{key}: {meta[key]}" for key in keys if meta.get(key) not in (None, "")]
    lines.append("columns: elapsed_s, power_W, head_temp_C (blank when not sampled)")
    return lines


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

    prompts = not args.no_prompt

    with PowerMeter(args.resource, channel=args.channel) as pm:
        # Thorlabs' example sleeps 2 s between opening the device and
        # configuring it: the console reads the sensor's non-volatile memory on
        # connect and is not ready straight away. Commands sent into that gap
        # are a good way to desync the response queue.
        time.sleep(2.0)

        identity = pm.identity
        sensor = pm.sensor
        print(f"Connected to {identity.model} (S/N {identity.serial_number})")
        print(f"Sensor:      {sensor}")
        print(f"Calibration: {sensor.calibration_message}")

        # Only heads that report a thermistor may be asked for a temperature.
        log_temperature = (sensor.has_temperature_sensor
                           and args.temperature_every_s > 0)
        if args.temperature_every_s > 0 and not sensor.has_temperature_sensor:
            print("             (no thermistor in this head -- temperature not logged)")

        pm.wavelength_nm = args.wavelength_nm
        pm.auto_range = True
        pm.power_unit = "W"
        if args.average_count is not None:
            pm.average_count = args.average_count

        average_count = pm.average_count
        integration_s = average_count / HARDWARE_SAMPLES_PER_S
        duty = integration_s / args.interval_s if args.interval_s > 0 else 1.0
        print(f"Averaging:   {average_count} samples (console setting) = "
              f"{integration_s * 1e3:.1f} ms per reading, taken every "
              f"{args.interval_s * 1e3:.0f} ms ({duty * 100:.0f} % of the time "
              f"integrating)")
        if args.attenuation_db is not None:
            pm.attenuation_db = args.attenuation_db
        if args.beam_diameter_mm is not None:
            pm.beam_diameter_mm = args.beam_diameter_mm

        low, high = pm.wavelength_range_nm
        if not low <= args.wavelength_nm <= high:
            print(
                f"\n!! {args.wavelength_nm:g} nm is outside this sensor's "
                f"calibrated range ({low:.0f}-{high:.0f} nm). The console will "
                f"extrapolate the responsivity and the reading is not traceable.\n"
            )
        print(f"Wavelength:  {pm.wavelength_nm:.1f} nm")

        # --- range FIRST, with the beam still on. The zero offset belongs to
        # the range it was measured on, so zeroing while auto-range has hunted
        # down to a sensitive range (which is what a blocked beam makes it do)
        # and then measuring on a coarser one leaves the console applying the
        # wrong offset -- negative readings and a blinking ZERO! warning.
        if args.range_w is not None:
            pm.power_range_w = args.range_w          # setting it disables auto-range
            range_mode = "pinned"
            print(f"Range pinned at {pm.power_range_w:.3e} W")
        elif args.lock_range:
            range_mode = "locked"
            pm.auto_range = True
            for _ in range(5):
                pm.read_power()                       # let it settle on the level
            pm.power_range_w = pm.power_range_w      # freeze wherever it landed
            print(f"Range locked at {pm.power_range_w:.3e} W")
        else:
            range_mode = "auto"                      # already enabled above
            print(f"Auto-range on (reading {pm.read_power() * 1e6:.4f} uW)")

        # --- now block the beam: zero on that range, then measure the floor
        dark_mean = dark_std = None
        took_dark = False
        if prompts and (args.dark_s > 0 or args.zero):
            ask("\nBlock the beam -- light-tight, not just a hand -- then press Enter: ", True)

            if args.zero:
                print("  Zeroing...")
                pm.zero()
                # The driver only reports that the routine FINISHED, never that
                # it succeeded, so read back what it actually stored.
                try:
                    offset = pm.dark_offset
                    print(f"  Stored zero offset: {offset * 1e9:+.4f} nW")
                except (AttributeError, PowerMeterError):
                    pass
                settled = pm.read_power()
                print(f"  Reading with the beam blocked: {settled * 1e9:+.4f} nW")
                if settled < 0:
                    print("  !! Negative after zeroing -- the console will show a")
                    print("     blinking ZERO! warning. Light reached the sensor during")
                    print("     the zero. Re-block properly and run again.")

            if args.dark_s > 0:
                print("  Letting the console auto-range down to the dark level "
                      "(this can take a few seconds)...", flush=True)
                for index in range(8):
                    settle = pm.read_power()
                    print(f"    settling {index + 1}/8: {settle * 1e9:+.3f} nW",
                          flush=True)
                print(f"  Measuring the dark level for {args.dark_s:.0f} s...",
                      flush=True)
                _, dark_powers, _, _ = sample_series(
                    pm, args.dark_s, args.interval_s, args.temperature_every_s,
                    dark_csv,
                    [f"segment: dark (beam blocked)",
                     f"wavelength_nm: {args.wavelength_nm:g}",
                     f"average_count: {average_count}"],
                    "dark", read_temperature=log_temperature,
                )
                if dark_powers.size:
                    took_dark = True
                    dark_mean = float(np.mean(dark_powers))
                    dark_std = float(np.std(dark_powers, ddof=1)) if dark_powers.size > 1 else 0.0
                    print(f"  Dark offset {dark_mean * 1e9:+.3f} nW, "
                          f"noise {dark_std * 1e9:.3f} nW rms")
                    if dark_std > 0 and dark_mean < -3 * dark_std:
                        print("  !! The dark level is significantly negative: the stored")
                        print("     zero is too large. Re-zero with the beam properly")
                        print("     blocked before trusting these numbers.")

            ask("Unblock the beam and press Enter to start: ", True)

        if args.warmup_s > 0:
            print(f"Settling for {args.warmup_s:.0f} s...")
            time.sleep(args.warmup_s)

        meta = describe(pm, args)
        meta.update({
            "schema_version": SCHEMA_VERSION,
            "script": "measure_intensity.py",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "requested_duration_s": args.duration_s,
            "requested_interval_s": args.interval_s,
            "note": args.note,
            "label": args.label,
            "range_mode": range_mode,
            "dark_file": dark_csv.name if took_dark else None,
            "dark_mean_w": dark_mean,
            "dark_std_w": dark_std,
            "dark_duration_s": args.dark_s if took_dark else 0.0,
            "zeroed": args.zero and prompts,
        })

        print(f"\nLogging for {args.duration_s:.0f} s "
              f"(Ctrl-C stops early and keeps the data)...")
        times, powers, temps, interrupted = sample_series(
            pm, args.duration_s, args.interval_s, args.temperature_every_s,
            power_csv, header_lines_from(meta), "run",
            read_temperature=log_temperature,
        )

    if powers.size == 0:
        print("No samples recorded.")
        return 1

    meta.update({
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "interrupted": interrupted,
        "n_samples": int(powers.size),
        "actual_duration_s": float(times[-1]) if times.size else 0.0,
        "median_interval_s": float(np.median(np.diff(times))) if times.size > 2 else None,
        "mean_w": float(np.mean(powers)),
        "std_w": float(np.std(powers, ddof=1)) if powers.size > 1 else 0.0,
        "power_file": power_csv.name,
    })
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mean = meta["mean_w"]
    std = meta["std_w"]
    print(f"\n{powers.size} samples over {meta['actual_duration_s']:.1f} s")
    print(f"Mean  {mean * 1e6:.4f} uW    rms spread {std / mean * 100:.3f} % "
          f"(includes drift)")
    if dark_std:
        print(f"Detector noise floor {dark_std / mean * 100:.4f} % of this level")
    print(f"\nSaved {power_csv}")
    if took_dark:
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
