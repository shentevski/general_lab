#!/usr/bin/env python3
"""
power_vs_rep_rate.py
====================
Average UV power against pulse repetition frequency, for the 266 nm pump of
the single-photon source.

Run it on the PC the Thorlabs meter is plugged into:

    python power_vs_rep_rate.py

It connects the meter, sets it to 266 nm, then loops: you set the laser's
repetition frequency, type it in, and the script averages a few readings and
keeps the point. Press Enter on an empty line (or type q) when you are done.

Two numbers per point:

* **mean power** P, what the meter reads (W). The meter averages over many
  pulses, so this is the average power, not a per-pulse peak.
* **mean energy per pulse** E = P / f (J). This is what "power per pulse"
  means for a pulsed laser: the meter cannot see a single pulse, so the
  per-pulse quantity is the average power divided by the repetition
  frequency f.

Saved into ~/Desktop/single_photon_source/, as power_vs_rep_rate_<stamp>.csv
and .png. The CSV is rewritten after every point, so a crash or a Ctrl-C
leaves the points you already took on disk.

Symbols used here:
    f  repetition frequency of the laser (Hz)
    P  mean (average) optical power at the meter (W)
    E  mean energy per pulse, E = P / f (J)
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thorlabs_powermeter import PowerMeter

# -------------------- settings --------------------
WAVELENGTH_NM = 266.0    # the pump wavelength; sets the sensor responsivity
READINGS = 20            # meter readings averaged per point
AVERAGE_COUNT = 100      # meter's own internal averaging, per reading
OUT_DIR = Path.home() / "Desktop" / "single_photon_source"
# --------------------------------------------------


def ask_frequency():
    """Repetition frequency in Hz, or None when the user is done.

    Accepts a bare number in Hz, or a k / M suffix: 5000, 5k and 5 kHz are
    the same frequency.
    """
    while True:
        text = input("\nRepetition frequency (Hz, or e.g. 5k) "
                     "[Enter = done]: ").strip().lower()
        if text in ("", "q", "quit", "done"):
            return None
        text = text.removesuffix("hz").strip()
        scale = 1.0
        if text.endswith("k"):
            scale, text = 1e3, text[:-1]
        elif text.endswith("m"):
            scale, text = 1e6, text[:-1]
        try:
            f = float(text) * scale
        except ValueError:
            print("  not a number -- try again")
            continue
        if f <= 0:
            print("  the frequency has to be positive")
            continue
        return f


def save(rows, csv_path):
    """Write every point taken so far to the CSV."""
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(f"# wavelength_nm: {WAVELENGTH_NM:g}\n")
        handle.write(f"# readings averaged per point: {READINGS}\n")
        handle.write("rep_rate_Hz,mean_power_W,std_power_W,"
                     "energy_per_pulse_J,n_readings\n")
        for f, p, s in rows:
            handle.write(f"{f:.6g},{p:.6e},{s:.6e},{p / f:.6e},{READINGS}\n")


def plot(rows, png_path):
    """Power against frequency, and energy per pulse against frequency."""
    f = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    s = np.array([r[2] for r in rows])
    order = np.argsort(f)
    f, p, s = f[order], p[order], s[order]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    left.errorbar(f * 1e-3, p * 1e6, yerr=s * 1e6, marker="o", capsize=3)
    left.set_xlabel("repetition frequency f (kHz)")
    left.set_ylabel("mean power P (uW)")
    left.set_title("average power")

    right.errorbar(f * 1e-3, p / f * 1e9, yerr=s / f * 1e9, marker="o", capsize=3)
    right.set_xlabel("repetition frequency f (kHz)")
    right.set_ylabel("mean energy per pulse E = P/f (nJ)")
    right.set_title("energy per pulse")

    for axis in (left, right):
        axis.grid(True, alpha=0.3)
        # absolute numbers on the y axis, never "+1" in the corner
        axis.ticklabel_format(axis="y", useOffset=False)
    figure.suptitle(f"266 nm pump, {png_path.stem}")
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)
    return figure


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"power_vs_rep_rate_{datetime.now():%Y%m%d-%H%M%S}"
    csv_path = OUT_DIR / f"{stem}.csv"
    png_path = OUT_DIR / f"{stem}.png"

    with PowerMeter(wavelength_nm=WAVELENGTH_NM) as pm:
        pm.auto_range = True
        pm.power_unit = "W"
        pm.average_count = AVERAGE_COUNT
        print(f"Connected to {pm.identity.model} "
              f"(S/N {pm.identity.serial_number}), sensor {pm.sensor.name}")
        print(f"Wavelength:  {pm.wavelength_nm:.1f} nm")

        low, high = pm.wavelength_range_nm
        if not low <= WAVELENGTH_NM <= high:
            print(f"\n!! {WAVELENGTH_NM:g} nm is outside this sensor's "
                  f"calibrated range ({low:.0f}-{high:.0f} nm). The reading is "
                  f"an extrapolation, not a traceable power.\n")

        rows = []
        try:
            while True:
                f = ask_frequency()
                if f is None:
                    break
                power = pm.read_powers(READINGS)
                mean, std = float(power.mean()), float(power.std(ddof=1))
                rows.append((f, mean, std))
                save(rows, csv_path)
                print(f"  f = {f * 1e-3:8.3f} kHz   "
                      f"P = {mean * 1e6:9.4f} +- {std * 1e6:.4f} uW   "
                      f"E = P/f = {mean / f * 1e9:9.4f} nJ/pulse")
        except KeyboardInterrupt:
            print("\n  Interrupted -- keeping the points taken so far.")

    if not rows:
        print("No points taken, nothing saved.")
        return

    print(f"\n{len(rows)} points")
    print(f"{'f (kHz)':>10} {'P (uW)':>12} {'E = P/f (nJ)':>14}")
    for f, p, _ in sorted(rows):
        print(f"{f * 1e-3:10.3f} {p * 1e6:12.4f} {p / f * 1e9:14.4f}")

    plot(rows, png_path)
    print(f"\nSaved {csv_path}")
    print(f"      {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
