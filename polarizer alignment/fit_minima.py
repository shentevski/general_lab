"""
fit_minima.py
=============
Reads every pol_alignment_N.csv written by polarizer_alignment.py, fits a
parabola to each run, and reports where the minimum sits.

Each run is expected to be a scan through the extinction point, so power
against measurement number is a parabola opening upwards. The fit is a plain
least-squares quadratic; the minimum is its vertex, which lands between
measurement numbers rather than on one of them -- that is the point of
fitting rather than just taking the smallest reading.

Prints a line per run, then the mean and standard deviation across runs:

    python fit_minima.py
    python fit_minima.py --folder "C:/Users/me/Desktop/polarizer_alignment"
"""

import argparse
import pathlib

import numpy as np

DEFAULT_FOLDER = pathlib.Path.home() / "Desktop" / "polarizer_alignment"


def fit_one(path):
    """Fit a parabola to one CSV. Returns (vertex_x, min_power, opens_up)."""
    numbers, powers = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    if len(numbers) < 3:
        raise ValueError("needs at least 3 points to fit a parabola")

    a, b, c = np.polyfit(numbers, powers, 2)
    if a == 0:
        raise ValueError("fit is a straight line -- no minimum")

    vertex = -b / (2 * a)
    return vertex, a * vertex**2 + b * vertex + c, a > 0


def main(folder):
    paths = sorted(
        folder.glob("pol_alignment_*.csv"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not paths:
        print(f"no pol_alignment_*.csv files in {folder}")
        return

    vertices, minima = [], []
    for path in paths:
        try:
            vertex, power, opens_up = fit_one(path)
        except (ValueError, IndexError) as error:
            print(f"{path.name}: skipped -- {error}")
            continue

        if not opens_up:
            print(f"{path.name}: skipped -- parabola opens downwards (a maximum)")
            continue

        print(f"{path.name}: minimum {power:.3f} uW at measurement {vertex:.2f}")
        vertices.append(vertex)
        minima.append(power)

    if not minima:
        print("\nnothing fitted")
        return

    vertices, minima = np.array(vertices), np.array(minima)
    spread = "" if len(minima) > 1 else "  (single run -- no spread)"
    deviation = np.std(minima, ddof=1) if len(minima) > 1 else 0.0
    vertex_deviation = np.std(vertices, ddof=1) if len(vertices) > 1 else 0.0

    print(f"\n{len(minima)} runs fitted{spread}")
    print(f"minimum power    : {minima.mean():.3f} +/- {deviation:.3f} uW")
    print(
        f"at measurement   : {vertices.mean():.2f} +/- {vertex_deviation:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        type=pathlib.Path,
        default=DEFAULT_FOLDER,
        help="folder holding the CSVs (default: %(default)s)",
    )
    main(parser.parse_args().folder)
