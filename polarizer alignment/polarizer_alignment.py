"""
polarizer_alignment.py
======================
Live power readout (left) plus a point-by-point plot (right), for checking a
polarizer against Malus' law or just peaking up an alignment.

Left panel  : the current power, big, with the wavelength it was measured at.
Right panel : every saved measurement -- power (vertical) against measurement
              number 1, 2, 3, ... (horizontal).

Press **s** in the window to save the current reading as the next point.
Press **d** to dump the points to ~/Desktop/polarizer_alignment/pol_alignment_N.csv
(N counts up from 1, never overwriting) and start a fresh run.
Close the window to stop.

Run it on the Windows PC the meter is plugged into:

    python polarizer_alignment.py --wavelength_nm 633
"""

import argparse
import pathlib

import matplotlib.pyplot as plt

from thorlabs_powermeter import PowerMeter

DEFAULT_WAVELENGTH_NM = 450.0
INTERVAL_S = 0.1
DESKTOP = pathlib.Path.home() / "Desktop"

# matplotlib grabs 's' for its own save-figure dialog -- leave that on ctrl+s.
plt.rcParams["keymap.save"] = ["ctrl+s"]

powers = []


def save_csv():
    """Write the points to the next free pol_alignment_N.csv and return it."""
    root = DESKTOP if DESKTOP.is_dir() else pathlib.Path.home()
    folder = root / "polarizer_alignment"
    folder.mkdir(exist_ok=True)

    index = 1
    while (folder / f"pol_alignment_{index}.csv").exists():
        index += 1

    path = folder / f"pol_alignment_{index}.csv"
    with open(path, "w") as file:
        file.write("measurement,power_uW\n")
        for number, power in enumerate(powers, start=1):
            file.write(f"{number},{power:.6f}\n")
    return path


def main(wavelength_nm):
    with PowerMeter(wavelength_nm=wavelength_nm) as pm:
        pm.average_count = 10
        pm.auto_range = True
        print("Connected to", pm.identity.model, "at", pm.wavelength_nm, "nm")
        print("Press 's' to save a point, 'd' to write a CSV and reset.")

        plt.ion()
        figure, (readout, graph) = plt.subplots(1, 2, figsize=(11, 5))

        readout.axis("off")
        number = readout.text(0.5, 0.55, "", ha="center", va="center", fontsize=44)
        label = readout.text(
            0.5, 0.25, f"at {wavelength_nm:.0f} nm", ha="center", fontsize=14
        )

        (line,) = graph.plot([], [], "o-")
        graph.set_xlabel("measurement number")
        graph.set_ylabel("power (uW)")
        graph.grid(True, alpha=0.3)

        def on_key(event):
            if event.key == "s":
                powers.append(pm.read_power() * 1e6)
                print(f"{len(powers)}: {powers[-1]:.3f} uW")
            elif event.key == "d":
                if not powers:
                    print("nothing to save yet")
                    return
                print(f"saved {len(powers)} points to {save_csv()} -- reset")
                powers.clear()

        figure.canvas.mpl_connect("key_press_event", on_key)

        while plt.fignum_exists(figure.number):
            number.set_text(f"{pm.read_power() * 1e6:.3f} uW")
            label.set_text(f"at {wavelength_nm:.0f} nm  |  {len(powers)} saved")

            line.set_data(range(1, len(powers) + 1), powers)
            graph.relim()
            graph.autoscale_view()
            plt.pause(INTERVAL_S)

        plt.ioff()

    for index, power in enumerate(powers, start=1):
        print(f"{index}\t{power:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wavelength_nm",
        type=float,
        default=DEFAULT_WAVELENGTH_NM,
        help="operating wavelength in nm (default: %(default)s)",
    )
    main(parser.parse_args().wavelength_nm)
