#!/usr/bin/env python3
"""Check the PSG: set a state, measure it, compare, plot the ellipse.

    python psg_check.py                 # type states one after the other
    python psg_check.py R H 30 -10      # or give them (the JSON's "states" too)
    python psg_check.py --simulate      # rehearse, no hardware

A state is typed as

    R              a name: H V D A R L, or P (both PSG plates along the polarizer)
    30 -10         two numbers: azimuth and ellipticity of the ellipse, deg
    0.5 0.5 0.707  three numbers: S1 S2 S3 (four: S0 S1 S2 S3)

and Enter with nothing quits. For each state the PSG sets it, the polarimeter
measures it (one sweep, laser-corrected when there is a reference meter),
and it prints

    target     what you asked for
    expected   what the PSG model says it makes, at the angles the stages reached
    measured   what the polarimeter sees

with the misses on the Poincare sphere: measured vs target is the whole
error, measured vs expected the part the PSG calibration does not explain.
The figure overlays the three ellipses (the measured one shrunk by its DOP).

NO SAMPLE in the beam. Hardware and the polarimeter calibration come from
measure_mueller.json, the PSG's from ~/.poltk (`poltk psg show`). Every sweep
and a results.json are saved in Desktop/psg_check_<date>_<time>/.

Note: the polarimeter that checks the PSG also calibrated it. This catches
everything except an error of the polarimeter itself -- for that, see the
extinction and handedness checks in the README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mueller_common import parse_with_config, write_sweep
from polarization_toolkit.errors import ToolkitError
from polarization_toolkit.psg import ellipse_of, sphere_angle_deg

HERE = Path(__file__).resolve().parent


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def parse_typed(txt: str):
    """A typed state -> a PSG state spec (raises ValueError)."""
    parts = txt.replace(",", " ").split()
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        if len(parts) == 1:
            return parts[0]                      # a name; the PSG checks it
        raise ValueError(f"'{txt}': a name, or 2, 3 or 4 numbers") from None
    if len(nums) == 2:
        return {"azimuth_deg": nums[0], "ellipticity_deg": nums[1]}
    if len(nums) in (3, 4):
        return nums
    raise ValueError(f"'{txt}': 2 numbers (azimuth, ellipticity) or 3 (S1 S2 S3)")


def rig_settings(a):
    """measure_mueller.json (hardware, polarimeter calibration, simulator), with
    this script's overrides, as the namespace the Rig expects."""
    path = Path(a.measure_config)
    if not path.is_absolute():
        path = HERE / path
    try:
        cfg = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"not found: {path}") from None
    cfg.update(use_psg=True, zero_meter=a.zero_meter, simulate=a.simulate)
    if a.qwp_steps:
        cfg["qwp_steps"] = a.qwp_steps
    defaults = {"qwp_sweep_deg": 360.0, "qwp_settle": 0.3, "s3_sign": 1,
                "ref_pm_serial": None, "sim_state_error_deg": 3.0, "sim_noise_rel": 1e-3,
                "sim_laser_drift_rel": 0.01, "sim_laser_noise_rel": 0.003}
    return SimpleNamespace(**{**defaults, **cfg})


# --------------------------------------------------------------------------- #
# printing and plotting
# --------------------------------------------------------------------------- #


def row(name, S):
    S = np.asarray(S, float)
    n = S / S[0]
    az, ell = ellipse_of(n[1:])
    return (f"  {name:9s} {n[1]:+7.3f} {n[2]:+7.3f} {n[3]:+7.3f}  {np.linalg.norm(n[1:]):6.3f}"
            f"  {az:8.2f}  {ell:+8.2f}")


def ellipse_xy(s, scale=1.0):
    """Points of the polarization ellipse of direction s = (S1, S2, S3), and the
    ellipticity angle (deg). x = H, y = V; azimuth counterclockwise from H.
    Semi-axes cos(ellipticity) and sin(ellipticity): unit intensity, so a
    linear state reaches the unit circle and a circular one has radius 0.71."""
    az, ell = np.deg2rad(ellipse_of(s))
    t = np.linspace(0, 2 * np.pi, 361)
    a, b = np.cos(ell), np.sin(ell)
    x = a * np.cos(t) * np.cos(az) - b * np.sin(t) * np.sin(az)
    y = a * np.cos(t) * np.sin(az) + b * np.sin(t) * np.cos(az)
    return scale * x, scale * y, np.degrees(ell)


def draw(ax, label, target, expected, measured, miss_t, miss_e):
    from matplotlib.patches import Circle
    ax.clear()
    lim = 1.25
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.add_patch(Circle((0, 0), 1, fill=False, color="0.9", lw=0.8))
    m = np.asarray(measured, float) / measured[0]
    dop = float(np.linalg.norm(m[1:]))
    curves = [(target, 1.0, dict(color="0.45", ls="--", lw=1.6), "target"),
              (np.asarray(expected)[1:], 1.0, dict(color="C0", ls=":", lw=2.2),
               "expected (PSG model)"),
              (m[1:], min(dop, 1.0), dict(color="C3", lw=2.0), f"measured (DOP {dop:.3f})")]
    for s, scale, style, name in curves:
        x, y, ell = ellipse_xy(s, scale)
        ax.plot(x, y, label=name, **style)
        if abs(ell) > 1.0:                    # sense of rotation, at the top of the ellipse
            k = int(np.argmax(y[:-7]))
            ax.annotate("", xy=(x[k + 6], y[k + 6]), xytext=(x[k], y[k]),
                        arrowprops=dict(arrowstyle="-|>", color=style["color"], lw=1.5))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("H")
    ax.set_ylabel("V")
    ax.set_title(f"{label}:  measured is {miss_t:.2f} deg from the target, "
                 f"{miss_e:.2f} deg from the expected\n(on the Poincare sphere; "
                 f"arrow: S3 > 0 drawn counterclockwise)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="*", default=[],
                    help="states to check, then stop (none: type them one by one)")
    ap.add_argument("--measure-config", default="measure_mueller.json",
                    dest="measure_config",
                    help="hardware + polarimeter calibration (default: measure_mueller.json)")
    ap.add_argument("--qwp-steps", type=int, default=None, dest="qwp_steps",
                    help="polarimeter steps per sweep (null: as in measure_mueller.json)")
    ap.add_argument("--zero-meter", type=str2bool, default=False, dest="zero_meter",
                    help="zero the power meter(s) at the start")
    ap.add_argument("--show-figures", type=str2bool, default=True, dest="show_figures")
    ap.add_argument("--save", type=str2bool, default=True)
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--simulate", action="store_true")
    a = parse_with_config(ap, argv, HERE / "psg_check.json")

    from measure_mueller import Rig, SimRig, analyse_sweep, sweep

    s = rig_settings(a)
    import matplotlib
    if not a.show_figures:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = None
    if a.save:
        root = Path(a.out) if a.out else Path.home() / "Desktop"
        run = root / time.strftime("psg_check_%Y%m%d_%H%M%S")
        run.mkdir(parents=True)
        print(f"saving in {run}")
    results = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "polarimeter": {"qwp_zero_deg": s.qwp_zero_deg,
                               "qwp_retardance_waves": s.qwp_retardance_waves,
                               "s3_sign": s.s3_sign},
               "states": []}

    rig = SimRig(s) if a.simulate else Rig(s)
    with rig as r:
        cal = r.psg.cal
        print(("PSG calibrated " + cal.meta.get("created", "")) if cal.calibrated else
              "PSG NOT calibrated (nominal model): expect large misses -- poltk psg calibrate")
        results["psg_calibration"] = cal.to_dict()
        if a.simulate:
            r.set_sample(False)
        else:
            input("\nNO SAMPLE in the beam. Press Enter when it is out...")
        if s.zero_meter:
            input("BLOCK the beam to zero the power meter(s), then press Enter...")
            r.zero_meter()
            input("UNBLOCK the beam, then press Enter...")

        fig = None
        if a.show_figures:
            plt.ion()
        queue = [x for x in a.states]
        interactive = not queue
        k = 0
        while True:
            if queue:
                spec = queue.pop(0)
                if isinstance(spec, str):
                    spec = parse_typed(spec)
            elif interactive:
                txt = input("\nstate (H V D A R L P | azimuth ellipticity | S1 S2 S3; "
                            "Enter = quit): ").strip()
                if not txt:
                    break
                try:
                    spec = parse_typed(txt)
                except ValueError as e:
                    print(f"  {e}")
                    continue
            else:
                break
            try:
                setting = r.set_state(spec)
            except ValueError as e:
                print(f"  {e}")
                continue
            k += 1
            print(f"\n=== {setting.label} ===")
            print(f"  PSG: HWP {setting.hwp_deg:.2f} deg, QWP {setting.qwp_deg:.2f} deg")
            sw = sweep(r, s, setting.label[:6], s.qwp_sweep_deg)
            f = analyse_sweep(sw, s)
            S = f["S"]
            miss_t = float(sphere_angle_deg(S, setting.target))
            miss_e = float(sphere_angle_deg(S, setting.predicted))
            print(f"  {'':9s} {'S1':>7s} {'S2':>7s} {'S3':>7s}  {'DOP':>6s}  "
                  f"{'azimuth':>8s}  {'ellipt.':>8s}  (deg)")
            print(row("target", np.r_[1.0, setting.target]))
            print(row("expected", setting.predicted))
            print(row("measured", S))
            fit_txt = (f"{100 * f['rms_raw']:.2f}% raw -> {100 * f['rms_rel']:.2f}% "
                       f"laser-corrected" if "rms_raw" in f else f"{100 * f['rms_rel']:.2f}%")
            print(f"  measured vs target {miss_t:.2f} deg, vs expected {miss_e:.2f} deg "
                  f"on the Poincare sphere   (fit rms {fit_txt})")

            entry = {"index": k, "label": setting.label, "psg": setting.to_dict(),
                     "measured": [float(x) for x in S / S[0]],
                     "dop": float(np.linalg.norm(S[1:]) / S[0]),
                     "miss_target_deg": miss_t, "miss_expected_deg": miss_e,
                     "fit_rms_rel": f["rms_rel"]}
            results["states"].append(entry)
            if run is not None:
                name = f"state_{k:02d}"
                write_sweep(run / f"{name}.csv", sw["commanded_deg"], sw["measured_deg"],
                            sw["power_W"], sw["ref_W"], sw["t_s"])
                entry["file"] = f"{name}.csv"
                (run / "results.json").write_text(json.dumps(results, indent=2,
                                                             default=float))

            if fig is None or not plt.fignum_exists(fig.number):
                fig, ax = plt.subplots(figsize=(6.4, 6.4))
            draw(ax, setting.label, setting.target, setting.predicted, S, miss_t, miss_e)
            fig.tight_layout()
            if run is not None:
                fig.savefig(run / f"state_{k:02d}_ellipse.png", dpi=130)
            if a.show_figures:
                plt.pause(0.1)

    if results["states"]:
        m = np.array([e["miss_target_deg"] for e in results["states"]])
        e = np.array([e["miss_expected_deg"] for e in results["states"]])
        print(f"\n{len(m)} state(s): miss vs target median {np.median(m):.2f} deg, "
              f"max {m.max():.2f} deg; vs expected median {np.median(e):.2f} deg")
    if a.show_figures and fig is not None:
        plt.ioff()
        plt.show()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolkitError as e:
        sys.exit(f"error: {e}")
