#!/usr/bin/env python3
"""Mueller matrix of a sample from a measure_mueller.py run -- and what it means.

    python analyze_mueller.py              # reads analyze_mueller.json
    python analyze_mueller.py <run_dir>    # or name the run folder directly

What it does
  1. Stokes vector of every input state and every through-sample state,
     with the PSA calibration recorded by the measurement -- or the one you
     put in the JSON (e.g. the QWP retardance from the manufacturer).
  2. M = S_out . pinv(S_in), least squares over all the states you took.
  3. Checks: how well the states cover the sphere (condition number),
     whether M predicts every S_out (per-state misfit -- catches a state
     that changed or a sample that moved), and whether M is physical
     (Cloude eigenvalues).
  4. Decompositions
       Lu-Chipman    M = M_depolarizer . M_retarder . M_diattenuator, with
                     the retarder split into linear retardance, fast axis
                     and optical rotation
       differential  logm(M) -> LB, LB', CB, LD, LD', CD of a homogeneous
                     medium (no fake CD from LD + LB mixing)
  5. Error bars: statistical (Monte Carlo from the fit residuals) and
     systematic (QWP retardance and QWP zero-angle uncertainty).

If the run has a reference meter (laser monitor), every sweep is divided by
it before anything else, which removes laser drift and noise; a summary of
what the laser did, and how much the correction helped, is printed and
plotted. Set use_reference to false to see the result without it.

Output goes to <run_dir>/analysis/: results.json, mueller.csv, figures.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from polarization_toolkit.analysis.mueller import (differential_decompose,
                                                   lu_chipman, mueller_from_stokes)
from mueller_common import (cloude, corrected_power, extract_stokes,
                            has_reference, parse_with_config, read_sweep,
                            split_retarder, waves_to_deg)

HERE = Path(__file__).resolve().parent


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y", "on")


# key, label, unit, period (axis angles are circular) -- in print order
PARAMS = [
    ("M00", "transmittance M00 (unpolarized light)", "", None),
    ("diattenuation", "diattenuation |D|", "", None),
    ("diattenuation_linear", "  linear part", "", None),
    ("diattenuation_circular", "  circular part D3", "", None),
    ("diattenuation_axis_deg", "  transmission axis", "deg", 180.0),
    ("polarizance", "polarizance |P|", "", None),
    ("retardance_deg", "total retardance", "deg", None),
    ("linear_retardance_deg", "  linear retardance", "deg", None),
    ("fast_axis_deg", "  fast axis", "deg", 180.0),
    ("optical_rotation_deg", "  optical rotation", "deg", 180.0),
    ("depolarization_index", "depolarization index (1 = none)", "", None),
    ("depolarization_power", "depolarization power (0 = none)", "", None),
    ("LB_deg", "LB   linear birefringence 0/90", "deg", None),
    ("LBp_deg", "LB'  linear birefringence +-45", "deg", None),
    ("CB_deg", "CB   circular birefringence", "deg", None),
    ("diff_linear_retardance_deg", "  linear retardance", "deg", None),
    ("diff_fast_axis_deg", "  fast axis", "deg", 180.0),
    ("diff_optical_rotation_deg", "  optical rotation = CB/2", "deg", 180.0),
    ("LD", "LD   linear dichroism 0/90", "", None),
    ("LDp", "LD'  linear dichroism +-45", "", None),
    ("CD", "CD   circular dichroism", "", None),
    ("diff_residual", "  model residual (0 = clean, homogeneous)", "", None),
    ("branch_margin_deg", "  branch margin (small = may wrap)", "deg", None),
    ("cloude_entropy", "entropy (0 = pure, 1 = fully random)", "", None),
    ("cloude_lambda_min", "smallest eigenvalue (< 0 = unphysical)", "", None),
]
SECTIONS = {"M00": "GENERAL",
            "retardance_deg": "LU-CHIPMAN RETARDER",
            "depolarization_index": "DEPOLARIZATION",
            "LB_deg": "DIFFERENTIAL (logm) -- homogeneous medium",
            "cloude_entropy": "CLOUDE (physical realizability)"}
PERIOD = {k: p for k, _, _, p in PARAMS if p}


def parameters(M) -> dict:
    """Every scalar we report, from one Mueller matrix."""
    M = np.asarray(M, float)
    deg = np.rad2deg
    m = M / M[0, 0]
    D, P = m[0, 1:], m[1:, 0]

    lc = lu_chipman(M)
    lin, fast, rot = split_retarder(lc.M_retarder)
    dd = differential_decompose(M)
    cl = cloude(M)
    return {
        "M00": M[0, 0],
        "diattenuation": np.linalg.norm(D),
        "diattenuation_linear": np.hypot(D[0], D[1]),
        "diattenuation_circular": D[2],
        "diattenuation_axis_deg": deg(0.5 * np.arctan2(D[1], D[0])) % 180,
        "polarizance": np.linalg.norm(P),
        "retardance_deg": deg(lc.retardance),
        "linear_retardance_deg": deg(lin),
        "fast_axis_deg": deg(fast) % 180,
        "optical_rotation_deg": deg(rot),
        "depolarization_index": lc.depolarization_index,
        "depolarization_power": 1 - abs(lc.depol_trace),
        "LB_deg": deg(dd.LB),
        "LBp_deg": deg(dd.LB_prime),
        "CB_deg": deg(dd.CB),
        "diff_linear_retardance_deg": deg(dd.linear_retardance),
        "diff_fast_axis_deg": deg(dd.linear_retardance_axis) % 180,
        "diff_optical_rotation_deg": deg(dd.optical_rotation),
        "LD": dd.LD,
        "LDp": dd.LD_prime,
        "CD": dd.CD,
        "diff_residual": dd.residual,
        "branch_margin_deg": deg(dd.branch_margin),
        "cloude_entropy": cl["entropy"],
        "cloude_lambda_min": cl["eigenvalues"].min(),
    } | {"_valid": bool(lc.valid) and bool(dd.valid)}


def spread(values, period=None):
    """(centre, std) of a sample; circular for axis angles."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    if period is None:
        return float(v.mean()), float(v.std(ddof=1))
    z = np.exp(2j * np.pi * v / period).mean()
    R = min(abs(z), 1.0)
    std = period / (2 * np.pi) * np.sqrt(-2 * np.log(R)) if R > 0 else period / 2
    return float(np.angle(z) * period / (2 * np.pi) % period), float(std)


def wrap_diff(a, b, period=None):
    d = a - b
    if period:
        d = (d + period / 2) % period - period / 2
    return d


# --------------------------------------------------------------------------- #


def load(run_dir: Path, exclude):
    meta = json.loads((run_dir / "run.json").read_text())
    states = []
    for s in meta["states"]:
        if str(s["index"]) in exclude or s["label"] in exclude:
            print(f"  excluding state {s['index']} ({s['label']})")
            continue
        states.append({**s, "in": read_sweep(run_dir / s["input"]),
                       "out": read_sweep(run_dir / s["output"])})
    return meta, states


def angles_of(sw):
    a = sw["measured_deg"]
    return np.where(np.isfinite(a), a, sw["commanded_deg"])


def stokes_all(states, cal):
    """Extract S_in / S_out (and the fits) of every state for one calibration."""
    out = []
    for s in states:
        r = {}
        for side in ("in", "out"):
            sw = s[side]
            r[side] = extract_stokes(sw["power_used"], angles_of(sw),
                                     qwp_zero_deg=cal["qwp_zero_deg"],
                                     retardance_deg=waves_to_deg(cal["qwp_retardance_waves"]),
                                     s3_sign=cal["s3_sign"])
        out.append(r)
    return out


def estimate(S_in, S_out):
    """M from (4, n) Stokes arrays via the toolkit."""
    keys = [f"{k:03d}" for k in range(S_in.shape[1])]
    res = mueller_from_stokes(dict(zip(keys, S_in.T)), dict(zip(keys, S_out.T)))
    return res.M


def coverage(S_in):
    Sn = S_in / S_in[0]
    sv = np.linalg.svd(Sn, compute_uv=False)
    return int((sv > 1e-2 * sv[0]).sum()), float(sv[0] / sv[-1])


def monte_carlo(states, fits, cal, trials, rng):
    """Re-draw every sweep from its fitted model + its own residual noise."""
    Ms, ps = [], []
    delta = waves_to_deg(cal["qwp_retardance_waves"])
    for _ in range(trials):
        S_in, S_out = [], []
        for s, f in zip(states, fits):
            for side, dest in (("in", S_in), ("out", S_out)):
                fit = f[side]
                y = fit["model"] + rng.normal(0, fit["residual_rms"], fit["model"].size)
                dest.append(extract_stokes(y, angles_of(s[side]),
                                           qwp_zero_deg=cal["qwp_zero_deg"],
                                           retardance_deg=delta,
                                           s3_sign=cal["s3_sign"])["S"])
        M = estimate(np.array(S_in).T, np.array(S_out).T)
        Ms.append(M / M[0, 0])
        ps.append(parameters(M))
    return np.array(Ms), ps


def apply_reference(states, use_reference):
    """Set sw["power_used"] on every sweep; returns (used?, present?)."""
    sweeps = [s[side] for s in states for side in ("in", "out")]
    present = bool(sweeps) and all(has_reference(sw) for sw in sweeps)
    used = present and use_reference
    level = np.mean([sw["ref_W"].mean() for sw in sweeps]) if present else None
    for sw in sweeps:
        sw["power_used"] = corrected_power(sw, level) if used else sw["power_W"]
    return used, present


def rel_rms(sw, power, cal):
    fit = extract_stokes(power, angles_of(sw), qwp_zero_deg=cal["qwp_zero_deg"],
                         retardance_deg=waves_to_deg(cal["qwp_retardance_waves"]),
                         s3_sign=cal["s3_sign"])
    return fit["residual_rms"] / np.mean(power)


def laser_summary(states, cal):
    """What the laser did during the run, from the reference meter."""
    sweeps = [(s, side, s[side]) for s in states for side in ("in", "out")]
    means = np.array([sw["ref_W"].mean() for _, _, sw in sweeps])
    noise = []
    for _, _, sw in sweeps:
        t = sw["t_s"] - sw["t_s"][0]
        r = sw["ref_W"]
        noise.append(np.std(r - np.polyval(np.polyfit(t, r, 1), t)) / r.mean())
    t_all = np.concatenate([sw["t_s"] for _, _, sw in sweeps])
    in_out = [s["out"]["ref_W"].mean() / s["in"]["ref_W"].mean() - 1 for s in states]
    rms_raw = [rel_rms(sw, sw["power_W"], cal) for _, _, sw in sweeps]
    rms_cor = [rel_rms(sw, corrected_power(sw, means.mean()), cal) for _, _, sw in sweeps]
    return {"run_minutes": float((t_all.max() - t_all.min()) / 60),
            "drift_peak_to_peak": float(np.ptp(means) / means.mean()),
            "noise_per_reading": float(np.median(noise)),
            "max_in_out_change": float(np.max(np.abs(in_out))),
            "in_out_change": [float(x) for x in in_out],
            "fit_rms_raw": [float(x) for x in rms_raw],
            "fit_rms_corrected": [float(x) for x in rms_cor]}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def fig_laser(plt, states, summary):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    level = np.mean([s[side]["ref_W"].mean() for s in states for side in ("in", "out")])
    for s in states:
        for side, c in (("in", "C0"), ("out", "C1")):
            sw = s[side]
            ax1.plot(sw["t_s"] / 60, 100 * (sw["ref_W"] / level - 1), ".", color=c, ms=2)
        ax1.text(s["in"]["t_s"][0] / 60, 0, s["label"], fontsize=8, va="bottom")
    ax1.plot([], [], ".", color="C0", label="input sweeps")
    ax1.plot([], [], ".", color="C1", label="through-sample sweeps")
    ax1.axhline(0, color="0.6", lw=0.6)
    ax1.set_xlabel("time since start of run (min)")
    ax1.set_ylabel("laser power - mean (%)")
    ax1.set_title(f"reference meter: drift {100 * summary['drift_peak_to_peak']:.2f}% "
                  f"peak-to-peak, noise {100 * summary['noise_per_reading']:.2f}% "
                  f"per reading", fontsize=10)
    ax1.legend(fontsize=8)
    names = [f"{s['label']} {side}" for s in states for side in ("in", "out")]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, 100 * np.array(summary["fit_rms_raw"]), 0.4, label="raw")
    ax2.bar(x + 0.2, 100 * np.array(summary["fit_rms_corrected"]), 0.4,
            label="laser-corrected")
    ax2.set_xticks(x, names, rotation=60, fontsize=7)
    ax2.set_ylabel("fit residual rms (%)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_matrix(plt, m, m_err, m_sys, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{m[i, j]:+.4f}\n±{m_err[i, j]:.4f} stat"
                          f"\n±{m_sys[i, j]:.4f} sys", ha="center",
                    va="center", fontsize=9,
                    color="white" if abs(m[i, j]) > 0.6 else "black")
    ax.set_xticks(range(4), [f"m{j}" for j in range(4)])
    ax.set_yticks(range(4), [f"m{i}" for i in range(4)])
    ax.set_xticks(np.arange(-.5, 4), minor=True)
    ax.set_yticks(np.arange(-.5, 4), minor=True)
    ax.grid(which="minor", color="w", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def fig_poincare(plt, S_in, S_out, labels):
    fig = plt.figure(figsize=(6.4, 6))
    ax = fig.add_subplot(projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="0.85", lw=0.4)
    for s, lab in zip(("S1", "S2", "S3"), range(3)):
        e = np.zeros(3); e[lab] = 1.25
        ax.plot(*np.c_[-e, e], color="0.6", lw=0.6)
        ax.text(*(e * 1.05), s, color="0.4")
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)
    for k, lab in enumerate(labels):
        a = S_in[1:, k] / S_in[0, k]
        b = S_out[1:, k] / S_out[0, k]
        ax.scatter(*a, color=colors[k], marker="o", s=45)
        ax.scatter(*b, color=colors[k], marker="^", s=55, edgecolor="k", lw=0.5)
        ax.plot(*np.c_[a, b], color=colors[k], lw=1.2, ls="--")
        ax.text(*(a * 1.12), lab, color=colors[k], fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.set_axis_off()
    ax.set_title("input (o) -> through sample (^)\ninside the sphere = depolarized",
                 fontsize=10)
    fig.tight_layout()
    return fig


def fig_fits(plt, states, fits):
    n = len(states)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows),
                            squeeze=False, sharex=True)
    for ax, s, f in zip(axs.flat, states, fits):
        for side, c in (("in", "C0"), ("out", "C1")):
            th = angles_of(s[side])
            order = np.argsort(th % 360)
            y = s[side]["power_used"]
            ax.plot(th % 360, 1e3 * y, ".", color=c, ms=3,
                    label=f"{side}: rms {100 * f[side]['residual_rms'] / np.mean(y):.2f}%")
            ax.plot((th % 360)[order], 1e3 * f[side]["model"][order], "-", color=c, lw=1)
        ax.set_title(f"state {s['index']}: {s['label']}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel("power (mW)")
    for ax in axs[-1]:
        ax.set_xlabel("QWP stage angle (deg)")
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="mueller_<timestamp> folder written by measure_mueller.py")
    ap.add_argument("--qwp-retardance-waves", type=float, default=None,
                    dest="qwp_retardance_waves",
                    help="override the recorded QWP retardance (waves); null = recorded")
    ap.add_argument("--qwp-zero-deg", type=float, default=None, dest="qwp_zero_deg",
                    help="override the recorded QWP zero; null = recorded")
    ap.add_argument("--s3-sign", type=int, default=None, choices=(1, -1), dest="s3_sign",
                    help="override the recorded S3 sign; null = recorded")
    ap.add_argument("--use-reference", type=str2bool, default=True, dest="use_reference",
                    help="divide by the reference (laser monitor) meter when the "
                         "run has one")
    ap.add_argument("--exclude-states", nargs="*", default=[], dest="exclude_states",
                    help="state numbers or labels to leave out")
    ap.add_argument("--monte-carlo-trials", type=int, default=500,
                    dest="monte_carlo_trials", help="0 = no statistical errors")
    ap.add_argument("--retardance-uncertainty-waves", type=float, default=1 / 300,
                    dest="retardance_uncertainty_waves")
    ap.add_argument("--qwp-zero-uncertainty-deg", type=float, default=0.1,
                    dest="qwp_zero_uncertainty_deg")
    ap.add_argument("--show-figures", type=str2bool, default=True, dest="show_figures")
    ap.add_argument("--save-figures", type=str2bool, default=True, dest="save_figures")
    a = parse_with_config(ap, argv, HERE / "analyze_mueller.json")

    if not a.run_dir:
        raise SystemExit("no run folder: set run_dir in analyze_mueller.json "
                         "or pass it on the command line")
    run_dir = Path(a.run_dir).expanduser()
    if not (run_dir / "run.json").is_file():
        raise SystemExit(f"{run_dir} has no run.json -- is it a measure_mueller.py run?")
    print(f"run: {run_dir}")

    meta, states = load(run_dir, {str(x) for x in a.exclude_states})
    rec = meta["calibration"]
    cal = {}
    print("\nPSA calibration")
    for key, unit in (("qwp_retardance_waves", "waves"), ("qwp_zero_deg", "deg"),
                      ("s3_sign", "")):
        mine = getattr(a, key)
        cal[key] = rec[key] if mine is None else mine
        src = "recorded" if mine is None else f"OVERRIDE (recorded {rec[key]})"
        extra = (f" = {waves_to_deg(cal[key]):.2f} deg"
                 if key == "qwp_retardance_waves" else "")
        print(f"  {key:22s} {cal[key]} {unit}{extra}   [{src}]")

    # ---- laser monitor ----------------------------------------------------
    ref_used, ref_present = apply_reference(states, a.use_reference)
    laser = laser_summary(states, cal) if ref_present else None
    print("\nlaser monitor (reference meter)")
    if not ref_present:
        print("  none in this run -- laser drift and noise are in every sweep and "
              "in M00")
    else:
        print(f"  run {laser['run_minutes']:.1f} min | laser drift "
              f"{100 * laser['drift_peak_to_peak']:.2f}% peak-to-peak | noise "
              f"{100 * laser['noise_per_reading']:.2f}% per reading")
        print(f"  largest laser change between a state's input and through-sample "
              f"sweeps: {100 * laser['max_in_out_change']:.2f}%")
        print(f"  fit rms (median over sweeps): "
              f"{100 * np.median(laser['fit_rms_raw']):.2f}% raw -> "
              f"{100 * np.median(laser['fit_rms_corrected']):.2f}% laser-corrected")
        print("  correction: " + ("ON" if ref_used else
                                  "OFF (use_reference is false) -- raw powers used"))

    # ---- Stokes vectors ---------------------------------------------------
    fits = stokes_all(states, cal)
    S_in = np.array([f["in"]["S"] for f in fits]).T
    S_out = np.array([f["out"]["S"] for f in fits]).T
    labels = [s["label"] for s in states]

    print(f"\n{'state':>5} {'label':8s} {'input S/S0':>30s}  {'DOP':>5}   "
          f"{'through sample S/S0':>30s}  {'DOP':>5}  fit rms in/out")
    for k, (s, f) in enumerate(zip(states, fits)):
        a_, b_ = S_in[:, k] / S_in[0, k], S_out[:, k] / S_out[0, k]
        rin = 100 * f["in"]["residual_rms"] / np.mean(s["in"]["power_used"])
        rout = 100 * f["out"]["residual_rms"] / np.mean(s["out"]["power_used"])
        dop_in = np.linalg.norm(a_[1:])
        flag = "  <- DOP > 1: retardance / zero wrong?" if dop_in > 1.03 else ""
        print(f"{s['index']:>5} {s['label']:8s} "
              f"[{a_[1]:+.3f} {a_[2]:+.3f} {a_[3]:+.3f}]  {dop_in:5.3f}   "
              f"[{b_[1]:+.3f} {b_[2]:+.3f} {b_[3]:+.3f}]  "
              f"{np.linalg.norm(b_[1:]):5.3f}  {rin:.2f}% / {rout:.2f}%{flag}")

    n = S_in.shape[1]
    rank, cond = coverage(S_in) if n else (0, np.inf)
    print(f"\ncoverage: {n} states, rank {rank} of 4, condition number {cond:.2f} "
          f"(1.73 = best possible)")
    if n < 4 or rank < 4:
        need = ("add a state with a circular component (R or L)"
                if n >= 4 and np.abs(S_in[3] / S_in[0]).max() < 0.2 else
                "measure at least 4 states")
        raise SystemExit(f"cannot reconstruct M: rank {rank} < 4 -- {need}.")
    if cond > 100:
        raise SystemExit("condition number > 100: the states are nearly "
                         "degenerate, M would be noise. Spread them out.")
    if cond > 10:
        print("  WARNING: condition number > 10 -- errors are strongly amplified; "
              "spread the states more (H V D A R L is ideal).")

    # ---- Mueller matrix ---------------------------------------------------
    M = estimate(S_in, S_out)
    m = M / M[0, 0]
    p0 = parameters(M)

    print("\nper-state consistency  |M.S_in - S_out| / S_out0")
    if n == 4:
        print("  exactly 4 states: M fits them perfectly, so there is no "
              "consistency check. Take 5+ states for one.")
        misfit = [0.0] * n
    else:
        misfit = list(np.linalg.norm(M @ S_in - S_out, axis=0) / S_out[0])
        med = np.median(misfit)
        for s, e in zip(states, misfit):
            flag = ("  <- outlier: did the state change or the sample move?"
                    if e > 3 * med and e > 0.01 else "")
            print(f"  state {s['index']:>2} {s['label']:8s} {100 * e:6.2f}%{flag}")

    # ---- errors -----------------------------------------------------------
    rng = np.random.default_rng(0)
    if a.monte_carlo_trials > 0:
        print(f"\nMonte Carlo: {a.monte_carlo_trials} trials ...", end="", flush=True)
        Ms, ps = monte_carlo(states, fits, cal, a.monte_carlo_trials, rng)
        m_err = Ms.std(axis=0, ddof=1)
        stat = {k: spread([p[k] for p in ps], PERIOD.get(k))[1] for k in p0 if k[0] != "_"}
        print(" done")
    else:
        m_err = np.zeros((4, 4))
        stat = {k: float("nan") for k in p0}

    # systematic: WORST CASE over the calibration tolerance box -- each error
    # alone at +-its limit, and both together at every sign combination
    unc = {"qwp_retardance_waves": a.retardance_uncertainty_waves,
           "qwp_zero_deg": a.qwp_zero_uncertainty_deg}
    unc = {k: v for k, v in unc.items() if v}
    keys = [k for k in p0 if k[0] != "_"]
    sys_err, sys_detail, m_sys = {k: 0.0 for k in keys}, {}, np.zeros((4, 4))
    worst_at = {}
    points = [{k: sg * u} for k, u in unc.items() for sg in (+1, -1)]
    points += [dict(zip(unc, sg * np.array(list(unc.values()))))
               for sg in itertools.product((+1, -1), repeat=len(unc))] if len(unc) > 1 else []
    for shift in points:
        c2 = dict(cal)
        for k, v in shift.items():
            c2[k] = cal[k] + v
        f2 = stokes_all(states, c2)
        M2 = estimate(np.array([f["in"]["S"] for f in f2]).T,
                      np.array([f["out"]["S"] for f in f2]).T)
        p2 = parameters(M2)
        m_sys = np.maximum(m_sys, np.abs(M2 / M2[0, 0] - m))
        src = next(iter(shift)) if len(shift) == 1 else "both"
        for k in keys:
            d = float(abs(wrap_diff(p2[k], p0[k], PERIOD.get(k))))
            sys_detail.setdefault(src, {}).setdefault(k, 0.0)
            sys_detail[src][k] = max(sys_detail[src][k], d)
            if d > sys_err[k]:
                sys_err[k], worst_at[k] = d, {kk: float(vv) for kk, vv in shift.items()}

    # ---- report -----------------------------------------------------------
    np.set_printoptions(precision=4, suppress=True, sign="+")
    print(f"\nMueller matrix, normalized (M00 = {M[0, 0]:.4f})   "
          f"value  ± stat  ± worst-case sys")
    for i in range(4):
        print("  " + "  ".join(f"{m[i, j]:+.4f} ±{m_err[i, j]:.4f} ±{m_sys[i, j]:.4f}"
                               for j in range(4)))

    if unc:
        print("\nworst-case systematics for: " + ", ".join(
            f"{k} ± {v:g}" for k, v in unc.items()))
    print(f"\n{'':44s} {'value':>10s} {'± stat':>9s} {'± sys':>9s}")
    for key, label, unit, _ in PARAMS:
        if key in SECTIONS:
            print(f"  {SECTIONS[key]}")
        st = f"{stat[key]:9.4f}" if np.isfinite(stat[key]) else f"{'-':>9s}"
        print(f"    {label:42s} {p0[key]:10.4f} {st} {sys_err[key]:9.4f}  {unit}")

    cl = cloude(M)
    lam_min, lam_err = cl["eigenvalues"].min(), stat["cloude_lambda_min"]
    print("\nverdict")
    if cl["physical"]:
        print("  M is physical (all Cloude eigenvalues >= 0).")
    elif np.isfinite(lam_err) and lam_min > -3 * lam_err:
        print(f"  smallest Cloude eigenvalue {lam_min:+.4f} is negative but within "
              f"3 sigma ({lam_err:.4f}) -- consistent with physical, it is noise.")
    else:
        print(f"  M is NOT physical: smallest Cloude eigenvalue {lam_min:+.4f} "
              f"(noise ~{lam_err:.4f}). Something systematic: PSA calibration, "
              f"a state that changed, or the sample moving between in/out.")
        print(f"  nearest physical M differs by {cl['filter_change']:.4f} "
              f"(max element, normalized) -- saved as M_physical in results.json.")
    if p0["depolarization_index"] < 0.95:
        print(f"  depolarizing sample (DI {p0['depolarization_index']:.3f}): the "
              f"differential numbers assume a clean homogeneous medium -- trust "
              f"Lu-Chipman here.")
    elif p0["diff_residual"] > 0.05:
        print("  large differential residual: the sample is not one homogeneous "
              "medium; Lu-Chipman is the better description.")
    else:
        print("  near non-depolarizing: the differential (LB/LD/CB/CD) numbers "
              "are meaningful; prefer its CD over Lu-Chipman's D3 (no LDLB artifact).")
    if abs(p0["CD"]) < 3 * np.nanmax([stat["CD"], sys_err["CD"], 1e-12]):
        print("  CD is not significant (below 3 x its error).")
    if p0["branch_margin_deg"] < 20:
        print("  retardance close to 180 deg: a thicker sample could wrap to the "
              "next branch -- values are modulo 360 deg.")

    # ---- save -------------------------------------------------------------
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    results = {
        "run_dir": str(run_dir), "calibration_used": cal,
        "reference_used": ref_used, "laser_monitor": laser,
        "calibration_recorded": rec, "states": labels,
        "excluded": list(a.exclude_states),
        "coverage": {"rank": rank, "condition_number": cond},
        "S_in": S_in.tolist(), "S_out": S_out.tolist(),
        "per_state_misfit": [float(x) for x in misfit],
        "M": M.tolist(), "M_normalized": m.tolist(), "M_normalized_err": m_err.tolist(),
        "M_normalized_sys_worst": m_sys.tolist(),
        "M_physical": cl["M_physical"].tolist(),
        "cloude_eigenvalues": cl["eigenvalues"].tolist(),
        "parameters": {k: {"value": float(p0[k]), "stat": float(stat[k]),
                           "sys": float(sys_err[k])} for k in sys_err},
        "systematics": {"method": "worst case over the tolerance box (each source alone and all sign combinations together)", "tolerances": unc,
                        "by_source": sys_detail, "worst_at": worst_at},
        "settings": vars(a),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    with open(out / "mueller.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# normalized Mueller matrix; M00 =", f"{M[0, 0]:.9g}"])
        for i in range(4):
            w.writerow([f"{x:.9g}" for x in m[i]])
        w.writerow(["# 1-sigma statistical errors"])
        for i in range(4):
            w.writerow([f"{x:.9g}" for x in m_err[i]])
    print(f"\nsaved: {out}")

    if a.show_figures or a.save_figures:
        import matplotlib
        if not a.show_figures:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figs = {
            "mueller_matrix": fig_matrix(
                plt, m, m_err, m_sys, f"M / M00   (M00 = {M[0, 0]:.4f}, cond {cond:.2f})"
                               f"\n{run_dir.name}"),
            "poincare": fig_poincare(plt, S_in, S_out, labels),
            "fits": fig_fits(plt, states, fits),
        }
        if laser:
            figs["laser_monitor"] = fig_laser(plt, states, laser)
        if a.save_figures:
            for name, f in figs.items():
                f.savefig(out / f"{name}.png", dpi=150)
        if a.show_figures:
            plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
