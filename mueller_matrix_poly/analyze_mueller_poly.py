#!/usr/bin/env python3
"""Mueller matrix of a sample at every wavelength, from a measure_mueller_poly.py run.

    python analyze_mueller_poly.py              # reads analyze_mueller_poly.json
    python analyze_mueller_poly.py <run_dir>    # or name the run folder directly

The spectral version of ../mueller_matrix/analyze_mueller.py -- the same
steps, once per wavelength bin (bin_nm wide, from wl_min_nm to wl_max_nm):

  1. counts minus the shutter dark, divided by the LED monitor when the run
     has one, summed over the pixels of each bin;
  2. the Stokes vector of every input and through-sample state, per bin, with
     the QWP retardance curve recorded by the measurement -- or the one set
     here (calibration_file: a newer characterization, or the Thorlabs data);
  3. M = S_out . pinv(S_in) per bin, and the checks: coverage (condition
     number per wavelength), per-state misfit, Cloude realizability;
  4. the same decompositions and parameters as the single-wavelength
     analysis, imported from it: Lu-Chipman, differential (logm), Cloude;
  5. error bars per bin: statistical (Monte Carlo from each sweep's residual)
     and worst-case systematic over the calibration tolerances. The QWP
     retardance tolerance is taken per wavelength from the calibration file
     when it has one (the characterization writes it), else
     retardance_uncertainty_waves.

Bins where the LED is weak (below min_signal_rel of the brightest bin), where
the states do not span all four Stokes directions, or where the condition
number is above 100 are left out, and the report says which.

A simulated run (sim_truth.npz in the run folder) is compared with the truth.

Output in <run_dir>/analysis/: results.json, mueller_spectrum.csv,
parameters.csv, and figures.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from poly_common import (angles_of, bin_edges, bin_matrix, binned_power,
                         calibration_at, centres, clipped, cloude, coverage,
                         extract_stokes, has_reference, parse_with_config,
                         psa_blind, read_sweep, resolve_path, str2bool)
from analyze_mueller import PARAMS, PERIOD, SECTIONS, parameters, spread, wrap_diff
from polarization_toolkit.analysis.mueller import mueller_from_stokes

HERE = Path(__file__).resolve().parent
KEYS = [k for k, _, _, _ in PARAMS]


# --------------------------------------------------------------------------- #
# data
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


def sweeps_of(states):
    return [(s, side, s[side]) for s in states for side in ("in", "out")]


def apply_bins(states, edges, use_reference):
    """sw["P"] (K, N) on every sweep; returns (reference used, present, level)."""
    sws = [sw for _, _, sw in sweeps_of(states)]
    present = bool(sws) and all(has_reference(sw) for sw in sws)
    used = present and use_reference
    level = np.mean([sw["ref_W"].mean() for sw in sws]) if present else None
    for sw in sws:
        sw["W"] = bin_matrix(sw["wavelengths"], edges)
        sw["P"] = binned_power(sw, sw["W"], level, used)
    return used, present, level


def restrict(states, keep):
    for _, _, sw in sweeps_of(states):
        sw["P"] = sw["P"][:, keep]
        sw["W"] = sw["W"][:, keep]


def stokes_all(states, cal_c, s3_sign):
    """Fits and Stokes arrays (N, 4, n) of all states for one calibration."""
    fits = [{side: extract_stokes(s[side]["P"], angles_of(s[side]),
                                  zero_deg=cal_c["zero_deg"],
                                  retardance_deg=cal_c["retardance_deg"], s3_sign=s3_sign)
             for side in ("in", "out")} for s in states]
    S_in = np.stack([f["in"]["S"].T for f in fits], axis=-1)
    S_out = np.stack([f["out"]["S"].T for f in fits], axis=-1)
    return fits, S_in, S_out


def estimate(S_in, S_out):
    """M per bin from (N, 4, n) Stokes arrays, via the toolkit."""
    keys = [f"{k:03d}" for k in range(S_in.shape[2])]
    res = mueller_from_stokes({k: S_in[..., i] for i, k in enumerate(keys)},
                              {k: S_out[..., i] for i, k in enumerate(keys)})
    return res.M


def param_arrays(M):
    """parameters() of every bin -> {key: (N,)}."""
    ps = [parameters(Mb) for Mb in M]
    return {k: np.array([p[k] for p in ps], float) for k in KEYS}


def shifted(cal_c, d_ret_deg=0.0, d_zero=0.0):
    return {**cal_c, "retardance_deg": cal_c["retardance_deg"] + d_ret_deg,
            "zero_deg": cal_c["zero_deg"] + d_zero}


def monte_carlo(states, fits, cal_c, s3_sign, trials, rng):
    """Re-draw every sweep from its fitted model + its own residual noise, bin
    by bin, and redo everything."""
    N = fits[0]["in"]["S"].shape[1]
    Ms = np.empty((trials, N, 4, 4))
    ps = {k: np.empty((trials, N)) for k in KEYS}
    for t in range(trials):
        S = {"in": [], "out": []}
        for s, f in zip(states, fits):
            for side in ("in", "out"):
                fit = f[side]
                y = fit["model"] + rng.normal(size=fit["model"].shape) * fit["resid_rms"]
                S[side].append(extract_stokes(y, angles_of(s[side]),
                                              zero_deg=cal_c["zero_deg"],
                                              retardance_deg=cal_c["retardance_deg"],
                                              s3_sign=s3_sign)["S"].T)
        M = estimate(np.stack(S["in"], -1), np.stack(S["out"], -1))
        Ms[t] = M / M[:, :1, :1]
        p = param_arrays(M)
        for k in KEYS:
            ps[k][t] = p[k]
    return Ms, ps


def led_summary(states, cal_c, s3_sign):
    """What the LED did during the run, from the reference meter."""
    sws = sweeps_of(states)
    means = np.array([sw["ref_W"].mean() for _, _, sw in sws])
    noise = []
    for _, _, sw in sws:
        t = sw["t_s"] - sw["t_s"][0]
        r = sw["ref_W"]
        noise.append(np.std(r - np.polyval(np.polyfit(t, r, 1), t)) / r.mean())
    t_all = np.concatenate([sw["t_s"] for _, _, sw in sws])
    in_out = [s["out"]["ref_W"].mean() / s["in"]["ref_W"].mean() - 1 for s in states]

    def rms(sw, P):
        f = extract_stokes(P, angles_of(sw), zero_deg=cal_c["zero_deg"],
                           retardance_deg=cal_c["retardance_deg"], s3_sign=s3_sign)
        return float(np.median(f["resid_rms"] / P.mean(axis=0)))

    raw = [rms(sw, binned_power(sw, sw["W"], use_reference=False)) for _, _, sw in sws]
    cor = [rms(sw, binned_power(sw, sw["W"], means.mean())) for _, _, sw in sws]
    return {"run_minutes": float(np.ptp(t_all) / 60),
            "drift_peak_to_peak": float(np.ptp(means) / means.mean()),
            "noise_per_reading": float(np.median(noise)),
            "max_in_out_change": float(np.max(np.abs(in_out))),
            "in_out_change": [float(x) for x in in_out],
            "fit_rms_raw": raw, "fit_rms_corrected": cor}


def truth_per_bin(run_dir, edges):
    """Simulated runs: the true M averaged over each bin, weighted by the LED."""
    f = run_dir / "sim_truth.npz"
    if not f.is_file():
        return None
    with np.load(f) as d:
        wl, M, led = d["wavelengths"], d["M"], d["led"]
    W = bin_matrix(wl, edges) * led[:, None]
    return np.einsum("pn,pij->nij", W, M) / W.sum(0)[:, None, None]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def fig_mueller(plt, wl, M00, m, m_err, m_sys, M00_err, M00_sys, title, truth=None):
    fig, axs = plt.subplots(4, 4, figsize=(13, 10), sharex=True)
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            if i == j == 0:
                y, st, sy, lab = M00, M00_err, M00_sys, "M00 (transmittance)"
            else:
                y, st, sy, lab = m[:, i, j], m_err[:, i, j], m_sys[:, i, j], f"m{i}{j}"
            tot = np.hypot(st, sy)
            ax.fill_between(wl, y - tot, y + tot, color="C0", alpha=0.2, lw=0)
            ax.fill_between(wl, y - st, y + st, color="C0", alpha=0.4, lw=0)
            ax.plot(wl, y, color="C0", lw=1.2)
            if truth is not None:
                tr = truth[:, 0, 0] if i == j == 0 else truth[:, i, j] / truth[:, 0, 0]
                ax.plot(wl, tr, "k--", lw=0.8)
            ax.set_title(lab, fontsize=9)
            if i != 0 or j != 0:
                ax.set_ylim(min(-0.05, np.nanmin(y - tot) - 0.05),
                            max(0.05, np.nanmax(y + tot) + 0.05))
            ax.axhline(0, color="0.8", lw=0.5)
            ax.tick_params(labelsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("wavelength (nm)", fontsize=8)
    fig.suptitle(title + ("   (dashed: simulated truth)" if truth is not None else "")
                 + "\nshaded: ±stat (dark), ±stat⊕sys (light)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_parameters(plt, wl, p0, stat, sys_err, cond):
    panels = [("M00", "transmittance M00"),
              (("diattenuation", "polarizance"), "diattenuation |D|, polarizance |P|"),
              (("retardance_deg", "linear_retardance_deg"), "retardance (deg): total, linear"),
              ("fast_axis_deg", "fast axis (deg)"),
              (("optical_rotation_deg", "diff_optical_rotation_deg"),
               "optical rotation (deg): Lu-Chipman, differential"),
              ("depolarization_index", "depolarization index (1 = none)"),
              (("LD", "LDp", "CD"), "differential: LD, LD', CD"),
              ("cloude_lambda_min", "smallest Cloude eigenvalue (< 0 unphysical)"),
              (None, "condition number of the input states")]
    fig, axs = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for ax, (keys, title) in zip(axs.flat, panels):
        if keys is None:
            ax.plot(wl, cond, color="C3")
            ax.axhline(1.73, color="0.6", ls="--", lw=0.8)
        else:
            for c, k in enumerate((keys,) if isinstance(keys, str) else keys):
                tot = np.hypot(stat[k], sys_err[k])
                ax.fill_between(wl, p0[k] - tot, p0[k] + tot, color=f"C{c}", alpha=0.25, lw=0)
                ax.plot(wl, p0[k], color=f"C{c}", lw=1.2, label=k)
            if not isinstance(keys, str):
                ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("wavelength (nm)", fontsize=8)
    fig.suptitle("band = ±(stat ⊕ worst-case sys)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_fits(plt, states, fits, b, wl_b):
    n = len(states)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows),
                            squeeze=False, sharex=True)
    for ax, s, f in zip(axs.flat, states, fits):
        for side, c in (("in", "C0"), ("out", "C1")):
            th = angles_of(s[side]) % 360
            order = np.argsort(th)
            y = s[side]["P"][:, b]
            ax.plot(th, y, ".", color=c, ms=3,
                    label=f"{side}: rms {100 * f[side]['resid_rms'][b] / np.mean(y):.2f}%")
            ax.plot(th[order], f[side]["model"][order, b], "-", color=c, lw=1)
        ax.set_title(f"state {s['index']}: {s['label']}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel("counts in bin")
    for ax in axs[-1]:
        ax.set_xlabel("QWP stage angle (deg)")
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"sweeps at {wl_b:.0f} nm", fontsize=10)
    fig.tight_layout()
    return fig


def fig_spectra(plt, states, edges, kept_wl, full_scale):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, s in enumerate(states):
        sw = s["in"]
        dark = 0.5 * (sw["dark_before"] + sw["dark_after"])
        ax.plot(sw["wavelengths"], (sw["counts"] - dark).mean(0), lw=0.8,
                color=f"C{k % 10}", label=f"{s['label']} (mean over the turn)")
        ax.plot(sw["wavelengths"], sw["counts"].max(0), lw=0.5, ls=":", color=f"C{k % 10}")
    ax.axhline(full_scale, color="k", ls="--", lw=0.8, label="full scale (raw)")
    ax.axvspan(edges[0], edges[-1], color="0.9", zorder=0, label="analysis range")
    ax.plot(kept_wl, np.zeros_like(kept_wl), "|", color="k", ms=8, label="bins analysed")
    ax.set_xlim(edges[0] - 60, edges[-1] + 60)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("counts")
    ax.set_title("input sweeps: dark-subtracted mean (solid), raw maximum (dotted)",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def fig_led(plt, states, summary):
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
    ax1.set_ylabel("LED power - mean (%)")
    ax1.set_title(f"reference meter: drift {100 * summary['drift_peak_to_peak']:.2f}% "
                  f"peak-to-peak, noise {100 * summary['noise_per_reading']:.2f}% "
                  f"per reading", fontsize=10)
    ax1.legend(fontsize=8)
    names = [f"{s['label']} {side}" for s in states for side in ("in", "out")]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, 100 * np.array(summary["fit_rms_raw"]), 0.4, label="raw")
    ax2.bar(x + 0.2, 100 * np.array(summary["fit_rms_corrected"]), 0.4,
            label="LED-corrected")
    ax2.set_xticks(x, names, rotation=60, fontsize=7)
    ax2.set_ylabel("fit residual rms (%), median over bins")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="mueller_poly_<timestamp> folder written by measure_mueller_poly.py")
    ap.add_argument("--calibration-file", default=None, dest="calibration_file",
                    help="QWP retardance curve to use instead of the recorded one "
                         "(relative paths: to this folder); null = recorded, "
                         "\"none\" = the constant qwp_retardance_waves")
    ap.add_argument("--qwp-retardance-waves", type=float, default=None,
                    dest="qwp_retardance_waves",
                    help="constant retardance (only without a file); null = recorded")
    ap.add_argument("--qwp-zero-deg", type=float, default=None, dest="qwp_zero_deg",
                    help="override the recorded QWP zero; null = recorded")
    ap.add_argument("--zero-source", default=None, choices=("constant", "file"),
                    dest="zero_source", help="null = recorded")
    ap.add_argument("--s3-sign", type=int, default=None, choices=(1, -1), dest="s3_sign",
                    help="override the recorded S3 sign; null = recorded")
    ap.add_argument("--wl-min-nm", type=float, default=480.0, dest="wl_min_nm")
    ap.add_argument("--wl-max-nm", type=float, default=820.0, dest="wl_max_nm")
    ap.add_argument("--bin-nm", type=float, default=5.0, dest="bin_nm",
                    help="wavelength bin width; the spectrometer resolves ~2 nm")
    ap.add_argument("--min-signal-rel", type=float, default=0.05, dest="min_signal_rel",
                    help="leave out bins with less input light than this fraction of "
                         "the brightest bin")
    ap.add_argument("--use-reference", type=str2bool, default=True, dest="use_reference",
                    help="divide by the reference (LED monitor) meter when the run has one")
    ap.add_argument("--exclude-states", nargs="*", default=[], dest="exclude_states",
                    help="state numbers or labels to leave out")
    ap.add_argument("--monte-carlo-trials", type=int, default=200,
                    dest="monte_carlo_trials", help="0 = no statistical errors")
    ap.add_argument("--retardance-uncertainty-waves", type=float, default=None,
                    dest="retardance_uncertainty_waves",
                    help="QWP retardance tolerance; null = the calibration file's "
                         "retardance_err_waves column, or 1/300 without one")
    ap.add_argument("--qwp-zero-uncertainty-deg", type=float, default=0.1,
                    dest="qwp_zero_uncertainty_deg")
    ap.add_argument("--report-wl-nm", type=float, nargs="*",
                    default=[500, 550, 600, 650, 700, 750, 800], dest="report_wl_nm")
    ap.add_argument("--fit-plot-wl-nm", type=float, default=650.0, dest="fit_plot_wl_nm",
                    help="wavelength of the sweep-fit figure and of the printed matrix")
    ap.add_argument("--show-figures", type=str2bool, default=True, dest="show_figures")
    ap.add_argument("--save-figures", type=str2bool, default=True, dest="save_figures")
    a = parse_with_config(ap, argv, HERE / "analyze_mueller_poly.json")

    if not a.run_dir:
        raise SystemExit("no run folder: set run_dir in analyze_mueller_poly.json "
                         "or pass it on the command line")
    run_dir = Path(a.run_dir).expanduser()
    if not (run_dir / "run.json").is_file():
        raise SystemExit(f"{run_dir} has no run.json -- is it a measure_mueller_poly.py run?")
    print(f"run: {run_dir}")
    meta, states = load(run_dir, {str(x) for x in a.exclude_states})
    if meta.get("kind") != "mueller_spectral":
        raise SystemExit("not a spectral run -- use ../mueller_matrix/analyze_mueller.py")
    full_scale = meta["settings"].get("full_scale_counts", 65535)

    # ---- calibration --------------------------------------------------------
    rec = meta["calibration"]
    cal = dict(rec)
    if a.calibration_file is not None:
        p = resolve_path(a.calibration_file, HERE)
        cal["calibration_file"] = None if p is None else str(p.resolve())
    for k in ("qwp_retardance_waves", "qwp_zero_deg", "zero_source", "s3_sign"):
        if getattr(a, k) is not None:
            cal[k] = getattr(a, k)
    edges = bin_edges(a.wl_min_nm, a.wl_max_nm, a.bin_nm)
    wl_all = centres(edges)
    cal_all = calibration_at(cal, wl_all, run_dir, strict=True)
    changed = {k: rec.get(k) for k in cal if cal[k] != rec.get(k)}
    print("\nPSA calibration")
    print(f"  {cal_all['source']}   "
          f"[{'recorded' if not changed else 'OVERRIDE of ' + str(changed)}]")
    print(f"  s3_sign {cal['s3_sign']}")

    # ---- counts -> bins -------------------------------------------------------
    ref_used, ref_present, _ = apply_bins(states, edges, a.use_reference)
    in_level = np.median([s["in"]["P"].mean(0) for s in states], axis=0)
    weak = in_level < a.min_signal_rel * in_level.max()
    band = (states[0]["in"]["wavelengths"] >= edges[0]) & \
           (states[0]["in"]["wavelengths"] <= edges[-1])
    clips = [(s["label"], side, int(clipped(sw["counts"][:, band], full_scale).sum()))
             for s, side, sw in sweeps_of(states)]
    clips = [c for c in clips if c[2]]

    fits, S_in, S_out = stokes_all(states, cal_all, cal["s3_sign"])
    n = S_in.shape[2]
    rank, cond = coverage(S_in) if n else (np.zeros(len(wl_all), int), np.full(len(wl_all), np.inf))
    blind = psa_blind(cal_all["retardance_deg"])
    keep = ~weak & (rank == 4) & (cond <= 100)

    print("\nspectrometer")
    print(f"  exposure {meta['hardware'].get('exposure_ms')} ms x "
          f"{meta['hardware'].get('hw_average')} frames, full scale {full_scale:g} counts")
    print("  clipping: " + ("none" if not clips else
                            "; ".join(f"{l} {sd}: {k} spectra" for l, sd, k in clips)
                            + "  <- those sweeps are NOT linear: redo them"))

    laser = None
    print("\nLED monitor (reference meter)")
    if not ref_present:
        print("  none in this run -- LED drift and noise are in every sweep and in M00")
    else:
        laser = led_summary(states, cal_all, cal["s3_sign"])
        print(f"  run {laser['run_minutes']:.1f} min | LED drift "
              f"{100 * laser['drift_peak_to_peak']:.2f}% peak-to-peak | noise "
              f"{100 * laser['noise_per_reading']:.2f}% per reading")
        print(f"  largest LED change between a state's input and through-sample sweeps: "
              f"{100 * laser['max_in_out_change']:.2f}%")
        print(f"  fit rms (median): {100 * np.median(laser['fit_rms_raw']):.2f}% raw -> "
              f"{100 * np.median(laser['fit_rms_corrected']):.2f}% LED-corrected")
        print("  correction: " + ("ON" if ref_used else
                                  "OFF (use_reference is false) -- raw counts used"))

    print(f"\nwavelength bins: {len(wl_all)} x {a.bin_nm:g} nm, "
          f"{edges[0]:g}-{edges[-1]:g} nm")
    for mask, why in ((weak, f"too little light (< {a.min_signal_rel:g} of the peak)"),
                      (~weak & (rank < 4), "rank < 4: the states miss a Stokes direction"),
                      (~weak & (rank == 4) & (cond > 100), "condition number > 100"),
                      (blind, "polarimeter QWP near 0 / 180 deg: nearly blind (kept)")):
        if mask.any():
            print(f"  {mask.sum():3d} bins, {wl_all[mask][0]:.0f}-{wl_all[mask][-1]:.0f} nm: {why}")
    if not keep.any():
        raise SystemExit("no usable wavelength bin")
    print(f"  analysed: {keep.sum()} bins, {wl_all[keep][0]:.0f}-{wl_all[keep][-1]:.0f} nm")

    # ---- everything below on the kept bins only -------------------------------
    restrict(states, keep)
    wl = wl_all[keep]
    cal_c = {**cal_all, "retardance_deg": cal_all["retardance_deg"][keep],
             "zero_deg": (cal_all["zero_deg"][keep] if np.ndim(cal_all["zero_deg"])
                          else cal_all["zero_deg"])}
    if cal_all["retardance_err_waves"] is not None:
        cal_c["retardance_err_waves"] = cal_all["retardance_err_waves"][keep]
    fits, S_in, S_out = stokes_all(states, cal_c, cal["s3_sign"])
    cond_all, cond = cond, cond[keep]
    N = len(wl)
    labels = [s["label"] for s in states]

    print(f"\ncoverage: {n} states, condition number median {np.median(cond):.2f}, "
          f"worst {cond.max():.2f} at {wl[np.argmax(cond)]:.0f} nm (1.73 = best possible)")
    if cond.max() > 10:
        print(f"  WARNING: condition number > 10 at {wl[cond > 10][0]:.0f}-"
              f"{wl[cond > 10][-1]:.0f} nm -- errors are strongly amplified there")

    M = estimate(S_in, S_out)
    M00 = M[:, 0, 0]
    m = M / M00[:, None, None]
    p0 = param_arrays(M)

    print("\nper-state consistency  |M.S_in - S_out| / S_out0, median (max) over wavelength")
    if n == 4:
        print("  exactly 4 states: M fits them perfectly, so there is no consistency "
              "check. Take 5+ states for one.")
        misfit = np.zeros((N, n))
    else:
        misfit = np.linalg.norm(M @ S_in - S_out, axis=1) / S_out[:, 0, :]
        med = np.median(misfit, axis=0)
        ref_med = np.median(med)
        for k, s in enumerate(states):
            flag = ("  <- outlier: did the state change or the sample move?"
                    if med[k] > 3 * ref_med and med[k] > 0.01 else "")
            print(f"  state {s['index']:>2} {s['label']:8s} {100 * med[k]:6.2f}% "
                  f"({100 * misfit[:, k].max():.2f}%){flag}")

    # ---- errors ---------------------------------------------------------------
    rng = np.random.default_rng(0)
    if a.monte_carlo_trials > 0:
        print(f"\nMonte Carlo: {a.monte_carlo_trials} trials x {N} bins ...", end="",
              flush=True)
        Ms, ps = monte_carlo(states, fits, cal_c, cal["s3_sign"], a.monte_carlo_trials, rng)
        m_err = Ms.std(axis=0, ddof=1)
        stat = {k: np.array([spread(ps[k][:, b], PERIOD.get(k))[1] for b in range(N)])
                for k in KEYS}
        M00_err = stat["M00"]
        print(" done")
    else:
        m_err = np.zeros((N, 4, 4))
        stat = {k: np.full(N, np.nan) for k in KEYS}
        M00_err = np.zeros(N)

    # systematic: WORST CASE over the calibration tolerance box, per bin
    if a.retardance_uncertainty_waves is not None:
        tol_ret, tol_src = np.full(N, a.retardance_uncertainty_waves), "setting"
    elif cal_c.get("retardance_err_waves") is not None:
        tol_ret, tol_src = np.nan_to_num(cal_c["retardance_err_waves"], nan=1 / 300), \
            "calibration file (per wavelength)"
    else:
        tol_ret, tol_src = np.full(N, 1 / 300), "default 1/300 wave"
    unc = {"ret": 360.0 * tol_ret, "zero": a.qwp_zero_uncertainty_deg}
    unc = {k: v for k, v in unc.items() if np.any(v)}
    points = [{k: sg * np.asarray(u)} for k, u in unc.items() for sg in (+1, -1)]
    if len(unc) > 1:
        points += [{k: sg[i] * np.asarray(u) for i, (k, u) in enumerate(unc.items())}
                   for sg in itertools.product((+1, -1), repeat=len(unc))]
    sys_err = {k: np.zeros(N) for k in KEYS}
    m_sys = np.zeros((N, 4, 4))
    for pt in points:
        c2 = shifted(cal_c, pt.get("ret", 0.0), pt.get("zero", 0.0))
        _, Si, So = stokes_all(states, c2, cal["s3_sign"])
        M2 = estimate(Si, So)
        m_sys = np.maximum(m_sys, np.abs(M2 / M2[:, :1, :1] - m))
        p2 = param_arrays(M2)
        for k in KEYS:
            sys_err[k] = np.maximum(sys_err[k], np.abs(wrap_diff(p2[k], p0[k], PERIOD.get(k))))
    M00_sys = sys_err["M00"]

    # ---- report ---------------------------------------------------------------
    b0 = int(np.argmin(abs(wl - a.fit_plot_wl_nm)))
    print(f"\nworst-case systematics for: QWP retardance ± "
          f"{np.median(tol_ret):.5f} waves (median; from the {tol_src}), "
          f"zero ± {a.qwp_zero_uncertainty_deg:g} deg")
    print(f"\nMueller matrix at {wl[b0]:.0f} nm, normalized (M00 = {M00[b0]:.4f})   "
          f"value  ± stat  ± worst-case sys")
    for i in range(4):
        print("  " + "  ".join(f"{m[b0, i, j]:+.4f} ±{m_err[b0, i, j]:.4f} "
                               f"±{m_sys[b0, i, j]:.4f}" for j in range(4)))

    cols = [int(np.argmin(abs(wl - w))) for w in a.report_wl_nm
            if wl[0] - a.bin_nm <= w <= wl[-1] + a.bin_nm]
    cols = list(dict.fromkeys(cols))
    print(f"\n{'value ± (stat ⊕ sys)':44s}" + "".join(f"{wl[c]:>14.0f}" for c in cols)
          + "  nm")
    for key, label, unit, _ in PARAMS:
        if key in SECTIONS:
            print(f"  {SECTIONS[key]}")
        cells = []
        for c in cols:
            tot = np.hypot(np.nan_to_num(stat[key][c]), sys_err[key][c])
            cells.append(f"{p0[key][c]:>8.3f}±{tot:<5.3f}"[:14].rjust(14))
        print(f"    {label[:40]:40s}" + "".join(cells) + f"  {unit}")

    print("\nverdict")
    lam, lam_err = p0["cloude_lambda_min"], stat["cloude_lambda_min"]
    unphys = (lam < -1e-3) & ~(np.isfinite(lam_err) & (lam > -3 * lam_err))
    if not unphys.any():
        print("  M is physical at every wavelength (Cloude eigenvalues >= 0 within noise).")
    else:
        print(f"  M is NOT physical at {unphys.sum()} of {N} wavelengths "
              f"({wl[unphys][0]:.0f}-{wl[unphys][-1]:.0f} nm): something systematic -- "
              f"the QWP retardance curve there, a state that changed, or the sample moving.")
    DI = p0["depolarization_index"]
    if DI.min() < 0.95:
        print(f"  depolarizing sample (DI down to {DI.min():.3f} at {wl[np.argmin(DI)]:.0f} "
              f"nm): the differential numbers assume a clean homogeneous medium -- trust "
              f"Lu-Chipman there.")
    elif p0["diff_residual"].max() > 0.05:
        print("  large differential residual somewhere: the sample is not one homogeneous "
              "medium; Lu-Chipman is the better description there.")
    else:
        print("  near non-depolarizing: the differential (LB/LD/CB/CD) numbers hold for ONE "
              "homogeneous medium. A stack of different elements (e.g. a diattenuator then a "
              "retarder) mixes them into a fake CD; M alone cannot tell -- use what you "
              "know about the sample to choose between differential and Lu-Chipman.")
    cd_err = np.fmax(np.nan_to_num(stat["CD"]), sys_err["CD"])
    sig = np.abs(p0["CD"]) > 3 * np.fmax(cd_err, 1e-12)
    print(f"  CD above 3 x its error at {sig.sum()} of {N} wavelengths.")
    if (p0["branch_margin_deg"] < 20).any():
        print("  retardance close to 180 deg somewhere: a thicker sample can wrap to the "
              "next branch -- values are modulo 360 deg.")

    truth = truth_per_bin(run_dir, edges)
    if truth is not None:
        truth = truth[keep]
        mt = truth / truth[:, :1, :1]
        dev = np.abs(m - mt).max(axis=(1, 2))
        pt = param_arrays(truth)
        print("\nSIMULATED RUN -- compared with the truth (sim_truth.npz)")
        print(f"  max |m_ij - true| per wavelength: median {np.median(dev):.4f}, "
              f"worst {dev.max():.4f} at {wl[np.argmax(dev)]:.0f} nm")
        for key in ("M00", "diattenuation", "retardance_deg", "fast_axis_deg",
                    "depolarization_index"):
            d = wrap_diff(p0[key], pt[key], PERIOD.get(key))
            tot = np.hypot(np.nan_to_num(stat[key]), sys_err[key])
            print(f"  {key:22s} measured - true: rms {np.sqrt(np.mean(d ** 2)):.4f}, "
                  f"max {np.abs(d).max():.4f}; within 3x its error at "
                  f"{int((np.abs(d) <= 3 * tot + 1e-12).sum())} of {N} wavelengths")

    # ---- save -----------------------------------------------------------------
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    tolist = lambda x: np.asarray(x, float).tolist()
    results = {
        "run_dir": str(run_dir), "calibration_used": cal, "calibration_source": cal_all["source"],
        "calibration_recorded": rec, "reference_used": ref_used, "led_monitor": laser,
        "states": labels, "excluded": list(a.exclude_states),
        "wavelength_nm": tolist(wl), "bin_nm": a.bin_nm,
        "bins_left_out": {"too_little_light": tolist(wl_all[weak]),
                          "rank_below_4": tolist(wl_all[~weak & (rank < 4)]),
                          "condition_over_100": tolist(wl_all[~weak & (rank == 4) & (cond_all > 100)])},
        "condition_number": tolist(cond),
        "qwp_retardance_deg": tolist(cal_c["retardance_deg"]),
        "S_in": tolist(S_in), "S_out": tolist(S_out),
        "per_state_misfit": tolist(misfit),
        "M": tolist(M), "M_normalized": tolist(m), "M_normalized_err": tolist(m_err),
        "M_normalized_sys_worst": tolist(m_sys),
        "M_physical": [tolist(cloude(Mb)["M_physical"]) for Mb in M],
        "parameters": {k: {"value": tolist(p0[k]), "stat": tolist(stat[k]),
                           "sys": tolist(sys_err[k])} for k in KEYS},
        "systematics": {"method": "worst case over the tolerance box (each source alone "
                                  "and all sign combinations together), per wavelength",
                        "retardance_tolerance_waves": tolist(tol_ret),
                        "retardance_tolerance_source": tol_src,
                        "zero_tolerance_deg": a.qwp_zero_uncertainty_deg},
        "settings": vars(a),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    with open(out / "mueller_spectrum.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = [f"m{i}{j}" for i in range(4) for j in range(4)][1:]
        w.writerow(["wavelength_nm", "M00", "M00_stat", "M00_sys"] + names
                   + [f"{x}_stat" for x in names] + [f"{x}_sys" for x in names])
        for b in range(N):
            flat = lambda A: [f"{v:.7g}" for v in A[b].ravel()[1:]]
            w.writerow([f"{wl[b]:.3f}", f"{M00[b]:.7g}", f"{M00_err[b]:.7g}",
                        f"{M00_sys[b]:.7g}"] + flat(m) + flat(m_err) + flat(m_sys))
    with open(out / "parameters.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wavelength_nm"] + [f"{k}{sfx}" for k in KEYS
                                        for sfx in ("", "_stat", "_sys")])
        for b in range(N):
            w.writerow([f"{wl[b]:.3f}"] + [f"{v:.7g}" for k in KEYS
                                           for v in (p0[k][b], stat[k][b], sys_err[k][b])])
    print(f"\nsaved: {out}")

    if a.show_figures or a.save_figures:
        import matplotlib
        if not a.show_figures:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figs = {
            "mueller_spectrum": fig_mueller(
                plt, wl, M00, m, m_err, m_sys, M00_err, M00_sys,
                f"M(λ) / M00   {run_dir.name}", truth),
            "parameters": fig_parameters(plt, wl, p0, stat, sys_err, cond),
            "fits": fig_fits(plt, states, fits, b0, wl[b0]),
            "spectra": fig_spectra(plt, states, edges, wl, full_scale),
        }
        if laser:
            figs["led_monitor"] = fig_led(plt, states, laser)
        if a.save_figures:
            for name, f in figs.items():
                f.savefig(out / f"{name}.png", dpi=130)
        if a.show_figures:
            plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
