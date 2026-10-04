#!/usr/bin/env python3
"""Measure a sample with DMD-selected wavelengths: sample in / sample out.

    python dmd_measure.py                     # reads dmd_measure.json, and the
                                              # shared hardware settings from
                                              # dmd_calibration.json
    python dmd_measure.py --simulate          # rehearse, no hardware
    python dmd_measure.py --analyze <file>    # analyse saved data again
                                              # (a *_meta.json or a scan folder)

    source -> prism -> spectrum on the DMD -> prism (recombine) -> [sample] -> spectrometer

The newest calibration from dmd_calibration.py (or calibration_file) turns the
wavelengths you type into line positions on the DMD.

A menu (q quits):
  1  single measurement: 1, 2 or 3 wavelengths at once, an exposure and a
     number of frames, then sample IN -> frames, sample OUT -> frames.
     Saved in <data_dir>/single/ as <date>_<time>_<n>wl_*.
  2  scan, saved in <data_dir>/scan/<date>_<time>_<type>/:
     1  one peak stepped from start to stop: the whole scan with the sample
        IN, then the whole scan with it OUT;
     2  the same, plus a stationary peak at a wavelength you choose, shown
        together with the scanning one at every step.

Frames: every spectrum is taken that many times (each one a separate
spectrum of hw_average x exposure). Results are the mean over the frames;
the error bars are +- their standard deviation (error "std"), or the
standard error of the mean with the background's own error added ("sem").

Every frame has a background subtracted (`background`): the DMD blocking
everything, or the spectrometer shutter, averaged over background_frames.
Single: right after the frames. Scan: before and after each pass,
interpolated in time.

After a measurement you choose the analysis, one or more:
  1  subtract    in - out
  2  divide      in / out                (the transmission)
  3  absorbance  -log10(in / out)        (positive when the sample absorbs)
Single: on the whole spectrum, and on the counts integrated over each peak.
Scan: on the counts integrated over the peak, at every step.
A peak is integrated over its centre +- band_halfwidth_nm (null: +- its
FWHM), the centre and FWHM taken from the sample-OUT spectrum. The integral
is taken frame by frame; its mean and spread give the error, propagated:
  subtract    sqrt(e_in^2 + e_out^2)
  divide      T * sqrt((e_in/I_in)^2 + (e_out/I_out)^2)
  absorbance  sqrt((e_in/I_in)^2 + (e_out/I_out)^2) / ln 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from dmd_common import (HERE, Calibration, ask_choice, ask_exposure, ask_float,
                        ask_int, clipped, data_root, fill_report, find_calibration,
                        find_peak, finish_figure, float_or_none, get_plt, hardware_args,
                        interp_background, make_rig, new_folder, parse_config,
                        read_csv, stamp, str2bool, write_csv)

SIDES = (("in", "Put the sample IN, then press Enter to measure..."),
         ("out", "Take the sample OUT, then press Enter to measure..."))
ANALYSES = {"1": "subtract", "2": "divide", "3": "absorbance"}
LABELS = {"subtract": "in - out (counts)", "divide": "in / out",
          "absorbance": "-log10(in / out)"}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def mean_err(x, bg_err, a, mode=None):
    """Mean over the frames (axis 0) and its error bar: the frames' standard
    deviation (error "std"), or the standard error of the mean with the
    background's own error added (error "sem"). mode overrides a.error.
    NaN with a single frame."""
    x = np.asarray(x, float)
    n = x.shape[0]
    mean = x.mean(axis=0)
    if n < 2:
        return mean, np.full(np.shape(mean), np.nan)
    std = x.std(axis=0, ddof=1)
    if (mode or a.error) == "std":
        return mean, std
    return mean, np.sqrt(std ** 2 / n + np.asarray(bg_err) ** 2)


def bg_mean_err(b):
    """Error of a background averaged over its frames (axis 0); 0 with one."""
    b = np.asarray(b, float)
    if b.shape[0] < 2:
        return np.zeros(b.shape[1:])
    return b.std(axis=0, ddof=1) / np.sqrt(b.shape[0])


def analyse(kind, i_in, i_out, min_out, e_in=np.nan, e_out=np.nan):
    """(value, error) of in - out, in / out or -log10(in / out), first-order
    error propagation (in and out are independent measurements). The ratios
    are NaN where the sample-out signal is below min_out."""
    i_in, i_out = np.asarray(i_in, float), np.asarray(i_out, float)
    e_in, e_out = np.asarray(e_in, float), np.asarray(e_out, float)
    if kind == "subtract":
        return i_in - i_out, np.hypot(e_in, e_out)
    with np.errstate(all="ignore"):
        ratio = np.where(i_out >= min_out, i_in / i_out, np.nan)
        rel = np.hypot(e_in / i_in, e_out / i_out)
        if kind == "divide":
            return ratio, ratio * rel
        return np.where(ratio > 0, -np.log10(ratio), np.nan), rel / np.log(10)


def pm(v, e, plot=False):
    """'value +- error': the error to 2 significant digits, the value to the
    same decimal place, large or small numbers with a common power of ten."""
    v, e = float(v), float(e)
    if not np.isfinite(v):
        return "nan"
    if not np.isfinite(e) or e <= 0:
        return f"{v:.4g}"
    sep = " ± " if plot else " +- "
    d = 1 - int(np.floor(np.log10(e)))
    p = int(np.floor(np.log10(max(abs(v), e))))
    if abs(p) >= 4:
        dd = max(d + p, 0)
        body = f"({v / 10 ** p:.{dd}f}{sep}{e / 10 ** p:.{dd}f})"
        return body + (f"$\\times10^{{{p}}}$" if plot else f"e{p}")
    return f"{v:.{max(d, 0)}f}{sep}{e:.{max(d, 0)}f}"


def error_note(a, n):
    if n < 2:
        return "1 frame: no error bars"
    return (f"± 1 standard deviation of {n} frames" if a.error == "std" else
            f"± standard error of the mean of {n} frames (+ background)")


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def ask_analyses():
    while True:
        txt = input("\nAnalysis: 1 = subtract (in - out), 2 = divide (in / out), "
                    "3 = absorbance -log10(in / out)\n"
                    "  one or more (e.g. 1,3), a = all, Enter = none: ").strip().lower()
        if not txt:
            return []
        if txt in ("a", "all"):
            return list(ANALYSES.values())
        keys = txt.replace(",", " ").split()
        if all(k in ANALYSES for k in keys):
            return [ANALYSES[k] for k in dict.fromkeys(keys)]
        print("  1, 2, 3, a or Enter")


def band_window(wl, out, target, nominal_hw, a, search):
    """(lo, hi, centre, fwhm) of the peak near target, from the sample-OUT
    spectrum. Without a clear peak there: the target and the line's nominal
    width."""
    p = find_peak(wl, out, target - search, target + search)
    found = (p is not None and p["peak"] >= a.min_signal_counts
             and np.isfinite(p["fwhm_nm"]))
    centre = p["centre_nm"] if found else target
    fwhm = p["fwhm_nm"] if found else np.nan
    hw = a.band_halfwidth_nm or (fwhm if found else nominal_hw)
    return centre - hw, centre + hw, centre, fwhm


def search_radius(a, target, others):
    """search_nm, but never as far as half-way to another peak."""
    d = [abs(target - o) for o in others if o != target]
    return min([a.search_nm] + [0.5 * x for x in d])


def load_meta(path: Path):
    path = Path(path)
    if path.is_dir():
        found = sorted(path.glob("*_meta.json"))
        if len(found) != 1:
            raise SystemExit(f"{path}: expected one *_meta.json, found {len(found)}")
        path = found[0]
    if not path.is_file():
        raise SystemExit(f"not found: {path}")
    meta = json.loads(path.read_text())
    if "frames" not in meta.get("files", {}):
        raise SystemExit(f"{path}: no frames file -- measured with an older version of "
                         f"this script")
    return path, meta


def prefix_of(meta_path: Path) -> str:
    return meta_path.name[:-len("_meta.json")]


def load_frames(folder, meta):
    with np.load(folder / meta["files"]["frames"]) as f:
        return {k: f[k].astype(float) for k in f.files}


def analyse_single(meta_path: Path, a, kinds):
    meta_path, meta = load_meta(meta_path)
    folder, pre, suf = meta_path.parent, prefix_of(meta_path), a.error
    d = load_frames(folder, meta)
    wl = d["wavelength_nm"]
    net = {s: d[f"{s}_frames"] - d[f"{s}_background"].mean(axis=0) for s in ("in", "out")}
    bg = {s: d[f"{s}_background"] for s in ("in", "out")}
    n = net["in"].shape[0]
    px = {s: mean_err(net[s], bg_mean_err(bg[s]), a) for s in ("in", "out")}   # per pixel
    cal_nm, nominal = meta["cal_nm"], meta["line_band_nm"]

    win = [band_window(wl, px["out"][0], c, nominal[i], a, search_radius(a, c, cal_nm))
           for i, c in enumerate(cal_nm)]
    overlap = [any(j != i and w[0] < v[1] and v[0] < w[1] for j, v in enumerate(win))
               for i, w in enumerate(win)]

    def integral(s, w):
        m = (wl >= w[0]) & (wl <= w[1])
        return mean_err(net[s][:, m].sum(axis=1), bg_mean_err(bg[s][:, m].sum(axis=1)), a)

    I = {s: np.array([integral(s, w) for w in win]) for s in ("in", "out")}   # (peaks, 2)
    peaks = {"target_nm": meta["targets_nm"], "cal_nm": cal_nm,
             "centre_nm": [w[2] for w in win], "fwhm_nm": [w[3] for w in win],
             "window_lo_nm": [w[0] for w in win], "window_hi_nm": [w[1] for w in win],
             "in_counts": I["in"][:, 0], f"in_counts_{suf}": I["in"][:, 1],
             "out_counts": I["out"][:, 0], f"out_counts_{suf}": I["out"][:, 1]}
    for k in kinds:
        peaks[k], peaks[f"{k}_{suf}"] = analyse(k, I["in"][:, 0], I["out"][:, 0],
                                                a.min_signal_counts, I["in"][:, 1],
                                                I["out"][:, 1])
    peaks["overlap"] = np.array(overlap, int)
    peaks["frames"] = np.full(len(cal_nm), n)

    print(f"\n  errors: {error_note(a, n).replace(chr(177), '+-')}")
    for i in range(len(cal_nm)):
        print(f"  {cal_nm[i]:.2f} nm: centre {win[i][2]:.2f} nm, FWHM {win[i][3]:.2f} nm, "
              f"window {win[i][0]:.2f}-{win[i][1]:.2f} nm"
              + ("   (windows overlap)" if overlap[i] else ""))
        print(f"      in   {pm(*I['in'][i])}      out  {pm(*I['out'][i])}")
        if kinds:
            print("      " + "      ".join(f"{k} {pm(peaks[k][i], peaks[f'{k}_{suf}'][i])}"
                                         for k in kinds))
    if any(overlap):
        print("  WARNING: the integration windows of some peaks overlap -- choose "
              "wavelengths further apart or set band_halfwidth_nm")

    write_csv(folder / f"{pre}_peaks.csv", peaks)
    per_px = {}
    for k in kinds:
        per_px[k] = analyse(k, px["in"][0], px["out"][0], a.min_signal_counts,
                            px["in"][1], px["out"][1])
        write_csv(folder / f"{pre}_{k}.csv", {"wavelength_nm": wl, k: per_px[k][0],
                                              f"{k}_{suf}": per_px[k][1]})
    print(f"  saved: {pre}_peaks.csv" + "".join(f", {pre}_{k}.csv" for k in kinds))
    if kinds:
        plot_single(a, folder / f"{pre}_analysis.png", meta, wl, px, per_px, win, peaks,
                    kinds, n)


def scan_patterns(off, stat_off):
    """What the DMD shows at one scan step, in this order. With a stationary
    peak: the scanning line alone, both lines, the stationary line alone --
    so 'both' can be compared with the sum of the two alone."""
    if stat_off is None:
        return {"scan": [off]}
    return {"scan": [off], "both": [off, stat_off], "stat": [stat_off]}


def analyse_scan(meta_path: Path, a, kinds):
    meta_path, meta = load_meta(meta_path)
    if "patterns" not in meta:
        raise SystemExit(f"{meta_path}: measured with an older version of this script")
    folder, pre, suf = meta_path.parent, prefix_of(meta_path), a.error
    d = load_frames(folder, meta)
    wl = d["wavelength_nm"]
    steps = read_csv(folder / meta["files"]["steps"])
    cal_nm, nominal = steps["cal_nm"], steps["line_band_nm"]
    stat, names = meta.get("stationary"), meta["patterns"]
    K, n = d[f"in_{names[0]}_frames"].shape[:2]

    net, bgs = {}, {}
    for s in ("in", "out"):
        b0, b1 = d[f"{s}_bg_before"], d[f"{s}_bg_after"]
        tb0, tb1 = d[f"t_bg_{s}"]
        for p in names:
            t = d[f"t_{s}_{p}"]
            f = np.clip((t - tb0) / (tb1 - tb0), 0, 1) if tb1 > tb0 else np.full(K, 0.5)
            bg = interp_background(t, tb0, tb1, b0.mean(axis=0), b1.mean(axis=0))
            net[s, p] = d[f"{s}_{p}_frames"] - bg[:, None, :]          # (K, n, P)
            bgs[s, p] = (f, b0, b1)

    def integral(s, p, k, m):
        """Counts of pattern p summed over the pixels m at step k:
        (mean, error bar, standard error of the mean)."""
        f, b0, b1 = bgs[s, p]
        e_bg = np.hypot((1 - f[k]) * bg_mean_err(b0[:, m].sum(axis=1)),
                        f[k] * bg_mean_err(b1[:, m].sum(axis=1)))
        x = net[s, p][k][:, m].sum(axis=1)
        return (*mean_err(x, e_bg, a), mean_err(x, e_bg, a, "sem")[1])

    def window_mask(w):
        return (wl >= w[0]) & (wl <= w[1])

    # each peak from the spectra where it is ALONE on the DMD
    res = {"step": np.arange(K), "cal_nm": cal_nm}
    out_scan = net["out", "scan"].mean(axis=1)
    scan_win = [band_window(wl, out_scan[k], cal_nm[k], nominal[k], a, a.search_nm)
                for k in range(K)]
    res["centre_nm"] = np.array([w[2] for w in scan_win])
    res["fwhm_nm"] = np.array([w[3] for w in scan_win])
    for s in ("in", "out"):
        v = np.array([integral(s, "scan", k, window_mask(scan_win[k])) for k in range(K)])
        res[f"{s}_counts"], res[f"{s}_counts_{suf}"] = v[:, 0], v[:, 1]
    if stat:
        out_stat = net["out", "stat"].mean(axis=1)
        stat_win = [band_window(wl, out_stat[k], stat["cal_nm"], stat["line_band_nm"], a,
                                a.search_nm) for k in range(K)]
        res["stat_centre_nm"] = np.array([w[2] for w in stat_win])
        for s in ("in", "out"):
            v = np.array([integral(s, "stat", k, window_mask(stat_win[k])) for k in range(K)])
            res[f"stat_{s}_counts"], res[f"stat_{s}_counts_{suf}"] = v[:, 0], v[:, 1]
    for kind in kinds:
        for p in ("", "stat_") if stat else ("",):
            res[f"{p}{kind}"], res[f"{p}{kind}_{suf}"] = analyse(
                kind, res[f"{p}in_counts"], res[f"{p}out_counts"], a.min_signal_counts,
                res[f"{p}in_counts_{suf}"], res[f"{p}out_counts_{suf}"])

    # both together vs the sum of the two alone
    delta = None
    if stat:
        lo_v, hi_v = meta["view_nm"]
        view = (wl >= lo_v) & (wl <= hi_v)
        peaks_m = [window_mask(scan_win[k]) | window_mask(stat_win[k]) for k in range(K)]
        R = {}
        for s in ("in", "out"):
            rows = []
            for k in range(K):
                B, A, S = (integral(s, p, k, peaks_m[k]) for p in ("both", "scan", "stat"))
                Q = A[0] + S[0]
                with np.errstate(all="ignore"):
                    r = B[0] / Q
                    e = r * np.hypot(B[1] / B[0], np.hypot(A[1], S[1]) / Q)
                    e_sem = r * np.hypot(B[2] / B[0], np.hypot(A[2], S[2]) / Q)
                Bo, Ao, So = (integral(s, p, k, view & ~peaks_m[k])
                              for p in ("both", "scan", "stat"))
                rows.append((r, e, e_sem, Bo[0] - Ao[0] - So[0],
                             np.sqrt(Bo[1] ** 2 + Ao[1] ** 2 + So[1] ** 2)))
            v = np.array(rows)
            R[s] = v
            res[f"excess_{s}_pct"] = 100 * (v[:, 0] - 1)
            res[f"excess_{s}_pct_{suf}"] = 100 * v[:, 1]
            res[f"excess_{s}_z"] = (v[:, 0] - 1) / v[:, 2]
            res[f"other_{s}_counts"], res[f"other_{s}_counts_{suf}"] = v[:, 3], v[:, 4]
        with np.errstate(all="ignore"):
            res["extra_absorbance"] = -np.log10(R["in"][:, 0] / R["out"][:, 0])
            res[f"extra_absorbance_{suf}"] = np.hypot(R["in"][:, 1] / R["in"][:, 0],
                                                      R["out"][:, 1] / R["out"][:, 0]) / np.log(10)
            sem = np.hypot(R["in"][:, 2] / R["in"][:, 0],
                           R["out"][:, 2] / R["out"][:, 0]) / np.log(10)
            res["extra_absorbance_z"] = res["extra_absorbance"] / sem
        res["windows_overlap"] = np.array([w[0] < v[1] and v[0] < w[1]
                                           for w, v in zip(scan_win, stat_win)]).astype(int)
        delta = {s: net[s, "both"].mean(axis=1) - net[s, "scan"].mean(axis=1)
                 - net[s, "stat"].mean(axis=1) for s in ("in", "out")}       # (K, P)

    # ---- report ----------------------------------------------------------- #
    shift = res["centre_nm"] - cal_nm
    good = np.isfinite(res["fwhm_nm"])
    print(f"\n  {K} steps, {cal_nm.min():.1f}-{cal_nm.max():.1f} nm; errors: "
          f"{error_note(a, n).replace(chr(177), '+-')}")
    if good.any():
        print(f"  peak centres vs calibration: mean {np.mean(shift[good]):+.2f} nm, "
              f"worst {shift[good][np.argmax(np.abs(shift[good]))]:+.2f} nm")
    if stat:
        r = analyse("divide", res["stat_in_counts"], res["stat_out_counts"],
                    a.min_signal_counts)[0]
        if np.isfinite(r).any():
            print(f"  stationary peak alone ({stat['cal_nm']:.1f} nm): in/out "
                  f"{np.nanmean(r):.4g}, spread {100 * np.nanstd(r) / np.nanmean(r):.2f}% "
                  f"over the scan")
    for kind in kinds:
        v, e = res[kind], res[f"{kind}_{suf}"]
        if np.isfinite(v).any():
            print(f"  {kind:10s}: {np.nanmin(v):.4g} to {np.nanmax(v):.4g}"
                  + (f", typical error +- {np.nanmedian(e):.2g}" if np.isfinite(e).any()
                     else ""))
    if stat:
        report_interaction(res, cal_nm, n)

    write_csv(folder / f"{pre}_results.csv", res)
    saved = [f"{pre}_results.csv"]
    if delta is not None:
        cols_names = [f"step{k:03d}_{c:.2f}nm" for k, c in enumerate(cal_nm)]
        for s in ("in", "out"):
            cols = {"wavelength_nm": wl}
            cols.update(zip(cols_names, delta[s]))
            write_csv(folder / f"{pre}_interaction_{s}.csv", cols)
            saved.append(f"{pre}_interaction_{s}.csv")
    print(f"  saved: {', '.join(saved)}")
    if kinds:
        plot_scan(a, folder / f"{pre}_analysis.png", meta, res, kinds, stat, n)
    if stat:
        plot_interaction(a, folder / f"{pre}_interaction.png", meta, res, wl, delta, stat, n)


def report_interaction(res, cal_nm, n):
    """Is 'both together' different from the sum of the two alone?"""
    print("\n  both wavelengths together vs the sum of each alone (light in the two peaks):")
    if n < 2:
        print("    1 frame: no errors, no significance")
    for s, name in (("out", "sample OUT (control)"), ("in", "sample IN")):
        v, z = res[f"excess_{s}_pct"], res[f"excess_{s}_z"]
        ok = np.isfinite(z)
        chi = f", chi2 per step {np.mean(z[ok] ** 2):.2f}" if ok.any() else ""
        print(f"    {name:21s}: excess light median {np.nanmedian(v):+.3f} %, "
              f"range {np.nanmin(v):+.3f} to {np.nanmax(v):+.3f} %{chi}")
    A, z = res["extra_absorbance"], res["extra_absorbance_z"]
    ok = np.isfinite(z)
    if not ok.any():
        return
    big = ok & (np.abs(z) > 3)
    print(f"    extra absorbance (IN corrected by the control): {np.nanmin(A):+.2e} to "
          f"{np.nanmax(A):+.2e}; chi2 per step {np.mean(z[ok] ** 2):.2f} (1 = noise only)")
    if big.any():
        print(f"    -> beyond 3 standard errors at {', '.join(f'{x:.1f}' for x in cal_nm[big])} "
              f"nm (largest {np.max(np.abs(z[ok])):.1f})")
    else:
        print(f"    -> no effect beyond the noise: all steps within 3 standard errors "
              f"(largest {np.max(np.abs(z[ok])):.1f})")
    print("       standard errors of the frames; drift between the three patterns of a "
          "step adds to them -- a control chi2 well above 1 means it matters")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

LABEL_BOX = dict(boxstyle="round,pad=0.3", fc="none", ec="0.6")     # see-through


def legend_with_error(ax, e):
    """Legend titled with the typical error bar (often smaller than the markers)."""
    e = np.asarray(e, float)
    title = (f"error bars: median \u00b1 {np.nanmedian(e):.2g}" if np.isfinite(e).any()
             else None)
    ax.legend(title=title, title_fontsize=12)


def plot_single(a, path, meta, wl, px, per_px, win, peaks, kinds, n):
    plt = get_plt(a.show_plots)
    if plt is None:
        return
    suf = a.error
    lo, hi = meta["view_nm"]
    m = (wl >= lo) & (wl <= hi)
    fig, ax = plt.subplots(1 + len(kinds), 1, figsize=(11, 4 * (1 + len(kinds))),
                           sharex=True, squeeze=False)
    ax = ax[:, 0]
    for s, label in (("out", "sample out"), ("in", "sample in")):
        y, e = px[s]
        line, = ax[0].plot(wl[m], y[m], label=label)
        ax[0].fill_between(wl[m], (y - e)[m], (y + e)[m], color=line.get_color(), alpha=0.3)
    for w in win:
        for x in ax:
            x.axvspan(w[0], w[1], color="0.88", zorder=0)
    ax[0].set(ylabel="counts - background",
              title=f"{meta['stamp']}: {', '.join(f'{c:.1f}' for c in meta['cal_nm'])} nm, "
                    f"{meta['exposure_ms']:g} ms x {n} frames\n{error_note(a, n)}")
    ax[0].legend()
    for x, k in zip(ax[1:], kinds):
        y, e = per_px[k]
        line, = x.plot(wl[m], y[m], lw=1, label="per pixel")
        x.fill_between(wl[m], (y - e)[m], (y + e)[m], color=line.get_color(), alpha=0.25)
        v, ev = np.asarray(peaks[k], float), np.asarray(peaks[f"{k}_{suf}"], float)
        if k == "subtract":          # integrated counts are not on the per-pixel scale
            for c, vi, ei in zip(peaks["centre_nm"], v, ev):
                x.annotate(f"integrated\n{pm(vi, ei, plot=True)}",
                           (c, 0.5 * np.interp(c, wl, y)), textcoords="offset points",
                           xytext=(12, 0), va="center", fontsize=13, bbox=LABEL_BOX)
        else:
            x.errorbar(peaks["centre_nm"], v, yerr=ev, fmt="o", color="C3", ms=7,
                       capsize=5, elinewidth=1.5, label="integrated peak")
            for c, vi, ei in zip(peaks["centre_nm"], v, ev):
                if np.isfinite(vi):
                    x.annotate(pm(vi, ei, plot=True), (c, vi), textcoords="offset points",
                               xytext=(12, 10), fontsize=13, bbox=LABEL_BOX)
        x.set(ylabel=LABELS[k])
        x.legend(loc="best")
    ax[-1].set(xlabel="wavelength (nm)")
    fig.tight_layout()
    finish_figure(plt, fig, path, a.show_plots)


def plot_scan(a, path, meta, res, kinds, stat, n):
    plt = get_plt(a.show_plots)
    if plt is None:
        return
    suf = a.error
    x = res["cal_nm"]
    fig, ax = plt.subplots(1 + len(kinds), 1, figsize=(11, 4 * (1 + len(kinds))),
                           sharex=True, squeeze=False)
    ax = ax[:, 0]
    bars = dict(ms=4, capsize=3, elinewidth=1)
    alone = " (alone)" if stat else ""
    ax[0].errorbar(x, res["out_counts"], yerr=res[f"out_counts_{suf}"], fmt="o-",
                   label=f"scanning peak{alone}, sample out", **bars)
    ax[0].errorbar(x, res["in_counts"], yerr=res[f"in_counts_{suf}"], fmt="o-",
                   label=f"scanning peak{alone}, sample in", **bars)
    if stat:
        ax[0].errorbar(x, res["stat_out_counts"], yerr=res[f"stat_out_counts_{suf}"],
                       fmt="s--", label="stationary peak (alone), sample out", **bars)
        ax[0].errorbar(x, res["stat_in_counts"], yerr=res[f"stat_in_counts_{suf}"],
                       fmt="s--", label="stationary peak (alone), sample in", **bars)
        for xx in ax:
            xx.axvline(stat["cal_nm"], color="0.6", ls=":", lw=1)
    ax[0].set(ylabel="integrated counts",
              title=f"{meta['stamp']}: scan, {meta['exposure_ms']:g} ms x {n} frames"
                    + (f", stationary peak {stat['cal_nm']:.1f} nm" if stat else "")
                    + f"\n{error_note(a, n)}")
    legend_with_error(ax[0], res[f"out_counts_{suf}"])
    for xx, k in zip(ax[1:], kinds):
        xx.errorbar(x, res[k], yerr=res[f"{k}_{suf}"], fmt="o-",
                    label=f"scanning peak{alone}", **bars)
        if stat:
            xx.errorbar(x, res[f"stat_{k}"], yerr=res[f"stat_{k}_{suf}"], fmt="s--",
                        label=f"stationary peak alone ({stat['cal_nm']:.1f} nm)", **bars)
        legend_with_error(xx, res[f"{k}_{suf}"])
        xx.set(ylabel=LABELS[k])
    ax[-1].set(xlabel="scanning peak wavelength (nm)")
    fig.tight_layout()
    finish_figure(plt, fig, path, a.show_plots)


def cell_edges(c):
    """Cell boundaries around increasing or decreasing centres, for pcolormesh."""
    c = np.asarray(c, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def plot_interaction(a, path, meta, res, wl, delta, stat, n):
    """Both wavelengths together vs the sum of each alone."""
    plt = get_plt(a.show_plots)
    if plt is None:
        return
    suf = a.error
    x = res["cal_nm"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 13), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1.3]})
    bars = dict(ms=4, capsize=3, elinewidth=1)
    for s, label in (("out", "sample out (control)"), ("in", "sample in")):
        ax[0].errorbar(x, res[f"excess_{s}_pct"], yerr=res[f"excess_{s}_pct_{suf}"],
                       fmt="o-", label=label, **bars)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(ylabel="excess light (%)",
              title=f"{meta['stamp']}: both together vs each alone, stationary peak "
                    f"{stat['cal_nm']:.1f} nm\n{error_note(a, n)}")
    legend_with_error(ax[0], res[f"excess_in_pct_{suf}"])
    ax[1].errorbar(x, res["extra_absorbance"], yerr=res[f"extra_absorbance_{suf}"],
                   fmt="o-", color="C3", label="sample in, corrected by the control", **bars)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(ylabel="extra absorbance\nwhen both are on")
    legend_with_error(ax[1], res[f"extra_absorbance_{suf}"])

    lo, hi = meta["view_nm"]
    m = (wl >= lo) & (wl <= hi)
    d = delta["in"][:, m]
    lim = np.nanpercentile(np.abs(d), 99.5) or 1.0
    im = ax[2].pcolormesh(cell_edges(x), cell_edges(wl[m]), d.T, cmap="RdBu_r",
                          vmin=-lim, vmax=lim, shading="flat")
    ax[2].plot(x, x, color="0.4", lw=0.8, ls="--")
    ax[2].axhline(stat["cal_nm"], color="0.4", lw=0.8, ls="--")
    fig.colorbar(im, ax=ax[2], label="both - scan - stat (counts)", pad=0.01)
    ax[2].set(ylabel="spectrometer wavelength (nm)",
              xlabel="scanning peak wavelength (nm)",
              title="per pixel, sample in (0 = no effect)")
    for xx in ax:
        xx.axvline(stat["cal_nm"], color="0.6", ls=":", lw=1)
    fig.tight_layout()
    finish_figure(plt, fig, path, a.show_plots)


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #


def lines_overlap(offsets, width):
    for i in range(len(offsets)):
        for j in range(i + 1, len(offsets)):
            if abs(offsets[i] - offsets[j]) < width:
                return i, j
    return None


def view_mask(r, cal):
    return (r.wl >= cal.lo - 20) & (r.wl <= cal.hi + 20)


def ask_exposure_frames(r, a, state):
    """Exposure and number of frames; Enter keeps the last ones."""
    ms = ask_exposure(state["exposure_ms"])
    n = ask_int("  frames (separate spectra: mean and standard deviation)",
                state["frames"], 1, 100000)
    if n < 2:
        print("  1 frame: no error bars")
    state["exposure_ms"], state["frames"] = ms, n
    r.set_exposure(ms)
    return n


def measure_lines(r, a, offsets, n):
    """n frames with the lines, then the background frames right after. The
    lines are already on the DMD (shown when the wavelengths were chosen) and
    go straight back on after a DMD background, so they stay on while you
    move the sample."""
    r.show_lines(offsets, a.line_width_rows)        # no-op: they are on already
    t = r.now()
    frames = r.frames(n)
    t_bg = r.now()
    bg = r.background_stack()
    r.show_lines(offsets, a.line_width_rows)        # back on (no-op after a shutter dark)
    return {"frames": frames, "bg": bg, "t": t, "t_bg": t_bg}


def common_meta(r, a, cal, kind, st, n):
    return {"kind": kind, "stamp": st, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "calibration_file": str(cal.path), "line_width_rows": a.line_width_rows,
            "exposure_ms": r.exposure_ms, "hw_average": r.hw_average, "frames": n,
            "background": a.background, "background_frames": a.background_frames,
            "view_nm": [cal.lo - 20, cal.hi + 20], "settings": vars(a),
            "hardware": r.describe(), "notes": a.notes}


def single(r, a, cal, state, root):
    nwl = int(ask_choice("Number of wavelengths (1, 2 or 3): ", ("1", "2", "3")))
    while True:
        targets = [ask_float(f"  wavelength {i + 1} (nm)", None, cal.lo, cal.hi)
                   for i in range(nwl)]
        offsets = [cal.offset_for(w) for w in targets]
        clash = lines_overlap(offsets, a.line_width_rows)
        if clash is None:
            break
        i, j = clash
        print(f"  {targets[i]:g} and {targets[j]:g} nm are only "
              f"{abs(offsets[i] - offsets[j])} rows apart on the DMD -- the "
              f"{a.line_width_rows}-row lines would overlap. Choose them further apart.")
    band = [a.line_width_rows * cal.nm_per_row(w) for w in targets]
    for w, o, b in zip(targets, offsets, band):
        print(f"  {w:g} nm -> line at offset {o:+d} rows ({cal.wl_at(o):.2f} nm, "
              f"band ~{b:.1f} nm)")
    r.show_lines(offsets, a.line_width_rows)
    print("  the lines are on the DMD and stay on while you move the sample")

    st = stamp()
    while True:
        n = ask_exposure_frames(r, a, state)
        data = {}
        for side, prompt in SIDES:
            input(prompt)
            r.set_sample(side == "in")
            data[side] = measure_lines(r, a, offsets, n)
            txt, data[side]["clipped"] = fill_report(data[side]["frames"], a,
                                                     view_mask(r, cal))
            print(f"  sample {side.upper():3s}: {txt}")
        if not (data["in"]["clipped"] or data["out"]["clipped"]):
            break
        if input("  CLIPPED: a clipped peak is not linear. Measure again with a shorter "
                 "exposure? [Y/n]: ").strip().lower() in ("", "y", "yes"):
            continue
        break

    folder = root / "single"
    folder.mkdir(parents=True, exist_ok=True)
    pre = f"{st}_{nwl}wl"
    while (folder / f"{pre}_meta.json").exists():
        pre += "b"
    files = {"frames": f"{pre}_frames.npz"}
    for side in ("in", "out"):
        fr, bg = data[side]["frames"], data[side]["bg"].mean(axis=0)
        files[side] = f"{pre}_sample_{side}.csv"
        write_csv(folder / files[side], {
            "wavelength_nm": r.wl, "net_counts": fr.mean(axis=0) - bg,
            "net_std": fr.std(axis=0, ddof=1) if n > 1 else np.full(bg.shape, np.nan),
            "raw_counts": fr.mean(axis=0), "background_counts": bg})
    np.savez_compressed(folder / files["frames"], wavelength_nm=r.wl,
                        **{f"{s}_{k}": data[s][key].astype(np.float32)
                           for s in ("in", "out")
                           for k, key in (("frames", "frames"), ("background", "bg"))})
    meta = common_meta(r, a, cal, "dmd_single", st, n)
    meta.update(targets_nm=targets, offsets_rows=offsets,
                cal_nm=[cal.wl_at(o) for o in offsets], line_band_nm=band, files=files,
                times_s={s: [data[s]["t"], data[s]["t_bg"]] for s in data},
                clipped={s: data[s]["clipped"] for s in data})
    meta_path = folder / f"{pre}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved: {folder / pre}_*")

    kinds = ask_analyses()
    analyse_single(meta_path, a, kinds)
    if not kinds:
        print(f"  no analysis -- later: python dmd_measure.py --analyze \"{meta_path}\"")


def scan_targets(start, stop, step):
    n = int(np.floor(abs(stop - start) / step + 1e-9))
    return start + (1 if stop >= start else -1) * step * np.arange(n + 1)


def run_scan(r, a, cal, offsets, stat_off, side, n):
    """One pass: a background, at every step each pattern of scan_patterns (n
    frames each), a background."""
    names = list(scan_patterns(0, stat_off))
    t0 = r.now()
    b0 = r.background_stack()
    frames = {p: [] for p in names}
    t = {p: np.empty(len(offsets)) for p in names}
    for k, off in enumerate(offsets):
        for p, lines in scan_patterns(off, stat_off).items():
            r.show_lines(lines, a.line_width_rows)
            t[p][k] = r.now()
            frames[p].append(r.frames(n))
        done = int(30 * (k + 1) / len(offsets))
        print(f"\r  sample {side.upper():3s} [{'#' * done}{'.' * (30 - done)}] "
              f"{k + 1}/{len(offsets)}  {cal.wl_at(off):6.1f} nm", end="", flush=True)
    print()
    t1 = r.now()
    b1 = r.background_stack()
    view = view_mask(r, cal)
    out = {"b0": b0, "b1": b1, "t_bg": np.array([t0, t1]), "p": {},
           "clipped": np.zeros(len(offsets), bool), "raw_max": 0.0}
    for p in names:
        fr = np.array(frames[p])                                    # (steps, n, pixels)
        net = fr - interp_background(t[p], t0, t1, b0.mean(axis=0), b1.mean(axis=0))[:, None]
        out["p"][p] = {"frames": fr, "t": t[p], "net_mean": net.mean(axis=1),
                       "net_std": (net.std(axis=1, ddof=1) if n > 1
                                   else np.full(net.shape[::2], np.nan))}
        out["clipped"] |= np.array([clipped(f[:, view], a.full_scale_counts).any()
                                    for f in fr])
        out["raw_max"] = max(out["raw_max"], float(fr[..., view].max()))
    return out


def scan(r, a, cal, state, root):
    kind = ask_choice("Scan: 1 = one peak, 2 = one scanning peak + one stationary peak: ",
                      ("1", "2"))
    start = ask_float("  start (nm)", a.scan_start_nm if a.scan_start_nm is not None
                      else round(cal.lo + 0.05, 1), cal.lo, cal.hi)
    stop = ask_float("  stop (nm)", a.scan_stop_nm if a.scan_stop_nm is not None
                     else round(cal.hi - 0.05, 1), cal.lo, cal.hi)
    step = ask_float("  step (nm)", a.scan_step_nm, 0.001)
    offsets = list(dict.fromkeys(cal.offset_for(w) for w in scan_targets(start, stop, step)))

    stat = None
    if kind == "2":
        ws = ask_float("  stationary peak (nm)", None, cal.lo, cal.hi)
        so = cal.offset_for(ws)
        stat = {"target_nm": ws, "offset_rows": so, "cal_nm": cal.wl_at(so),
                "line_band_nm": a.line_width_rows * cal.nm_per_row(ws)}
        keep = [o for o in offsets if abs(o - so) >= a.line_width_rows]
        if len(keep) < len(offsets):
            print(f"  {len(offsets) - len(keep)} steps skipped where the scanning line "
                  f"would overlap the stationary one on the DMD")
        offsets = keep
        print(f"  stationary: {ws:g} nm -> offset {so:+d} rows ({stat['cal_nm']:.2f} nm)")
        print("  at every step: scanning line alone, both lines, stationary line alone")
    if not offsets:
        print("  nothing to scan")
        return
    stat_off = stat and stat["offset_rows"]
    names = list(scan_patterns(0, stat_off))
    cal_nm = [cal.wl_at(o) for o in offsets]
    print(f"  {len(offsets)} steps, {cal_nm[0]:.1f} to {cal_nm[-1]:.1f} nm")

    n = ask_exposure_frames(r, a, state)
    est = len(offsets) * len(names) * (a.dmd_settle_s + 0.3
                                       + n * r.exposure_ms * r.hw_average / 1e3)
    print(f"  about {est / 60:.1f} min per pass")

    st = stamp()
    data = {}
    for side, prompt in SIDES:
        input(prompt)
        r.set_sample(side == "in")
        data[side] = run_scan(r, a, cal, offsets, stat_off, side, n)
        nclip = int(data[side]["clipped"].sum())
        print(f"  brightest pixel {100 * data[side]['raw_max'] / a.full_scale_counts:.0f}% "
              f"of full scale"
              + (f" -- WARNING: {nclip} steps CLIPPED (lower the exposure)" if nclip else ""))

    folder = new_folder(root / "scan", f"{st}_{'1peak' if kind == '1' else '2peak'}")
    step_names = [f"step{k:03d}_{c:.2f}nm" for k, c in enumerate(cal_nm)]
    files = {"steps": f"{st}_steps.csv", "background": f"{st}_background.csv",
             "frames": f"{st}_frames.npz"}
    for side in ("in", "out"):
        for p in names:
            sfx = "" if len(names) == 1 else f"_{p}"
            for key, what, end in ((f"{side}{sfx}", "net_mean", ""),
                                   (f"{side}{sfx}_std", "net_std", "_std")):
                files[key] = f"{st}_sample_{side}{sfx}{end}.csv"
                cols = {"wavelength_nm": r.wl}
                cols.update(zip(step_names, data[side]["p"][p][what]))
                write_csv(folder / files[key], cols)
    steps_cols = {"step": np.arange(len(offsets)), "offset_rows": offsets, "cal_nm": cal_nm,
                  "line_band_nm": [a.line_width_rows * cal.nm_per_row(c) for c in cal_nm]}
    for side in ("in", "out"):
        for p in names:
            steps_cols[f"t_{side}_{p}_s"] = data[side]["p"][p]["t"]
        steps_cols[f"clipped_{side}"] = data[side]["clipped"].astype(int)
    write_csv(folder / files["steps"], steps_cols)
    write_csv(folder / files["background"], {
        "wavelength_nm": r.wl, "in_before": data["in"]["b0"].mean(axis=0),
        "in_after": data["in"]["b1"].mean(axis=0),
        "out_before": data["out"]["b0"].mean(axis=0),
        "out_after": data["out"]["b1"].mean(axis=0)})
    arrays = {"wavelength_nm": r.wl}
    for s in ("in", "out"):
        arrays.update({f"{s}_bg_before": data[s]["b0"].astype(np.float32),
                       f"{s}_bg_after": data[s]["b1"].astype(np.float32),
                       f"t_bg_{s}": data[s]["t_bg"]})
        for p in names:
            arrays[f"{s}_{p}_frames"] = data[s]["p"][p]["frames"].astype(np.float32)
            arrays[f"t_{s}_{p}"] = data[s]["p"][p]["t"]
    np.savez_compressed(folder / files["frames"], **arrays)
    meta = common_meta(r, a, cal, "dmd_scan", st, n)
    meta.update(scan_type="one peak" if kind == "1" else "scanning + stationary peak",
                start_nm=start, stop_nm=stop, step_nm=step, stationary=stat,
                patterns=names, files=files)
    meta_path = folder / f"{st}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved: {folder}")

    kinds = ask_analyses()
    analyse_scan(meta_path, a, kinds)
    if not kinds:
        print(f"  no analysis -- later: python dmd_measure.py --analyze \"{folder}\"")


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    hardware_args(ap)
    g = ap.add_argument_group("measurement")
    g.add_argument("--calibration-file", default=None, dest="calibration_file",
                   help="dmd_calibration.csv (or its folder); null = the newest in "
                        "<data_dir>/calibrations/")
    g.add_argument("--line-width-rows", type=int, default=20, dest="line_width_rows")
    g.add_argument("--exposure-ms", type=float_or_none, default=None, dest="exposure_ms",
                   help="offered at the first exposure prompt (then the last one used)")
    g.add_argument("--frames", type=int, default=10,
                   help="offered at the first frames prompt (then the last one used)")
    g.add_argument("--error", choices=("std", "sem"), default="std",
                   help="error bars: std = standard deviation of the frames; sem = "
                        "standard error of the mean, with the background's error")
    g.add_argument("--scan-start-nm", type=float_or_none, default=None, dest="scan_start_nm",
                   help="offered at the scan prompt; null = the calibrated range")
    g.add_argument("--scan-stop-nm", type=float_or_none, default=None, dest="scan_stop_nm")
    g.add_argument("--scan-step-nm", type=float, default=5.0, dest="scan_step_nm")
    g.add_argument("--band-halfwidth-nm", type=float_or_none, default=None,
                   dest="band_halfwidth_nm",
                   help="peaks are integrated over centre +- this; null = +- the FWHM")
    g.add_argument("--search-nm", type=float, default=5.0, dest="search_nm",
                   help="a peak is looked for this far from its calibrated wavelength")
    g.add_argument("--min-signal-counts", type=float, default=100.0, dest="min_signal_counts",
                   help="in/out and absorbance only where sample-out has this much")
    g.add_argument("--show-plots", type=str2bool, default=True, dest="show_plots")
    g.add_argument("--notes", default="")
    g.add_argument("--analyze", metavar="FILE", default=None,
                   help="analyse saved data again: a *_meta.json, or a scan folder")
    a = parse_config(ap, argv, HERE / "dmd_measure.json", HERE / "dmd_calibration.json")

    if a.analyze:
        path, meta = load_meta(Path(a.analyze))
        kinds = ask_analyses()
        (analyse_single if meta["kind"] == "dmd_single" else analyse_scan)(path, a, kinds)
        return 0

    cal = Calibration(find_calibration(a, a.calibration_file))
    cal.check(a)
    root = data_root(a)
    print(f"\ncalibration: {cal.describe()}")
    print(f"data       : {root}")
    state = {"exposure_ms": a.exposure_ms, "frames": a.frames}
    with make_rig(a) as r:
        r.set_exposure(a.exposure_ms or 10.0)
        while True:
            try:
                choice = ask_choice("\nMode: 1 = single measurement, 2 = scan, q = quit: ",
                                    ("1", "2", "q"))
                if choice == "q":
                    break
                (single if choice == "1" else scan)(r, a, cal, state, root)
            except KeyboardInterrupt:
                print("\n  stopped -- back to the menu (q quits)")
            except EOFError:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
