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
  1  single measurement: 1, 2 or 3 wavelengths at once and an exposure, then
     sample IN -> spectrum, sample OUT -> spectrum.
     Saved in <data_dir>/single/ as <date>_<time>_<n>wl_*.
  2  scan, saved in <data_dir>/scan/<date>_<time>_<type>/:
     1  one peak stepped from start to stop: the whole scan with the sample
        IN, then the whole scan with it OUT;
     2  the same, plus a stationary peak at a wavelength you choose, shown
        together with the scanning one at every step.

Every spectrum has a background subtracted (`background`): the DMD blocking
everything, or the spectrometer shutter. Single: one right after each
spectrum. Scan: one before and one after each pass, interpolated in time.

After a measurement you choose the analysis, one or more:
  1  subtract    in - out
  2  divide      in / out                (the transmission)
  3  absorbance  -log10(in / out)        (positive when the sample absorbs)
Single: on the whole spectrum, and on the counts integrated over each peak.
Scan: on the counts integrated over the peak, at every step.
A peak is integrated over its centre +- band_halfwidth_nm (null: +- its
FWHM), the centre and FWHM taken from the sample-OUT spectrum.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from dmd_common import (HERE, Calibration, ask_choice, ask_exposure, ask_float,
                        clipped, data_root, fill_report, find_calibration, find_peak,
                        finish_figure, float_or_none, get_plt, hardware_args,
                        interp_background, make_rig, new_folder, parse_config,
                        read_csv, stamp, str2bool, write_csv)

SIDES = (("in", "Put the sample IN, then press Enter to measure..."),
         ("out", "Take the sample OUT, then press Enter to measure..."))
ANALYSES = {"1": "subtract", "2": "divide", "3": "absorbance"}
LABELS = {"subtract": "in - out (counts)", "divide": "in / out",
          "absorbance": "-log10(in / out)"}


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


def analyse(kind, i_in, i_out, min_out):
    """in - out, in / out or -log10(in / out); the ratios are NaN where the
    sample-out signal is below min_out (no light there to divide by)."""
    i_in, i_out = np.asarray(i_in, float), np.asarray(i_out, float)
    if kind == "subtract":
        return i_in - i_out
    with np.errstate(all="ignore"):
        ratio = np.where(i_out >= min_out, i_in / i_out, np.nan)
        if kind == "divide":
            return ratio
        return np.where(ratio > 0, -np.log10(ratio), np.nan)


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


def band_sum(wl, y, lo, hi):
    return np.asarray(y)[..., (wl >= lo) & (wl <= hi)].sum(axis=-1)


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
    return path, json.loads(path.read_text())


def prefix_of(meta_path: Path) -> str:
    return meta_path.name[:-len("_meta.json")]


def analyse_single(meta_path: Path, a, kinds):
    meta_path, meta = load_meta(meta_path)
    folder, pre = meta_path.parent, prefix_of(meta_path)
    din = read_csv(folder / meta["files"]["in"])
    dout = read_csv(folder / meta["files"]["out"])
    wl, y_in, y_out = din["wavelength_nm"], din["net_counts"], dout["net_counts"]
    cal_nm, nominal = meta["cal_nm"], meta["line_band_nm"]

    win = [band_window(wl, y_out, c, nominal[i], a, search_radius(a, c, cal_nm))
           for i, c in enumerate(cal_nm)]
    overlap = [any(j != i and w[0] < v[1] and v[0] < w[1] for j, v in enumerate(win))
               for i, w in enumerate(win)]
    i_in = np.array([band_sum(wl, y_in, w[0], w[1]) for w in win])
    i_out = np.array([band_sum(wl, y_out, w[0], w[1]) for w in win])
    peaks = {"target_nm": meta["targets_nm"], "cal_nm": cal_nm,
             "centre_nm": [w[2] for w in win], "fwhm_nm": [w[3] for w in win],
             "window_lo_nm": [w[0] for w in win], "window_hi_nm": [w[1] for w in win],
             "in_counts": i_in, "out_counts": i_out}
    for k in kinds:
        peaks[k] = analyse(k, i_in, i_out, a.min_signal_counts)
    peaks["overlap"] = np.array(overlap, int)

    print("\n  peak (nm)  centre   FWHM      in counts     out counts"
          + "".join(f"  {k:>10s}" for k in kinds))
    for i in range(len(cal_nm)):
        print(f"  {cal_nm[i]:9.2f} {win[i][2]:7.2f} {win[i][3]:6.2f} {i_in[i]:14.0f} "
              f"{i_out[i]:14.0f}" + "".join(f"  {peaks[k][i]:10.4g}" for k in kinds)
              + ("   windows overlap" if overlap[i] else ""))
    if any(overlap):
        print("  WARNING: the integration windows of some peaks overlap -- choose "
              "wavelengths further apart or set band_halfwidth_nm")

    write_csv(folder / f"{pre}_peaks.csv", peaks)
    for k in kinds:
        write_csv(folder / f"{pre}_{k}.csv",
                  {"wavelength_nm": wl, k: analyse(k, y_in, y_out, a.min_signal_counts)})
    print(f"  saved: {pre}_peaks.csv" + "".join(f", {pre}_{k}.csv" for k in kinds))
    plot_single(a, folder / f"{pre}_analysis.png", meta, wl, y_in, y_out, win, peaks, kinds)


def analyse_scan(meta_path: Path, a, kinds):
    meta_path, meta = load_meta(meta_path)
    folder, pre = meta_path.parent, prefix_of(meta_path)
    din = read_csv(folder / meta["files"]["in"])
    dout = read_csv(folder / meta["files"]["out"])
    steps = read_csv(folder / meta["files"]["steps"])
    wl = din["wavelength_nm"]
    cols = [c for c in din if c != "wavelength_nm"]
    Y_in = np.array([din[c] for c in cols])
    Y_out = np.array([dout[c] for c in cols])
    cal_nm, nominal = steps["cal_nm"], steps["line_band_nm"]
    stat = meta.get("stationary")
    n = len(cal_nm)

    res = {"step": np.arange(n), "cal_nm": cal_nm}
    scan_win, stat_win = [], []
    for k in range(n):
        others = [stat["cal_nm"]] if stat else []
        scan_win.append(band_window(wl, Y_out[k], cal_nm[k], nominal[k], a,
                                    search_radius(a, cal_nm[k], others)))
        if stat:
            stat_win.append(band_window(wl, Y_out[k], stat["cal_nm"], stat["line_band_nm"],
                                        a, search_radius(a, stat["cal_nm"], [cal_nm[k]])))
    res["centre_nm"] = np.array([w[2] for w in scan_win])
    res["fwhm_nm"] = np.array([w[3] for w in scan_win])
    res["in_counts"] = np.array([band_sum(wl, Y_in[k], *scan_win[k][:2]) for k in range(n)])
    res["out_counts"] = np.array([band_sum(wl, Y_out[k], *scan_win[k][:2]) for k in range(n)])
    overlap = np.zeros(n, bool)
    if stat:
        overlap = np.array([w[0] < v[1] and v[0] < w[1] for w, v in zip(scan_win, stat_win)])
        res["stat_centre_nm"] = np.array([w[2] for w in stat_win])
        res["stat_in_counts"] = np.array([band_sum(wl, Y_in[k], *stat_win[k][:2])
                                          for k in range(n)])
        res["stat_out_counts"] = np.array([band_sum(wl, Y_out[k], *stat_win[k][:2])
                                           for k in range(n)])
    for kind in kinds:
        v = analyse(kind, res["in_counts"], res["out_counts"], a.min_signal_counts)
        res[kind] = np.where(overlap, np.nan, v)
        if stat:
            v = analyse(kind, res["stat_in_counts"], res["stat_out_counts"],
                        a.min_signal_counts)
            res[f"stat_{kind}"] = np.where(overlap, np.nan, v)
    if stat:
        res["overlap"] = overlap.astype(int)

    shift = res["centre_nm"] - cal_nm
    good = np.isfinite(res["fwhm_nm"])
    print(f"\n  {n} steps, {cal_nm.min():.1f}-{cal_nm.max():.1f} nm")
    if good.any():
        print(f"  peak centres vs calibration: mean {np.mean(shift[good]):+.2f} nm, "
              f"worst {shift[good][np.argmax(np.abs(shift[good]))]:+.2f} nm")
    if stat:
        r = analyse("divide", res["stat_in_counts"], res["stat_out_counts"],
                    a.min_signal_counts)[~overlap]
        if np.isfinite(r).any():
            print(f"  stationary peak ({stat['cal_nm']:.1f} nm): in/out {np.nanmean(r):.4g}, "
                  f"spread {100 * np.nanstd(r) / np.nanmean(r):.2f}% over the scan")
        if overlap.any():
            print(f"  {overlap.sum()} steps where the two peaks' windows overlap: left out "
                  f"of the analysis (NaN)")
    for kind in kinds:
        v = res[kind]
        if np.isfinite(v).any():
            print(f"  {kind:10s}: {np.nanmin(v):.4g} to {np.nanmax(v):.4g}")

    write_csv(folder / f"{pre}_results.csv", res)
    print(f"  saved: {pre}_results.csv")
    if kinds:
        plot_scan(a, folder / f"{pre}_analysis.png", meta, res, kinds, stat)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def plot_single(a, path, meta, wl, y_in, y_out, win, peaks, kinds):
    plt = get_plt(a.show_plots)
    if plt is None:
        return
    lo, hi = meta["view_nm"]
    m = (wl >= lo) & (wl <= hi)
    fig, ax = plt.subplots(1 + len(kinds), 1, figsize=(10, 3.2 * (1 + len(kinds))),
                           sharex=True, squeeze=False)
    ax = ax[:, 0]
    ax[0].plot(wl[m], y_out[m], label="sample out")
    ax[0].plot(wl[m], y_in[m], label="sample in")
    for w in win:
        for x in ax:
            x.axvspan(w[0], w[1], color="0.85", zorder=0)
    ax[0].set(ylabel="counts - background",
              title=f"{meta['stamp']}: {', '.join(f'{c:.1f}' for c in meta['cal_nm'])} nm, "
                    f"{meta['exposure_ms']:g} ms")
    ax[0].legend()
    for x, k in zip(ax[1:], kinds):
        per_px = analyse(k, y_in, y_out, a.min_signal_counts)
        x.plot(wl[m], per_px[m], lw=0.8, label="per pixel")
        if k == "subtract":          # integrated counts are not on the per-pixel scale
            for c, v in zip(peaks["centre_nm"], peaks[k]):
                y = np.interp(c, wl, per_px)
                x.annotate(f"integrated\n{v:.4g}", (c, y), textcoords="offset points",
                           xytext=(8, 0), va="center", fontsize=8)
        else:
            x.plot(peaks["centre_nm"], peaks[k], "o", color="C3", label="integrated peak")
            for c, v in zip(peaks["centre_nm"], peaks[k]):
                if np.isfinite(v):
                    x.annotate(f"{v:.4g}", (c, v), textcoords="offset points",
                               xytext=(6, 6))
        x.set(ylabel=LABELS[k])
        x.legend(loc="best")
    ax[-1].set(xlabel="wavelength (nm)")
    fig.tight_layout()
    finish_figure(plt, fig, path, a.show_plots)


def plot_scan(a, path, meta, res, kinds, stat):
    plt = get_plt(a.show_plots)
    if plt is None:
        return
    x = res["cal_nm"]
    hide = res.get("overlap", np.zeros(len(x))).astype(bool)   # the two peaks merged
    fig, ax = plt.subplots(1 + len(kinds), 1, figsize=(10, 3.2 * (1 + len(kinds))),
                           sharex=True, squeeze=False)
    ax = ax[:, 0]
    show = lambda key: np.where(hide, np.nan, res[key])           # noqa: E731
    ax[0].plot(x, show("out_counts"), "o-", ms=3, label="scanning peak, sample out")
    ax[0].plot(x, show("in_counts"), "o-", ms=3, label="scanning peak, sample in")
    if stat:
        ax[0].plot(x, show("stat_out_counts"), "--", label="stationary peak, sample out")
        ax[0].plot(x, show("stat_in_counts"), "--", label="stationary peak, sample in")
        for xx in ax:
            xx.axvline(stat["cal_nm"], color="0.6", ls=":", lw=1)
    ax[0].set(ylabel="integrated counts",
              title=f"{meta['stamp']}: scan, {meta['exposure_ms']:g} ms"
                    + (f", stationary peak {stat['cal_nm']:.1f} nm" if stat else ""))
    ax[0].legend(fontsize=8)
    for xx, k in zip(ax[1:], kinds):
        xx.plot(x, res[k], "o-", ms=3, label="scanning peak")
        if stat:
            xx.plot(x, res[f"stat_{k}"], "s--", ms=3,
                    label=f"stationary peak ({stat['cal_nm']:.1f} nm)")
            xx.legend(fontsize=8)
        xx.set(ylabel=LABELS[k])
    ax[-1].set(xlabel="scanning peak wavelength (nm)")
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


def measure_lines(r, a, offsets):
    """The lines, then the background right after."""
    r.show_lines(offsets, a.line_width_rows)
    t = r.now()
    raw = r.snap()
    t_bg = r.now()
    bg = r.background()
    return {"raw": raw, "background": bg, "t": t, "t_background": t_bg}


def common_meta(r, a, cal, kind, st, exposure_ms):
    return {"kind": kind, "stamp": st, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "calibration_file": str(cal.path), "line_width_rows": a.line_width_rows,
            "exposure_ms": exposure_ms, "hw_average": a.hw_average,
            "background": a.background, "view_nm": [cal.lo - 20, cal.hi + 20],
            "settings": vars(a), "hardware": r.describe(), "notes": a.notes}


def single(r, a, cal, state, root):
    n = int(ask_choice("Number of wavelengths (1, 2 or 3): ", ("1", "2", "3")))
    while True:
        targets = [ask_float(f"  wavelength {i + 1} (nm)", None, cal.lo, cal.hi)
                   for i in range(n)]
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

    st = stamp()
    while True:
        ms = ask_exposure(state["exposure_ms"])
        state["exposure_ms"] = ms
        r.set_exposure(ms)
        data = {}
        for side, prompt in SIDES:
            input(prompt)
            r.set_sample(side == "in")
            data[side] = measure_lines(r, a, offsets)
            txt, data[side]["clipped"] = fill_report(data[side]["raw"], a, view_mask(r, cal))
            print(f"  sample {side.upper():3s}: {txt}")
        if not (data["in"]["clipped"] or data["out"]["clipped"]):
            break
        if input("  CLIPPED: a clipped peak is not linear. Measure again with a shorter "
                 "exposure? [Y/n]: ").strip().lower() in ("", "y", "yes"):
            continue
        break

    folder = root / "single"
    folder.mkdir(parents=True, exist_ok=True)
    pre = f"{st}_{n}wl"
    while (folder / f"{pre}_meta.json").exists():
        pre += "b"
    files = {}
    for side in ("in", "out"):
        d = data[side]
        files[side] = f"{pre}_sample_{side}.csv"
        write_csv(folder / files[side], {
            "wavelength_nm": r.wl, "net_counts": d["raw"] - d["background"],
            "raw_counts": d["raw"], "background_counts": d["background"]})
    meta = common_meta(r, a, cal, "dmd_single", st, r.exposure_ms)
    meta.update(targets_nm=targets, offsets_rows=offsets,
                cal_nm=[cal.wl_at(o) for o in offsets], line_band_nm=band, files=files,
                times_s={s: [data[s]["t"], data[s]["t_background"]] for s in data},
                clipped={s: data[s]["clipped"] for s in data})
    meta_path = folder / f"{pre}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved: {folder / pre}_*")

    kinds = ask_analyses()
    if kinds:
        analyse_single(meta_path, a, kinds)
    else:
        print(f"  no analysis -- later: python dmd_measure.py --analyze \"{meta_path}\"")


def scan_targets(start, stop, step):
    n = int(np.floor(abs(stop - start) / step + 1e-9))
    return start + (1 if stop >= start else -1) * step * np.arange(n + 1)


def run_scan(r, a, cal, offsets, stat_off, side):
    t0 = r.now()
    bg0 = r.background()
    raw, t = [], np.empty(len(offsets))
    for k, off in enumerate(offsets):
        r.show_lines([off] if stat_off is None else [off, stat_off], a.line_width_rows)
        t[k] = r.now()
        raw.append(r.snap())
        done = int(30 * (k + 1) / len(offsets))
        print(f"\r  sample {side.upper():3s} [{'#' * done}{'.' * (30 - done)}] "
              f"{k + 1}/{len(offsets)}  {cal.wl_at(off):6.1f} nm", end="", flush=True)
    print()
    t1 = r.now()
    bg1 = r.background()
    raw = np.array(raw)
    net = raw - interp_background(t, t0, t1, bg0, bg1)
    clip = clipped(raw[:, view_mask(r, cal)], a.full_scale_counts)
    return {"net": net, "raw_max": raw.max(axis=1), "t": t, "bg": (bg0, bg1),
            "t_bg": [t0, t1], "clipped": clip}


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
    if not offsets:
        print("  nothing to scan")
        return
    cal_nm = [cal.wl_at(o) for o in offsets]
    print(f"  {len(offsets)} steps, {cal_nm[0]:.1f} to {cal_nm[-1]:.1f} nm")

    ms = ask_exposure(state["exposure_ms"])
    state["exposure_ms"] = ms
    r.set_exposure(ms)
    est = len(offsets) * (a.dmd_settle_s + 0.3 + ms * a.hw_average / 1e3)
    print(f"  about {est / 60:.1f} min per pass")

    st = stamp()
    data = {}
    for side, prompt in SIDES:
        input(prompt)
        r.set_sample(side == "in")
        data[side] = run_scan(r, a, cal, offsets, stat and stat["offset_rows"], side)
        fill = data[side]["raw_max"].max() / a.full_scale_counts
        nclip = int(data[side]["clipped"].sum())
        print(f"  brightest pixel {100 * fill:.0f}% of full scale"
              + (f" -- WARNING: {nclip} steps CLIPPED (lower the exposure)" if nclip else ""))

    folder = new_folder(root / "scan", f"{st}_{'1peak' if kind == '1' else '2peak'}")
    names = [f"step{k:03d}_{c:.2f}nm" for k, c in enumerate(cal_nm)]
    files = {"in": f"{st}_sample_in.csv", "out": f"{st}_sample_out.csv",
             "steps": f"{st}_steps.csv", "background": f"{st}_background.csv"}
    for side in ("in", "out"):
        cols = {"wavelength_nm": r.wl}
        cols.update(zip(names, data[side]["net"]))
        write_csv(folder / files[side], cols)
    write_csv(folder / files["steps"], {
        "step": np.arange(len(offsets)), "offset_rows": offsets, "cal_nm": cal_nm,
        "line_band_nm": [a.line_width_rows * cal.nm_per_row(c) for c in cal_nm],
        "t_in_s": data["in"]["t"], "t_out_s": data["out"]["t"],
        "clipped_in": data["in"]["clipped"].astype(int),
        "clipped_out": data["out"]["clipped"].astype(int)})
    write_csv(folder / files["background"], {
        "wavelength_nm": r.wl, "in_before": data["in"]["bg"][0],
        "in_after": data["in"]["bg"][1], "out_before": data["out"]["bg"][0],
        "out_after": data["out"]["bg"][1]})
    meta = common_meta(r, a, cal, "dmd_scan", st, r.exposure_ms)
    meta.update(scan_type="one peak" if kind == "1" else "scanning + stationary peak",
                start_nm=start, stop_nm=stop, step_nm=step, stationary=stat, files=files,
                background_times_s={s: data[s]["t_bg"] for s in data})
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
    state = {"exposure_ms": a.exposure_ms}
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
