#!/usr/bin/env python3
"""Calibrate the polarimeter at every wavelength: QWP retardance delta(lambda)
and QWP zero, plus beam walk and analyzer leak.

    python QWP_analyzer_characterization_poly.py                  # measure, then analyse
    python QWP_analyzer_characterization_poly.py --simulate       # rehearse, no hardware
    python QWP_analyzer_characterization_poly.py <run_dir> ...    # analyse existing runs

The spectral version of ../mueller_matrix/QWP_analyzer_characterization.py:
the same idea, once per wavelength bin. Hardware and the calibration currently
ASSUMED come from measure_mueller_poly.json; this script's own settings from
QWP_analyzer_characterization_poly.json. Existing runs can be characterization
runs or spectral Mueller runs (their input sweeps, sample out, are used).

The idea
  NO SAMPLE in the beam. Every state from LED -> polarizer -> wave plates is
  taken as fully polarized (DOP = 1), so whatever differs from that is the
  polarimeter. Each state is swept `repeats` times without touching anything.

  H    (both state wave plates OUT: the polarizer alone makes the same linear
       state at EVERY wavelength; turn the polarizer -- or the HWP, which is
       exact only near its design wavelength -- for maximum signal)
       -> QWP RETARDANCE delta(lambda). The depth of its sweep is cos^2(delta/2):
          its DOP moves ~3.4 % per degree of retardance error.
  R/L  -> QWP ZERO: the S3 term must be a pure sin(2 theta); a zero off by eps
          adds cos(2 theta) with c2/c1 = -tan(2 eps), independent of the
          retardance, the LED power and the DOP. R and L need not be circular
          at every wavelength, only |S3| > s3_min there.
  D/A  (optional) cross-check: must give the same retardance as H. How much
          the states disagree is the real accuracy of the calibration.
  V    (optional) analyzer leak: an upper bound on 1 / extinction ratio.

DOP = 1 is an assumption. An unpolarized LED through a polarizer of extinction
ratio ER is polarized to DOP = 1 - 2/ER (0.998 at 1000:1), which makes the
retardance come out ~0.06 deg LOW. That is the one-sided "input DOP" term of
the error budget (input_dop_min). Outside the polarizer's band (LPVISC:
510-800 nm) the extinction drops and this term grows: read the budget there.

Error budget, per wavelength, for the zero and the retardance separately
  statistical      scatter of the repeats of one state
  states           half the spread between states -- the real accuracy test
  cos 2t leftover  a cos(2 theta) term the zero does not explain (seen in
                   states without S3) would shift the zero this much
  zero             the zero's own uncertainty, propagated into the retardance
  input DOP        if the states are at input_dop_min instead of 1 (one-sided)
  analyzer leak    the same for the leak measured with V (one-sided)

The retardance is then smoothed across wavelength (a polynomial in
1/lambda, smooth_order; a wave plate's retardance is a smooth curve) and the
zero is one number (zero_smooth_order 0: zero-order and quartz/MgF2
achromatic plates have an axis that does not move with wavelength).

Output in <run>/characterization/:
  qwp_calibration.csv   wavelength_nm, retardance_waves, retardance_err_waves,
                        zero_deg, zero_err_deg, and the raw per-bin values --
                        set calibration_file in measure_mueller_poly.json to it
  characterization.json, characterization.png
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from poly_common import (angles_of, bin_edges, bin_matrix, binned_power,
                         calibration_at, centres, dop, fit_coeffs,
                         float_or_none, full_turn, has_reference,
                         parse_with_config, read_curve, read_sweep,
                         resolve_path, stokes_from_coeffs, str2bool,
                         write_curve, write_sweep, zero_offset)

HERE = Path(__file__).resolve().parent
# bins without data (too little light, no sensitive state) are NaN on purpose
warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)


# --------------------------------------------------------------------------- #
# per-bin physics
# --------------------------------------------------------------------------- #


def retardance_from_dop(c, guess_deg, target=1.0, span=40.0, s3_sign=1):
    """Per bin: the QWP retardance that makes the DOP equal `target` (1 for a
    fully polarized state), the root nearest the guess; and how much the DOP
    moves per degree of retardance there (the sensitivity).
    c (5, N), guess_deg (N,), target number or (N,) -> (delta (N,), slope (N,))."""
    guess = np.asarray(guess_deg, float)
    N = guess.size
    tgt = np.broadcast_to(np.asarray(target, float), guess.shape)

    def f(d):
        two = d.ndim == 2
        S = stokes_from_coeffs(c[:, :, None] if two else c, d, s3_sign)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.linalg.norm(S[1:], axis=0) / S[0]
                    - (tgt[:, None] if two else tgt))

    with np.errstate(divide="ignore", invalid="ignore"):
        grid = guess[:, None] + np.arange(-span, span + 1e-9, 0.5)[None, :]
        v = f(grid)
        slope = f(guess + 0.5) - f(guess - 0.5)
    cross = ((np.sign(v[:, :-1]) != np.sign(v[:, 1:]))
             & np.isfinite(v[:, :-1]) & np.isfinite(v[:, 1:]))
    mid = 0.5 * (grid[:, :-1] + grid[:, 1:])
    dist = np.where(cross, np.abs(mid - guess[:, None]), np.inf)
    i = np.argmin(dist, axis=1)
    r = np.arange(N)
    found = np.isfinite(dist[r, i])
    lo, hi = grid[r, i], grid[r, i + 1]
    flo = f(lo)
    for _ in range(40):                              # bisection to ~1e-11 deg
        m = 0.5 * (lo + hi)
        fm = f(m)
        same = np.sign(fm) == np.sign(flo)
        lo, flo, hi = np.where(same, m, lo), np.where(same, fm, flo), np.where(same, hi, m)
    return np.where(found, 0.5 * (lo + hi), np.nan), slope


def combine(x, w, labels):
    """Weighted mean over sweeps, bin by bin, with its error split in two:

    stat    scatter of the repeats WITHIN each state, as an error of the mean
            (nan where no state was repeated)
    states  half the spread between the per-state means: what a perfect
            polarimeter would make zero, so the test of accuracy (nan for
            one state)
    x, w (n_sweeps, N); w = 0 where a sweep does not count in that bin.
    Returns mean, stat, states, per-state means {label: (N,)}, sweeps used (N,).
    """
    w = np.where(np.isfinite(x) & (w > 0), w, 0.0)
    x = np.nan_to_num(x)
    L = np.array(labels)
    W = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(W > 0, (w * x).sum(0) / W, np.nan)
        per, fitted = {}, np.zeros_like(x)
        for lab in dict.fromkeys(labels):
            m = L == lab
            Wl = w[m].sum(0)
            per[lab] = np.where(Wl > 0, (w[m] * x[m]).sum(0) / Wl, np.nan)
            fitted[m] = np.nan_to_num(per[lab])
        used = (w > 0).sum(0)
        n_st = np.array([np.isfinite(v) for v in per.values()]).sum(0)
        dof = used - n_st
        sd = np.sqrt((w * (x - fitted) ** 2).sum(0) / W * used / np.maximum(dof, 1))
        n_eff = W ** 2 / (w ** 2).sum(0)
        stat = np.where(dof > 0, sd / np.sqrt(n_eff), np.nan)
        pm = np.array(list(per.values()))
        states = np.where(n_st > 1, 0.5 * (np.nanmax(np.where(np.isfinite(pm), pm, -np.inf), 0)
                                           - np.nanmin(np.where(np.isfinite(pm), pm, np.inf), 0)),
                          np.nan)
    return mean, stat, states, per, used


def wmean(x, w):
    ok = np.isfinite(x) & (w > 0)
    W = np.where(ok, w, 0.0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(W > 0, (np.where(ok, w, 0.0) * np.nan_to_num(x)).sum(0) / W, np.nan)


def quad(*terms):
    """Quadrature sum of the terms that are known (nan = not measured), per bin."""
    t = np.array([np.broadcast_to(np.asarray(v, float), np.shape(terms[0]))
                  for v in terms if v is not None])
    s = np.nansum(t ** 2, axis=0)
    return np.where(np.isfinite(t).any(axis=0), np.sqrt(s), np.nan)


def smooth(wl, y, sigma, order):
    """Weighted polynomial in 1/lambda through y(wl), evaluated inside the
    range of the data; order None = no smoothing."""
    ok = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if order is None or ok.sum() <= order + 1:
        return y.copy(), None
    x = 1000.0 / wl
    p = np.polyfit(x[ok], y[ok], int(order), w=1 / sigma[ok])
    out = np.polyval(p, x)
    inside = (wl >= wl[ok].min()) & (wl <= wl[ok].max())
    return np.where(inside, out, np.nan), p


def _num(x, fmt="{:.3f}"):
    return fmt.format(x) if np.isfinite(x) else "-"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_runs(run_dirs, include_through, edges):
    """Every usable sweep, dark-subtracted, LED-corrected when possible, binned."""
    sweeps = []
    for d in run_dirs:
        run = Path(d).expanduser()
        meta = json.loads((run / "run.json").read_text())
        if meta.get("kind") not in ("qwp_characterization_spectral", "mueller_spectral"):
            raise SystemExit(f"{run.name} is not a spectral run")
        files = []
        for s in meta["states"]:
            if "sweeps" in s:                           # characterization run
                files += [(s["label"], f) for f in s["sweeps"]]
            else:                                       # Mueller run
                files.append((s["label"], s["input"]))
                if include_through:
                    files.append((s["label"] + "+sample", s["output"]))
        raw = [(lab, f, read_sweep(run / f)) for lab, f in files]
        ref = bool(raw) and all(has_reference(sw) for _, _, sw in raw)
        level = np.mean([sw["ref_W"].mean() for _, _, sw in raw]) if ref else None
        for lab, f, sw in raw:
            ang = angles_of(sw)
            sweeps.append({"run": run.name, "label": lab, "file": f, "sw": sw,
                           "P": binned_power(sw, bin_matrix(sw["wavelengths"], edges),
                                             level, ref),
                           "angles": ang, "walk": full_turn(ang), "laser_corrected": ref})
        print(f"  {run.name}: {len(raw)} sweeps"
              f"{' (LED-corrected)' if ref else ' (no reference meter)'}")
    return sweeps


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def analyse(run_dirs, a, out_dir: Path):
    edges = bin_edges(a.wl_min_nm, a.wl_max_nm, a.bin_nm)
    wl = centres(edges)
    N = wl.size
    print("\nloading")
    sweeps = load_runs(run_dirs, a.include_through_sample, edges)
    if not sweeps:
        raise SystemExit("no sweeps found")
    labels = [s["label"] for s in sweeps]
    sign = a.s3_sign
    assumed = {"qwp_zero_deg": a.qwp_zero_deg, "qwp_retardance_waves": a.qwp_retardance_waves,
               "calibration_file": a.calibration_file, "zero_source": a.zero_source,
               "s3_sign": sign}
    c0 = calibration_at(assumed, wl, HERE, strict=False)
    z0, d0 = c0["zero_deg"], c0["retardance_deg"]
    level = np.median([s["P"].mean(0) for s in sweeps], axis=0)
    weak = level < a.min_signal_rel * level.max()
    live = ~weak
    print(f"  assumed: {c0['source']}")
    print(f"  {N} bins x {a.bin_nm:g} nm, {edges[0]:g}-{edges[-1]:g} nm"
          + (f"; {weak.sum()} with too little light left out" if weak.any() else ""))

    # ---- 1. QWP zero from the phase of the S3 term (needs no retardance) ----
    for s in sweeps:
        f = fit_coeffs(s["P"], s["angles"], z0, s["walk"])
        S = stokes_from_coeffs(f["c"], d0, sign)
        s["s_assumed"] = S / S[0]
        s["dop_assumed"] = dop(S)
        s["zero"] = z0 + zero_offset(f["c"])
        s["w_zero"] = (f["c"][1] / f["c"][0]) ** 2
    Z = np.array([s["zero"] for s in sweeps])
    Wz = np.array([np.where((np.abs(s["s_assumed"][3]) > a.s3_min) & live, s["w_zero"], 0)
                   for s in sweeps])
    zero = {"assumed": z0 if np.ndim(z0) == 0 else list(np.asarray(z0))}
    z_raw = z_stat = z_states = np.full(N, np.nan)
    z_per, z_n = {}, np.zeros(N, int)
    if a.fix_zero_deg is not None:
        z1 = np.full(N, float(a.fix_zero_deg))
        zero["source"] = "fix_zero_deg"
    elif (Wz > 0).any():
        z_raw, z_stat, z_states, z_per, z_n = combine(Z, Wz, labels)
        zw = np.sqrt(Wz.sum(0))                       # ~ 1 / sigma of each bin
        if a.zero_smooth_order == 0:
            ok = np.isfinite(z_raw)
            z1 = np.full(N, np.sum(zw[ok] ** 2 * z_raw[ok]) / np.sum(zw[ok] ** 2))
            zero["source"] = "S3-term phase, one value for all wavelengths"
        else:
            z1, _ = smooth(wl, z_raw, 1 / np.where(zw > 0, zw, np.nan), a.zero_smooth_order)
            z1 = np.where(np.isfinite(z1), z1, np.nanmean(z1))
            zero["source"] = (f"S3-term phase, polynomial order {a.zero_smooth_order} "
                              f"in 1/lambda" if a.zero_smooth_order is not None
                              else "S3-term phase, per wavelength")
    else:
        z1 = np.broadcast_to(np.asarray(z0, float), (N,)).copy()
        zero["source"] = ("assumed (no bin with |S3| > s3_min: measure R or L to "
                          "calibrate the zero)")

    # ---- 2. QWP retardance: DOP = 1, at the zero just found -----------------
    cache = {}

    def retardances(zero_deg, target):
        """Per-sweep retardance (n_sweeps, N) for a zero and a DOP target."""
        key = (np.asarray(zero_deg).tobytes(), np.asarray(target).tobytes())
        if key not in cache:
            D, SL = [], []
            for s in sweeps:
                c = fit_coeffs(s["P"], s["angles"], zero_deg, s["walk"])["c"]
                d, sl = retardance_from_dop(c, d0, target, a.search_span_deg, sign)
                D.append(d)
                SL.append(sl)
            cache[key] = (np.array(D), np.array(SL))
        return cache[key]

    for s in sweeps:
        s["fit"] = fit_coeffs(s["P"], s["angles"], z1, s["walk"])
    D, SL = retardances(z1, 1.0)
    Wd = np.where((np.abs(SL) >= a.min_dop_slope) & np.isfinite(D) & live, SL ** 2, 0.0)
    ret = {"assumed_deg": list(d0)}
    d_raw, d_stat, d_states, d_per, d_n = combine(D, Wd, labels)
    if not (Wd > 0).any():
        print("  no sweep sensitive to the retardance: measure H")

    # ---- 3. error budget ---------------------------------------------------
    for s in sweeps:
        s["c2_rel"] = s["fit"]["c"][2] / s["fit"]["c"][0]
    low = np.array([np.where(np.abs(s["s_assumed"][3]) < a.s3_min, s["c2_rel"], np.nan)
                    for s in sweeps])
    with np.errstate(invalid="ignore", divide="ignore"):
        low_rms = np.sqrt(np.nanmean(low ** 2, axis=0))
        c2_left = np.degrees(0.5 * low_rms * np.sqrt(Wz).sum(0) / Wz.sum(0))
    c2_left = np.where(np.isfinite(c2_left) & (Wz.sum(0) > 0), c2_left, np.nan)
    z_total = quad(z_stat, z_states, c2_left)
    if a.fix_zero_deg is not None or not (Wz > 0).any():
        z_unc = np.full(N, np.nan)
    elif a.zero_smooth_order == 0:
        z_unc = np.full(N, np.nanmedian(z_total))     # one zero: its typical error
    else:
        z_unc = z_total

    from_zero = np.full(N, np.nan)
    zt = np.nan_to_num(z_unc)
    if (zt > 0).any():
        from_zero = 0.5 * np.abs(wmean(retardances(z1 + zt, 1.0)[0], Wd)
                                 - wmean(retardances(z1 - zt, 1.0)[0], Wd))
    input_dop = wmean(retardances(z1, a.input_dop_min)[0], Wd) - d_raw

    for s in sweeps:                                   # at the new calibration
        S = stokes_from_coeffs(s["fit"]["c"], d_raw, sign)
        s["S"], s["dop"] = S / S[0], dop(S)
        s["S0"] = S[0]
    vlike = np.array([(s["S"][1] < -0.95) for s in sweeps])
    leak_each = np.array([s["P"].min(0) / s["S0"] for s in sweeps])
    leak = np.where(vlike.any(0), np.where(vlike, leak_each, np.inf).min(0), np.nan)
    leak_term = np.full(N, np.nan)
    if np.isfinite(leak).any():
        leak_term = wmean(retardances(z1, np.where(np.isfinite(leak), 1 - leak, 1.0))[0],
                          Wd) - d_raw
        leak_term = np.where(np.isfinite(leak), leak_term, np.nan)
    d_total = quad(d_stat, d_states, from_zero, input_dop, leak_term)

    # ---- 4. smooth, and what is left at the new calibration -----------------
    d1, poly = smooth(wl, d_raw, d_total, a.smooth_order)
    ret.update(raw_deg=list(d_raw), stat_deg=list(d_stat), state_spread_deg=list(d_states),
               from_zero_deg=list(from_zero), input_dop_deg=list(input_dop),
               analyzer_leak_deg=list(leak_term), total_deg=list(d_total),
               value_deg=list(d1), smooth_order=a.smooth_order,
               per_state_deg={k: list(v) for k, v in d_per.items()})
    with np.errstate(invalid="ignore"):
        chi = (d_raw - d1) / d_total
    for s in sweeps:
        S = stokes_from_coeffs(s["fit"]["c"], d1, sign)
        s["dop_new"] = dop(S)
        c = s["fit"]["c"]
        s["walk_rel"] = (np.sqrt(np.sum(s["fit"]["walk"] ** 2, axis=0) / 2) / c[0]
                         if s["walk"] else np.full(N, np.nan))
        s["resid_rel"] = s["fit"]["resid_rms"] / c[0]
        r = s["P"] - s["fit"]["model"]
        with np.errstate(invalid="ignore", divide="ignore"):
            s["autocorr"] = (np.sum(r[:-1] * r[1:], 0)
                             / np.sqrt(np.sum(r[:-1] ** 2, 0) * np.sum(r[1:] ** 2, 0)))

    compare = None
    cf = resolve_path(a.compare_file, HERE)
    if cf is not None:
        cc = read_curve(cf)
        inside = (wl >= cc["wavelength_nm"][0]) & (wl <= cc["wavelength_nm"][-1])
        cmp_deg = np.where(inside, 360 * np.interp(wl, cc["wavelength_nm"],
                                                   cc["retardance_waves"]), np.nan)
        compare = {"file": str(cf), "deg": cmp_deg}

    # ---- report ------------------------------------------------------------
    mid = int(np.argmin(abs(wl - np.median(wl))))
    print(f"\n{'state':8s} {'sweeps':>6} {'S/S0 at %.0f nm' % wl[mid]:>26s} "
          f"{'zero bins':>9} {'retard. bins':>12} {'DOP-1 median, assumed -> new':>30}")
    for lab in dict.fromkeys(labels):
        m = np.array([l == lab for l in labels])
        S = np.mean([s["S"][:, mid] for s, k in zip(sweeps, m) if k], axis=0)
        before = np.nanmedian([s["dop_assumed"][live] - 1 for s, k in zip(sweeps, m) if k])
        after = np.nanmedian([s["dop_new"][live] - 1 for s, k in zip(sweeps, m) if k])
        print(f"{lab:8s} {m.sum():>6} [{S[1]:+.3f} {S[2]:+.3f} {S[3]:+.3f}] "
              f"{int((Wz[m] > 0).any(0).sum()):>9} {int((Wd[m] > 0).any(0).sum()):>12} "
              f"{100 * before:+13.2f}% -> {100 * after:+.2f}%")

    cols = [int(np.argmin(abs(wl - w))) for w in a.report_wl_nm
            if wl[0] - a.bin_nm <= w <= wl[-1] + a.bin_nm]
    cols = list(dict.fromkeys(cols))
    head = "".join(f"{wl[c]:>11.0f}" for c in cols)
    print(f"\n{'':34s}{head}  nm")
    rows = [("QWP ZERO (deg), per wavelength", z_raw, "{:11.3f}"),
            ("  total error", z_total, "{:11.3f}"),
            ("  zero used", z1, "{:11.3f}"),
            ("QWP RETARDANCE (deg), raw", d_raw, "{:11.3f}"),
            ("  total error", d_total, "{:11.3f}"),
            ("  smoothed (written to the file)", d1, "{:11.3f}"),
            ("  assumed", d0, "{:11.3f}")]
    if compare:
        rows.append(("  compare_file", compare["deg"], "{:11.3f}"))
    rows += [("  per state: " + k, v, "{:11.3f}") for k, v in d_per.items()]
    for name, v, fmt in rows:
        print(f"  {name[:32]:32s}" + "".join(fmt.format(v[c]) if np.isfinite(v[c])
                                              else f"{'-':>11}" for c in cols))

    print(f"\nERROR BUDGET, retardance (deg){'':4s}{head}")
    for name, v in (("statistical (scatter of repeats)", d_stat),
                    ("disagreement between states", d_states),
                    ("zero uncertainty, propagated", from_zero),
                    (f"input DOP {a.input_dop_min:g} (one-sided +)", input_dop),
                    ("analyzer leak (one-sided +)", leak_term),
                    ("total (quadrature)", d_total)):
        print(f"  {name:32s}" + "".join(f"{_num(v[c]):>11}" for c in cols))
    print(f"ERROR BUDGET, zero (deg){'':10s}{head}")
    for name, v in (("statistical (scatter of repeats)", z_stat),
                    ("disagreement between states", z_states),
                    ("cos 2t the zero does not explain", c2_left),
                    ("total (quadrature)", z_total)):
        print(f"  {name:32s}" + "".join(f"{_num(v[c]):>11}" for c in cols))
    if not np.isfinite(d_states).any() and (Wd > 0).any():
        print("  only one state calibrates the retardance: its accuracy is NOT checked "
              "-- add D or A")
    if not np.isfinite(d_stat).any() and (Wd > 0).any():
        print("  no state was repeated: the statistical part is not measured (repeats >= 2)")

    print("\nSUMMARY")
    if np.isfinite(z_raw).any():
        spread_z = np.nanmax(z_raw) - np.nanmin(z_raw)
        print(f"  zero: {z1[0]:.3f} deg" + (f" ± {np.nanmedian(z_total):.3f}"
                                            if a.zero_smooth_order == 0 else " (per wavelength)")
              + f" (assumed {np.mean(z0):g}); per-wavelength values span "
              f"{spread_z:.3f} deg over {wl[np.isfinite(z_raw)][0]:.0f}-"
              f"{wl[np.isfinite(z_raw)][-1]:.0f} nm")
    else:
        print(f"  zero: {zero['source']}")
    if poly is not None:
        print(f"  retardance: smoothed with a polynomial of order {a.smooth_order} in "
              f"1/lambda; raw - smoothed = {np.sqrt(np.nanmean(chi ** 2)):.2f} x the error "
              f"(rms; ~1 or less = the curve is smooth to within the errors)")
    if compare:
        dd = d1 - compare["deg"]
        if np.isfinite(dd).any():
            print(f"  vs {Path(compare['file']).name}: measured - file = "
                  f"{np.nanmean(dd):+.2f} deg mean, {np.nanmax(np.abs(dd)):.2f} deg max "
                  f"({np.nanmean(dd) / 360:+.4f} waves)")
    dop_b = np.sqrt(np.nanmean([(s["dop_assumed"][live] - 1) ** 2 for s in sweeps]))
    dop_a = np.sqrt(np.nanmean([(s["dop_new"][live] - 1) ** 2 for s in sweeps]))
    print(f"  DOP - 1 rms over all sweeps and wavelengths: {100 * dop_b:.2f}% at the "
          f"assumed calibration -> {100 * dop_a:.2f}%")
    if np.isfinite(low).any():
        print(f"  cos(2 theta) term in sweeps without S3: rms {100 * np.nanmedian(low_rms):.2f}% "
              f"of c0, median over wavelength (should be 0; not a zero error -- QWP "
              f"diattenuation or drift)")
    print(f"  fit residual: median {100 * np.nanmedian([s['resid_rel'][live] for s in sweeps]):.2f}% "
          f"of c0, lag-1 autocorrelation "
          f"{np.nanmedian([s['autocorr'][live] for s in sweeps]):.2f} (0 = noise, near 1 = "
          f"systematic)")
    walks = [s for s in sweeps if s["walk"]]
    if walks:
        wr = np.array([s["walk_rel"] for s in walks])[:, live]
        print(f"  beam walk (once-per-turn 1, 3, 5 theta), rms: median "
              f"{100 * np.nanmedian(wr):.3f}% of c0, max {100 * np.nanmax(wr):.3f}%  "
              f"(the fibre coupling makes this larger than with a power meter)")
    else:
        print("  beam walk: no 360-deg sweeps -- set qwp_sweep_deg to 360")
    if np.isfinite(leak).any():
        print(f"  analyzer: leak (min power / S0, V-like input) median "
              f"{np.nanmedian(leak):.2e} -> extinction >= {1 / np.nanmax(leak):.0f}:1 "
              f"everywhere measured (upper bound on the leak)")
    else:
        print("  analyzer: no V state measured -- add V for an extinction check")

    # ---- save --------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_file = out_dir / "qwp_calibration.csv"
    write_curve(cal_file, {
        "wavelength_nm": wl, "retardance_waves": d1 / 360,
        "retardance_err_waves": d_total / 360, "zero_deg": z1,
        "zero_err_deg": z_unc if np.isfinite(z_unc).any() else z_total,
        "retardance_raw_waves": d_raw / 360, "retardance_stat_waves": d_stat / 360,
        "retardance_states_waves": d_states / 360, "zero_raw_deg": z_raw,
        "sweeps_retardance": d_n, "sweeps_zero": z_n})
    zero.update(raw_deg=list(z_raw), stat_deg=list(z_stat), state_spread_deg=list(z_states),
                c2_leftover_deg=list(c2_left), total_deg=list(z_total), value_deg=list(z1),
                per_state_deg={k: list(v) for k, v in z_per.items()})
    results = {"run_dirs": [str(d) for d in run_dirs], "wavelength_nm": list(wl),
               "zero": zero, "retardance": ret, "analyzer_leak_bound": list(leak),
               "dop_rms_assumed": dop_b, "dop_rms_new": dop_a,
               "compare": None if compare is None else
               {"file": compare["file"], "deg": list(compare["deg"])},
               "settings": vars(a)}
    text = json.dumps(results, indent=2, default=lambda o: float(o)
                      if isinstance(o, (np.floating, np.integer)) else str(o))
    (out_dir / "characterization.json").write_text(text.replace("NaN", "null"))
    if a.save_figures or a.show_figures:
        figure(wl, d_raw, d_total, d1, d0, d_per, z_raw, z_total, z1, z0, z_per, sweeps,
               labels, live, compare, (d_stat, d_states, from_zero, input_dop, leak_term),
               out_dir, a)

    print(f"\nsaved in {out_dir}")
    print("\npaste into measure_mueller_poly.json (future runs) -- and, for old runs, "
          "into analyze_mueller_poly.json:")
    print(f'  "calibration_file": "{cal_file.resolve().as_posix()}",')
    if a.zero_smooth_order == 0 and np.isfinite(z_raw).any():
        print(f'  "qwp_zero_deg": {z1[0]:.2f},')
        print('  "zero_source": "constant",')
        print("and the zero tolerance into analyze_mueller_poly.json (the retardance "
              "tolerance comes from the file, per wavelength):")
        print(f'  "qwp_zero_uncertainty_deg": {np.nanmedian(z_total):.3f},')
    elif np.isfinite(z_raw).any():
        print('  "zero_source": "file",')
    return results


def figure(wl, d_raw, d_total, d1, d0, d_per, z_raw, z_total, z1, z0, z_per, sweeps,
           labels, live, compare, budget, out_dir, a):
    import matplotlib
    if not a.show_figures:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = list(dict.fromkeys(labels))
    col = {lab: f"C{i % 10}" for i, lab in enumerate(labs)}
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    a0 = ax[0, 0]
    for lab, v in d_per.items():
        a0.plot(wl, v, ".", color=col[lab], ms=4)
    a0.errorbar(wl, d_raw, d_total, fmt="none", ecolor="0.5", lw=0.6)
    a0.plot(wl, d1, "k-", lw=1.2, label="calibration (smoothed)")
    a0.plot(wl, d0, color="gray", ls="--", lw=1, label="assumed")
    if compare:
        a0.plot(wl, compare["deg"], color="C3", ls=":", lw=1.2,
                label=Path(compare["file"]).name)
    a0.set(title="QWP retardance from DOP = 1 (dots: per state)",
           xlabel="wavelength (nm)", ylabel="retardance (deg)")
    a0.legend(fontsize=7)
    sec = a0.secondary_yaxis("right", functions=(lambda d: d / 360, lambda w: w * 360))
    sec.set_ylabel("waves")

    a1 = ax[0, 1]
    for lab, v in z_per.items():
        a1.plot(wl, v, ".", color=col[lab], ms=4)
    a1.errorbar(wl, z_raw, z_total, fmt="none", ecolor="0.5", lw=0.6)
    a1.plot(wl, z1, "k-", lw=1.2, label="zero used")
    a1.plot(wl, np.broadcast_to(z0, wl.shape), color="gray", ls="--", lw=1, label="assumed")
    a1.set(title="QWP zero from the S3-term phase (R, L)", xlabel="wavelength (nm)",
           ylabel="stage angle (deg)")
    a1.legend(fontsize=7)

    a2 = ax[1, 0]
    for lab in labs:
        m = [s for s in sweeps if s["label"] == lab]
        a2.plot(wl, np.nanmedian([s["dop_assumed"] for s in m], 0) - 1, "--",
                color=col[lab], lw=0.8)
        a2.plot(wl, np.nanmedian([s["dop_new"] for s in m], 0) - 1, "-",
                color=col[lab], lw=1.2)
    a2.axhline(0, color="k", lw=0.6)
    a2.set(title="DOP - 1 per state: assumed calibration (dashed), new (solid)",
           xlabel="wavelength (nm)", ylabel="DOP - 1")
    lo = np.nanpercentile([s["dop_new"][live] - 1 for s in sweeps], 1)
    hi = np.nanpercentile([s["dop_assumed"][live] - 1 for s in sweeps], 99)
    if np.isfinite(lo) and np.isfinite(hi):
        a2.set_ylim(min(lo, -0.01) * 1.5, max(hi, 0.01) * 1.5)

    a3 = ax[1, 1]
    for v, name in zip(budget, ("statistical", "between states", "zero, propagated",
                                "input DOP (+)", "analyzer leak (+)")):
        if np.isfinite(v).any():
            a3.plot(wl, np.abs(v), lw=1, label=name)
    a3.plot(wl, d_total, "k-", lw=1.5, label="total")
    a3.axhline(1.2, color="gray", ls=":", lw=0.8, label="λ/300")
    a3.set_yscale("log")
    a3.set(title="retardance error budget", xlabel="wavelength (nm)", ylabel="deg")
    a3.legend(fontsize=7)

    handles = [plt.Line2D([], [], marker="o", ls="", color=col[l], label=l) for l in labs]
    fig.legend(handles=handles, loc="upper center", ncol=len(labs), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if a.save_figures:
        fig.savefig(out_dir / "characterization.png", dpi=130)
    if a.show_figures:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def quick_look(sw, a, cal, base):
    """Live feedback in 20-nm bins at the ASSUMED calibration; the analysis at
    the end redoes it properly."""
    from measure_mueller_poly import quick_stokes
    f = quick_stokes(sw, a, cal, base, 20.0)
    c = calibration_at(cal, f["wl"], base, strict=False)
    d, slope = retardance_from_dop(f["c"], c["retardance_deg"], 1.0, a.search_span_deg,
                                   cal["s3_sign"])
    z = c["zero_deg"] + zero_offset(f["c"])
    for w in a.report_wl_nm:
        if not (f["wl"][0] - 1e-9 <= w <= f["wl"][-1] + 1e-9):
            continue
        k = int(np.argmin(abs(f["wl"] - w)))
        S = f["S"][:, k] / f["S"][0, k]
        txt = (f"    {f['wl'][k]:5.0f} nm  [{S[1]:+.3f} {S[2]:+.3f} {S[3]:+.3f}]  "
               f"DOP {np.linalg.norm(S[1:]):.4f}")
        if abs(S[3]) > a.s3_min:
            txt += f"  | zero -> {np.atleast_1d(z)[k if np.ndim(z) else 0]:.2f}"
        if abs(slope[k]) >= a.min_dop_slope and np.isfinite(d[k]):
            txt += f"  | retardance -> {d[k]:.2f}"
        print(txt)


def measure(a) -> Path:
    from measure_mueller_poly import (Rig, SimRig, ambient_report, check_exposure,
                                      copy_calibration, start, sweep, sweep_health)

    span = float(a.qwp_sweep_deg)
    root = Path(a.out) if a.out else Path.home() / "Desktop"
    run = root / time.strftime("qwp_char_poly_%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    cal = copy_calibration(a, run)
    c = calibration_at(cal, np.array([a.wl_min_nm, a.wl_max_nm]), run, strict=False)
    print(f"\nrun folder : {run}")
    print(f"assumed    : {c['source']}")
    print(f"sweep      : {a.qwp_steps} steps over {span:g} deg, {a.repeats} repeats per state")

    if a.simulate:
        # the simulator's polarimeter is the TRUE one; the analysis starts
        # from the assumed values and must find these
        truth = copy.copy(a)
        truth.qwp_zero_deg = a.qwp_zero_deg + a.sim_true_zero_offset_deg
        print(f"simulated  : TRUE zero {truth.qwp_zero_deg:g} deg, TRUE retardance "
              f"{a.sim_true_retardance_scale:g} x the assumed curve")
        rig = SimRig(truth, retardance_scale=a.sim_true_retardance_scale)
    else:
        rig = Rig(a)

    meta = {"kind": "qwp_characterization_spectral",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_file": str(Path(a.config).resolve()), "settings": vars(a),
            "calibration": cal, "qwp_sweep_deg": span, "notes": a.notes, "states": []}

    with rig as r:
        def save():
            meta["hardware"] = r.describe()
            (run / "run.json").write_text(json.dumps(meta, indent=2, default=str))

        if a.simulate:
            np.savez_compressed(run / "sim_truth.npz", **r.truth())
        r.set_sample(False)
        amb = start(r, a)
        input("\nNO SAMPLE in the beam for the whole run. Set H (both state wave "
              "plates OUT), then press Enter to check the exposure...")
        check_exposure(r, a)
        ambient_report(amb, r, a)
        print(f"  exposure for the whole run: {r.exposure_ms:g} ms x {r.hw_average} frames")
        save()

        print("\nSuggested states (approximate is fine -- each is measured):\n"
              "  H    both state wave plates OUT (polarizer only), max signal -> retardance\n"
              "  R/L  state QWP in, as circular as is easy               -> zero\n"
              "  D/A  cross-check of the retardance;  V  analyzer leak\n"
              "Don't touch anything while a state's repeats run.")
        idx = 0
        while True:
            label = input(f"\nstate {idx + 1}: set it by hand, then type a label "
                          f"(Enter = finish): ").strip()
            if not label:
                break
            idx += 1
            r.set_state(label)
            meta["states"].append({"index": idx, "label": label, "sweeps": []})
            for k in range(int(a.repeats)):
                sw = sweep(r, a, f"{label} {k + 1}/{a.repeats}", span)
                name = f"state_{idx:02d}_rep_{k + 1:02d}.npz"
                write_sweep(run / name, sw)
                meta["states"][-1]["sweeps"].append(name)
                save()
                print(f"    {sweep_health(sw, a)}")
                quick_look(sw, a, cal, run)

    if not meta["states"]:
        raise SystemExit("nothing measured")
    print(f"\n{sum(len(s['sweeps']) for s in meta['states'])} sweeps saved in {run}")
    return run


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    from measure_mueller_poly import hardware_args

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", default=[],
                    help="analyse these runs instead of measuring (characterization "
                         "or spectral Mueller runs)")
    ap.add_argument("--repeats", type=int, default=5,
                    help="sweeps per state, nothing touched in between")
    ap.add_argument("--bin-nm", type=float, default=10.0, dest="bin_nm",
                    help="wavelength bin of the calibration curve")
    ap.add_argument("--min-signal-rel", type=float, default=0.05, dest="min_signal_rel")
    ap.add_argument("--s3-min", type=float, default=0.3, dest="s3_min",
                    help="|S3/S0| a sweep needs, in a bin, to calibrate the zero there")
    ap.add_argument("--min-dop-slope", type=float, default=0.01, dest="min_dop_slope",
                    help="DOP change per deg of retardance a sweep needs to calibrate "
                         "the retardance (H ~0.035, D/A ~0.02, V/R/L ~0)")
    ap.add_argument("--search-span-deg", type=float, default=40.0, dest="search_span_deg",
                    help="look for the retardance within this of the assumed curve")
    ap.add_argument("--include-through-sample", type=str2bool, default=False,
                    dest="include_through_sample",
                    help="Mueller runs: also use the through-sample sweeps (only for a "
                         "non-depolarizing sample)")
    ap.add_argument("--fix-zero-deg", type=float_or_none, default=None, dest="fix_zero_deg",
                    help="use this zero instead of estimating it")
    ap.add_argument("--zero-smooth-order", type=float_or_none, default=0,
                    dest="zero_smooth_order",
                    help="0 = one zero for all wavelengths; n = polynomial in 1/lambda "
                         "(superachromatic plates); null = every bin on its own")
    ap.add_argument("--smooth-order", type=float_or_none, default=4, dest="smooth_order",
                    help="retardance curve: polynomial order in 1/lambda; null = the raw "
                         "per-bin values")
    ap.add_argument("--input-dop-min", type=float, default=0.998, dest="input_dop_min",
                    help="worst-case DOP of the input states, for the error budget: "
                         "1 - 2/ER for an unpolarized LED (0.998 for 1000:1)")
    ap.add_argument("--compare-file", default=None, dest="compare_file",
                    help="a retardance curve to compare with, e.g. the Thorlabs data")
    ap.add_argument("--report-wl-nm", type=float, nargs="*",
                    default=[500, 550, 600, 650, 700, 750, 800], dest="report_wl_nm")
    ap.add_argument("--show-figures", type=str2bool, default=True, dest="show_figures")
    ap.add_argument("--save-figures", type=str2bool, default=True, dest="save_figures")
    ap.add_argument("--sim-true-zero-offset-deg", type=float, default=-0.7,
                    dest="sim_true_zero_offset_deg",
                    help="simulator only: true zero = assumed + this")
    ap.add_argument("--sim-true-retardance-scale", type=float, default=0.985,
                    dest="sim_true_retardance_scale",
                    help="simulator only: true retardance = this x the assumed curve")
    hardware_args(ap)

    # measure_mueller_poly.json first (hardware, assumed calibration), then
    # this script's own JSON, then the command line
    shared = HERE / "measure_mueller_poly.json"
    if shared.is_file():
        known = {x.dest for x in ap._actions}
        data = json.loads(shared.read_text())
        ap.set_defaults(**{k: v for k, v in data.items() if k in known})
        print(f"hardware + assumed calibration: {shared}")
    a = parse_with_config(ap, argv, HERE / "QWP_analyzer_characterization_poly.json")
    for k in ("zero_smooth_order", "smooth_order"):
        v = getattr(a, k)
        setattr(a, k, None if v is None else int(v))

    if a.run_dirs:
        dirs = [Path(d) for d in a.run_dirs]
        out = (dirs[0] / "characterization" if len(dirs) == 1 else
               (Path(a.out) if a.out else Path.home() / "Desktop")
               / time.strftime("qwp_char_poly_analysis_%Y%m%d_%H%M%S"))
    else:
        run = measure(a)
        dirs, out = [run], run / "characterization"
    analyse(dirs, a, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
