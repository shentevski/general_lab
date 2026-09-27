#!/usr/bin/env python3
"""Characterize the polarimeter itself (rotating QWP + fixed analyzer):
QWP zero, QWP retardance, beam walk, analyzer leak, reference coupling.

    python QWP_analyzer_characterization.py                  # measure, then analyse
    python QWP_analyzer_characterization.py --simulate       # rehearse, no hardware
    python QWP_analyzer_characterization.py <run_dir> ...    # analyse existing runs

Hardware, and the calibration currently ASSUMED, come from measure_mueller.json
(one place for serials, motor and calibration); the settings of this script from
QWP_analyzer_characterization.json. Existing runs can be characterization runs
or Mueller runs -- of a Mueller run the input sweeps (sample out) are used.

The idea
  NO SAMPLE in the beam. Every input state comes from laser -> polarizer ->
  wave plates, so it is fully polarized: DOP = 1 up to the polarizer's leak
  times the laser's unpolarized fraction (input_dop_min is the worst case),
  and anything else is the polarimeter's error. Each state is swept `repeats`
  times without touching anything. The states only need to be approximate:
  each one is measured, not assumed.

  H    (state QWP removed; HWP turned for maximum signal at the QWP zero)
       -> QWP RETARDANCE. The depth of H's sweep is cos^2(delta/2): its DOP
          moves 3.5% per degree of retardance error. V, R and L are blind to it.
  R/L  -> QWP ZERO. The S3 term must be a pure sin(2 theta). A zero that is
          off by eps adds a cos(2 theta) part with c2/c1 = -tan(2 eps) --
          independent of the retardance and of the laser power.
  D/A  (optional) cross-check: must give the same retardance as H. How much
          the states disagree is the real accuracy of the calibration.
  V    (optional) analyzer leak: an upper bound on 1 / extinction ratio.

  Sweep 360 deg (qwp_sweep_deg): a wedge or tilt of the rotating QWP moves the
  beam on the detector with a 360-deg period (beam walk). A 360-deg sweep fits
  it as its own term and compares the two half turns; 180 deg cannot separate
  it from the Stokes terms.

Error budget, for the zero and the retardance separately
  statistical      scatter of the repeats of one state (needs repeats)
  states           half the spread between states -- the real accuracy test
  cos 2t leftover  a cos(2 theta) term that the zero does not explain (seen in
                   states without S3) would shift the zero this much
  zero             the zero's own uncertainty, propagated into the retardance
  input DOP        if the input states are at input_dop_min instead of 1, the
                   retardance is higher by this much (one-sided)
  analyzer leak    the same for the leak measured with V (one-sided)

Output in <run>/characterization/: characterization.json, characterization.png,
and the values to paste into measure_mueller.json and analyze_mueller.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

from mueller_common import (corrected_power, extract_stokes, full_turn,
                            has_reference, parse_with_config, read_sweep,
                            stokes_from_coeffs, waves_to_deg, write_sweep)

HERE = Path(__file__).resolve().parent


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y", "on")


def float_or_none(s):
    return None if str(s).lower() in ("", "none", "null") else float(s)


# --------------------------------------------------------------------------- #
# per-sweep physics
# --------------------------------------------------------------------------- #


def fit_sweep(power, angles_deg, zero_deg, walk=None):
    """The Mueller scripts' own fit (extract_stokes), beam-walk harmonics
    included over a full turn. Only its coefficients c0..c4 are used: the
    scans over the retardance convert them with stokes_from_coeffs."""
    f = extract_stokes(power, angles_deg, qwp_zero_deg=zero_deg,
                       retardance_deg=90.0, walk=walk)
    return {"c": f["coeffs"], "walk": f["walk"], "model": f["model"],
            "resid": np.asarray(power, float) - f["model"]}


def zero_offset(c):
    """How far the true QWP zero is from the one used in the fit (deg).

    The S3 term is K sin(2(theta - eps)) = K cos2eps sin2theta - K sin2eps cos2theta,
    so c1 = K cos 2eps, c2 = -K sin 2eps. K cancels: no retardance, no power."""
    s = 1.0 if c[1] >= 0 else -1.0
    return float(-0.5 * np.degrees(np.arctan2(s * c[2], s * c[1])))


def dop(c, delta_deg, s3_sign=1):
    S = stokes_from_coeffs(c, delta_deg, s3_sign)
    return float(np.linalg.norm(S[1:]) / S[0])


def retardance_from_dop(c, guess_deg, target=1.0, lo=60.0, hi=120.0):
    """The QWP retardance that makes this state's DOP equal `target` (1 for a
    fully polarized state), and how much its DOP moves per degree of
    retardance (its sensitivity)."""
    f = lambda d: dop(c, d) - target
    slope = f(guess_deg + 0.5) - f(guess_deg - 0.5)
    grid = np.arange(lo, hi + 1e-9, 0.5)
    v = np.array([f(g) for g in grid])
    roots = []
    for i in np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        a, b = grid[i], grid[i + 1]
        for _ in range(40):                     # bisection to ~1e-12 deg
            m = 0.5 * (a + b)
            if np.sign(f(m)) == np.sign(f(a)):
                a = m
            else:
                b = m
        roots.append(0.5 * (a + b))
    if not roots:
        return float("nan"), float(slope)
    return float(min(roots, key=lambda r: abs(r - guess_deg))), float(slope)


def ref_coupling(sw):
    """Reference change per watt of signal change, slow laser drift removed.
    Nonzero = light from the polarimeter (e.g. the wire grid reflecting the
    rejected polarization) reaches the reference, or feeds back into the laser."""
    t = sw["t_s"] - sw["t_s"].mean()
    t = t / max(np.ptp(t), 1e-9)
    X = np.column_stack([np.ones_like(t), t, t ** 2, sw["power_W"]])
    c, *_ = np.linalg.lstsq(X, sw["ref_W"], rcond=None)
    res = sw["ref_W"] - X @ c
    dof = max(len(t) - X.shape[1], 1)
    err = np.sqrt(res @ res / dof * np.linalg.inv(X.T @ X)[-1, -1])
    return float(-c[-1]), float(err)


def combine(items, key, weight):
    """Weighted mean over sweeps, with its error split in two:

    stat    scatter of the repeats WITHIN each state, as an error of the mean
            (nan when no state was repeated)
    states  half the spread between the per-state means: what a perfect
            polarimeter would make zero, so the test of accuracy (nan for one state)
    """
    x = np.array([s[key] for s in items], float)
    w = np.array([s[weight] for s in items], float)
    labels = [s["label"] for s in items]
    mean = float(np.sum(w * x) / np.sum(w))
    per = {}
    for lab in dict.fromkeys(labels):
        m = np.array([l == lab for l in labels])
        pm = float(np.sum(w[m] * x[m]) / np.sum(w[m]))
        sp = (float(np.sqrt(np.sum(w[m] * (x[m] - pm) ** 2) / np.sum(w[m])))
              if m.sum() > 1 else float("nan"))
        per[lab] = {"mean": pm, "spread": sp, "n": int(m.sum())}
    dof = len(x) - len(per)
    stat = float("nan")
    if dof > 0:
        r = x - np.array([per[l]["mean"] for l in labels])
        sd = np.sqrt(np.sum(w * r ** 2) / np.sum(w) * len(x) / dof)
        stat = float(sd / np.sqrt(w.sum() ** 2 / np.sum(w ** 2)))
    states = 0.5 * float(np.ptp([v["mean"] for v in per.values()])) if len(per) > 1 else float("nan")
    return mean, stat, states, per


def _num(x, fmt="{:.3f}"):
    return fmt.format(x) if np.isfinite(x) else "-"


def quad(*terms):
    """Quadrature sum of the terms that are known (nan = not measured)."""
    t = [v for v in terms if v is not None and np.isfinite(v)]
    return float(np.sqrt(np.sum(np.square(t)))) if t else float("nan")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_runs(run_dirs, include_through):
    """Every usable sweep of the given runs, laser-corrected when possible."""
    sweeps = []
    for d in run_dirs:
        run = Path(d).expanduser()
        meta = json.loads((run / "run.json").read_text())
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
            n = sw["commanded_deg"].size
            com = sw["commanded_deg"]
            span = (com[-1] - com[0]) * n / (n - 1) if n > 1 else 0.0
            ang = np.where(np.isfinite(sw["measured_deg"]), sw["measured_deg"], com)
            sweeps.append({"run": run.name, "label": lab, "file": f, "sw": sw,
                           "power": corrected_power(sw, level) if ref else sw["power_W"],
                           "angles": ang, "span": float(span), "walk": full_turn(ang),
                           "laser_corrected": ref, "t_mid": float(np.mean(sw["t_s"]))})
        print(f"  {run.name}: {len(raw)} sweeps"
              f"{' (laser-corrected)' if ref else ' (no reference meter)'}")
    return sweeps


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def analyse(run_dirs, a, out_dir: Path):
    print("\nloading")
    sweeps = load_runs(run_dirs, a.include_through_sample)
    if not sweeps:
        raise SystemExit("no sweeps found")
    z0, d0, sign = a.qwp_zero_deg, waves_to_deg(a.qwp_retardance_waves), a.s3_sign

    # ---- 1. QWP zero from the phase of the S3 term (needs no retardance) ----
    for s in sweeps:
        f = fit_sweep(s["power"], s["angles"], z0, s["walk"])
        S = stokes_from_coeffs(f["c"], d0, sign)
        s["s_assumed"] = S / S[0]
        s["dop_assumed"] = float(np.linalg.norm(S[1:]) / S[0])
        s["zero"] = z0 + zero_offset(f["c"])
        s["w_zero"] = float((f["c"][1] / f["c"][0]) ** 2)   # S3-term size
    zs = [s for s in sweeps if abs(s["s_assumed"][3]) > a.s3_min]
    zero = {"assumed": z0, "n": len(zs)}
    if a.fix_zero_deg is not None:
        z1 = a.fix_zero_deg
        zero.update(value=z1, source="fix_zero_deg")
    elif zs:
        z1, stat, states, per = combine(zs, "zero", "w_zero")
        zero.update(value=z1, stat=stat, state_spread=states, per_state=per,
                    source="S3-term phase")
    else:
        z1 = z0
        zero.update(value=z0, source="assumed (no sweep with |S3| > s3_min: "
                                     "measure R or L to calibrate the zero)")

    # ---- 2. QWP retardance: DOP = 1, at the zero just found -----------------
    def retardances(zero_deg, target):
        """Per-sweep retardance for a given zero and DOP target."""
        out = []
        for s in sweeps:
            c = (s["fit"]["c"] if zero_deg == z1 else
                 fit_sweep(s["power"], s["angles"], zero_deg, s["walk"])["c"])
            out.append(retardance_from_dop(c, d0, target))
        return out

    for s in sweeps:
        s["fit"] = fit_sweep(s["power"], s["angles"], z1, s["walk"])
    for s, (d, slope) in zip(sweeps, retardances(z1, 1.0)):
        s["delta"], s["slope"], s["w_delta"] = d, slope, slope ** 2
    ds = [s for s in sweeps if abs(s["slope"]) >= a.min_dop_slope and np.isfinite(s["delta"])]
    ret = {"assumed_deg": d0, "n": len(ds)}
    if ds:
        d1, stat, states, per = combine(ds, "delta", "w_delta")
        ret.update(value_deg=d1, value_waves=d1 / 360.0, stat=stat,
                   state_spread=states, per_state=per)
    else:
        d1 = d0
        ret.update(value_deg=d0, value_waves=d0 / 360.0,
                   note="no sweep sensitive to the retardance: measure H")

    # ---- 3. what is left at the new calibration ---------------------------
    for s in sweeps:
        f = s["fit"]
        S = stokes_from_coeffs(f["c"], d1, sign)
        s["S"] = S / S[0]
        s["dop"] = float(np.linalg.norm(S[1:]) / S[0])
        s["S0"] = float(S[0])
        s["c2_rel"] = float(f["c"][2] / f["c"][0])
        # rms of the odd-harmonic (1, 3, 5 theta) part: what repeats once per turn
        s["walk_rel"] = (float(np.sqrt(np.sum(np.square(f["walk"])) / 2) / f["c"][0])
                         if s["walk"] else float("nan"))
        r = f["resid"]
        s["resid_rel"] = float(np.std(r) / f["c"][0])
        s["resid_autocorr"] = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if np.std(r) > 0 else 0.0
        s["half_diff"] = float("nan")
        if s["walk"]:                                   # compare the two half turns
            com = s["sw"]["commanded_deg"]
            first = (com - com[0]) < s["span"] / 2
            halves = []
            for m in (first, ~first):
                c = fit_sweep(s["power"][m], s["angles"][m], z1, False)["c"]
                Sh = stokes_from_coeffs(c, d1, sign)
                halves.append(Sh / Sh[0])
            s["half_diff"] = float(np.linalg.norm(halves[0][1:] - halves[1][1:]))
        s["k_ref"], s["k_ref_err"] = (ref_coupling(s["sw"]) if has_reference(s["sw"])
                                      else (float("nan"), float("nan")))

    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")
    dop_before = rms([s["dop_assumed"] - 1 for s in sweeps])
    dop_after = rms([s["dop"] - 1 for s in sweeps])
    low_s3 = [s["c2_rel"] for s in sweeps if abs(s["S"][3]) < a.s3_min]
    vlike = [s for s in sweeps if s["S"][1] < -0.95]
    leak = (min(float(np.min(s["power"]) / s["S0"]) for s in vlike) if vlike else float("nan"))
    walks = [s for s in sweeps if s["walk"]]
    refs = [s for s in sweeps if np.isfinite(s["k_ref"])]
    in_zs, in_ds = {id(s) for s in zs}, {id(s) for s in ds}
    nan = float("nan")

    # ---- 4. error budget ---------------------------------------------------
    # zero: a cos(2t) term that is not a zero error (it shows in states without
    # S3) shifts a sweep's zero by ~ 0.5 * c2 / (c1/c0) rad; weight like the mean
    zero["c2_leftover"] = nan
    if zs and low_s3 and a.fix_zero_deg is None:
        w = np.array([s["w_zero"] for s in zs])
        zero["c2_leftover"] = float(np.degrees(0.5 * rms(low_s3) * np.sum(w / np.sqrt(w)) / np.sum(w)))
    zero["total"] = (quad(zero.get("stat", nan), zero.get("state_spread", nan), zero["c2_leftover"])
                     if "per_state" in zero else nan)

    for key in ("from_zero", "input_dop", "analyzer_leak"):
        ret[key] = nan
    if ds:
        idx = [i for i, s in enumerate(sweeps) if id(s) in in_ds]
        wts = np.array([sweeps[i]["w_delta"] for i in idx])

        def mean_of(results):
            v = np.array([results[i][0] for i in idx])
            ok = np.isfinite(v)
            return float(np.sum(wts[ok] * v[ok]) / np.sum(wts[ok])) if ok.any() else nan

        sz = zero["total"]
        if np.isfinite(sz) and sz > 0:
            ret["from_zero"] = 0.5 * abs(mean_of(retardances(z1 + sz, 1.0))
                                         - mean_of(retardances(z1 - sz, 1.0)))
        # one-sided: a true DOP below 1 means the real retardance is higher
        ret["input_dop"] = mean_of(retardances(z1, a.input_dop_min)) - d1
        if np.isfinite(leak):
            ret["analyzer_leak"] = mean_of(retardances(z1, 1.0 - leak)) - d1
        ret["total"] = quad(ret.get("stat", nan), ret.get("state_spread", nan),
                            ret["from_zero"], ret["input_dop"], ret["analyzer_leak"])

    # ---- report ------------------------------------------------------------
    print(f"\n{'#':>3} {'run':>6} {'state':6s} {'S1/S0':>7} {'S2/S0':>7} {'S3/S0':>7} "
          f"{'DOP':>7} {'zero':>7} {'retard':>7} {'dDOP/deg':>8} {'cos2t':>7} "
          f"{'walk':>6} {'halves':>6}")
    for i, s in enumerate(sweeps):
        z = f"{s['zero']:7.2f}" if id(s) in in_zs else f"{'-':>7}"
        d = f"{s['delta']:7.2f}" if id(s) in in_ds else f"{'-':>7}"
        w = f"{100 * s['walk_rel']:5.2f}%" if s["walk"] else f"{'-':>6}"
        h = f"{s['half_diff']:6.3f}" if s["walk"] else f"{'-':>6}"
        print(f"{i + 1:>3} {s['run'][-6:]:>6} {s['label']:6s} {s['S'][1]:+7.3f} "
              f"{s['S'][2]:+7.3f} {s['S'][3]:+7.3f} {s['dop']:7.4f} {z} {d} "
              f"{s['slope']:+8.4f} {s['c2_rel']:+7.4f} {w} {h}")

    print("\nQWP ZERO  (phase of the S3 term; needs R or L)")
    if "per_state" in zero:
        for k, v in zero["per_state"].items():
            print(f"  {k:8s} {v['mean']:8.3f} deg   repeats scatter {_num(v['spread'])}   "
                  f"({v['n']} sweeps)")
        print(f"  zero     {z1:8.3f} deg   (assumed {z0:g})")
    else:
        print(f"  {zero['value']:.3f} deg -- {zero['source']}")

    print("\nQWP RETARDANCE  (DOP = 1; needs H, cross-checked by D / A)")
    if "per_state" in ret:
        for k, v in ret["per_state"].items():
            print(f"  {k:8s} {v['mean']:8.3f} deg   repeats scatter {_num(v['spread'])}   "
                  f"({v['n']} sweeps)")
        print(f"  retardance {d1:.3f} deg = {d1 / 360:.5f} waves   (assumed {d0:.2f} deg)")
    else:
        print(f"  {ret['note']}")

    print(f"\nERROR BUDGET{'':30s}{'zero (deg)':>12}{'retardance (deg)':>18}")
    rows = [("statistical (scatter of repeats)", zero.get("stat", nan), ret.get("stat", nan)),
            ("disagreement between states", zero.get("state_spread", nan),
             ret.get("state_spread", nan)),
            ("cos 2t the zero does not explain", zero["c2_leftover"], None),
            ("zero uncertainty, propagated", None, ret["from_zero"]),
            (f"input DOP down to {a.input_dop_min:g} (one-sided +)", None, ret["input_dop"]),
            ("analyzer leak (one-sided +)", None, ret["analyzer_leak"])]
    cell = lambda v: f"{'':>12}" if v is None else f"{_num(v):>12}"
    for name, zv, dv in rows:
        print(f"  {name:40s}{cell(zv)}{cell(dv):>18}")
    print(f"  {'total (quadrature)':40s}{cell(zero['total'])}{cell(ret.get('total', nan)):>18}")
    if "per_state" in ret and not np.isfinite(ret.get("state_spread", nan)):
        print("  only one state calibrates the retardance: its accuracy is NOT checked "
              "-- add D or A")
    if "per_state" in ret and not np.isfinite(ret.get("stat", nan)):
        print("  no state was repeated: the statistical part is not measured "
              "(repeats >= 2)")
    if "per_state" in zero and not np.isfinite(zero.get("state_spread", nan)):
        print("  only one state calibrates the zero: measure both R and L")

    print("\nWHAT IS LEFT at the new calibration")
    print(f"  DOP - 1 rms over all sweeps: {100 * dop_before:.2f}% at the assumed "
          f"calibration -> {100 * dop_after:.2f}%")
    if low_s3:
        print(f"  cos(2 theta) term in sweeps without S3: rms {100 * rms(low_s3):.2f}% of c0 "
              f"(should be 0; not a zero error -- QWP diattenuation or drift)")
    print(f"  fit residual: median {100 * np.median([s['resid_rel'] for s in sweeps]):.2f}% "
          f"of c0, lag-1 autocorrelation {np.median([s['resid_autocorr'] for s in sweeps]):.2f} "
          f"(0 = noise, near 1 = systematic)")

    print("\nBEAM WALK  (360-deg sweeps)")
    if walks:
        print(f"  once-per-turn part (1, 3, 5 theta), rms: median "
              f"{100 * np.median([s['walk_rel'] for s in walks]):.3f}% of c0, "
              f"max {100 * max(s['walk_rel'] for s in walks):.3f}%")
        print(f"  first vs second half turn: |dS|/S0 median "
              f"{np.median([s['half_diff'] for s in walks]):.4f}")
    else:
        print("  no 360-deg sweeps -- set qwp_sweep_deg to 360 to measure it")

    print("\nANALYZER")
    if vlike:
        print(f"  leak (min power / S0 for V-like input): {leak:.2e} -> extinction "
              f"ratio >= {1 / leak:.0f}:1 (upper bound on the leak)")
    else:
        print("  no V state measured -- add V for an extinction check")
    if refs:
        k = np.array([s["k_ref"] for s in refs])
        sig = sum(abs(s["k_ref"]) > 3 * s["k_ref_err"] for s in refs)
        print(f"  back-reflection into the reference: {np.median(k):+.4f} W per W of "
              f"signal (median), significant in {sig} of {len(refs)} sweeps")
    else:
        print("  no reference meter data")

    print("\npaste into measure_mueller.json (future runs) and analyze_mueller.json "
          "(re-analysis of old runs):")
    print(f'  "qwp_zero_deg": {z1:.2f},')
    print(f'  "qwp_retardance_waves": {d1 / 360:.4f},')
    if np.isfinite(ret.get("total", nan)) and np.isfinite(zero["total"]):
        print("and the tolerances (the totals above) into analyze_mueller.json:")
        print(f'  "retardance_uncertainty_waves": {ret["total"] / 360:.5f},')
        print(f'  "qwp_zero_uncertainty_deg": {zero["total"]:.3f},')

    # ---- save --------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sweep = [{"run": s["run"], "label": s["label"], "file": s["file"],
                  "S_normalized": s["S"].tolist(), "dop": s["dop"],
                  "dop_assumed_cal": s["dop_assumed"],
                  "zero_deg": s["zero"] if id(s) in in_zs else None,
                  "retardance_deg": s["delta"] if id(s) in in_ds else None,
                  "dop_per_deg_retardance": s["slope"], "cos2t_rel": s["c2_rel"],
                  "walk_rel": s["walk_rel"], "half_turn_diff": s["half_diff"],
                  "resid_rel": s["resid_rel"], "resid_autocorr": s["resid_autocorr"],
                  "ref_coupling": s["k_ref"], "ref_coupling_err": s["k_ref_err"],
                  "laser_corrected": s["laser_corrected"]} for s in sweeps]
    results = {"run_dirs": [str(d) for d in run_dirs], "zero": zero, "retardance": ret,
               "dop_rms_assumed": dop_before, "dop_rms_new": dop_after,
               "analyzer_leak_bound": leak, "sweeps": per_sweep, "settings": vars(a)}
    clean = lambda o: (None if isinstance(o, float) and not np.isfinite(o) else o)
    text = json.dumps(results, indent=2, default=lambda o: clean(float(o))
                      if isinstance(o, (np.floating, np.integer)) else str(o))
    (out_dir / "characterization.json").write_text(text.replace("NaN", "null"))
    if a.save_figures or a.show_figures:
        figure(sweeps, zs, ds, z1, d1, out_dir, a)
    print(f"\nsaved in {out_dir}")
    return results


def figure(sweeps, zs, ds, z1, d1, out_dir, a):
    import matplotlib
    if not a.show_figures:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(s["label"] for s in sweeps))
    col = {lab: f"C{i % 10}" for i, lab in enumerate(labels)}
    idx = {id(s): i + 1 for i, s in enumerate(sweeps)}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    a0 = ax[0, 0]
    for s in zs:
        a0.plot(idx[id(s)], s["zero"], "o", color=col[s["label"]])
    a0.axhline(z1, color="k", lw=1)
    a0.axhline(a.qwp_zero_deg, color="gray", ls="--", lw=1, label="assumed")
    a0.set(title="QWP zero from the S3-term phase", xlabel="sweep", ylabel="zero (deg)")

    a1 = ax[0, 1]
    for s in ds:
        a1.plot(idx[id(s)], s["delta"], "o", color=col[s["label"]])
    a1.axhline(d1, color="k", lw=1)
    a1.axhline(waves_to_deg(a.qwp_retardance_waves), color="gray", ls="--", lw=1)
    a1.set(title="QWP retardance from DOP = 1", xlabel="sweep", ylabel="retardance (deg)")

    a2 = ax[1, 0]
    grid = np.linspace(d1 - 4, d1 + 4, 81)
    for s in ds:
        a2.plot(grid, [dop(s["fit"]["c"], g) - 1 for g in grid], color=col[s["label"]], lw=0.8)
    a2.axhline(0, color="k", lw=0.8)
    a2.axvline(d1, color="k", lw=1)
    a2.set(title="DOP - 1 vs assumed retardance (must cross 0 together)",
           xlabel="QWP retardance (deg)", ylabel="DOP - 1")

    a3 = ax[1, 1]
    for s in sweeps:
        a3.plot(idx[id(s)], s["dop_assumed"] - 1, "x", color=col[s["label"]], alpha=0.5)
        a3.plot(idx[id(s)], s["dop"] - 1, "o", color=col[s["label"]])
    a3.axhline(0, color="k", lw=0.8)
    a3.set(title="DOP - 1: assumed cal (x) vs new cal (o)", xlabel="sweep", ylabel="DOP - 1")

    handles = [plt.Line2D([], [], marker="o", ls="", color=col[l], label=l) for l in labels]
    fig.legend(handles=handles, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if a.save_figures:
        fig.savefig(out_dir / "characterization.png", dpi=130)
    if a.show_figures:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def quick_look(sw, a, span):
    """One line of live feedback, at the ASSUMED zero -- H's retardance barely
    depends on the zero, D/A's does; the analysis at the end corrects both."""
    P = corrected_power(sw) if has_reference(sw) else sw["power_W"]
    f = fit_sweep(P, sw["measured_deg"], a.qwp_zero_deg)
    S = stokes_from_coeffs(f["c"], waves_to_deg(a.qwp_retardance_waves), a.s3_sign)
    n = S / S[0]
    txt = f"S/S0 [{n[1]:+.3f} {n[2]:+.3f} {n[3]:+.3f}]  DOP {np.linalg.norm(n[1:]):.4f}"
    if abs(n[3]) > a.s3_min:
        txt += f"  | zero -> {a.qwp_zero_deg + zero_offset(f['c']):.2f} deg"
    d, slope = retardance_from_dop(f["c"], waves_to_deg(a.qwp_retardance_waves))
    if abs(slope) >= a.min_dop_slope and np.isfinite(d):
        txt += f"  | retardance -> {d:.2f} deg"
    print("    " + txt)


def measure(a) -> Path:
    from measure_mueller import Rig, SimRig, sweep

    span = float(a.qwp_sweep_deg)
    root = Path(a.out) if a.out else Path.home() / "Desktop"
    run = root / time.strftime("qwp_char_%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    print(f"\nrun folder : {run}")
    print(f"assumed    : QWP zero {a.qwp_zero_deg:g} deg, retardance "
          f"{a.qwp_retardance_waves:g} waves = {waves_to_deg(a.qwp_retardance_waves):.2f} deg")
    print(f"sweep      : {a.qwp_steps} steps over {span:g} deg, {a.repeats} repeats per state")

    if a.simulate:
        # the simulator's polarimeter is the TRUE one; the analysis starts
        # from the assumed values and must find these
        truth = copy.copy(a)
        truth.qwp_zero_deg = (a.sim_true_zero_deg if a.sim_true_zero_deg is not None
                              else a.qwp_zero_deg - 0.7)
        truth.qwp_retardance_waves = (a.sim_true_retardance_waves
                                      if a.sim_true_retardance_waves is not None else 0.2385)
        print(f"simulated  : TRUE zero {truth.qwp_zero_deg:g} deg, TRUE retardance "
              f"{waves_to_deg(truth.qwp_retardance_waves):.2f} deg")
        rig = SimRig(truth)
        a_sweep = copy.copy(a)
        a_sweep.qwp_settle = 0.0                 # the simulator keeps its own clock
    else:
        rig = Rig(a)
        a_sweep = a

    meta = {"kind": "qwp_characterization",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_file": str(Path(a.config).resolve()), "settings": vars(a),
            "calibration": {"qwp_zero_deg": a.qwp_zero_deg,
                            "qwp_retardance_waves": a.qwp_retardance_waves,
                            "s3_sign": a.s3_sign, "analyzer_deg": 0.0,
                            "wavelength_nm": a.wavelength_nm},
            "qwp_sweep_deg": span, "notes": a.notes, "states": []}

    with rig as r:
        def save():
            meta["hardware"] = r.describe()
            (run / "run.json").write_text(json.dumps(meta, indent=2, default=str))

        if a.zero_meter:
            what = ("both power meters (block the laser BEFORE the beamsplitter)"
                    if r.has_ref else "the power meter")
            input(f"\nBLOCK the beam to zero {what}, then press Enter...")
            r.zero_meter()
            input("UNBLOCK the beam, then press Enter...")
        r.set_sample(False)
        save()

        print("\nNO SAMPLE in the beam for the whole run. Suggested states "
              "(approximate is fine -- each is measured):\n"
              "  H    state QWP out, HWP turned for maximum signal -> retardance\n"
              "  R/L  state QWP in, as circular as is easy       -> zero\n"
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
                sw = sweep(r, a_sweep, f"{label} {k + 1}/{a.repeats}", span)
                name = f"state_{idx:02d}_rep_{k + 1:02d}.csv"
                write_sweep(run / name, sw["commanded_deg"], sw["measured_deg"],
                            sw["power_W"], sw["ref_W"], sw["t_s"])
                meta["states"][-1]["sweeps"].append(name)
                save()
                quick_look(sw, a, span)

    if not meta["states"]:
        raise SystemExit("nothing measured")
    print(f"\n{sum(len(s['sweeps']) for s in meta['states'])} sweeps saved in {run}")
    return run


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", default=[],
                    help="analyse these runs instead of measuring (characterization "
                         "or Mueller runs)")
    ap.add_argument("--repeats", type=int, default=10,
                    help="sweeps per state, nothing touched in between")
    ap.add_argument("--qwp-sweep-deg", type=float, default=360.0, dest="qwp_sweep_deg",
                    help="sweep span: 360 separates beam walk (as the Mueller sweeps do), 180 cannot")
    ap.add_argument("--s3-min", type=float, default=0.3, dest="s3_min",
                    help="|S3/S0| a sweep needs to calibrate the zero")
    ap.add_argument("--min-dop-slope", type=float, default=0.01, dest="min_dop_slope",
                    help="DOP change per deg of retardance a sweep needs to "
                         "calibrate the retardance (H 0.035, D/A ~0.02, V/R/L ~0)")
    ap.add_argument("--include-through-sample", type=str2bool, default=False,
                    dest="include_through_sample",
                    help="Mueller runs: also use the through-sample sweeps (only for a "
                         "non-depolarizing sample)")
    ap.add_argument("--fix-zero-deg", type=float_or_none, default=None, dest="fix_zero_deg",
                    help="use this zero instead of estimating it")
    ap.add_argument("--input-dop-min", type=float, default=0.998, dest="input_dop_min",
                    help="worst-case DOP of the input states, for the error budget. "
                         "About 1 - 2/ER of the polarizer: 0.998 for 1000:1, the "
                         "LPVISC guarantee at 510-520 nm")
    ap.add_argument("--show-figures", type=str2bool, default=True, dest="show_figures")
    ap.add_argument("--save-figures", type=str2bool, default=True, dest="save_figures")
    ap.add_argument("--sim-true-zero-deg", type=float_or_none, default=None,
                    dest="sim_true_zero_deg", help="simulator only; null = assumed - 0.7")
    ap.add_argument("--sim-true-retardance-waves", type=float_or_none, default=None,
                    dest="sim_true_retardance_waves", help="simulator only; null = 0.2385")

    # hardware + assumed calibration: same keys as measure_mueller.py
    ap.add_argument("--qwp-motor", default="28000005", dest="qwp_motor")
    ap.add_argument("--qwp-stage", default="none", dest="qwp_stage")
    ap.add_argument("--home", type=str2bool, default=True)
    ap.add_argument("--qwp-steps", type=int, default=100, dest="qwp_steps")
    ap.add_argument("--qwp-settle", type=float, default=0.3, dest="qwp_settle")
    ap.add_argument("--qwp-zero-deg", type=float, default=93.6, dest="qwp_zero_deg")
    ap.add_argument("--qwp-retardance-waves", type=float, default=0.25,
                    dest="qwp_retardance_waves")
    ap.add_argument("--s3-sign", type=int, default=1, choices=(1, -1), dest="s3_sign")
    ap.add_argument("--wavelength-nm", type=float, default=520.0, dest="wavelength_nm")
    ap.add_argument("--pm-serial", default=None, dest="pm_serial")
    ap.add_argument("--pm-average-count", type=int, default=300, dest="pm_average_count")
    ap.add_argument("--power-range-w", type=float_or_none, default=None, dest="power_range_w")
    ap.add_argument("--ref-pm-serial", default=None, dest="ref_pm_serial")
    ap.add_argument("--ref-power-range-w", type=float_or_none, default=None,
                    dest="ref_power_range_w")
    ap.add_argument("--zero-meter", type=str2bool, default=True, dest="zero_meter")
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-state-error-deg", type=float, default=3.0, dest="sim_state_error_deg")
    ap.add_argument("--sim-noise-rel", type=float, default=0.001, dest="sim_noise_rel")
    ap.add_argument("--sim-laser-drift-rel", type=float, default=0.01, dest="sim_laser_drift_rel")
    ap.add_argument("--sim-laser-noise-rel", type=float, default=0.003, dest="sim_laser_noise_rel")

    # measure_mueller.json first (hardware, assumed calibration), then this
    # script's own JSON, then the command line
    shared = HERE / "measure_mueller.json"
    if shared.is_file():
        known = {x.dest for x in ap._actions}
        data = json.loads(shared.read_text())
        ap.set_defaults(**{k: v for k, v in data.items() if k in known})
        print(f"hardware + assumed calibration: {shared}")
    a = parse_with_config(ap, argv, HERE / "QWP_analyzer_characterization.json")

    if a.run_dirs:
        dirs = [Path(d) for d in a.run_dirs]
        out = (dirs[0] / "characterization" if len(dirs) == 1 else
               (Path(a.out) if a.out else Path.home() / "Desktop")
               / time.strftime("qwp_char_analysis_%Y%m%d_%H%M%S"))
    else:
        run = measure(a)
        dirs, out = [run], run / "characterization"
    analyse(dirs, a, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
