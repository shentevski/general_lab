"""Shared maths and file handling for the spectral (LED + CCT10) Mueller scripts.

The physics is the SAME as in ../mueller_matrix and is imported from there,
not copied -- two copies of the maths would be two things that can silently
disagree: the Stokes inversion for a QWP of any retardance
(stokes_from_coeffs), the retarder split, the Cloude check and the config
handling. What is new here is only that a sweep is now a spectrum at every
QWP angle, so every step runs once per wavelength bin:

  * the fit is one matrix product for all bins at once,
  * the QWP retardance is a curve delta(lambda): from a file (the Thorlabs
    data, or the output of QWP_analyzer_characterization_poly.py), or one
    number for the whole band,
  * the QWP zero is one number -- a zero-order or quartz/MgF2 achromatic
    plate has an axis that does not move with wavelength -- or, with
    zero_source "file", a curve too (superachromatic plates).

The detector is a spectrometer: raw counts, a dark measured with its shutter
before and after every sweep, and nothing depends on its spectral response
(every number is a ratio at the same pixel).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_MM = Path(__file__).resolve().parent.parent / "mueller_matrix"
if str(_MM) not in sys.path:
    sys.path.insert(0, str(_MM))

# cloude and parse_with_config are re-exported for the other poly scripts
from mueller_common import (cloude, full_turn, parse_with_config,  # noqa: E402,F401
                            stokes_from_coeffs, waves_to_deg)
from polarization_toolkit.analysis.extract import design_matrix    # noqa: E402


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y", "on")


def float_or_none(s):
    return None if str(s).lower() in ("", "none", "null") else float(s)


# --------------------------------------------------------------------------- #
# fit of one sweep, every wavelength bin at once
# --------------------------------------------------------------------------- #


def _design(theta, walk):
    A = design_matrix(theta)
    if walk:
        A = np.column_stack([A] + [f(k * theta) for k in (1, 3, 5)
                                   for f in (np.cos, np.sin)])
    return A


def fit_coeffs(power, angles_deg, zero_deg, walk=None):
    """Least-squares fit of every bin of one sweep.

    power (K, N): K QWP angles x N wavelength bins. zero_deg: one number, or
    one per bin. The model is extract_stokes' in ../mueller_matrix: the five
    polarization terms (0, 2t, 4t) and, over a full turn, the beam-walk terms
    (1t, 3t, 5t) that polarization never produces (walk=None: automatically
    when the sweep covers a full turn).

    Returns dict: c (5, N), walk (6 or 0, N), model (K, N), resid_rms (N,).
    """
    y = np.asarray(power, float)
    if y.ndim == 1:
        y = y[:, None]
    ang = np.asarray(angles_deg, float)
    if walk is None:
        walk = full_turn(ang)
    z = np.broadcast_to(np.asarray(zero_deg, float), (y.shape[1],))
    coef = np.empty((11 if walk else 5, y.shape[1]))
    model = np.empty_like(y)
    for zv in np.unique(z):                      # one design matrix per zero
        cols = z == zv
        A = _design(np.deg2rad(ang - zv), walk)
        coef[:, cols] = np.linalg.pinv(A) @ y[:, cols]
        model[:, cols] = A @ coef[:, cols]
    return {"c": coef[:5], "walk": coef[5:], "model": model,
            "resid_rms": np.sqrt(np.mean((y - model) ** 2, axis=0))}


def extract_stokes(power, angles_deg, *, zero_deg, retardance_deg, s3_sign=1,
                   walk=None):
    """Stokes vector of every bin: fit_coeffs + stokes_from_coeffs.
    retardance_deg: one per bin. Adds S (4, N) to fit_coeffs' dict."""
    f = fit_coeffs(power, angles_deg, zero_deg, walk)
    f["S"] = stokes_from_coeffs(f["c"], retardance_deg, s3_sign)
    return f


def dop(S):
    """Degree of polarization of (4, ...) Stokes arrays."""
    return np.linalg.norm(S[1:], axis=0) / S[0]


def coverage(S_in):
    """Rank and condition number of the input states, per bin.
    S_in (N, 4, n) -> rank (N,), cond (N,)."""
    Sn = S_in / S_in[:, :1, :]
    sv = np.linalg.svd(Sn, compute_uv=False)
    rank = (sv > 1e-2 * sv[:, :1]).sum(axis=1)
    cond = (sv[:, 0] / sv[:, -1] if S_in.shape[2] >= 4
            else np.full(S_in.shape[0], np.inf))
    return rank, cond


def psa_blind(retardance_deg, limit=0.2):
    """Bins where the rotating QWP hardly sees linear or circular light:
    the linear terms scale with (1 - cos d)/2, the circular one with sin d.
    A multi-order plate crosses 0 and 180 deg and is blind there."""
    d = np.deg2rad(retardance_deg)
    return ((1 - np.cos(d)) / 2 < limit) | (np.abs(np.sin(d)) < limit)


# --------------------------------------------------------------------------- #
# wavelength bins
# --------------------------------------------------------------------------- #


def bin_edges(wl_min, wl_max, width):
    n = max(int(round((wl_max - wl_min) / width)), 1)
    return wl_min + (wl_max - wl_min) / n * np.arange(n + 1)


def bin_matrix(wavelengths, edges):
    """(P, N) 0/1 matrix: counts @ W sums the pixels of every bin."""
    idx = np.digitize(wavelengths, edges) - 1
    N = len(edges) - 1
    W = (idx[:, None] == np.arange(N)[None, :]).astype(float)
    empty = np.where(W.sum(0) == 0)[0]
    if empty.size:
        raise SystemExit(f"wavelength bins with no spectrometer pixel: "
                         f"{0.5 * (edges[empty] + edges[empty + 1])} nm -- "
                         f"make the bins wider or the range narrower")
    return W


def centres(edges):
    return 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------------- #
# sweep files: one .npz per sweep, raw counts + the darks around it
# --------------------------------------------------------------------------- #

# counts (K, P): raw counts, no dark subtracted, amplitude correction off.
# dark_before / dark_after (P,): shutter-closed spectra at t_dark[0] / [1].
# ref_W (K,): reference meter (LED monitor), NaN without one.
SWEEP_KEYS = ("commanded_deg", "measured_deg", "counts", "ref_W", "t_s",
              "wavelengths", "dark_before", "dark_after", "t_dark",
              "exposure_ms", "hw_average")


def write_sweep(path: Path, sw: dict):
    data = {k: np.asarray(sw[k]) for k in SWEEP_KEYS}
    data["counts"] = data["counts"].astype(np.float32)   # exact to < 0.01 count
    np.savez_compressed(path, **data)


def read_sweep(path: Path) -> dict:
    with np.load(path) as d:
        return {k: d[k].astype(float) for k in SWEEP_KEYS}


def signal_counts(sw):
    """Counts minus the dark, interpolated in time between the shutter darks
    taken before and after the sweep (the dark drifts with the sensor
    temperature). A leftover offset looks exactly like lost polarization."""
    tb, ta = sw["t_dark"]
    f = (np.clip((sw["t_s"] - tb) / (ta - tb), 0, 1) if ta > tb
         else np.full(sw["t_s"].shape, 0.5))
    dark = (1 - f)[:, None] * sw["dark_before"] + f[:, None] * sw["dark_after"]
    return sw["counts"] - dark


def clipped(counts, full_scale):
    """Spectra (rows) that reached full scale: at >= 98 % of it, or with a
    plateau of >= 3 pixels exactly at the spectrum maximum while that maximum
    is in the upper half of the range (what a clipped detector gives even if
    full_scale_counts is set wrong; a dim, noisy spectrum can repeat a value
    by chance, so a plateau low down is not clipping)."""
    c = np.atleast_2d(counts)
    peak = c.max(axis=1)
    at_max = (c == peak[:, None]).sum(axis=1) >= 3
    return (peak >= 0.98 * full_scale) | (at_max & (peak > 0.5 * full_scale))


def has_reference(sw) -> bool:
    r = sw["ref_W"]
    return bool(np.isfinite(r).all() and (r > 0).all())


def reference_factor(sw, level=None):
    """(K,) factor that divides the LED fluctuations out: level / ref_W.
    level = the run's mean reference, so the result stays in counts at the
    average LED power and M00 stays a transmittance. Ones without a reference."""
    if not has_reference(sw):
        return np.ones(sw["ref_W"].shape)
    r = sw["ref_W"]
    return (np.mean(r) if level is None else level) / r


def angles_of(sw):
    a = sw["measured_deg"]
    return np.where(np.isfinite(a), a, sw["commanded_deg"])


def binned_power(sw, W, level=None, use_reference=True):
    """Dark-subtracted, laser-(LED-)corrected counts summed per bin: (K, N)."""
    p = signal_counts(sw) @ W
    if use_reference:
        p = p * reference_factor(sw, level)[:, None]
    return p


# --------------------------------------------------------------------------- #
# calibration curves: QWP retardance (waves) and zero (deg) vs wavelength
# --------------------------------------------------------------------------- #


def _num(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def read_curve(path: Path) -> dict:
    """A calibration file: comma, tab or semicolon separated.

    Columns are found by name in the header row (the first row with a cell
    containing "wavelength"): wavelength (nm), retardance (waves) -- the
    Thorlabs retardance data saved from Excel as CSV works as it is, product
    text in the other columns and all -- and, when present, zero_deg and
    retardance_err_waves (written by the characterization). Without a header
    the first two columns are wavelength and retardance. Empty cells are
    missing values; rows without a wavelength and a retardance are skipped.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        table = [[x.strip() for x in row] for row in csv.reader(fh, dialect)]
    h = next((i for i, row in enumerate(table)
              if any("wavelength" in x.lower() for x in row)), None)
    header = [x.lower() for x in table[h]] if h is not None else []

    def col(pred, default=None):
        return next((i for i, name in enumerate(header) if pred(name)), default)

    iw = col(lambda n: "wavelength" in n, 0)
    ir = col(lambda n: n == "retardance_waves" or
             ("retardance" in n and "err" not in n and "raw" not in n
              and "stat" not in n and "states" not in n), 1)
    iz = col(lambda n: n == "zero_deg")
    ie = col(lambda n: n == "retardance_err_waves")
    width = max([iw, ir] + [i for i in (iz, ie) if i is not None]) + 1
    rows = [[_num(x) for x in (row + [""] * width)[:width]]
            for row in table[(h + 1 if h is not None else 0):]]
    data = np.array([r for r in rows if np.isfinite(r[iw]) and np.isfinite(r[ir])])
    if data.size == 0:
        raise SystemExit(f"{path}: no rows with a wavelength and a retardance")
    data = data[np.argsort(data[:, iw])]
    out = {"wavelength_nm": data[:, iw], "retardance_waves": data[:, ir],
           "zero_deg": data[:, iz] if iz is not None else None,
           "retardance_err_waves": data[:, ie] if ie is not None else None,
           "path": str(path)}
    if np.nanmax(out["retardance_waves"]) > 50:
        raise SystemExit(f"{path}: retardance values up to "
                         f"{np.nanmax(out['retardance_waves']):g} -- expected WAVES "
                         f"(0.25 = quarter wave); nm or degrees?")
    if not (300 < np.median(out["wavelength_nm"]) < 1200):
        raise SystemExit(f"{path}: wavelengths should be in nm")
    return out


def resolve_path(p, base: Path):
    if p is None or str(p).lower() in ("", "none", "null"):
        return None
    p = Path(p).expanduser()
    return p if p.is_absolute() else (base / p)


def calibration_at(cal: dict, wl_nm, base: Path, strict=True) -> dict:
    """The calibration at the given wavelengths.

    cal: {"qwp_zero_deg", "qwp_retardance_waves", "calibration_file",
          "zero_source", "s3_sign"} -- as recorded in run.json.
    strict: refuse wavelengths outside the file (the analysis); otherwise
    hold the end values (the simulator, and a starting guess).

    Returns retardance_deg (N,), zero_deg (number, or (N,) for zero_source
    "file"), retardance_err_waves (N,) or None, and a one-line description.
    """
    wl = np.asarray(wl_nm, float)
    path = resolve_path(cal.get("calibration_file"), base)
    zero = float(cal["qwp_zero_deg"])
    if path is None:
        if cal.get("zero_source", "constant") == "file":
            raise SystemExit("zero_source is 'file' but there is no calibration_file")
        w = float(cal["qwp_retardance_waves"])
        return {"retardance_deg": np.full(wl.shape, waves_to_deg(w)), "zero_deg": zero,
                "retardance_err_waves": None,
                "source": f"constant {w:g} waves at every wavelength, zero {zero:g} deg"}
    if not path.is_file():
        raise SystemExit(f"calibration file not found: {path}")
    c = read_curve(path)
    lo, hi = c["wavelength_nm"][0], c["wavelength_nm"][-1]
    if strict and (wl.min() < lo - 1e-6 or wl.max() > hi + 1e-6):
        raise SystemExit(f"{path.name} covers {lo:g}-{hi:g} nm but "
                         f"{wl.min():.1f}-{wl.max():.1f} nm is asked for: narrow "
                         f"wl_min_nm / wl_max_nm")
    ret = np.interp(wl, c["wavelength_nm"], c["retardance_waves"])
    err = (np.interp(wl, c["wavelength_nm"], c["retardance_err_waves"])
           if c["retardance_err_waves"] is not None else None)
    src = f"{path.name} ({lo:g}-{hi:g} nm)"
    if cal.get("zero_source", "constant") == "file":
        if c["zero_deg"] is None:
            raise SystemExit(f"zero_source is 'file' but {path.name} has no zero_deg column")
        zero = np.interp(wl, c["wavelength_nm"], c["zero_deg"])
        src += ", zero from the file"
    else:
        src += f", zero {zero:g} deg"
    return {"retardance_deg": waves_to_deg(1.0) * ret, "zero_deg": zero,
            "retardance_err_waves": err, "source": src}


def write_curve(path: Path, columns: dict):
    """CSV with a header, one row per wavelength; NaN written as empty."""
    keys = list(columns)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for row in zip(*(np.asarray(columns[k], float) for k in keys)):
            w.writerow(["" if not np.isfinite(v) else f"{v:.7g}" for v in row])


def zero_offset(c):
    """How far the true QWP zero is from the one used in the fit (deg), per bin.

    The S3 term is K sin(2(theta - eps)) = K cos2eps sin2theta - K sin2eps cos2theta,
    so c1 = K cos 2eps, c2 = -K sin 2eps. K cancels: no retardance, no power,
    no degree of polarization."""
    s = np.where(c[1] >= 0, 1.0, -1.0)
    return -0.5 * np.degrees(np.arctan2(s * c[2], s * c[1]))
