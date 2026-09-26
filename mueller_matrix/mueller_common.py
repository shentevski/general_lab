"""Shared physics and file handling for measure_mueller.py and analyze_mueller.py.

Kept in one place so the measurement's live feedback and the analysis use
the SAME Stokes extraction -- two copies of this maths would be two things
that can silently disagree.

The polarimeter (PSA) is a rotating QWP in front of a fixed analyzer, read
by a power meter. The analyzer axis DEFINES H, so the analyzer angle is 0
by construction; qwp_zero_deg is the stage reading at which the QWP is
aligned with it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from polarization_toolkit.analysis.extract import design_matrix


# --------------------------------------------------------------------------- #
# config: JSON read by default, command line overrides
# --------------------------------------------------------------------------- #


def parse_with_config(ap, argv, default_cfg: Path):
    """Two-pass parse: the JSON supplies the defaults, flags override it.

    Unknown keys are rejected with the valid list, because a silently
    ignored typo would run the measurement with a default value.
    """
    ap.add_argument("--config", metavar="JSON", default=str(default_cfg),
                    help=f"settings file (default: {default_cfg.name} next to "
                         f"this script, read automatically)")
    pre, _ = ap.parse_known_args(argv)
    path = Path(pre.config)
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} is not valid JSON: {e}")
        valid = {a.dest for a in ap._actions} - {"help", "config"}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise SystemExit(f"{path}: unknown key(s) {unknown}.\n"
                             f"valid keys: {sorted(valid)}")
        ap.set_defaults(**data)
        print(f"config: {path}")
    elif path != default_cfg:
        raise SystemExit(f"config file not found: {path}")
    else:
        print(f"config: none found at {path} -- using built-in defaults")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# Stokes extraction for ANY QWP retardance
# --------------------------------------------------------------------------- #


def extract_stokes(power, angles_deg, *, qwp_zero_deg, retardance_deg,
                   s3_sign=1):
    """Least-squares Stokes vector from one QWP sweep.

    The intensity behind a rotating retarder of retardance delta and an
    analyzer at 0 is linear in five coefficients, and for general delta

        c0 = (S0 + S1 (1+cos d)/2) / 2      c1 = S3 sin d / 2
        c3 = S2 (1-cos d) / 4               c4 = S1 (1-cos d) / 4

    Inverting that removes the QWP's retardance error EXACTLY instead of
    assuming a perfect quarter-wave. At delta = 90 deg it reduces to the
    toolkit's own projection (checked in _self_check).

    Returns dict: S (4,), coeffs (5,), model (K,), residual_rms.
    """
    y = np.asarray(power, float)
    theta = np.deg2rad(np.asarray(angles_deg, float) - qwp_zero_deg)
    A = design_matrix(theta)
    c = np.linalg.pinv(A) @ y
    cd, sd = np.cos(np.deg2rad(retardance_deg)), np.sin(np.deg2rad(retardance_deg))
    S1 = 4 * c[4] / (1 - cd)
    S2 = 4 * c[3] / (1 - cd)
    S3 = s3_sign * 2 * c[1] / sd
    S0 = 2 * c[0] - S1 * (1 + cd) / 2
    model = A @ c
    return {"S": np.array([S0, S1, S2, S3]), "coeffs": c, "model": model,
            "residual_rms": float(np.sqrt(np.mean((y - model) ** 2)))}


# --------------------------------------------------------------------------- #
# retarder split and Cloude realizability
# --------------------------------------------------------------------------- #


def split_retarder(M_R):
    """Lu-Chipman retarder -> (linear retardance, fast axis, optical rotation),
    radians, with M_R = M_linear_retarder @ M_rotator.

    Verified exact against constructed matrices; the optical-rotation sign
    agrees with the toolkit's differential decomposition.
    """
    from polarization_toolkit.analysis.mueller import lu_chipman
    from polarization_toolkit.hardware.simulated import rotator_mueller

    a = M_R[1, 1] + M_R[2, 2]
    b = M_R[2, 1] - M_R[1, 2]
    delta = float(np.arccos(np.clip(np.hypot(a, b) - 1.0, -1.0, 1.0)))
    psi = float(0.5 * np.arctan2(b, a))
    M_lin = M_R @ rotator_mueller(psi).T
    ax = lu_chipman(M_lin).retardance_axis
    theta = float(0.5 * np.arctan2(ax[1], ax[0])) if delta > 1e-9 else float("nan")
    return delta, theta, psi


_PAULI = [np.eye(2, dtype=complex),
          np.array([[1, 0], [0, -1]], complex),
          np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]], complex)]
_BASIS = [[np.kron(_PAULI[i], _PAULI[j].conj()) for j in range(4)] for i in range(4)]


def coherency(M):
    """Cloude coherency matrix. MUST use conj(sigma_j): the unconjugated form
    gives air a negative eigenvalue and flags perfect data as unphysical."""
    return 0.25 * sum(M[i, j] * _BASIS[i][j] for i in range(4) for j in range(4))


def mueller_from_coherency(H):
    return np.array([[np.trace(H @ _BASIS[i][j]).real for j in range(4)]
                     for i in range(4)])


def cloude(M):
    """Physical realizability of a measured M.

    A physical Mueller matrix has a coherency matrix with NO negative
    eigenvalues; noise or systematic error can push one below zero. Also
    returns the Cloude entropy (0 = non-depolarizing, 1 = total depolarizer)
    and the nearest physical M, obtained by clipping negative eigenvalues.
    """
    H = coherency(M)
    w, V = np.linalg.eigh(H)
    lam = np.sort(w)[::-1] / max(np.trace(H).real, 1e-15)
    pos = np.clip(lam, 0, None)
    pos = pos / pos.sum() if pos.sum() > 0 else pos
    nz = pos[pos > 1e-15]
    entropy = max(0.0, float(-(nz * np.log(nz) / np.log(4)).sum()))
    M_phys = mueller_from_coherency((V * np.clip(w, 0, None)) @ V.conj().T)
    return {"eigenvalues": lam, "physical": bool(lam.min() > -1e-3),
            "entropy": entropy, "M_physical": M_phys,
            "filter_change": float(np.abs(M_phys - M).max() / max(M[0, 0], 1e-15))}


# --------------------------------------------------------------------------- #
# sweep files
# --------------------------------------------------------------------------- #

SWEEP_COLUMNS = ("commanded_deg", "measured_deg", "power_W", "t_s")


def write_sweep(path: Path, commanded, measured, power, t):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SWEEP_COLUMNS)
        for row in zip(commanded, measured, power, t):
            w.writerow([f"{v:.9g}" for v in row])


def read_sweep(path: Path) -> dict:
    d = np.genfromtxt(path, delimiter=",", names=True)
    return {k: np.atleast_1d(d[k]).astype(float) for k in SWEEP_COLUMNS}


def waves_to_deg(waves: float) -> float:
    return float(waves) * 360.0


def _self_check():
    """At 90 deg the general extraction must equal the toolkit's."""
    from polarization_toolkit.analysis.extract import stokes_from_stack
    rng = np.random.default_rng(3)
    ang = np.linspace(0, 180, 100, endpoint=False)
    y = rng.normal(1.0, 0.2, 100)
    mine = extract_stokes(y, ang, qwp_zero_deg=12.0, retardance_deg=90.0)["S"]
    tk = stokes_from_stack(y, ang, qwp_zero_deg=12.0).stokes
    return float(np.abs(mine - tk).max())
