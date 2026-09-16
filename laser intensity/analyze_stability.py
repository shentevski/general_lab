#!/usr/bin/env python3
"""
analyze_stability.py
====================
Noise and drift analysis for a run recorded by ``measure_intensity.py``, aimed
at one question: **how well do you actually know the power?**

    python analyze_stability.py                  # newest run on the Desktop
    python analyze_stability.py path/to/run.csv

Writes three files next to the data:

    <stem>_report.txt      the whole analysis in words and numbers
    <stem>_stability.png   six panels: time series, histogram, Allan, PSD, QQ, temperature
    <stem>_metrics.json    every number above, for scripting

What it separates, and why each part is here:

* **Drift from noise.** A least-squares line through P(t) is the drift; the
  residuals are the noise. Quoting one rms for both is what makes a "1 % stable"
  laser turn out to be 0.05 % noisy and 1 %/hour drifting -- two different
  problems with two different fixes.
* **The laser from the meter.** The beam-blocked segment is the detector's own
  noise floor. If the residual noise is barely above it, you are characterising
  the power meter, not the laser.
* **Allan deviation.** Straight statistics assume independent samples, and
  power readings are not. The overlapping Allan deviation says how much
  averaging actually buys you: it falls as 1/sqrt(tau) while white noise
  dominates, flattens at the flicker floor, and turns back up where drift takes
  over. Its minimum is the best you can do, and the tau at that minimum is how
  long you should average each point.
* **Correlation time.** sigma/sqrt(N) is optimistic when samples are
  correlated. The integrated autocorrelation time gives the effective N.
* **Spectrum.** Where the noise lives -- 60/120 Hz pickup, a chopper, a fan,
  1/f. You cannot fix what you cannot name.
* **Temperature.** If head temperature was logged, drift is regressed against
  it. Drift that tracks temperature is usually the room, not the laser.
* **An uncertainty budget.** Type A from the Allan deviation, Type B from the
  datasheet numbers you pass in, combined per the GUM. Every Type B term
  defaults to ZERO, so out of the box this reports the statistical spread of
  your data and says so -- that is repeatability, not accuracy. Pass the
  figures off your own spec sheet (``--help`` lists the published PM100D /
  S130VC ones) to turn it into a real budget.

Everything printed here is computed from the recorded samples and the
instrument's own reported settings. Nothing rests on an assumed integration
time, bandwidth or per-sample duration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

# PM100D manual, utility software section: "a rate of 3000 averages the
# incoming measurement values for approx. 1 second". (Its SCPI section says
# 3 ms per sample instead; two of the three statements in the manual agree on
# 3000/s, so that is what is used here.)
HARDWARE_SAMPLES_PER_S = 3000.0
ELEMENTARY_CHARGE = 1.602176634e-19

# np.trapz was renamed in numpy 2.0; the Windows PC may have either.
_integrate = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------- loading


class Run:
    """One acquisition: timebase, power, optional temperature, metadata."""

    def __init__(self, path: Path, t, p, temp, meta, dark):
        self.path = path
        self.t = t
        self.p = p
        self.temp = temp
        self.meta = meta
        self.dark = dark          # array of beam-blocked powers, or None
        self.residual = None      # filled in by analyse(), reused by the figure


def read_csv(path: Path):
    """Return (elapsed_s, power_W, head_temp_C, header dict) from one CSV."""
    header = {}
    comment_lines = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("#"):
            break
        comment_lines += 1
        if ":" in line:
            key, _, value = line.lstrip("# ").partition(":")
            header[key.strip()] = value.strip()

    # skip_header, not comments='#': with names=True genfromtxt takes the field
    # names from the very first line it sees, comment marker or not, and our
    # metadata comments sit above the real header row.
    table = np.genfromtxt(
        path, delimiter=",", skip_header=comment_lines, names=True, dtype=float,
    )
    if table.size == 0:
        raise SystemExit(f"{path.name} has no data rows.")

    names = table.dtype.names
    power_column = "power_W" if "power_W" in names else names[1]
    t = np.atleast_1d(table["elapsed_s"] if "elapsed_s" in names else table[names[0]])
    p = np.atleast_1d(table[power_column])
    temp = np.atleast_1d(table["head_temp_C"]) if "head_temp_C" in names else None
    return t, p, temp, header


def newest_run(directory: Path) -> Path:
    candidates = [
        path for path in directory.glob("power_*.csv")
        if not path.name.endswith("_dark.csv")
    ]
    if not candidates:
        raise SystemExit(
            f"No runs found in {directory}. Record one first:\n"
            f"  python measure_intensity.py --wavelength_nm 450"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_run(path: Path) -> Run:
    t, p, temp, header = read_csv(path)

    meta_path = path.with_name(path.with_suffix("").name + "_meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # CSV comments are the fallback when the sidecar went missing.
    for key, value in header.items():
        meta.setdefault(key, value)

    dark = None
    dark_path = path.with_name(path.with_suffix("").name + "_dark.csv")
    if dark_path.exists():
        dark = read_csv(dark_path)[1]

    return Run(path, t, p, temp, meta, dark)


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------- maths


def linear_fit(x, y):
    """Least-squares slope, intercept and R^2."""
    slope, intercept = np.polyfit(x, y, 1)
    model = slope * x + intercept
    residual = y - model
    total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - np.sum(residual ** 2) / total if total > 0 else float("nan")
    return float(slope), float(intercept), float(r_squared), residual


def overlapping_adev(y, tau0, points=50, min_estimates=8):
    """Overlapping Allan deviation of a frequency-like series.

    ``y`` is the measurement itself (power), not a phase, so the standard
    frequency-data estimator applies. Working from the cumulative sum turns the
    double sum into three slices, which keeps long runs fast.

    Returns (tau, adev, relative 1-sigma error bars).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 16:
        return np.array([]), np.array([]), np.array([])

    # x[k] is the "phase": the integral of y up to sample k.
    x = np.concatenate(([0.0], np.cumsum(y))) * tau0

    max_m = max(1, n // min_estimates)
    ms = np.unique(np.round(np.logspace(0, math.log10(max_m), points)).astype(int))

    taus, devs, errors = [], [], []
    for m in ms:
        if n - 2 * m + 1 < 2:
            continue
        differences = x[2 * m:] - 2 * x[m:-m] + x[:-2 * m]
        tau = m * tau0
        variance = np.sum(differences ** 2) / (2 * tau ** 2 * differences.size)
        taus.append(tau)
        devs.append(math.sqrt(variance))
        # Rough confidence: the usual 1/sqrt(2(N/m - 1)) rule of thumb.
        errors.append(1.0 / math.sqrt(2 * max(n / m - 1, 1)))

    return np.array(taus), np.array(devs), np.array(errors)


def adev_at(taus, devs, tau, slack=1.01):
    """Log-log interpolation of the Allan curve; NaN outside the measured range."""
    if taus.size == 0 or tau < taus[0] / slack or tau > taus[-1] * slack:
        return float("nan")
    tau = min(max(tau, taus[0]), taus[-1])
    return float(np.exp(np.interp(math.log(tau), np.log(taus), np.log(devs))))


def autocorrelation(d):
    """Normalised autocorrelation of a detrended series, via FFT."""
    d = np.asarray(d, dtype=float)
    d = d - d.mean()
    n = d.size
    size = 1 << int(2 * n - 1).bit_length()
    spectrum = np.fft.rfft(d, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    if correlation[0] == 0:
        return np.ones(n)
    return correlation / correlation[0]


def correlation_time(d, c=6.0):
    """Integrated autocorrelation time, in samples, with Sokal windowing.

    Samples that far apart are effectively independent; N/this is the effective
    number of independent measurements, and the honest divisor for sigma/sqrt(N).
    """
    rho = autocorrelation(d)
    running = 1.0
    for k in range(1, rho.size):
        if rho[k] <= 0:                       # first zero crossing: stop summing
            break
        running += 2.0 * rho[k]
        if k >= c * running:                  # Sokal's automatic window
            break
    return max(float(running), 1.0)


def welch_psd(x, fs, nperseg):
    """One-sided power spectral density, Hann window, 50 % overlap.

    Each segment is linearly detrended first, so drift does not leak across the
    whole spectrum. Scaled as a density, so integrating it returns the variance.
    """
    x = np.asarray(x, dtype=float)
    nperseg = int(min(nperseg, x.size))
    if nperseg < 16:
        return np.array([]), np.array([])

    window = np.hanning(nperseg + 1)[:-1]     # periodic form
    step = nperseg // 2
    scale = 1.0 / (fs * np.sum(window ** 2))
    index = np.arange(nperseg)

    total = None
    count = 0
    for start in range(0, x.size - nperseg + 1, step):
        segment = x[start:start + nperseg]
        slope, intercept = np.polyfit(index, segment, 1)
        segment = (segment - (slope * index + intercept)) * window
        spectrum = np.abs(np.fft.rfft(segment)) ** 2 * scale
        spectrum[1:-1] *= 2.0                 # fold the negative frequencies in
        total = spectrum if total is None else total + spectrum
        count += 1

    if not count:
        return np.array([]), np.array([])
    return np.fft.rfftfreq(nperseg, 1.0 / fs), total / count


def local_median(y, width):
    """Sliding-median baseline, used to find spectral lines above the grass."""
    n = y.size
    half = max(1, width // 2)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(y[max(0, i - half):min(n, i + half + 1)])
    return out


def spectral_peaks(freqs, psd, count=5, threshold=6.0):
    """Narrow lines standing well above the local noise floor."""
    if freqs.size < 32:
        return []
    baseline = local_median(psd, max(16, psd.size // 40))
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(baseline > 0, psd / baseline, 0.0)

    peaks = []
    for i in range(2, psd.size - 1):
        if excess[i] > threshold and psd[i] > psd[i - 1] and psd[i] >= psd[i + 1]:
            peaks.append((freqs[i], psd[i], excess[i]))
    peaks.sort(key=lambda item: item[2], reverse=True)
    return peaks[:count]


def band_rms(freqs, psd, low, high):
    """Fractional rms contributed by one frequency band."""
    mask = (freqs >= low) & (freqs < high) & np.isfinite(psd)
    if mask.sum() < 2:
        return float("nan")
    return float(math.sqrt(max(_integrate(psd[mask], freqs[mask]), 0.0)))


def moments(d):
    """Skewness and excess kurtosis -- is the noise actually Gaussian?"""
    sigma = np.std(d)
    if sigma == 0:
        return 0.0, 0.0
    z = (d - np.mean(d)) / sigma
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


# ---------------------------------------------------------------- the budget


DIVISORS = {"rect": math.sqrt(3.0), "normal": 1.0, "k2": 2.0}


def build_budget(args, analysis, meta):
    """Type B terms, each converted from a datasheet tolerance to 1 sigma.

    Datasheets quote bounds ("+/-3 %"), not standard deviations. The GUM way to
    use one is to assume a distribution inside the bound and divide: sqrt(3) for
    a rectangular one (the default, and the conservative reading of a spec),
    1 if the number really is a standard uncertainty, 2 if it is an expanded
    k=2 uncertainty.
    """
    divisor = DIVISORS[args.type_b_distribution]
    entries = []

    def add(name, tolerance_pct, note, cancels_in_ratio=False, from_data=False):
        if not tolerance_pct:
            return
        entries.append({
            "name": name,
            "tolerance_pct": float(tolerance_pct),
            "standard_pct": float(tolerance_pct) / divisor,
            "cancels_in_ratio": cancels_in_ratio,
            "from_data": from_data,
            "note": note,
        })

    add("Sensor calibration", args.cal_tolerance_pct,
        "absolute responsivity of this head at this wavelength; "
        "cancels in ratios taken with the same head and wavelength",
        cancels_in_ratio=True)
    add("Console readout / linearity", args.console_tolerance_pct,
        "meter electronics, ADC and range linearity")
    add("Detector nonlinearity", args.linearity_pct,
        "extra term if you are comparing very different power levels")

    wavelength_term = args.wavelength_error_nm * args.responsivity_slope_pct_per_nm
    add("Wavelength setting", wavelength_term,
        f"{args.wavelength_error_nm:g} nm uncertain x "
        f"{args.responsivity_slope_pct_per_nm:g} %/nm responsivity slope; "
        "matters near the band edges, negligible mid-band",
        cancels_in_ratio=True)

    excursion = args.temp_excursion_c
    if excursion is None:
        excursion = analysis["temperature"].get("excursion_c") or 0.0
    temperature_term = args.temp_coeff_pct_per_c * excursion
    add("Head temperature", temperature_term,
        f"{args.temp_coeff_pct_per_c:g} %/C x {excursion:.2f} C observed swing")

    # The residual dark offset is a real bias on the reading, sized from the
    # blocked-beam segment rather than assumed.
    dark = analysis["dark"]
    if dark["available"] and analysis["level"]["mean_w"]:
        offset_pct = abs(dark["mean_w"]) / analysis["level"]["mean_w"] * 100.0
        add("Residual dark offset", offset_pct,
            "measured with the beam blocked, after zeroing", from_data=True)

    combined_b = math.sqrt(sum(entry["standard_pct"] ** 2 for entry in entries))
    return entries, combined_b


def type_a_at(analysis, tau_s):
    """Statistical uncertainty of a tau-long average, from the Allan curve.

    For white noise the Allan deviation equals the standard deviation of a
    tau-long mean, which is exactly what is wanted here -- and unlike
    sigma/sqrt(N) it stays honest when the noise is not white.
    """
    taus = np.array(analysis["allan"]["tau_s"])
    devs = np.array(analysis["allan"]["adev_frac"])
    return adev_at(taus, devs, tau_s) * 100.0


# -------------------------------------------------------------- the analysis


def analyse(run: Run, args) -> dict:
    t, p = run.t, run.p
    n = p.size
    meta = run.meta

    if args.skip_s > 0 or args.max_s is not None:
        end = args.max_s if args.max_s is not None else t[-1]
        keep = (t >= args.skip_s) & (t <= end)
        t, p = t[keep] - args.skip_s, p[keep]
        if run.temp is not None:
            run.temp = run.temp[keep]
        # The figure plots exactly what was analysed, not the raw file.
        run.t, run.p = t, p
        n = p.size
    if n < 16:
        raise SystemExit("Fewer than 16 samples in the analysis window.")

    duration = float(t[-1] - t[0])
    dt = np.diff(t)
    dt_median = float(np.median(dt))
    jitter = float(np.std(dt) / dt_median) if dt_median > 0 else float("nan")
    fs = 1.0 / dt_median

    mean = float(np.mean(p))

    integration = as_float(meta.get("integration_time_s"))
    if integration is None:
        integration = as_float(meta.get("average_count"), 1.0) / HARDWARE_SAMPLES_PER_S

    # Averaging for time T is a boxcar filter, whose equivalent noise bandwidth
    # is exactly 1/(2T). But the console also has an analogue HI/LO filter in
    # front of it (LO is ~15 Hz and is what Thorlabs recommends for photodiode
    # heads), and the driver cannot read that setting -- so this is an UPPER
    # bound on the real bandwidth unless you pass --console_bandwidth_hz.
    averaging_bw = 1.0 / (2.0 * integration)
    bandwidth = (min(averaging_bw, args.console_bandwidth_hz)
                 if args.console_bandwidth_hz else averaging_bw)

    # --- drift: a straight line through the run, and what is left over
    slope, intercept, r_squared, residual = linear_fit(t, p)
    run.residual = residual
    drift_pct_per_hour = slope / mean * 3600.0 * 100.0 if mean else float("nan")
    drift_total_pct = slope * duration / mean * 100.0 if mean else float("nan")

    block = max(n // 10, 2)
    first_block = float(np.mean(p[:block]))
    last_block = float(np.mean(p[-block:]))
    block_change_pct = (last_block - first_block) / mean * 100.0 if mean else float("nan")

    # Everything below is computed on the residuals: noise, with drift removed.
    noise = residual if args.detrend else p - mean
    sigma = float(np.std(noise, ddof=1))
    skew, excess_kurtosis = moments(noise)
    tau_int = correlation_time(noise)
    n_effective = n / tau_int
    sem_naive = sigma / math.sqrt(n)
    sem_corrected = sigma / math.sqrt(n_effective)

    # --- Allan deviation on the fractional signal
    taus, adevs, adev_errors = overlapping_adev(p / mean if mean else p, dt_median)
    optimal_tau = floor_adev = float("nan")
    if taus.size:
        best = int(np.argmin(adevs))
        optimal_tau, floor_adev = float(taus[best]), float(adevs[best])

    # --- spectrum of the fractional fluctuations
    nperseg = min(args.nperseg, max(16, n // 8))
    freqs, psd = welch_psd(noise / mean if mean else noise, fs, nperseg)
    peaks = spectral_peaks(freqs, psd)
    nyquist = fs / 2.0
    bands = {
        "0.01-0.1 Hz": band_rms(freqs, psd, 0.01, 0.1),
        "0.1-1 Hz": band_rms(freqs, psd, 0.1, 1.0),
        "1-10 Hz": band_rms(freqs, psd, 1.0, 10.0),
        f"10-{nyquist:.1f} Hz": band_rms(freqs, psd, 10.0, nyquist),
    }

    # --- the detector's own noise floor, from the blocked-beam segment
    dark_info = {"available": False}
    if run.dark is not None and run.dark.size > 1:
        dark_mean = float(np.mean(run.dark))
        dark_sigma = float(np.std(run.dark, ddof=1))
        variance_share = (dark_sigma / sigma) ** 2 if sigma > 0 else float("nan")
        dark_info = {
            "available": True,
            "n": int(run.dark.size),
            "mean_w": dark_mean,
            "std_w": dark_sigma,
            "std_frac_of_signal": dark_sigma / mean if mean else float("nan"),
            "variance_share": variance_share,
            "snr": mean / dark_sigma if dark_sigma > 0 else float("inf"),
            "detection_limit_w": 3.0 * dark_sigma,
        }

    # --- shot noise: the floor no amount of engineering gets you below
    photocurrent = as_float(meta.get("photocurrent_a"))
    shot_frac = float("nan")
    if photocurrent and photocurrent > 0:
        shot_frac = math.sqrt(2 * ELEMENTARY_CHARGE * bandwidth / photocurrent)

    # --- temperature, if the head had a thermistor
    temperature = {"available": False}
    if run.temp is not None and np.isfinite(run.temp).sum() > 3:
        valid = np.isfinite(run.temp)
        temp_t, temp_v = t[valid], run.temp[valid]
        interpolated = np.interp(t, temp_t, temp_v)
        excursion = float(np.max(temp_v) - np.min(temp_v))

        # Bin before correlating: temperature moves on a thermal timescale, and
        # per-sample white noise only dilutes r toward zero.
        bin_s = max(duration / 100.0, float(np.median(np.diff(temp_t))) if temp_t.size > 2 else 0.0)
        edges = np.arange(t[0], t[-1] + bin_s, bin_s)
        index = np.clip(np.digitize(t, edges) - 1, 0, edges.size - 2)
        binned_p = np.array([p[index == k].mean() for k in range(edges.size - 1)
                             if np.any(index == k)])
        binned_temp = np.array([interpolated[index == k].mean()
                                for k in range(edges.size - 1) if np.any(index == k)])

        correlation = (float(np.corrcoef(binned_temp, binned_p)[0, 1])
                       if binned_p.size > 3 else float("nan"))
        sensitivity_w_per_c = float(np.polyfit(interpolated, p, 1)[0])
        sensitivity_pct = sensitivity_w_per_c / mean * 100.0 if mean else float("nan")
        explained = sensitivity_pct * excursion
        temperature = {
            "available": True,
            "bin_s": bin_s,
            "start_c": float(temp_v[0]),
            "end_c": float(temp_v[-1]),
            "min_c": float(np.min(temp_v)),
            "max_c": float(np.max(temp_v)),
            "excursion_c": excursion,
            "correlation": correlation,
            "sensitivity_pct_per_c": sensitivity_pct,
            "explained_drift_pct": explained,
            "explained_fraction": (explained / drift_total_pct
                                   if abs(drift_total_pct) > 1e-12 else float("nan")),
            # A head that only ever warms gives temperature and elapsed time the
            # same shape, and no regression can tell those two apart.
            "monotonic": bool(abs(temp_v[-1] - temp_v[0]) > 0.8 * excursion),
        }

    analysis = {
        "file": str(run.path),
        "acquisition": {
            "n_samples": n,
            "duration_s": duration,
            "median_interval_s": dt_median,
            "interval_jitter": jitter,
            "sample_rate_hz": fs,
            "integration_time_s": integration,
            "duty_cycle": integration / dt_median if dt_median else float("nan"),
            "averaging_bandwidth_hz": averaging_bw,
            "bandwidth_hz": bandwidth,
            "bandwidth_is_upper_bound": not bool(args.console_bandwidth_hz),
            "skipped_s": args.skip_s,
        },
        "level": {
            "mean_w": mean,
            "median_w": float(np.median(p)),
            "min_w": float(np.min(p)),
            "max_w": float(np.max(p)),
            "peak_to_peak_pct": (float(np.max(p) - np.min(p)) / mean * 100.0) if mean else float("nan"),
            "raw_std_pct": (float(np.std(p, ddof=1)) / mean * 100.0) if mean else float("nan"),
        },
        "noise": {
            "sigma_w": sigma,
            "sigma_pct": sigma / mean * 100.0 if mean else float("nan"),
            "detrended": bool(args.detrend),
            "skew": skew,
            "excess_kurtosis": excess_kurtosis,
            "skew_se": math.sqrt(6.0 / n),
            "kurtosis_se": math.sqrt(24.0 / n),
            "correlation_time_samples": tau_int,
            "correlation_time_s": tau_int * dt_median,
            "n_effective": n_effective,
            "sem_naive_pct": sem_naive / mean * 100.0 if mean else float("nan"),
            "sem_corrected_pct": sem_corrected / mean * 100.0 if mean else float("nan"),
            "shot_noise_pct": shot_frac * 100.0,
        },
        "drift": {
            "slope_w_per_s": slope,
            "intercept_w": intercept,
            "pct_per_hour": drift_pct_per_hour,
            "total_pct": drift_total_pct,
            "r_squared": r_squared,
            "first_block_w": first_block,
            "last_block_w": last_block,
            "block_change_pct": block_change_pct,
            "block_s": block * dt_median,
        },
        "allan": {
            "tau_s": taus.tolist(),
            "adev_frac": adevs.tolist(),
            "adev_rel_error": adev_errors.tolist(),
            "optimal_tau_s": optimal_tau,
            "floor_pct": floor_adev * 100.0,
        },
        "spectrum": {
            "nperseg": nperseg,
            "nyquist_hz": nyquist,
            "peaks": [
                {"frequency_hz": float(f), "psd": float(s), "excess": float(e)}
                for f, s, e in peaks
            ],
            "band_rms_pct": {key: value * 100.0 for key, value in bands.items()},
        },
        "dark": dark_info,
        "temperature": temperature,
    }

    # Repeatability at the averaging times people actually use.
    analysis["repeatability_pct"] = {
        f"{tau:g} s": value * 100.0
        for tau, value in ((tau, adev_at(taus, adevs, tau))
                           for tau in (0.1, 1.0, 10.0, 60.0, 600.0, 3600.0))
        if math.isfinite(value)
    }

    entries, combined_b = build_budget(args, analysis, meta)
    type_a = type_a_at(analysis, args.averaging_time_s)
    if not math.isfinite(type_a):
        type_a = analysis["noise"]["sem_corrected_pct"]
    combined = math.sqrt(type_a ** 2 + combined_b ** 2)
    # A ratio of two readings taken with the same head at the same wavelength
    # divides the calibration out, so those terms leave the budget. Offset,
    # linearity and temperature do not: they act differently on the two
    # readings.
    surviving = sum(
        entry["standard_pct"] ** 2
        for entry in entries if not entry["cancels_in_ratio"]
    )
    relative_only = math.sqrt(type_a ** 2 + surviving)
    analysis["budget"] = {
        "averaging_time_s": args.averaging_time_s,
        "n_supplied": sum(1 for entry in entries if not entry["from_data"]),
        "type_a_pct": type_a,
        "type_b": entries,
        "type_b_combined_pct": combined_b,
        "type_b_distribution": args.type_b_distribution,
        "combined_pct": combined,
        "expanded_k2_pct": 2.0 * combined,
        "relative_combined_pct": relative_only,
        "relative_expanded_k2_pct": 2.0 * relative_only,
    }

    analysis["warnings"] = collect_warnings(analysis, run, args)
    return analysis


def collect_warnings(analysis, run: Run, args) -> list:
    warnings = []
    meta = run.meta
    level = analysis["level"]
    acquisition = analysis["acquisition"]

    range_w = as_float(meta.get("power_range_w"))
    if range_w:
        if level["max_w"] > 0.95 * range_w:
            warnings.append(
                f"Readings reach {level['max_w'] / range_w * 100:.0f} % of the "
                f"{range_w:.3e} W range -- clipping is likely; drop the range or "
                f"add attenuation."
            )
        elif level["max_w"] < 0.01 * range_w:
            warnings.append(
                f"Readings use only {level['max_w'] / range_w * 100:.2f} % of the "
                f"range, so you are throwing away ADC resolution. Pin a lower "
                f"range with --range_w."
            )

    # range_mode is what the script chose; auto_range is what the console said.
    auto = (meta.get("range_mode") == "auto" if "range_mode" in meta
            else str(meta.get("auto_range", "")).lower() in ("true", "1"))
    if auto and run.residual is not None and run.residual.size > 2:
        steps = np.abs(np.diff(run.residual))
        sigma = float(np.std(run.residual, ddof=1))
        jumps = int(np.sum(steps > 10 * sigma)) if sigma > 0 else 0
        if jumps:
            warnings.append(
                f"Auto-range was on and there are {jumps} single-sample jumps of "
                f"more than 10 sigma -- almost certainly range switches. They read "
                f"back as drift. Re-record with --lock_range."
            )

    nyquist = acquisition["sample_rate_hz"] / 2.0
    if not acquisition["bandwidth_is_upper_bound"]:
        if acquisition["bandwidth_hz"] > nyquist:
            warnings.append(
                f"The console passes {acquisition['bandwidth_hz']:.0f} Hz but you "
                f"sampled to {nyquist:.1f} Hz, so noise between the two folds back "
                f"and appears as slow noise. Harmless for drift (white noise folds "
                f"to white noise), but sample at --interval_s "
                f"{1 / (2 * acquisition['bandwidth_hz']):.3f} or faster to see that "
                f"band honestly."
            )
    elif math.isfinite(acquisition["duty_cycle"]) and acquisition["duty_cycle"] < 0.5:
        warnings.append(
            f"Unknown console bandwidth, so aliasing cannot be ruled out: the "
            f"averaging alone passes {acquisition['averaging_bandwidth_hz']:.0f} Hz "
            f"against a {nyquist:.1f} Hz Nyquist. Read the BW setting off the panel "
            f"and pass --console_bandwidth_hz (15 for LO) to make this check exact: "
            f"if that number is below {nyquist:.1f} Hz, nothing folds at all."
        )

    if acquisition["interval_jitter"] > 0.1:
        warnings.append(
            f"Sample spacing varies by {acquisition['interval_jitter'] * 100:.0f} % "
            f"rms. Allan deviation and the spectrum assume even sampling, so "
            f"treat them as approximate."
        )

    noise = analysis["noise"]
    if abs(noise["skew"]) > 3 * noise["skew_se"] or abs(noise["excess_kurtosis"]) > 3 * noise["kurtosis_se"]:
        warnings.append(
            "The noise is not Gaussian (see the skew/kurtosis and the QQ panel). "
            "A single rms understates how often you land far from the mean -- "
            "look for mode hopping, spikes or a range switch."
        )

    dark = analysis["dark"]
    if not dark["available"]:
        warnings.append(
            "No beam-blocked segment, so the meter's own noise floor is unknown "
            "and none of this can be attributed to the laser rather than the "
            "detector. Re-record without --dark_s 0."
        )
    elif math.isfinite(dark["variance_share"]) and dark["variance_share"] > 0.5:
        warnings.append(
            f"The detector accounts for {dark['variance_share'] * 100:.0f} % of "
            f"the observed variance -- you are mostly measuring the power meter, "
            f"not the laser. More optical power or more averaging."
        )

    drift = analysis["drift"]
    if abs(drift["total_pct"]) > 5 * analysis["noise"]["sigma_pct"] and abs(drift["total_pct"]) > 0.1:
        warnings.append(
            f"Drift ({drift['total_pct']:+.2f} % over the run) dominates the "
            f"noise. If the run started cold, re-analyse with --skip_s to drop "
            f"the warm-up."
        )

    if not meta.get("zeroed") and analysis["dark"]["available"]:
        dark = analysis["dark"]
        warnings.append(
            f"This run was not zeroed, so the {dark['mean_w'] * 1e9:+.3f} nW dark "
            f"offset above is still sitting in every reading "
            f"({dark['mean_w'] / analysis['level']['mean_w'] * 100:+.4f} % of "
            f"signal). Subtract it if you are quoting an absolute power; it "
            f"cancels in a ratio."
        )
    return warnings


# -------------------------------------------------------------- the report


def rule(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def format_report(analysis, run: Run, args) -> str:
    meta = run.meta
    out = []
    add = out.append

    acquisition = analysis["acquisition"]
    level = analysis["level"]
    noise = analysis["noise"]
    drift = analysis["drift"]
    allan = analysis["allan"]
    dark = analysis["dark"]
    budget = analysis["budget"]

    add("=" * 72)
    add("LASER POWER STABILITY REPORT")
    add("=" * 72)
    add(f"File        {run.path.name}")
    add(f"Recorded    {meta.get('started_at', 'unknown')}")
    if meta.get("note"):
        add(f"Note        {meta['note']}")

    add(rule("1. INSTRUMENT"))
    add(f"  Meter            {meta.get('model', '?')} (S/N {meta.get('serial_number', '?')})")
    add(f"  Sensor           {meta.get('sensor_name', '?')} "
        f"(S/N {meta.get('sensor_serial', '?')}, {meta.get('sensor_type', '?')})")
    add(f"  Calibration      {meta.get('calibration_message', 'unknown')}")
    add(f"  Wavelength       {as_float(meta.get('wavelength_nm'), float('nan')):.1f} nm")
    add(f"  Averaging        {meta.get('average_count', '?')} samples at 3000/s "
        f"= {acquisition['integration_time_s'] * 1e3:.2f} ms integration")
    if acquisition["bandwidth_is_upper_bound"]:
        add(f"  Bandwidth        {acquisition['bandwidth_hz']:.1f} Hz from the averaging, "
            f"an UPPER bound --")
        add("                   the console's own HI/LO filter (LO is ~15 Hz) sits in")
        add("                   front of it and the driver cannot read it. Pass")
        add("                   --console_bandwidth_hz if you know the panel setting.")
    else:
        add(f"  Bandwidth        {acquisition['bandwidth_hz']:.1f} Hz "
            f"(averaging {acquisition['averaging_bandwidth_hz']:.1f} Hz, "
            f"console filter {args.console_bandwidth_hz:.1f} Hz)")
    add(f"  Range            {as_float(meta.get('power_range_w'), float('nan')):.3e} W"
        f"   ({meta.get('range_mode', 'auto-range ' + str(meta.get('auto_range', '?')))})")
    add(f"  Samples          {acquisition['n_samples']} over "
        f"{acquisition['duration_s']:.1f} s "
        f"at {acquisition['sample_rate_hz']:.2f} Hz "
        f"(spacing jitter {acquisition['interval_jitter'] * 100:.1f} %)")
    if acquisition["skipped_s"]:
        add(f"  Skipped          first {acquisition['skipped_s']:.0f} s")

    add(rule("2. SIGNAL LEVEL"))
    add(f"  Mean             {level['mean_w'] * 1e6:.4f} uW")
    add(f"  Median           {level['median_w'] * 1e6:.4f} uW")
    add(f"  Min / max        {level['min_w'] * 1e6:.4f} / {level['max_w'] * 1e6:.4f} uW"
        f"   (peak-to-peak {level['peak_to_peak_pct']:.3f} %)")
    add(f"  Raw rms spread   {level['raw_std_pct']:.4f} %   <- noise AND drift together")

    add(rule("3. SHORT-TERM NOISE  " + ("(drift removed by a linear fit)"
                                       if noise["detrended"] else
                                       "(--no_detrend: drift still included)")))
    add(f"  sigma            {noise['sigma_w'] * 1e9:.3f} nW  =  {noise['sigma_pct']:.4f} % rms")
    add(f"  Distribution     skew {noise['skew']:+.3f} (+/-{noise['skew_se']:.3f}), "
        f"excess kurtosis {noise['excess_kurtosis']:+.3f} (+/-{noise['kurtosis_se']:.3f})")
    gaussian = (abs(noise["skew"]) < 3 * noise["skew_se"]
                and abs(noise["excess_kurtosis"]) < 3 * noise["kurtosis_se"])
    add(f"                   {'consistent with Gaussian' if gaussian else 'NOT Gaussian -- see the QQ panel'}")
    add(f"  Correlation time {noise['correlation_time_s']:.3f} s "
        f"({noise['correlation_time_samples']:.1f} samples)")
    add(f"  Effective N      {noise['n_effective']:.0f} of {acquisition['n_samples']} "
        f"samples are independent")
    add(f"  Mean uncertainty {noise['sem_corrected_pct']:.5f} % "
        f"(naive sigma/sqrt(N) would claim {noise['sem_naive_pct']:.5f} %)")
    if math.isfinite(noise["shot_noise_pct"]) and noise["shot_noise_pct"] > 0:
        ratio = noise["sigma_pct"] / noise["shot_noise_pct"]
        bound = " at most" if acquisition["bandwidth_is_upper_bound"] else ""
        add(f"  Shot-noise limit {noise['shot_noise_pct']:.5f} %{bound} -- the measured "
            f"noise is {ratio:.0f}x larger, so")
        add("                   it is technical and in principle fixable, not fundamental.")

    add(rule("4. DETECTOR FLOOR  (beam blocked)"))
    if dark["available"]:
        add(f"  Dark offset      {dark['mean_w'] * 1e9:+.4f} nW "
            f"({dark['mean_w'] / level['mean_w'] * 100:+.4f} % of signal)")
        add(f"  Dark noise       {dark['std_w'] * 1e9:.4f} nW rms "
            f"= {dark['std_frac_of_signal'] * 100:.5f} % of signal")
        add(f"  Share of variance  {dark['variance_share'] * 100:.1f} % of the observed "
            f"noise is the detector")
        add(f"  Verdict          {'mostly the METER' if dark['variance_share'] > 0.5 else 'mostly the LASER'}")
        add(f"  SNR              {dark['snr']:.3e}")
        add(f"  Detection limit  {dark['detection_limit_w'] * 1e9:.4f} nW (3 sigma dark)")
    else:
        add("  No beam-blocked segment in this run, so laser noise and detector")
        add("  noise cannot be separated. Record one with --dark_s.")

    add(rule("5. DRIFT"))
    add(f"  Linear slope     {drift['pct_per_hour']:+.4f} %/hour  "
        f"(R^2 = {drift['r_squared']:.3f})")
    add(f"  Over this run    {drift['total_pct']:+.4f} % in "
        f"{acquisition['duration_s'] / 60:.1f} min")
    add(f"  First vs last    {drift['block_change_pct']:+.4f} % "
        f"(means of the first and last {drift['block_s']:.0f} s)")
    temperature = analysis["temperature"]
    if temperature["available"]:
        add(f"  Head temperature {temperature['start_c']:.2f} -> {temperature['end_c']:.2f} C "
            f"(swing {temperature['excursion_c']:.2f} C)")
        add(f"  Power vs temp    r = {temperature['correlation']:+.3f}, "
            f"{temperature['sensitivity_pct_per_c']:+.3f} %/C")
        fraction = temperature.get("explained_fraction", float("nan"))
        add(f"  Explains         {temperature['explained_drift_pct']:+.3f} % of the "
            f"{drift['total_pct']:+.3f} % observed"
            + (f"  ({fraction * 100:.0f} %)" if math.isfinite(fraction) else ""))
        thermal = (math.isfinite(fraction) and 0.5 < fraction < 2.0) or \
                  (math.isfinite(temperature["correlation"]) and abs(temperature["correlation"]) > 0.7)
        if thermal:
            add("  Verdict          drift tracks temperature -- suspect the room and the")
            add("                   head, not the laser. Stabilise or re-reference.")
            if temperature.get("monotonic"):
                add("                   Caveat: the head warmed monotonically, so thermal")
                add("                   and source drift are collinear here and the %/C")
                add("                   above absorbs both. To separate them, let the room")
                add("                   cycle, or run long enough to see it turn around.")
        else:
            add("  Verdict          drift does not track head temperature -- look at the")
            add("                   source, the alignment or the pointing.")
    else:
        add("  No head-temperature log (this head may have no thermistor), so")
        add("  thermal drift cannot be separated from source drift.")

    add(rule("6. ALLAN DEVIATION  (how much averaging actually buys you)"))
    taus = np.array(allan["tau_s"])
    devs = np.array(allan["adev_frac"])
    if taus.size:
        add("  Average each point for...      ...and it repeats to")
        for tau in (0.1, 1.0, 10.0, 60.0, 600.0, 3600.0):
            value = adev_at(taus, devs, tau)
            if math.isfinite(value):
                marker = "  <- best" if abs(tau - allan["optimal_tau_s"]) < tau / 2 else ""
                add(f"  {tau:22.1f} s {value * 100:20.5f} %{marker}")
        add(f"\n  Best averaging   {allan['optimal_tau_s']:.2f} s "
            f"-> floor {allan['floor_pct']:.5f} % rms")
        if acquisition["duration_s"] < 60 or acquisition["n_samples"] < 500:
            add("  (Short run: the long-tau end rests on only a few independent")
            add("   differences. Record minutes, not seconds, before trusting it.)")
        if allan["optimal_tau_s"] >= taus[-1] * 0.99:
            add("  The curve is still falling at the longest tau this run supports,")
            add("  so that floor is a ceiling on your real one -- record for longer")
            add("  to find where drift takes over.")
        else:
            add("  Averaging longer than that buys nothing: drift has taken over.")
        add("  Read the log-log panel: -1/2 slope = white noise, flat = flicker,")
        add("  +1 slope = drift.")
    else:
        add("  Too few samples for an Allan deviation.")

    add(rule("7. FREQUENCY CONTENT"))
    spectrum = analysis["spectrum"]
    add(f"  Nyquist          {spectrum['nyquist_hz']:.2f} Hz -- anything faster than")
    add("                   this in the signal folds back into the bands below.")
    add("  rms by band:")
    for label, value in spectrum["band_rms_pct"].items():
        if math.isfinite(value):
            add(f"    {label:>14}  {value:.5f} %")
    if spectrum["peaks"]:
        add("  Lines above the noise floor:")
        for peak in spectrum["peaks"]:
            hint = ""
            if abs(peak["frequency_hz"] - 60.0) < 1.0 or abs(peak["frequency_hz"] - 120.0) < 1.0:
                hint = "   <- mains pickup"
            elif peak["frequency_hz"] < 0.05:
                hint = "   <- slow, likely thermal/HVAC"
            add(f"    {peak['frequency_hz']:8.3f} Hz   {peak['excess']:5.1f}x floor{hint}")
    else:
        add("  No narrow lines -- the noise is broadband.")

    add(rule("8. UNCERTAINTY BUDGET"))
    add(f"  For a single measurement averaged over {budget['averaging_time_s']:g} s.")
    if not budget["n_supplied"]:
        add("")
        add("  NO TYPE B TERMS SUPPLIED. What follows is the STATISTICAL spread of")
        add("  your own data and nothing else -- it is NOT your accuracy. The")
        add("  calibration of the head alone is a few percent and dwarfs it. Pass")
        add("  --cal_tolerance_pct and friends off your spec sheet to get a real")
        add("  budget; see --help for the published PM100D / S130VC figures.")
    else:
        add(f"  Type B terms are treated as {budget['type_b_distribution']} bounds.")
    add("")
    add(f"  {'Term':<32}{'tolerance':>11}{'u (1 sigma)':>13}")
    add(f"  {'-' * 56}")
    add(f"  {'Type A: statistical (Allan)':<32}{'':>11}{budget['type_a_pct']:>12.4f} %")
    for entry in budget["type_b"]:
        add(f"  {entry['name']:<32}{entry['tolerance_pct']:>10.3f} %"
            f"{entry['standard_pct']:>12.4f} %")
    add(f"  {'-' * 56}")
    add(f"  {'Combined standard u_c':<32}{'':>11}{budget['combined_pct']:>12.4f} %")
    add(f"  {'Expanded U (k=2, 95 %)':<32}{'':>11}{budget['expanded_k2_pct']:>12.4f} %")
    add("")
    label = "ABSOLUTE" if budget["n_supplied"] else "STATISTICAL ONLY"
    add(f"  {label}   P = {level['mean_w'] * 1e6:.4f} uW "
        f"+/- {level['mean_w'] * 1e6 * budget['expanded_k2_pct'] / 100:.4f} uW  (k=2)")
    if any(entry["cancels_in_ratio"] for entry in budget["type_b"]):
        add(f"  RELATIVE   ratios and changes measured with this same head and")
        add(f"             wavelength are good to {budget['relative_expanded_k2_pct']:.4f} % (k=2) --")
        add(f"             the calibration term cancels.")
    add("")
    for entry in budget["type_b"]:
        add(f"  * {entry['name']}: {entry['note']}")

    add(rule("9. WHAT TO DO WITH THIS"))
    dominant = max(budget["type_b"], key=lambda entry: entry["standard_pct"], default=None)
    if not budget["n_supplied"]:
        add("  This run tells you how REPEATABLE the meter is, not how ACCURATE.")
        add("  For accuracy you need the spec-sheet numbers in the budget above.")
    elif dominant and dominant["standard_pct"] > budget["type_a_pct"]:
        add(f"  Absolute accuracy is limited by {dominant['name'].lower()} "
            f"({dominant['standard_pct']:.2f} %), not by")
        add(f"  the laser ({budget['type_a_pct']:.4f} %). Averaging longer will not help; only a")
        add("  fresh calibration or a side-by-side reference detector will.")
    else:
        add("  Statistics dominate the budget -- average longer or quiet the source.")
    if math.isfinite(allan["optimal_tau_s"]):
        add(f"  Average each data point for ~{allan['optimal_tau_s']:.1f} s; "
            f"beyond that drift wins.")
    if abs(drift["pct_per_hour"]) > 0.1:
        add(f"  At {drift['pct_per_hour']:+.2f} %/hour, re-reference your power every "
            f"~{abs(0.1 / drift['pct_per_hour'] * 60):.0f} min to hold 0.1 %.")

    warnings = analysis["warnings"]
    if warnings:
        add(rule("WARNINGS"))
        for warning in warnings:
            add(f"  ! {warning}")

    add("")
    return "\n".join(out)


# --------------------------------------------------------------- the figure


def make_figure(analysis, run: Run, args, path: Path):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, p = run.t, run.p
    mean = analysis["level"]["mean_w"]
    drift = analysis["drift"]

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    figure.suptitle(
        f"{run.path.name}    {mean * 1e6:.4f} uW    "
        f"noise {analysis['noise']['sigma_pct']:.4f} % rms    "
        f"drift {drift['pct_per_hour']:+.3f} %/h",
        fontsize=11,
    )

    # 1. the run itself, with the drift line on top
    ax = axes[0, 0]
    ax.plot(t, p * 1e6, lw=0.6, color="0.4")
    ax.plot(t, (drift["slope_w_per_s"] * t + drift["intercept_w"]) * 1e6,
            color="crimson", lw=1.5, label=f"{drift['pct_per_hour']:+.3f} %/h")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("power (uW)")
    ax.set_title("Power vs time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. residual histogram against the Gaussian it is supposed to be
    ax = axes[0, 1]
    residual = run.residual
    residual_pct = residual / mean * 100
    ax.hist(residual_pct, bins=60, density=True, color="steelblue", alpha=0.75)
    sigma = np.std(residual_pct, ddof=1)
    grid = np.linspace(residual_pct.min(), residual_pct.max(), 200)
    ax.plot(grid, np.exp(-0.5 * (grid / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)),
            color="crimson", lw=1.5, label="Gaussian")
    if analysis["dark"]["available"]:
        ax.axvline(analysis["dark"]["std_frac_of_signal"] * 100, color="darkgreen",
                   ls="--", lw=1, label="detector sigma")
        ax.axvline(-analysis["dark"]["std_frac_of_signal"] * 100, color="darkgreen",
                   ls="--", lw=1)
    ax.set_xlabel("deviation from trend (%)")
    ax.set_ylabel("density")
    ax.set_title(f"Noise distribution (sigma = {sigma:.4f} %)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Allan deviation, with the slopes you are meant to recognise
    ax = axes[0, 2]
    taus = np.array(analysis["allan"]["tau_s"])
    devs = np.array(analysis["allan"]["adev_frac"]) * 100
    errors = np.array(analysis["allan"]["adev_rel_error"])
    if taus.size:
        ax.errorbar(taus, devs, yerr=devs * errors, fmt="o-", ms=3, lw=1,
                    color="navy", ecolor="0.7", capsize=2)
        reference = devs[0] * (taus / taus[0]) ** -0.5
        ax.plot(taus, reference, ls=":", color="crimson", lw=1, label="white noise, -1/2")
        best = analysis["allan"]["optimal_tau_s"]
        if math.isfinite(best):
            ax.axvline(best, color="darkgreen", ls="--", lw=1,
                       label=f"best tau = {best:.1f} s")
        if analysis["dark"]["available"]:
            ax.axhline(analysis["dark"]["std_frac_of_signal"] * 100, color="0.5",
                       ls="-.", lw=1, label="detector floor")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("averaging time tau (s)")
    ax.set_ylabel("Allan deviation (%)")
    ax.set_title("Allan deviation")
    ax.grid(alpha=0.3, which="both")

    # 4. where the noise lives
    ax = axes[1, 0]
    fs = analysis["acquisition"]["sample_rate_hz"]
    freqs, psd = welch_psd(residual / mean, fs, analysis["spectrum"]["nperseg"])
    if freqs.size > 1:
        ax.loglog(freqs[1:], psd[1:], lw=0.7, color="darkslategray")
        for peak in analysis["spectrum"]["peaks"][:3]:
            ax.axvline(peak["frequency_hz"], color="crimson", ls=":", lw=1)
            ax.text(peak["frequency_hz"], ax.get_ylim()[1],
                    f" {peak['frequency_hz']:.2f} Hz", fontsize=7,
                    rotation=90, va="top", color="crimson")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("RIN (1/Hz)")
    ax.set_title("Relative intensity noise spectrum")
    ax.grid(alpha=0.3, which="both")

    # 5. normal probability plot -- tails show up here, not in the histogram
    ax = axes[1, 1]
    sample = np.sort(residual_pct)
    if sample.size > 4000:
        sample = sample[np.linspace(0, sample.size - 1, 4000).astype(int)]
    n = sample.size
    normal = NormalDist()
    theoretical = np.array([normal.inv_cdf((i + 0.5) / n) for i in range(n)])
    ax.plot(theoretical, sample, ".", ms=2, color="steelblue")
    limit = max(abs(theoretical[0]), abs(theoretical[-1]))
    ax.plot([-limit, limit], [-limit * sigma, limit * sigma], color="crimson", lw=1)
    ax.set_xlabel("theoretical quantile (sigma)")
    ax.set_ylabel("observed deviation (%)")
    ax.set_title("Normal probability plot")
    ax.grid(alpha=0.3)

    # 6. temperature if we have it, otherwise the autocorrelation
    ax = axes[1, 2]
    temperature = analysis["temperature"]
    if temperature["available"]:
        valid = np.isfinite(run.temp)
        interpolated = np.interp(t, t[valid], run.temp[valid])
        ax.plot(interpolated, p * 1e6, ".", ms=2, alpha=0.4, color="darkorange")
        fit = np.polyfit(interpolated, p * 1e6, 1)
        grid = np.linspace(interpolated.min(), interpolated.max(), 10)
        ax.plot(grid, np.polyval(fit, grid), color="crimson", lw=1.5,
                label=f"{temperature['sensitivity_pct_per_c']:+.3f} %/C\n"
                      f"r = {temperature['correlation']:+.3f}")
        ax.set_xlabel("head temperature (C)")
        ax.set_ylabel("power (uW)")
        ax.set_title("Power vs head temperature")
        ax.legend(fontsize=8)
    else:
        rho = autocorrelation(residual)
        lags = np.arange(rho.size) * analysis["acquisition"]["median_interval_s"]
        show = min(rho.size, max(50, int(20 * analysis["noise"]["correlation_time_samples"])))
        ax.plot(lags[:show], rho[:show], lw=1, color="purple")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(analysis["noise"]["correlation_time_s"], color="crimson", ls="--",
                   lw=1, label=f"tau_int = {analysis['noise']['correlation_time_s']:.2f} s")
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("autocorrelation")
        ax.set_title("Autocorrelation of the noise")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=140)
    if args.show:
        plt.show()
    plt.close(figure)


# ----------------------------------------------------------------- the entry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Noise, drift and uncertainty analysis for a power-meter run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv", nargs="?", type=Path, default=None,
        help="the run to analyse; default is the newest one in --data_dir",
    )
    parser.add_argument(
        "--data_dir", type=Path,
        default=Path.home() / "Desktop" / "power_meter_data",
        help="where to look when no file is given",
    )

    parser.add_argument("--skip_s", type=float, default=0.0, metavar="S",
                        help="drop this much from the start (warm-up)")
    parser.add_argument("--max_s", type=float, default=None, metavar="S",
                        help="analyse only up to this time")
    parser.add_argument("--no_detrend", dest="detrend", action="store_false",
                        help="quote noise without removing the linear drift")
    parser.add_argument("--nperseg", type=int, default=2048, metavar="N",
                        help="segment length for the Welch spectrum")
    parser.add_argument("--averaging_time_s", type=float, default=1.0, metavar="S",
                        help="the budget is quoted for a measurement this long")
    parser.add_argument("--console_bandwidth_hz", type=float, default=None, metavar="HZ",
                        help="the console's analogue BW setting, which the driver "
                             "cannot read: 15 for a PM100D on LO (Thorlabs' advice "
                             "for photodiode heads). Unset assumes the averaging "
                             "alone sets the bandwidth, an upper bound")

    budget = parser.add_argument_group(
        "uncertainty budget (Type B)",
        "All default to zero -- nothing enters the budget unless you put it there, "
        "off your own spec sheet. For a PM100D + S130VC the published figures are: "
        "calibration +/-3 % over 451-1000 nm and +/-5 % over 200-450 and "
        "1001-1100 nm; linearity +/-0.5 %; console +/-0.2 % of FULL SCALE "
        "(+/-0.5 % on the 50 nA range) -- full scale, so at a tenth of the range "
        "that is 2 % of your reading, and you must work it out for your range.",
    )
    budget.add_argument("--cal_tolerance_pct", type=float, default=0.0,
                        help="sensor calibration tolerance, from your spec sheet")
    budget.add_argument("--console_tolerance_pct", type=float, default=0.0,
                        help="console readout error AS A PERCENT OF YOUR READING "
                             "(convert it yourself from the %% of full scale spec)")
    budget.add_argument("--linearity_pct", type=float, default=0.0,
                        help="detector nonlinearity across power levels")
    budget.add_argument("--wavelength_error_nm", type=float, default=0.0,
                        help="how well you know the wavelength you set")
    budget.add_argument("--responsivity_slope_pct_per_nm", type=float, default=0.0,
                        help="local slope of the responsivity curve")
    budget.add_argument("--temp_coeff_pct_per_c", type=float, default=0.0,
                        help="sensor temperature coefficient")
    budget.add_argument("--temp_excursion_c", type=float, default=None,
                        help="temperature swing to assume; default is the logged one")
    budget.add_argument("--type_b_distribution", choices=sorted(DIVISORS),
                        default="rect",
                        help="how to read a datasheet tolerance: rect (/sqrt3), "
                             "normal (/1) or k2 (/2)")

    parser.add_argument("--show", action="store_true", help="open the figure window")
    parser.add_argument("--no_figure", action="store_true", help="skip the PNG")
    return parser.parse_args(argv)


def json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main(argv=None) -> int:
    args = parse_args(argv)
    path = args.csv or newest_run(args.data_dir)
    if not path.exists():
        raise SystemExit(f"{path} does not exist.")

    run = load_run(path)
    analysis = analyse(run, args)
    report = format_report(analysis, run, args)
    print(report)

    stem = path.with_suffix("")
    report_path = Path(f"{stem}_report.txt")
    metrics_path = Path(f"{stem}_metrics.json")
    report_path.write_text(report, encoding="utf-8")
    metrics_path.write_text(json.dumps(analysis, indent=2, default=json_safe),
                            encoding="utf-8")

    print(f"Saved {report_path}")
    print(f"      {metrics_path}")
    if not args.no_figure:
        figure_path = Path(f"{stem}_stability.png")
        make_figure(analysis, run, args, figure_path)
        print(f"      {figure_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
