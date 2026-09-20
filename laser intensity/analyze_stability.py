#!/usr/bin/env python3
"""
analyze_stability.py
====================
Noise and drift analysis for a run recorded by ``measure_intensity.py``, aimed
at one question: **how well do you actually know the power?**

    python analyze_stability.py                  # newest run on the Desktop
    python analyze_stability.py path/to/run.csv

Writes ``<stem>_report.txt``, ``<stem>_stability.png`` and
``<stem>_metrics.json`` beside the data.

What it separates, and why:

* **Drift from noise.** A line is fitted and removed; noise is quoted on the
  residuals, drift as %/hour. One rms for both hides which you have.
* **The laser from the meter.** The beam-blocked segment is the detector's own
  noise floor. If the residual noise is barely above it, you are
  characterising the power meter.
* **Allan deviation.** Straight statistics assume independent samples, and
  power readings are not. The slope names the noise: -1/2 white, flat flicker,
  +1/2 random walk, +1 drift. The minimum is the best you can do, and the tau
  there is how long to average each point.
* **Correlation time.** sigma/sqrt(N) is optimistic when samples are
  correlated. The integrated autocorrelation time gives the effective N.
* **Abrupt steps.** Mode hops, a slipping mount, a range switch. Not noise, so
  they never average away.
* **Spectrum.** Where the noise lives -- mains pickup, a chopper, a fan, 1/f.
* **An uncertainty budget.** Type A from the Allan deviation, Type B from the
  datasheet numbers you pass in. Every Type B term defaults to ZERO, so out of
  the box this reports repeatability, not accuracy, and says so.

The console's bandwidth is taken to be 15 Hz throughout -- the PM100D's LO
setting, which is what Thorlabs recommends for photodiode heads. Put the front
panel on LO and this is exact; leave it on HI and the bandwidth is larger and
unknown, because the driver cannot read the setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

# PM100D, Meas Config -> BW -> LO. The driver exposes no bandwidth function, so
# this cannot be read back; it is an assumption about your front panel.
CONSOLE_BANDWIDTH_HZ = 15.0
ELEMENTARY_CHARGE = 1.602176634e-19

# A datasheet "+/-3 %" is a bound, not a standard deviation. With nothing else
# known, treat it as a rectangular distribution: standard uncertainty = a/sqrt(3).
RECTANGULAR = math.sqrt(3.0)

# np.trapz was renamed in numpy 2.0; the Windows PC may have either.
_integrate = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------- loading


class Run:
    """One acquisition: timebase, power, metadata, optional dark segment."""

    def __init__(self, path, t, p, meta, dark):
        self.path = path
        self.t = t
        self.p = p
        self.meta = meta
        self.dark = dark
        self.residual = None       # filled in by analyse(), reused by the figure


def read_csv(path: Path):
    """Return (elapsed_s, power_W, header dict) from one CSV."""
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
    # names from the very first line it sees, comment marker or not.
    table = np.genfromtxt(path, delimiter=",", skip_header=comment_lines,
                          names=True, dtype=float)
    if table.size == 0:
        raise SystemExit(f"{path.name} has no data rows.")

    names = table.dtype.names
    power = "power_W" if "power_W" in names else names[1]
    time = "elapsed_s" if "elapsed_s" in names else names[0]
    return np.atleast_1d(table[time]), np.atleast_1d(table[power]), header


def newest_run(directory: Path) -> Path:
    candidates = [path for path in directory.glob("power_*.csv")
                  if not path.name.endswith("_dark.csv")]
    if not candidates:
        raise SystemExit(
            f"No runs found in {directory}. Record one first:\n"
            f"  python measure_intensity.py --wavelength_nm 520")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_run(path: Path) -> Run:
    t, p, header = read_csv(path)
    stem = path.with_suffix("").name

    meta = {}
    meta_path = path.with_name(stem + "_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, value in header.items():       # CSV comments are the fallback
        meta.setdefault(key, value)

    dark_path = path.with_name(stem + "_dark.csv")
    dark = read_csv(dark_path)[1] if dark_path.exists() else None
    return Run(path, t, p, meta, dark)


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------- maths


def linear_fit(x, y):
    """Least-squares slope, intercept, R^2 and residuals."""
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - np.sum(residual ** 2) / total if total > 0 else float("nan")
    return float(slope), float(intercept), float(r_squared), residual


def overlapping_adev(y, tau0, points=50, min_estimates=8):
    """Overlapping Allan deviation of a frequency-like series.

    ``y`` is the measurement itself, so the standard frequency-data estimator
    applies. Working from the cumulative sum turns the double sum into three
    slices, which keeps long runs fast.

    Returns (tau, adev, relative 1-sigma error bars).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 16:
        return np.array([]), np.array([]), np.array([])

    x = np.concatenate(([0.0], np.cumsum(y))) * tau0      # the "phase"
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
    d = np.asarray(d, dtype=float) - np.mean(d)
    n = d.size
    spectrum = np.fft.rfft(d, 1 << int(2 * n - 1).bit_length())
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    return np.ones(n) if correlation[0] == 0 else correlation / correlation[0]


def correlation_time(d, c=6.0):
    """Integrated autocorrelation time in samples, with Sokal windowing.

    Samples that far apart are effectively independent, so N/this is the honest
    divisor for sigma/sqrt(N).
    """
    rho = autocorrelation(d)
    running = 1.0
    for k in range(1, rho.size):
        if rho[k] <= 0:                       # first zero crossing: stop
            break
        running += 2.0 * rho[k]
        if k >= c * running:                  # Sokal's automatic window
            break
    return max(float(running), 1.0)


def welch_psd(x, fs, nperseg):
    """One-sided power spectral density, Hann window, 50 % overlap.

    Each segment is linearly detrended first so drift does not leak across the
    spectrum, and scaled as a density so integrating it returns the variance.
    A single periodogram is an inconsistent estimator -- its variance does not
    fall with more data -- which is why segments are averaged.
    """
    x = np.asarray(x, dtype=float)
    nperseg = int(min(nperseg, x.size))
    if nperseg < 16:
        return np.array([]), np.array([])

    window = np.hanning(nperseg + 1)[:-1]     # periodic form
    scale = 1.0 / (fs * np.sum(window ** 2))
    index = np.arange(nperseg)
    total, count = None, 0

    for start in range(0, x.size - nperseg + 1, nperseg // 2):
        segment = x[start:start + nperseg]
        slope, intercept = np.polyfit(index, segment, 1)
        segment = (segment - (slope * index + intercept)) * window
        spectrum = np.abs(np.fft.rfft(segment)) ** 2 * scale
        spectrum[1:-1] *= 2.0                 # fold in the negative frequencies
        total = spectrum if total is None else total + spectrum
        count += 1

    if not count:
        return np.array([]), np.array([])
    return np.fft.rfftfreq(nperseg, 1.0 / fs), total / count


def spectral_peaks(freqs, psd, count=5, threshold=6.0):
    """Narrow lines standing well above the local noise floor."""
    if freqs.size < 32:
        return []
    width = max(16, psd.size // 40)
    half = max(1, width // 2)
    baseline = np.array([np.median(psd[max(0, i - half):i + half + 1])
                         for i in range(psd.size)])
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(baseline > 0, psd / baseline, 0.0)

    peaks = [(freqs[i], psd[i], excess[i]) for i in range(2, psd.size - 1)
             if excess[i] > threshold and psd[i] > psd[i - 1] and psd[i] >= psd[i + 1]]
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


def find_steps(t, p, block_s=2.0, threshold=6.0):
    """Abrupt level changes: mode hops, range switches, a mount slipping.

    Averages into ``block_s`` blocks and compares each block with the one two
    places later, which captures the whole of a step completing inside a single
    block whatever its phase. The yardstick is the median absolute deviation of
    those differences -- robust, so a few genuine steps cannot inflate it and
    hide themselves.

    Returns [(time_s, size as a fraction of the mean)].
    """
    n = p.size
    step = (t[-1] - t[0]) / (n - 1)
    size = max(1, int(round(block_s / step)))
    count = n // size
    if count < 10:
        return []

    means = p[:count * size].reshape(count, size).mean(axis=1)
    diffs = means[2:] - means[:-2]
    centred = diffs - np.median(diffs)
    scale = 1.4826 * float(np.median(np.abs(centred)))
    if scale <= 0:
        return []

    strength = np.abs(centred) / scale
    found = []
    for i in np.argsort(strength)[::-1]:
        if strength[i] < threshold:
            break
        if any(abs(i - j) <= 3 for j, _ in found):   # one transition, one entry
            continue
        found.append((int(i), float(diffs[i])))
    found.sort()
    mean = float(np.mean(p))
    return [(float(t[0] + (i + 1.5) * size * step), d / mean) for i, d in found]


# ------------------------------------------------------------- the analysis


def analyse(run: Run, args) -> dict:
    t, p, meta = run.t, run.p, run.meta

    if args.skip_s > 0 or args.max_s is not None:
        end = args.max_s if args.max_s is not None else t[-1]
        keep = (t >= args.skip_s) & (t <= end)
        t, p = t[keep] - args.skip_s, p[keep]
        run.t, run.p = t, p          # the figure plots what was analysed
    n = p.size
    if n < 16:
        raise SystemExit("Fewer than 16 samples in the analysis window.")

    duration = float(t[-1] - t[0])
    dt = np.diff(t)
    # The acquisition loop catches up after a slow read, so gaps come in bursts
    # and the median understates the true spacing. The mean is the real rate.
    dt_step = duration / (n - 1)
    jitter = float(np.std(dt) / dt_step)
    fs = 1.0 / dt_step
    mean = float(np.mean(p))

    # --- drift: a straight line through the run, and what is left over
    slope, intercept, r_squared, residual = linear_fit(t, p)
    run.residual = residual
    sigma = float(np.std(residual, ddof=1))
    skew, excess_kurtosis = moments(residual)
    tau_int = correlation_time(residual)
    n_effective = n / tau_int

    block = max(n // 10, 2)
    first_block, last_block = float(np.mean(p[:block])), float(np.mean(p[-block:]))

    # --- Allan deviation on the fractional signal
    taus, adevs, adev_errors = overlapping_adev(p / mean if mean else p, dt_step)
    optimal_tau = floor_adev = float("nan")
    if taus.size:
        best = int(np.argmin(adevs))
        optimal_tau, floor_adev = float(taus[best]), float(adevs[best])

    # --- spectrum of the fractional fluctuations. Segments sized from the run
    # so a long record automatically reaches lower frequencies.
    nperseg = int(min(1 << 17, max(256, n // 8)))
    freqs, psd = welch_psd(residual / mean if mean else residual, fs, nperseg)
    nyquist = fs / 2.0
    bands = {"0.01-0.1 Hz": band_rms(freqs, psd, 0.01, 0.1),
             "0.1-1 Hz": band_rms(freqs, psd, 0.1, 1.0),
             "1-10 Hz": band_rms(freqs, psd, 1.0, 10.0),
             f"10-{nyquist:.1f} Hz": band_rms(freqs, psd, 10.0, nyquist)}

    # --- the detector's own noise floor, from the blocked-beam segment
    dark = {"available": False}
    if run.dark is not None and run.dark.size > 1:
        dark_mean = float(np.mean(run.dark))
        dark_sigma = float(np.std(run.dark, ddof=1))
        dark = {
            "available": True,
            "mean_w": dark_mean,
            "std_w": dark_sigma,
            "std_frac_of_signal": dark_sigma / mean if mean else float("nan"),
            # Independent variances add, so this is the detector's share.
            "variance_share": (dark_sigma / sigma) ** 2 if sigma > 0 else float("nan"),
            "snr": mean / dark_sigma if dark_sigma > 0 else float("inf"),
            "detection_limit_w": 3.0 * dark_sigma,
        }

    # --- shot noise: the floor no engineering gets you below
    photocurrent = as_float(meta.get("photocurrent_a"))
    shot_frac = (math.sqrt(2 * ELEMENTARY_CHARGE * CONSOLE_BANDWIDTH_HZ / photocurrent)
                 if photocurrent and photocurrent > 0 else float("nan"))

    analysis = {
        "file": str(run.path),
        "acquisition": {
            "n_samples": n, "duration_s": duration, "sample_rate_hz": fs,
            "mean_interval_s": dt_step, "interval_jitter": jitter,
            "bandwidth_hz": CONSOLE_BANDWIDTH_HZ, "nyquist_hz": nyquist,
            "skipped_s": args.skip_s,
        },
        "level": {
            "mean_w": mean, "median_w": float(np.median(p)),
            "min_w": float(np.min(p)), "max_w": float(np.max(p)),
            "peak_to_peak_pct": float(np.max(p) - np.min(p)) / mean * 100,
            "raw_std_pct": float(np.std(p, ddof=1)) / mean * 100,
        },
        "noise": {
            "sigma_w": sigma, "sigma_pct": sigma / mean * 100,
            "skew": skew, "excess_kurtosis": excess_kurtosis,
            "skew_se": math.sqrt(6.0 / n), "kurtosis_se": math.sqrt(24.0 / n),
            "correlation_time_s": tau_int * dt_step, "n_effective": n_effective,
            "sem_naive_pct": sigma / math.sqrt(n) / mean * 100,
            "sem_corrected_pct": sigma / math.sqrt(n_effective) / mean * 100,
            "shot_noise_pct": shot_frac * 100,
        },
        "drift": {
            "slope_w_per_s": slope, "intercept_w": intercept,
            "pct_per_hour": slope / mean * 3600 * 100,
            "total_pct": slope * duration / mean * 100,
            "r_squared": r_squared,
            "block_change_pct": (last_block - first_block) / mean * 100,
            "block_s": block * dt_step,
        },
        "allan": {
            "tau_s": taus.tolist(), "adev_frac": adevs.tolist(),
            "adev_rel_error": adev_errors.tolist(),
            "optimal_tau_s": optimal_tau, "floor_pct": floor_adev * 100,
        },
        "spectrum": {
            "nperseg": nperseg,
            "peaks": [{"frequency_hz": float(f), "excess": float(e)}
                      for f, _, e in spectral_peaks(freqs, psd)],
            "band_rms_pct": {k: v * 100 for k, v in bands.items()},
        },
        "dark": dark,
        "steps": [{"time_s": time_s, "size_pct": size * 100}
                  for time_s, size in find_steps(t, p)],
    }

    analysis["repeatability_pct"] = {
        f"{tau:g} s": value * 100
        for tau, value in ((tau, adev_at(taus, adevs, tau))
                           for tau in (0.1, 1.0, 10.0, 60.0, 600.0, 3600.0))
        if math.isfinite(value)
    }
    analysis["budget"] = build_budget(args, analysis)
    analysis["warnings"] = collect_warnings(analysis, meta)
    return analysis


def build_budget(args, analysis) -> dict:
    """Type A from the Allan floor, Type B from whatever you supplied.

    Datasheets quote bounds, not standard deviations, so each Type B tolerance
    is divided by sqrt(3): the standard deviation of a rectangular distribution,
    the least-committed assumption consistent with "somewhere in this interval".
    """
    entries = []

    def add(name, tolerance_pct, note, cancels_in_ratio=False, from_data=False):
        if tolerance_pct:
            entries.append({"name": name, "tolerance_pct": float(tolerance_pct),
                            "standard_pct": float(tolerance_pct) / RECTANGULAR,
                            "cancels_in_ratio": cancels_in_ratio,
                            "from_data": from_data, "note": note})

    add("Sensor calibration", args.cal_tolerance_pct,
        "absolute responsivity of this head at this wavelength; cancels in "
        "ratios taken with the same head and wavelength", cancels_in_ratio=True)
    add("Console readout", args.console_tolerance_pct,
        "meter electronics and range linearity, as a percent of YOUR reading")
    add("Detector nonlinearity", args.linearity_pct,
        "matters when you compare very different power levels")

    # The residual dark offset is a real bias, sized from the blocked-beam
    # segment rather than assumed.
    dark, level = analysis["dark"], analysis["level"]
    if dark["available"] and level["mean_w"]:
        add("Residual dark offset", abs(dark["mean_w"]) / level["mean_w"] * 100,
            "measured with the beam blocked", from_data=True)

    # For white noise the Allan deviation IS the standard deviation of a
    # tau-long mean, and unlike sigma/sqrt(N) it stays honest when the noise is
    # not white. Quote it at the averaging time that actually works best.
    type_a = analysis["allan"]["floor_pct"]
    if not math.isfinite(type_a):
        type_a = analysis["noise"]["sem_corrected_pct"]

    combined_b = math.sqrt(sum(e["standard_pct"] ** 2 for e in entries))
    combined = math.sqrt(type_a ** 2 + combined_b ** 2)
    # A ratio with the same head at the same wavelength divides the calibration
    # out. Offset, linearity and readout do not cancel: they act differently on
    # the two readings.
    surviving = sum(e["standard_pct"] ** 2 for e in entries if not e["cancels_in_ratio"])
    relative = math.sqrt(type_a ** 2 + surviving)

    return {
        "averaging_time_s": analysis["allan"]["optimal_tau_s"],
        "n_supplied": sum(1 for e in entries if not e["from_data"]),
        "type_a_pct": type_a, "type_b": entries,
        "type_b_combined_pct": combined_b,
        "combined_pct": combined, "expanded_k2_pct": 2 * combined,
        "relative_combined_pct": relative, "relative_expanded_k2_pct": 2 * relative,
    }


def collect_warnings(analysis, meta) -> list:
    warnings = []
    level, acquisition, noise = analysis["level"], analysis["acquisition"], analysis["noise"]

    range_w = as_float(meta.get("power_range_w"))
    if range_w:
        if level["max_w"] > 0.95 * range_w:
            warnings.append(
                f"Readings reach {level['max_w'] / range_w * 100:.0f} % of the "
                f"{range_w:.3e} W range -- clipping is likely. Add attenuation.")
        elif level["max_w"] < 0.01 * range_w:
            warnings.append(
                f"Readings use only {level['max_w'] / range_w * 100:.2f} % of the "
                f"range, throwing away ADC resolution and inflating the console's "
                f"percent-of-full-scale error.")

    if analysis["steps"]:
        warnings.append(
            f"{len(analysis['steps'])} abrupt step(s) in the power (section 5). A "
            f"step is a mode hop, a mount or fibre slipping, or a range switch. "
            f"None of those is noise, so none averages away, and each inflates "
            f"the long-tau end of the Allan curve.")

    if acquisition["nyquist_hz"] < CONSOLE_BANDWIDTH_HZ:
        warnings.append(
            f"The console passes {CONSOLE_BANDWIDTH_HZ:.0f} Hz but you sampled to "
            f"{acquisition['nyquist_hz']:.1f} Hz, so noise in between folds back and "
            f"looks slow. Harmless for drift; use --interval_s "
            f"{1 / (2 * CONSOLE_BANDWIDTH_HZ):.3f} or faster to see that band.")

    if acquisition["interval_jitter"] > 0.1:
        warnings.append(
            f"Sample spacing varies by {acquisition['interval_jitter'] * 100:.0f} % "
            f"rms. Allan deviation and the spectrum assume even sampling, so the "
            f"short-tau end and the high-frequency end are approximate.")

    if (abs(noise["skew"]) > 3 * noise["skew_se"]
            or abs(noise["excess_kurtosis"]) > 3 * noise["kurtosis_se"]):
        warnings.append(
            "The noise is not Gaussian, so a single rms understates how often you "
            "land far from the mean -- look for mode hopping or spikes. But check "
            "the correlation time first: with few independent samples this test "
            "fires on perfectly Gaussian data.")

    dark = analysis["dark"]
    if not dark["available"]:
        warnings.append(
            "No beam-blocked segment, so laser noise and detector noise cannot be "
            "separated. Re-record without --dark_s 0.")
    else:
        if math.isfinite(dark["variance_share"]) and dark["variance_share"] > 0.5:
            warnings.append(
                f"The detector accounts for {dark['variance_share'] * 100:.0f} % of "
                f"the observed variance -- you are mostly measuring the power meter. "
                f"More optical power, or a more sensitive range.")
        offset_pct = dark["mean_w"] / level["mean_w"] * 100
        if abs(offset_pct) > 0.01:
            warnings.append(
                f"The {dark['mean_w'] * 1e9:+.3f} nW dark offset ({offset_pct:+.3f} % "
                f"of signal) sits in every reading. Subtract it for an absolute "
                f"power; it cancels in a ratio.")

    drift = analysis["drift"]
    if abs(drift["total_pct"]) > 5 * noise["sigma_pct"] and abs(drift["total_pct"]) > 0.1:
        warnings.append(
            f"Drift ({drift['total_pct']:+.2f} % over the run) dominates the noise. "
            f"If the run started cold, re-analyse with --skip_s to drop the warm-up.")
    return warnings


# --------------------------------------------------------------- the report


def rule(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def format_report(analysis, run: Run) -> str:
    meta = run.meta
    acquisition, level, noise = (analysis["acquisition"], analysis["level"],
                                 analysis["noise"])
    drift, allan, dark, budget = (analysis["drift"], analysis["allan"],
                                  analysis["dark"], analysis["budget"])
    out = []
    add = out.append

    add("=" * 72)
    add("LASER POWER STABILITY REPORT")
    add("=" * 72)
    add(f"File        {run.path.name}")
    add(f"Recorded    {meta.get('started_at', 'unknown')}")
    if meta.get("note"):
        add(f"Note        {meta['note']}")

    add(rule("1. INSTRUMENT"))
    add(f"  Meter            {meta.get('model', '?')} (S/N {meta.get('serial_number', '?')})")
    add(f"  Sensor           {meta.get('sensor_name', '?')} ({meta.get('sensor_type', '?')})")
    add(f"  Calibration      {meta.get('calibration_message', 'unknown')}")
    add(f"  Wavelength       {as_float(meta.get('wavelength_nm'), float('nan')):.1f} nm")
    add(f"  Averaging        console setting: {meta.get('average_count', '?')}")
    add(f"  Range            {as_float(meta.get('power_range_w'), float('nan')):.3e} W")
    add(f"  Bandwidth        {CONSOLE_BANDWIDTH_HZ:.0f} Hz, assumed -- the console's LO")
    add("                   setting. The driver cannot read it; check the panel.")
    add(f"  Samples          {acquisition['n_samples']} over {acquisition['duration_s']:.1f} s "
        f"at {acquisition['sample_rate_hz']:.2f} Hz "
        f"(spacing jitter {acquisition['interval_jitter'] * 100:.0f} %)")
    if acquisition["skipped_s"]:
        add(f"  Skipped          first {acquisition['skipped_s']:.0f} s")

    add(rule("2. SIGNAL LEVEL"))
    add(f"  Mean             {level['mean_w'] * 1e6:.4f} uW")
    add(f"  Min / max        {level['min_w'] * 1e6:.4f} / {level['max_w'] * 1e6:.4f} uW"
        f"   (peak-to-peak {level['peak_to_peak_pct']:.3f} %)")
    add(f"  Raw rms spread   {level['raw_std_pct']:.4f} %   <- noise AND drift together")

    add(rule("3. SHORT-TERM NOISE  (drift removed by a linear fit)"))
    add(f"  sigma            {noise['sigma_w'] * 1e9:.3f} nW  =  {noise['sigma_pct']:.4f} % rms")
    gaussian = (abs(noise["skew"]) < 3 * noise["skew_se"]
                and abs(noise["excess_kurtosis"]) < 3 * noise["kurtosis_se"])
    add(f"  Distribution     skew {noise['skew']:+.3f} (+/-{noise['skew_se']:.3f}), "
        f"excess kurtosis {noise['excess_kurtosis']:+.3f} (+/-{noise['kurtosis_se']:.3f})")
    add(f"                   {'consistent with Gaussian' if gaussian else 'NOT Gaussian -- see the QQ panel'}")
    add(f"  Correlation time {noise['correlation_time_s']:.3f} s")
    add(f"  Effective N      {noise['n_effective']:.0f} of {acquisition['n_samples']} "
        f"samples are independent")
    add(f"  Mean uncertainty {noise['sem_corrected_pct']:.5f} % "
        f"(naive sigma/sqrt(N) would claim {noise['sem_naive_pct']:.5f} %)")
    if math.isfinite(noise["shot_noise_pct"]) and noise["shot_noise_pct"] > 0:
        add(f"  Shot-noise limit {noise['shot_noise_pct']:.5f} % -- the measured noise "
            f"is {noise['sigma_pct'] / noise['shot_noise_pct']:.0f}x larger, so it is")
        add("                   technical and in principle fixable, not fundamental.")

    add(rule("4. DETECTOR FLOOR  (beam blocked)"))
    if dark["available"]:
        add(f"  Dark offset      {dark['mean_w'] * 1e9:+.4f} nW "
            f"({dark['mean_w'] / level['mean_w'] * 100:+.4f} % of signal)")
        add(f"  Dark noise       {dark['std_w'] * 1e9:.4f} nW rms "
            f"= {dark['std_frac_of_signal'] * 100:.5f} % of signal")
        add(f"  Share of variance  {dark['variance_share'] * 100:.1f} % of the observed "
            f"noise is the detector")
        add(f"  Verdict          {'mostly the METER' if dark['variance_share'] > 0.5 else 'mostly the LASER'}")
        add(f"  Detection limit  {dark['detection_limit_w'] * 1e9:.4f} nW (3 sigma dark)")
    else:
        add("  No beam-blocked segment, so laser noise and detector noise cannot")
        add("  be separated. Record one with --dark_s.")

    add(rule("5. DRIFT AND STEPS"))
    add(f"  Linear slope     {drift['pct_per_hour']:+.4f} %/hour  (R^2 = {drift['r_squared']:.3f})")
    add(f"  Over this run    {drift['total_pct']:+.4f} % in {acquisition['duration_s'] / 60:.1f} min")
    add(f"  First vs last    {drift['block_change_pct']:+.4f} % "
        f"(means of the first and last {drift['block_s']:.0f} s)")
    if drift["r_squared"] < 0.3:
        add("                   R^2 is low: this is wander that happens to lean, not")
        add("                   a trend. Measure again and the slope will differ.")
    steps = analysis["steps"]
    if steps:
        add(f"  Abrupt steps     {len(steps)} found:")
        for step in steps:
            add(f"                   t = {step['time_s']:8.0f} s  "
                f"({step['time_s'] / 3600:4.2f} h)   {step['size_pct']:+.2f} %")
    else:
        add("  Abrupt steps     none found")

    add(rule("6. ALLAN DEVIATION  (how much averaging actually buys you)"))
    taus, devs = np.array(allan["tau_s"]), np.array(allan["adev_frac"])
    if taus.size:
        add("  Average each point for...      ...and it repeats to")
        for tau in (0.1, 1.0, 10.0, 60.0, 600.0, 3600.0):
            value = adev_at(taus, devs, tau)
            if math.isfinite(value):
                marker = "  <- best" if abs(tau - allan["optimal_tau_s"]) < tau / 2 else ""
                add(f"  {tau:22.1f} s {value * 100:20.5f} %{marker}")
        add(f"\n  Best averaging   {allan['optimal_tau_s']:.3g} s "
            f"-> floor {allan['floor_pct']:.5f} % rms")
        if allan["optimal_tau_s"] <= taus[0] * 1.01:
            add("  The curve NEVER falls: there is no white-noise region at all.")
            add("  Averaging helps at no timescale. Take readings back-to-back with")
            add("  their reference instead of averaging.")
        elif allan["optimal_tau_s"] >= taus[-1] * 0.99:
            add("  Still falling at the longest tau this run supports, so that floor")
            add("  is a ceiling on your real one -- record for longer.")
        else:
            add("  Averaging longer than that buys nothing: drift has taken over.")
        add("  Read the log-log panel: -1/2 slope = white noise, flat = flicker,")
        add("  +1/2 = random walk, +1 = drift.")
    else:
        add("  Too few samples for an Allan deviation.")

    add(rule("7. FREQUENCY CONTENT"))
    spectrum = analysis["spectrum"]
    add(f"  Nyquist          {acquisition['nyquist_hz']:.2f} Hz; console passes "
        f"{CONSOLE_BANDWIDTH_HZ:.0f} Hz")
    add("  rms by band:")
    for label, value in spectrum["band_rms_pct"].items():
        if math.isfinite(value):
            add(f"    {label:>14}  {value:.5f} %")
    if spectrum["peaks"]:
        add("  Lines above the noise floor:")
        for peak in spectrum["peaks"]:
            hint = ""
            if min(abs(peak["frequency_hz"] - f) for f in (50, 60, 100, 120)) < 1.0:
                hint = "   <- mains pickup"
            add(f"    {peak['frequency_hz']:8.3f} Hz   {peak['excess']:5.1f}x floor{hint}")
    else:
        add("  No narrow lines -- the noise is broadband.")

    add(rule("8. UNCERTAINTY BUDGET"))
    add(f"  For one measurement averaged over {budget['averaging_time_s']:.3g} s "
        f"(the Allan minimum).")
    if not budget["n_supplied"]:
        add("")
        add("  NO TYPE B TERMS SUPPLIED. What follows is the STATISTICAL spread of")
        add("  your own data -- it is NOT your accuracy. The head's calibration")
        add("  alone is a few percent and dwarfs it. Pass --cal_tolerance_pct and")
        add("  friends off your spec sheet for a real budget.")
    else:
        add("  Type B tolerances are read as rectangular bounds (divided by sqrt 3).")
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
    if any(e["cancels_in_ratio"] for e in budget["type_b"]):
        add(f"  RELATIVE   ratios with this same head and wavelength are good to")
        add(f"             {budget['relative_expanded_k2_pct']:.4f} % (k=2) -- the "
            f"calibration cancels.")
    add("")
    for entry in budget["type_b"]:
        add(f"  * {entry['name']}: {entry['note']}")

    add(rule("9. WHAT TO DO WITH THIS"))
    dominant = max(budget["type_b"], key=lambda e: e["standard_pct"], default=None)
    if not budget["n_supplied"]:
        add("  This run tells you how REPEATABLE the meter is, not how ACCURATE.")
    elif dominant and dominant["standard_pct"] > budget["type_a_pct"]:
        add(f"  Absolute accuracy is limited by {dominant['name'].lower()} "
            f"({dominant['standard_pct']:.2f} %), not by")
        add(f"  the laser ({budget['type_a_pct']:.4f} %). Averaging will not help; only a")
        add("  fresh calibration or a side-by-side reference detector will.")
    if taus.size and allan["optimal_tau_s"] <= taus[0] * 1.01:
        add("  Averaging buys nothing. Take each reading back-to-back with its reference.")
    elif math.isfinite(allan["optimal_tau_s"]):
        add(f"  Average each data point for ~{allan['optimal_tau_s']:.3g} s.")
    if abs(drift["pct_per_hour"]) > 0.1:
        add(f"  At {drift['pct_per_hour']:+.2f} %/hour, re-reference every "
            f"~{abs(0.1 / drift['pct_per_hour'] * 60):.0f} min to hold 0.1 %.")
    if steps:
        add(f"  {len(steps)} step(s) of around {max(abs(s['size_pct']) for s in steps):.1f} % "
            f"will not average away. Interleave")
        add("  signal and reference so a step lands in both and divides out.")

    if analysis["warnings"]:
        add(rule("WARNINGS"))
        for warning in analysis["warnings"]:
            add(f"  ! {warning}")
    add("")
    return "\n".join(out)


# --------------------------------------------------------------- the figure


def make_figure(analysis, run: Run, args, path: Path):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, p, mean = run.t, run.p, analysis["level"]["mean_w"]
    drift, residual = analysis["drift"], run.residual
    residual_pct = residual / mean * 100
    sigma = float(np.std(residual_pct, ddof=1))

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    figure.suptitle(f"{run.path.name}    {mean * 1e6:.4f} uW    "
                    f"noise {analysis['noise']['sigma_pct']:.4f} % rms    "
                    f"drift {drift['pct_per_hour']:+.3f} %/h", fontsize=11)

    # 1. the run itself, with the drift line and any steps
    ax = axes[0, 0]
    ax.plot(t, p * 1e6, lw=0.6, color="0.4")
    ax.plot(t, (drift["slope_w_per_s"] * t + drift["intercept_w"]) * 1e6,
            color="crimson", lw=1.5, label=f"{drift['pct_per_hour']:+.3f} %/h")
    for index, step in enumerate(analysis["steps"]):
        ax.axvline(step["time_s"], color="darkorange", lw=1, ls="--",
                   label="abrupt step" if index == 0 else None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("power (uW)")
    ax.set_title("Power vs time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. residual histogram against the Gaussian it is supposed to be
    ax = axes[0, 1]
    ax.hist(residual_pct, bins=60, density=True, color="steelblue", alpha=0.75)
    grid = np.linspace(residual_pct.min(), residual_pct.max(), 200)
    ax.plot(grid, np.exp(-0.5 * (grid / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)),
            color="crimson", lw=1.5, label="Gaussian")
    if analysis["dark"]["available"]:
        for sign in (-1, 1):
            ax.axvline(sign * analysis["dark"]["std_frac_of_signal"] * 100,
                       color="darkgreen", ls="--", lw=1,
                       label="detector sigma" if sign > 0 else None)
    ax.set_xlabel("deviation from trend (%)")
    ax.set_ylabel("density")
    ax.set_title(f"Noise distribution (sigma = {sigma:.4f} %)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Allan deviation, with the slopes you are meant to recognise
    ax = axes[0, 2]
    taus = np.array(analysis["allan"]["tau_s"])
    devs = np.array(analysis["allan"]["adev_frac"]) * 100
    if taus.size:
        ax.errorbar(taus, devs, yerr=devs * np.array(analysis["allan"]["adev_rel_error"]),
                    fmt="o-", ms=3, lw=1, color="navy", ecolor="0.7", capsize=2)
        ax.plot(taus, devs[0] * (taus / taus[0]) ** -0.5, ls=":", color="crimson",
                lw=1, label="white noise, -1/2")
        best = analysis["allan"]["optimal_tau_s"]
        if math.isfinite(best):
            ax.axvline(best, color="darkgreen", ls="--", lw=1,
                       label=f"best tau = {best:.3g} s")
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
    freqs, psd = welch_psd(residual / mean, analysis["acquisition"]["sample_rate_hz"],
                           analysis["spectrum"]["nperseg"])
    if freqs.size > 1:
        ax.loglog(freqs[1:], psd[1:], lw=0.7, color="darkslategray")
        for peak in analysis["spectrum"]["peaks"][:3]:
            ax.axvline(peak["frequency_hz"], color="crimson", ls=":", lw=1)
    ax.axvline(CONSOLE_BANDWIDTH_HZ, color="0.5", ls="-.", lw=1, label="console BW")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("RIN (1/Hz)")
    ax.set_title("Relative intensity noise spectrum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # 5. normal probability plot -- the tails show up here, not in a histogram
    ax = axes[1, 1]
    sample = np.sort(residual_pct)
    if sample.size > 4000:
        sample = sample[np.linspace(0, sample.size - 1, 4000).astype(int)]
    normal = NormalDist()
    theoretical = np.array([normal.inv_cdf((i + 0.5) / sample.size)
                            for i in range(sample.size)])
    ax.plot(theoretical, sample, ".", ms=2, color="steelblue")
    limit = max(abs(theoretical[0]), abs(theoretical[-1]))
    ax.plot([-limit, limit], [-limit * sigma, limit * sigma], color="crimson", lw=1)
    ax.set_xlabel("theoretical quantile (sigma)")
    ax.set_ylabel("observed deviation (%)")
    ax.set_title("Normal probability plot")
    ax.grid(alpha=0.3)

    # 6. how long the noise remembers itself
    ax = axes[1, 2]
    rho = autocorrelation(residual)
    lags = np.arange(rho.size) * analysis["acquisition"]["mean_interval_s"]
    tau_int = analysis["noise"]["correlation_time_s"]
    show = min(rho.size, max(50, int(20 * tau_int / analysis["acquisition"]["mean_interval_s"])))
    ax.plot(lags[:show], rho[:show], lw=1, color="purple")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(tau_int, color="crimson", ls="--", lw=1, label=f"tau_int = {tau_int:.2f} s")
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path, default=None,
                        help="the run to analyse; default is the newest in --data_dir")
    parser.add_argument("--data_dir", type=Path,
                        default=Path.home() / "Desktop" / "power_meter_data",
                        help="where to look when no file is given")
    parser.add_argument("--skip_s", type=float, default=0.0, metavar="S",
                        help="drop this much from the start (warm-up)")
    parser.add_argument("--max_s", type=float, default=None, metavar="S",
                        help="analyse only up to this time")

    budget = parser.add_argument_group(
        "uncertainty budget (Type B)",
        "All default to zero -- nothing enters the budget unless you put it there, "
        "off your own spec sheet. For a PM100D + S130VC: calibration +/-3 % over "
        "451-1000 nm and +/-5 % outside it; linearity +/-0.5 %; console +/-0.2 % of "
        "FULL SCALE, so at a third of the range that is ~0.6 % of your reading.")
    budget.add_argument("--cal_tolerance_pct", type=float, default=0.0,
                        help="sensor calibration tolerance")
    budget.add_argument("--console_tolerance_pct", type=float, default=0.0,
                        help="console readout error as a percent of YOUR reading")
    budget.add_argument("--linearity_pct", type=float, default=0.0,
                        help="detector nonlinearity across power levels")

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
    report = format_report(analysis, run)
    print(report)

    stem = path.with_suffix("")
    Path(f"{stem}_report.txt").write_text(report, encoding="utf-8")
    Path(f"{stem}_metrics.json").write_text(
        json.dumps(analysis, indent=2, default=json_safe), encoding="utf-8")
    print(f"Saved {stem}_report.txt")
    print(f"      {stem}_metrics.json")
    if not args.no_figure:
        make_figure(analysis, run, args, Path(f"{stem}_stability.png"))
        print(f"      {stem}_stability.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
