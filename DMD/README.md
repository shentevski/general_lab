# DMD wavelength selection

A prism spreads the spectrum across the DMD. A line of mirrors picks a narrow band, and a
second prism recombines what the line passes. The light then goes through the sample to the
spectrometer.

```
source → prism → spectrum on the DMD (lines pick bands) → prism (recombine) → [sample] → CCT10
```

| file | what |
|---|---|
| `dmd_calibration.py` + `dmd_calibration.json` | steps one line across the DMD, finds each peak's centre wavelength → `dmd_calibration.csv` |
| `dmd_measure.py` + `dmd_measure.json` | single measurements (1–3 wavelengths) and scans, sample in / out, analysis |
| `dmd_common.py` | hardware, simulator, peak finder, calibration file |

Hardware: [`dlp6500-hid`](https://github.com/shentevski/dlp6500-hid) (`import dlpc900_hid`), the
same library as `toggle_dmd_patterns.py`, and
[`thorlabs_spectrometer`](https://github.com/shentevski/thorlabs_spectrometer) (Windows only).
Close the TI LightCrafter GUI first. `--simulate` runs anywhere and rehearses every prompt,
file and figure.

Line positions are **offsets in mirror rows from the chip centre**, drawn with the `*_rows`
patterns (exactly `width` rows at every offset), as in `toggle_dmd_patterns.py`.

## Order of work

1. **Check `dmd_calibration.json`.** `orientation` (90 = vertical lines, like the toggle
   script), `line_on`, and the offset range. `line_on: false` means the lines are OFF mirrors
   on an ON field, as in the toggle script. If that's the wrong way round, the calibration
   stops and tells you to flip it.
2. **Rehearse:** `python dmd_calibration.py --simulate`, then `python dmd_measure.py --simulate`.
3. **Calibrate:** `python dmd_calibration.py`. Look at `calibration.png`: the residuals
   should be small and without structure.
4. **Measure:** `python dmd_measure.py`. It takes the newest calibration automatically.

Recalibrate after anything moves: the prisms, the DMD, or the fibre.

## Calibration

1. The DMD passes everything (the whole spectrum). The script compares this with blocking
   everything. If blocking is brighter, `line_on` is the wrong way round and the script stops.
2. You type an exposure and see how full the brightest pixel gets. Pass‑everything is the
   brightest any line can be, so nothing clips during the scan. Aim for 50–70 %.
3. The line steps from `offset_start` to `offset_stop`, `offset_step` rows at a time. A
   background is taken before and after (interpolated in time) and subtracted.
4. For each position it finds the peak in `wl_min_nm`–`wl_max_nm`: its centre
   (`peak_method`), FWHM and height. A position isn't used if the peak is dimmer than
   `min_peak_counts`, clipped, or at the band edge.
5. It fits a polynomial wavelength(offset) of `fit_order`. A prism's dispersion is smooth but
   not linear: in the simulation it goes from 0.08 to 0.18 nm per row across the chip.
   Positions more than `max_residual_nm` off the fit are dropped one at a time and the fit is
   redone. The fit must be monotonic, or the measurement script refuses it.

Output in `<data_dir>/calibrations/<date>_<time>/`:

| file | contents |
|---|---|
| `dmd_calibration.csv` | `offset_rows`, `wavelength_nm` (the fit, blank outside the measured range), `measured_nm`, `fwhm_nm`, `peak_counts`, `residual_nm`, `used`, `status` |
| `calibration_spectra.csv` | every background‑subtracted spectrum, one column per offset |
| `background.csv` | the backgrounds before and after |
| `calibration_run.json` | settings, fit coefficients, summary; the measurement checks the geometry against it |
| `calibration.png` | spectra, fit, residuals, FWHM |

`dmd_measure.py` only reads `offset_rows` and `wavelength_nm`, and interpolates linearly
between rows. A wavelength goes to the nearest whole mirror row; the script prints the
wavelength at that row.

## Measurement

`python dmd_measure.py` opens a menu that stays connected between measurements (`q` quits;
Ctrl‑C during a measurement returns to the menu).

After the exposure, every measurement also asks for the **number of frames**. Each frame is
a separate spectrum (`hw_average` × exposure). Results are the mean over the frames, and the
spread of the frames gives the error bars (see [Error bars](#error-bars)). Enter keeps the
last exposure and number of frames.

**1 = single measurement.** Choose 1, 2 or 3 wavelengths, the exposure and the frames. Then
**sample IN** → Enter → frames, and **sample OUT** → Enter → frames. If anything clipped, it
offers to measure again with a shorter exposure. Lines that would overlap on the DMD are
refused. Files go in `<data_dir>/single/` as `<date>_<time>_<n>wl_*`:
- `sample_in.csv`, `sample_out.csv`: `wavelength_nm`, `net_counts` (mean of the frames −
  background), `net_std` (standard deviation of the frames, per pixel), `raw_counts`,
  `background_counts`;
- `frames.npz`: every frame and every background frame, raw (what the analysis reads);
- `meta.json`.

**2 = scan.** Choose start, stop and step (nm), the exposure and the frames. The whole scan
runs with the sample IN, then the whole scan with it OUT, so the sample is moved only once.
- **1 = one peak:** one line steps across the range.
- **2 = scanning + stationary peak:** tests whether the two wavelengths together do
  something that each alone doesn't. At every step the DMD shows three patterns, each for
  the full number of frames:
  1. the scanning line alone;
  2. both lines;
  3. the stationary line alone.

  A step takes about 3× as long as in scan 1. Steps where the two lines would overlap on
  the DMD (closer than `line_width_rows`, ~3 nm) are skipped. That is the only gap: each
  peak's own transmission comes from the spectra where it is alone, so overlapping peaks
  in the spectrometer don't matter. See [Two wavelengths together](#two-wavelengths-together).

Files go in `<data_dir>/scan/<date>_<time>_1peak/` (or `_2peak/`), all prefixed with the
date and time:
- `sample_in.csv`, `sample_out.csv`: mean of the frames − background, one column per step.
  In scan 2 there is one file per pattern: `sample_in_scan.csv`, `sample_in_both.csv`,
  `sample_in_stat.csv`, and the same for `out`;
- the same with `_std`: the per‑pixel standard deviation of the frames;
- `steps.csv`, `background.csv`, `frames.npz` (every frame, raw) and `meta.json`.

The scan prints how far the measured peak centres are from the calibration, as a running
check of it. In scan 2, it also prints how steady the stationary peak's in/out ratio is over
the scan (alone): a spread there is source drift.

### Analysis

After every measurement it asks which analysis to run (one or more, e.g. `1,3`; `a` = all;
Enter = none):

| | | |
|---|---|---|
| 1 | subtract | in − out |
| 2 | divide | in / out (the transmission) |
| 3 | absorbance | −log10(in / out), positive when the sample absorbs |

"in" and "out" are the background‑subtracted sample‑in and sample‑out spectra. The ratios are
left out (NaN) where the sample‑out signal is below `min_signal_counts`.

- **Single:** on the counts integrated over each peak, which are the results
  (`<…>_peaks.csv`, printed as `value +- error`). The same formulas are also applied pixel by
  pixel across the spectrum, for looking at only (`<…>_subtract.csv`, …). Plus
  `<…>_analysis.png`.
- **Scan:** on the counts integrated over the peak at every step (`<…>_results.csv`), against
  the wavelength, plus `<…>_analysis.png`.

A peak is integrated over its centre ± `band_halfwidth_nm` (null: ± its FWHM). The centre is
found in the sample‑OUT spectrum within `search_nm` of the calibrated wavelength.

### Two wavelengths together

Scan 2 only. If the wavelengths don't act on each other, the light with both lines on is
the sum of each alone, pixel by pixel:

```
both = scan + stat        →   excess light  = Σ both / Σ (scan + stat) − 1
```

Σ runs over the two peaks' windows. It is computed for every step with the sample IN, and
with the sample OUT as the **control**. With nothing in the beam, the control must be zero;
whatever is left is the instrument (detector nonlinearity where the peaks overlap, scatter
that depends on the pattern, drift between the three patterns). The result corrected by
the control is

```
extra absorbance = −log10[ (1 + excess_in) / (1 + excess_out) ]
```

It is positive if the sample absorbs more when both wavelengths are on.

Output:
- **Printed:**
  - the excess light for the control and the sample, each with a chi2 per step. Near 1
    means noise only. A control well above 1 means the instrument or drift adds noise, and
    the significance below is overstated;
  - the extra absorbance, with the steps beyond 3 standard errors listed, or "no effect
    beyond the noise". Significance always uses the standard error of the mean, whatever
    `error` is set to.
- **`<…>_results.csv`:** `excess_in_pct`, `excess_out_pct`, `extra_absorbance` (each with
  its error), `extra_absorbance_z` (value / standard error), and `other_in_counts` /
  `other_out_counts`: excess light **outside** the two peaks, which is where new light such
  as emission would appear.
- **`<…>_interaction_in.csv`, `<…>_interaction_out.csv`:** both − scan − stat per pixel, one
  column per step.
- **`<…>_interaction.png`:**
  - the excess light (IN and control);
  - the extra absorbance;
  - a map of both − scan − stat per pixel (sample IN) against the scanning wavelength. With
    no effect it is noise everywhere. An effect shows up as a feature on the scanning
    peak's diagonal or on the stationary peak's horizontal line.

Checked in simulation:
- **No effect:** all steps within 3 standard errors (largest 2.3), chi2 per step 0.83.
- **Injected effect** (1 % extra absorption when both are on, around a scanning wavelength
  of 650 nm, i.e. extra absorbance 0.0044): found at 640–660 nm, 0.0047 ± 0.0009 at 650 nm.
  The residual from the injected curve had chi2 per step 0.81.

### Error bars

1. Every frame has the background subtracted, and the peak is integrated frame by frame.
2. The mean of the N integrals is I. Their spread gives the error e:
   - `error: "std"` (default): e = standard deviation of the N frames;
   - `error: "sem"`: e = standard error of the mean, √(std²/N + e_bg²), where e_bg is the
     error of the averaged background.
3. The error is propagated, with sample in and sample out independent:

| | value | error |
|---|---|---|
| subtract | I_in − I_out | √(e_in² + e_out²) |
| divide | T = I_in / I_out | T · √((e_in/I_in)² + (e_out/I_out)²) |
| absorbance | A = −log10 T | √((e_in/I_in)² + (e_out/I_out)²) / ln 10 |

The CSV columns carry the type in their name, e.g. `absorbance_std` or `absorbance_sem`.
`--analyze` with the other `error` setting recomputes them from the saved frames. In the
figures, the values are labelled `value ± error` and the title states which error it is. The
scan legends give the median error bar, because the bars are often smaller than the markers.

**What the error bars don't include:** source drift and sample repositioning between the IN and
OUT measurements. The frames are taken back to back, so they can't see either. In simulation
the frame errors match the real scatter of repeated measurements once drift is switched off;
with the simulator's 0.3 % drift the scatter was about 20 % larger. To include drift, repeat
the whole in/out measurement a few times, or run the empty test (air for both), and use the
spread of the results.

To analyse saved data again, with another analysis or other settings, and no hardware:

```
python dmd_measure.py --analyze <path to …_meta.json, or a scan folder>
```

## Settings

`dmd_calibration.json`. The first block is **shared**: the measurement reads it from this
file too, so the DMD geometry is set in one place (putting these keys in `dmd_measure.json`
is refused).

| key | meaning |
|---|---|
| `dmd_width`, `dmd_height` | 1920 × 1080 (DLP6500) |
| `orientation` | 0, 90, 45 or −45 degrees (90 = vertical lines) |
| `line_on` | `false`: lines are OFF mirrors on an ON field (as in the toggle script) |
| `dmd_settle_s` | wait after a pattern change before a spectrum |
| `spectrometer_id` | `null` = the first one found |
| `hw_average` | frames the spectrometer averages per spectrum |
| `full_scale_counts` | raw counts at saturation |
| `background` | `dmd`: spectrum with the DMD blocking everything (removes dark, ambient light and DMD scatter); `shutter`: spectrometer shutter (dark only) |
| `background_frames` | spectra averaged per background |
| `data_dir` | `null` = `Desktop/DMD_data` (simulated runs: `…/simulated`) |
| `offset_start`, `offset_stop`, `offset_step` | the calibration scan, in mirror rows from the centre |
| `line_width_rows` | line width for the calibration |
| `exposure_ms` | offered at the exposure prompt (Enter takes it) |
| `wl_min_nm`, `wl_max_nm` | where peaks are looked for |
| `min_peak_counts` | dimmer peaks are not used |
| `peak_method` | `centroid` (weights: counts − threshold), `gaussian` (parabola through ln counts), `max` (brightest pixel, refined) |
| `peak_threshold_rel` | pixels above this fraction of the peak give its centre |
| `fit_order` | polynomial order; `null` = no fit, straight lines between the measured centres |
| `max_residual_nm` | positions further than this from the fit are dropped |

`dmd_measure.json`:

| key | meaning |
|---|---|
| `calibration_file` | `null` = the newest in `<data_dir>/calibrations/`; or a CSV (or its folder) |
| `line_width_rows` | line width for measurements (narrower = narrower band, less light) |
| `exposure_ms` | offered at the first exposure prompt; after that, the last one you used |
| `frames` | offered at the first frames prompt; after that, the last one you used |
| `error` | `std`: error bars = standard deviation of the frames; `sem`: standard error of the mean (+ background) |
| `scan_start_nm`, `scan_stop_nm`, `scan_step_nm` | offered at the scan prompts (`null` = the calibrated range) |
| `band_halfwidth_nm` | peaks are integrated over centre ± this; `null` = ± the FWHM |
| `search_nm` | how far from the calibrated wavelength a peak is looked for |
| `min_signal_counts` | in/out and absorbance only where sample‑out has at least this |

Any key can be overridden for one run on the command line, e.g. `--line-width-rows 10`.
A misspelled key in a JSON is rejected with the list of valid keys.

## Rehearsal

`--simulate` uses a white‑LED‑like source, a non‑linear prism, 0.2 % DMD scatter, a sample
with absorption bands at 560 and 650 nm, and a noisy 2048‑pixel spectrometer that clips at
full scale. It models OFF mirrors as the ones that pass light, so `line_on: true` is caught.
In simulation the calibration matches the true line centres to about 0.015 nm. The measured
transmissions match the sample's, e.g. 0.684 vs 0.685 at 620 nm.
