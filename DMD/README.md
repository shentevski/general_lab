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

**1 = single measurement.** Choose 1, 2 or 3 wavelengths, then the exposure. Then **sample
IN** → Enter → spectrum, and **sample OUT** → Enter → spectrum. If anything clipped, it
offers to measure again with a shorter exposure. Lines that would overlap on the DMD are
refused. Files go in `<data_dir>/single/` as `<date>_<time>_<n>wl_*`:
`sample_in.csv`, `sample_out.csv` (`wavelength_nm`, `net_counts`, `raw_counts`,
`background_counts`) and `meta.json`.

**2 = scan.** Choose start, stop and step (nm) and the exposure. The whole scan runs with the
sample IN, then the whole scan with it OUT, so the sample is moved only once.
- **1 = one peak:** one line steps across the range.
- **2 = scanning + stationary peak:** a second line stays at the wavelength you choose,
  displayed together with the scanning one at every step. Steps where the two lines would
  overlap on the DMD are skipped. Steps where their peaks' integration windows overlap are
  NaN in the analysis.

Files go in `<data_dir>/scan/<date>_<time>_1peak/` (or `_2peak/`):
`sample_in.csv` and `sample_out.csv` (one column per step), `steps.csv`, `background.csv`
and `meta.json`, all prefixed with the date and time.

The scan prints how far the measured peak centres are from the calibration, as a running
check of it. In scan 2, it also prints how steady the stationary peak's in/out ratio is over
the scan: a spread there is source drift between the two passes, or the scanning light
acting on the sample.

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

- **Single:** on the whole spectrum (`<…>_subtract.csv`, …), and on the counts integrated
  over each peak (`<…>_peaks.csv`, printed as a table), plus `<…>_analysis.png`.
- **Scan:** on the counts integrated over the peak at every step (`<…>_results.csv`), against
  the wavelength, plus `<…>_analysis.png`.

A peak is integrated over its centre ± `band_halfwidth_nm` (null: ± its FWHM). The centre is
found in the sample‑OUT spectrum within `search_nm` of the calibrated wavelength.

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
