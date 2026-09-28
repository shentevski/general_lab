# Mueller matrix at every wavelength (LED + spectrometer)

The spectral version of [`../mueller_matrix`](../mueller_matrix): the laser is replaced by a
broadband LED (MBB1F1, 470–850 nm) and the signal power meter by a CCT10 spectrometer. A
whole spectrum is recorded at every QWP step, so one run gives M(λ), about 480–820 nm.

```
LED → collimator → polarizer → beamsplitter → HWP (H V D A) / QWP (R L) → [sample] → QWP (rotating) → analyzer → fibre → CCT10
                                    │
                                    └→ reference power meter (optional LED monitor)
```

| file | what |
|---|---|
| `measure_mueller_poly.py` + `.json` | interactive measurement |
| `analyze_mueller_poly.py` + `.json` | M(λ), decompositions, error bars, figures |
| `QWP_analyzer_characterization_poly.py` + `.json` | calibrates the polarimeter at every wavelength: QWP retardance δ(λ), QWP zero, beam walk, analyzer leak |
| `poly_common.py` | wavelength bins, spectrum files, calibration curves |

The physics is **imported** from `../mueller_matrix`: the Stokes inversion for any QWP
retardance, the retarder split, the Cloude check and every reported parameter. That folder
must sit next to this one. Hardware: `thorlabs_spectrometer` (Windows only), `kcube`,
`thorlabs_powermeter` (only with a reference meter). `--simulate` runs anywhere.

## What changes with an LED and a spectrometer

**What stays easy.** Three things carry over from the laser setup unchanged:

- The analyzer is fixed, so the light entering the spectrometer always has the same
  polarization. The grating's and the fibre's polarization dependence don't matter.
- Every input state is measured, not assumed. The HWP and QWP that make the states need
  no calibration at any wavelength. A 520 nm HWP makes an elliptical "D" at 800 nm, and
  that's fine. The coverage check after each state tells you whether the states still
  spread over the sphere at every wavelength.
- The spectrometer's spectral response cancels, since every number is a ratio at the same
  pixel.

**What now needs a curve instead of a number.** The rotating QWP's retardance δ(λ). The
zero stays one number for zero‑order and quartz/MgF₂ achromatic plates, because their axis
doesn't move with wavelength. For a superachromatic plate, set `zero_source: "file"`.

**Spectrometer.**
- **One exposure for the whole run.** The Stokes fit assumes a linear detector. At the
  start, the script checks the exposure on a bright state. With `exposure_ms: null` it
  sets the brightest pixel to `target_fill` (60%) of full scale.
- **A shutter dark before and after every sweep,** interpolated in time. A leftover
  offset looks exactly like lost polarization.
- **Raw counts are saved,** with amplitude correction off and no SDK dark subtraction.
- **Every spectrum is checked for clipping.**

**LED.**
- Warm it up for 30–60 min at constant current.
- The reference meter corrects the total power, not a change in spectral shape.
- At the start, with the LED blocked, the script measures the ambient light reaching the
  spectrometer. Unmodulated light also looks like lost polarization.

**Fibre coupling** is more sensitive to beam position than the power meter's large sensor,
so beam walk from the rotating QWP is larger. The 360° sweeps fit it separately, and the
characterization measures it. A diffuser or a large‑core fibre reduces it.

## Order of work

1. **Thorlabs retardance curve → CSV.** Open the Thorlabs Excel file for your QWP and save
   it as CSV. The reader finds the columns by the header row ("Wavelength", "Retardance"),
   even with product text beside the data. Put the file in this folder and set
   `"calibration_file": "<name>.csv"` in `measure_mueller_poly.json`. It is the starting
   guess for the calibration.
2. **`full_scale_counts`, once.** 65535 (16‑bit) is assumed. Saturate a pixel and read the
   raw value to check it. The clipping check also detects a flat‑topped spectrum on its
   own.
3. **Calibrate:** `python QWP_analyzer_characterization_poly.py`. No sample. States:
   - **H**, with both state wave plates OUT: the polarizer alone gives the same linear
     state at every wavelength. This gives the retardance.
   - **R and L**, which give the zero.
   - Optionally **D and A** (a cross‑check) and **V** (the analyzer leak).

   Paste the printed lines into the JSON files.
4. **Measure:** `python measure_mueller_poly.py`. Rehearse first with `--simulate`.
5. **Analyze:** set `run_dir` in `analyze_mueller_poly.json`, then
   `python analyze_mueller_poly.py`.

## The calibration assumes DOP = 1

The characterization finds the retardance that makes every sweep's degree of polarization
(DOP) equal 1, as in the laser version. An unpolarized LED through a polarizer of
extinction ratio ER only reaches DOP = 1 − 2/ER: 0.998 at 1000:1. That makes the retardance
come out about 0.06° low, well inside λ/300 (1.2°). The error budget shows this as a
one‑sided term, "input DOP" (`input_dop_min`).

Outside the polarizer's band (LPVISC: 510–800 nm) the extinction drops and the term grows.
In simulation, the calibrated retardance was within about 0.06° across the middle of the
band and 0.3° off at 495 nm. Read the budget at the edges, or narrow `wl_min_nm` /
`wl_max_nm`. The zero doesn't depend on the DOP: it comes from the phase of the circular
term.

The retardance is smoothed across wavelength with a polynomial in 1/λ (`smooth_order`),
since a wave plate's retardance is a smooth curve. The per‑wavelength error goes into the
file, and the analysis uses it as the systematic tolerance at each wavelength.

## Key settings

`measure_mueller_poly.json` (the characterization reads hardware and the assumed
calibration from it too):

| key | meaning |
|---|---|
| `qwp_zero_deg` | stage reading where the QWP fast axis is along the analyzer |
| `calibration_file` | QWP retardance vs wavelength (CSV); `null` = `qwp_retardance_waves` at every wavelength |
| `zero_source` | `constant` (`qwp_zero_deg`) or `file` (the `zero_deg` column) |
| `exposure_ms` | `null` = set automatically at the start; a number fixes it |
| `hw_average` | frames the spectrometer averages per QWP step |
| `target_fill`, `full_scale_counts` | automatic exposure target; raw counts at saturation |
| `dark_frames` | spectra averaged per shutter dark |
| `wl_min_nm`, `wl_max_nm` | band for checks and live feedback |
| `ref_pm_serial` | reference meter behind the beamsplitter; `null` = none |
| `block_at_start` | block the LED at the start: zero the reference meter, measure ambient light |

`analyze_mueller_poly.json`:

| key | meaning |
|---|---|
| `calibration_file`, `qwp_zero_deg`, … | `null` = what the run recorded; a value overrides it for re‑analysis (`"none"` = no file) |
| `wl_min_nm`, `wl_max_nm`, `bin_nm` | analysis band and bin width (5 nm; the CCT10 resolves ~2 nm) |
| `min_signal_rel` | leave out bins with less light than this fraction of the brightest |
| `retardance_uncertainty_waves` | `null` = from the calibration file, per wavelength (else 1/300) |
| `qwp_zero_uncertainty_deg` | zero tolerance for the systematic errors |
| `report_wl_nm`, `fit_plot_wl_nm` | wavelengths of the printed table, the printed matrix and the fit figure |

`QWP_analyzer_characterization_poly.json`:

| key | meaning |
|---|---|
| `repeats` | sweeps per state |
| `bin_nm` | bin width of the calibration curve |
| `smooth_order` | polynomial order in 1/λ for the retardance (`null` = raw per bin) |
| `zero_smooth_order` | `0` = one zero for all wavelengths |
| `input_dop_min` | 1 − 2/ER of the input polarizer, for the error budget |
| `compare_file` | a curve to compare with, e.g. the Thorlabs CSV |

## Output

- **Each run folder** (`Desktop/mueller_poly_<date>_<time>/`) holds:
  - `run.json`;
  - one `.npz` per sweep, with raw counts (100 × 2048), the two darks, the reference meter,
    angles and times;
  - `qwp_calibration_used.csv`, the calibration file copied in.
- **`analysis/`** holds `results.json`, `mueller_spectrum.csv`, `parameters.csv`, and the
  figures `mueller_spectrum.png`, `parameters.png`, `fits.png`, `spectra.png` and
  `led_monitor.png`.
- **`characterization/`** holds `qwp_calibration.csv` (point `calibration_file` at it),
  `characterization.json` and `characterization.png`.

## Rehearsal

`--simulate` on all three scripts uses:

- **LED:** MBB1F1‑like spectrum, drifting.
- **Input polarizer:** LPVISC‑like, 1000:1, falling off outside 510–800 nm.
- **State plates:** zero‑order, designed for 520 nm.
- **Sample:** a dispersive diattenuator, retarder and depolarizer.
- **Spectrometer:** noise, a drifting dark, stray light and clipping.

A simulated run is compared with the truth at the end of the analysis. In one simulated
chain, the characterization found the zero to 0.001° and the retardance to about 0.06°
mid‑band. The Mueller matrix measured with that calibration was within 0.002 of the truth
(median over wavelength). The same data analysed with only the "Thorlabs" curve (1.5% off)
and the old zero (0.7° off) was off by 0.034, with a 1.5° error in the fast axis, and came
out unphysical at 11 wavelengths.

**The calibration trap still applies.** Air measures as a perfect identity whatever the
calibration, so check the calibration with a known sample that isn't the identity.
