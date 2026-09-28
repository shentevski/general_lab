# Mueller matrix of a sample

Rotating-QWP polarimeter (QWP on K-Cube 28000005 → fixed analyzer → PM100D + S130VC),
input states made by hand, a second PM100D as laser monitor. Built on `polarization_toolkit`.

```
laser → polarizer → beamsplitter → HWP (H V D A) / QWP (R L) → [sample] → QWP (rotating) → analyzer → signal meter
                         │
                         └→ reference meter (laser monitor)
```

| file | what |
|---|---|
| `measure_mueller.py` + `measure_mueller.json` | interactive measurement |
| `analyze_mueller.py` + `analyze_mueller.json` | M, decompositions, error bars, figures |
| `QWP_analyzer_characterization.py` + `.json` | calibrates the polarimeter itself: QWP zero and retardance, beam walk, analyzer leak |
| `measure_transmission.py` + `.json` | sample transmittance with the two power meters only, no polarimeter (checks M00) |
| `mueller_common.py` | Stokes extraction shared by all (one copy of the maths) |

Both scripts read their JSON automatically. A flag on the command line overrides
the JSON for that run; `--config other.json` uses a different file. A misspelled
key in the JSON is rejected with the list of valid keys.

## Measure

```
python measure_mueller.py              # rehearse first with: --simulate
```

For each input state:

1. set the state by hand, type a label (`H`, `D`, `R`, anything)
2. **remove the sample** → Enter → the input state is measured
3. **put the sample back** → Enter → the state is measured through the sample

After each state it prints the rank and condition number of what you have. Press
Enter with no label to finish. Everything is saved after every state, in
`Desktop/mueller_<date>_<time>/` (`run.json` + one CSV per sweep).

**Laser monitor.** Set `ref_pm_serial` once and the rest is automatic: at every QWP
step both meters are read at the same moment, over the same averaging window
(`pm_average_count`), and saved side by side (`power_W`, `ref_W`). The live Stokes
vectors are already laser-corrected, and after each sweep it prints the laser noise
and drift, the fit residual before → after correction, and how much the laser changed
between the input and through-sample sweeps. The first run prints the serial numbers
of the meters it finds; copy them into the JSON. `ref_pm_serial: null` runs without
a reference, exactly as before.

The beamsplitter belongs where it is, after the polarizer: the light it splits
always has the same polarization, so its split ratio does not change with the state
you prepare. Its effect on the transmitted polarization does not matter either —
every input state is measured after it.

**The states do not have to be exact** — each is measured, not assumed. What matters:

* the state must not change between its "input" and "through sample" sweeps;
* at least 4 states, including one with a real circular component
  (linear states alone give rank 3 and M cannot be reconstructed);
* spread out on the sphere. Condition number 1.73 is the best possible
  (H V D A R L); below ~4 is good; above 10 the errors blow up.
  Five or more states also give a per-state consistency check.

Key settings in `measure_mueller.json`:

| key | meaning |
|---|---|
| `qwp_zero_deg` | stage reading where the QWP fast axis is along the analyzer (93.6) |
| `qwp_retardance_waves` | QWP retardance at the wavelength, in waves as the manufacturer quotes it (0.24 = 86.4°) |
| `s3_sign` | flip to -1 if a known RCP reads as LCP |
| `qwp_sweep_deg` | sweep span, 360 (a full turn) or 180 — over a full turn the beam-walk term is fitted separately |
| `qwp_steps` | positions per sweep, spread over `qwp_sweep_deg` (100 over 360° = 3.6° steps) |
| `pm_average_count` | PM100D internal averages per reading (both meters) |
| `pm_serial` | signal meter, after the analyzer (`null` = the one that isn't the reference) |
| `power_range_w` | signal range: `null` = auto; a number fixes it (avoids range switching mid-sweep) |
| `ref_pm_serial` | reference meter after the beamsplitter; `null` = no laser monitor |
| `ref_power_range_w` | reference range, same rules |
| `zero_meter` | zero the meter(s) at the start — block the laser before the beamsplitter |

## Analyze

Put the run folder in `analyze_mueller.json` (`"run_dir": "C:/Users/.../mueller_..."`)
and run

```
python analyze_mueller.py
```

It uses the calibration recorded by the measurement unless you set
`qwp_retardance_waves`, `qwp_zero_deg` or `s3_sign` in the JSON (`null` = recorded),
so a better retardance value can be applied to old data without re-measuring.

Reported, each with a statistical (Monte Carlo) and a systematic error:

* **general** — transmittance M00, diattenuation (linear, circular, axis), polarizance
* **Lu-Chipman retarder** — total retardance, split into linear retardance, fast axis, optical rotation
* **depolarization** — depolarization index, depolarization power
* **differential (logm)** — LB, LB', CB, LD, LD', CD for a homogeneous medium.
  Its CD is free of the LD+LB mixing that fakes CD in M03 and in Lu-Chipman's D3.
* **Cloude** — entropy, and whether M is physical (smallest eigenvalue ≥ 0,
  compared with its own noise)

When the run has reference data every sweep is divided by it first
(`use_reference: false` to compare without it), and a laser-monitor summary is printed:
drift over the run, noise per reading, the largest change between a state's input
and through-sample sweeps, and the fit residual raw → corrected.

Plus a per-state misfit (catches a state that changed or a sample that moved) and a
short verdict. Systematic errors are the WORST CASE over the calibration
tolerances `retardance_uncertainty_waves` (default λ/300) and
`qwp_zero_uncertainty_deg` (default 0.1°): each error alone at ± its limit, and both
together at every sign combination, re-analysed; the largest shift is reported. Output in `<run_dir>/analysis/`: `results.json`,
`mueller.csv`, `mueller_matrix.png`, `poincare.png`, `fits.png`, `laser_monitor.png`.

## Characterize the polarimeter

```
python QWP_analyzer_characterization.py                 # measure, then analyse (--simulate to rehearse)
python QWP_analyzer_characterization.py <run_dir> ...   # analyse existing runs (Mueller runs too)
```

Measures the two numbers every Stokes vector depends on, `qwp_zero_deg` and
`qwp_retardance_waves`, from the setup itself. **No sample in the beam.** Every
state from laser → polarizer → wave plates is fully polarized (DOP = 1), so
whatever differs from that is the polarimeter. Each state is swept `repeats`
times (default 10) without touching anything, over 360° by default.

| state | what it calibrates | how |
|---|---|---|
| **H** (state QWP out, HWP for max signal) | QWP retardance | the depth of its sweep is cos²(δ/2); DOP moves 3.5 % per degree |
| **R** or **L** | QWP zero | a zero error ε gives the S3 term a cos 2θ part, c2/c1 = −tan 2ε, independent of retardance and power |
| D / A (optional) | cross-check | must give the same retardance as H — their disagreement is the real accuracy |
| V (optional) | analyzer leak | upper bound on 1 / extinction ratio |

Also reported: beam walk (the 360°-period term, and first vs second half turn),
the cos 2θ term that a zero error does not explain, whether the fit residual is noise
or systematic, and how much light from the polarimeter reaches the reference meter
(the wire grid reflects the rejected polarization back).

The states only need to be approximate: each is measured. What matters is not
touching anything during a state's repeats, H close to the analyzer axis (it is
the most sensitive state), and R/L reasonably circular.

**Error budget.** Both numbers come with a budget: the scatter of repeats, the
disagreement between states (the real accuracy test: zero for a polarimeter whose
only errors are these two numbers), a cos 2θ term the zero doesn't explain, the
zero's uncertainty propagated into the retardance, and two one-sided terms. The
input states' DOP is not exactly 1: it is at least `input_dop_min` (default 0.998,
≈ 1 − 2/ER for a 1000:1 polarizer, the LPVISC guarantee at 510–520 nm), and the
analyzer leak measured with V lowers every DOP too. Both would make the true
retardance slightly higher. The totals are what it suggests as tolerances.

Hardware settings and the calibration currently assumed are read from
`measure_mueller.json`; `QWP_analyzer_characterization.json` holds only this
script's settings. At the end it prints the values to paste into
`measure_mueller.json` and `analyze_mueller.json`, with tolerances. Output in
`<run>/characterization/`.

## Measure transmittance directly

```
python measure_transmission.py            # rehearse first with: --simulate
```

An independent check of M00. **Take the polarimeter's QWP and analyzer out of the
beam** (or put the sensor right after the sample): the signal meter must see all the
light, since a wave plate changes the polarization and anything polarizing after it
fakes a transmission change. The script asks for the sample out, in, out, in, …, out
(`cycles` times in): every "in" is bracketed by two "out" readings so slow drift
cancels, each reading is divided by the reference meter, and the dark offset (beam
blocked) is subtracted. It prints the transmittance per cycle and in total, with and
without the reference, and compares with a Mueller run's M00 if `compare_run` is set.
Meter serials and wavelength come from `measure_mueller.json`. Output in
`Desktop/transmission_<date>_<time>/`: `run.json`, `readings.csv`, `transmission.png`.

## The calibration trap

An error in the polarimeter's own calibration (QWP retardance, zero angle) makes it
report every Stokes vector through the same wrong matrix `A`. Then the measured
matrix is `A·M·A⁻¹`, not `M`. For air, `M = I`, so `A·I·A⁻¹ = I` **exactly** — a
perfect identity for air proves nothing about the calibration. Check it instead with
a sample you know that is not the identity (e.g. a waveplate of known retardance, or
a polarizer), and get the QWP retardance from the manufacturer's data.
