# Mueller matrix of a film on a substrate — translation stage

Rotating-QWP polarimeter (QWP on K-Cube 28000005 → fixed analyzer → PM100D + S130VC),
input states set by the motorized PSG (polarizer → HWP → QWP), a second PM100D as
laser monitor, and a translation stage (ZST225 on a KST101) that puts the empty
position, the bare substrate or the film on the substrate into ONE beam. Built on
`polarization_toolkit`. The two-aperture version with flippers is
[`../mueller_matrix_flippers`](../mueller_matrix_flippers).

```
laser → polarizer (V) → beamsplitter → HWP (DDR25) → QWP (PRM1Z8) → [stage: E / S / F] → QWP (rotating) → analyzer → signal meter
                             │         └──────── PSG ────────┘                          └──────── polarimeter ────────┘
                             └→ reference meter (laser monitor)
```

## Order of work

1. **PSG settings, once.** Run `poltk psg show`: it writes `~/.poltk/psg_settings.json`.
   Put the HWP and QWP motor serials in it, and set `psa_config` to the full path of
   this folder's `measure_mueller.json` (e.g.
   `C:/Users/naumeshe/general_lab/mueller_matrix_t_stage/measure_mueller.json`).
2. **Calibrate the polarimeter:** `python QWP_analyzer_characterization.py` (no sample;
   automatic: the stage parks at the empty position and the PSG sets `psg_states`,
   H R L D A V by default, `repeats` sweeps each).
   Paste the printed values into `measure_mueller.json`. If the PSG is already
   calibrated, it also prints the `poltk psg refit` command that updates the PSG with the
   new polarimeter values, without measuring again.
3. **Calibrate the PSG:** `poltk psg calibrate` (no sample, ~25 min). It saves
   `~/.poltk/psg_calibration.json`; every script reads it automatically, and the next
   calibration replaces it (the old one goes to `~/.poltk/psg_history/`).
4. **Check the PSG** (optional): `python psg_check.py`, then type states.
5. **Measure:** `python measure_mueller.py`. Fully automatic (see *Empty, substrate,
   film* below).
6. **Analyze:** set `run_dir` in `analyze_mueller.json`, then `python analyze_mueller.py`.

The first time, step 3 runs with the polarimeter values you already have (93.6°,
0.24 waves) and step 2 with an uncalibrated PSG: both work, because the polarimeter
calibration only needs fully polarized states, not exact ones. If step 2 changes the
polarimeter values a lot, run step 3 again (or `poltk psg refit` on its run folder).

| file | what |
|---|---|
| `measure_mueller.py` + `measure_mueller.json` | interactive measurement |
| `analyze_mueller.py` + `analyze_mueller.json` | M, decompositions, error bars, figures |
| `QWP_analyzer_characterization.py` + `.json` | calibrates the polarimeter itself: QWP zero and retardance, beam walk, analyzer leak |
| `psg_check.py` + `.json` | type a state: the PSG sets it, the polarimeter measures it; target / expected / measured and the ellipses |
| `measure_transmission.py` + `.json` | sample transmittance with one power meter, no polarimeter (checks M00) |
| `mueller_common.py` | shared helpers; the Stokes extraction itself lives in the toolkit (`polarization_toolkit.analysis.psa`) |

Both scripts read their JSON automatically. A flag on the command line overrides
the JSON for that run; `--config other.json` uses a different file. A misspelled
key in the JSON is rejected with the list of valid keys.

## Measure

```
python measure_mueller.py              # rehearse first with: --simulate
```

Before measuring, it prints the PSG settings of every state in `states` and how well
they will pin down M (the condition number). Then, for each state, the PSG sets it and
the stage runs through `sequence` (E S F F S E) with one sweep at each stop. Nothing is
done by hand. After each state it prints how far the input state (E) is from its target,
how far each kind's first and last sweeps are apart (drift), the coverage so far, and
the time left. Everything is saved after every state, in
`Desktop/mueller_<date>_<time>/` (`run.json` plus one CSV per sweep:
`state_01_E1.csv`, `state_01_S1.csv`, `state_01_F1.csv`, ...).

`use_psg: false` sets the states by hand instead: you type a label for each (the stage
still does the rest), and Enter with no label finishes.

**Laser monitor.** With `ref_pm_serial` set, both meters are read at every QWP step,
at the same moment and over the same averaging window; every reading is divided by the
reference, which removes laser power drift and noise.

**The states do not have to be exact** — each is measured (E). What matters: at least 4
states, including one with a real circular component, spread out on the sphere
(condition number 1.73 is the best possible).

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
| `use_psg` | the PSG sets the states (`false`: by hand) |
| `states` | the states, in order: names (`"H"` … `"L"`), `[S1, S2, S3]`, or `{"azimuth_deg": 30, "ellipticity_deg": 10}` |
| `use_stage` | the translation stage puts E / S / F in the beam |
| `stage_motor`, `stage_name` | KST101 serial (starts with 26); Kinesis actuator name, e.g. `"ZST225"` (`null` = the settings stored on the controller) |
| `empty_mm`, `substrate_mm`, `film_mm` | the three stage positions (4.5866, 12.7767, 21.4397) |
| `dark_mm` | optional: a position where the holder blocks the beam — automatic zeroing and a dark check at the end (`null`: you block the beam at the start) |
| `stage_approach_mm` | every move arrives from this far below: no backlash |
| `sequence` | stops per state; `["E","S","F","F","S","E"]` (below) |
| `live_view` | a live window during the run (below) |
| `dark_readings` | readings averaged for the dark check at the end |

## Empty, substrate, film (translation stage)

One beam; the stage moves the holder so that the beam goes through the empty position
(E), the bare substrate (S), or the film on the substrate (F). For each PSG state the
stage runs through `sequence`, **E S F F S E** by default, one sweep per stop:

- **Every kind has the same mean time.** A slow, linear drift of anything — laser,
  polarization, the PSG plates, the polarimeter — cancels to first order in every
  comparison: film vs substrate, substrate vs empty, film vs empty.
- **One PSG setting for all six sweeps**, so E, S and F see identical input states,
  minutes apart instead of an hour (the empty-aperture pass of the flipper version).
- **Two sweeps of each kind:** equal noise, fitted together as one double sweep; how far
  each kind's first and last sweeps are apart is printed per state as a drift monitor.
- **Every stage move arrives from below** (`stage_approach_mm`), so the lead screw's
  backlash cannot shift a position.

Shorter alternatives: `["S","F","F","S"]` (film only, no empty) or
`["E","S","F","S","E"]` (5 sweeps, still a palindrome). A sequence that is not a
palindrome gets a warning: its drift does not cancel.

`analyze_mueller.py` reports (`result` in its JSON; null = `film`):

| `result` | from | meaning |
|---|---|---|
| `film` | S_F · pinv(S_S) | the film alone. The same beam and the same states go through substrate and film, so the input states cancel exactly; M00 is the film's transmittance |
| `substrate` | S_S · pinv(S_E) | the substrate alone |
| `film_substrate` | S_F · pinv(S_E) | film and substrate together |
| `film_via_empty` | (S_F · pinv(S_E)) · (S_S · pinv(S_E))⁻¹ | cross-check: should agree with `film` |

`film` assumes the light crosses the **substrate first**, i.e. the film faces the
polarimeter (exact even for a birefringent substrate). It prints all four briefly and
fully analyzes the one chosen; `results.json` keeps all.

The film is measured on a different spot of the substrate than the bare substrate:
`film` assumes the substrate is the same at both positions (thickness, stress).

In simulation (14 states, E S F F S E, 1 % laser drift, 0.1 % noise) the film came out
within 0.0002 of the truth, the substrate within 0.0003 — against 0.0005 and 0.0021 with
the flippers and an empty pass an hour later.

Setup, once: the stage needs `thorlabs_motion` with KST101 support (`git pull` there).
Run `python kcube.py` in `thorlabs_motion` to see the KST101 and its serial. If the
actuator is not stored on the controller, set `stage_name` (e.g. `"ZST225"`). Check
once that the three positions land on the holder slots.

## Precise measurement

Polarimeter noise is already small (about 0.05° per sweep, against about 0.5° of
systematic effects), so averaging longer alone buys little. What helps:

- **The palindromic sequence** E S F F S E (above): drift cancels, all three kinds are
  measured minutes apart with one PSG setting, and each kind gets two sweeps.
- **More states:** the six plus the eight cube corners give the same ideal conditioning
  (1.73), about 1.5× lower noise in M, state-specific errors averaged out, and a
  per-state consistency check:
  ```json
  "states": ["H", "V", "D", "A", "R", "L",
             [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
             [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]],
  ```
- **Longer averaging and settling:** `"pm_average_count": 1000`, `"qwp_settle": 0.5`,
  `"stage_settle": 0.5` (all set in this folder's JSON).
- **A fixed meter range:** auto-ranging can switch mid-sweep and put a step in the data.
  After a first run, set `power_range_w` a little above the largest signal you saw.
- **Lights off, zeroing on** (`zero_meter`; automatic with `dark_mm`), and the laser
  warmed up.

With 14 states and 6 sweeps each, a run takes a few hours. After each state it prints
the time so far and an estimate of the time left.

## Live view

With `live_view: true` a window shows the run as it happens:

- **the sweep:** signal vs polarimeter QWP angle, point by point, with the Fourier fit
  growing as points come in, and the Stokes vector and DOP so far;
- **the polarization ellipse** of the state being measured, on top of its target;
- **the Poincaré sphere** filling up: empty (s), substrate (o), film (^), a dashed line
  from each state's substrate point to its film point;
- **a live film Mueller matrix** (film vs substrate) as soon as four states pin it
  down, with the condition number and M00;
- **status:** which position the stage holds, the PSG angles, the laser monitor,
  elapsed time and time left.

Closing the window does not stop the run. The final picture is saved as
`live_view.png` in the run folder; at the end the window stays open until you close it.

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
| **H** (the PSG sets it; by hand: state QWP out, HWP for max signal) | QWP retardance | the depth of its sweep is cos²(δ/2); DOP moves 3.5 % per degree |
| **R** or **L** | QWP zero | a zero error ε gives the S3 term a cos 2θ part, c2/c1 = −tan 2ε, independent of retardance and power |
| D / A (optional) | cross-check | must give the same retardance as H — their disagreement is the real accuracy |
| V or P (optional) | analyzer leak | upper bound on 1 / extinction ratio (V and R/L carry no retardance information: their DOP stays 1 whatever retardance is assumed) |

Also reported: beam walk (the 360°-period term, and first vs second half turn),
the cos 2θ term that a zero error does not explain, whether the fit residual is noise
or systematic, and how much light from the polarimeter reaches the reference meter
(the wire grid reflects the rejected polarization back).

With the PSG (`use_psg`, read from `measure_mueller.json`) it runs on its own: the PSG
sets every state in `psg_states` (H R L D A V by default) and each is swept `repeats`
times. With 100 steps per sweep that is about 1 min per sweep, so 6 states × 10 repeats
is about an hour; lower `repeats` for a quicker run. An empty `psg_states` lets you type
the states instead (`P` is both PSG plates along the polarizer). The PSG's own accuracy
does not matter here: any fully polarized state works.

**PSG miss vs polarimeter error.** If `poltk psg calibrate` reports a large miss, the
measured DOP line tells you where it comes from. A polarimeter error large enough to
make the PSG miss by several degrees (e.g. a retardance 0.03 waves off) also spreads
the measured DOP far from 1 (0.73–1.04 in simulation). A DOP close to 1 with a large
miss points at the PSG side instead. On the bench it was the input state: the polarizer
sat 20° from the analyzer's V and the beamsplitter added −1° of ellipticity, which a fit
fixed at V cannot absorb (4.5° rms); with `fit_polarizer: true` in
`~/.poltk/psg_settings.json` (the default for new files) the fit finds both, 0.6° rms.
Other suspects: the QWP stage setting, a plate wobbling in its mount, a stage not
reaching its position. The states only need to be
approximate: each is measured. What matters is not touching anything during a
state's repeats, H close to the analyzer axis (it is the most sensitive state), and
R/L reasonably circular.

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
python measure_transmission.py
```

An independent check of M00 with one power meter (the reference PM unless
`meter_serial` is set). Put the meter directly behind the sample, with no polarizing
optics in between. The script zeroes the meter (beam blocked), then asks for the sample
out, in, out, in, …, out (`cycles` times in); each "in" is compared with the mean of
the "out" readings just before and after it, so slow laser drift cancels. It prints
the transmittance per cycle and the mean ± its error, and saves `readings.csv` and
`run.json` in `Desktop/transmission_<date>_<time>/`.

## Check the PSG

```
python psg_check.py                  # type states one after the other (Enter quits)
python psg_check.py R H "30 -10"     # or give them; also "states" in psg_check.json
```

A state is a name (`H V D A R L P`), two numbers (azimuth and ellipticity of the
ellipse, deg) or three (S1 S2 S3). For each one it prints the target, the expected
state (the PSG model at the angles the stages reached) and the measured state, with
DOP, azimuth and ellipticity, and the misses on the Poincaré sphere: measured vs target
is the whole error, measured vs expected the part the PSG calibration does not explain.
The figure overlays the three ellipses. No sample in the beam; everything is saved in
`Desktop/psg_check_<date>_<time>/`.

## The calibration trap

An error in the polarimeter's own calibration (QWP retardance, zero angle) makes it
report every Stokes vector through the same wrong matrix `A`. Then the measured
matrix is `A·M·A⁻¹`, not `M`. For air, `M = I`, so `A·I·A⁻¹ = I` **exactly** — a
perfect identity for air proves nothing about the calibration. Check it instead with
a sample you know that is not the identity (e.g. a waveplate of known retardance, or
a polarizer), and get the QWP retardance from the manufacturer's data.
