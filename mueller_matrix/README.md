# Mueller matrix of a sample

Rotating-QWP polarimeter (QWP on K-Cube 28000005 → fixed analyzer → PM100D + S130VC),
input states set by the motorized PSG (polarizer → HWP → QWP), a second PM100D as
laser monitor. Built on `polarization_toolkit`.

```
laser → polarizer (V) → beamsplitter → HWP (DDR25) → QWP (PRM1Z8) → [sample] → QWP (rotating) → analyzer → signal meter
                             │         └──────── PSG ────────┘                 └──────── polarimeter ────────┘
                             └→ reference meter (laser monitor)
```

## Order of work

1. **PSG settings, once.** Run `poltk psg show`: it writes `~/.poltk/psg_settings.json`.
   Put the HWP and QWP motor serials in it, and set `psa_config` to the full path of
   this folder's `measure_mueller.json`.
2. **Calibrate the polarimeter:** `python QWP_analyzer_characterization.py` (no sample;
   automatic: the PSG sets `psg_states`, H R L D A V by default, `repeats` sweeps each).
   Paste the printed values into `measure_mueller.json`. If the PSG is already
   calibrated, it also prints the `poltk psg refit` command that updates the PSG with the
   new polarimeter values, without measuring again.
3. **Calibrate the PSG:** `poltk psg calibrate` (no sample, ~25 min). It saves
   `~/.poltk/psg_calibration.json`; every script reads it automatically, and the next
   calibration replaces it (the old one goes to `~/.poltk/psg_history/`).
4. **Check the PSG** (optional): `python psg_check.py`, then type states.
5. **Measure:** `python measure_mueller.py`. With the PSG and the flippers it is fully
   automatic (see *Film and substrate* below).
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
they will pin down M (the condition number). Then, for each state:

1. the PSG sets the state
2. **remove the sample** → Enter → the input state is measured
3. **put the sample back** → Enter → the state is measured through the sample

After each state it prints how far the measured input is from the target (a check on
the PSG calibration) and the rank and condition number of what you have so far.
Everything is saved after every state, in `Desktop/mueller_<date>_<time>/` (`run.json`
with the PSG angles of every state, plus one CSV per sweep).

`use_psg: false` goes back to setting the states by hand: you type a label for each,
and Enter with no label finishes.

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
| `use_psg` | the PSG sets the states (`false`: by hand) |
| `states` | the states, in order: names (`"H"` … `"L"`), `[S1, S2, S3]`, or `{"azimuth_deg": 30, "ellipticity_deg": 10}` |
| `use_flippers` | the two flippers choose the film / substrate half: fully automatic (`false`: you move the sample) |
| `film_flipper`, `substrate_flipper` | serials of the flippers that block the film half and the substrate half (MFF, start with 37) |
| `flipper_film_block_position`, `flipper_subst_block_position` | the position, 1 or 2, that puts each flipper's block INTO the beam (null: `flipper_block_position` for both) |
| `live_view` | a live window during the run (below) |
| `empty_at_end` | at the end, remove film + substrate once; every state is measured again through each empty aperture |
| `dark_readings` | readings averaged for the dark check at the end |

## Film and substrate (flippers)

The sample is half film, half bare substrate, in one beam; two motorized flippers
(MFF10x, driven through `thorlabs_motion`'s `flipper.py`) each block one half. A run
then needs nothing from you:

1. **Start:** both halves blocked, so the signal meter is zeroed. That also removes
   room light and the meter's dark offset (room light otherwise lowers every DOP and
   looks like depolarization).
2. **Each state:** the PSG sets it. Film half blocked: the *reference* sweep, through
   the substrate. Substrate half blocked: the *sample* sweep, through film + substrate.
3. **Empty apertures** (`empty_at_end`, on by default): it asks you once to remove the
   film and the substrate, leaving the two apertures; then every state is measured again
   through each empty aperture. This is the only thing you do by hand.
4. **End:** both blocked again; the dark reading should be about 0. If it isn't, the
   room light or the zero changed during the run, and it says by how much. Then both
   flippers open, so the beam is clear for the other scripts.

With the empty apertures, `analyze_mueller.py` reports (`result` in its JSON; null =
`film`):

| `result` | from | meaning |
|---|---|---|
| `film` | (film half ÷ its empty aperture) · (substrate half ÷ its empty aperture)⁻¹ | the film alone; each half referenced to its own aperture, so differences between the halves cancel, and M00 is the film's transmittance |
| `substrate` | substrate half ÷ its empty aperture | the substrate alone |
| `film_substrate` | film half ÷ its empty aperture | film and substrate together |
| `relative` | film half ÷ substrate half | the film relative to the substrate (all there is without the empty pass) |
| `null` | empty film aperture ÷ empty substrate aperture | should be the identity; M00 = how the beam is split between the apertures |

It prints all of them briefly and fully analyzes the one chosen; `results.json` keeps all.

Without the empty pass, the result is the film relative to the substrate,
M = M_(film+substrate) · M_substrate⁻¹:

- **Exact when the light crosses the substrate before the film**, i.e. the film faces
  the polarimeter, even for a birefringent substrate. The other way round it holds for
  an isotropic substrate (glass) and is otherwise the film seen through the substrate.
- **M00** is the film half's power over the substrate half's: the film's transmittance
  times however unevenly the beam is split between the halves. The normalized M does
  not depend on that.
- **The two halves are different parts of the beam** and reach the polarimeter at
  slightly different places. The empty pass measures that (`null`) and the `film`
  result removes it.

Setup, once: `python flipper.py` in `thorlabs_motion` lists the flippers and their
serials. Check which position blocks (`flipper_block_position`). If `thorlabs_motion`
was installed before the flipper support, reinstall it (`pip install -e .` in that
folder) so `flipper` can be imported.

## Live view

With `live_view: true` a window shows the run as it happens:

- **the sweep:** signal vs polarimeter QWP angle, point by point, with the Fourier fit
  growing as points come in, and the Stokes vector and DOP so far;
- **the polarization ellipse** of the state being measured, on top of its target;
- **the Poincaré sphere** filling up: reference (o), through the sample (^), empty
  apertures (s, D), a dashed line from each state's reference to its sample point;
- **a live Mueller matrix** (sample relative to reference) as soon as four states pin it
  down, with the condition number and M00;
- **status:** which half the flippers let through, the PSG angles, the laser monitor,
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
