# HB phase 0: wavelength-tuned LED source

A Thorlabs LED on a **DC40** driver shines through an element on a **PRMTZ8** rotation
stage that passes one narrow band; turning the stage tunes the band's wavelength. The
output goes to a **CCT10** spectrometer (calibration, `set_wavelength.py`) or a **power
meter** placed in the output beam (`constant_power_scan.py`).

```
DC40 + LED → element on the PRMTZ8 (one band, tuned by the angle) → CCT10 spectrometer
                                                                    or power meter
```

| file | what |
|---|---|
| `calibration.py` + `calibration.json` | every angle × every LED current → a spectrum: the wavelength map (angle → wavelength) and the amplitude map (current → peak amplitude at each angle) |
| `set_wavelength.py` + `set_wavelength.json` | asks for a wavelength and a peak amplitude, sets the stage angle and the LED current from the calibration, checks with a spectrum |
| `constant_power_scan.py` + `constant_power_scan.json` | steps through the wavelengths, holding the power constant with the power meter |
| `hb_common.py` | hardware, simulator, peak finder, calibration file reader |

Hardware libraries (in `SCRIPTS/HARDWARE`, Windows only, `pip install -e .` in each):
[`thorlabs_motion`](https://github.com/shentevski/thorlabs_motional) (`kcube`, for the
KDC101 driving the PRMTZ8), [`thorlabs_led`](https://github.com/shentevski/thorlabs_led)
(the DC40), [`thorlabs_spectrometer`](https://github.com/shentevski/thorlabs_spectrometer)
and [`thorlabs_powermeter`](https://github.com/shentevski/thorlabs_powermeter). Close the
Thorlabs GUIs (Kinesis, DC40, ThorSpectra, Optical Power Monitor) first: each holds its
device open. `--simulate` runs anywhere, without hardware, and rehearses every prompt,
file and figure.

**Words used here.**

| term | meaning |
|---|---|
| peak amplitude | the height of the band's peak in spectrometer counts above the background, at the calibration's exposure |
| band light | the counts summed over the band (counts × nm); proportional to the band's power at one wavelength |
| background | a spectrum without the LED's light, subtracted from every spectrum |
| FWHM | full width at half maximum of the band |
| EEPROM | the memory chip in a Thorlabs mounted LED that holds its name and current limit; the DC40 reads it |
| angle | the PRMTZ8 position in degrees, as the stage reports it |

## Order of work

1. **Check `calibration.json`**: the angle range, the LED currents, and the shared hardware
   block (serial numbers can stay `null` when only one of each device is connected).
2. **Rehearse:** `python calibration.py --simulate`, then `python set_wavelength.py
   --simulate` and `python constant_power_scan.py --simulate`.
3. **Calibrate** with the spectrometer in the output beam: `python calibration.py`. Look
   at `calibration.png`: the residuals should be small and without structure.
4. **Use it:** `python set_wavelength.py` (spectrometer in the beam), or put the power meter
   in the output beam and run `python constant_power_scan.py`. Both take the newest
   calibration automatically.

Recalibrate after anything in the light path moves, or after the LED is changed.

## Calibration

1. The LED goes on at the highest current and warms up (`led_warmup_s`): its light drops
   for a while as it heats.
2. **Prescan:** the stage visits every angle once at the highest current to find the
   brightest spectrum of the calibration.
3. At that angle you type an exposure; Enter takes the one offered, which puts the
   brightest pixel at `target_fill` (70 %) of full scale. **One exposure is used for the
   whole calibration**, so every amplitude is in the same counts; `set_wavelength.py`
   measures at the same exposure.
4. A background is taken before and after the scan (the LED off, or the spectrometer
   shutter closed: `background`) and interpolated in time.
5. **Scan:** at every angle the LED steps through the currents, with `led_settle_s` after
   each change, and a spectrum is recorded at each current. Ctrl-C stops it; the angles
   already finished are analysed and saved.
6. **Drift check:** the first angle is measured again at the end. The report gives how much
   its peak and centre changed over the calibration.
7. For every spectrum it finds the peak in `wl_min_nm`–`wl_max_nm`: its centre
   (`peak_method`), FWHM and height. A spectrum is not used for the wavelength map if it is
   dimmer than `min_peak_counts`, clipped, or touches the edge of the window.
8. **Wavelength map:** a polynomial wavelength(angle) of `fit_order` through the centres at
   `reference_current_ma` (default: the highest current). Angles more than
   `max_residual_nm` off the fit are dropped one at a time and the fit is redone. The map
   must be monotonic: an angle range that crosses normal incidence is refused.

Output in `<data_dir>/calibrations/<date>_<time>/`:

| file | contents |
|---|---|
| `hb_calibration.csv` | one row per angle and current: `centre_nm`, `fwhm_nm`, `peak_counts`, `band_counts`, `time_s`, `status` |
| `wavelength_map.csv` | one row per angle: `angle_read_deg` (stage readback), `wavelength_nm` (the fit), `measured_nm`, `residual_nm`, `fwhm_nm`, `peak_counts`, `centre_shift_nm` (how far the centre moves between currents), `used`, `status` |
| `spectra.npz` | every raw spectrum (angles × currents × pixels), the two backgrounds, wavelengths, times |
| `calibration_run.json` | settings, hardware, exposure, fit coefficients, summary, drift check |
| `calibration.png` | spectra, wavelength map, residuals, peak amplitude against wavelength at each current |

## Setting a wavelength

```
python set_wavelength.py                                              # asks, again and again
python set_wavelength.py --wavelength-nm 650 --amplitude-counts 20000 # once, then exits
```

1. You type a wavelength; the stage turns to the angle the wavelength map gives for it.
2. You type a peak amplitude (the prompt shows the most the calibration reached at that
   wavelength). The LED current is set to what the amplitude map gives for it at this
   angle, interpolated between the two nearest calibrated angles. Amplitudes beyond the
   calibration are refused (no extrapolation); 0 switches the LED off.
3. A spectrum at the calibration's exposure checks the result: measured centre, peak and
   FWHM. With `refine_iterations` > 0, the angle is corrected by the wavelength error
   (using the local nm per degree from the map) and the current by the amplitude error,
   until both are within `wl_tolerance_nm` and `amplitude_tolerance_rel`. A reading far
   from what the calibration expects is reported and not used for corrections.
4. When you quit (Enter or `q` at the wavelength prompt), it asks whether to leave the LED
   on at the last setting.

Each setting is logged in `<data_dir>/set_wavelength/<date>_<time>/log.csv` (asked and
measured values, angle and current from the calibration and after refinement), with its
spectrum in `spectra.npz`.

## Constant-power scan

The power meter goes in the output beam; the spectrometer is not used.

1. The power meter is zeroed with the LED off (`zero_pm`), then the LED warms up at the
   highest current.
2. **Probe:** every wavelength once at the highest current — the most power each can give.
   The weakest one limits a constant power; `probe_margin` (90 %) of it is offered as the
   target. You type the target in µW; Enter takes the one offered.
3. **Scan:** at every wavelength the stage turns to the calibrated angle, the power meter
   is told the wavelength (its sensor-response correction), and the LED current is
   adjusted until the power is within `power_tol_rel` (1 %) of the target. The calibration
   gives the shape of band light against current at each angle; only its scale in µW is
   unknown, and the probe gives it, so one reading is usually enough.
4. A wavelength where even the highest current falls short is marked `unreachable`. The
   highest current is the calibration's highest (or the LED limit, if lower).
5. At the end the LED goes off and the power meter is read once more: what is left is the
   zero's drift plus any change in room light.

Output in `<data_dir>/constant_power/<date>_<time>/`: `constant_power_scan.csv`
(wavelength, angle, current, measured LED current, power, its standard deviation, error,
readings needed, status; rewritten after every wavelength, so a stopped scan keeps what it
measured), `probe.csv`, `scan_run.json`, `constant_power_scan.png`.

## Settings

`calibration.json`. The first block is **shared**: the other two scripts read it from
this file, so the hardware is described in one place (putting these keys in their own
JSON is refused).

| key | meaning |
|---|---|
| `stage_serial` | KDC101 serial (27...); `null` = the only KDC101 connected |
| `stage_name` | Kinesis stage name, `PRMTZ8` |
| `home` | `auto`: home only if the controller says it isn't homed; `always`; `never` |
| `stage_settle_s` | wait after every stage move |
| `led_serial` | DC40 serial; `null` = the first one found |
| `led_max_ma` | never drive the LED above this; `null` = the LED's own limit (EEPROM) |
| `led_settle_s` | wait after every LED current change |
| `spectrometer_id` | `null` = the first one found |
| `hw_average` | frames the spectrometer averages per spectrum |
| `full_scale_counts` | raw counts at saturation |
| `background` | `led_off`: spectrum with the LED switched off (removes dark and room light); `shutter`: spectrometer shutter closed (dark only) |
| `background_frames` | spectra averaged per background |
| `data_dir` | `null` = `Desktop/HB_data` (simulated runs: `…/simulated`) |
| `simulate` | `true` = always rehearse |

The calibration's own keys:

| key | meaning |
|---|---|
| `angle_start_deg`, `angle_stop_deg`, `angle_step_deg` | the angles, in degrees on the stage scale |
| `angles_deg` | an explicit list instead, e.g. `[0, 5, 10, 20]` |
| `current_start_ma`, `current_stop_ma`, `current_count` | `current_count` currents evenly spaced; `current_stop_ma: null` = the LED limit |
| `currents_ma` | an explicit list instead, e.g. `[50, 100, 200, 400]` |
| `reference_current_ma` | the current the wavelength map is made at; `null` = the highest |
| `led_warmup_s` | LED on at the highest current this long before the prescan |
| `prescan` | `false` = no prescan; the exposure is set at the first angle |
| `target_fill` | the exposure offered fills the brightest pixel to this fraction of full scale |
| `exposure_ms` | offered at the exposure prompt instead (Enter takes it) |
| `drift_check` | measure the first angle again at the end |
| `wl_min_nm`, `wl_max_nm` | where peaks are looked for |
| `min_peak_counts` | dimmer spectra are not used for the wavelength map |
| `peak_method` | `centroid` (weights: counts − threshold), `gaussian` (parabola through ln counts), `max` (brightest pixel, refined) |
| `peak_threshold_rel` | pixels above this fraction of the peak give its centre |
| `band_halfwidth_nm` | band light is summed over centre ± this; `null` = 1.5 × the median FWHM |
| `fit_order` | polynomial order of the wavelength map; `null` = straight lines between the measured centres |
| `max_residual_nm` | angles further than this from the fit are dropped |
| `show_plots`, `notes` | show the figure; free text saved with the run |

`set_wavelength.json`:

| key | meaning |
|---|---|
| `calibration_file` | `null` = the newest in `<data_dir>/calibrations/`; or a calibration folder (or a file in it) |
| `wavelength_nm`, `amplitude_counts` | set these once and exit (usually given on the command line) |
| `verify` | check every setting with a spectrum |
| `refine_iterations` | corrections after the check; 0 = only check |
| `wl_tolerance_nm`, `amplitude_tolerance_rel` | when the setting counts as reached |
| `background_refresh_s` | retake the background before a setting once it is this old |
| `leave_led_on` | the answer offered when you quit (used directly in a one-shot run) |

`constant_power_scan.json`:

| key | meaning |
|---|---|
| `calibration_file` | as above |
| `pm_serial` | power meter serial; `null` = the first found |
| `pm_average_count` | samples the meter averages per reading (about 1 ms each) |
| `pm_readings` | readings averaged per power value |
| `zero_pm` | zero the meter with the LED off before the scan |
| `wl_start_nm`, `wl_stop_nm`, `wl_step_nm` | the wavelengths; start/stop `null` = the calibrated range |
| `wavelengths_nm` | an explicit list instead |
| `target_power_uw` | offered at the target prompt (µW); `null` = offer the probe's suggestion |
| `probe`, `probe_margin` | the highest-current pass and the fraction of the weakest it suggests |
| `power_tol_rel` | how close to the target counts as held |
| `max_iterations` | power readings allowed per wavelength |
| `led_warmup_s` | LED on at the highest current this long before the probe |
| `dwell_s` | stay this long at each wavelength once it is held (the power is read again at the end) |
| `show_plots`, `notes` | as above |

Any key can be overridden for one run on the command line, e.g. `--angle-step-deg 1` or
`--currents-ma 100,300,500`. A misspelled key in a JSON is rejected with the list of
valid keys.

## Rehearsal

`--simulate` uses a white LED (blue pump + phosphor, 1000 mA limit) whose light grows less
than linearly with current and overshoots for a moment after every current step, an
angle-tuned band-pass filter (12–15 nm wide, 700 nm at 0° → 603 nm at 60°), room-light
lines, a noisy 2048-pixel spectrometer that clips, and a silicon power meter. Time runs on
a virtual clock, so a rehearsal takes seconds. With the default settings:

- the calibration (31 angles × 10 currents; about 6 minutes of scanning by the simulator's
  clock) matches the true band centres to 0.007 nm with `fit_order` 5 (a cubic leaves
  0.3 nm of systematic error over 0–60°);
- `set_wavelength.py` sets 650 nm / 20000 counts and measures 650.00 nm / 20079 counts.
  With the LED made 15 % dimmer and the stage 0.5° off, one correction brings it back to
  650.00 nm and −0.2 %;
- `constant_power_scan.py` holds 65 µW at 19 wavelengths (605–695 nm) with 0.15 % spread,
  one reading each. With the LED 20 % dimmer than at calibration (and drooping more at
  high current) and no probe, every wavelength is still held within 1 %, with 1.4
  readings on average.

On the real setup, nothing here corrects for backlash in the stage; the wavelength check in
`set_wavelength.py` shows whether that matters.
