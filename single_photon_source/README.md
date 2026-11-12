# single_photon_source

Everything for the single-photon source: the 266 nm Q-switched pump, the
Arduino that triggers it, and the photon-pair measurements behind it.

Run these on the Windows PC the hardware is plugged into — the Time Tagger
and Arduino scripts use `msvcrt` for the keypresses and a `COM` port for the
serial link, both set at the top of each script.

| script | what it does |
| --- | --- |
| `power_vs_rep_rate.py` | mean 266 nm power and mean energy per pulse against the laser's pulse repetition frequency |
| `measure_delay.py` | start–stop delay between the laser sync and the detector, fired shot by shot |
| `laser_trigger/laser_trigger.ino` | the Arduino sketch: one 5 V, 1 ms pulse on D8 per character received |
| `laser_trigger/upload.py` | compiles and uploads the sketch with `arduino-cli`, detecting board and port |
| `debug/fire_trigger.py` | fires the trigger by hand, no Time Tagger — checks the Arduino → laser link alone |
| `debug/tag_pulses.py` | sends the Arduino pulse into the tagger and prints its width and spacing |
| `debug/single_photons_v1.py` | records all tagger channels while you fire by hand; saves CSV + raw `.ttbin` |

Only one program can hold the COM port, so close the others (and the Arduino
Serial Monitor) before starting one.

## power_vs_rep_rate.py

```
python power_vs_rep_rate.py
```

Connects the Thorlabs meter, sets it to 266 nm, then loops: set the laser's
repetition frequency, type it in (`5000`, `5k` and `5 kHz` all work), and the
script averages 20 readings and keeps the point. Enter on an empty line (or
`q`) finishes.

Per point it records the mean power P and the mean energy per pulse
E = P / f, f being the repetition frequency. The meter averages over many
pulses, so P is the average power and E is the only per-pulse quantity it can
give you.

Saved in `~/Desktop/single_photon_source/` as
`power_vs_rep_rate_<stamp>.csv` and `.png`. The CSV is rewritten after every
point, so a Ctrl-C keeps what you already measured.

Needs the `thorlabs_powermeter` package (`SCRIPTS/HARDWARE/thorlabs_powermeter`),
the same one the polarimetry scripts use.

## measure_delay.py

```
python measure_delay.py
```

Fires the laser through the Arduino one shot at a time (`s` fires, `q` stops
and saves) and histograms the detector arrival time against the laser sync on
the Time Tagger. Prints the peak delay per shot and, at the end, the peak and
count-weighted mean delay; the histogram goes to `delays_ps.txt`.

The peak is the delay you want. The weighted mean only means something once
the peak stands well clear of the dark-count background, since a dark count
landing before the real photon is recorded as a pair too.

Needs `TimeTagger` (Swabian) and `pyserial`.
