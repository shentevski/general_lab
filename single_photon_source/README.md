# single_photon_source

Characterization scripts for the single-photon source: the 266 nm Q-switched
pump and, later, the BBO pair source behind it.

Run them on the PC the hardware is plugged into.

| script | what it does |
| --- | --- |
| `power_vs_rep_rate.py` | mean 266 nm power and mean energy per pulse against the laser's pulse repetition frequency |

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
