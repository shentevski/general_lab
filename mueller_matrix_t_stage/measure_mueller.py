#!/usr/bin/env python3
"""Measure the Mueller matrix of a film on a substrate -- one beam, a translation stage.

    python measure_mueller.py                  # reads measure_mueller.json
    python measure_mueller.py --simulate       # rehearse, no hardware

Optical layout

    laser -> polarizer (V) -> beamsplitter -> HWP -> QWP -> [stage] -> QWP (rotating) -> analyzer -> power meter
                                   |          '-- PSG --'      |       '------------ polarimeter ------------'
                                   '-> reference power meter   '-- ZST225 / KST101 moves into the beam:
                                       (laser monitor)              E  empty      (empty_mm)
                                                                    S  substrate  (substrate_mm)
                                                                    F  film on the substrate (film_mm)

Nothing is done by hand. For each input state (the PSG sets it, from `states`):

    the stage runs through `sequence` -- E S F F S E by default -- with one
    QWP sweep at each stop, all with the SAME PSG setting.

Why E S F F S E:
  * every kind (E, S, F) has the same mean time, so a slow, linear drift of
    anything -- laser, polarization, plates, polarimeter -- cancels to first
    order in every comparison (film vs substrate, substrate vs empty, ...);
  * one PSG setting for all six sweeps: the three see identical input states,
    minutes apart instead of hours;
  * two sweeps of each kind: equal noise, and their difference is a drift
    monitor printed for every state;
  * every stage move arrives from below (stage_approach_mm): no backlash.

The analysis gets the film directly, M_film = S_F . pinv(S_S): the same beam
and the same states go through the substrate and through film + substrate,
so the input states cancel exactly (light crosses the substrate first, i.e.
the film faces the polarimeter). The empty position gives the substrate and
film + substrate on their own, and an independent cross-check of the film.

Zeroing: with dark_mm (a stage position where the holder blocks the beam) the
signal meter is zeroed there automatically and a dark check ends the run;
otherwise you block the beam once at the start. Either way the stage parks
at the empty position at the end, so the beam is clear for other scripts.

Laser monitor: set ref_pm_serial and the second power meter is read at
every QWP step; dividing by it removes laser power drift and noise.

The states do NOT need to be exact: each is measured (E). What matters is
at least 4 of them, with a circular one, well spread on the Poincare sphere.
With use_psg false you set each state by hand and type a label; the stage
still does the rest. Every state is saved the moment it is measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from mueller_common import (corrected_power, extract_stokes, has_reference,
                            parse_with_config, waves_to_deg, write_sweep)
from polarization_toolkit.errors import ToolkitError
from polarization_toolkit.hardware.pm_polarimeter import (  # noqa: F401  (_setup_meter:
    PowerMeterPolarimeter, PSAConfig, _setup_meter, pm_sweep)  # measure_transmission)
from polarization_toolkit.psg import (PSG, SIX_STATES, SimulatedPSG, coverage as psg_coverage,
                                      sphere_angle_deg, state_label)

HERE = Path(__file__).resolve().parent

PLACEHOLDER = "26XXXXXX"
KINDS = {"E": "empty", "S": "substrate", "F": "film"}
SIDE = {"E": "in0", "S": "in", "F": "out"}        # names the analysis uses


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def view_of(x):
    """A stage view: E / S / F / dark (or empty / substrate / film); True /
    False (sample in / out), as the other scripts pass it."""
    if isinstance(x, bool):
        return "F" if x else "E"
    key = {"empty": "E", "substrate": "S", "film": "F"}.get(str(x).lower(), str(x))
    if key not in ("E", "S", "F", "dark"):
        raise ValueError(f"unknown stage view {x!r}: E, S, F or dark")
    return key


def stage_configured(a) -> bool:
    return bool(getattr(a, "use_stage", False))


def check_sequence(seq):
    seq = [str(s).upper() for s in seq]
    bad = [s for s in seq if s not in KINDS]
    if bad or "S" not in seq or "F" not in seq:
        raise SystemExit(f"sequence {seq}: use only E, S, F, with at least one S and one F")
    if seq != seq[::-1]:
        print(f"NOTE: sequence {' '.join(seq)} is not a palindrome -- slow drift will not "
              f"cancel between its kinds (E S F F S E does that)")
    return seq


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


class Stage:
    """The ZST225 actuator on a KST101 K-Cube, moving the sample holder.
    Every move arrives from below (stage_approach_mm), so the lead screw's
    backlash cannot shift a position."""

    def __init__(self, a):
        self.serial = str(a.stage_motor)
        self.name = a.stage_name
        self.pos = {"E": float(a.empty_mm), "S": float(a.substrate_mm), "F": float(a.film_mm)}
        if a.dark_mm is not None:
            self.pos["dark"] = float(a.dark_mm)
        self.approach = float(a.stage_approach_mm)
        self.settle = float(a.stage_settle)
        self.home = bool(a.stage_home)
        self.m = None
        self.view = None

    def __enter__(self):
        if self.serial in ("", "None", PLACEHOLDER):
            raise ToolkitError("no stage serial -- set stage_motor in measure_mueller.json "
                               "(KST101 serials start with 26), or use_stage: false")
        import kcube
        name = None if self.name is None or str(self.name).lower() in ("", "none") else self.name
        try:
            self.m = kcube.KCube(self.serial, stage_name=name).connect()
            if self.home:
                self.m.home()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        print(f"stage           : {self.serial} ({self.m.model}), E {self.pos['E']} / "
              f"S {self.pos['S']} / F {self.pos['F']} mm")
        self.select("E")
        return self

    def __exit__(self, *exc):
        try:
            if self.m is not None:
                self.select("E")                 # park: the beam is clear for other scripts
        except Exception:
            pass
        try:
            self.m and self.m.close()
        except Exception:
            pass
        self.m = None
        return False

    def select(self, view):
        view = view_of(view)
        if view not in self.pos:
            raise ValueError("no dark position -- set dark_mm")
        target = self.pos[view]
        if self.approach > 0:
            self.m.move_to(target - self.approach)
        self.m.move_to(target)
        time.sleep(self.settle)
        self.view = view

    def position(self):
        return float(self.m.get_position())

    def describe(self):
        return {"stage_motor": self.serial, "stage_name": self.name, "positions_mm": self.pos,
                "approach_mm": self.approach}


class Rig(PowerMeterPolarimeter):
    """The polarimeter (QWP stage + signal / reference PM100D, from the
    toolkit), with use_psg the PSG, which sets the input states, and with
    use_stage the translation stage, which puts E / S / F in the beam."""

    def __init__(self, a):
        super().__init__(PSAConfig.from_mapping(a))
        self.psg = PSG.from_files() if getattr(a, "use_psg", False) else None
        self.stage = Stage(a) if stage_configured(a) else None

    def __enter__(self):
        super().__enter__()
        try:
            for part in (self.psg, self.stage):
                if part is not None:
                    part.__enter__()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for part in (self.stage, self.psg):
            if part is not None:
                try:
                    part.__exit__(*exc)
                except Exception:
                    pass
        return super().__exit__(*exc)

    @property
    def has_dark(self):
        return self.stage is not None and "dark" in self.stage.pos

    def set_state(self, spec):
        return self.psg.set_state(spec) if self.psg is not None else None

    def set_sample(self, view):
        if self.stage is not None:
            self.stage.select(view)

    def zero_signal(self):
        """Zero the signal meter only (the reference sees the laser before the
        stage, so the stage cannot darken it)."""
        self.pm.zero()

    def describe(self):
        return {**super().describe(), "psg": self.psg.describe() if self.psg else None,
                "stage": self.stage.describe() if self.stage else None}


class SimRig:
    """Rehearsal stand-in with a known film and substrate, so a run can be
    checked. The states come from a simulated PSG (polarization_toolkit)
    whose true optics differ from its calibration in ~/.poltk/simulated; the
    substrate is weakly birefringent; the laser drifts and fluctuates (seen
    by both meters); room light reaches the signal meter until it is zeroed.
    Time runs on a virtual clock, so drift is realistic even when the
    rehearsal itself runs in seconds."""

    simulated = True
    READ_S = 0.3
    PSG_S = 5.0
    STAGE_S = 4.0
    STATE_S = 60.0
    DRIFT_PERIOD_S = 900.0

    NOMINAL = {"H": (1, 1, 0, 0), "V": (1, -1, 0, 0), "D": (1, 0, 1, 0),
               "A": (1, 0, -1, 0), "R": (1, 0, 0, 1), "RCP": (1, 0, 0, 1),
               "L": (1, 0, 0, -1), "LCP": (1, 0, 0, -1)}

    def __init__(self, a):
        from polarization_toolkit.hardware.simulated import (
            retarder_mueller, rotator_mueller, analyzer_row)
        self._ret, self._rot, self._row = retarder_mueller, rotator_mueller, analyzer_row
        self.a = a
        self.rng = np.random.default_rng(1)
        d = np.deg2rad
        # the film: a partial diattenuator (D = 0.3 at 15 deg), a 60 deg retarder
        # at 20 deg and a weak depolarizer, 80 % transmission
        D, ax = 0.3, np.array([np.cos(d(30)), np.sin(d(30)), 0.0])
        MD = np.eye(4); MD[0, 1:] = MD[1:, 0] = D * ax
        sq = np.sqrt(1 - D * D)
        MD[1:, 1:] = sq * np.eye(3) + (1 - sq) * np.outer(ax, ax)
        self.film = 0.8 * np.diag([1, .95, .95, .95]) @ retarder_mueller(d(60), d(20)) @ MD
        self.substrate = 0.92 * retarder_mueller(d(3.0), d(10))
        self.state = np.array([1., 1, 0, 0])
        self.view = "E"
        self.pos = 0.0
        self.P0 = 1e-3
        self.clock = 0.0
        self.has_ref = bool(a.ref_pm_serial)
        self.has_dark = getattr(a, "dark_mm", None) is not None
        self.parallel = True
        self.ambient = getattr(a, "sim_ambient_rel", 0.0) * self.P0
        self.zero = 0.0
        self.psg = (SimulatedPSG.from_files(simulate=True)
                    if getattr(a, "use_psg", False) else None)

    def __enter__(self):
        if self.psg is not None:
            self.psg.__enter__()
        print("signal meter    : simulated")
        print("reference meter : " + ("simulated (laser monitor)" if self.has_ref else
                                      "none (ref_pm_serial is null) -- no laser correction"))
        print("stage           : simulated")
        return self

    def __exit__(self, *exc):
        return False

    def set_state(self, spec):
        if self.psg is not None:
            self.clock += self.PSG_S
            setting = self.psg.set_state(spec)
            self.state = self.psg.stokes()
            return setting
        self.clock += self.STATE_S
        nom = self.NOMINAL.get(str(spec).strip().upper())
        if nom is None:
            v = self.rng.normal(size=3); v /= np.linalg.norm(v)
            nom = (1, *v)
        e = np.deg2rad(self.a.sim_state_error_deg)
        axis = self.rng.uniform(0, np.pi)
        wobble = self._rot(self.rng.normal(0, e)) @ self._ret(self.rng.normal(0, 2 * e), axis)
        self.state = wobble @ np.array(nom, float)
        return None

    def set_sample(self, view):
        self.clock += self.STAGE_S
        self.view = view_of(view)

    def zero_signal(self):
        self.zero = self.ambient

    zero_meter = zero_signal              # you blocking the beam by hand

    def now(self):                 return self.clock
    def move_to(self, deg):        self.pos = float(deg)
    def move_relative(self, d):    self.pos += float(d)
    def position(self):            return self.pos % 360.0

    def read(self):
        self.clock += self.a.qwp_settle + self.READ_S
        laser = self.P0 * (1 + self.a.sim_laser_drift_rel
                           * np.sin(2 * np.pi * self.clock / self.DRIFT_PERIOD_S + 0.7))
        laser *= 1 + self.rng.normal(0, self.a.sim_laser_noise_rel)
        M = {"E": np.eye(4), "S": self.substrate, "F": self.film @ self.substrate,
             "dark": np.zeros((4, 4))}[self.view]
        row = self._row(np.deg2rad(self.pos - self.a.qwp_zero_deg), 0.0,
                        np.deg2rad(waves_to_deg(self.a.qwp_retardance_waves)))
        sig = laser * float(row @ M @ self.state) * (1 + self.rng.normal(0, self.a.sim_noise_rel))
        sig += self.ambient - self.zero
        ref = (0.5 * laser * (1 + self.rng.normal(0, self.a.sim_noise_rel))
               if self.has_ref else float("nan"))
        return sig, ref

    def describe(self):
        return {"backend": "simulated", "film_mueller": self.film.tolist(),
                "substrate_mueller": self.substrate.tolist(),
                "true_sample_mueller": self.film.tolist(), "ambient_W": self.ambient,
                "reference_read": "parallel" if self.has_ref else None,
                "sim_noise_rel": self.a.sim_noise_rel,
                "sim_laser_drift_rel": self.a.sim_laser_drift_rel,
                "sim_laser_noise_rel": self.a.sim_laser_noise_rel,
                "psg": self.psg.describe() if self.psg is not None else None}


# --------------------------------------------------------------------------- #


def sweep(rig, a, tag, span_deg=360.0, on_point=None):
    """One QWP sweep of qwp_steps positions over span_deg -> dict with the
    sweep CSV columns. on_point(k, sw) is called after every point."""
    return pm_sweep(rig, steps=a.qwp_steps, span_deg=span_deg, zero_deg=a.qwp_zero_deg,
                    settle_s=a.qwp_settle, tag=tag, on_point=on_point)


def analyse_sweep(sw, a):
    """Stokes vector for live feedback (laser-corrected when possible)."""
    kw = dict(qwp_zero_deg=a.qwp_zero_deg,
              retardance_deg=waves_to_deg(a.qwp_retardance_waves), s3_sign=a.s3_sign)
    fit = extract_stokes(corrected_power(sw), sw["measured_deg"], **kw)
    if has_reference(sw):
        raw = extract_stokes(sw["power_W"], sw["measured_deg"], **kw)
        fit["rms_raw"] = raw["residual_rms"] / np.mean(sw["power_W"])
    fit["rms_rel"] = fit["residual_rms"] / np.mean(corrected_power(sw))
    return fit


def laser_line(sw):
    r = sw["ref_W"]
    t = sw["t_s"] - sw["t_s"][0]
    slope, icpt = np.polyfit(t, r, 1)
    noise = np.std(r - (slope * t + icpt)) / np.mean(r)
    drift = slope * t[-1] / np.mean(r)
    return f"laser: noise {100 * noise:.2f}%, drift {100 * drift:+.2f}% over the sweep"


def fmt(S):
    n = S / S[0]
    return (f"[{n[0]:.3f} {n[1]:+.3f} {n[2]:+.3f} {n[3]:+.3f}]  "
            f"DOP {np.linalg.norm(n[1:]):.3f}")


def coverage(S_list):
    S = np.column_stack(S_list)
    sv = np.linalg.svd(S / S[0], compute_uv=False)
    rank = int((sv > 1e-2 * sv[0]).sum())
    return rank, (sv[0] / sv[-1] if len(S_list) >= 4 else float("inf"))


def plan_states(psg, specs):
    print("\nstates (PSG)" + ("" if psg.cal.calibrated else
                              "  -- NOMINAL model, the PSG is not calibrated"))
    plans = []
    for k, spec in enumerate(specs, 1):
        try:
            s = psg.solve(spec)
        except ValueError as e:
            raise SystemExit(f"states, entry {k}: {e}") from None
        plans.append(s)
        print(f"  {k}. {s.label:10s} HWP {s.hwp_deg:7.2f}  QWP {s.qwp_deg:7.2f}   "
              f"model miss {s.error_deg:.3f} deg")
    rank, cond = psg_coverage([s.predicted for s in plans])
    if rank < 4:
        raise SystemExit(f"these states cannot reconstruct M (rank {rank} of 4): use at "
                         f"least 4, including a circular one (R or L)")
    print(f"  predicted coverage: rank 4, condition number {cond:.2f} (1.73 = best possible)")
    if not psg.cal.calibrated:
        print("  NOTE: not calibrated, so the states can be far from their targets. They\n"
              "  are measured, so M is still right -- but run `poltk psg calibrate` first.")
    return plans


def concat(*sws):
    """Several sweeps of the same state as one (fitted together)."""
    return {k: np.concatenate([sw[k] for sw in sws]) for k in sws[0]}


def measure_state(r, a, run, idx, seq, live=None, setting=None, label=""):
    """Run the stage through `seq` with one QWP sweep at every stop. Saves
    every sweep (state_NN_E1.csv, ...) and returns (fits, files), keyed E /
    S / F: the sweeps of each kind fitted together, and their file names."""
    sweeps, files, fits = {}, {}, {}
    for k in "ESF":
        sweeps[k], files[k] = [], []
    for kind in seq:
        r.set_sample(kind)
        n = len(sweeps[kind]) + 1
        if live is not None:
            live.start_sweep(f"{label or idx}" + (f" ({n})" if seq.count(kind) > 1 else ""),
                             SIDE[kind], kind,
                             target=None if setting is None else setting.target, psg=setting)
        sw = sweep(r, a, f"{kind}{n}", a.qwp_sweep_deg,
                   on_point=None if live is None else live.point)
        name = f"state_{idx:02d}_{kind}{n}.csv"
        write_sweep(run / name, sw["commanded_deg"], sw["measured_deg"], sw["power_W"],
                    sw["ref_W"], sw["t_s"])
        f = analyse_sweep(sw, a)
        sweeps[kind].append((sw, f))
        files[kind].append(name)
        if live is not None:
            live.end_sweep(idx, SIDE[kind], f["S"])
        fit_txt = (f"fit rms {100 * f['rms_raw']:.2f}% raw -> "
                   f"{100 * f['rms_rel']:.2f}% laser-corrected"
                   if "rms_raw" in f else f"fit rms {100 * f['rms_rel']:.2f}%")
        print(f"  {kind}{n} {KINDS[kind]:9s} {fmt(f['S'])}   {fit_txt}")
    for kind, lst in sweeps.items():
        if not lst:
            continue
        if len(lst) > 1:
            drift = float(sphere_angle_deg(lst[0][1]["S"], lst[-1][1]["S"]))
            print(f"  {KINDS[kind]:9s} first vs last sweep: {drift:.3f} deg apart "
                  f"(drift during this state)")
        fits[kind] = analyse_sweep(concat(*[sw for sw, _ in lst]), a) if len(lst) > 1 \
            else lst[0][1]
    return fits, files


def report_coverage(S_list):
    rank, cond = coverage(S_list)
    if rank < 4:
        hint = ("add a state with a circular component (e.g. R or L)"
                if len(S_list) >= 3 and abs(np.column_stack(S_list)[3]).max() < 0.2
                else "need at least 4 states")
        print(f"  saved. states {len(S_list)} | rank {rank} of 4 -- {hint}")
    else:
        verdict = ("good" if cond < 4 else "usable -- spread the states more"
                   if cond < 10 else "POOR -- states too similar")
        print(f"  saved. states {len(S_list)} | rank 4, condition "
              f"number {cond:.2f} ({verdict}; 1.73 is the best possible)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwp-motor", default="28000005", dest="qwp_motor")
    ap.add_argument("--qwp-stage", default="none", dest="qwp_stage")
    ap.add_argument("--home", type=str2bool, default=True)
    ap.add_argument("--qwp-steps", type=int, default=100, dest="qwp_steps",
                    help="QWP positions per sweep (a count, spread over qwp_sweep_deg)")
    ap.add_argument("--qwp-sweep-deg", type=float, default=360.0, dest="qwp_sweep_deg")
    ap.add_argument("--qwp-settle", type=float, default=0.3, dest="qwp_settle")
    ap.add_argument("--qwp-zero-deg", type=float, default=93.6, dest="qwp_zero_deg",
                    help="stage angle at which the QWP is aligned with the analyzer")
    ap.add_argument("--qwp-retardance-waves", type=float, default=0.25,
                    dest="qwp_retardance_waves",
                    help="QWP retardance at the working wavelength, in WAVES")
    ap.add_argument("--s3-sign", type=int, default=1, choices=(1, -1), dest="s3_sign")
    ap.add_argument("--wavelength-nm", type=float, default=520.0, dest="wavelength_nm")
    ap.add_argument("--pm-serial", default=None, dest="pm_serial",
                    help="signal meter (after the analyzer)")
    ap.add_argument("--pm-average-count", type=int, default=300, dest="pm_average_count",
                    help="averages per reading, on both meters")
    ap.add_argument("--power-range-w", type=float, default=None, dest="power_range_w",
                    help="signal meter range in W; null = auto range")
    ap.add_argument("--ref-pm-serial", default=None, dest="ref_pm_serial",
                    help="reference meter (laser monitor after the polarizer); null = none")
    ap.add_argument("--ref-power-range-w", type=float, default=None, dest="ref_power_range_w")
    ap.add_argument("--zero-meter", type=str2bool, default=True, dest="zero_meter")
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--live-view", type=str2bool, default=True, dest="live_view")
    ap.add_argument("--use-psg", type=str2bool, default=True, dest="use_psg",
                    help="the PSG sets the states (false: you set them by hand)")
    ap.add_argument("--states", nargs="+", default=list(SIX_STATES),
                    help="input states for the PSG: names here; in the JSON also "
                         "[S1, S2, S3] or {\"azimuth_deg\": .., \"ellipticity_deg\": ..}")
    # the translation stage
    ap.add_argument("--use-stage", type=str2bool, default=True, dest="use_stage",
                    help="the translation stage puts empty / substrate / film in the beam")
    ap.add_argument("--stage-motor", default=PLACEHOLDER, dest="stage_motor",
                    help="KST101 serial (starts with 26)")
    ap.add_argument("--stage-name", default=None, dest="stage_name",
                    help="Kinesis actuator name (e.g. ZST225); null = the settings stored "
                         "on the controller")
    ap.add_argument("--stage-home", type=str2bool, default=True, dest="stage_home")
    ap.add_argument("--empty-mm", type=float, default=4.5866, dest="empty_mm")
    ap.add_argument("--substrate-mm", type=float, default=12.7767, dest="substrate_mm")
    ap.add_argument("--film-mm", type=float, default=21.4397, dest="film_mm")
    ap.add_argument("--dark-mm", type=float, default=None, dest="dark_mm",
                    help="a stage position where the holder blocks the beam: automatic "
                         "zeroing and dark check (null: you block the beam at the start)")
    ap.add_argument("--stage-approach-mm", type=float, default=0.2, dest="stage_approach_mm",
                    help="every move arrives from this far below (no backlash)")
    ap.add_argument("--stage-settle", type=float, default=0.5, dest="stage_settle")
    ap.add_argument("--sequence", nargs="+", default=["E", "S", "F", "F", "S", "E"],
                    help="stage stops per state; a palindrome cancels slow drift")
    ap.add_argument("--dark-readings", type=int, default=5, dest="dark_readings")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-state-error-deg", type=float, default=3.0,
                    dest="sim_state_error_deg")
    ap.add_argument("--sim-noise-rel", type=float, default=0.001, dest="sim_noise_rel")
    ap.add_argument("--sim-laser-drift-rel", type=float, default=0.01,
                    dest="sim_laser_drift_rel")
    ap.add_argument("--sim-laser-noise-rel", type=float, default=0.003,
                    dest="sim_laser_noise_rel")
    ap.add_argument("--sim-ambient-rel", type=float, default=0.01, dest="sim_ambient_rel")
    a = parse_with_config(ap, argv, HERE / "measure_mueller.json")
    seq = check_sequence(a.sequence)
    if not stage_configured(a) and not a.simulate:
        raise SystemExit("use_stage is false -- this version needs the translation stage "
                         "(the flippers version is ../mueller_matrix_flippers)")

    delta = waves_to_deg(a.qwp_retardance_waves)
    root = Path(a.out) if a.out else Path.home() / "Desktop"
    run = root / time.strftime("mueller_%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    print(f"\nrun folder : {run}")
    print(f"PSA QWP    : zero at {a.qwp_zero_deg:g} deg, retardance "
          f"{a.qwp_retardance_waves:g} waves = {delta:.2f} deg")
    print(f"sweep      : {a.qwp_steps} steps over {a.qwp_sweep_deg:g} deg, "
          f"{a.wavelength_nm:g} nm")
    print(f"sequence   : {' '.join(seq)} per state ({len(seq)} sweeps)\n")

    meta = {"kind": "mueller_powermeter",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_file": str(Path(a.config).resolve()),
            "settings": vars(a),
            "calibration": {"qwp_zero_deg": a.qwp_zero_deg,
                            "qwp_retardance_waves": a.qwp_retardance_waves,
                            "s3_sign": a.s3_sign, "analyzer_deg": 0.0,
                            "wavelength_nm": a.wavelength_nm},
            "qwp_sweep_deg": a.qwp_sweep_deg, "notes": a.notes, "states": [],
            "sample_mode": {"kind": "t_stage", "sequence": seq,
                            "positions_mm": {"E": a.empty_mm, "S": a.substrate_mm,
                                             "F": a.film_mm, "dark": a.dark_mm}}}

    rig = SimRig(a) if a.simulate else Rig(a)
    with rig as r:
        def save():
            meta["hardware"] = r.describe()
            (run / "run.json").write_text(json.dumps(meta, indent=2, default=str))

        if a.zero_meter and r.has_dark:
            r.set_sample("dark")
            r.zero_signal()
            print("stage at the dark position: signal meter zeroed (room light and dark "
                  "offset removed)")
        elif a.zero_meter:
            if a.simulate:
                r.zero_meter()
            else:
                what = ("both power meters (block the laser BEFORE the beamsplitter)"
                        if r.has_ref else "the power meter")
                input(f"\nBLOCK the beam to zero {what}, then press Enter...")
                r.zero_meter()
                input("UNBLOCK the beam, then press Enter -- the rest is automatic...")
        save()

        from live_view import LiveView
        live = LiveView(a, len(a.states), len(seq), enabled=a.live_view)
        S_list = []
        t_run = r.now()

        def one_state(idx, spec, setting, label):
            fits, files = measure_state(r, a, run, idx, seq, live=live, setting=setting,
                                        label=label)
            entry = {"index": idx, "label": label,
                     **{KINDS[k]: v for k, v in files.items() if v}}
            if setting is not None:
                entry["psg"] = setting.to_dict()
            if "E" in fits:
                entry["input_miss_deg"] = float(sphere_angle_deg(fits["E"]["S"],
                                                                 setting.target)) \
                    if setting is not None else None
                if setting is not None:
                    print(f"  the input state (empty) is {entry['input_miss_deg']:.2f} deg "
                          f"from its target")
            meta["states"].append(entry)
            save()
            S_list.append(fits["E" if "E" in fits else "S"]["S"])
            report_coverage(S_list)

        if r.psg is not None:
            plan_states(r.psg, a.states)
            for idx, spec in enumerate(a.states, 1):
                print(f"\n=== state {idx} of {len(a.states)}: {state_label(spec)} ===")
                setting = r.set_state(spec)
                print(f"  PSG: HWP {setting.hwp_deg:.2f} deg, QWP {setting.qwp_deg:.2f} deg")
                one_state(idx, spec, setting, setting.label)
                done = r.now() - t_run
                print(f"  {done / 60:.1f} min so far, about "
                      f"{done / idx * (len(a.states) - idx) / 60:.0f} min left")
        else:
            idx = 0
            while True:
                idx += 1
                print(f"\n=== state {idx} ===")
                r.set_sample("E")
                label = input("Set the input state by hand, then type a label for it"
                              "\n  (Enter = finish): ").strip()
                if not label:
                    break
                r.set_state(label)
                one_state(idx, label, None, label)

        if r.has_dark and meta["states"]:
            r.set_sample("dark")
            dark = np.mean([r.read()[0] for _ in range(max(1, a.dark_readings))])
            level = np.mean([S[0] for S in S_list]) / 2
            meta["dark_end_W"] = float(dark)
            print(f"\ndark at the end: {dark * 1e9:+.1f} nW = {100 * dark / level:+.2f}% of "
                  f"the signal" + ("  -- the zero drifted (room light changed?)"
                                   if abs(dark) > 0.005 * level else "  (ok)"))
            save()
        r.set_sample("E")                         # park: the beam is clear
        live.finish(run / "live_view.png")

    print(f"\n{len(meta['states'])} states saved in {run}")
    print("Analyse with: python analyze_mueller.py   (set run_dir in analyze_mueller.json)")
    if live.enabled and live.interactive and not live.closed:
        print("(close the live window to finish)")
        live.plt.show()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolkitError as e:
        sys.exit(f"error: {e}")
