#!/usr/bin/env python3
"""Measure the Mueller matrix of a sample -- fully automatic with the PSG and flippers.

    python measure_mueller.py                  # reads measure_mueller.json
    python measure_mueller.py --simulate       # rehearse, no hardware

Optical layout

    laser -> polarizer (V) -> beamsplitter -> HWP -> QWP -> [sample] -> QWP (rotating) -> analyzer -> power meter
                                   |          '-- PSG --'              '------------ polarimeter ------------'
                                   '-> reference power meter (laser monitor)

With use_psg (the default) the PSG -- the motorized HWP and QWP, driven by
polarization_toolkit -- sets the input states listed in `states` (names
H V D A R L, Stokes vectors [S1, S2, S3], or {"azimuth_deg", "ellipticity_deg"}).
Its settings and calibration are read from ~/.poltk (`poltk psg show`). For
each state:

    1. the PSG sets the state
    2. you REMOVE the sample      -> the script measures the input state
    3. you PUT THE SAMPLE BACK    -> the script measures through the sample

With use_psg false you set each state by hand and type a label for it
instead, as many as you like (Enter with no label finishes). Either way it
reports after every state how well the states so far pin down M (rank and
condition number).

Flippers (use_flippers): the sample is half film, half bare substrate, in
one beam, and two motorized flippers (MFF10x) each block one half. Then
nothing is done by hand:

    start       both halves blocked -> the signal meter is zeroed (this also
                removes room light and the meter's dark offset)
    each state  the PSG sets it; film half blocked -> the REFERENCE sweep
                (through the substrate); substrate half blocked -> the SAMPLE
                sweep (through film + substrate)
    empty       (empty_at_end) you remove the film and the substrate, once;
                every state is measured again through each EMPTY aperture
    end         both blocked -> a dark reading checks the zero did not
                drift; then both open

With the empty apertures the analysis references each half to its own
empty aperture -- M_film = (film half / its empty) . (substrate half / its
empty)^-1 -- so differences between the two halves of the beam cancel and
M00 is the film's own transmittance; the two empty apertures against each
other are a null test (should be the identity). Without them,
M = S_sample . pinv(S_reference) is the film relative to the substrate,
M_film = M_(film+substrate) . M_substrate^-1. It is exact when the light
crosses the substrate BEFORE the film (film facing the polarimeter), even for
a birefringent substrate; the other way round it holds for an isotropic
substrate (glass). M00 is the film half's power over the substrate half's:
the film's transmittance times however unevenly the beam is split between
the halves -- the normalized M does not depend on that.

The states do NOT need to be exact. Each one is measured, not assumed, so an
approximate H or R is as good as a perfect one (the PSG's calibration makes
them accurate, but M does not depend on it). What matters is that

  * the state does not change between its "input" and "through sample"
    measurements -- don't touch the polarizer optics in between,
  * together the states span everything: at least 4, and at least one with
    a real circular component (linear states alone give rank 3),
  * they are well spread on the Poincare sphere (H V D A R L is ideal).

Laser monitor: set ref_pm_serial in the JSON and the second power meter is
read at every QWP step, at the same moment and over the same averaging
window as the signal meter. Dividing by it removes laser power drift and
noise -- within a sweep, and between the input and through-sample sweeps
(which is what keeps M00 an honest transmittance). The beamsplitter sits
after the polarizer, so the light it splits has a fixed polarization and
its split ratio does not change with the state you prepare.

Every state is saved the moment it is measured, so an interrupted run keeps
what it has.
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


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


PLACEHOLDER = "37XXXXXX"

# which halves of the beam are BLOCKED for each view of the sample
VIEWS = {"reference": ("film",),            # through the substrate only
         "sample": ("substrate",),          # through film + substrate only
         "dark": ("film", "substrate"),     # no laser light at the polarimeter
         "open": ()}


def view_of(x):
    """set_sample accepts a view name, or True / False (sample in / out) as
    the other scripts pass it."""
    if isinstance(x, bool):
        return "sample" if x else "reference"
    if x not in VIEWS:
        raise ValueError(f"unknown sample view {x!r}: {sorted(VIEWS)}")
    return x


def flippers_configured(a) -> bool:
    return bool(getattr(a, "use_flippers", False))


class Flippers:
    """The two MFF10x flippers in front of the half film / half substrate
    sample: film_flipper blocks the film half, substrate_flipper the
    substrate half. flipper_film_block_position / flipper_subst_block_position
    is the position (1 or 2) that puts each one's block INTO the beam (null:
    flipper_block_position, for both)."""

    def __init__(self, a):
        self.serials = {"film": str(a.film_flipper), "substrate": str(a.substrate_flipper)}
        common = getattr(a, "flipper_block_position", 1)
        film = getattr(a, "flipper_film_block_position", None)
        subst = getattr(a, "flipper_subst_block_position", None)
        self.block = {"film": int(common if film is None else film),
                      "substrate": int(common if subst is None else subst)}
        if any(b not in (1, 2) for b in self.block.values()):
            raise ValueError("flipper block positions must be 1 or 2")
        self.settle = float(a.flipper_settle)
        self.dev = {}
        self.view = None

    def __enter__(self):
        for half, serial in self.serials.items():
            if serial in ("", "None", PLACEHOLDER):
                raise ToolkitError(f"no serial for the {half} flipper -- set {half}_flipper "
                                   f"in measure_mueller.json (MFF serials start with 37), "
                                   f"or use_flippers: false")
        from flipper import FilterFlipper
        try:
            for half, serial in self.serials.items():
                self.dev[half] = FilterFlipper(serial).connect()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        print(f"flippers        : film {self.serials['film']}, substrate "
              f"{self.serials['substrate']} (block = position {self.block['film']} / "
              f"{self.block['substrate']})")
        self.select("open")
        return self

    def __exit__(self, *exc):
        try:
            if len(self.dev) == 2:
                self.select("open")          # leave the beam clear for other scripts
        except Exception:
            pass
        for d in self.dev.values():
            try:
                d.close()
            except Exception:
                pass
        self.dev = {}
        return False

    def select(self, view):
        view = view_of(view)
        for half, d in self.dev.items():
            b = self.block[half]
            d.move_to_position(b if half in VIEWS[view] else 3 - b)
        time.sleep(self.settle)
        self.view = view

    def describe(self):
        return {"film_flipper": self.serials["film"],
                "substrate_flipper": self.serials["substrate"],
                "film_block_position": self.block["film"],
                "subst_block_position": self.block["substrate"]}


class Rig(PowerMeterPolarimeter):
    """The polarimeter (QWP stage + signal / reference PM100D, from the
    toolkit), with use_psg the PSG, which sets the input states, and with
    use_flippers the two flippers, which choose the sample half. Without the
    PSG you set the states by hand; without the flippers you move the sample."""

    def __init__(self, a):
        super().__init__(PSAConfig.from_mapping(a))
        self.psg = PSG.from_files() if getattr(a, "use_psg", False) else None
        self.flippers = Flippers(a) if flippers_configured(a) else None

    def __enter__(self):
        super().__enter__()
        try:
            for part in (self.psg, self.flippers):
                if part is not None:
                    part.__enter__()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for part in (self.flippers, self.psg):
            if part is not None:
                try:
                    part.__exit__(*exc)
                except Exception:
                    pass
        return super().__exit__(*exc)

    def set_state(self, spec):
        return self.psg.set_state(spec) if self.psg is not None else None

    def set_sample(self, view):
        if self.flippers is not None:
            self.flippers.select(view)

    def zero_signal(self):
        """Zero the signal meter only (the reference sees the laser before the
        flippers, so it cannot be darkened by them)."""
        self.pm.zero()

    def describe(self):
        return {**super().describe(), "psg": self.psg.describe() if self.psg else None,
                "flippers": self.flippers.describe() if self.flippers else None}


class SimRig:
    """Rehearsal stand-in with a known sample, so a run can be checked.

    With use_psg the states come from a simulated PSG (polarization_toolkit),
    whose true optics differ from its calibration in ~/.poltk/simulated as a
    real one's would. Without it, hand-made states are imperfect on purpose
    (sim_state_error_deg), to show that approximate states still give the
    right M. The laser drifts and
    fluctuates (sim_laser_*), seen by both meters, to show what the
    reference removes. Time runs on a virtual clock, so drift is realistic
    even when the rehearsal itself runs in seconds.
    """

    simulated = True
    READ_S = 0.3             # one meter reading
    PSG_S = 5.0              # the PSG stages moving
    FLIP_S = 1.0             # a flipper flipping
    SWAP_S = 20.0            # you taking the sample out / putting it back
    STATE_S = 60.0           # you setting a new state
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
        # a partial diattenuator (D = 0.3 at 15 deg) ...
        D, ax = 0.3, np.array([np.cos(d(30)), np.sin(d(30)), 0.0])
        MD = np.eye(4); MD[0, 1:] = MD[1:, 0] = D * ax
        sq = np.sqrt(1 - D * D)
        MD[1:, 1:] = sq * np.eye(3) + (1 - sq) * np.outer(ax, ax)
        # ... then a 60 deg retarder at 20 deg, then a weak depolarizer
        self.sample = np.diag([1, .95, .95, .95]) @ retarder_mueller(d(60), d(20)) @ MD
        self.sample *= 0.8                                   # 80% transmission
        self.state = np.array([1., 1, 0, 0])
        self.inside = False
        self.pos = 0.0
        self.P0 = 1e-3                                       # 1 mW
        self.clock = 0.0
        self.has_ref = bool(a.ref_pm_serial)
        self.parallel = True
        self.psg = (SimulatedPSG.from_files(simulate=True)
                    if getattr(a, "use_psg", False) else None)
        # flippers: a half film / half substrate sample. The substrate is a
        # weakly birefringent plate; the film is the sample above; the beam
        # is split unevenly between the halves; room light reaches the meter
        self.flippers = flippers_configured(a)
        self.removed = False                     # film + substrate taken out
        self.view = "open"
        self.substrate = 0.92 * retarder_mueller(d(3.0), d(10))
        self.split = {"reference": 0.52, "sample": 0.48}
        self.ambient = getattr(a, "sim_ambient_rel", 0.0) * self.P0
        self.zero = 0.0
        if self.flippers:
            # what the analysis should find: the film relative to the substrate,
            # scaled by the uneven split
            self.sample_rel = self.split["sample"] / self.split["reference"] * self.sample

    def __enter__(self):
        if self.psg is not None:
            self.psg.__enter__()
        print("signal meter    : simulated")
        print("reference meter : " + ("simulated (laser monitor)" if self.has_ref else
                                      "none (ref_pm_serial is null) -- no laser correction"))
        return self

    def __exit__(self, *exc):
        return False

    def set_state(self, spec):
        if self.psg is not None:
            self.clock += self.PSG_S
            setting = self.psg.set_state(spec)
            self.state = self.psg.stokes()
            return setting
        label = spec
        self.clock += self.STATE_S
        nom = self.NOMINAL.get(label.strip().upper())
        if nom is None:
            v = self.rng.normal(size=3); v /= np.linalg.norm(v)
            nom = (1, *v)
            print(f"  [sim] '{label}' is not H/V/D/A/R/L -- using a random state")
        e = np.deg2rad(self.a.sim_state_error_deg)
        axis = self.rng.uniform(0, np.pi)
        wobble = self._rot(self.rng.normal(0, e)) @ self._ret(self.rng.normal(0, 2 * e), axis)
        self.state = wobble @ np.array(nom, float)          # imperfect, like by hand
        return None

    def set_sample(self, view):
        if self.flippers:
            self.clock += self.FLIP_S
            self.view = view_of(view)
        else:
            self.clock += self.SWAP_S
            self.inside = bool(view) if isinstance(view, bool) else view == "sample"

    def zero_signal(self):
        self.zero = self.ambient if self.view == "dark" else self.zero

    def remove_sample(self):
        self.clock += 60.0
        self.removed = True

    def now(self):                 return self.clock
    def move_to(self, deg):        self.pos = float(deg)
    def move_relative(self, d):    self.pos += float(d)
    def position(self):            return self.pos % 360.0
    def zero_meter(self):          pass

    def read(self):
        self.clock += self.a.qwp_settle + self.READ_S
        laser = self.P0 * (1 + self.a.sim_laser_drift_rel
                           * np.sin(2 * np.pi * self.clock / self.DRIFT_PERIOD_S + 0.7))
        laser *= 1 + self.rng.normal(0, self.a.sim_laser_noise_rel)   # both meters see it
        if self.flippers:
            Sin = self.state if self.removed else self.substrate @ self.state
            film = Sin if self.removed else self.sample @ Sin
            S = {"reference": self.split["reference"] * Sin,
                 "sample": self.split["sample"] * film,
                 "dark": 0.0 * Sin,
                 "open": self.split["reference"] * Sin
                 + self.split["sample"] * film}[self.view]
        else:
            S = (self.sample @ self.state) if self.inside else self.state
        row = self._row(np.deg2rad(self.pos - self.a.qwp_zero_deg), 0.0,
                        np.deg2rad(waves_to_deg(self.a.qwp_retardance_waves)))
        sig = laser * float(row @ S) * (1 + self.rng.normal(0, self.a.sim_noise_rel))
        if self.flippers:
            sig += self.ambient - self.zero          # room light, minus the meter's zero
        ref = (0.5 * laser * (1 + self.rng.normal(0, self.a.sim_noise_rel))
               if self.has_ref else float("nan"))
        return sig, ref

    def describe(self):
        truth = self.sample_rel if self.flippers else self.sample
        return {"backend": "simulated", "true_sample_mueller": truth.tolist(),
                "film_mueller": self.sample.tolist(),
                "flippers": ({"substrate_mueller": self.substrate.tolist(),
                              "beam_split": self.split, "ambient_W": self.ambient}
                             if self.flippers else None),
                "reference_read": "parallel" if self.has_ref else None,
                "sim_state_error_deg": self.a.sim_state_error_deg,
                "sim_noise_rel": self.a.sim_noise_rel,
                "sim_laser_drift_rel": self.a.sim_laser_drift_rel,
                "sim_laser_noise_rel": self.a.sim_laser_noise_rel,
                "psg": self.psg.describe() if self.psg is not None else None}


# --------------------------------------------------------------------------- #


def sweep(rig, a, tag, span_deg=360.0, on_point=None):
    """One QWP sweep of qwp_steps positions over span_deg -> dict with the
    sweep CSV columns. 360 deg also separates beam walk (360-deg period).
    on_point(k, sw) is called after every point (the live view)."""
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
    """Laser noise and drift during one sweep, from the reference meter."""
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
    """Rank and condition number of the input states measured so far."""
    S = np.column_stack(S_list)
    sv = np.linalg.svd(S / S[0], compute_uv=False)
    rank = int((sv > 1e-2 * sv[0]).sum())
    return rank, (sv[0] / sv[-1] if len(S_list) >= 4 else float("inf"))


def plan_states(psg, specs):
    """The PSG settings of every state, and whether together they can
    reconstruct M -- before anything is measured."""
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
    rank, cond = psg_coverage([s.predicted for s in plans]) if len(plans) >= 1 else (0, 0)
    if rank < 4:
        raise SystemExit(f"these states cannot reconstruct M (rank {rank} of 4): use at "
                         f"least 4, including a circular one (R or L)")
    print(f"  predicted coverage: rank 4, condition number {cond:.2f} (1.73 = best possible)")
    if not psg.cal.calibrated:
        print("  NOTE: not calibrated, so the states can be far from their targets. They\n"
              "  are measured, so M is still right -- but run `poltk psg calibrate` first.")
    return plans


def measure_state(r, a, run, idx, auto=False, empty=False, live=None, setting=None,
                  label=""):
    """The input (reference) sweep, then the through-sample sweep. With the
    flippers (auto) they choose the substrate / film half; otherwise you take
    the sample out and put it back. Saves both and prints the live feedback;
    returns the two fits."""
    sweeps, fits = {}, {}
    steps = ((("in0", "reference", "empty"), ("out0", "sample", "empty")) if empty else
             (("in", "reference", "substr"), ("out", "sample", "film")) if auto else
             (("in", "REMOVE the sample, then press Enter to measure the input state...",
               "input"),
              ("out", "PUT THE SAMPLE BACK, then press Enter to measure through it...",
               "output")))
    for side, what, tag in steps:
        if auto:
            r.set_sample(what)
        else:
            input(what)
            r.set_sample(side == "out")
        if live is not None:
            live.start_sweep(label or str(idx), side, what if auto else "",
                             target=None if setting is None else setting.target, psg=setting)
        sw = sweep(r, a, tag, a.qwp_sweep_deg,
                   on_point=None if live is None else live.point)
        write_sweep(run / f"state_{idx:02d}_{side}.csv", sw["commanded_deg"],
                    sw["measured_deg"], sw["power_W"], sw["ref_W"], sw["t_s"])
        f = analyse_sweep(sw, a)
        sweeps[side], fits[side] = sw, f
        if live is not None:
            live.end_sweep(idx, side, f["S"])
        name = {"in": "S_ref" if auto else "S_in ", "out": "S_out",
                "in0": "S_e,s", "out0": "S_e,f"}[side]
        fit_txt = (f"fit rms {100 * f['rms_raw']:.2f}% raw -> "
                   f"{100 * f['rms_rel']:.2f}% laser-corrected"
                   if "rms_raw" in f else f"fit rms {100 * f['rms_rel']:.2f}%")
        print(f"  {name} = {fmt(f['S'])}   {fit_txt}")
        if has_reference(sw):
            print(f"          {laser_line(sw)}")
    a_, b_ = ("in0", "out0") if empty else ("in", "out")
    if has_reference(sweeps[a_]) and not empty:
        change = np.mean(sweeps[b_]["ref_W"]) / np.mean(sweeps[a_]["ref_W"]) - 1
        print(f"  laser changed {100 * change:+.2f}% between the input and "
              f"through-sample sweeps (corrected)")
    return fits


def report_coverage(S_in_list):
    rank, cond = coverage(S_in_list)
    if rank < 4:
        hint = ("add a state with a circular component (e.g. R or L)"
                if len(S_in_list) >= 3 and abs(np.column_stack(S_in_list)[3]).max() < 0.2
                else "need at least 4 states")
        print(f"  saved. states {len(S_in_list)} | rank {rank} of 4 -- {hint}")
    else:
        verdict = ("good" if cond < 4 else "usable -- spread the states more"
                   if cond < 10 else "POOR -- states too similar")
        print(f"  saved. states {len(S_in_list)} | rank 4, condition "
              f"number {cond:.2f} ({verdict}; 1.73 is the best possible)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwp-motor", default="28000005", dest="qwp_motor")
    ap.add_argument("--qwp-stage", default="none", dest="qwp_stage")
    ap.add_argument("--home", type=lambda s: str(s).lower() in ("1", "true", "yes"),
                    default=True)
    ap.add_argument("--qwp-steps", type=int, default=100, dest="qwp_steps",
                    help="QWP positions per sweep (a count, spread over qwp_sweep_deg)")
    ap.add_argument("--qwp-sweep-deg", type=float, default=360.0, dest="qwp_sweep_deg",
                    help="QWP sweep span: 360 (a full turn, beam walk separated) "
                         "or 180")
    ap.add_argument("--qwp-settle", type=float, default=0.3, dest="qwp_settle")
    ap.add_argument("--qwp-zero-deg", type=float, default=93.6, dest="qwp_zero_deg",
                    help="stage angle at which the QWP is aligned with the analyzer")
    ap.add_argument("--qwp-retardance-waves", type=float, default=0.25,
                    dest="qwp_retardance_waves",
                    help="QWP retardance at the working wavelength, in WAVES "
                         "as the manufacturer quotes it (0.25 = ideal)")
    ap.add_argument("--s3-sign", type=int, default=1, choices=(1, -1), dest="s3_sign")
    ap.add_argument("--wavelength-nm", type=float, default=520.0, dest="wavelength_nm")
    ap.add_argument("--pm-serial", default=None, dest="pm_serial",
                    help="signal meter (after the analyzer); null = the only "
                         "meter that is not the reference")
    ap.add_argument("--pm-average-count", type=int, default=300, dest="pm_average_count",
                    help="averages per reading, on both meters")
    ap.add_argument("--power-range-w", type=float, default=None, dest="power_range_w",
                    help="signal meter range in W; null = auto range")
    ap.add_argument("--ref-pm-serial", default=None, dest="ref_pm_serial",
                    help="reference meter (laser monitor after the polarizer); "
                         "null = no reference")
    ap.add_argument("--ref-power-range-w", type=float, default=None,
                    dest="ref_power_range_w",
                    help="reference meter range in W; null = auto range")
    ap.add_argument("--zero-meter", type=lambda s: str(s).lower() in ("1", "true", "yes"),
                    default=True, dest="zero_meter")
    ap.add_argument("--out", default=None, help="output root (default: Desktop)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--live-view", type=str2bool, default=True, dest="live_view",
                    help="a live window: the sweep as it comes in, the ellipse, the "
                         "Poincare sphere, a live Mueller matrix, the flippers and time left")
    ap.add_argument("--use-psg", type=str2bool, default=True, dest="use_psg",
                    help="the PSG sets the states (false: you set them by hand)")
    ap.add_argument("--use-flippers", type=str2bool, default=False, dest="use_flippers",
                    help="two flippers choose the film / substrate half of the sample "
                         "(fully automatic run)")
    ap.add_argument("--film-flipper", default=PLACEHOLDER, dest="film_flipper",
                    help="serial of the flipper that blocks the FILM half")
    ap.add_argument("--substrate-flipper", default=PLACEHOLDER, dest="substrate_flipper",
                    help="serial of the flipper that blocks the SUBSTRATE half")
    ap.add_argument("--flipper-block-position", type=int, default=1, choices=(1, 2),
                    dest="flipper_block_position",
                    help="block position for both flippers, unless each has its own")
    ap.add_argument("--flipper-film-block-position", type=int, default=None, choices=(1, 2),
                    dest="flipper_film_block_position",
                    help="position that puts the FILM flipper's block into the beam "
                         "(null: flipper_block_position)")
    ap.add_argument("--flipper-subst-block-position", type=int, default=None, choices=(1, 2),
                    dest="flipper_subst_block_position",
                    help="position that puts the SUBSTRATE flipper's block into the beam "
                         "(null: flipper_block_position)")
    ap.add_argument("--flipper-settle", type=float, default=0.5, dest="flipper_settle")
    ap.add_argument("--empty-at-end", type=str2bool, default=True, dest="empty_at_end",
                    help="flippers: at the end you remove film + substrate, and every state "
                         "is measured again through each empty aperture")
    ap.add_argument("--dark-readings", type=int, default=5, dest="dark_readings",
                    help="flippers: readings averaged for the dark check at the end")
    ap.add_argument("--states", nargs="+", default=list(SIX_STATES),
                    help="input states for the PSG: names here; in the JSON also "
                         "[S1, S2, S3] or {\"azimuth_deg\": .., \"ellipticity_deg\": ..}")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-state-error-deg", type=float, default=3.0,
                    dest="sim_state_error_deg")
    ap.add_argument("--sim-noise-rel", type=float, default=0.001, dest="sim_noise_rel",
                    help="independent noise of each meter (relative)")
    ap.add_argument("--sim-laser-drift-rel", type=float, default=0.01,
                    dest="sim_laser_drift_rel", help="laser drift amplitude (relative)")
    ap.add_argument("--sim-laser-noise-rel", type=float, default=0.003,
                    dest="sim_laser_noise_rel", help="laser noise (relative)")
    ap.add_argument("--sim-ambient-rel", type=float, default=0.01, dest="sim_ambient_rel",
                    help="flippers: room light at the signal meter (relative)")
    a = parse_with_config(ap, argv, HERE / "measure_mueller.json")

    delta = waves_to_deg(a.qwp_retardance_waves)
    root = Path(a.out) if a.out else Path.home() / "Desktop"
    run = root / time.strftime("mueller_%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    print(f"\nrun folder : {run}")
    print(f"PSA QWP    : zero at {a.qwp_zero_deg:g} deg, retardance "
          f"{a.qwp_retardance_waves:g} waves = {delta:.2f} deg")
    print(f"sweep      : {a.qwp_steps} steps over {a.qwp_sweep_deg:g} deg "
          f"({a.qwp_sweep_deg / a.qwp_steps:.2f} deg each), {a.wavelength_nm:g} nm\n")

    meta = {"kind": "mueller_powermeter",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_file": str(Path(a.config).resolve()),
            "settings": vars(a),
            "calibration": {"qwp_zero_deg": a.qwp_zero_deg,
                            "qwp_retardance_waves": a.qwp_retardance_waves,
                            "s3_sign": a.s3_sign, "analyzer_deg": 0.0,
                            "wavelength_nm": a.wavelength_nm},
            "qwp_sweep_deg": a.qwp_sweep_deg, "notes": a.notes, "states": []}
    auto = flippers_configured(a)
    if auto:
        meta["sample_mode"] = {
            "kind": "flippers", "reference": "substrate half (film half blocked)",
            "sample": "film half (substrate half blocked)",
            "result": "M = M_(film+substrate) . M_substrate^-1, the film relative to the "
                      "substrate; M00 includes the beam's split between the halves"}

    rig = SimRig(a) if a.simulate else Rig(a)
    with rig as r:
        def save():
            meta["hardware"] = r.describe()
            (run / "run.json").write_text(json.dumps(meta, indent=2, default=str))

        if a.zero_meter and auto:
            r.set_sample("dark")
            r.zero_signal()
            print("\nboth halves blocked: signal meter zeroed (room light and dark "
                  "offset removed)")
        elif a.zero_meter:
            what = ("both power meters (block the laser BEFORE the beamsplitter)"
                    if r.has_ref else "the power meter")
            input(f"\nBLOCK the beam to zero {what}, then press Enter...")
            r.zero_meter()
            input("UNBLOCK the beam, then press Enter...")
        save()

        S_in_list = []
        from live_view import LiveView
        per_state = 2 + (2 if auto and a.empty_at_end and r.psg is not None else 0)
        live = LiveView(a, len(a.states), per_state, enabled=a.live_view)
        if r.psg is not None:
            plan_states(r.psg, a.states)
            for idx, spec in enumerate(a.states, 1):
                print(f"\n=== state {idx} of {len(a.states)}: {state_label(spec)} ===")
                setting = r.set_state(spec)
                print(f"  PSG: HWP {setting.hwp_deg:.2f} deg, QWP {setting.qwp_deg:.2f} deg "
                      f"(model: {setting.error_deg:.2f} deg from the target)")
                fits = measure_state(r, a, run, idx, auto, live=live, setting=setting,
                                     label=setting.label)
                miss = float(sphere_angle_deg(fits["in"]["S"], setting.target))
                if auto:
                    print(f"  through the substrate the state is {miss:.2f} deg from the "
                          f"target (the substrate's own effect included)")
                else:
                    print(f"  measured input is {miss:.2f} deg from the target on the "
                          f"Poincare sphere" + (" -- recalibrate the PSG? (poltk psg "
                                                 "calibrate)" if miss > 2.0
                                                 and r.psg.cal.calibrated else ""))
                meta["states"].append({"index": idx, "label": setting.label,
                                       "input": f"state_{idx:02d}_in.csv",
                                       "output": f"state_{idx:02d}_out.csv",
                                       "psg": setting.to_dict(), "input_miss_deg": miss})
                save()
                S_in_list.append(fits["in"]["S"])
                report_coverage(S_in_list)
        else:
            idx = 0
            while True:
                idx += 1
                print(f"\n=== state {idx} ===")
                label = input("Set the input state by hand, then type a label for it"
                              "\n  (e.g. H, D, R -- approximate is fine; Enter = finish): "
                              ).strip()
                if not label:
                    rank = coverage(S_in_list)[0] if S_in_list else 0
                    if rank < 4:
                        ans = input(f"Only rank {rank} of 4 -- the analysis cannot "
                                    f"reconstruct M. Finish anyway? [y/N]: ").strip().lower()
                        if ans != "y":
                            idx -= 1
                            continue
                    break
                r.set_state(label)
                fits = measure_state(r, a, run, idx, auto, live=live, label=label)
                meta["states"].append({"index": idx, "label": label,
                                       "input": f"state_{idx:02d}_in.csv",
                                       "output": f"state_{idx:02d}_out.csv"})
                save()
                S_in_list.append(fits["in"]["S"])
                report_coverage(S_in_list)

        if auto and a.empty_at_end and r.psg is not None and meta["states"]:
            print("\n=== EMPTY APERTURES ===")
            if a.simulate:
                r.remove_sample()
            else:
                input("REMOVE the film and the substrate (leave the apertures), then press "
                      "Enter -- the rest is automatic...")
            if a.zero_meter:
                r.set_sample("dark")
                r.zero_signal()
            for entry, spec in zip(meta["states"], a.states):
                idx = entry["index"]
                print(f"\n--- empty, state {idx} of {len(meta['states'])}: {entry['label']} ---")
                setting = r.set_state(spec)
                fits = measure_state(r, a, run, idx, auto, empty=True, live=live,
                                     setting=setting, label=entry["label"])
                null = float(sphere_angle_deg(fits["in0"]["S"], fits["out0"]["S"]))
                ratio = fits["out0"]["S"][0] / fits["in0"]["S"][0]
                print(f"  the two empty apertures differ by {null:.2f} deg on the Poincare "
                      f"sphere; power film/substrate aperture {ratio:.3f}")
                entry.update(input_empty=f"state_{idx:02d}_in0.csv",
                             output_empty=f"state_{idx:02d}_out0.csv",
                             empty_null_deg=null)
                save()
            meta["sample_mode"]["empty"] = ("each state measured again through the empty "
                                            "apertures (in0: substrate's, out0: film's)")
            save()

        if auto and meta["states"]:
            r.set_sample("dark")
            dark = np.mean([r.read()[0] for _ in range(max(1, a.dark_readings))])
            level = np.mean([S[0] for S in S_in_list]) / 2      # ~ mean signal
            meta["dark_end_W"] = float(dark)
            print(f"\ndark at the end (both halves blocked): {dark * 1e9:+.1f} nW = "
                  f"{100 * dark / level:+.2f}% of the signal"
                  + ("  -- the zero drifted (room light changed?): DOP and M00 are off "
                     "by about that much" if abs(dark) > 0.005 * level else "  (ok)"))
            r.set_sample("open")
            save()

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
