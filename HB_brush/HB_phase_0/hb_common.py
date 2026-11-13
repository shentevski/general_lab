"""Shared pieces of the HB_phase_0 scripts.

    LED (DC40) -> element on the PRMTZ8 rotation stage -> output -> CCT10 spectrometer
                  (passes one band; its wavelength            (calibration, set_wavelength)
                   changes with the angle)                 or power meter
                                                              (constant_power_scan)

calibration.py          for every angle, a spectrum at every LED current:
                        wavelength(angle) and peak amplitude(angle, current)
set_wavelength.py       a wavelength -> angle; a peak amplitude -> LED current
constant_power_scan.py  steps the wavelength; the power meter holds the power

Here: the settings, the hardware (stage + LED + spectrometer + power meter), a
simulator of the whole setup for rehearsal (--simulate), the peak finder and
the calibration files.

Hardware libraries (SCRIPTS/HARDWARE, Windows only, `pip install -e .` each):
thorlabs_motion (kcube), thorlabs_led, thorlabs_spectrometer, thorlabs_powermeter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y", "on")


def float_or_none(s):
    return None if str(s).lower() in ("", "none", "null") else float(s)


def int_or_none(s):
    return None if str(s).lower() in ("", "none", "null") else int(s)


def float_list(s):
    """'1, 2, 3' (command line) or [1, 2, 3] (JSON) -> [1.0, 2.0, 3.0]; empty / null -> None."""
    if s is None:
        return None
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s] or None
    s = str(s).strip()
    if s.lower() in ("", "none", "null"):
        return None
    return [float(x) for x in s.strip("[]").replace(",", " ").split()]


def unique_keep_order(values):
    """The values without repeats, in their first order."""
    out, seen = [], set()
    for v in values:
        k = round(float(v), 6)
        if k not in seen:
            seen.add(k)
            out.append(float(v))
    return out


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #


def hardware_args(ap):
    """Settings shared by all three scripts. They live in calibration.json only,
    so the hardware is described in one place."""
    g = ap.add_argument_group("shared settings (kept in calibration.json)")
    g.add_argument("--stage-serial", default=None, dest="stage_serial",
                   help="KDC101 serial of the PRMTZ8 (27...); null = the only KDC101 found")
    g.add_argument("--stage-name", default="PRMTZ8", dest="stage_name",
                   help="Kinesis stage name; 'none' = settings stored on the controller")
    g.add_argument("--home", choices=("auto", "always", "never"), default="auto",
                   help="auto: home only if the controller says it isn't homed")
    g.add_argument("--stage-settle-s", type=float, default=0.3, dest="stage_settle_s",
                   help="wait after every stage move")
    g.add_argument("--led-serial", default=None, dest="led_serial",
                   help="DC40 serial; null = the first one found")
    g.add_argument("--led-max-ma", type=float_or_none, default=None, dest="led_max_ma",
                   help="never drive the LED above this; null = the LED's own limit")
    g.add_argument("--led-settle-s", type=float, default=1.0, dest="led_settle_s",
                   help="wait after every LED current change (the LED heats and dims "
                        "slightly after a step)")
    g.add_argument("--spectrometer-id", default=None, dest="spectrometer_id",
                   help="CCT device ID; null = the first one found")
    g.add_argument("--hw-average", type=int, default=5, dest="hw_average",
                   help="frames the spectrometer averages per spectrum")
    g.add_argument("--full-scale-counts", type=float, default=65535,
                   dest="full_scale_counts", help="raw counts at saturation")
    g.add_argument("--background", choices=("led_off", "shutter"), default="led_off",
                   help="led_off: spectrum with the LED off (dark + room light); "
                        "shutter: spectrometer shutter closed (dark only)")
    g.add_argument("--background-frames", type=int, default=3, dest="background_frames",
                   help="spectra averaged per background")
    g.add_argument("--data-dir", default=None, dest="data_dir",
                   help="where data is saved; null = Desktop/HB_data")
    g.add_argument("--simulate", action="store_true",
                   help="rehearse without hardware (saves under <data_dir>/simulated)")


def _dests(add):
    p = argparse.ArgumentParser(add_help=False)
    add(p)
    return {x.dest for x in p._actions}


HARDWARE_KEYS = _dests(hardware_args)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}")


def parse_config(ap, argv, own_cfg: Path, hardware_cfg: Path | None = None):
    """Defaults from the JSON files, flags on the command line override them.

    own_cfg holds this script's settings. hardware_cfg (set_wavelength and
    constant_power_scan) is calibration.json: only its shared hardware keys are
    taken. Unknown keys are rejected: a silently ignored typo would run with a
    default value.
    """
    ap.add_argument("--config", metavar="JSON", default=str(own_cfg),
                    help=f"settings file (default: {own_cfg.name} next to this script)")
    if hardware_cfg is not None:
        ap.add_argument("--hardware-config", metavar="JSON", default=str(hardware_cfg),
                        dest="hardware_config",
                        help=f"shared hardware settings (default: {hardware_cfg.name})")
    pre, _ = ap.parse_known_args(argv)

    if hardware_cfg is not None:
        hpath = Path(pre.hardware_config)
        if not hpath.is_file():
            raise SystemExit(f"hardware settings not found: {hpath}")
        data = _read_json(hpath)
        ap.set_defaults(**{k: v for k, v in data.items() if k in HARDWARE_KEYS})
        print(f"hardware: {hpath}")

    path = Path(pre.config)
    if path.is_file():
        data = _read_json(path)
        valid = {x.dest for x in ap._actions} - {"help", "config", "hardware_config"}
        if hardware_cfg is not None:
            shared = sorted(set(data) & HARDWARE_KEYS)
            if shared:
                raise SystemExit(f"{path}: {shared} belong in {Path(hardware_cfg).name} "
                                 f"(shared by all the scripts)")
        unknown = sorted(set(data) - valid)
        if unknown:
            raise SystemExit(f"{path}: unknown key(s) {unknown}.\nvalid keys: {sorted(valid)}")
        ap.set_defaults(**data)
        print(f"config  : {path}")
    elif path != own_cfg:
        raise SystemExit(f"config file not found: {path}")
    else:
        print(f"config  : none found at {path} -- using built-in defaults")
    return ap.parse_args(argv)


def data_root(a) -> Path:
    """Desktop/HB_data (or data_dir); simulated runs go to a subfolder so a
    rehearsal calibration is never picked up by a real measurement."""
    root = Path(a.data_dir).expanduser() if a.data_dir else Path.home() / "Desktop" / "HB_data"
    return root / "simulated" if a.simulate else root


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def new_folder(parent: Path, name: str) -> Path:
    """parent/name, created; _2, _3 ... appended if it already exists."""
    parent.mkdir(parents=True, exist_ok=True)
    path, i = parent / name, 1
    while True:
        try:
            path.mkdir()
            return path
        except FileExistsError:
            i += 1
            path = parent / f"{name}_{i}"


def steps(start, stop, step, what="step"):
    """start, start + step, ... up to stop (inclusive, either direction)."""
    step = abs(float(step))
    if step == 0:
        raise SystemExit(f"{what} must not be 0")
    n = int(math.floor(abs(float(stop) - float(start)) / step + 1e-9)) + 1
    sign = 1.0 if stop >= start else -1.0
    return np.round(float(start) + sign * step * np.arange(n), 6)


def progress(k, n, text="", elapsed_s=None):
    """[#####.....] k/n text  m:ss left -- on one line, rewritten in place."""
    done = int(30 * k / max(n, 1))
    left = ""
    if elapsed_s is not None and 0 < k < n:
        rest = elapsed_s / k * (n - k)
        left = f"  {int(rest // 60)}:{int(rest % 60):02d} left"
    print(f"\r  [{'#' * done}{'.' * (30 - done)}] {k}/{n}  {text}{left}     ", end="", flush=True)
    if k >= n:
        print()


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


class _RigBase:
    """What the real and the simulated setup do the same way."""

    leave_led_on = False

    def frames(self, n):
        """n spectra in a row: (n, pixels)."""
        return np.array([self.snap() for _ in range(max(int(n), 1))])

    def background(self, n=None):
        """Spectrum without the LED light: the LED switched off for a moment
        (dark + room light), or the spectrometer shutter closed (dark only).
        The LED is back on afterwards if it was on -- wait led_settle_s before
        the next spectrum."""
        n = n or self.a.background_frames
        if self.a.background == "shutter":
            self.shutter(False)
            try:
                return self.frames(n).mean(axis=0)
            finally:
                self.shutter(True)
        was_on = self.led_is_on
        if was_on:
            self.led_off()
        try:
            self.sleep(0.2)
            return self.frames(n).mean(axis=0)
        finally:
            if was_on:
                self.led_on()


class Rig(_RigBase):
    """PRMTZ8 on a KDC101 (kcube) + DC40 (thorlabs_led) + CCT10
    (thorlabs_spectrometer) + power meter (thorlabs_powermeter). Only the
    instruments a script needs are opened. Raw counts: the SDK dark and
    amplitude correction are off; the scripts take their own backgrounds."""

    simulated = False

    def __init__(self, a, spectrometer=True, power_meter=False):
        self.a = a
        self.need_spec, self.need_pm = bool(spectrometer), bool(power_meter)
        self.stage = self.led = self.spec = self.pm = None
        self.wl = None
        self.t0 = time.time()
        self.exposure_ms = self.hw_average = None
        self.current_ma = 0.0
        self.led_is_on = False
        self.led_max_ma = None
        self.stage_serial = None

    def __enter__(self):
        try:
            self._open_led()
            self._open_stage()
            if self.need_spec:
                self._open_spec()
            if self.need_pm:
                self._open_pm()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        if self.led is not None:
            try:
                self.led.close(switch_off=not self.leave_led_on)
            except Exception:
                pass
        for dev in (self.stage, self.spec, self.pm):
            if dev is not None:
                try:
                    dev.close()
                except Exception:
                    pass
        return False

    # -- LED (DC40) ---------------------------------------------------------
    def _open_led(self):
        from thorlabs_led import LEDDriver

        self.led = LEDDriver(serial=self.a.led_serial or None)
        if self.led.output:
            self.led.off()                       # start dark, in a known state
        self.led.mode = "CW"
        info = self.led.led
        ident = self.led.identity
        if not info.has_eeprom:
            if self.a.led_max_ma is None:
                raise SystemExit("the LED has no EEPROM (unmounted or non-Thorlabs), so its "
                                 "limit is unknown: set led_max_ma in calibration.json")
            self.led.use_non_thorlabs_led = True
            self.led.current_limit_ma = float(self.a.led_max_ma)
        top = self.led.max_current_ma
        if self.a.led_max_ma is not None:
            top = min(top, float(self.a.led_max_ma))
        self.led_max_ma = top
        self.led_name = str(info)
        print(f"LED driver   : {ident.model} {ident.serial_number}")
        print(f"LED          : {info}; used up to {top:g} mA")

    def set_current(self, ma):
        ma = float(ma)
        if ma > self.led_max_ma + 1e-9:
            raise SystemExit(f"{ma:g} mA is above the LED limit of {self.led_max_ma:g} mA")
        self.led.current_ma = ma
        self.current_ma = ma

    def led_on(self):
        self.led.on()
        self.led_is_on = True

    def led_off(self):
        self.led.off()
        self.led_is_on = False

    def measure_current_ma(self):
        return self.led.measure_current_ma()

    # -- rotation stage (PRMTZ8 on a KDC101) ----------------------------------
    def _open_stage(self):
        import kcube

        serial = self.a.stage_serial
        if not serial:
            devices = kcube.list_devices()
            cubes = [d for d in devices if d["model"] == "KDC101"]
            if len(cubes) != 1:
                found = ", ".join(f"{d['serial']} ({d['model']})" for d in devices) or "nothing"
                raise SystemExit(f"set stage_serial in calibration.json -- found: {found}")
            serial = cubes[0]["serial"]
        name = None if str(self.a.stage_name).lower() in ("", "none", "null") else self.a.stage_name
        self.stage = kcube.KCube(str(serial), stage_name=name)
        self.stage.connect()
        self.stage_serial = str(serial)
        print(f"stage        : {self.stage.model} {serial} ({name or 'controller settings'})")
        self._home()

    def _home(self):
        if self.a.home == "never":
            return
        if self.a.home == "auto":
            try:
                homed = bool(self.stage._device.Status.IsHomed)
            except Exception:
                homed = False
            if homed:
                print("stage        : already homed")
                return
        self.stage.home()

    def move_to(self, deg):
        self.stage.move_to(float(deg) % 360.0)
        self.sleep(self.a.stage_settle_s)

    def angle(self):
        return float(self.stage.get_position())

    # -- spectrometer (CCT10) -----------------------------------------------
    def _open_spec(self):
        from thorlabs_spectrometer import Spectrometer

        self.spec = Spectrometer(self.a.spectrometer_id or None)
        print(f"spectrometer : {self.spec.device_id}")
        self.spec.amplitude_correction = False      # raw counts
        try:
            self.spec.clear_dark()                  # we take our own backgrounds
        except Exception:
            pass
        self.spec.open_shutter()
        self.set_exposure(10.0)

    def set_exposure(self, ms, hw_average=None):
        self.spec.exposure_ms = float(ms)
        self.spec.hw_average = int(hw_average or self.a.hw_average)
        self.exposure_ms = float(self.spec.exposure_ms)
        self.hw_average = int(self.spec.hw_average)

    def snap(self):
        s = self.spec.snap()
        if self.wl is None:
            self.wl = np.asarray(s.wavelengths, float)
        return np.asarray(s.intensities, float)

    def shutter(self, open_):
        self.spec.set_shutter(bool(open_))

    # -- power meter ----------------------------------------------------------
    def _open_pm(self):
        from thorlabs_powermeter import PowerMeter, list_power_meters

        want = getattr(self.a, "pm_serial", None)
        found = list_power_meters()
        hits = [d for d in found if not want or d.serial_number == str(want)
                or str(want) in d.resource_name]
        if not hits:
            listing = "\n".join(f"    {d.serial_number:12s} {d.model:10s} {d.resource_name}"
                                for d in found) or "    (none)"
            raise SystemExit(f"power meter {want or ''} not found. Found:\n{listing}")
        self.pm = PowerMeter(hits[0].resource_name)
        self.pm.power_unit = "W"
        self.pm.average_count = int(getattr(self.a, "pm_average_count", 100))
        self.pm.auto_range = True
        ident = self.pm.identity
        print(f"power meter  : {ident.model} {ident.serial_number}, sensor {self.pm.sensor}")

    def pm_wavelength_range(self):
        return tuple(float(x) for x in self.pm.wavelength_range_nm)

    def pm_wavelength(self, nm):
        """The wavelength the power meter corrects its sensor's response for."""
        self.pm.wavelength_nm = float(nm)

    def read_power_w(self, n=1):
        """(mean, standard deviation) of n power readings, W."""
        v = self.pm.read_powers(max(int(n), 1))
        return float(np.mean(v)), float(np.std(v))

    def zero_pm(self):
        self.pm.zero()

    # -- time -----------------------------------------------------------------
    def sleep(self, s):
        if s and s > 0:
            time.sleep(float(s))

    def now(self):
        return time.time() - self.t0

    def describe(self):
        def ident(dev):
            i = dev.identity
            return f"{i.model} S/N {i.serial_number}, firmware {i.firmware}"

        d = {"backend": "hardware", "stage_serial": self.stage_serial,
             "stage_name": self.a.stage_name, "led": getattr(self, "led_name", None),
             "led_driver": ident(self.led) if self.led else None,
             "led_max_ma": self.led_max_ma}
        if self.spec is not None:
            d.update(spectrometer=self.spec.device_id, exposure_ms=self.exposure_ms,
                     hw_average=self.hw_average)
        if self.pm is not None:
            d.update(power_meter=ident(self.pm), pm_sensor=str(self.pm.sensor))
        return d


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #


class SimRig(_RigBase):
    """Rehearsal stand-in for the whole setup, so every prompt, file and plot
    can be tried without hardware (and the calibration checked against a truth).

    * LED: white (455 nm pump + broad phosphor), limit 1000 mA. Light grows as
      I (1 - 0.12 I / 1000 mA) (droop) and red-shifts 2 nm at full current.
      After every current step the light overshoots by 3 % of the step and
      settles with a 0.6 s time constant (the junction heats); 0.3 % slow
      drift, 0.1 % flicker;
    * element on the stage: an angle-tuned bandpass filter, normal incidence
      at -3 deg on the stage scale. Centre 700 nm x sqrt(1 - (sin(a + 3 deg)/1.75)^2):
      700 nm at 0 deg -> 603 nm at 60 deg; flat-topped, FWHM 12 -> 15 nm;
    * room light: weak lines at 546 and 611 nm (the LED-off background removes them);
    * spectrometer: 2048 px over 200-1000 nm, 2 nm resolution, 16 bit, dark
      ~1000 counts that drifts, read + shot noise, clipping;
    * power meter: silicon photodiode (responsivity ~ wavelength) corrected at
      the wavelength set on it, 0.1 % noise, 50 nW of room light until zeroed;
    * stage: 25 deg/s; homing takes 8 s.
    Time runs on a virtual clock (nothing really waits).
    """

    simulated = True
    LED_LIMIT_MA = 1000.0
    THETA0_DEG, LAMBDA0_NM, N_EFF = -3.0, 700.0, 1.75
    COUNTS_PER_MS = 7.6
    WATT_PER_UNIT = 2.5e-8
    RESOLUTION_NM = 2.0

    def __init__(self, a, spectrometer=True, power_meter=False):
        self.a = a
        self.need_spec, self.need_pm = bool(spectrometer), bool(power_meter)
        self.rng = np.random.default_rng(11)
        self.wl = np.linspace(200.0, 1000.0, 2048)
        wl = self.wl
        self.dl = np.gradient(wl)
        self.led_shape = (np.exp(-0.5 * ((wl - 455) / 9) ** 2)
                          + 0.75 * np.exp(-0.5 * ((wl - 590) / 75) ** 2))
        self.room = 4.0 * (np.exp(-0.5 * ((wl - 546) / 1.0) ** 2)
                           + 0.6 * np.exp(-0.5 * ((wl - 611) / 1.2) ** 2)) + 0.3
        sig = self.RESOLUTION_NM / 2.3548 / (wl[1] - wl[0])
        k = np.arange(-int(5 * sig) - 1, int(5 * sig) + 2)
        self.kernel = np.exp(-0.5 * (k / sig) ** 2)
        self.kernel /= self.kernel.sum()
        self.fixed = self.rng.normal(0, 15, wl.size)
        self.clock = 0.0
        self.theta = 0.0
        self.current_ma = 0.0
        self.led_is_on = False
        self._step_t, self._step_frac = -1e9, 0.0
        self.exposure_ms, self.hw_average = 10.0, int(a.hw_average)
        self.shutter_open = True
        self.pm_wl, self.pm_zero_w = 550.0, 0.0
        top = self.LED_LIMIT_MA
        if a.led_max_ma is not None:
            top = min(top, float(a.led_max_ma))
        self.led_max_ma = top

    def __enter__(self):
        print(f"LED          : simulated white LED, limit {self.LED_LIMIT_MA:g} mA; "
              f"used up to {self.led_max_ma:g} mA")
        print("stage        : simulated (angle-tuned filter, 700 nm at 0 deg -> 603 nm at 60 deg)")
        if self.a.home != "never":
            self.clock += 8.0
        if self.need_spec:
            print("spectrometer : simulated (2048 px, 200-1000 nm)")
        if self.need_pm:
            print("power meter  : simulated (silicon photodiode, 400-1100 nm)")
        return self

    def __exit__(self, *exc):
        if not self.leave_led_on:
            self.led_is_on = False
        return False

    # -- the optics -------------------------------------------------------------
    def centre_nm(self, theta):
        s = np.sin(np.deg2rad(np.asarray(theta, float) - self.THETA0_DEG)) / self.N_EFF
        return self.LAMBDA0_NM * np.sqrt(1 - s * s)

    def _filter(self, theta):
        x = abs(theta - self.THETA0_DEG) / 60.0
        fwhm = 12.0 * (1 + 0.25 * x)
        t = (0.92 - 0.1 * x) * np.exp(-math.log(2) * (2 * (self.wl - self.centre_nm(theta)) / fwhm) ** 4)
        return t + 1e-4

    def _led(self, current, noisy=True):
        """Spectral density of the LED light (arbitrary units)."""
        i = float(current)
        light = i * (1 - 0.12 * i / self.LED_LIMIT_MA)
        shape = np.interp(self.wl - 2.0 * i / self.LED_LIMIT_MA, self.wl, self.led_shape)
        if not noisy:
            return light * shape
        transient = 1 + 0.03 * self._step_frac * math.exp(-(self.clock - self._step_t) / 0.6)
        drift = ((1 + 0.003 * math.sin(2 * math.pi * self.clock / 600))
                 * (1 + 0.001 * self.rng.normal()))
        return light * transient * drift * shape

    def _output(self):
        """Spectral density of the light leaving the setup now."""
        if not self.led_is_on:
            return np.zeros_like(self.wl)
        return self._led(self.current_ma) * self._filter(self.theta)

    def true_centre(self, theta, current, a):
        """Peak centre of the noise-free spectrum (the 'truth' of a calibration)."""
        light = self._led(current, noisy=False) * self._filter(theta)
        y = np.convolve(light, self.kernel, "same")
        p = find_peak(self.wl, y, a.wl_min_nm, a.wl_max_nm, a.peak_method, a.peak_threshold_rel)
        return float("nan") if p is None else p["centre_nm"]

    # -- LED --------------------------------------------------------------------
    def _step(self, delta_ma):
        self._step_frac = float(delta_ma) / self.LED_LIMIT_MA
        self._step_t = self.clock

    def set_current(self, ma):
        ma = float(ma)
        if ma > self.led_max_ma + 1e-9:
            raise SystemExit(f"{ma:g} mA is above the LED limit of {self.led_max_ma:g} mA")
        if self.led_is_on:
            self._step(ma - self.current_ma)
        self.current_ma = ma
        self.clock += 0.01

    def led_on(self):
        if not self.led_is_on:
            self._step(self.current_ma)
        self.led_is_on = True
        self.clock += 0.01

    def led_off(self):
        self.led_is_on = False
        self.clock += 0.01

    def measure_current_ma(self):
        return self.current_ma if self.led_is_on else 0.0

    # -- stage --------------------------------------------------------------------
    def move_to(self, deg):
        self.clock += abs(float(deg) - self.theta) / 25.0 + 0.1
        self.theta = float(deg)
        self.sleep(self.a.stage_settle_s)

    def angle(self):
        return self.theta

    # -- spectrometer -----------------------------------------------------------
    def set_exposure(self, ms, hw_average=None):
        self.exposure_ms = float(np.clip(ms, 0.01, 30000))
        self.hw_average = int(hw_average or self.a.hw_average)

    def shutter(self, open_):
        self.shutter_open = bool(open_)

    def snap(self):
        self.clock += self.exposure_ms * self.hw_average / 1e3 + 0.03
        sig = np.zeros_like(self.wl)
        if self.shutter_open:
            light = self.COUNTS_PER_MS * np.convolve(self._output(), self.kernel, "same")
            sig = self.exposure_ms * (light + self.room)
        dark = (1000 + self.fixed + 0.5 * self.exposure_ms
                + 10 * math.sin(2 * math.pi * self.clock / 1800))
        noise = np.sqrt(12.0 ** 2 + 0.56 * sig) / math.sqrt(max(self.hw_average, 1))
        c = dark + sig + noise * self.rng.normal(0, 1, sig.size)
        return np.clip(np.round(c), 0, self.a.full_scale_counts)

    # -- power meter ----------------------------------------------------------------
    @staticmethod
    def _resp(nm):
        return np.clip(np.asarray(nm, float) / 1000.0, 0.05, None)    # A/W, silicon-like

    def pm_wavelength_range(self):
        return 400.0, 1100.0

    def pm_wavelength(self, nm):
        self.pm_wl = float(nm)
        self.clock += 0.02

    def _pm_reading(self):
        photo = self.WATT_PER_UNIT * float(np.sum(self._output() * self._resp(self.wl) * self.dl))
        room = 5e-8 * float(self._resp(600.0))
        w = (photo + room) / float(self._resp(self.pm_wl)) - self.pm_zero_w
        return w * (1 + 0.001 * self.rng.normal()) + 2e-9 * self.rng.normal()

    def read_power_w(self, n=1):
        vals = []
        for _ in range(max(int(n), 1)):
            self.clock += getattr(self.a, "pm_average_count", 100) * 1e-3 + 0.01
            vals.append(self._pm_reading())
        return float(np.mean(vals)), float(np.std(vals))

    def zero_pm(self):
        self.clock += 3.0
        self.pm_zero_w = 0.0
        self.pm_zero_w = float(np.mean([self._pm_reading() for _ in range(20)]))

    # -- time -----------------------------------------------------------------------
    def sleep(self, s):
        if s and s > 0:
            self.clock += float(s)

    def now(self):
        return self.clock

    def describe(self):
        return {"backend": "simulated", "led_max_ma": self.led_max_ma,
                "exposure_ms": self.exposure_ms, "hw_average": self.hw_average,
                "filter": "700 nm * sqrt(1 - (sin(angle + 3 deg)/1.75)^2)"}


def make_rig(a, spectrometer=True, power_meter=False):
    cls = SimRig if a.simulate else Rig
    return cls(a, spectrometer=spectrometer, power_meter=power_meter)


# --------------------------------------------------------------------------- #
# spectra
# --------------------------------------------------------------------------- #


def clipped(counts, full_scale):
    """Spectra (rows) that reached full scale: >= 98 % of it, or a plateau of
    >= 3 pixels exactly at the maximum while the maximum is in the upper half
    (what a clipped detector gives even if full_scale_counts is set wrong)."""
    c = np.atleast_2d(counts)
    peak = c.max(axis=1)
    at_max = (c == peak[:, None]).sum(axis=1) >= 3
    return (peak >= 0.98 * full_scale) | (at_max & (peak > 0.5 * full_scale))


def interp_background(t, t_before, t_after, bg_before, bg_after):
    """Background at time(s) t, linear between the one before and the one after."""
    t = np.atleast_1d(np.asarray(t, float))
    f = (np.clip((t - t_before) / (t_after - t_before), 0, 1) if t_after > t_before
         else np.full(t.shape, 0.5))
    return (1 - f)[:, None] * bg_before + f[:, None] * bg_after


def _edge(wl, y, k, level, step, stop):
    """Walk from pixel k while y >= level; the interpolated crossing (nm) and
    the last pixel at or above level."""
    j = k
    while j != stop and y[j + step] >= level:
        j += step
    if j == stop:
        return float(wl[j]), j
    f = (y[j] - level) / (y[j] - y[j + step])
    return float(wl[j] + f * (wl[j + step] - wl[j])), j


def find_peak(wl, y, lo, hi, method="centroid", rel=0.5):
    """The peak of a background-subtracted spectrum between lo and hi nm.

    The maximum is located on a 3-pixel running mean (a lone hot pixel is not a
    peak); the pixels around it above rel x maximum give the centre:
      centroid  intensity-weighted mean wavelength (weights: counts - threshold)
      gaussian  parabola through ln(counts), weighted by the counts
      max       the brightest pixel, refined with a parabola through 3 pixels
    Returns None when no pixel is in range, else a dict with centre_nm, fwhm_nm,
    peak (counts, smoothed), lo_nm / hi_nm (the pixels used).
    """
    idx = np.flatnonzero((wl >= lo) & (wl <= hi))
    if idx.size < 3:
        return None
    ys = np.convolve(y, np.ones(3) / 3, mode="same")
    k = int(idx[np.argmax(ys[idx])])
    peak = float(ys[k])
    if peak <= 0:
        return {"centre_nm": float(wl[k]), "fwhm_nm": np.nan, "peak": peak,
                "lo_nm": float(wl[k]), "hi_nm": float(wl[k])}
    left, _ = _edge(wl, ys, k, 0.5 * peak, -1, idx[0])
    right, _ = _edge(wl, ys, k, 0.5 * peak, +1, idx[-1])
    thr = rel * peak
    _, i0 = _edge(wl, ys, k, thr, -1, idx[0])
    _, i1 = _edge(wl, ys, k, thr, +1, idx[-1])
    sl = slice(i0, i1 + 1)
    w, v = wl[sl], y[sl]

    centre = None
    if method == "gaussian" and (v > 0).sum() >= 3:
        m = v > 0
        p = np.polyfit(w[m] - wl[k], np.log(v[m]), 2, w=v[m])
        if p[0] < 0:
            centre = float(wl[k] - p[1] / (2 * p[0]))
    elif method == "max" and 0 < k < len(wl) - 1:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        d = y0 - 2 * y1 + y2
        if d < 0:
            centre = float(wl[k] + 0.5 * (y0 - y2) / d * (wl[k + 1] - wl[k]))
    if centre is None or not (w[0] - 1 <= centre <= w[-1] + 1):
        wt = np.clip(v - thr, 0, None)
        centre = float((wt * w).sum() / wt.sum()) if wt.sum() > 0 else float(wl[k])
    return {"centre_nm": centre, "fwhm_nm": right - left, "peak": peak,
            "lo_nm": float(w[0]), "hi_nm": float(w[-1])}


def band_counts(wl, y, centre, halfwidth):
    """Counts x nm under the spectrum within centre +- halfwidth: the light in
    the band (proportional to its power at a given wavelength)."""
    if not (np.isfinite(centre) and np.isfinite(halfwidth)) or halfwidth <= 0:
        return np.nan
    m = np.abs(wl - centre) <= halfwidth
    return float(np.sum(y[m] * np.gradient(wl)[m]))


def fill_report(raw, a, mask=None):
    """'brightest pixel N counts = X % of full scale (verdict)' and whether it clipped."""
    c = raw if mask is None else raw[..., mask]
    clip = bool(clipped(c, a.full_scale_counts).any())
    fill = float(np.max(c)) / a.full_scale_counts
    verdict = ("CLIPPED" if clip else "near full scale" if fill > 0.85 else
               "dim -- a longer exposure gives less noise" if fill < 0.2 else "good")
    return (f"brightest pixel {np.max(c):.0f} counts = {100 * fill:.0f}% of full scale "
            f"({verdict})"), clip


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _fmt(v):
    if isinstance(v, (str, np.str_)):
        return str(v)
    v = float(v)
    return "" if not np.isfinite(v) else f"{v:.7g}"


def write_csv(path: Path, columns: dict):
    """CSV with a header, one row per entry; NaN written as an empty cell."""
    keys = list(columns)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for row in zip(*(np.asarray(columns[k]) for k in keys)):
            w.writerow([_fmt(v) for v in row])


def read_csv(path: Path) -> dict:
    """{column: array}: numbers as floats (empty = NaN), anything else as text."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header, body = rows[0], rows[1:]
    out = {}
    for j, name in enumerate(header):
        col = [r[j].strip() if j < len(r) else "" for r in body]
        try:
            out[name.strip()] = np.array([float(x) if x else np.nan for x in col])
        except ValueError:
            out[name.strip()] = np.array(col)
    return out


def append_csv(path: Path, row: dict):
    """Add one row to a CSV log, writing the header if the file is new."""
    new = not path.is_file()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(list(row))
        w.writerow([_fmt(v) if not isinstance(v, str) else v for v in row.values()])


# --------------------------------------------------------------------------- #
# calibration files
# --------------------------------------------------------------------------- #

CAL_TABLE = "hb_calibration.csv"      # one row per (angle, LED current)
CAL_MAP = "wavelength_map.csv"        # one row per angle: the wavelength map
CAL_RUN = "calibration_run.json"      # settings, exposure, fit, summary


class Calibration:
    """What calibration.py measured, ready to be used backwards:

      wavelength_at(angle)          the wavelength map (the fit, or straight
                                    lines between the measured centres)
      angle_for(wavelength)         its inverse
      value_at(angle, current)      peak amplitude ("peak", counts) or the light
                                    in the band ("band", counts x nm), linear
                                    between the two nearest calibrated angles
      current_for(angle, value)     its inverse in the current

    Amplitudes are counts at the calibration's exposure (exposure_ms). The LED
    off gives no light, so (0 mA, 0) is part of every curve: below the lowest
    calibrated current the curve is a straight line to zero.
    """

    def __init__(self, path: Path):
        path = Path(path)
        folder = path if path.is_dir() else path.parent
        for name in (CAL_RUN, CAL_TABLE):
            if not (folder / name).is_file():
                raise SystemExit(f"{folder}: {name} is missing -- not a calibration folder")
        self.folder = folder
        self.run = _read_json(folder / CAL_RUN)
        self.settings = self.run.get("settings", {})
        self.exposure_ms = float(self.run["exposure_ms"])
        self.hw_average = int(self.run["hw_average"])
        m = self.run["map"]
        self.angle_lo, self.angle_hi = (float(x) for x in m["angle_range_deg"])
        self.coef = m.get("coef_low_to_high")
        self.map_angles = np.asarray(m["measured_angles_deg"], float)
        self.map_wl = np.asarray(m["measured_nm"], float)

        t = read_csv(folder / CAL_TABLE)
        ok = t["status"] == "ok"
        amp = np.isin(t["status"], ["ok", "dim"])      # dim: amplitude fine, centre noisy
        self.angles = np.unique(t["angle_deg"])
        self.currents = np.unique(t["current_ma"])
        shape = (self.angles.size, self.currents.size)
        self.peak = np.full(shape, np.nan)
        self.band = np.full(shape, np.nan)
        self.centre = np.full(shape, np.nan)
        ia = np.searchsorted(self.angles, t["angle_deg"])
        ic = np.searchsorted(self.currents, t["current_ma"])
        self.peak[ia[amp], ic[amp]] = t["peak_counts"][amp]
        self.band[ia[amp], ic[amp]] = t["band_counts"][amp]
        self.centre[ia[ok], ic[ok]] = t["centre_nm"][ok]

        self._th = np.linspace(self.angle_lo, self.angle_hi, 20001)
        self._lam = self.wavelength_at(self._th)
        d = np.diff(self._lam)
        if not ((d > 0).all() or (d < 0).all()):
            raise SystemExit(f"{folder}: the wavelength is not monotonic in the angle -- "
                             f"recalibrate (narrower angles or a lower fit_order)")
        self.wl_lo, self.wl_hi = float(self._lam.min()), float(self._lam.max())

    # -- wavelength <-> angle -------------------------------------------------
    def wavelength_at(self, deg):
        deg = np.asarray(deg, float)
        if self.coef:
            return np.polynomial.polynomial.polyval(deg, self.coef)
        o = np.argsort(self.map_angles)
        return np.interp(deg, self.map_angles[o], self.map_wl[o])

    def angle_for(self, nm) -> float:
        nm = float(nm)
        if not self.wl_lo - 1e-9 <= nm <= self.wl_hi + 1e-9:
            raise ValueError(f"{nm:g} nm is outside the calibrated "
                             f"{self.wl_lo:.1f}-{self.wl_hi:.1f} nm")
        o = np.argsort(self._lam)
        return float(np.interp(nm, self._lam[o], self._th[o]))

    def nm_per_deg(self, deg) -> float:
        return abs(float(self.wavelength_at(deg + 0.05) - self.wavelength_at(deg - 0.05))) / 0.1

    # -- amplitude <-> current --------------------------------------------------
    def curve_at(self, deg, kind="peak"):
        """(currents mA, values) at an angle, starting with (0, 0); linear
        between the two nearest calibrated angles."""
        grid = self.peak if kind == "peak" else self.band
        deg = float(np.clip(deg, self.angles[0], self.angles[-1]))
        i = int(np.clip(np.searchsorted(self.angles, deg) - 1, 0, max(self.angles.size - 2, 0)))
        if self.angles.size == 1:
            vals = grid[0]
        else:
            f = (deg - self.angles[i]) / (self.angles[i + 1] - self.angles[i])
            vals = (1 - f) * grid[i] + f * grid[i + 1]
        good = np.isfinite(vals)
        if good.sum() == 0:
            raise ValueError(f"no usable calibration points near {deg:.2f} deg")
        cur = np.concatenate([[0.0], self.currents[good]])
        val = np.concatenate([[0.0], vals[good]])
        return cur, val

    def value_at(self, deg, current_ma, kind="peak") -> float:
        cur, val = self.curve_at(deg, kind)
        if current_ma > cur[-1]:
            return float(val[-1] + (val[-1] - val[-2]) / (cur[-1] - cur[-2]) * (current_ma - cur[-1]))
        return float(np.interp(current_ma, cur, val))

    def value_range(self, deg, kind="peak"):
        cur, val = self.curve_at(deg, kind)
        return 0.0, float(np.max(val))

    def current_for(self, deg, value, kind="peak") -> float:
        """The LED current that gives `value` at this angle. The curve is made
        non-decreasing first (droop at the top would make it ambiguous)."""
        cur, val = self.curve_at(deg, kind)
        mono = np.maximum.accumulate(val)
        if value > mono[-1] * (1 + 1e-9):
            raise ValueError(f"{value:.0f} is above the {mono[-1]:.0f} the calibration "
                             f"reaches at {self.wavelength_at(deg):.1f} nm (at {cur[-1]:g} mA)")
        keep = np.concatenate([[True], np.diff(mono) > 0])
        return float(np.interp(value, mono[keep], cur[keep]))

    def check(self, a):
        """Refuse a rehearsal calibration for real hardware (and vice versa);
        warn if the stage is not the one it was made with."""
        s = self.settings
        if "simulate" in s and bool(s["simulate"]) != bool(a.simulate):
            raise SystemExit(f"the calibration was made with simulate = {s['simulate']}, "
                             f"now it is {a.simulate}")
        for key in ("stage_name", "stage_serial"):
            if s.get(key) and getattr(a, key, None) and s[key] != getattr(a, key):
                print(f"  WARNING: the calibration was made with {key} = {s[key]}, now "
                      f"{getattr(a, key)} -- recalibrate if the stage changed")

    def describe(self):
        return (f"{self.folder}\n             {self.wl_lo:.1f}-{self.wl_hi:.1f} nm at angles "
                f"{self.angle_lo:g} to {self.angle_hi:g} deg; currents "
                f"{self.currents[0]:g}-{self.currents[-1]:g} mA; exposure "
                f"{self.exposure_ms:g} ms x {self.hw_average}")


def find_calibration(a, explicit=None) -> Path:
    """calibration_file (a folder or a file in it), or the newest calibration
    in <data_dir>/calibrations/."""
    root = data_root(a) / "calibrations"
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = next((q for q in (Path.cwd() / p, root / p, HERE / p) if q.exists()), p)
        if p.is_file():
            p = p.parent
        if not (p / CAL_RUN).is_file():
            raise SystemExit(f"no calibration found at {p}")
        return p
    found = sorted(q.parent for q in root.glob(f"*/{CAL_RUN}"))
    if not found:
        raise SystemExit(f"no calibration in {root} -- run calibration.py first "
                         f"(or set calibration_file)")
    return found[-1]


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #


def ask_float(prompt, default=None, lo=None, hi=None):
    """Ask until a number in [lo, hi] is typed; Enter takes the default."""
    while True:
        txt = input(f"{prompt}" + (f" [Enter = {default:g}]" if default is not None else "")
                    + ": ").strip()
        if not txt and default is not None:
            return float(default)
        try:
            v = float(txt)
        except ValueError:
            print("  a number, please")
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print(f"  between {lo:.6g} and {hi:.6g}, please")
            continue
        return v


def ask_yes(prompt, default=False):
    """y/n question; Enter takes the default."""
    hint = "[Y/n]" if default else "[y/N]"
    txt = input(f"{prompt} {hint}: ").strip().lower()
    return default if not txt else txt in ("y", "yes")


def ask_exposure(suggested=None):
    """Exposure in ms (the CCT10 takes 0.01-30000)."""
    return ask_float("  exposure (ms)", suggested, 0.01, 30000)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

PLOT_STYLE = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
              "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11}


def get_plt(show):
    """pyplot with larger text, or None (with a note) when matplotlib is missing."""
    try:
        import matplotlib
        if not show:
            try:
                matplotlib.use("Agg")
            except Exception:
                pass
        import matplotlib.pyplot as plt
        plt.rcParams.update(PLOT_STYLE)
        return plt
    except ImportError:
        print("  (matplotlib is not installed -- no figure)")
        return None


def finish_figure(plt, fig, path: Path, show):
    fig.savefig(path, dpi=130)
    print(f"  figure: {path}")
    if show:
        print("  (close the figure window to continue)")
        plt.show()
    plt.close(fig)
