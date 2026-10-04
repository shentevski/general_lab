"""Shared pieces of the DMD wavelength-selection scripts.

    source -> prism -> spectrum on the DMD -> prism (recombine) -> [sample] -> spectrometer

A line of mirrors on the DMD picks a band of the spectrum. dmd_calibration.py
finds which wavelength every line position sends to the spectrometer;
dmd_measure.py uses that to put lines at the wavelengths you ask for.

Here: the hardware (DMD over USB-HID + CCT10), a simulator of the whole setup
for rehearsal (--simulate), the shared settings, the peak finder and the
calibration file.

Line positions are OFFSETS in mirror rows from the chip centre, drawn with the
*_rows patterns of dlpc900_hid (exactly `width` rows at every offset), the same
as in toggle_dmd_patterns.py.
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


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #


def hardware_args(ap):
    """Settings shared by both scripts. They live in dmd_calibration.json only:
    the measurement must use the same DMD geometry as the calibration."""
    g = ap.add_argument_group("shared settings (kept in dmd_calibration.json)")
    g.add_argument("--dmd-width", type=int, default=1920, dest="dmd_width")
    g.add_argument("--dmd-height", type=int, default=1080, dest="dmd_height")
    g.add_argument("--orientation", type=float, default=90.0,
                   help="line orientation in degrees: 0, 90, 45 or -45")
    g.add_argument("--line-on", type=str2bool, default=False, dest="line_on",
                   help="true: lines are ON mirrors on an OFF field; false: OFF mirrors "
                        "on an ON field (as in toggle_dmd_patterns.py)")
    g.add_argument("--dmd-settle-s", type=float, default=0.3, dest="dmd_settle_s",
                   help="wait after a pattern change before a spectrum")
    g.add_argument("--spectrometer-id", default=None, dest="spectrometer_id",
                   help="CCT device ID; null = the first one found")
    g.add_argument("--hw-average", type=int, default=5, dest="hw_average",
                   help="frames the spectrometer averages per spectrum")
    g.add_argument("--full-scale-counts", type=float, default=65535,
                   dest="full_scale_counts", help="raw counts at saturation")
    g.add_argument("--background", choices=("dmd", "shutter"), default="dmd",
                   help="dmd: spectrum with the DMD blocking everything (dark + ambient "
                        "+ DMD scatter); shutter: spectrometer shutter closed (dark only)")
    g.add_argument("--background-frames", type=int, default=3, dest="background_frames",
                   help="spectra averaged per background")
    g.add_argument("--data-dir", default=None, dest="data_dir",
                   help="where data is saved; null = Desktop/DMD_data")
    g.add_argument("--simulate", action="store_true",
                   help="rehearse without hardware (saves under <data_dir>/simulated)")


def _dests(add):
    p = argparse.ArgumentParser(add_help=False)
    add(p)
    return {x.dest for x in p._actions}


HARDWARE_KEYS = _dests(hardware_args)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}")


def parse_config(ap, argv, own_cfg: Path, hardware_cfg: Path | None = None):
    """Defaults from the JSON files, flags on the command line override them.

    own_cfg holds this script's settings. hardware_cfg (the measurement
    script) is dmd_calibration.json: only its shared hardware keys are taken,
    so the DMD geometry is set in one place. Unknown keys are rejected: a
    silently ignored typo would run with a default value.
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
        if hpath.is_file():
            data = _read_json(hpath)
            ap.set_defaults(**{k: v for k, v in data.items() if k in HARDWARE_KEYS})
            print(f"hardware: {hpath}")
        else:
            raise SystemExit(f"hardware settings not found: {hpath}")

    path = Path(pre.config)
    if path.is_file():
        data = _read_json(path)
        valid = {x.dest for x in ap._actions} - {"help", "config", "hardware_config"}
        if hardware_cfg is not None:
            shared = sorted(set(data) & HARDWARE_KEYS)
            if shared:
                raise SystemExit(f"{path}: {shared} belong in {Path(hardware_cfg).name} "
                                 f"(shared with the calibration)")
        unknown = sorted(set(data) - valid)
        if unknown:
            raise SystemExit(f"{path}: unknown key(s) {unknown}.\n"
                             f"valid keys: {sorted(valid)}")
        ap.set_defaults(**data)
        print(f"config  : {path}")
    elif path != own_cfg:
        raise SystemExit(f"config file not found: {path}")
    else:
        print(f"config  : none found at {path} -- using built-in defaults")
    return ap.parse_args(argv)


def data_root(a) -> Path:
    """Desktop/DMD_data (or data_dir); simulated runs go to a subfolder so a
    rehearsal calibration is never picked up by a real measurement."""
    root = Path(a.data_dir).expanduser() if a.data_dir else Path.home() / "Desktop" / "DMD_data"
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


# --------------------------------------------------------------------------- #
# DMD line geometry (the same arithmetic as patterns.MixWavelengths._lines_rows)
# --------------------------------------------------------------------------- #


def row_limits(a):
    """First and last mirror row on the chip, counted from the centre row."""
    ang = np.deg2rad(float(a.orientation))
    s, c = np.sin(ang), np.cos(ang)
    sgn = lambda v: 0 if abs(v) < 1e-6 else (1 if v > 0 else -1)   # noqa: E731
    ss, cc = sgn(s), sgn(c)
    if (ss and cc and abs(abs(s) - abs(c)) > 1e-6) or not (ss or cc):
        raise SystemExit(f"orientation {a.orientation:g}: mirror-row lines need 0, 90, "
                         f"45 or -45 degrees")
    w, h = a.dmd_width - 1, a.dmd_height - 1
    centre = int(round(ss * w / 2 + cc * h / 2))
    lo = min(0, ss * w) + min(0, cc * h)
    hi = max(0, ss * w) + max(0, cc * h)
    return lo - centre, hi - centre


def line_rows(offset, width):
    """First and last row of a line (as _lines_rows draws it)."""
    start = int(round(offset)) - (int(width) - 1) // 2
    return start, start + int(width) - 1


def offset_limits(a, width):
    """Offsets whose whole line is on the chip."""
    lo, hi = row_limits(a)
    h = (int(width) - 1) // 2
    return lo + h, hi + h - (int(width) - 1)


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


class _RigBase:
    """What the real and the simulated setup do the same way."""

    def frames(self, n):
        """n spectra in a row: (n, pixels)."""
        return np.array([self.snap() for _ in range(max(int(n), 1))])

    def background_stack(self):
        """background_frames spectra without the line light: (n, pixels). The
        DMD blocking everything ("dmd") or the spectrometer shutter closed."""
        n = self.a.background_frames
        if self.a.background == "shutter":
            self._shutter(False)
            try:
                return self.frames(n)
            finally:
                self._shutter(True)
        self.show_all(False)
        return self.frames(n)

    def background(self):
        return self.background_stack().mean(axis=0)


class Rig(_RigBase):
    """DMD (dlpc900_hid) + CCT10 (thorlabs_spectrometer). Raw counts: the SDK
    dark and amplitude correction are off, the scripts take their own
    backgrounds. set_sample is a no-op: on the bench YOU move the sample."""

    simulated = False

    def __init__(self, a):
        self.a = a
        self.dmd = self.spec = None
        self.wl = None
        self.t0 = time.time()
        self._encoded = {}

    def __enter__(self):
        try:
            from dlpc900_hid import DMD, patterns
            from thorlabs_spectrometer import Spectrometer

            self.dmd = DMD()
            print(f"DMD          : {self.dmd.get_hardware()[0]}")
            self.mw = patterns.MixWavelengths(self.a.dmd_width, self.a.dmd_height)
            self.dmd.enter_otf_mode()
            self.spec = Spectrometer(self.a.spectrometer_id or None)
            print(f"spectrometer : {self.spec.device_id}")
            self.spec.amplitude_correction = False
            try:
                self.spec.clear_dark()
            except Exception:
                pass
            self.spec.open_shutter()
            self.set_exposure(10.0)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        if self.dmd is not None:
            for step in (self.dmd.stop_pattern, self.dmd.standby, self.dmd.close):
                try:
                    step()
                except Exception:
                    pass
        if self.spec is not None:
            try:
                self.spec.close()
            except Exception:
                pass
        return False

    # -- DMD ---------------------------------------------------------------
    def _image(self, key):
        kind, offsets, width = key
        mw, on, o = self.mw, self.a.line_on, self.a.orientation
        if kind == "all":                      # offsets is True: pass all, False: block all
            return mw.solid_on() if offsets == on else mw.solid_off()
        if len(offsets) == 1:
            return mw.one_line_rows(offset=offsets[0], width=width, on=on, orientation=o)
        make = {2: mw.two_lines_rows, 3: mw.three_lines_rows}[len(offsets)]
        return make(offsets=list(offsets), widths=[width] * len(offsets), on=on,
                    orientation=o)

    def _show(self, key):
        enc = self._encoded.get(key)
        if enc is None:
            enc = self._encoded[key] = self.dmd._encode_image(self._image(key))
        # display_pattern streams what load_patterns cached in _patterns. Setting
        # it directly skips load_patterns' 18-image limit and its OTF mode change
        # on every call (OTF mode was entered once, in __enter__).
        self.dmd._patterns = [enc]
        self.dmd.display_pattern(0)
        time.sleep(self.a.dmd_settle_s)

    def show_lines(self, offsets, width):
        self._show(("lines", tuple(int(round(o)) for o in offsets), int(width)))

    def show_all(self, passing):
        """Every mirror in the line state (passing) or in the field state."""
        self._show(("all", bool(passing), 0))

    # -- spectrometer --------------------------------------------------------
    def set_exposure(self, ms):
        self.spec.exposure_ms = float(ms)
        self.spec.hw_average = int(self.a.hw_average)
        self.exposure_ms = float(self.spec.exposure_ms)
        self.hw_average = int(self.spec.hw_average)

    def snap(self):
        s = self.spec.snap()
        if self.wl is None:
            self.wl = np.asarray(s.wavelengths, float)
        return np.asarray(s.intensities, float)

    def _shutter(self, open_):
        self.spec.set_shutter(open_)

    def now(self):
        return time.time() - self.t0

    def set_sample(self, inside):
        pass

    def describe(self):
        return {"backend": "hardware", "dmd": getattr(self.dmd, "hardware", None),
                "spectrometer": self.spec.device_id, "exposure_ms": self.exposure_ms,
                "hw_average": self.hw_average}


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #

_erf = np.vectorize(math.erf, otypes=[float])


class SimRig(_RigBase):
    """Rehearsal stand-in for the whole setup, so every prompt, file and plot
    can be tried without hardware (and the calibration checked against a truth).

    * source: white-LED-like (blue pump at 455 nm + broad phosphor), drifting 0.3 %
      and flickering 0.1 % from one spectrum to the next;
    * prism: row x on the DMD (from the centre row) carries
      lambda = 575 + 0.13 x + 3e-5 x^2 nm -- non-linear, more dispersion in the blue;
      each wavelength is a spot of 4 rows (sigma) on the DMD;
    * DMD: OFF mirrors send light on to the recombining prism (so line_on false
      is right, as in toggle_dmd_patterns.py; true inverts every pattern);
      0.2 % of the light reaches the spectrometer whatever the pattern (scatter);
    * sample: 85 % transmission with absorption bands at 560 nm and 650 nm;
    * spectrometer: 2048 px over 200-1000 nm, 2 nm resolution, 16 bit, dark
      offset ~1000 counts that drifts, read + shot noise, clipping.
    Time runs on a virtual clock.
    """

    simulated = True
    OFF_MIRRORS_PASS = True
    SPOT_ROWS = 4.0
    STRAY = 2e-3
    COUNTS_PER_MS = 2000.0
    RESOLUTION_NM = 2.0

    def __init__(self, a):
        self.a = a
        self.rng = np.random.default_rng(7)
        self.wl = np.linspace(200.0, 1000.0, 2048)
        wl = self.wl
        self.source = (np.exp(-0.5 * ((wl - 620) / 110) ** 2)
                       + 0.6 * np.exp(-0.5 * ((wl - 455) / 11) ** 2))
        self.sample_T = (0.85 * (1 - 0.7 * np.exp(-0.5 * ((wl - 560) / 12) ** 2))
                         * (1 - 0.4 * np.exp(-0.5 * ((wl - 650) / 25) ** 2)))
        arg = 0.13 ** 2 + 4 * 3e-5 * (wl - 575)
        x = (-0.13 + np.sqrt(np.clip(arg, 0, None))) / (2 * 3e-5)
        self.x = np.where(arg > 0, x, 1e9)           # rows on the DMD, 1e9 = misses it
        sig = self.RESOLUTION_NM / 2.3548 / (wl[1] - wl[0])
        k = np.arange(-int(5 * sig) - 1, int(5 * sig) + 2)
        self.kernel = np.exp(-0.5 * (k / sig) ** 2)
        self.kernel /= self.kernel.sum()
        self.fixed = self.rng.normal(0, 15, wl.size)
        self.T = np.zeros_like(wl)
        self.clock = 0.0
        self.inside = False
        self.exposure_ms, self.hw_average = 10.0, int(a.hw_average)
        self.shutter_open = True

    @staticmethod
    def true_wavelength(rows):
        rows = np.asarray(rows, float)
        return 575 + 0.13 * rows + 3e-5 * rows ** 2

    def true_centre(self, offset, width):
        """Wavelength at the middle row of a line (the 'truth' of a calibration)."""
        first, last = line_rows(offset, width)
        return float(self.true_wavelength(0.5 * (first + last)))

    def __enter__(self):
        print("DMD          : simulated")
        print("spectrometer : simulated (2048 px, 200-1000 nm)")
        return self

    def __exit__(self, *exc):
        return False

    def _band(self, first, last):
        s = self.SPOT_ROWS * math.sqrt(2)
        return 0.5 * (_erf((last + 0.5 - self.x) / s) - _erf((first - 0.5 - self.x) / s))

    def _line_mirrors_pass(self):
        return self.a.line_on != self.OFF_MIRRORS_PASS

    def show_lines(self, offsets, width):
        self.clock += 0.3
        T = np.clip(sum(self._band(*line_rows(o, width)) for o in offsets), 0, 1)
        self.T = T if self._line_mirrors_pass() else self._band(*row_limits(self.a)) - T

    def show_all(self, passing):
        self.clock += 0.3
        light = passing == self._line_mirrors_pass()
        self.T = self._band(*row_limits(self.a)) if light else np.zeros_like(self.wl)

    def set_exposure(self, ms):
        self.exposure_ms = float(np.clip(ms, 0.01, 30000))
        self.hw_average = int(self.a.hw_average)

    def set_sample(self, inside):
        self.clock += 10.0
        self.inside = bool(inside)

    def now(self):
        return self.clock

    def snap(self):
        self.clock += self.exposure_ms * self.hw_average / 1e3 + 0.03
        drift = ((1 + 0.003 * math.sin(2 * math.pi * self.clock / 600))
                 * (1 + 0.001 * self.rng.normal()))
        light = drift * self.source * (self.T + self.STRAY)
        if self.inside:
            light = light * self.sample_T
        if not self.shutter_open:
            light = 0 * light
        sig = self.COUNTS_PER_MS * self.exposure_ms * np.convolve(light, self.kernel, "same")
        dark = (1000 + self.fixed + 0.5 * self.exposure_ms
                + 10 * math.sin(2 * math.pi * self.clock / 1800))
        noise = np.sqrt(12.0 ** 2 + 0.56 * sig) / math.sqrt(max(self.hw_average, 1))
        c = dark + sig + noise * self.rng.normal(0, 1, sig.size)
        return np.clip(np.round(c), 0, self.a.full_scale_counts)

    def _shutter(self, open_):
        self.shutter_open = bool(open_)

    def describe(self):
        return {"backend": "simulated", "exposure_ms": self.exposure_ms,
                "hw_average": self.hw_average,
                "prism": "lambda = 575 + 0.13 x + 3e-5 x^2 (x in rows)"}


def make_rig(a):
    return SimRig(a) if a.simulate else Rig(a)


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


# --------------------------------------------------------------------------- #
# calibration file: line offset (mirror rows) <-> wavelength (nm)
# --------------------------------------------------------------------------- #

CAL_NAME = "dmd_calibration.csv"
CAL_RUN = "calibration_run.json"


class Calibration:
    """dmd_calibration.csv: the rows with a wavelength_nm (the fit, within the
    measured range) map offset_rows <-> wavelength, linear in between."""

    def __init__(self, path: Path):
        d = read_csv(path)
        if "offset_rows" not in d or "wavelength_nm" not in d:
            raise SystemExit(f"{path}: needs the columns offset_rows and wavelength_nm")
        ok = np.isfinite(d["offset_rows"]) & np.isfinite(d["wavelength_nm"])
        off, wl = d["offset_rows"][ok], d["wavelength_nm"][ok]
        if off.size < 2:
            raise SystemExit(f"{path}: fewer than 2 calibrated positions")
        o = np.argsort(off)
        self.off, self.wl = off[o], wl[o]
        dw = np.diff(self.wl)
        if not ((dw > 0).all() or (dw < 0).all()):
            raise SystemExit(f"{path}: wavelength is not monotonic in the offset -- "
                             f"redo the calibration or lower fit_order")
        self.path = Path(path)
        self.lo, self.hi = float(self.wl.min()), float(self.wl.max())
        run = self.path.parent / CAL_RUN
        self.run = _read_json(run) if run.is_file() else None

    def offset_for(self, wl_nm) -> int:
        """The mirror row (integer) closest to wl_nm."""
        s = np.argsort(self.wl)
        return int(round(float(np.interp(wl_nm, self.wl[s], self.off[s]))))

    def wl_at(self, offset) -> float:
        return float(np.interp(offset, self.off, self.wl))

    def nm_per_row(self, wl_nm) -> float:
        o = self.offset_for(wl_nm)
        return abs(self.wl_at(o + 0.5) - self.wl_at(o - 0.5))

    def check(self, a):
        """Refuse a calibration made with another DMD geometry: its offsets would
        put the lines at other wavelengths."""
        if self.run is None:
            print(f"  (no {CAL_RUN} next to the calibration: geometry not checked)")
            return
        s = self.run.get("settings", {})
        for key in ("orientation", "line_on", "dmd_width", "dmd_height", "simulate"):
            if key in s and s[key] != getattr(a, key):
                raise SystemExit(f"the calibration was made with {key} = {s[key]}, now it "
                                 f"is {getattr(a, key)} -- recalibrate, or set it back")

    def describe(self):
        return (f"{self.path}\n             {self.lo:.1f}-{self.hi:.1f} nm, offsets "
                f"{self.off[0]:.0f} to {self.off[-1]:.0f} rows")


def find_calibration(a, explicit=None) -> Path:
    """calibration_file, or the newest one in <data_dir>/calibrations/."""
    if explicit:
        p = Path(explicit).expanduser()
        p = p if p.is_absolute() else HERE / p
        if p.is_dir():
            p = p / CAL_NAME
        if not p.is_file():
            raise SystemExit(f"calibration file not found: {p}")
        return p
    root = data_root(a) / "calibrations"
    found = sorted(root.glob(f"*/{CAL_NAME}"))
    if not found:
        raise SystemExit(f"no calibration in {root} -- run dmd_calibration.py first "
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
            print(f"  between {round(lo, 2):g} and {round(hi, 2):g}, please")
            continue
        return v


def ask_int(prompt, default=None, lo=None, hi=None):
    while True:
        v = ask_float(prompt, default, lo, hi)
        if v == int(v):
            return int(v)
        print("  a whole number, please")


def ask_choice(prompt, choices):
    while True:
        txt = input(prompt).strip().lower()
        if txt in choices:
            return txt
        print(f"  one of: {', '.join(choices)}")


def ask_exposure(suggested=None):
    """Exposure in ms (the CCT10 takes 0.01-30000)."""
    return ask_float("  exposure (ms)", suggested, 0.01, 30000)


def fill_report(raw, a, mask=None):
    """'peak N counts = X % of full scale (verdict)' and whether it clipped."""
    c = raw if mask is None else raw[..., mask]
    clip = bool(clipped(c, a.full_scale_counts).any())
    fill = float(np.max(c)) / a.full_scale_counts
    verdict = ("CLIPPED" if clip else "near full scale" if fill > 0.85 else
               "dim -- a longer exposure gives less noise" if fill < 0.2 else "good")
    return (f"brightest pixel {np.max(c):.0f} counts = {100 * fill:.0f}% of full scale "
            f"({verdict})"), clip


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
