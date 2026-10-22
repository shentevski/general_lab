"""Live window for measure_mueller.py: watch the measurement happen.

    sweep        power vs QWP angle, point by point, with the Fourier fit
                 growing as the points come in
    ellipse      the state being measured (from the fit so far) on top of
                 the target the PSG was asked for
    Poincare     every finished sweep: empty (s), substrate (o), film (^);
                 a line from substrate to film shows what the film does to
                 each state
    Mueller      a live estimate of the film (film vs substrate) as soon as
                 four states pin it down, with its condition number
    status       which position the stage holds (empty / substrate / film),
                 the PSG angles, the laser monitor, elapsed time and time left

It never decides anything: closing the window does not stop the run. The
final picture is saved in the run folder (live_view.png).
"""

from __future__ import annotations

import time

import numpy as np

from mueller_common import stokes_from_coeffs
from polarization_toolkit.analysis.extract import design_matrix

MARKERS = {"in": "o", "out": "^", "in0": "s"}
NAMES = {"in": "substrate", "out": "film on substrate", "in0": "empty"}


def ellipse_xy(s):
    """Polarization ellipse of the unit direction s = (S1, S2, S3)."""
    s = np.asarray(s, float)
    s = s / max(np.linalg.norm(s), 1e-12)
    az = 0.5 * np.arctan2(s[1], s[0])
    ell = 0.5 * np.arcsin(np.clip(s[2], -1, 1))
    t = np.linspace(0, 2 * np.pi, 181)
    a, b = np.cos(ell), np.sin(ell)
    return (a * np.cos(t) * np.cos(az) - b * np.sin(t) * np.sin(az),
            a * np.cos(t) * np.sin(az) + b * np.sin(t) * np.cos(az))


class LiveView:
    REDRAW_S = 0.3                      # at most ~3 redraws per second

    def __init__(self, a, n_states, sweeps_per_state, enabled=True):
        self.a = a
        self.enabled = enabled
        self.total = n_states * sweeps_per_state
        self.done = 0
        self.t0 = time.time()
        self.sweep_times = []
        self.state_label, self.side, self.view, self.target, self.psg = "", "in", "", None, None
        self.ref0 = None
        self.points = []                # finished sweeps: (state index, side, S)
        self.pairs = {}                 # state index -> {"in": S, "out": S}
        self.last_draw = 0.0
        self.closed = False
        self.stage = bool(getattr(a, "use_stage", False))
        if not enabled:
            return
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        self.interactive = matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps")
        if self.interactive:
            plt.ion()
        self.fig = plt.figure(figsize=(14, 8.2))
        self.fig.canvas.manager.set_window_title("measure_mueller -- live") \
            if hasattr(self.fig.canvas, "manager") and self.fig.canvas.manager else None
        g = self.fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.35, wspace=0.3)
        self.ax_sweep = self.fig.add_subplot(g[0, 0:2])
        self.ax_ell = self.fig.add_subplot(g[0, 2])
        self.ax_sph = self.fig.add_subplot(g[1, 0], projection="3d")
        self.ax_m = self.fig.add_subplot(g[1, 1])
        self.ax_st = self.fig.add_subplot(g[1, 2])
        self._draw_sphere()
        self._draw_mueller(None, None)
        self._draw_status()
        if self.interactive:
            plt.show(block=False)
            self._show()

    # ------------------------------------------------------------------ events

    def start_sweep(self, state_label, side, view="", target=None, psg=None):
        self.state_label, self.side, self.view = state_label, side, view
        self.target, self.psg = target, psg
        self.sweep_t0 = time.time()

    def point(self, k, sw):
        """Called by the sweep after every point."""
        if not self.enabled or self.closed:
            return
        if not self.plt.fignum_exists(self.fig.number):
            self.closed = True
            return
        now = time.time()
        n = len(sw["power_W"])
        if now - self.last_draw < self.REDRAW_S and k < n - 1:
            return
        self.last_draw = now
        self._draw_sweep(k, sw)
        self._draw_status(k + 1, n, sw)
        self._show()

    def end_sweep(self, state_index, side, S):
        self.done += 1
        self.sweep_times.append(time.time() - self.sweep_t0)
        self.points.append((state_index, side, np.asarray(S, float)))
        if side in ("in", "out"):
            self.pairs.setdefault(state_index, {})[side] = np.asarray(S, float)
        if not self.enabled or self.closed:
            return
        self._draw_sphere()
        S_in = [p["in"] for p in self.pairs.values() if "in" in p and "out" in p]
        S_out = [p["out"] for p in self.pairs.values() if "in" in p and "out" in p]
        self._draw_mueller(S_in, S_out)
        self._draw_status()
        self._show()

    def finish(self, path=None):
        if not self.enabled:
            return
        if path is not None and not self.closed:
            try:
                self.fig.savefig(path, dpi=110)
            except Exception:
                pass
        if self.interactive and not self.closed:
            self.plt.ioff()

    # ------------------------------------------------------------------ panels

    def _fit(self, ang, y):
        th = np.deg2rad(np.asarray(ang) - self.a.qwp_zero_deg)
        A = design_matrix(th)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return c

    def _draw_sweep(self, k, sw):
        ax = self.ax_sweep
        ax.clear()
        ang = np.asarray(sw["measured_deg"][:k + 1]) % 360
        y = np.asarray(sw["power_W"][:k + 1]) * 1e6
        ax.plot(ang, y, "o", ms=3, color="C3" if self.side in ("out",) else "C0")
        S = None
        if k >= 11:                                        # enough for a 5-term fit
            c = self._fit(sw["measured_deg"][:k + 1], y)
            t = np.linspace(0, 360, 361)
            ax.plot(t, design_matrix(np.deg2rad(t - self.a.qwp_zero_deg)) @ c, "-",
                    color="0.3", lw=1)
            S = stokes_from_coeffs(c, 360 * self.a.qwp_retardance_waves, self.a.s3_sign)
        ax.set_xlim(0, 360)
        ax.set_xlabel("polarimeter QWP angle (deg)")
        ax.set_ylabel("signal (µW)")
        what = NAMES.get(self.side, self.side)
        txt = f"state {self.state_label} -- {what}"
        if S is not None and S[0] > 0:
            n = S / S[0]
            txt += (f"    S/S0 so far [{n[1]:+.3f} {n[2]:+.3f} {n[3]:+.3f}]  DOP "
                    f"{np.linalg.norm(n[1:]):.3f}")
        ax.set_title(txt, fontsize=10)
        self._draw_ellipse(S)

    def _draw_ellipse(self, S):
        ax = self.ax_ell
        ax.clear()
        ax.add_patch(self.plt.Circle((0, 0), 1, fill=False, color="0.88"))
        if self.target is not None:
            x, y = ellipse_xy(self.target)
            ax.plot(x, y, "--", color="0.5", lw=1.4, label="target")
        if S is not None and S[0] > 0 and np.linalg.norm(S[1:]) > 0:
            dop = min(np.linalg.norm(S[1:]) / S[0], 1.0)
            x, y = ellipse_xy(S[1:])
            ax.plot(dop * x, dop * y, color="C3", lw=2, label=f"measured (DOP {dop:.3f})")
            k = int(np.argmax(y[:-6]))
            if abs(S[3] / S[0]) > 0.02:
                ax.annotate("", xy=(dop * x[k + 5], dop * y[k + 5]),
                            xytext=(dop * x[k], dop * y[k]),
                            arrowprops=dict(arrowstyle="-|>", color="C3"))
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("H"); ax.set_ylabel("V")
        ax.set_title("polarization ellipse", fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="lower right")

    def _draw_sphere(self):
        ax = self.ax_sph
        ax.clear()
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
        ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                          color="0.88", lw=0.4)
        for i, lab in enumerate(("S1", "S2", "S3")):
            e = np.zeros(3); e[i] = 1.25
            ax.plot(*np.c_[-e, e], color="0.7", lw=0.6)
            ax.text(*(e * 1.05), lab, color="0.45", fontsize=8)
        colors = self.plt.cm.tab10
        for idx, side, S in self.points:
            p = S[1:] / S[0]
            ax.scatter(*p, color=colors((idx - 1) % 10), marker=MARKERS.get(side, "o"), s=30,
                       edgecolor="k" if side == "out" else None, lw=0.4)
        for idx, pr in self.pairs.items():
            if "in" in pr and "out" in pr:
                a_, b_ = pr["in"][1:] / pr["in"][0], pr["out"][1:] / pr["out"][0]
                ax.plot(*np.c_[a_, b_], "--", color=colors((idx - 1) % 10), lw=1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
        ax.set_axis_off()
        ax.set_title("Poincaré sphere: s empty, o substrate, ^ film", fontsize=9)

    def _draw_mueller(self, S_in, S_out):
        ax = self.ax_m
        ax.clear()
        title = "live film Mueller matrix (film vs substrate)"
        m = np.full((4, 4), np.nan)
        if S_in and len(S_in) >= 4:
            Si, So = np.column_stack(S_in), np.column_stack(S_out)
            sv = np.linalg.svd(Si / Si[0], compute_uv=False)
            if sv[-1] > 1e-2 * sv[0]:
                M = So @ np.linalg.pinv(Si)
                m = M / M[0, 0]
                title += f"\n{len(S_in)} states, cond {sv[0] / sv[-1]:.2f}, M00 {M[0, 0]:.3f}"
            else:
                title += f"\n{len(S_in)} states -- not enough yet (add a circular one)"
        else:
            title += f"\nwaiting for 4 states ({len(S_in or [])} so far)"
        ax.imshow(np.nan_to_num(m), cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(4):
            for j in range(4):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(m[i, j]) > 0.6 else "black")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_title(title, fontsize=9)

    def _draw_status(self, k=None, n=None, sw=None):
        ax = self.ax_st
        ax.clear()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        # the sample holder: the position in the beam is lit
        if self.stage:
            for x0, key, name in ((0.04, "E", "empty"), (0.36, "S", "substrate"),
                                  (0.68, "F", "film")):
                lit = self.view == key
                ax.add_patch(self.plt.Rectangle((x0, 0.62), 0.28, 0.3,
                                                color="#ffd84d" if lit else "0.85"))
                ax.text(x0 + 0.14, 0.77, f"{name}\n{'IN BEAM' if lit else ''}",
                        ha="center", va="center", fontsize=9)
        lines = [f"sweeps {self.done} of {self.total}"
                 + (f"   (point {k} of {n})" if k else "")]
        elapsed = time.time() - self.t0
        if self.sweep_times:
            left = np.median(self.sweep_times) * (self.total - self.done)
            lines.append(f"elapsed {elapsed / 60:.1f} min, about {left / 60:.1f} min left")
        else:
            lines.append(f"elapsed {elapsed / 60:.1f} min")
        if self.psg is not None:
            lines.append(f"PSG: HWP {self.psg.hwp_deg:.2f}°, QWP {self.psg.qwp_deg:.2f}°")
        if sw is not None and k:
            r = np.asarray(sw["ref_W"][:k])
            if np.isfinite(r).all() and r.size:
                if self.ref0 is None:
                    self.ref0 = r[0]
                lines.append(f"laser monitor {r[-1] * 1e6:.3f} µW "
                             f"({100 * (r[-1] / self.ref0 - 1):+.2f}% since start)")
        ax.text(0.02, 0.5, "\n".join(lines), va="top", fontsize=9, family="monospace")

    def _show(self):
        try:
            if self.interactive:                     # no plt.pause: it raises the window
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            else:
                self.fig.canvas.draw()
        except Exception:
            self.closed = True
