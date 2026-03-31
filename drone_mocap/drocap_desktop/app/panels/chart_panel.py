"""
ChartPanel — bottom-left quadrant.

Plots Hip / Knee / Ankle angle time-series on a white pyqtgraph canvas.

Live mode (during analysis Pass 1)
    push_live_frame() appends pre-computed angles from the worker and calls
    setData() on dotted "preview" curves.  Y-axis is pinned to (-30, 180) so
    data is always visible regardless of pyqtgraph's autoRange state.

Post-analysis (after analysis_complete)
    load_angles() reads the final CSV and sets solid bold curves, then calls
    add_gait_regions() with the GaitCycles object from biomechanics.py.

Cursor
    Dashed Jira-navy InfiniteLine (#0747A6) on the knee plot, x-linked to
    hip and ankle via mirror lines.  Dragging emits cursor_moved(float s);
    set_cursor(float s) moves it without re-emitting.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.theme import PG_COLORS

_CURSOR_COLOR = "#0747A6"
_STRIDE_ODD   = (7,  82, 204,  25)   # RGBA faint Jira-blue
_STRIDE_EVEN  = (94, 108, 132, 18)   # RGBA faint slate

# Anatomically-safe default y-range shown before any data arrives.
# Angles outside this range still auto-expand the view via autoRange().
_Y_MIN, _Y_MAX = -30.0, 180.0

_VIS_IDX = {"hip": (23, 24), "knee": (25, 26), "ankle": (27, 28)}


class ChartPanel(QWidget):
    cursor_moved  = pyqtSignal(float)
    stride_jumped = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("KINEMATIC WAVEFORMS")
        title.setObjectName("section_title")
        self._live_badge = QLabel("  LIVE  ")
        self._live_badge.setObjectName("section_title")
        self._live_badge.setStyleSheet(
            "background:#36B37E; color:#FFFFFF; border-radius:3px;"
            "padding:1px 6px; font-size:10px; font-weight:700;"
        )
        self._live_badge.setVisible(False)
        self._stride_label = QLabel("")
        self._stride_label.setStyleSheet("color:#5E6C84; font-size:10px;")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self._stride_label)
        hdr.addWidget(self._live_badge)
        layout.addLayout(hdr)

        # ── pyqtgraph ─────────────────────────────────────────────────────────
        pg.setConfigOptions(antialias=True, foreground=PG_COLORS["axis_text"])
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(PG_COLORS["bg"])
        layout.addWidget(self._gw, stretch=1)

        self._plots:       dict[str, pg.PlotItem]     = {}
        self._curves:      dict[str, pg.PlotDataItem] = {}
        self._live_curves: dict[str, pg.PlotDataItem] = {}
        self._cursor:      pg.InfiniteLine | None = None
        self._gait_items:  list = []

        # Live data buffers
        self._live_t:    deque[float]            = deque(maxlen=4000)
        self._live_vals: dict[str, deque[float]] = {
            j: deque(maxlen=4000) for j in ("hip", "knee", "ankle")
        }
        self._live_side: str   = "right"
        self._live_fps:  float = 30.0

        self._build_plots()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_plots(self) -> None:
        joints_cfg = [
            ("hip",   "Hip  (°)"),
            ("knee",  "Knee  (°)"),
            ("ankle", "Ankle  (°)"),
        ]
        prev = None
        for row, (joint, ylabel) in enumerate(joints_cfg):
            p = self._gw.addPlot(row=row, col=0)
            p.setMenuEnabled(False)
            p.setLabel("left", ylabel, color=PG_COLORS[joint], size="10pt")
            p.showGrid(x=True, y=True, alpha=0.35)
            p.getViewBox().setBackgroundColor(PG_COLORS["bg"])

            for ax_name in ("bottom", "left", "top", "right"):
                ax = p.getAxis(ax_name)
                ax.setPen(pg.mkPen(PG_COLORS["grid"], width=1))
                ax.setTextPen(pg.mkPen(PG_COLORS["axis_text"]))

            # Pin Y to anatomical range — prevents the "collapsed to [0,0]" trap.
            # setClipToView is safe here because the range is always defined.
            p.setYRange(_Y_MIN, _Y_MAX, padding=0.05)

            if prev is not None:
                p.setXLink(prev)
            prev = p

            # Final smoothed curve (bold solid)
            curve = p.plot(pen=pg.mkPen(PG_COLORS[joint], width=2.2), name=joint)
            curve.setDownsampling(auto=True)

            # Live preview curve (thin dotted) — no setClipToView because the
            # X-range grows as frames arrive and clipping would hide all data
            live = p.plot(
                pen=pg.mkPen(PG_COLORS[joint], width=1.0,
                             style=Qt.PenStyle.DotLine),
                name=f"{joint}_live",
            )
            live.setDownsampling(auto=True)

            self._plots[joint]       = p
            self._curves[joint]      = curve
            self._live_curves[joint] = live

        # Cursor on knee plot (x-linked to all via setXLink)
        knee_plot = self._plots["knee"]
        self._cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen(_CURSOR_COLOR, width=1.5,
                         style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(_CURSOR_COLOR, width=2.5),
        )
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        knee_plot.addItem(self._cursor)

        for joint in ("hip", "ankle"):
            m = pg.InfiniteLine(
                pos=0.0, angle=90, movable=False,
                pen=pg.mkPen(_CURSOR_COLOR, width=1.5,
                             style=Qt.PenStyle.DashLine),
            )
            self._plots[joint].addItem(m)
            setattr(self, f"_mirror_{joint}", m)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_live_mode(self, active: bool, side: str = "right", fps: float = 30.0) -> None:
        self._live_badge.setVisible(active)
        self._live_side = side
        self._live_fps  = fps
        if active:
            self._live_t.clear()
            for b in self._live_vals.values():
                b.clear()
            # Reset X-range to show from t=0; Y stays pinned to (_Y_MIN, _Y_MAX)
            for p in self._plots.values():
                p.setXRange(0, 5, padding=0)          # show first 5 s; auto-expands
                p.setYRange(_Y_MIN, _Y_MAX, padding=0.05)

    def push_live_frame(
        self,
        frame_idx: int,
        total_frames: int,
        vis: np.ndarray,
        angles: object,    # dict[str, float] from worker, or {}
    ) -> None:
        """
        Called on every throttled progress signal.
        angles are pre-computed in the worker thread (not GUI thread).
        """
        t = frame_idx / self._live_fps
        self._live_t.append(t)

        if angles:
            ang = angles  # type: ignore[assignment]
            self._live_vals["hip"].append(ang.get("hip", np.nan))
            self._live_vals["knee"].append(ang.get("knee", np.nan))
            self._live_vals["ankle"].append(ang.get("ankle", np.nan))

            # Debug: confirm data is flowing
            print(f"UI received angles: hip={ang.get('hip'):.1f}  "
                  f"knee={ang.get('knee'):.1f}  ankle={ang.get('ankle'):.1f}  t={t:.2f}s")
        else:
            self._push_vis_proxy(vis)

        t_arr = np.fromiter(self._live_t, dtype=float)
        for joint, buf in self._live_vals.items():
            v_arr = np.fromiter(buf, dtype=float)
            n = min(len(t_arr), len(v_arr))
            if n >= 1:
                self._live_curves[joint].setData(t_arr[:n], v_arr[:n])

        # Auto-expand X-axis as time advances (Y stays pinned)
        if len(t_arr) > 1:
            self._plots["knee"].setXRange(
                float(t_arr[0]), float(t_arr[-1]) + 0.5, padding=0
            )

    def _push_vis_proxy(self, vis: np.ndarray) -> None:
        for joint, (i, j) in _VIS_IDX.items():
            avg = float((vis[i] + vis[j]) / 2) if len(vis) > max(i, j) else 0.0
            self._live_vals[joint].append(avg * 100.0 - 20.0)

    def clear(self) -> None:
        for c in list(self._curves.values()) + list(self._live_curves.values()):
            c.setData([], [])
        self._remove_gait_items()
        self._live_badge.setVisible(False)
        self._stride_label.setText("")
        # Reset Y-range to default so incoming live data is visible
        for p in self._plots.values():
            p.setYRange(_Y_MIN, _Y_MAX, padding=0.05)

    def load_angles(self, csv_path: str) -> None:
        """Load final smoothed angles from CSV; replace live dotted curves."""
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return
        if "time_s" not in df.columns:
            return

        t = df["time_s"].to_numpy(dtype=float)

        col_map: dict[str, str] = {}
        for col in df.columns:
            lc = col.lower()
            if   "hip"   in lc and "hip"   not in col_map: col_map["hip"]   = col
            elif "knee"  in lc and "knee"  not in col_map: col_map["knee"]  = col
            elif "ankle" in lc and "ankle" not in col_map: col_map["ankle"] = col

        for joint, col in col_map.items():
            vals = df[col].to_numpy(dtype=float)
            if joint in self._curves:
                self._curves[joint].setData(t, vals)
            if joint in self._live_curves:
                self._live_curves[joint].setData([], [])

        # Refit X-range to data; Y re-pins to anatomical range with padding
        for p in self._plots.values():
            p.setYRange(_Y_MIN, _Y_MAX, padding=0.05)
        if len(t):
            self._plots["knee"].setXRange(float(t[0]), float(t[-1]), padding=0.02)

        if self._cursor is not None and len(t):
            self._cursor.setValue(float(t[0]))

        self._live_badge.setVisible(False)

    def add_gait_regions(self, cycles) -> None:
        """
        Draw stride bands and event lines from a GaitCycles object.

        Parameters
        ----------
        cycles : app.utils.biomechanics.GaitCycles
        """
        self._remove_gait_items()
        hs = cycles.heel_strike_times
        sw = cycles.swing_peak_times

        if len(hs) == 0:
            return

        # ── Stride bands ─────────────────────────────────────────────────────
        boundaries = list(hs)
        if len(sw) and sw[-1] > boundaries[-1]:
            boundaries.append(float(sw[-1]))
        elif len(boundaries) > 1:
            boundaries.append(boundaries[-1] + cycles.mean_stride_s)

        for idx in range(len(boundaries) - 1):
            t_start = float(boundaries[idx])
            t_end   = float(boundaries[idx + 1])
            color   = _STRIDE_ODD if idx % 2 == 0 else _STRIDE_EVEN
            for plot in self._plots.values():
                region = _ClickableRegion(t_start, t_end, color,
                                          self._on_stride_clicked)
                plot.addItem(region)
                self._gait_items.append(region)

        # ── Heel-strike lines ─────────────────────────────────────────────────
        for t_hs in hs:
            for plot in self._plots.values():
                line = pg.InfiniteLine(
                    pos=float(t_hs), angle=90, movable=False,
                    pen=pg.mkPen(PG_COLORS["gait_hs"], width=1.0,
                                 style=Qt.PenStyle.DotLine),
                    label="HS", labelOpts={"position": 0.92, "color": PG_COLORS["gait_hs"],
                                           "fill": (255, 255, 255, 120)},
                )
                plot.addItem(line)
                self._gait_items.append(line)

        # ── Swing-peak lines ──────────────────────────────────────────────────
        for t_sw in sw:
            for plot in self._plots.values():
                line = pg.InfiniteLine(
                    pos=float(t_sw), angle=90, movable=False,
                    pen=pg.mkPen(PG_COLORS["gait_swing"], width=0.8,
                                 style=Qt.PenStyle.DotLine),
                )
                plot.addItem(line)
                self._gait_items.append(line)

        # ── Stride statistics label ───────────────────────────────────────────
        n = cycles.n_strides
        if n > 0:
            cadence = cycles.cadence_steps_per_min
            mean_s  = cycles.mean_stride_s
            self._stride_label.setText(
                f"{n} stride{'s' if n != 1 else ''}  ·  "
                f"{mean_s:.2f} s/stride  ·  "
                f"{cadence:.0f} steps/min"
            )
        else:
            self._stride_label.setText(f"{len(hs)} heel-strike detected")

    def set_cursor(self, time_s: float) -> None:
        if self._cursor is None:
            return
        self._cursor.blockSignals(True)
        self._cursor.setValue(time_s)
        self._cursor.blockSignals(False)
        self._sync_mirrors(time_s)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _on_cursor_moved(self, line: pg.InfiniteLine) -> None:
        t = float(line.value())
        self._sync_mirrors(t)
        self.cursor_moved.emit(t)

    def _sync_mirrors(self, t: float) -> None:
        for joint in ("hip", "ankle"):
            m = getattr(self, f"_mirror_{joint}", None)
            if m is not None:
                m.setValue(t)

    def _remove_gait_items(self) -> None:
        for item in self._gait_items:
            for plot in self._plots.values():
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
        self._gait_items.clear()

    def _on_stride_clicked(self, t_start: float) -> None:
        self.set_cursor(t_start)
        self.stride_jumped.emit(t_start)


class _ClickableRegion(pg.LinearRegionItem):
    def __init__(self, t_start, t_end, color, callback):
        super().__init__(
            values=(t_start, t_end),
            movable=False,
            brush=pg.mkBrush(*color),
            pen=pg.mkPen(None),
        )
        self._t_start = t_start
        self._cb = callback
        self.setZValue(-10)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._cb(self._t_start)
            ev.accept()
        else:
            super().mouseClickEvent(ev)
