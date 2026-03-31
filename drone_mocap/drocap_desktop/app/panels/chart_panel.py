"""
ChartPanel — bottom-left quadrant.

Responsibilities:
  • White background pyqtgraph plots for Hip / Knee / Ankle
  • Real-time waveform updates during Pass 1 via push_live_frame():
      - Attempts on-the-fly angle estimation from raw XY keypoints
      - Falls back to joint visibility score if drone_mocap is unavailable
  • Gait-cycle event markers (heel-strike purple, swing-peak amber)
  • Draggable red cursor InfiniteLine synced bi-directionally with VideoPanel
  • Emits cursor_moved(float seconds) on drag
  • Receives set_cursor(float seconds) from VideoPanel
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout

from app.theme import PG_COLORS

try:
    from scipy.signal import find_peaks
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# Try importing the angle estimator for live preview
try:
    from drone_mocap.angles.saggital2D import joint_angles_sagittal
    _ANGLE_OK = True
except ImportError:
    _ANGLE_OK = False

# Joint keypoint indices used for live visibility fallback
_VIS_IDX = {
    "hip":   (23, 24),
    "knee":  (25, 26),
    "ankle": (27, 28),
}


def _detect_gait_events(
    knee_deg: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not _SCIPY_OK or len(knee_deg) < 10:
        return np.array([]), np.array([])
    valid = np.isfinite(knee_deg)
    if valid.sum() < 10:
        return np.array([]), np.array([])
    min_distance = max(5, int(len(knee_deg) * 0.04))
    hs_idx, _ = find_peaks(-knee_deg, distance=min_distance, prominence=5.0)
    sw_idx, _ = find_peaks( knee_deg, distance=min_distance, prominence=5.0)
    return time_s[hs_idx], time_s[sw_idx]


class ChartPanel(QWidget):
    cursor_moved = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # Header row: title + live indicator
        header = QHBoxLayout()
        title = QLabel("KINEMATIC WAVEFORMS")
        title.setObjectName("section_title")
        self._live_badge = QLabel("  LIVE  ")
        self._live_badge.setObjectName("section_title")
        self._live_badge.setStyleSheet(
            "background-color: #36B37E; color: #FFFFFF; border-radius: 3px;"
            "padding: 1px 5px; font-size: 10px; font-weight: 700;"
        )
        self._live_badge.setVisible(False)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._live_badge)
        layout.addLayout(header)

        # pyqtgraph — white background
        pg.setConfigOptions(antialias=True, foreground=PG_COLORS["axis_text"])
        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.setBackground(PG_COLORS["bg"])
        layout.addWidget(self._plot_widget, stretch=1)

        self._plots:  dict[str, pg.PlotItem]     = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._live_curves: dict[str, pg.PlotDataItem] = {}
        self._cursor: pg.InfiniteLine | None = None
        self._gait_lines: list = []

        # Live data buffers
        self._live_t:    deque[float] = deque(maxlen=3000)
        self._live_vals: dict[str, deque[float]] = {
            j: deque(maxlen=3000) for j in ("hip", "knee", "ankle")
        }
        self._fps_estimate: float = 30.0
        self._live_side: str = "right"

        self._build_plots()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_plots(self) -> None:
        joints = [
            ("hip",   "Hip (°)",   0),
            ("knee",  "Knee (°)",  1),
            ("ankle", "Ankle (°)", 2),
        ]
        prev_plot = None
        for joint, ylabel, row in joints:
            plot = self._plot_widget.addPlot(row=row, col=0)
            plot.setMenuEnabled(False)
            plot.setLabel("left", ylabel,
                          color=PG_COLORS[joint], size="10pt")
            plot.showGrid(x=True, y=True, alpha=0.4)

            # White background per plot
            plot.getViewBox().setBackgroundColor(PG_COLORS["bg"])

            # Axis styling (light-theme axes)
            for axis_name in ("bottom", "left", "top", "right"):
                ax = plot.getAxis(axis_name)
                ax.setPen(pg.mkPen(PG_COLORS["grid"], width=1))
                ax.setTextPen(pg.mkPen(PG_COLORS["axis_text"]))

            if prev_plot is not None:
                plot.setXLink(prev_plot)
            prev_plot = plot

            # Final smoothed curve (bold, solid)
            curve = plot.plot(
                pen=pg.mkPen(PG_COLORS[joint], width=2.0),
                name=joint,
            )
            # Live preview curve (thinner, semi-transparent)
            live_curve = plot.plot(
                pen=pg.mkPen(PG_COLORS[joint], width=1.0, style=Qt.PenStyle.DotLine),
                name=f"{joint}_live",
            )
            self._plots[joint]       = plot
            self._curves[joint]      = curve
            self._live_curves[joint] = live_curve

        # Cursor on knee plot (xLinked to all others)
        knee_plot = self._plots["knee"]
        self._cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen(PG_COLORS["cursor"], width=1.5,
                         style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(PG_COLORS["cursor"], width=2.5),
        )
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        knee_plot.addItem(self._cursor)

        for joint in ("hip", "ankle"):
            mirror = pg.InfiniteLine(
                pos=0.0, angle=90, movable=False,
                pen=pg.mkPen(PG_COLORS["cursor"], width=1.5,
                             style=Qt.PenStyle.DashLine),
            )
            self._plots[joint].addItem(mirror)
            setattr(self, f"_cursor_mirror_{joint}", mirror)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_live_mode(self, active: bool, side: str = "right", fps: float = 30.0) -> None:
        self._live_badge.setVisible(active)
        self._live_side = side
        self._fps_estimate = fps
        if active:
            self._live_t.clear()
            for buf in self._live_vals.values():
                buf.clear()

    def push_live_frame(
        self,
        frame_idx: int,
        total_frames: int,
        xy: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        """
        Called on every progress signal during Pass 1.
        Attempts live angle estimation; falls back to visibility proxy.
        """
        t = frame_idx / self._fps_estimate
        self._live_t.append(t)

        if _ANGLE_OK:
            try:
                ang = joint_angles_sagittal(
                    xy, vis,
                    visible_side=self._live_side,
                    min_vis=0.2,
                )
                self._live_vals["hip"].append(ang.get("hip", np.nan))
                self._live_vals["knee"].append(ang.get("knee", np.nan))
                self._live_vals["ankle"].append(ang.get("ankle", np.nan))
            except Exception:
                self._push_vis_proxy(xy, vis)
        else:
            self._push_vis_proxy(xy, vis)

        # Update live curves every 3 frames to avoid hammering the GUI thread
        if frame_idx % 3 == 0:
            t_arr = np.fromiter(self._live_t, dtype=float)
            for joint, buf in self._live_vals.items():
                v_arr = np.fromiter(buf, dtype=float)
                if len(t_arr) == len(v_arr) and len(t_arr) > 1:
                    self._live_curves[joint].setData(t_arr, v_arr)

    def _push_vis_proxy(self, xy: np.ndarray, vis: np.ndarray) -> None:
        """Fallback: scale visibility [0,1] to [-20, 80] as a proxy waveform."""
        for joint, (i, j) in _VIS_IDX.items():
            avg_vis = float((vis[i] + vis[j]) / 2) if len(vis) > max(i, j) else 0.0
            self._live_vals[joint].append(avg_vis * 100.0 - 20.0)

    def clear(self) -> None:
        for curve in list(self._curves.values()) + list(self._live_curves.values()):
            curve.setData([], [])
        for line in self._gait_lines:
            for plot in self._plots.values():
                try:
                    plot.removeItem(line)
                except Exception:
                    pass
        self._gait_lines.clear()
        self._live_badge.setVisible(False)

    def load_angles(self, csv_path: str) -> None:
        """Load final smoothed angles from the output CSV and replace live curves."""
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return
        if "time_s" not in df.columns:
            return

        t = df["time_s"].to_numpy(float)
        col_map: dict[str, str] = {}
        for col in df.columns:
            lc = col.lower()
            if "hip"   in lc: col_map["hip"]   = col
            elif "knee" in lc: col_map["knee"]  = col
            elif "ankle" in lc: col_map["ankle"] = col

        for joint, col in col_map.items():
            if joint in self._curves and col in df.columns:
                self._curves[joint].setData(t, df[col].to_numpy(float))
            # Clear live preview — real data has replaced it
            self._live_curves[joint].setData([], [])

        if "knee" in col_map:
            knee_deg = df[col_map["knee"]].to_numpy(float)
            hs_times, sw_times = _detect_gait_events(knee_deg, t)
            self._draw_gait_events(hs_times, sw_times)

        if self._cursor is not None and len(t):
            self._cursor.setValue(float(t[0]))

        self._live_badge.setVisible(False)

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
            m = getattr(self, f"_cursor_mirror_{joint}", None)
            if m is not None:
                m.setValue(t)

    def _draw_gait_events(
        self, hs_times: np.ndarray, sw_times: np.ndarray
    ) -> None:
        for plot in self._plots.values():
            for t_hs in hs_times:
                line = pg.InfiniteLine(
                    pos=float(t_hs), angle=90, movable=False,
                    pen=pg.mkPen(PG_COLORS["gait_hs"], width=0.9,
                                 style=Qt.PenStyle.DotLine),
                )
                plot.addItem(line)
                self._gait_lines.append(line)
            for t_sw in sw_times:
                line = pg.InfiniteLine(
                    pos=float(t_sw), angle=90, movable=False,
                    pen=pg.mkPen(PG_COLORS["gait_swing"], width=0.9,
                                 style=Qt.PenStyle.DotLine),
                )
                plot.addItem(line)
                self._gait_lines.append(line)
