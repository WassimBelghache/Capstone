"""
ChartPanel — bottom-left quadrant.

Responsibilities:
  • Display hip / knee / ankle angle time-series (pyqtgraph)
  • Detect gait cycles: heel-strike minima (knee) and swing peaks
  • Mark gait events as vertical lines
  • Draggable cursor InfiniteLine synced with VideoPanel
  • Emit cursor_moved(float seconds) on drag
  • Receive set_cursor(float seconds) from VideoPanel time changes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from app.theme import PG_COLORS

try:
    from scipy.signal import find_peaks
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def _detect_gait_events(
    knee_deg: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (heel_strike_times, swing_peak_times) in seconds.

    Heel-strikes  → minima of knee angle (leg straightens at contact)
    Swing peaks   → maxima of knee angle (max flexion during swing)
    """
    if not _SCIPY_OK or len(knee_deg) < 10:
        return np.array([]), np.array([])
    valid = np.isfinite(knee_deg)
    if valid.sum() < 10:
        return np.array([]), np.array([])

    # Use a prominence-based peak finder on the NEGATED signal for minima
    min_distance = max(5, int(len(knee_deg) * 0.04))
    hs_idx, _ = find_peaks(-knee_deg, distance=min_distance, prominence=5.0)
    sw_idx, _ = find_peaks( knee_deg, distance=min_distance, prominence=5.0)
    return time_s[hs_idx], time_s[sw_idx]


class ChartPanel(QWidget):
    cursor_moved = pyqtSignal(float)   # seconds

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        pg.setConfigOptions(antialias=True, background=PG_COLORS["bg"])

        self._plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self._plot_widget)

        self._plots: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._cursor: pg.InfiniteLine | None = None
        self._gait_lines: list[pg.InfiniteLine] = []

        self._build_plots()

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_plots(self) -> None:
        joints = [("hip", "Hip  (°)"), ("knee", "Knee  (°)"), ("ankle", "Ankle  (°)")]
        prev_plot = None
        for i, (joint, ylabel) in enumerate(joints):
            plot = self._plot_widget.addPlot(row=i, col=0)
            plot.setLabel("left", ylabel, color=PG_COLORS[joint])
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.getAxis("bottom").setPen(pg.mkPen(PG_COLORS["grid"]))
            plot.getAxis("left").setPen(pg.mkPen(PG_COLORS["grid"]))
            if prev_plot is not None:
                plot.setXLink(prev_plot)
            prev_plot = plot

            curve = plot.plot(
                pen=pg.mkPen(PG_COLORS[joint], width=1.6),
                name=joint,
            )
            self._plots[joint]  = plot
            self._curves[joint] = curve

        # Add cursor InfiniteLine to the knee plot (linked via setXLink to all)
        knee_plot = self._plots["knee"]
        self._cursor = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen(PG_COLORS["cursor"], width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
            label="",
        )
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        knee_plot.addItem(self._cursor)
        # Mirror cursor lines to hip and ankle for visual clarity
        for joint in ("hip", "ankle"):
            mirror = pg.InfiniteLine(
                pos=0.0,
                angle=90,
                movable=False,
                pen=pg.mkPen(PG_COLORS["cursor"], width=1,
                             style=pg.QtCore.Qt.PenStyle.DashLine),
            )
            self._plots[joint].addItem(mirror)
            # Keep reference so we can update them
            setattr(self, f"_cursor_mirror_{joint}", mirror)

    # ── Public API ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        for curve in self._curves.values():
            curve.setData([], [])
        for line in self._gait_lines:
            for plot in self._plots.values():
                try:
                    plot.removeItem(line)
                except Exception:
                    pass
        self._gait_lines.clear()

    def load_angles(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        if "time_s" not in df.columns:
            return
        t = df["time_s"].to_numpy(float)

        col_map = {}
        for col in df.columns:
            lc = col.lower()
            if "hip" in lc:
                col_map["hip"] = col
            elif "knee" in lc:
                col_map["knee"] = col
            elif "ankle" in lc:
                col_map["ankle"] = col

        for joint, col in col_map.items():
            if joint in self._curves and col in df.columns:
                self._curves[joint].setData(t, df[col].to_numpy(float))

        # Gait events from knee signal
        if "knee" in col_map:
            knee_deg = df[col_map["knee"]].to_numpy(float)
            hs_times, sw_times = _detect_gait_events(knee_deg, t)
            self._draw_gait_events(hs_times, sw_times)

        # Move cursor to start
        if self._cursor is not None:
            self._cursor.setValue(float(t[0]) if len(t) else 0.0)

    def set_cursor(self, time_s: float) -> None:
        """Called by VideoPanel — move cursor without emitting cursor_moved."""
        if self._cursor is None:
            return
        self._cursor.blockSignals(True)
        self._cursor.setValue(time_s)
        self._cursor.blockSignals(False)
        self._sync_mirror_cursors(time_s)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _on_cursor_moved(self, line: pg.InfiniteLine) -> None:
        t = float(line.value())
        self._sync_mirror_cursors(t)
        self.cursor_moved.emit(t)

    def _sync_mirror_cursors(self, t: float) -> None:
        for joint in ("hip", "ankle"):
            mirror = getattr(self, f"_cursor_mirror_{joint}", None)
            if mirror is not None:
                mirror.setValue(t)

    def _draw_gait_events(
        self, hs_times: np.ndarray, sw_times: np.ndarray
    ) -> None:
        for plot in self._plots.values():
            for t_hs in hs_times:
                line = pg.InfiniteLine(
                    pos=float(t_hs),
                    angle=90,
                    movable=False,
                    pen=pg.mkPen(PG_COLORS["gait_hs"], width=0.8,
                                 style=pg.QtCore.Qt.PenStyle.DotLine),
                )
                plot.addItem(line)
                self._gait_lines.append(line)
            for t_sw in sw_times:
                line = pg.InfiniteLine(
                    pos=float(t_sw),
                    angle=90,
                    movable=False,
                    pen=pg.mkPen(PG_COLORS["gait_swing"], width=0.8,
                                 style=pg.QtCore.Qt.PenStyle.DotLine),
                )
                plot.addItem(line)
                self._gait_lines.append(line)
