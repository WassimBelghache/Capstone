"""
MetricsPanel — bottom-right quadrant.

Displays metrics from metrics_mocap.json:
  • System Accuracy badge (top) — overall grade based on median Pearson r
  • QTableWidget with RMSE / MAE / r per joint
  • Conditional colour formatting on the r column:
      Green  #00C896 : r > 0.90  (Excellent)
      Amber  #FF8C42 : 0.75 ≤ r ≤ 0.90  (Good)
      Red    #FF4757 : r < 0.75  (Requires review)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_JOINTS = ["HIP", "KNEE", "ANKLE"]
_COLS   = ["Joint", "RMSE (°)", "MAE (°)", "r", "Matched MoCap column"]

# Pearson r thresholds
_R_EXCELLENT = 0.90
_R_GOOD      = 0.75

# Text colours
_CLR_EXCELLENT = QColor("#00C896")
_CLR_GOOD      = QColor("#FF8C42")
_CLR_POOR      = QColor("#FF4757")
_CLR_DEFAULT   = QColor("#172B4D")


def _fmt(val: object, precision: int = 2) -> str:
    try:
        f = float(val)   # type: ignore[arg-type]
        if f != f:        # NaN
            return "—"
        return f"{f:.{precision}f}"
    except (TypeError, ValueError):
        return "—"


def _r_color(r_str: str) -> QColor:
    try:
        r = float(r_str)
        if r > _R_EXCELLENT:
            return _CLR_EXCELLENT
        if r >= _R_GOOD:
            return _CLR_GOOD
        return _CLR_POOR
    except ValueError:
        return _CLR_DEFAULT


def _overall_grade(r_values: list[float]) -> tuple[str, str, str]:
    """Return (grade_letter, label, hex_color) based on median Pearson r."""
    if not r_values:
        return "–", "No data", "#A5ADBA"
    med = float(np.nanmedian(r_values))
    if med > _R_EXCELLENT:
        return "A", f"Excellent  (r = {med:.3f})", "#00C896"
    if med >= _R_GOOD:
        return "B", f"Good  (r = {med:.3f})", "#FF8C42"
    return "C", f"Requires Review  (r = {med:.3f})", "#FF4757"


class _AccuracyBadge(QFrame):
    """Horizontal summary card: grade letter + label."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedHeight(52)
        self.setStyleSheet(
            "QFrame { background: #F4F5F7; border: 1px solid #DFE1E6;"
            " border-radius: 4px; }"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(12)

        cap_label = QLabel("SYSTEM ACCURACY")
        cap_label.setObjectName("section_title")
        cap_label.setStyleSheet("color:#5E6C84; font-size:10px; font-weight:700;"
                                " letter-spacing:1px; background:transparent;"
                                " border:none;")
        row.addWidget(cap_label)

        self._grade_lbl = QLabel("–")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        self._grade_lbl.setFont(font)
        self._grade_lbl.setStyleSheet("color:#A5ADBA; background:transparent; border:none;")

        self._desc_lbl = QLabel("Run an analysis to see results.")
        self._desc_lbl.setStyleSheet("color:#5E6C84; font-size:11px;"
                                     " background:transparent; border:none;")

        row.addWidget(self._grade_lbl)
        row.addWidget(self._desc_lbl, stretch=1)

    def update(self, grade: str, label: str, color: str) -> None:
        self._grade_lbl.setText(grade)
        self._grade_lbl.setStyleSheet(
            f"color:{color}; background:transparent; border:none;"
        )
        self._desc_lbl.setText(label)
        self._desc_lbl.setStyleSheet(
            f"color:{color}; font-size:11px; font-weight:600;"
            f" background:transparent; border:none;"
        )

    def reset(self) -> None:
        self._grade_lbl.setText("–")
        self._grade_lbl.setStyleSheet("color:#A5ADBA; background:transparent; border:none;")
        self._desc_lbl.setText("Run an analysis to see results.")
        self._desc_lbl.setStyleSheet("color:#5E6C84; font-size:11px;"
                                     " background:transparent; border:none;")


class MetricsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Accuracy badge ────────────────────────────────────────────────────
        self._badge = _AccuracyBadge()
        layout.addWidget(self._badge)

        # ── Metrics table ─────────────────────────────────────────────────────
        grp = QGroupBox("Comparison Metrics")
        inner = QVBoxLayout(grp)
        inner.setContentsMargins(6, 6, 6, 6)

        self._table = QTableWidget(len(_JOINTS), len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setShowGrid(False)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(len(_COLS) - 1, QHeaderView.ResizeMode.Stretch)
        for col in range(len(_COLS) - 1):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        inner.addWidget(self._table)

        # Footer: alignment notes
        self._footer = QLabel("")
        self._footer.setStyleSheet("color:#5E6C84; font-size:10px;")
        self._footer.setAlignment(Qt.AlignmentFlag.AlignRight)
        inner.addWidget(self._footer)

        layout.addWidget(grp, stretch=1)

        self.clear()

    # ── Public API ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        for row, joint in enumerate(_JOINTS):
            self._set_row(row, joint.capitalize(), "—", "—", "—", "—")
        self._badge.reset()
        self._footer.setText("")

    def load_metrics(self, json_path: str) -> None:
        try:
            payload = json.loads(Path(json_path).read_text())
        except Exception:
            self._footer.setText("Failed to load metrics.")
            return

        metrics = payload.get("metrics", {})
        r_values: list[float] = []

        for row, joint in enumerate(_JOINTS):
            jm   = metrics.get(joint, {})
            rmse = _fmt(jm.get("rmse"))
            mae  = _fmt(jm.get("mae"))
            corr = _fmt(jm.get("corr"), precision=3)
            col  = str(jm.get("matched_col") or "—")
            self._set_row(row, joint.capitalize(), rmse, mae, corr, col)

            # Collect valid r values for badge
            try:
                r_val = float(jm.get("corr", float("nan")))
                if r_val == r_val:     # not NaN
                    r_values.append(r_val)
            except (TypeError, ValueError):
                pass

        # Update accuracy badge
        grade, label, color = _overall_grade(r_values)
        self._badge.update(grade, label, color)

        # Footer alignment notes
        parts = []
        shift = metrics.get("_time_shift_frames")
        if shift is not None:
            parts.append(f"Time shift: {shift} frames")
        ankle_tf = metrics.get("_ankle_transform", "")
        if ankle_tf and ankle_tf != "raw":
            parts.append(f"Ankle transform: {ankle_tf}")
        self._footer.setText("  ·  ".join(parts))

    # ── Internals ──────────────────────────────────────────────────────────────

    def _set_row(
        self,
        row: int,
        joint: str,
        rmse: str,
        mae: str,
        corr: str,
        col: str,
    ) -> None:
        values = [joint, rmse, mae, corr, col]
        r_col_idx = 3   # index of the Pearson r column

        for c, val in enumerate(values):
            item = QTableWidgetItem(val)

            # Alignment
            if c < len(values) - 1:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )

            # Conditional colour on r column
            if c == r_col_idx and val != "—":
                item.setForeground(_r_color(val))
                f = item.font()
                f.setBold(True)
                item.setFont(f)

            # Joint name bold
            if c == 0:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
                item.setForeground(QColor(_get_joint_color(joint.upper())))

            self._table.setItem(row, c, item)


def _get_joint_color(joint: str) -> str:
    return {
        "HIP":   PG_COLORS["hip"],
        "KNEE":  PG_COLORS["knee"],
        "ANKLE": PG_COLORS["ankle"],
    }.get(joint, "#172B4D")


# Late import to avoid circular; theme is always available
from app.theme import PG_COLORS  # noqa: E402
