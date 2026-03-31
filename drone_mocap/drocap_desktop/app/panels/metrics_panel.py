"""
MetricsPanel — bottom-right quadrant.

Displays the RMSE / MAE / r comparison metrics from metrics_mocap.json
in a styled QTableWidget.  Shows "—" when no MoCap comparison was run.
"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox,
)


_JOINTS = ["HIP", "KNEE", "ANKLE"]
_COLS   = ["Joint", "RMSE (°)", "MAE (°)", "r", "MoCap column"]


def _fmt(val: object, precision: int = 2) -> str:
    try:
        f = float(val)  # type: ignore[arg-type]
        return f"{f:.{precision}f}" if not (f != f) else "—"  # NaN check
    except (TypeError, ValueError):
        return "—"


class MetricsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        grp = QGroupBox("COMPARISON METRICS")
        inner = QVBoxLayout(grp)

        self._table = QTableWidget(len(_JOINTS), len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            len(_COLS) - 1, QHeaderView.ResizeMode.Stretch
        )
        for col in range(len(_COLS) - 1):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        inner.addWidget(self._table)

        self._status_label = QLabel("No comparison data.")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setObjectName("section_title")
        inner.addWidget(self._status_label)

        layout.addWidget(grp)
        self.clear()

    def clear(self) -> None:
        for row, joint in enumerate(_JOINTS):
            self._set_row(row, joint, "—", "—", "—", "—")
        self._status_label.setText("No comparison data.")

    def load_metrics(self, json_path: str) -> None:
        try:
            payload = json.loads(Path(json_path).read_text())
        except Exception:
            self._status_label.setText("Failed to load metrics.")
            return

        metrics = payload.get("metrics", {})
        for row, joint in enumerate(_JOINTS):
            jm = metrics.get(joint, {})
            rmse = _fmt(jm.get("rmse"))
            mae  = _fmt(jm.get("mae"))
            corr = _fmt(jm.get("corr"), precision=3)
            col  = str(jm.get("matched_col") or "—")
            self._set_row(row, joint.capitalize(), rmse, mae, corr, col)

        shift = metrics.get("_time_shift_frames")
        ankle_tf = metrics.get("_ankle_transform", "")
        parts = []
        if shift is not None:
            parts.append(f"shift={shift} fr")
        if ankle_tf:
            parts.append(f"ankle={ankle_tf}")
        self._status_label.setText("  |  ".join(parts) if parts else "Aligned.")

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
        for c, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
                if c < len(values) - 1
                else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, c, item)
