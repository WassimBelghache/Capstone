"""
ControlPanel — left-top quadrant.

Responsibilities:
  • File selection (video + optional MoCap reference)
  • Pipeline parameter inputs (side, cutoff Hz, order, min_vis, height)
  • Output directory selection
  • RUN / ABORT button + progress bar
  • Live skeleton preview overlay (painted on top of a still frame grab)
  • Emits run_requested(dict) and abort_requested()
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QFileDialog, QDoubleSpinBox, QSpinBox, QComboBox,
    QProgressBar, QSizePolicy,
)


# MediaPipe skeleton links (same as run.py)
_SKELETON_LINKS = [
    (11,12),(11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),
    (11,13),(12,14),(13,15),(14,16),
]


class SkeletonPreview(QLabel):
    """Small canvas that draws a stick figure from keypoint arrays."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._xy:  np.ndarray | None = None
        self._vis: np.ndarray | None = None
        self._set_blank()

    def _set_blank(self) -> None:
        px = QPixmap(self.width() or 200, self.height() or 120)
        px.fill(QColor("#0d0d18"))
        self.setPixmap(px)

    def update_skeleton(self, xy: np.ndarray, vis: np.ndarray) -> None:
        self._xy  = xy
        self._vis = vis
        self._redraw()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._xy is None:
            self._set_blank()
        else:
            self._redraw()

    def _redraw(self) -> None:
        if self._xy is None:
            return
        xy  = self._xy
        vis = self._vis
        w, h = max(self.width(), 1), max(self.height(), 1)

        # Normalize keypoint positions to canvas
        valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        if valid.sum() < 2:
            self._set_blank()
            return

        x_vals = xy[valid, 0]
        y_vals = xy[valid, 1]
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        span_x = max(x_max - x_min, 1.0)
        span_y = max(y_max - y_min, 1.0)
        margin = 16

        def to_canvas(pt):
            nx = (pt[0] - x_min) / span_x * (w - 2 * margin) + margin
            ny = (pt[1] - y_min) / span_y * (h - 2 * margin) + margin
            return int(nx), int(ny)

        img = QImage(w, h, QImage.Format.Format_RGB32)
        img.fill(QColor("#0d0d18"))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        def kp_color(v: float) -> QColor:
            if v >= 0.8:
                return QColor("#30d158")
            if v >= 0.3:
                return QColor("#ffd60a")
            return QColor("#ff453a")

        for i, j in _SKELETON_LINKS:
            pi, pj = xy[i], xy[j]
            if not (np.all(np.isfinite(pi)) and np.all(np.isfinite(pj))):
                continue
            avg_vis = float((vis[i] + vis[j]) / 2)
            pen = QPen(kp_color(avg_vis), 1, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(*to_canvas(pi), *to_canvas(pj))

        for idx in range(33):
            pt = xy[idx]
            if not np.all(np.isfinite(pt)):
                continue
            cx, cy = to_canvas(pt)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(kp_color(float(vis[idx])))
            painter.drawEllipse(cx - 2, cy - 2, 4, 4)

        painter.end()
        self.setPixmap(QPixmap.fromImage(img))


class ControlPanel(QWidget):
    run_requested   = pyqtSignal(dict)
    abort_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._build_files_group())
        layout.addWidget(self._build_params_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_preview_group())
        layout.addStretch()

    # ── File selection ────────────────────────────────────────────────────────

    def _build_files_group(self) -> QGroupBox:
        grp = QGroupBox("INPUT FILES")
        g = QGridLayout(grp)
        g.setSpacing(4)

        self._video_edit = QLineEdit()
        self._video_edit.setPlaceholderText("Select video…")
        self._video_edit.setReadOnly(True)
        video_btn = QPushButton("…")
        video_btn.setFixedWidth(28)
        video_btn.clicked.connect(self._pick_video)

        self._mocap_edit = QLineEdit()
        self._mocap_edit.setPlaceholderText("MoCap reference (optional)")
        self._mocap_edit.setReadOnly(True)
        mocap_btn = QPushButton("…")
        mocap_btn.setFixedWidth(28)
        mocap_btn.clicked.connect(self._pick_mocap)

        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output directory…")
        self._out_edit.setReadOnly(True)
        out_btn = QPushButton("…")
        out_btn.setFixedWidth(28)
        out_btn.clicked.connect(self._pick_outdir)

        g.addWidget(QLabel("Video"), 0, 0)
        g.addWidget(self._video_edit, 0, 1)
        g.addWidget(video_btn, 0, 2)
        g.addWidget(QLabel("MoCap"), 1, 0)
        g.addWidget(self._mocap_edit, 1, 1)
        g.addWidget(mocap_btn, 1, 2)
        g.addWidget(QLabel("Output"), 2, 0)
        g.addWidget(self._out_edit, 2, 1)
        g.addWidget(out_btn, 2, 2)
        return grp

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.MP4);;All Files (*)"
        )
        if path:
            self._video_edit.setText(path)
            # Default output dir to same folder
            if not self._out_edit.text():
                self._out_edit.setText(str(Path(path).parent))

    def _pick_mocap(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MoCap File", "",
            "MoCap Files (*.txt *.csv);;All Files (*)"
        )
        if path:
            self._mocap_edit.setText(path)

    def _pick_outdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._out_edit.setText(path)

    def video_path(self) -> str | None:
        p = self._video_edit.text().strip()
        return p if p else None

    # ── Parameters ────────────────────────────────────────────────────────────

    def _build_params_group(self) -> QGroupBox:
        grp = QGroupBox("PIPELINE PARAMETERS")
        g = QGridLayout(grp)
        g.setSpacing(4)

        self._side_combo = QComboBox()
        self._side_combo.addItems(["right", "left"])

        self._cutoff_spin = QDoubleSpinBox()
        self._cutoff_spin.setRange(1.0, 30.0)
        self._cutoff_spin.setValue(6.0)
        self._cutoff_spin.setSuffix(" Hz")
        self._cutoff_spin.setSingleStep(0.5)

        self._order_spin = QSpinBox()
        self._order_spin.setRange(1, 8)
        self._order_spin.setValue(4)

        self._minvis_spin = QDoubleSpinBox()
        self._minvis_spin.setRange(0.0, 1.0)
        self._minvis_spin.setValue(0.3)
        self._minvis_spin.setSingleStep(0.05)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(0.0, 2.5)
        self._height_spin.setValue(0.0)
        self._height_spin.setSuffix(" m")
        self._height_spin.setSingleStep(0.01)
        self._height_spin.setSpecialValueText("(none)")

        rows = [
            ("Side",        self._side_combo),
            ("Cutoff",      self._cutoff_spin),
            ("Filter order",self._order_spin),
            ("Min vis",     self._minvis_spin),
            ("Athlete h.",  self._height_spin),
        ]
        for row, (label, widget) in enumerate(rows):
            g.addWidget(QLabel(label), row, 0)
            g.addWidget(widget, row, 1)
        return grp

    # ── Run / progress ────────────────────────────────────────────────────────

    def _build_run_group(self) -> QGroupBox:
        grp = QGroupBox("EXECUTION")
        v = QVBoxLayout(grp)
        v.setSpacing(4)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("RUN ANALYSIS")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self.abort_requested)

        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._abort_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress_label = QLabel("Idle")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        v.addLayout(btn_row)
        v.addWidget(self._progress)
        v.addWidget(self._progress_label)
        return grp

    def _on_run_clicked(self) -> None:
        video = self._video_edit.text().strip()
        out   = self._out_edit.text().strip()
        if not video or not out:
            return
        height = self._height_spin.value()
        params = {
            "video_path":       video,
            "out_root":         out,
            "mocap_path":       self._mocap_edit.text().strip() or None,
            "visible_side":     self._side_combo.currentText(),
            "cutoff_hz":        self._cutoff_spin.value(),
            "filter_order":     self._order_spin.value(),
            "min_vis":          self._minvis_spin.value(),
            "athlete_height_m": height if height > 0.0 else None,
        }
        self.run_requested.emit(params)

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._abort_btn.setEnabled(running)
        if not running:
            self._progress.setValue(0)
            self._progress_label.setText("Idle")

    def update_progress(self, frame_idx: int, total_frames: int) -> None:
        if total_frames > 0:
            pct = int(frame_idx / total_frames * 100)
            self._progress.setValue(pct)
            self._progress_label.setText(f"Frame {frame_idx} / {total_frames}")

    # ── Skeleton preview ──────────────────────────────────────────────────────

    def _build_preview_group(self) -> QGroupBox:
        grp = QGroupBox("LIVE SKELETON PREVIEW")
        v = QVBoxLayout(grp)
        self._skeleton = SkeletonPreview()
        self._skeleton.setMinimumHeight(140)
        v.addWidget(self._skeleton)
        return grp

    def update_skeleton(self, xy: np.ndarray, vis: np.ndarray) -> None:
        self._skeleton.update_skeleton(xy, vis)
