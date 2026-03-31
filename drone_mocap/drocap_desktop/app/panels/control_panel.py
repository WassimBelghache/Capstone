"""
ControlPanel — left-top quadrant.

Responsibilities:
  • File selection (video + optional MoCap reference)
  • Pipeline parameter inputs (side, cutoff Hz, order, min_vis, height)
  • Output directory selection
  • RUN / ABORT button + progress bar
  • Live skeleton overlay: QPainter draws 33 MediaPipe landmarks over a
    dimmed BGR frame — proving the AI sees the athlete in real-time.
  • Emits run_requested(dict) and abort_requested()
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QBrush
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QSizePolicy,
    QSpinBox, QVBoxLayout, QWidget,
)

# ── MediaPipe 33-keypoint skeleton connectivity ───────────────────────────
_SKELETON_LINKS = [
    (11,12),(11,23),(12,24),(23,24),          # torso
    (23,25),(24,26),(25,27),(26,28),          # legs
    (27,29),(28,30),(29,31),(30,32),          # feet
    (11,13),(12,14),(13,15),(14,16),          # arms
    (0,1),(1,2),(2,3),(3,7),                  # face left
    (0,4),(4,5),(5,6),(6,8),                  # face right
    (9,10),                                   # mouth
]

# Jira-blue for limbs; white for joints
_LIMB_COLOR  = QColor("#0052CC")
_JOINT_COLOR = QColor("#FFFFFF")
_JOINT_LOW   = QColor("#FF5630")   # red when vis < 0.3


class SkeletonPreview(QWidget):
    """
    Renders a live skeleton overlaid on the most-recent BGR video frame.

    The frame is dimmed to 55% brightness so the skeleton stands out.
    Limbs are drawn in Jira-blue; joints are white (red when low-confidence).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frame_bgr: np.ndarray | None = None
        self._xy:        np.ndarray | None = None
        self._vis:       np.ndarray | None = None

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        xy: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        self._frame_bgr = frame_bgr
        self._xy        = xy
        self._vis       = vis
        self.update()   # triggers paintEvent

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # ── Background ──────────────────────────────────────────────────────
        if self._frame_bgr is not None:
            frame = self._frame_bgr
            fh, fw = frame.shape[:2]
            # Convert BGR → RGB
            rgb = frame[:, :, ::-1].copy()
            # Dim to 55% brightness
            rgb = (rgb.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)
            # Ensure contiguous memory for QImage
            rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.data, fw, fh, fw * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            # Centre the frame
            x_off = (w - pixmap.width())  // 2
            y_off = (h - pixmap.height()) // 2
            painter.drawPixmap(x_off, y_off, pixmap)

            # ── Scale keypoints to the displayed region ──────────────────────
            scale  = min(w / fw, h / fh)
            ox     = (w - fw * scale) / 2
            oy     = (h - fh * scale) / 2

            def to_canvas(pt: np.ndarray) -> tuple[int, int]:
                return int(pt[0] * scale + ox), int(pt[1] * scale + oy)

        else:
            # Blank slate before first frame arrives
            painter.fillRect(0, 0, w, h, QColor("#EBECF0"))
            painter.setPen(QColor("#A5ADBA"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for analysis…",
            )
            return

        if self._xy is None or self._vis is None:
            return

        xy, vis = self._xy, self._vis

        # ── Limbs ────────────────────────────────────────────────────────────
        limb_pen = QPen(_LIMB_COLOR, 2, Qt.PenStyle.SolidLine)
        limb_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(limb_pen)
        for i, j in _SKELETON_LINKS:
            pi, pj = xy[i], xy[j]
            if not (np.all(np.isfinite(pi)) and np.all(np.isfinite(pj))):
                continue
            avg_vis = float((vis[i] + vis[j]) / 2)
            if avg_vis < 0.1:
                continue
            painter.drawLine(*to_canvas(pi), *to_canvas(pj))

        # ── Joints ───────────────────────────────────────────────────────────
        painter.setPen(Qt.PenStyle.NoPen)
        for idx in range(33):
            pt = xy[idx]
            if not np.all(np.isfinite(pt)):
                continue
            v = float(vis[idx])
            if v < 0.1:
                continue
            cx, cy = to_canvas(pt)
            color = _JOINT_COLOR if v >= 0.3 else _JOINT_LOW
            # Outer white/red ring
            painter.setBrush(QBrush(color))
            r = 4
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)
            # Inner dark dot for depth cue
            painter.setBrush(QBrush(QColor("#172B4D")))
            painter.drawEllipse(cx - 2, cy - 2, 4, 4)


# ── ControlPanel ──────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    from PyQt6.QtCore import pyqtSignal
    run_requested   = pyqtSignal(dict)
    abort_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_files_group())
        layout.addWidget(self._build_params_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_preview_group(), stretch=1)

    # ── File selection ────────────────────────────────────────────────────────

    def _build_files_group(self) -> QGroupBox:
        grp = QGroupBox("Input Files")
        g = QGridLayout(grp)
        g.setSpacing(6)
        g.setColumnStretch(1, 1)

        self._video_edit = QLineEdit()
        self._video_edit.setPlaceholderText("Select video file…")
        self._video_edit.setReadOnly(True)
        video_btn = QPushButton("Browse")
        video_btn.setFixedWidth(60)
        video_btn.clicked.connect(self._pick_video)

        self._mocap_edit = QLineEdit()
        self._mocap_edit.setPlaceholderText("MoCap reference (optional)")
        self._mocap_edit.setReadOnly(True)
        mocap_btn = QPushButton("Browse")
        mocap_btn.setFixedWidth(60)
        mocap_btn.clicked.connect(self._pick_mocap)

        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output directory…")
        self._out_edit.setReadOnly(True)
        out_btn = QPushButton("Browse")
        out_btn.setFixedWidth(60)
        out_btn.clicked.connect(self._pick_outdir)

        for row, (lbl, edit, btn) in enumerate([
            ("Video",  self._video_edit, video_btn),
            ("MoCap",  self._mocap_edit, mocap_btn),
            ("Output", self._out_edit,   out_btn),
        ]):
            g.addWidget(QLabel(lbl), row, 0)
            g.addWidget(edit, row, 1)
            g.addWidget(btn,  row, 2)

        return grp

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.MP4);;All Files (*)"
        )
        if path:
            self._video_edit.setText(path)
            from pathlib import Path
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
        grp = QGroupBox("Pipeline Parameters")
        g = QGridLayout(grp)
        g.setSpacing(6)
        g.setColumnStretch(1, 1)

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
            ("Visible side",   self._side_combo),
            ("Filter cutoff",  self._cutoff_spin),
            ("Filter order",   self._order_spin),
            ("Min visibility", self._minvis_spin),
            ("Athlete height", self._height_spin),
        ]
        for row, (label, widget) in enumerate(rows):
            lbl = QLabel(label)
            g.addWidget(lbl,    row, 0)
            g.addWidget(widget, row, 1)
        return grp

    # ── Run / progress ────────────────────────────────────────────────────────

    def _build_run_group(self) -> QGroupBox:
        grp = QGroupBox("Execution")
        v = QVBoxLayout(grp)
        v.setSpacing(6)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Start Analysis")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.clicked.connect(self._on_run_clicked)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setObjectName("abort_btn")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self.abort_requested)

        btn_row.addWidget(self._run_btn, stretch=3)
        btn_row.addWidget(self._abort_btn, stretch=1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%  (%v / %m frames)")

        self._progress_label = QLabel("Idle")
        self._progress_label.setObjectName("status_label")

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
            self._progress.setMaximum(100)
            self._progress.setFormat("%p%")
            self._progress_label.setText("Idle")

    def update_progress(self, frame_idx: int, total_frames: int) -> None:
        if self._progress.maximum() != total_frames:
            self._progress.setMaximum(max(total_frames, 1))
            self._progress.setFormat(f"%v / {total_frames} frames")
        self._progress.setValue(frame_idx)
        self._progress_label.setText(f"Processing frame {frame_idx} of {total_frames}…")

    # ── Live skeleton preview ─────────────────────────────────────────────────

    def _build_preview_group(self) -> QGroupBox:
        grp = QGroupBox("Live Skeleton Preview")
        v = QVBoxLayout(grp)
        v.setContentsMargins(4, 4, 4, 4)
        self._skeleton = SkeletonPreview()
        self._skeleton.setMinimumHeight(160)
        v.addWidget(self._skeleton, stretch=1)
        return grp

    def update_skeleton(
        self,
        frame_bgr: np.ndarray,
        xy: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        self._skeleton.update_frame(frame_bgr, xy, vis)
