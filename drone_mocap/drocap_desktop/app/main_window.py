"""
DroCap Mission Control — MainWindow

Layout:
    ┌─────────────────┬───────────────────────┐
    │  ControlPanel   │     VideoPanel         │
    │  (top-left)     │     (top-right)        │
    ├─────────────────┼───────────────────────┤
    │  ChartPanel     │     MetricsPanel       │
    │  (bottom-left)  │     (bottom-right)     │
    └─────────────────┴───────────────────────┘
    [== Global Progress Bar ====================]  ← Jira-style bottom bar
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.panels.control_panel  import ControlPanel
from app.panels.video_panel    import VideoPanel
from app.panels.chart_panel    import ChartPanel
from app.panels.metrics_panel  import MetricsPanel
from app.workers.analysis_worker import AnalysisWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DroCap Mission Control")
        self.resize(1440, 860)

        self._worker: AnalysisWorker | None = None

        # ── Panels ───────────────────────────────────────────────────────────
        self.control_panel = ControlPanel()
        self.video_panel   = VideoPanel()
        self.chart_panel   = ChartPanel()
        self.metrics_panel = MetricsPanel()

        # ── Splitters ────────────────────────────────────────────────────────
        left_split = QSplitter(Qt.Orientation.Vertical)
        left_split.addWidget(self.control_panel)
        left_split.addWidget(self.chart_panel)
        left_split.setSizes([340, 460])

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(self.video_panel)
        right_split.addWidget(self.metrics_panel)
        right_split.setSizes([500, 300])

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.addWidget(left_split)
        h_split.addWidget(right_split)
        h_split.setSizes([440, 1000])

        # ── Global progress bar (bottom of central widget) ───────────────────
        self._global_progress = QProgressBar()
        self._global_progress.setRange(0, 100)
        self._global_progress.setValue(0)
        self._global_progress.setFixedHeight(6)
        self._global_progress.setTextVisible(False)
        self._global_progress.setStyleSheet(
            "QProgressBar { background: #EBECF0; border: none; border-radius: 0; }"
            "QProgressBar::chunk { background: #0052CC; }"
        )

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(h_split, stretch=1)
        central_layout.addWidget(self._global_progress)
        self.setCentralWidget(central)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_msg = QLabel("Ready")
        status_bar = QStatusBar()
        status_bar.addWidget(self._status_msg, 1)
        self.setStatusBar(status_bar)

        # ── Signal wiring ─────────────────────────────────────────────────────
        self.control_panel.run_requested.connect(self._start_analysis)
        self.control_panel.abort_requested.connect(self._abort_analysis)

        self.chart_panel.cursor_moved.connect(self.video_panel.seek_to_time)
        self.video_panel.time_changed.connect(self.chart_panel.set_cursor)

    # ── Analysis lifecycle ────────────────────────────────────────────────────

    def _start_analysis(self, params: dict) -> None:
        if self._worker and self._worker.isRunning():
            return

        self.chart_panel.clear()
        self.metrics_panel.clear()
        self.control_panel.set_running(True)
        self._global_progress.setValue(0)
        self._set_status("Running analysis…")

        # Tell ChartPanel we are in live mode so it shows the LIVE badge
        self.chart_panel.set_live_mode(
            active=True,
            side=params["visible_side"],
            fps=30.0,   # will be refined once the first frame arrives
        )

        self._worker = AnalysisWorker(
            video_path       = params["video_path"],
            out_root         = params["out_root"],
            mocap_path       = params.get("mocap_path"),
            visible_side     = params["visible_side"],
            cutoff_hz        = params["cutoff_hz"],
            filter_order     = params["filter_order"],
            min_vis          = params["min_vis"],
            athlete_height_m = params.get("athlete_height_m"),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _abort_analysis(self) -> None:
        if self._worker:
            self._worker.abort()

    def _on_progress(
        self,
        frame_idx: int,
        total_frames: int,
        frame_bgr: np.ndarray,
        xy: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        # Global progress bar
        if total_frames > 0:
            self._global_progress.setMaximum(total_frames)
            self._global_progress.setValue(frame_idx)

        # Control panel: per-panel progress text + live skeleton overlay
        self.control_panel.update_progress(frame_idx, total_frames)
        self.control_panel.update_skeleton(frame_bgr, xy, vis)

        # Video panel: show skeleton on blank canvas during inference
        self.video_panel.update_skeleton(frame_bgr, xy, vis)

        # Chart: stream live angles / visibility proxy
        self.chart_panel.push_live_frame(frame_idx, total_frames, xy, vis)

        # Update status bar every 30 frames to avoid flicker
        if frame_idx % 30 == 0 and total_frames > 0:
            pct = int(frame_idx / total_frames * 100)
            self._set_status(f"Analysing…  {pct}%  (frame {frame_idx} / {total_frames})")

    def _on_finished(self, out_dir: str) -> None:
        self.control_panel.set_running(False)
        self._global_progress.setValue(self._global_progress.maximum())
        self._set_status(f"Analysis complete  —  {out_dir}")

        out = Path(out_dir)
        angles_csv   = out / "derived" / "angles_sagittal.csv"
        metrics_json = out / "reports"  / "metrics_mocap.json"

        if angles_csv.exists():
            self.chart_panel.load_angles(str(angles_csv))

        if metrics_json.exists():
            self.metrics_panel.load_metrics(str(metrics_json))

        video_path = self.control_panel.video_path()
        if video_path:
            self.video_panel.load_video(video_path)

    def _on_error(self, message: str) -> None:
        self.control_panel.set_running(False)
        self._global_progress.setValue(0)
        self._set_status(f"Error: {message}")

    def _set_status(self, msg: str) -> None:
        self._status_msg.setText(msg)
