"""
DroCap Mission Control — MainWindow

Layout (4-quadrant QSplitter):
    ┌─────────────────┬───────────────────────┐
    │  ControlPanel   │     VideoPanel         │
    │  (top-left)     │     (top-right)        │
    ├─────────────────┼───────────────────────┤
    │  ChartPanel     │     MetricsPanel       │
    │  (bottom-left)  │     (bottom-right)     │
    └─────────────────┴───────────────────────┘
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QStatusBar,
    QWidget,
    QVBoxLayout,
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
        self.resize(1400, 820)

        self._worker: AnalysisWorker | None = None

        # ── Panels ──────────────────────────────────────────────────────────
        self.control_panel = ControlPanel()
        self.video_panel   = VideoPanel()
        self.chart_panel   = ChartPanel()
        self.metrics_panel = MetricsPanel()

        # ── Splitters ────────────────────────────────────────────────────────
        left_split  = QSplitter(Qt.Orientation.Vertical)
        left_split.addWidget(self.control_panel)
        left_split.addWidget(self.chart_panel)
        left_split.setSizes([320, 480])

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(self.video_panel)
        right_split.addWidget(self.metrics_panel)
        right_split.setSizes([480, 320])

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.addWidget(left_split)
        h_split.addWidget(right_split)
        h_split.setSizes([420, 980])

        self.setCentralWidget(h_split)

        # ── Status bar ───────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

        # ── Signal wiring ────────────────────────────────────────────────────
        self.control_panel.run_requested.connect(self._start_analysis)
        self.control_panel.abort_requested.connect(self._abort_analysis)

        # Chart ↔ Video cursor sync
        self.chart_panel.cursor_moved.connect(self.video_panel.seek_to_time)
        self.video_panel.time_changed.connect(self.chart_panel.set_cursor)

    # ── Analysis lifecycle ────────────────────────────────────────────────────

    def _start_analysis(self, params: dict) -> None:
        if self._worker and self._worker.isRunning():
            return

        self.chart_panel.clear()
        self.metrics_panel.clear()
        self.control_panel.set_running(True)
        self.status_bar.showMessage("Running analysis…")

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
        xy: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        self.control_panel.update_progress(frame_idx, total_frames)
        self.video_panel.update_skeleton(xy, vis)

    def _on_finished(self, out_dir: str) -> None:
        self.control_panel.set_running(False)
        self.status_bar.showMessage(f"Done — {out_dir}")

        out = Path(out_dir)
        angles_csv = out / "derived" / "angles_sagittal.csv"
        metrics_json = out / "reports" / "metrics_mocap.json"

        if angles_csv.exists():
            self.chart_panel.load_angles(str(angles_csv))

        if metrics_json.exists():
            self.metrics_panel.load_metrics(str(metrics_json))

        video_path = self.control_panel.video_path()
        if video_path:
            self.video_panel.load_video(video_path)

    def _on_error(self, message: str) -> None:
        self.control_panel.set_running(False)
        self.status_bar.showMessage(f"Error: {message}")
