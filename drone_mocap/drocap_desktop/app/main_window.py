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
    [== Global Progress Bar ====================]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.panels.control_panel    import ControlPanel
from app.panels.video_panel      import VideoPanel
from app.panels.chart_panel      import ChartPanel
from app.panels.metrics_panel    import MetricsPanel
from app.workers.analysis_worker import AnalysisWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DroCap Mission Control")
        self.resize(1440, 860)

        self._worker: AnalysisWorker | None = None
        self._video_fps: float = 30.0
        self._last_out_dir: str | None = None   # for PDF export
        self._last_gait_cycles = None           # GaitCycles | None

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

        # ── Global progress bar ───────────────────────────────────────────────
        self._global_progress = QProgressBar()
        self._global_progress.setRange(0, 100)
        self._global_progress.setValue(0)
        self._global_progress.setFixedHeight(6)
        self._global_progress.setTextVisible(False)
        self._global_progress.setStyleSheet(
            "QProgressBar { background:#EBECF0; border:none; border-radius:0; }"
            "QProgressBar::chunk { background:#0052CC; }"
        )

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(h_split, stretch=1)
        cl.addWidget(self._global_progress)
        self.setCentralWidget(central)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_msg = QLabel("Ready")
        sb = QStatusBar()
        sb.addWidget(self._status_msg, 1)
        self.setStatusBar(sb)

        # ── Signal wiring ─────────────────────────────────────────────────────
        self.control_panel.run_requested.connect(self._start_analysis)
        self.control_panel.abort_requested.connect(self._abort_analysis)

        self.chart_panel.cursor_moved.connect(self.video_panel.seek_to_time)
        self.video_panel.time_changed.connect(self.chart_panel.set_cursor)
        self.chart_panel.stride_jumped.connect(self.video_panel.seek_to_time)

        self.metrics_panel.export_requested.connect(self._export_pdf)

    # ── Analysis lifecycle ────────────────────────────────────────────────────

    def _disconnect_worker(self) -> None:
        """Disconnect all signals from the current worker to prevent zombie signals."""
        if self._worker is None:
            return
        try:
            self._worker.progress.disconnect()
        except Exception:
            pass
        try:
            self._worker.finished.disconnect()
        except Exception:
            pass
        try:
            self._worker.error.disconnect()
        except Exception:
            pass

    def _start_analysis(self, params: dict) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._disconnect_worker()
        self.chart_panel.clear()
        self.metrics_panel.clear()
        self.control_panel.set_running(True)
        self.video_panel.enter_live_mode()
        self._global_progress.setValue(0)
        self._set_status("Running analysis\u2026")

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

        # Resolve real FPS + frame count once — avoids per-frame overhead
        try:
            from drone_mocap.io.video import get_video_meta
            meta = get_video_meta(Path(params["video_path"]))
            self._video_fps = meta.fps
            self._global_progress.setMaximum(max(meta.frame_count, 1))
        except Exception:
            self._video_fps = 30.0
            self._global_progress.setMaximum(1000)

        self.chart_panel.set_live_mode(
            active=True,
            side=params["visible_side"],
            fps=self._video_fps,
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
        rgb_small: np.ndarray,
        xy: np.ndarray,
        vis: np.ndarray,
        angles: object,
    ) -> None:
        self._global_progress.setValue(frame_idx)
        self.control_panel.update_progress(frame_idx, total_frames)
        self.video_panel.update_skeleton(rgb_small, xy, vis)
        self.chart_panel.push_live_frame(frame_idx, total_frames, vis, angles)

        # Status bar every 15 emitted signals (= 15 × _EMIT_STRIDE source frames)
        if total_frames > 0 and (frame_idx // 6) % 15 == 0:
            pct = int(frame_idx / total_frames * 100)
            self._set_status(
                f"Analysing\u2026  {pct}%  ({frame_idx} / {total_frames} frames)"
            )

    def _on_finished(self, out_dir: str) -> None:
        # ── Immediate: sever the worker ───────────────────────────────────────
        # Disconnect signals first — no more zombie progress/error signals.
        self._disconnect_worker()

        # Transition button to "Finalizing…" while results load.
        # set_running(False) is deferred to the last waterfall step.
        self.control_panel.set_finalizing()
        self._global_progress.setValue(self._global_progress.maximum())
        self._set_status("Processing results\u2026")
        self._last_out_dir = out_dir

        out          = Path(out_dir)
        angles_csv   = out / "derived" / "angles_sagittal.csv"
        metrics_json = out / "reports"  / "metrics_mocap.json"

        # Tear down the QThread object safely.
        # run() has already returned (that's what caused this signal), so
        # wait() returns immediately — but we defer to the next event loop
        # iteration (delay=0) so the current signal delivery is fully unwound
        # before we join the thread.
        if self._worker is not None:
            _w = self._worker
            self._worker = None
            QTimer.singleShot(0, lambda: (_w.wait(), _w.deleteLater()))

        # ── Waterfall loading ─────────────────────────────────────────────────
        # Each step runs in a separate event-loop turn so the UI stays
        # responsive and QMediaPlayer / pandas file handles don't race.

        # Step 1 — 100 ms: metrics JSON (fast; makes the badge appear quickly)
        def _load_metrics() -> None:
            if metrics_json.exists():
                self.metrics_panel.load_metrics(str(metrics_json))

        # Step 2 — 300 ms: final smoothed angle curves into chart
        def _load_chart() -> None:
            if angles_csv.exists():
                self.chart_panel.load_angles(str(angles_csv))

        # Step 3 — 600 ms: gait detection + video player init (heaviest pair)
        # Gait must come after chart so add_gait_regions paints over solid curves.
        # Video is last so QMediaPlayer gets a fully-flushed .mp4.
        def _load_gait_and_video() -> None:
            if angles_csv.exists():
                self._run_gait_detection(angles_csv)

            diagnostic_mp4 = out / "diagnostic.mp4"
            if diagnostic_mp4.exists():
                self.video_panel.load_video(str(diagnostic_mp4))
            else:
                src = self.control_panel.video_path()
                if src:
                    self.video_panel.load_video(src)

            # Re-enable Run button and reset status only after all loading done
            self.control_panel.set_running(False)
            self._set_status(f"Analysis complete  \u2014  {out_dir}")

        QTimer.singleShot(100, _load_metrics)
        QTimer.singleShot(300, _load_chart)
        QTimer.singleShot(600, _load_gait_and_video)

    def _run_gait_detection(self, angles_csv: Path) -> None:
        """
        Read the final knee angle column, run detect_gait_cycles, and pass
        the GaitCycles object to ChartPanel.add_gait_regions().
        """
        try:
            from app.utils.biomechanics import detect_gait_cycles
            df = pd.read_csv(angles_csv)
            if "time_s" not in df.columns:
                return

            t = df["time_s"].to_numpy(dtype=float)
            knee_col = next(
                (c for c in df.columns if "knee" in c.lower()), None
            )
            if knee_col is None:
                return

            knee_deg = df[knee_col].to_numpy(dtype=float)
            cycles   = detect_gait_cycles(knee_deg, t, fps=self._video_fps)
            self._last_gait_cycles = cycles
            self.chart_panel.add_gait_regions(cycles)

            if cycles.n_strides > 0:
                self._set_status(
                    f"Analysis complete  —  "
                    f"{cycles.n_strides} strides  ·  "
                    f"{cycles.mean_stride_s:.2f} s/stride  ·  "
                    f"{cycles.cadence_steps_per_min:.0f} steps/min"
                )
        except Exception as exc:
            # Gait detection is non-critical; don't block the UI
            print(f"Gait detection skipped: {exc}")

    def _on_error(self, message: str) -> None:
        self._disconnect_worker()
        if self._worker is not None:
            _w = self._worker
            self._worker = None
            QTimer.singleShot(0, lambda: (_w.wait(), _w.deleteLater()))
        self.control_panel.set_running(False)
        self._global_progress.setValue(0)
        self._set_status(f"Error: {message}")

    def _export_pdf(self) -> None:
        """Export a branded PDF report for the most recent analysis."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        if not self._last_out_dir:
            QMessageBox.warning(self, "No Analysis", "Run an analysis first.")
            return

        # Suggest a default filename inside the output directory
        default_path = str(Path(self._last_out_dir) / "reports" / "drocap_report.pdf")
        pdf_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", default_path,
            "PDF Files (*.pdf);;All Files (*)"
        )
        if not pdf_path:
            return

        self._set_status("Generating PDF report\u2026")
        try:
            from app.utils.report_gen import generate_report

            # Load metrics dict for the report
            metrics_json = Path(self._last_out_dir) / "reports" / "metrics_mocap.json"
            metrics_data: dict | None = None
            if metrics_json.exists():
                import json
                payload = json.loads(metrics_json.read_text())
                metrics_data = payload.get("metrics")

            out = generate_report(
                out_path         = pdf_path,
                chart_panel      = self.chart_panel,
                gait_cycles      = self._last_gait_cycles,
                metrics          = metrics_data,
                video_path       = self.control_panel.video_path(),
                athlete_height_m = None,
            )
            self._set_status(f"Report saved  \u2014  {out}")
            QMessageBox.information(
                self, "Report Saved",
                f"PDF report written to:\n{out}"
            )
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "reportlab is required for PDF export.\n\n"
                "Install it with:\n  pip install reportlab"
            )
            self._set_status("PDF export failed — reportlab not installed.")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            self._set_status(f"PDF export failed: {exc}")

    def _set_status(self, msg: str) -> None:
        self._status_msg.setText(msg)
