"""
QThread worker that runs the drone_mocap pipeline off the GUI thread.

Signals emitted:
    progress(frame_idx: int, total_frames: int, xy: np.ndarray, vis: np.ndarray)
        — after each pose-inference frame (Pass 1)
    finished(out_dir: str)
        — pipeline completed successfully
    error(message: str)
        — unhandled exception from run_pipeline
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    progress  = pyqtSignal(int, int, np.ndarray, np.ndarray)   # frame_idx, total, xy, vis
    finished  = pyqtSignal(str)                                  # out_dir path
    error     = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        out_root: str,
        mocap_path: str | None = None,
        visible_side: str = "right",
        cutoff_hz: float = 6.0,
        filter_order: int = 4,
        min_vis: float = 0.3,
        athlete_height_m: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_path        = video_path
        self.out_root          = out_root
        self.mocap_path        = mocap_path
        self.visible_side      = visible_side
        self.cutoff_hz         = cutoff_hz
        self.filter_order      = filter_order
        self.min_vis           = min_vis
        self.athlete_height_m  = athlete_height_m
        self._abort            = False

    def abort(self) -> None:
        self._abort = True

    def _on_progress(
        self,
        frame_idx: int,
        total_frames: int,
        _frame_bgr: np.ndarray,
        xy: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        if self._abort:
            raise InterruptedError("Aborted by user.")
        self.progress.emit(frame_idx, total_frames, xy.copy(), vis.copy())

    def run(self) -> None:
        try:
            from drone_mocap.pipeline.run import run_pipeline

            out_dir = run_pipeline(
                video            = Path(self.video_path),
                out_root         = Path(self.out_root),
                mocap_txt        = Path(self.mocap_path) if self.mocap_path else None,
                visible_side     = self.visible_side,
                cutoff_hz        = self.cutoff_hz,
                filter_order     = self.filter_order,
                min_vis          = self.min_vis,
                athlete_height_m = self.athlete_height_m,
                diagnostic_video = True,
                on_progress      = self._on_progress,
            )
            self.finished.emit(str(out_dir))
        except InterruptedError:
            self.error.emit("Analysis cancelled.")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
