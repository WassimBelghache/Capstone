"""
QThread worker that runs the drone_mocap pipeline off the GUI thread.

Segfault prevention
-------------------
MediaPipe stores landmarks in Protobuf objects backed by C++ memory.
`pr.xy` / `pr.vis` are already NumPy arrays (created fresh in predict()),
but to be absolutely safe we re-copy as float64 and pass only plain Python
objects (dicts, numpy arrays, ints) across the thread boundary.  No
MediaPipe or Protobuf objects ever touch the Qt signal.

Signals
-------
progress(frame_idx, total_frames, rgb_small, xy, vis, angles)
    Emitted every _EMIT_STRIDE source frames.
    rgb_small : (H', W', 3) uint8 RGB — pre-downsampled ≤480px wide
    xy        : (33, 2) float64 — pixel coords, plain NumPy
    vis       : (33,)   float64 — visibility scores, plain NumPy
    angles    : dict[str, float] {"hip":…, "knee":…, "ankle":…} or {}

finished(out_dir: str)
error(message: str)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Emit a live-preview update every N source frames.
# At ~15 fps MediaPipe throughput, stride=6 gives ~2.5 UI updates/sec —
# fast enough to look live, slow enough to keep the event loop free.
_EMIT_STRIDE = 6

# Downscale preview to this width before sending over the signal.
_PREVIEW_MAX_W = 480


def _resize_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 → RGB uint8 and downsample to ≤_PREVIEW_MAX_W wide.
    All processing happens in the worker thread.
    Returns a contiguous uint8 RGB array.
    """
    h, w = bgr.shape[:2]
    # BGR → RGB (no allocation: just reverse last axis view, then make contiguous)
    rgb = bgr[:, :, ::-1]
    if w > _PREVIEW_MAX_W:
        step_x = max(1, w  // _PREVIEW_MAX_W)
        step_y = max(1, h  // (h * _PREVIEW_MAX_W // w))
        rgb = rgb[::step_y, ::step_x, :]
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _safe_angles(xy: np.ndarray, vis: np.ndarray, side: str) -> dict:
    """
    Compute hip/knee/ankle angles in the worker thread.
    Returns a plain Python dict — no MediaPipe objects.
    """
    try:
        from drone_mocap.angles.saggital2D import joint_angles_sagittal
        result = joint_angles_sagittal(xy, vis, visible_side=side, min_vis=0.2)
        # Materialise to plain Python floats so nothing exotic crosses threads
        return {k: float(v) for k, v in result.items()}
    except Exception:
        return {}


class AnalysisWorker(QThread):
    progress = pyqtSignal(int, int, np.ndarray, np.ndarray, np.ndarray, object)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

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
        self.video_path       = video_path
        self.out_root         = out_root
        self.mocap_path       = mocap_path
        self.visible_side     = visible_side
        self.cutoff_hz        = cutoff_hz
        self.filter_order     = filter_order
        self.min_vis          = min_vis
        self.athlete_height_m = athlete_height_m
        self._abort           = False

    def abort(self) -> None:
        self._abort = True

    def _on_progress(
        self,
        frame_idx: int,
        total_frames: int,
        frame_bgr: np.ndarray,
        xy: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        if self._abort:
            raise InterruptedError("Aborted by user.")

        if frame_idx % _EMIT_STRIDE != 0:
            return

        # ── Materialise as plain float64 NumPy — severs any MediaPipe memory ──
        xy_safe  = np.array(xy,  dtype=np.float64)   # (33, 2)
        vis_safe = np.array(vis, dtype=np.float64)   # (33,)

        rgb_small = _resize_to_rgb(frame_bgr)
        angles    = _safe_angles(xy_safe, vis_safe, self.visible_side)

        self.progress.emit(
            int(frame_idx),
            int(total_frames),
            rgb_small,
            xy_safe,
            vis_safe,
            angles,
        )

    def run(self) -> None:
        try:
            import gc
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

            # ── Clean-exit buffer ────────────────────────────────────────────
            # run_pipeline has returned; all cv2 captures/writers are already
            # released inside the pipeline.  gc.collect() forces any lingering
            # reference-counted C-extension objects (OpenCV, MediaPipe) to be
            # finalized before we signal the UI — this flushes OS write buffers
            # for the .mp4 and .csv so QMediaPlayer / pandas won't see partial files.
            gc.collect()

            # msleep gives Qt's event loop a chance to drain any queued
            # progress signals that arrived just before the pipeline returned.
            # Using QThread.msleep (not time.sleep) keeps the sleep cooperative
            # with the Qt event system and avoids blocking the OS scheduler.
            self.msleep(500)

            self.finished.emit(str(out_dir))

        except InterruptedError:
            self.error.emit("Analysis cancelled.")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
