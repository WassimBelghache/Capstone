from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter


def butterworth_smooth_xy(
    xy_series: np.ndarray,
    vis_series: np.ndarray,
    fps: float,
    cutoff_hz: float = 6.0,
    order: int = 4,
    min_vis: float = 0.3,
) -> np.ndarray:
    """
    Zero-phase dual-pass Butterworth filter with confidence-weighted cubic spline interpolation.

    Pipeline per joint coordinate:
      1. Identify high-confidence frames (vis >= min_vis AND finite coords).
      2. Fit a CubicSpline through high-confidence anchor points to fill gaps
         caused by occlusions or low-confidence detections.
      3. Apply sosfiltfilt (zero-phase, forward+backward Butterworth) using a
         cutoff defined in Hz — automatically normalized to the video's Nyquist.
      4. Restore NaN at frames that were entirely missing (no signal at all).

    Args:
        xy_series:  (T, J, 2)  pixel coordinates; NaN where keypoint not detected.
        vis_series: (T, J)     MediaPipe visibility scores in [0, 1].
        fps:        Video frame rate in Hz — used to compute normalized cutoff.
        cutoff_hz:  Low-pass cutoff in Hz.
                      ~6 Hz   for walking / slow movement
                      ~10 Hz  for jogging
                      ~12 Hz  for sprinting / high-velocity athletics
        order:      Butterworth filter order. 4 is standard; higher → sharper rolloff.
        min_vis:    Frames with vis < min_vis are treated as unreliable and excluded
                    from spline anchors (their positions are interpolated instead).

    Returns:
        (T, J, 2)  smoothed pixel coordinates.
    """
    T, J, C = xy_series.shape
    out = xy_series.copy()

    nyq = fps / 2.0
    cutoff_norm = float(np.clip(cutoff_hz / nyq, 1e-4, 0.99))
    sos = butter(order, cutoff_norm, btype="low", output="sos")

    t_idx = np.arange(T, dtype=float)

    min_samples_for_filter = 3 * (order + 1)

    for j in range(J):
        conf_mask = (vis_series[:, j] >= min_vis) & np.all(
            np.isfinite(xy_series[:, j, :]), axis=1
        )
        n_good = int(conf_mask.sum())

        if n_good < 4:
            continue

        totally_absent = ~np.any(np.isfinite(xy_series[:, j, :]), axis=1)

        for c in range(C):
            y_raw = xy_series[:, j, c]

            if n_good < T:
                t_good = t_idx[conf_mask]
                y_good = y_raw[conf_mask]
                cs = CubicSpline(t_good, y_good, extrapolate=True)
                y_interp = cs(t_idx)
            else:
                y_interp = y_raw.copy()

            if T < min_samples_for_filter:
                out[:, j, c] = y_interp
                out[totally_absent, j, c] = np.nan
                continue

            y_smooth = sosfiltfilt(sos, y_interp)

            out[:, j, c] = y_smooth
            out[totally_absent, j, c] = np.nan

    return out


def median_smooth_angles(
    angles_dict: dict[str, np.ndarray],
    kernel_sizes: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Apply joint-specific median filtering to angle time-series.

    Median filtering is particularly effective for removing impulse noise
    (sudden spikes) from keypoint jitter without distorting the underlying
    waveform shape. It is applied AFTER Butterworth filtering and anatomical
    clamping.

    The ankle benefits most from this because the foot/heel keypoints are
    small and fast-moving, causing occasional large spike errors that the
    Butterworth filter alone cannot remove.

    Args:
        angles_dict:  {"hip": (T,), "knee": (T,), "ankle": (T,)} arrays.
        kernel_sizes: Per-joint median filter kernel size (must be odd).
                      Defaults:
                        hip:   3  (light smoothing, already clean)
                        knee:  3  (light smoothing, already clean)
                        ankle: 13 (stronger smoothing, empirically derived
                                   to maximise correlation with MoCap Z axis)

    Returns:
        Same structure with median filtering applied.
    """
    if kernel_sizes is None:
        kernel_sizes = {
            "hip":   3,
            "knee":  3,
            "ankle": 13,
        }

    out: dict[str, np.ndarray] = {}
    for joint, arr in angles_dict.items():
        kernel = kernel_sizes.get(joint, 1)
        if kernel > 1:
            # Replace NaNs temporarily for median filter then restore
            finite_mask = np.isfinite(arr)
            if finite_mask.sum() < kernel:
                out[joint] = arr.copy()
                continue
            arr_filled = arr.copy()
            # Fill NaNs with nearest valid value before filtering
            if not finite_mask.all():
                idx = np.arange(len(arr))
                valid_idx = idx[finite_mask]
                arr_filled = np.interp(idx, valid_idx, arr[finite_mask])
            smoothed = median_filter(arr_filled, size=kernel)
            # Restore NaNs
            smoothed[~finite_mask] = np.nan
            out[joint] = smoothed
        else:
            out[joint] = arr.copy()

    return out