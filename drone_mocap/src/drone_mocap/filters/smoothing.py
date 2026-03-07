from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import CubicSpline


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
    # Guard: cutoff must be strictly below Nyquist
    cutoff_norm = float(np.clip(cutoff_hz / nyq, 1e-4, 0.99))
    sos = butter(order, cutoff_norm, btype="low", output="sos")

    t_idx = np.arange(T, dtype=float)

    # sosfiltfilt requires at least 3*(order+1) samples for padding
    min_samples_for_filter = 3 * (order + 1)

    for j in range(J):
        # High-confidence mask: vis >= min_vis AND both x,y are finite
        conf_mask = (vis_series[:, j] >= min_vis) & np.all(
            np.isfinite(xy_series[:, j, :]), axis=1
        )
        n_good = int(conf_mask.sum())

        # Need at least 4 anchor points for a cubic spline (k=3 requires k+1)
        if n_good < 4:
            continue

        # Track frames that are truly absent (no detection at all, regardless of vis)
        totally_absent = ~np.any(np.isfinite(xy_series[:, j, :]), axis=1)

        for c in range(C):
            y_raw = xy_series[:, j, c]

            if n_good < T:
                # --- Confidence-weighted cubic spline gap-filling ---
                t_good = t_idx[conf_mask]
                y_good = y_raw[conf_mask]
                cs = CubicSpline(t_good, y_good, extrapolate=True)
                y_interp = cs(t_idx)
            else:
                y_interp = y_raw.copy()

            if T < min_samples_for_filter:
                # Too short to filter safely — keep interpolated values
                out[:, j, c] = y_interp
                out[totally_absent, j, c] = np.nan
                continue

            # --- Zero-phase dual-pass Butterworth ---
            y_smooth = sosfiltfilt(sos, y_interp)

            out[:, j, c] = y_smooth
            # Restore NaN at frames where the joint was never detected at all
            out[totally_absent, j, c] = np.nan

    return out
