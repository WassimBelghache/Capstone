from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

def savgol_smooth_xy(xy_series: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """
    xy_series: (T, J, 2) with NaNs allowed.
    Smooth each joint coordinate across time using Savitzky-Golay.
    NaNs are linearly interpolated before smoothing, then restored where entire joint is missing.
    """
    T, J, C = xy_series.shape
    out = xy_series.copy()

    for j in range(J):
        for c in range(C):
            y = out[:, j, c]
            mask = np.isfinite(y)
            if mask.sum() < max(5, poly + 2):
                continue

            # interpolate NaNs
            idx = np.arange(T)
            y_interp = y.copy()
            y_interp[~mask] = np.interp(idx[~mask], idx[mask], y[mask])

            # ensure window is odd and <= T
            win = min(window, T if T % 2 == 1 else T - 1)
            if win < poly + 2:
                continue
            if win % 2 == 0:
                win -= 1
            y_smooth = savgol_filter(y_interp, window_length=win, polyorder=poly)

            out[:, j, c] = y_smooth

    return out
