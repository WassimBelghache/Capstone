"""
Gait cycle detection from kinematic waveforms.

Public API
----------
detect_gait_cycles(knee_angles, time_s, fps) -> GaitCycles
    Identifies heel-strikes (knee minima) and swing peaks (knee maxima)
    from the smoothed knee angle time-series.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.signal import find_peaks
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


@dataclass
class GaitCycles:
    """Container for detected gait events."""
    heel_strike_times: np.ndarray   # seconds — local minima (knee near extension)
    swing_peak_times:  np.ndarray   # seconds — local maxima (knee in flexion)
    stride_durations:  np.ndarray   # seconds — diff between consecutive heel-strikes

    @property
    def n_strides(self) -> int:
        return max(0, len(self.heel_strike_times) - 1)

    @property
    def mean_stride_s(self) -> float:
        if len(self.stride_durations) == 0:
            return float("nan")
        return float(np.nanmean(self.stride_durations))

    @property
    def cadence_steps_per_min(self) -> float:
        if self.mean_stride_s <= 0 or np.isnan(self.mean_stride_s):
            return float("nan")
        # One stride = two steps; 60 s/min
        return 60.0 / self.mean_stride_s * 2.0


_EMPTY = GaitCycles(
    heel_strike_times=np.array([], dtype=float),
    swing_peak_times =np.array([], dtype=float),
    stride_durations =np.array([], dtype=float),
)


def detect_gait_cycles(
    knee_angles: np.ndarray,
    time_s:      np.ndarray,
    fps:         float = 30.0,
) -> GaitCycles:
    """
    Identify heel-strikes and swing peaks from a smoothed knee angle series.

    Heel-strikes  — local minima of knee angle (leg approaching full extension
                    at initial contact); anatomically constrained to < 30°.
    Swing peaks   — local maxima of knee angle (peak flexion during swing);
                    anatomically constrained to > 30°.

    Parameters
    ----------
    knee_angles : (T,) float array — knee angle in degrees (smoothed).
    time_s      : (T,) float array — corresponding time in seconds.
    fps         : video frame-rate in Hz — used to set minimum inter-peak
                  distance (0.5 s = half a typical stride).

    Returns
    -------
    GaitCycles dataclass; returns empty arrays if scipy is unavailable or
    the signal is too short / has insufficient quality.
    """
    if not _SCIPY_OK:
        return _EMPTY

    if len(knee_angles) < 20 or len(knee_angles) != len(time_s):
        return _EMPTY

    finite = np.isfinite(knee_angles)
    if finite.sum() < 20:
        return _EMPTY

    # Fill NaNs with linear interpolation so find_peaks works uninterrupted
    idx = np.arange(len(knee_angles), dtype=float)
    valid_idx = idx[finite]
    filled = np.interp(idx, valid_idx, knee_angles[finite])

    # Minimum inter-peak distance: 0.5 s — handles strides up to 120 steps/min
    min_dist = max(5, int(fps * 0.5))

    # ── Heel-strikes: minima of knee angle ───────────────────────────────────
    hs_idx, _ = find_peaks(
        -filled,                   # invert to find minima as peaks
        distance=min_dist,
        prominence=5.0,            # ignore trivial wobbles
    )
    # Anatomical gate: knee near extension at heel-strike
    hs_idx = hs_idx[filled[hs_idx] < 30.0]

    # ── Swing peaks: maxima of knee angle ────────────────────────────────────
    sw_idx, _ = find_peaks(
        filled,
        distance=min_dist,
        prominence=5.0,
    )
    # Anatomical gate: meaningful knee flexion
    sw_idx = sw_idx[filled[sw_idx] > 30.0]

    hs_times = time_s[hs_idx]
    sw_times = time_s[sw_idx]
    stride_durations = np.diff(hs_times) if len(hs_times) > 1 else np.array([], dtype=float)

    return GaitCycles(
        heel_strike_times=hs_times,
        swing_peak_times =sw_times,
        stride_durations =stride_durations,
    )
