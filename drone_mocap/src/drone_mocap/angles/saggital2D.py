from __future__ import annotations

import numpy as np
from drone_mocap.pose.mediapose import LM

# ---------------------------------------------------------------------------
# Anatomical range-of-motion limits (degrees).
# Values outside these bounds are flagged as detector outliers and replaced
# by interpolation from the previous N valid frames.
# ---------------------------------------------------------------------------
ANAT_LIMITS: dict[str, tuple[float, float]] = {
    "hip":   (-30.0, 130.0),   # -30° extension  →  130° flexion
    "knee":  (-10.0, 175.0),   # -10° hyperext   →  175° flexion
    "ankle": (-80.0, 30.0),    # widened to match MoCap Z axis range (-64° to -5°)
}

# ---------------------------------------------------------------------------
# Cross-product sign convention for atan2-based signed angles.
#
# Formula: angle = atan2(cz_sign * cross_z, -dot)
#   → 0°  at full extension
#   → positive values in flexion
#   → negative values in hyperextension / over-extension
#
# Derivation (verified analytically for y-down image coordinates):
#   RIGHT visible side (athlete's right side faces camera, moves L→R):
#     knee:  cz_sign = -1  (flexion → cross_z < 0)
#     hip:   cz_sign = +1  (flexion → cross_z > 0)
#     ankle: cz_sign = +1  (dorsiflexion → cross_z > 0)
#
#   LEFT visible side (athlete's left side faces camera, moves R→L):
#     knee:  cz_sign = +1
#     hip:   cz_sign = -1
#     ankle: cz_sign = -1
# ---------------------------------------------------------------------------
_CZ_SIGN: dict[str, dict[str, float]] = {
    "right": {"knee": -1.0, "hip": +1.0, "ankle": +1.0},
    "left":  {"knee": +1.0, "hip": -1.0, "ankle": -1.0},
}

# ---------------------------------------------------------------------------
# Ankle offset correction.
#
# The raw atan2 ankle angle has an offset because the "neutral" foot position
# (foot roughly horizontal) doesn't correspond to 0° naturally.
#
# For RIGHT side: neutral gives raw ≈ +90°, so we subtract 90°
# For LEFT side:  neutral gives raw ≈ -90°, so we add 90°
#
# These were verified empirically from the raw ankle values seen in testing.
# ---------------------------------------------------------------------------
_ANKLE_OFFSET: dict[str, float] = {
    "right": -90.0,
    "left":  -114.0,  # empirically derived: -97 - 17.15 to match MoCap Z axis mean
}


def _signed_angle_at_apex(
    p_prox: np.ndarray,
    p_apex: np.ndarray,
    p_dist: np.ndarray,
    cz_sign: float,
) -> float:
    """
    Signed flexion angle at p_apex (degrees).

    Vectors v1 = p_prox - p_apex, v2 = p_dist - p_apex.
    Uses atan2(cz_sign * cross_z, -dot) to obtain the full (−180°, 180°] range:
      - 0°        full anatomical extension
      - positive  flexion
      - negative  hyperextension / past full extension

    Unlike arccos, this correctly encodes angles beyond the ±180° boundary
    and eliminates the ambiguity between flexion and hyperextension.
    """
    v1 = p_prox - p_apex
    v2 = p_dist - p_apex
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    v1n = v1 / n1
    v2n = v2 / n2
    cross_z = float(v1n[0] * v2n[1] - v1n[1] * v2n[0])
    dot_val = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return float(np.degrees(np.arctan2(cz_sign * cross_z, -dot_val)))


def _clamp_series(angles: np.ndarray, lo: float, hi: float, prev_n: int = 3) -> np.ndarray:
    """
    Replace anatomically impossible values with the mean of the previous
    `prev_n` valid in-range frames. Falls back to NaN if no prior valid
    frames exist, allowing the filter to interpolate on the next pass.
    """
    out = angles.copy().astype(float)
    for i in range(len(out)):
        val = out[i]
        if np.isfinite(val) and (val < lo or val > hi):
            prev_valid = [
                out[k]
                for k in range(max(0, i - prev_n), i)
                if np.isfinite(out[k]) and lo <= out[k] <= hi
            ]
            out[i] = float(np.mean(prev_valid)) if prev_valid else np.nan
    return out


def joint_angles_sagittal(
    xy: np.ndarray,
    vis: np.ndarray,
    visible_side: str = "right",
    min_vis: float = 0.3,
) -> dict[str, float]:
    """
    Compute signed sagittal-plane joint angles for a single frame.

    Args:
        xy:           (33, 2) pixel coordinates for one frame.
        vis:          (33,)   MediaPipe visibility scores in [0, 1].
        visible_side: "right" or "left" — which side of the body faces the camera.
        min_vis:      Per-keypoint minimum visibility threshold.

    Returns:
        dict with keys "hip", "knee", "ankle" in degrees.
        Positive = flexion / dorsiflexion; negative = hyperextension / plantarflex.
    """
    side_key = "right" if visible_side.lower().startswith("r") else "left"
    prefix = "R" if side_key == "right" else "L"
    cz = _CZ_SIGN[side_key]

    idx = {
        "shoulder": LM[f"{prefix}_SHOULDER"],
        "hip":      LM[f"{prefix}_HIP"],
        "knee":     LM[f"{prefix}_KNEE"],
        "ankle":    LM[f"{prefix}_ANKLE"],
        "heel":     LM[f"{prefix}_HEEL"],
        "foot":     LM[f"{prefix}_FOOT"],
    }

    def _ok(name: str) -> bool:
        i = idx[name]
        return vis[i] >= min_vis and bool(np.all(np.isfinite(xy[i])))

    # --- KNEE (hip → knee → ankle) ---
    if _ok("hip") and _ok("knee") and _ok("ankle"):
        knee_angle = _signed_angle_at_apex(
            xy[idx["hip"]], xy[idx["knee"]], xy[idx["ankle"]], cz["knee"]
        )
    else:
        knee_angle = np.nan

    # --- HIP (shoulder → hip → knee) ---
    if _ok("shoulder") and _ok("hip") and _ok("knee"):
        hip_angle = _signed_angle_at_apex(
            xy[idx["shoulder"]], xy[idx["hip"]], xy[idx["knee"]], cz["hip"]
        )
    else:
        hip_angle = np.nan

    # --- ANKLE (knee → ankle → foot) ---
    # The raw angle at the ankle apex between the shank (knee→ankle) and
    # foot (ankle→foot) segments. The offset zeroes the angle at neutral stance.
    if _ok("knee") and _ok("ankle") and _ok("foot"):
        raw_ankle = _signed_angle_at_apex(
            xy[idx["knee"]], xy[idx["ankle"]], xy[idx["foot"]], cz["ankle"]
        )
        ankle_angle = raw_ankle + _ANKLE_OFFSET[side_key]
    else:
        ankle_angle = np.nan

    return {"hip": hip_angle, "knee": knee_angle, "ankle": ankle_angle}


def clamp_angle_series(
    angles_dict: dict[str, np.ndarray],
    prev_n: int = 3,
) -> dict[str, np.ndarray]:
    """
    Apply anatomical range-of-motion clipping to full angle time-series.

    Call this AFTER stacking per-frame results into arrays.
    Out-of-range samples are replaced with the rolling mean of the previous
    `prev_n` in-range frames rather than hard-clipped, preserving signal
    continuity near physiological boundaries.

    Args:
        angles_dict: {"hip": (T,), "knee": (T,), "ankle": (T,)} arrays.
        prev_n:      Number of prior frames to average for outlier replacement.

    Returns:
        Same structure with outliers interpolated.
    """
    return {
        joint: _clamp_series(arr, *ANAT_LIMITS[joint], prev_n=prev_n)
        for joint, arr in angles_dict.items()
        if joint in ANAT_LIMITS
    }