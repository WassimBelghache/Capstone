from __future__ import annotations
import numpy as np
from drone_mocap.pose.mediapose import LM

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    # returns angle in degrees between vectors v1 and v2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def joint_angles_sagittal(xy: np.ndarray, vis: np.ndarray, visible_side: str = "right", min_vis: float = 0.5):
    """
    xy: (33,2) pixels for one frame
    vis: (33,) visibility
    visible_side: "right" or "left" (which side is facing camera)

    Returns dict with hip/knee/ankle angles in degrees (flex/ext style conventions can be adjusted later).
    For now:
      knee_angle: angle between thigh (hip->knee) and shank (ankle->knee). Full extension ~180.
      hip_angle: angle between trunk (shoulder->hip) and thigh (knee->hip). More flexion -> smaller? depends.
      ankle_angle: angle between shank (knee->ankle) and foot (foot->ankle). Neutral ~90-ish.
    """
    side = "R" if visible_side.lower().startswith("r") else "L"

    shoulder = LM[f"{side}_SHOULDER"]
    hip = LM[f"{side}_HIP"]
    knee = LM[f"{side}_KNEE"]
    ankle = LM[f"{side}_ANKLE"]
    foot = LM[f"{side}_FOOT"]
    heel = LM[f"{side}_HEEL"]

    needed = [shoulder, hip, knee, ankle, heel, foot]
    if any(vis[i] < min_vis or not np.all(np.isfinite(xy[i])) for i in needed):
        return {"hip": np.nan, "knee": np.nan, "ankle": np.nan}

    v_trunk = xy[shoulder] - xy[hip]      # hip -> shoulder (up)
    v_thigh = xy[knee] - xy[hip]          # hip -> knee
    v_shank = xy[ankle] - xy[knee]        # knee -> ankle
    
    # Foot segment: heel -> toe (more stable)
    v_foot = xy[foot] - xy[heel]

    # Shank at ankle: ankle -> knee
    v_shank_at_ankle = xy[knee] - xy[ankle]

    # Knee: angle between thigh (knee->hip) and shank (knee->ankle)
    knee_angle = _angle_between(xy[hip] - xy[knee], xy[ankle] - xy[knee])

    # Hip: angle between trunk (hip->shoulder) and thigh (hip->knee)
    hip_angle = _angle_between(v_trunk, v_thigh)
    
    ankle_angle = _angle_between(v_shank_at_ankle, v_foot)
    
    knee_flex = 180.0 - knee_angle

    hip_flex = 180.0 - hip_angle
    
    ankle_dorsi = 90.0 - ankle_angle

    return {"hip": hip_flex, "knee": knee_flex, "ankle": ankle_dorsi}
