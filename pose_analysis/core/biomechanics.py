import numpy as np
from typing import Dict, Tuple
from config import LANDMARK_IDX, LEG_LANDMARKS


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate angle between two vectors in degrees.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Angle in degrees (0-180)
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    v1_norm = float(np.linalg.norm(v1))
    v2_norm = float(np.linalg.norm(v2))
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    
    cos_theta = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def get_body_frame(leg_pts: Dict[str, np.ndarray], shoulder_pts: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Transform landmarks to body-centered coordinate system.
    
    Uses hip center as origin when hips are well-separated,
    falls back to shoulder center when hips are close (facing camera).
    
    Args:
        leg_pts: Leg landmark coordinates in pixels
        shoulder_pts: Shoulder landmark coordinates in pixels
    
    Returns:
        Tuple of (body_coords, hip_distance)
        - body_coords: Landmarks in body frame
        - hip_distance: Distance between hips in pixels
    """
    if "LEFT_HIP" not in leg_pts or "RIGHT_HIP" not in leg_pts:
        return {}, 0.0
    
    left_hip = leg_pts["LEFT_HIP"]
    right_hip = leg_pts["RIGHT_HIP"]
    hip_dist = float(np.linalg.norm(right_hip - left_hip))
    
    if hip_dist < 10.0 and "LEFT_SHOULDER" in shoulder_pts and "RIGHT_SHOULDER" in shoulder_pts:
        center = (shoulder_pts["LEFT_SHOULDER"] + shoulder_pts["RIGHT_SHOULDER"]) / 2.0
        x_vec = shoulder_pts["RIGHT_SHOULDER"] - shoulder_pts["LEFT_SHOULDER"]
    else:
        center = (left_hip + right_hip) / 2.0
        x_vec = right_hip - left_hip
    
    if np.linalg.norm(x_vec) < 1e-6:
        x_axis = np.array([1.0, 0.0], dtype=np.float32)
    else:
        x_axis = x_vec / np.linalg.norm(x_vec)
    
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
    
    body_coords = {}
    for name, point in leg_pts.items():
        rel = point - center
        body_coords[name] = np.array([
            float(np.dot(rel, x_axis)),
            float(np.dot(rel, y_axis))
        ], dtype=np.float32)
    
    return body_coords, hip_dist


def compute_joint_angles(body_coords: Dict[str, np.ndarray], 
                        hip_distance: float,
                        view_mode: str) -> Dict[str, float]:
    """
    Compute biomechanical angles based on view mode.
    
    Args:
        body_coords: Landmarks in body frame
        hip_distance: Hip separation in pixels
        view_mode: "frontal" or "sagittal"
    
    Returns:
        Dictionary of angle measurements (in degrees)
    """
    if not body_coords:
        return {}
    
    leg_lengths = []
    for side in ["LEFT", "RIGHT"]:
        hip_key = f"{side}_HIP"
        ankle_key = f"{side}_ANKLE"
        if hip_key in body_coords and ankle_key in body_coords:
            leg_len = np.linalg.norm(body_coords[ankle_key] - body_coords[hip_key])
            leg_lengths.append(leg_len)
    
    leg_length = float(np.mean(leg_lengths)) if leg_lengths else 50.0
    leg_length = max(leg_length, 1e-3)
    
    stability = min(1.0, hip_distance / leg_length)
    
    angles = {}
    
    if view_mode == "frontal":
        angles.update(_compute_frontal_angles(body_coords, stability))
    else:  # sagittal
        angles.update(_compute_sagittal_angles(body_coords, stability))
    
    return angles


def _compute_frontal_angles(coords: Dict[str, np.ndarray], 
                           stability: float) -> Dict[str, float]:
    """Compute adduction and valgus angles (frontal plane)."""
    angles = {}
    vertical = np.array([0.0, -1.0], dtype=np.float32)
    
    # Hip adduction (thigh angle from vertical)
    for side in ["LEFT", "RIGHT"]:
        hip = f"{side}_HIP"
        knee = f"{side}_KNEE"
        if hip in coords and knee in coords:
            thigh = coords[knee] - coords[hip]
            raw_angle = angle_between(thigh, vertical)
            angles[f"{side}_HIP_adduction"] = raw_angle * stability
    
    # Knee valgus (mechanical axis deviation)
    for side in ["LEFT", "RIGHT"]:
        hip = f"{side}_HIP"
        knee = f"{side}_KNEE"
        ankle = f"{side}_ANKLE"
        if {hip, knee, ankle} <= coords.keys():
            mech_axis = coords[ankle] - coords[hip]
            thigh_seg = coords[knee] - coords[hip]
            raw_angle = angle_between(mech_axis, thigh_seg)
            angles[f"{side}_KNEE_valgus"] = raw_angle * stability
    
    return angles


def _compute_sagittal_angles(coords: Dict[str, np.ndarray], 
                            stability: float) -> Dict[str, float]:
    """Compute flexion/extension angles (sagittal plane)."""
    angles = {}
    horizontal = np.array([1.0, 0.0], dtype=np.float32)
    
    # Hip flexion (thigh angle from horizontal)
    for side in ["LEFT", "RIGHT"]:
        hip = f"{side}_HIP"
        knee = f"{side}_KNEE"
        if hip in coords and knee in coords:
            thigh = coords[knee] - coords[hip]
            raw_angle = angle_between(thigh, horizontal)
            angles[f"{side}_HIP_flexion"] = raw_angle * stability
    
    # Knee flexion (angle between thigh and shank)
    for side in ["LEFT", "RIGHT"]:
        hip = f"{side}_HIP"
        knee = f"{side}_KNEE"
        ankle = f"{side}_ANKLE"
        if {hip, knee, ankle} <= coords.keys():
            thigh = coords[hip] - coords[knee]
            shank = coords[ankle] - coords[knee]
            raw_angle = angle_between(thigh, shank)
            angles[f"{side}_KNEE_flexion"] = raw_angle * stability
    
    # Ankle flexion (angle between foot and shank)
    for side in ["LEFT", "RIGHT"]:
        knee = f"{side}_KNEE"
        ankle = f"{side}_ANKLE"
        foot = f"{side}_FOOT_INDEX"
        if {knee, ankle, foot} <= coords.keys():
            foot_vec = coords[foot] - coords[ankle]
            shank_vec = coords[ankle] - coords[knee]
            raw_angle = angle_between(foot_vec, shank_vec)
            angles[f"{side}_ANKLE_flexion"] = raw_angle * stability
    
    return angles