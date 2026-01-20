from typing import List, Tuple


LANDMARK_IDX = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

LEG_LANDMARKS = [
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX",
    "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"
]

SHOULDER_LANDMARKS = ["LEFT_SHOULDER", "RIGHT_SHOULDER"]

BONE_CONNECTIONS: List[Tuple[str, str]] = [
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
]


FRONTAL_ANGLES = [
    "LEFT_HIP_adduction", 
    "RIGHT_HIP_adduction", 
    "LEFT_KNEE_valgus", 
    "RIGHT_KNEE_valgus"
]

SAGITTAL_ANGLES = [
    "LEFT_HIP_flexion", 
    "RIGHT_HIP_flexion", 
    "LEFT_KNEE_flexion", 
    "RIGHT_KNEE_flexion",
    "LEFT_ANKLE_flexion", 
    "RIGHT_ANKLE_flexion"
]


VIEW_MODES = ["frontal", "sagittal"]


def get_csv_header() -> List[str]:
    """Generate CSV header with all data columns."""
    cols = ["frame", "timestamp_ms", "hip_dist_px", "motion_px", "state"]
    

    for landmark in LEG_LANDMARKS:
        cols.extend([f"{landmark}_x", f"{landmark}_y"])
    

    cols.extend(FRONTAL_ANGLES)
    cols.extend(SAGITTAL_ANGLES)
    
    return cols


# Processing parameters
class ProcessingConfig:
    """Configuration for pose processing."""
    MIN_POSE_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    

    MOTION_STATIC_THRESHOLD = 2.0
    MOTION_HIGH_SPEED_THRESHOLD = 10.0
    

    HIP_CLOSE_THRESHOLD = 10.0
    HIP_UNSTABLE_THRESHOLD = 20.0
    

    SMOOTHING_FACTOR = 0.7
    HISTORY_SIZE = 10
    GRAPH_SMOOTHING_WINDOW = 5
    

    DEFAULT_FPS = 30.0
    PROGRESS_UPDATE_INTERVAL = 10 