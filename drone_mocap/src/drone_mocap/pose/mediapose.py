from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
import mediapipe as mp

LM = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANKLE": 27, "R_ANKLE": 28,
    "L_HEEL": 29, "R_HEEL": 30,
    "L_FOOT": 31, "R_FOOT": 32,
}

@dataclass
class PoseResult:
    # shape: (33, 2) in pixel coords, NaN if missing
    xy: np.ndarray
    # shape: (33,) visibility/confidence in [0,1], 0 if missing
    vis: np.ndarray

class MediaPipePoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            smooth_landmarks=False,  # we do our own smoothing
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def predict(self, frame_bgr: np.ndarray) -> PoseResult:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(frame_rgb)

        xy = np.full((33, 2), np.nan, dtype=np.float32)
        vis = np.zeros((33,), dtype=np.float32)

        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                xy[i, 0] = lm.x * w
                xy[i, 1] = lm.y * h
                vis[i] = float(lm.visibility) if lm.visibility is not None else 1.0

        return PoseResult(xy=xy, vis=vis)

    def close(self):
        self.pose.close()
