import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
PoseLandmarker = mp_tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp_tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp_tasks.vision.RunningMode

class PoseDetector:
    """Wrapper for MediaPipe Pose Landmarker."""
    
    def __init__(self, model_path: str, mode: str = "video"):
        """
        Initialize pose detector.
        
        Args:
            model_path: Path to .task model file
            mode: "video" or "live"
        """
        from config import ProcessingConfig
        
        self.model_path = model_path
        
        running_mode = (VisionRunningMode.VIDEO if mode == "video" else VisionRunningMode.LIVE_STREAM)
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=ProcessingConfig.MIN_POSE_CONFIDENCE,
            min_pose_presence_confidence=ProcessingConfig.MIN_POSE_CONFIDENCE,
            min_tracking_confidence=ProcessingConfig.MIN_TRACKING_CONFIDENCE,
        )
        
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def detect(self, frame: np.ndarray, timestamp_ms: int):
        """
        Detect pose in frame.
        
        Args:
            frame: BGR image frame
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            Detection result with pose_landmarks
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()