from typing import Dict, Optional
import numpy as np


class AngleStabilizer:
    """Smooth angles based on motion speed and hip stability."""
    
    def __init__(self, smoothing_factor: float = 0.7, history_size: int = 10):
        """
        Initialize stabilizer.
        
        Args:
            smoothing_factor: Base smoothing coefficient (0-1)
            history_size: Number of frames to keep in history
        """
        from config import ProcessingConfig
        
        self.base_smoothing = smoothing_factor
        self.history_size = history_size
        self.previous_angles: Optional[Dict[str, float]] = None
        self.angle_history: Dict[str, list] = {}
    
    def stabilize(self, new_angles: Dict[str, float], hip_distance: float, motion_speed: float) -> Dict[str, float]:
        """
        Apply motion-aware smoothing to angles.
        
        Args:
            new_angles: Raw computed angles
            hip_distance: Current hip separation (px)
            motion_speed: Average joint displacement (px/frame)
        
        Returns:
            Stabilized angles
        """
        from config import ProcessingConfig
        
        if self.previous_angles is None:
            self._initialize_history(new_angles)
            return new_angles
        
        # Trust factor based on hip stability
        trust = min(1.0, hip_distance / 30.0)
        
        # Adjust smoothing for motion speed
        if motion_speed > ProcessingConfig.MOTION_HIGH_SPEED_THRESHOLD:
            # Fast motion - more responsive
            smoothing = min(self.base_smoothing, 0.5)
        elif motion_speed < ProcessingConfig.MOTION_STATIC_THRESHOLD:
            # Slow/static - more stable
            smoothing = max(self.base_smoothing, 0.85)
        else:
            # Adaptive based on hip stability
            smoothing = self.base_smoothing * (1 - trust) + 0.9 * trust
        
        stabilized = {}
        for key, value in new_angles.items():
            # Update history
            if key not in self.angle_history:
                self.angle_history[key] = []
            self.angle_history[key].append(value)
            if len(self.angle_history[key]) > self.history_size:
                self.angle_history[key].pop(0)
            
            # Use median for very unstable conditions
            if hip_distance < ProcessingConfig.HIP_UNSTABLE_THRESHOLD:
                reference = float(np.median(self.angle_history[key]))
            else:
                reference = self.previous_angles.get(key, value)
            
            # Apply exponential smoothing
            stabilized[key] = smoothing * value + (1 - smoothing) * reference
        
        self.previous_angles = stabilized
        return stabilized
    
    def _initialize_history(self, angles: Dict[str, float]):
        """Initialize on first frame."""
        self.previous_angles = angles.copy()
        for key, value in angles.items():
            self.angle_history[key] = [value]
    
    def reset(self):
        """Reset state for new video."""
        self.previous_angles = None
        self.angle_history.clear()