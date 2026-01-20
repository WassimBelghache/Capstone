import csv
from pathlib import Path
from typing import Dict, Optional
from PyQt5 import QtCore


class AnalysisWorker(QtCore.QThread):
    """Worker thread for Pass 1: full video analysis with pose detection."""
    
    progress = QtCore.pyqtSignal(int, int, str)  # current, total, status_msg
    finished = QtCore.pyqtSignal(str)            # csv_path
    failed = QtCore.pyqtSignal(str)              # error_message
    
    def __init__(self, 
                 video_path: str, 
                 model_path: str, 
                 view_mode: str,
                 output_csv: Optional[str] = None):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.view_mode = view_mode
        self.output_csv = output_csv or str(Path(video_path).with_suffix(".angles.csv"))
        self._should_stop = False
    
    def stop(self):
        """Request worker to stop processing."""
        self._should_stop = True
    
    def run(self):
        """Main analysis loop."""
        from core.pose_detector import PoseDetector
        from core.biomechanics import compute_joint_angles, get_body_frame
        from core.video_utils import fix_rotation, to_pixel_coords
        from config import (
            get_csv_header, 
            ProcessingConfig, 
            LANDMARK_IDX, 
            LEG_LANDMARKS, 
            SHOULDER_LANDMARKS
        )
        
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.video_path}")
            
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or ProcessingConfig.DEFAULT_FPS
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = -1
            
            self.progress.emit(0, total_frames, "Initializing pose detector...")
            
            with PoseDetector(self.model_path, mode="video") as detector:
                with open(self.output_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=get_csv_header())
                    writer.writeheader()
                    
                    prev_leg_pts: Optional[Dict[str, np.ndarray]] = None
                    frame_idx = 0
                    
                    while not self._should_stop:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame = fix_rotation(frame, cap)
                        h, w = frame.shape[:2]
                        
                        # Detect pose
                        timestamp_ms = int((frame_idx / fps) * 1000.0)
                        result = detector.detect(frame, timestamp_ms)
                        
                        # Prepare CSV row
                        row = self._create_empty_row(frame_idx, timestamp_ms)
                        
                        # Process landmarks if detected
                        if result.pose_landmarks and len(result.pose_landmarks) > 0:
                            landmarks = result.pose_landmarks[0]
                            
                            # Extract coordinates
                            leg_pts = {
                                name: to_pixel_coords(landmarks[LANDMARK_IDX[name]], w, h)
                                for name in LEG_LANDMARKS
                            }
                            shoulder_pts = {
                                name: to_pixel_coords(landmarks[LANDMARK_IDX[name]], w, h)
                                for name in SHOULDER_LANDMARKS
                            }
                            
                            # Calculate motion
                            motion = self._calculate_motion(leg_pts, prev_leg_pts)
                            prev_leg_pts = {k: v.copy() for k, v in leg_pts.items()}
                            
                            # Compute angles
                            body_coords, hip_dist = get_body_frame(leg_pts, shoulder_pts)
                            angles = compute_joint_angles(body_coords, hip_dist, self.view_mode)
                            
                            # Fill row
                            self._fill_row_data(row, leg_pts, angles, hip_dist, motion)
                        
                        writer.writerow(row)
                        
                        # Update progress
                        if frame_idx % ProcessingConfig.PROGRESS_UPDATE_INTERVAL == 0:
                            status = f"Analyzing frame {frame_idx}"
                            if total_frames > 0:
                                status += f" of {total_frames}"
                            self.progress.emit(frame_idx, total_frames, status)
                        
                        frame_idx += 1
            
            if cap:
                cap.release()
            
            if self._should_stop:
                self.failed.emit("Analysis cancelled by user")
            else:
                self.progress.emit(frame_idx, total_frames, "Analysis complete!")
                self.finished.emit(self.output_csv)
        
        except Exception as e:
            if cap:
                cap.release()
            self.failed.emit(f"Analysis error: {str(e)}")
    
    def _create_empty_row(self, frame_idx: int, timestamp_ms: int) -> Dict[str, str]:
        """Create empty CSV row with frame metadata."""
        from config import get_csv_header
        row = {col: "" for col in get_csv_header()}
        row["frame"] = str(frame_idx)
        row["timestamp_ms"] = str(timestamp_ms)
        return row
    
    def _calculate_motion(self, 
                         current: Dict[str, np.ndarray],
                         previous: Optional[Dict[str, np.ndarray]]) -> float:
        """Calculate average motion between frames."""
        if not previous or not current:
            return 0.0
        
        total_motion = 0.0
        count = 0
        for key, pt in current.items():
            if key in previous:
                total_motion += float(np.linalg.norm(pt - previous[key]))
                count += 1
        
        return total_motion / count if count > 0 else 0.0
    
    def _fill_row_data(self,
                       row: Dict[str, str],
                       landmarks: Dict[str, np.ndarray],
                       angles: Dict[str, float],
                       hip_distance: float,
                       motion: float):
        """Fill CSV row with detection data."""
        from config import ProcessingConfig
        
        # Landmark coordinates
        for name, pt in landmarks.items():
            row[f"{name}_x"] = f"{pt[0]:.2f}"
            row[f"{name}_y"] = f"{pt[1]:.2f}"
        
        # Measurements
        row["hip_dist_px"] = f"{hip_distance:.2f}"
        row["motion_px"] = f"{motion:.2f}"
        row["state"] = "static" if motion < ProcessingConfig.MOTION_STATIC_THRESHOLD else "moving"
        
        # Angles
        for angle_name, value in angles.items():
            row[angle_name] = f"{value:.2f}"