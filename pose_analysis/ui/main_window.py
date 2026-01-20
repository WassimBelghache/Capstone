import os
import csv
from pathlib import Path
from typing import Dict, Optional
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from processing.analyzer import AnalysisWorker
from processing.stabilizer import AngleStabilizer
from core.pose_detector import PoseDetector
from core.biomechanics import (
    compute_joint_angles, 
    get_body_frame, 
    LANDMARK_IDX, 
    LEG_LANDMARKS, 
    SHOULDER_LANDMARKS
)
from core.video_utils import fix_rotation, to_pixel_coords
from ui.plot_widgets import HipAngleCanvas, KneeAngleCanvas, AnkleAngleCanvas
from config import (
    VIEW_MODES, 
    FRONTAL_ANGLES, 
    SAGITTAL_ANGLES, 
    BONE_CONNECTIONS,
    ProcessingConfig
)


class PoseAnalysisGUI(QtWidgets.QWidget):
    """Main application with two-pass analysis and real-time modes."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lower-Limb Motion Capture Analysis")
        self.resize(1400, 800)
        
        # File paths
        self.video_path: Optional[str] = None
        self.model_path: Optional[str] = None
        self.csv_path: Optional[str] = None
        self.view_mode = "frontal"
        self.processing_mode = "realtime"
        
        # Playback state
        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._playback_tick)
        self.current_frame_idx = 0
        self.current_frame_data: Optional[np.ndarray] = None
        self.csv_data: Dict[int, dict] = {}
        self.source_fps = 30.0
        
        # Real-time processing
        self.detector: Optional[PoseDetector] = None
        self.stabilizer = AngleStabilizer()
        self.prev_landmarks: Optional[Dict[str, np.ndarray]] = None
        
        # Worker thread
        self.analysis_worker: Optional[AnalysisWorker] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the user interface."""
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # Left panel
        left_panel = QtWidgets.QVBoxLayout()
        
        # Video display
        self.video_label = QtWidgets.QLabel("No video loaded")
        self.video_label.setMinimumSize(720, 540)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #555; background: #222; color: #aaa;")
        left_panel.addWidget(self.video_label, 1)
        
        # Controls
        left_panel.addLayout(self._create_controls())
        
        # Status
        self.status_label = QtWidgets.QLabel("Ready. Select video and model to begin.")
        self.status_label.setStyleSheet("padding: 5px; background: #333; color: #0f0;")
        left_panel.addWidget(self.status_label)
        
        # Progress
        self.progress_bar = QtWidgets.QProgressBar()
        left_panel.addWidget(self.progress_bar)
        
        # Right panel - graphs
        right_panel = self._create_graphs()
        
        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 2)
    
    def _create_controls(self) -> QtWidgets.QVBoxLayout:
        """Create control panel."""
        layout = QtWidgets.QVBoxLayout()
        
        # File selection
        file_row = QtWidgets.QHBoxLayout()
        self.video_btn = QtWidgets.QPushButton("📁 Video")
        self.video_btn.clicked.connect(self._select_video)
        file_row.addWidget(self.video_btn)
        
        self.model_btn = QtWidgets.QPushButton("🤖 Model")
        self.model_btn.clicked.connect(self._select_model)
        file_row.addWidget(self.model_btn)
        layout.addLayout(file_row)
        
        # Settings
        settings_row = QtWidgets.QHBoxLayout()
        settings_row.addWidget(QtWidgets.QLabel("View:"))
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(VIEW_MODES)
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        settings_row.addWidget(self.view_combo)
        
        settings_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Real-time + Graphs", "Fast Playback (CSV)"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        settings_row.addWidget(self.mode_combo)
        settings_row.addStretch()
        layout.addLayout(settings_row)
        
        # Actions
        action_row = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton("⚡ Analyze")
        self.analyze_btn.clicked.connect(self._start_analysis)
        action_row.addWidget(self.analyze_btn)
        
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._start_playback)
        action_row.addWidget(self.play_btn)
        
        self.stop_btn = QtWidgets.QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_all)
        action_row.addWidget(self.stop_btn)
        
        layout.addLayout(action_row)
        return layout
    
    def _create_graphs(self) -> QtWidgets.QVBoxLayout:
        """Create graph panel."""
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("📊 Angle Tracking"))
        
        self.hip_canvas = HipAngleCanvas(self)
        self.hip_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.hip_canvas, 1)
        
        self.knee_canvas = KneeAngleCanvas(self)
        self.knee_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.knee_canvas, 1)
        
        self.ankle_canvas = AnkleAngleCanvas(self)
        self.ankle_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.ankle_canvas.setVisible(False)
        layout.addWidget(self.ankle_canvas, 1)
        
        return layout
    
    # Event handlers
    def _select_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_path = path
            self.csv_path = str(Path(path).with_suffix(".angles.csv"))
            self.status_label.setText(f"📁 {os.path.basename(path)}")
    
    def _select_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", "MediaPipe (*.task)"
        )
        if path:
            self.model_path = path
            self.status_label.setText(f"🤖 {os.path.basename(path)}")
    
    def _on_view_changed(self, view: str):
        self.view_mode = view
        self.ankle_canvas.setVisible(view == "sagittal")
    
    def _on_mode_changed(self, mode: str):
        self.processing_mode = "realtime" if "Real-time" in mode else "playback"
    
    # Analysis (Pass 1)
    def _start_analysis(self):
        if not self.video_path or not self.model_path:
            QtWidgets.QMessageBox.warning(self, "Missing Input", "Select video and model first.")
            return
        
        self._stop_all()
        self.analyze_btn.setEnabled(False)
        
        self.analysis_worker = AnalysisWorker(
            self.video_path, self.model_path, self.view_mode, self.csv_path
        )
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.failed.connect(self._on_analysis_failed)
        self.analysis_worker.start()
    
    def _on_analysis_progress(self, current: int, total: int, msg: str):
        self.status_label.setText(msg)
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
    
    def _on_analysis_finished(self, csv_path: str):
        self.csv_path = csv_path
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"✓ Done: {os.path.basename(csv_path)}")
    
    def _on_analysis_failed(self, error: str):
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("✗ Analysis failed")
        QtWidgets.QMessageBox.critical(self, "Error", error)
    
    # Playback
    def _start_playback(self):
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "No Video", "Select a video first.")
            return
        
        if self.processing_mode == "playback":
            self._start_fast_playback()
        else:
            self._start_realtime_playback()
    
    def _start_realtime_playback(self):
        """Real-time with pose detection."""
        if not self.model_path:
            QtWidgets.QMessageBox.warning(self, "No Model", "Select model first.")
            return
        
        self._stop_all()
        
        try:
            self.detector = PoseDetector(self.model_path)
            self.stabilizer.reset()
            self.prev_landmarks = None
            
            self.cap = cv2.VideoCapture(self.video_path)
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.current_frame_idx = 0
            
            self.timer.start(int(1000 / self.source_fps))
            self.status_label.setText(f"▶ Real-time @ {self.source_fps:.1f} FPS")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
    
    def _start_fast_playback(self):
        """Fast playback from CSV."""
        if not self.csv_path or not os.path.exists(self.csv_path):
            QtWidgets.QMessageBox.warning(self, "No Data", "Run analysis first.")
            return
        
        self._stop_all()
        self._load_csv()
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame_idx = 0
        
        self.timer.start(int(1000 / self.source_fps))
        self.status_label.setText(f"▶ Playback @ {self.source_fps:.1f} FPS")
    
    def _playback_tick(self):
        """Process one frame."""
        if not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self._stop_all()
            self.status_label.setText("⏹ Finished")
            return
        
        frame = fix_rotation(frame, self.cap)
        
        if self.processing_mode == "realtime":
            self._process_frame_realtime(frame)
        else:
            self._process_frame_playback(frame)
        
        self.current_frame_idx += 1
    
    def _process_frame_realtime(self, frame: np.ndarray):
        """Real-time pose detection and angle calculation."""
        h, w = frame.shape[:2]
        ts_ms = int((self.current_frame_idx / self.source_fps) * 1000)
        
        result = self.detector.detect(frame, ts_ms)
        
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lms = result.pose_landmarks[0]
            
            # Extract landmarks
            leg_pts = {name: to_pixel_coords(lms[LANDMARK_IDX[name]], w, h) 
                      for name in LEG_LANDMARKS}
            shoulder_pts = {name: to_pixel_coords(lms[LANDMARK_IDX[name]], w, h)
                           for name in SHOULDER_LANDMARKS}
            
            # Calculate motion
            motion = self._calc_motion(leg_pts)
            
            # Compute angles
            body_coords, hip_dist = get_body_frame(leg_pts, shoulder_pts)
            angles = compute_joint_angles(body_coords, hip_dist, self.view_mode)
            angles = self.stabilizer.stabilize(angles, hip_dist, motion)
            
            # Draw overlay
            self._draw_skeleton(frame, leg_pts, hip_dist, motion, angles)
            
            # Update graphs
            self._update_graphs(angles)
        else:
            cv2.putText(frame, "No pose detected", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        self._display_frame(frame)
    
    def _process_frame_playback(self, frame: np.ndarray):
        """Fast playback using CSV data."""
        row = self.csv_data.get(self.current_frame_idx)
        if row:
            # Draw from CSV
            leg_pts = {}
            for name in LEG_LANDMARKS:
                x = row.get(f"{name}_x")
                y = row.get(f"{name}_y")
                if x and y:
                    leg_pts[name] = np.array([float(x), float(y)])
            
            hip_dist = float(row.get("hip_dist_px", 0))
            motion = float(row.get("motion_px", 0))
            
            angles = {}
            angle_keys = FRONTAL_ANGLES if self.view_mode == "frontal" else SAGITTAL_ANGLES
            for key in angle_keys:
                val = row.get(key)
                if val:
                    angles[key] = float(val)
            
            self._draw_skeleton(frame, leg_pts, hip_dist, motion, angles)
            self._update_graphs(angles)
        
        self._display_frame(frame)
    
    def _calc_motion(self, landmarks: Dict[str, np.ndarray]) -> float:
        """Calculate motion between frames."""
        if not self.prev_landmarks:
            self.prev_landmarks = {k: v.copy() for k, v in landmarks.items()}
            return 0.0
        
        total = sum(np.linalg.norm(landmarks[k] - self.prev_landmarks[k])
                   for k in landmarks if k in self.prev_landmarks)
        
        self.prev_landmarks = {k: v.copy() for k, v in landmarks.items()}
        return total / len(landmarks) if landmarks else 0.0
    
    def _draw_skeleton(self, frame: np.ndarray, 
                      landmarks: Dict[str, np.ndarray],
                      hip_dist: float,
                      motion: float,
                      angles: Dict[str, float]):
        """Draw pose overlay."""
        # Draw bones
        for name1, name2 in BONE_CONNECTIONS:
            if name1 in landmarks and name2 in landmarks:
                pt1 = tuple(landmarks[name1].astype(int))
                pt2 = tuple(landmarks[name2].astype(int))
                color = (0, 0, 255) if (name1 in ["LEFT_HIP", "RIGHT_HIP"] and 
                                       name2 in ["LEFT_HIP", "RIGHT_HIP"] and 
                                       hip_dist < 20) else (255, 0, 0)
                cv2.line(frame, pt1, pt2, color, 3)
        
        # Draw joints
        for pt in landmarks.values():
            cv2.circle(frame, tuple(pt.astype(int)), 6, (0, 255, 0), -1)
        
        # Draw text overlay
        y = 30
        state = "Static" if motion < 2.0 else "Moving"
        color = (0, 255, 0) if hip_dist >= 20 else (0, 0, 255)
        
        self._draw_text(frame, f"Hip: {hip_dist:.1f}px | Motion: {motion:.1f}px | {state}", 
                       (20, y), color)
        y += 35
        
        for name, val in angles.items():
            self._draw_text(frame, f"{name}: {val:.1f}°", (20, y), (255, 255, 255))
            y += 28
    
    def _draw_text(self, frame: np.ndarray, text: str, pos: tuple, color: tuple):
        """Draw text with outline."""
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _update_graphs(self, angles: Dict[str, float]):
        """Update angle graphs."""
        if self.view_mode == "frontal":
            self.hip_canvas.update_frontal(
                angles.get("LEFT_HIP_adduction", 0),
                angles.get("RIGHT_HIP_adduction", 0)
            )
            self.knee_canvas.update_frontal(
                angles.get("LEFT_KNEE_valgus", 0),
                angles.get("RIGHT_KNEE_valgus", 0)
            )
        else:
            self.hip_canvas.update_sagittal(
                angles.get("LEFT_HIP_flexion", 0),
                angles.get("RIGHT_HIP_flexion", 0)
            )
            self.knee_canvas.update_sagittal(
                angles.get("LEFT_KNEE_flexion", 0),
                angles.get("RIGHT_KNEE_flexion", 0)
            )
            self.ankle_canvas.update_sagittal(
                angles.get("LEFT_ANKLE_flexion", 0),
                angles.get("RIGHT_ANKLE_flexion", 0)
            )
    
    def _display_frame(self, frame: np.ndarray):
        """Show frame in video label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        pix = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, 
                        QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.current_frame_data = frame
    
    def _load_csv(self):
        """Load CSV data into memory."""
        self.csv_data.clear()
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    self.csv_data[int(row["frame"])] = row
                except:
                    pass
    
    def _stop_all(self):
        """Stop all processing."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.detector:
            self.detector.close()
            self.detector = None
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame_data is not None:
            self._display_frame(self.current_frame_data)
    
    def closeEvent(self, event):
        self._stop_all()
        event.accept()