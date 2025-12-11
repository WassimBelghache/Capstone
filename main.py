import sys
import cv2
import csv
import numpy as np
import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ------------------------------
# MediaPipe Setup
# ------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# View mode: "frontal" (adduction/valgus) or "sagittal" (flexion/extension)
VIEW_MODES = ["frontal", "sagittal"]

leg_landmarks = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
]

landmark_names = {
    mp_pose.PoseLandmark.LEFT_HIP: "LEFT_HIP",
    mp_pose.PoseLandmark.LEFT_KNEE: "LEFT_KNEE",
    mp_pose.PoseLandmark.LEFT_ANKLE: "LEFT_ANKLE",
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX: "LEFT_FOOT_INDEX",
    mp_pose.PoseLandmark.RIGHT_HIP: "RIGHT_HIP",
    mp_pose.PoseLandmark.RIGHT_KNEE: "RIGHT_KNEE",
    mp_pose.PoseLandmark.RIGHT_ANKLE: "RIGHT_ANKLE",
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: "RIGHT_FOOT_INDEX"
}

# ------------------------------
# Helper Functions
# ------------------------------
def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms < 1e-6:
        return 0.0
    cos_theta = np.clip(dot_product / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def get_2d_joint_coords(results, img_shape):
    h, w, _ = img_shape
    coords = {}
    if results.pose_landmarks:
        for lm in leg_landmarks:
            lm_data = results.pose_landmarks.landmark[lm]
            x_px = int(lm_data.x * w)
            y_px = int(lm_data.y * h)
            coords[landmark_names[lm]] = np.array([x_px, y_px], dtype=np.float32)
    return coords

def get_stable_body_coords(results, img_shape):
    """Get body coordinates with robust handling for hip joint issues"""
    h, w, _ = img_shape
    coords = {}
    
    if not results.pose_landmarks:
        return coords, 0.0
    
    landmarks = results.pose_landmarks.landmark
    
    left_hip = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h
    ])
    right_hip = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h
    ])
    
    hip_distance = np.linalg.norm(right_hip - left_hip)
    
    if hip_distance < 10:
        left_shoulder = np.array([
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
        ])
        right_shoulder = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
        ])
        
        body_center = (left_shoulder + right_shoulder) / 2.0
        shoulder_vector = right_shoulder - left_shoulder
        
        if np.linalg.norm(shoulder_vector) > 10:
            body_x = shoulder_vector / np.linalg.norm(shoulder_vector)
        else:
            body_x = np.array([1.0, 0.0])
        body_y = np.array([-body_x[1], body_x[0]])
    else:
        body_center = (left_hip + right_hip) / 2.0
        hip_vector = right_hip - left_hip
        body_x = hip_vector / np.linalg.norm(hip_vector)
        body_y = np.array([-body_x[1], body_x[0]])
    
    for lm in leg_landmarks:
        world_coord = np.array([
            landmarks[lm].x * w,
            landmarks[lm].y * h
        ])
        rel_coord = world_coord - body_center
        body_coord_x = np.dot(rel_coord, body_x)
        body_coord_y = np.dot(rel_coord, body_y)
        coords[landmark_names[lm]] = np.array([body_coord_x, body_coord_y])
    
    return coords, hip_distance

def compute_stable_angles(coords, hip_distance, view_mode="frontal"):
    """
    Compute angles with protection against hip joint collapse,
    normalized by approximate leg length and supporting frontal/sagittal views.
    """
    angles = {}
    if not coords:
        return angles
    
    leg_lengths = []
    if "LEFT_HIP" in coords and "LEFT_ANKLE" in coords:
        leg_lengths.append(np.linalg.norm(coords["LEFT_ANKLE"] - coords["LEFT_HIP"]))
    if "RIGHT_HIP" in coords and "RIGHT_ANKLE" in coords:
        leg_lengths.append(np.linalg.norm(coords["RIGHT_ANKLE"] - coords["RIGHT_HIP"]))
    
    leg_length = np.mean(leg_lengths) if leg_lengths else 50.0
    leg_length = max(leg_length, 1e-3)
    
    normalized_hip = hip_distance / leg_length
    stability_factor = min(1.0, normalized_hip)  # 0–1
    
    try:
        if view_mode == "frontal":
            # HIP AD/ABDUCTION
            if "LEFT_HIP" in coords and "LEFT_KNEE" in coords:
                thigh_vector = coords["LEFT_KNEE"] - coords["LEFT_HIP"]
                vertical_ref = np.array([0.0, -1.0])
                raw_angle = angle_between(thigh_vector, vertical_ref)
                angles["LEFT_HIP_adduction"] = raw_angle * stability_factor
            
            if "RIGHT_HIP" in coords and "RIGHT_KNEE" in coords:
                thigh_vector = coords["RIGHT_KNEE"] - coords["RIGHT_HIP"]
                vertical_ref = np.array([0.0, -1.0])
                raw_angle = angle_between(thigh_vector, vertical_ref)
                angles["RIGHT_HIP_adduction"] = raw_angle * stability_factor

            # KNEE VALGUS
            if "LEFT_HIP" in coords and "LEFT_KNEE" in coords and "LEFT_ANKLE" in coords:
                mech_axis = coords["LEFT_ANKLE"] - coords["LEFT_HIP"]
                thigh_segment = coords["LEFT_KNEE"] - coords["LEFT_HIP"]
                raw_angle = angle_between(mech_axis, thigh_segment)
                angles["LEFT_KNEE_valgus"] = raw_angle * stability_factor
            
            if "RIGHT_HIP" in coords and "RIGHT_KNEE" in coords and "RIGHT_ANKLE" in coords:
                mech_axis = coords["RIGHT_ANKLE"] - coords["RIGHT_HIP"]
                thigh_segment = coords["RIGHT_KNEE"] - coords["RIGHT_HIP"]
                raw_angle = angle_between(mech_axis, thigh_segment)
                angles["RIGHT_KNEE_valgus"] = raw_angle * stability_factor
        
        elif view_mode == "sagittal":
            # Sagittal plane: flexion/extension
            if "LEFT_HIP" in coords and "LEFT_KNEE" in coords:
                thigh_vector = coords["LEFT_KNEE"] - coords["LEFT_HIP"]
                ref_axis = np.array([1.0, 0.0])  # horizontal reference
                raw_angle = angle_between(thigh_vector, ref_axis)
                angles["LEFT_HIP_flexion"] = raw_angle * stability_factor
            
            if "RIGHT_HIP" in coords and "RIGHT_KNEE" in coords:
                thigh_vector = coords["RIGHT_KNEE"] - coords["RIGHT_HIP"]
                ref_axis = np.array([1.0, 0.0])
                raw_angle = angle_between(thigh_vector, ref_axis)
                angles["RIGHT_HIP_flexion"] = raw_angle * stability_factor

            # Knee flexion: angle between thigh and shank
            if "LEFT_HIP" in coords and "LEFT_KNEE" in coords and "LEFT_ANKLE" in coords:
                thigh = coords["LEFT_HIP"] - coords["LEFT_KNEE"]
                shank = coords["LEFT_ANKLE"] - coords["LEFT_KNEE"]
                raw_angle = angle_between(thigh, shank)
                angles["LEFT_KNEE_flexion"] = raw_angle * stability_factor
            
            if "RIGHT_HIP" in coords and "RIGHT_KNEE" in coords and "RIGHT_ANKLE" in coords:
                thigh = coords["RIGHT_HIP"] - coords["RIGHT_KNEE"]
                shank = coords["RIGHT_ANKLE"] - coords["RIGHT_KNEE"]
                raw_angle = angle_between(thigh, shank)
                angles["RIGHT_KNEE_flexion"] = raw_angle * stability_factor

            # Ankle / foot segment angle (toe segment)
            if "LEFT_ANKLE" in coords and "LEFT_FOOT_INDEX" in coords:
                foot_vec = coords["LEFT_FOOT_INDEX"] - coords["LEFT_ANKLE"]
                shank_vec = coords["LEFT_ANKLE"] - coords["LEFT_KNEE"] if "LEFT_KNEE" in coords else np.array([0.0, -1.0])
                raw_angle = angle_between(foot_vec, shank_vec)
                angles["LEFT_ANKLE_flexion"] = raw_angle * stability_factor

            if "RIGHT_ANKLE" in coords and "RIGHT_FOOT_INDEX" in coords:
                foot_vec = coords["RIGHT_FOOT_INDEX"] - coords["RIGHT_ANKLE"]
                shank_vec = coords["RIGHT_ANKLE"] - coords["RIGHT_KNEE"] if "RIGHT_KNEE" in coords else np.array([0.0, -1.0])
                raw_angle = angle_between(foot_vec, shank_vec)
                angles["RIGHT_ANKLE_flexion"] = raw_angle * stability_factor

    except Exception as e:
        print(f"Angle computation error: {e}")
    return angles

def fix_video_rotation(frame, cap):
    try:
        rotation_code = None
        try:
            rotation_code = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        except Exception:
            pass
        if rotation_code is not None and rotation_code != 0:
            if rotation_code == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_code == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_code == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"Rotation fix error: {e}")
    return frame

# ------------------------------
# Smoothing function
# ------------------------------
def smooth_data(data, window=5):
    if len(data) < 2:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i+1]
        smoothed.append(sum(window_data) / len(window_data))
    return smoothed

# ------------------------------
# Advanced Stabilizer
# ------------------------------
class AdvancedStabilizer:
    def __init__(self):
        self.previous_angles = None
        self.smoothing_factor = 0.7
        self.angle_history = {}
        self.max_history = 10
        
    def stabilize_angles(self, new_angles, hip_distance, motion_level):
        """
        Dynamic smoothing based on hip distance (stability) and motion speed.
        motion_level: average joint displacement in pixels per frame.
        """
        # If hips are too close, rely more on history
        trust_factor = min(1.0, hip_distance / 30.0)  # 0–1
        
        if self.previous_angles is None:
            self.previous_angles = new_angles
            for key in new_angles:
                self.angle_history[key] = [new_angles[key]]
            return new_angles
        
        stabilized = {}
        # Simple motion thresholds
        high_speed_threshold = 10.0
        low_speed_threshold = 2.0
        
        for key in new_angles:
            if key in self.previous_angles:
                # Base smoothing from hip stability
                base_smoothing = (
                    self.smoothing_factor * (1 - trust_factor) +
                    0.9 * trust_factor
                )
                
                # Adjust smoothing based on motion speed:
                # - High speed → less smoothing (more responsive)
                # - Low speed → more smoothing (more stable)
                if motion_level > high_speed_threshold:
                    current_smoothing = min(base_smoothing, 0.5)
                elif motion_level < low_speed_threshold:
                    current_smoothing = max(base_smoothing, 0.85)
                else:
                    current_smoothing = base_smoothing
                
                if key not in self.angle_history:
                    self.angle_history[key] = []
                self.angle_history[key].append(new_angles[key])
                if len(self.angle_history[key]) > self.max_history:
                    self.angle_history[key].pop(0)
                
                # When hips are very close, use median history
                if hip_distance < 20:
                    historical_value = float(np.median(self.angle_history[key]))
                else:
                    historical_value = self.previous_angles[key]
                
                stabilized[key] = (
                    current_smoothing * new_angles[key] +
                    (1 - current_smoothing) * historical_value
                )
            else:
                stabilized[key] = new_angles[key]
        
        self.previous_angles = stabilized
        return stabilized

# ------------------------------
# GUI Canvases
# ------------------------------
class AngleCanvas(FigureCanvas):
    """Hip angle graph: adduction or flexion, depending on mode."""
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.tight_layout(pad=3.0)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.left_hip_data = []
        self.right_hip_data = []
        self.frame_idx = 0

    def update_plot(self, angles, view_mode):
        self.frame_idx += 1
        
        # Use the appropriate angle keys depending on mode
        if view_mode == "frontal":
            left_val = angles.get("LEFT_HIP_adduction", 0.0)
            right_val = angles.get("RIGHT_HIP_adduction", 0.0)
            title = "Hip Adduction Over Time"
            ylim_max = 60
        else:
            left_val = angles.get("LEFT_HIP_flexion", 0.0)
            right_val = angles.get("RIGHT_HIP_flexion", 0.0)
            title = "Hip Flexion Over Time"
            ylim_max = 180
        
        self.left_hip_data.append(left_val)
        self.right_hip_data.append(right_val)

        # Limit data length
        max_len = 500
        if len(self.left_hip_data) > max_len:
            self.left_hip_data = self.left_hip_data[-max_len:]
            self.right_hip_data = self.right_hip_data[-max_len:]

        self.ax.cla()
        
        # Slightly larger window when movement is slower; simple fixed window here
        left_smoothed = smooth_data(self.left_hip_data, window=5)
        right_smoothed = smooth_data(self.right_hip_data, window=5)

        self.ax.plot(left_smoothed, label="Left Hip", color='r')
        self.ax.plot(right_smoothed, label="Right Hip", color='m')
        self.ax.set_xlabel("Frame", fontsize=12)
        self.ax.set_ylabel("Angle (deg)", fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.legend(loc='upper right', fontsize=10)

        all_data = left_smoothed + right_smoothed
        if all_data:
            ymin = max(0, min(all_data) - 5)
            ymax = min(ylim_max, max(all_data) + 5)
            self.ax.set_ylim(ymin, ymax)

        self.draw()

class ValgusCanvas(FigureCanvas):
    """Knee angle graph: valgus or flexion, depending on mode."""
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.tight_layout(pad=3.0)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.left_knee_data = []
        self.right_knee_data = []
        self.frame_idx = 0

    def update_plot(self, angles, view_mode):
        self.frame_idx += 1
        
        if view_mode == "frontal":
            left_val = angles.get("LEFT_KNEE_valgus", 0.0)
            right_val = angles.get("RIGHT_KNEE_valgus", 0.0)
            title = "Knee Valgus Over Time"
            ylim_max = 90
        else:
            left_val = angles.get("LEFT_KNEE_flexion", 0.0)
            right_val = angles.get("RIGHT_KNEE_flexion", 0.0)
            title = "Knee Flexion Over Time"
            ylim_max = 180
        
        self.left_knee_data.append(left_val)
        self.right_knee_data.append(right_val)

        max_len = 500
        if len(self.left_knee_data) > max_len:
            self.left_knee_data = self.left_knee_data[-max_len:]
            self.right_knee_data = self.right_knee_data[-max_len:]

        self.ax.cla()
        left_smoothed = smooth_data(self.left_knee_data, window=5)
        right_smoothed = smooth_data(self.right_knee_data, window=5)

        self.ax.plot(left_smoothed, label="Left Knee", color='b')
        self.ax.plot(right_smoothed, label="Right Knee", color='c')
        self.ax.set_xlabel("Frame", fontsize=12)
        self.ax.set_ylabel("Angle (deg)", fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.legend(loc='upper right', fontsize=10)

        all_data = left_smoothed + right_smoothed
        if all_data:
            ymin = max(0, min(all_data) - 5)
            ymax = min(ylim_max, max(all_data) + 5)
            self.ax.set_ylim(ymin, ymax)

        self.draw()

# ------------------------------
# Main App
# ------------------------------
class MocapApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lower-Limb Motion Capture (Robust Version)")
        self.resize(1300, 700)

        main_layout = QtWidgets.QHBoxLayout(self)

        # Left panel: video & controls
        left_panel = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid gray;")
        self.video_label.setText("Load a video or start live mode to begin")
        left_panel.addWidget(self.video_label)

        # Control buttons
        controls_layout = QtWidgets.QHBoxLayout()

        self.load_btn = QtWidgets.QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        controls_layout.addWidget(self.load_btn)

        self.live_btn = QtWidgets.QPushButton("Live Mode")
        self.live_btn.clicked.connect(self.start_live_mode)
        controls_layout.addWidget(self.live_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        controls_layout.addWidget(self.stop_btn)

        left_panel.addLayout(controls_layout)


        # View mode selector
        view_layout = QtWidgets.QHBoxLayout()
        view_label = QtWidgets.QLabel("View mode:")
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItems(VIEW_MODES)
        self.view_mode_combo.currentTextChanged.connect(self.on_view_mode_changed)
        view_layout.addWidget(view_label)
        view_layout.addWidget(self.view_mode_combo)
        left_panel.addLayout(view_layout)

        # Right panel: stacked graphs
        right_panel = QtWidgets.QVBoxLayout()
        self.adduction_canvas = AngleCanvas(self, width=5, height=3)
        self.adduction_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Expanding)
        self.valgus_canvas = ValgusCanvas(self, width=5, height=3)
        self.valgus_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                         QtWidgets.QSizePolicy.Expanding)

        right_panel.addWidget(self.adduction_canvas, 1)
        right_panel.addWidget(self.valgus_canvas, 1)

        self.status_label = QtWidgets.QLabel("Angle graphs will appear here")
        right_panel.addWidget(self.status_label)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)

        # Timer & video
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.current_frame = None
        self.stabilizer = AdvancedStabilizer()
        self.detection_count = 0
        self.frame_idx = 0
        self.view_mode = "frontal"
        self.prev_abs_coords = None
        self.csv_path = "angle_output.csv"

    def on_view_mode_changed(self, mode):
        if mode in VIEW_MODES:
            self.view_mode = mode

    def init_csv_log(self):
        """Initialize CSV file for saving angle data for lab validation."""
        self.csv_path = "angle_output.csv"
        try:
            with open(self.csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "joint", "angle_deg",
                                 "hip_distance_px", "motion_px", "state"])
        except Exception as e:
            print(f"CSV init error: {e}")

    def append_csv_log(self, frame_idx, angles, hip_distance, motion_level, state):
        try:
            with open(self.csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                for joint, angle in angles.items():
                    writer.writerow([
                        frame_idx,
                        joint,
                        f"{angle:.2f}",
                        f"{hip_distance:.2f}",
                        f"{motion_level:.2f}",
                        state
                    ])
        except Exception as e:
            print(f"CSV append error: {e}")

    def load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(self, "Error", "Could not open video")
                return
            self.timer.start(30)
            self.status_label.setText("Processing video...")
            self.stabilizer = AdvancedStabilizer()
            self.detection_count = 0
            self.frame_idx = 0
            self.prev_abs_coords = None
            self.init_csv_log()

    def start_live_mode(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Error", "Could not access camera")
            return
        self.timer.start(30)
        self.status_label.setText("Live mode: processing webcam stream...")
        self.stabilizer = AdvancedStabilizer()
        self.detection_count = 0
        self.frame_idx = 0
        self.prev_abs_coords = None
        self.init_csv_log()

    def compute_motion_level(self, abs_coords_2d):
        """Average per-joint displacement between frames in pixels."""
        if self.prev_abs_coords is None:
            self.prev_abs_coords = {k: v.copy() for k, v in abs_coords_2d.items()}
            return 0.0
        
        total_motion = 0.0
        count = 0
        for k, v in abs_coords_2d.items():
            if k in self.prev_abs_coords:
                total_motion += float(np.linalg.norm(v - self.prev_abs_coords[k]))
                count += 1
        self.prev_abs_coords = {k: v.copy() for k, v in abs_coords_2d.items()}
        if count == 0:
            return 0.0
        return total_motion / count

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.timer.stop()
            self.status_label.setText("Video finished")
            return

        self.frame_idx += 1

        frame = fix_video_rotation(frame, self.cap)
        self.current_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Get absolute coordinates for display
        abs_coords_2d = get_2d_joint_coords(results, frame.shape)
        motion_level = self.compute_motion_level(abs_coords_2d) if abs_coords_2d else 0.0
        state = "static" if motion_level < 2.0 else "moving"

        display_frame = frame.copy()
        
        if results.pose_landmarks and len(abs_coords_2d) >= 4:
            self.detection_count += 1
            
            # Get stable body coordinates with hip distance monitoring
            body_coords, hip_distance = get_stable_body_coords(results, frame.shape)
            angles = compute_stable_angles(body_coords, hip_distance, self.view_mode)
            
            # Advanced stabilization with motion information
            stabilized_angles = self.stabilizer.stabilize_angles(
                angles, hip_distance, motion_level
            )
            
            # Draw skeleton with hip distance warning
            for name, pt in abs_coords_2d.items():
                cv2.circle(display_frame, tuple(pt.astype(int)), 8, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    name.split('_')[-1],
                    tuple((pt + np.array([10, -10])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
            
            bone_pairs = [
                ("LEFT_HIP", "RIGHT_HIP"),
                ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
                ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
                ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
                ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX")
            ]
            for j1, j2 in bone_pairs:
                if j1 in abs_coords_2d and j2 in abs_coords_2d:
                    color = (255, 0, 0)
                    if j1 in ["LEFT_HIP", "RIGHT_HIP"] and j2 in ["LEFT_HIP", "RIGHT_HIP"]:
                        # Highlight hip line in red if hips are too close
                        if hip_distance < 20:
                            color = (0, 0, 255)
                    cv2.line(
                        display_frame,
                        tuple(abs_coords_2d[j1].astype(int)),
                        tuple(abs_coords_2d[j2].astype(int)),
                        color,
                        3
                    )

            # Display angles, hip status, and motion state
            y_offset = 40
            status_text = f"Hip distance: {hip_distance:.1f}px | Motion: {motion_level:.1f}px | State: {state}"
            color = (0, 255, 0) if hip_distance >= 20 else (0, 0, 255)
            cv2.putText(
                display_frame, status_text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3
            )
            cv2.putText(
                display_frame, status_text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            y_offset += 35
            
            for joint, angle in stabilized_angles.items():
                text = f"{joint}: {angle:.1f} deg"
                cv2.putText(
                    display_frame, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3
                )
                cv2.putText(
                    display_frame, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                y_offset += 30
            
            self.adduction_canvas.update_plot(stabilized_angles, self.view_mode)
            self.valgus_canvas.update_plot(stabilized_angles, self.view_mode)

            self.append_csv_log(self.frame_idx, stabilized_angles, hip_distance, motion_level, state)

            self.status_label.setText(
                f"Tracking frames: {self.detection_count} | Hip distance: {hip_distance:.1f}px | Mode: {self.view_mode.capitalize()}"
            )
        else:
            cv2.putText(
                display_frame, "Searching for pose...", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            self.status_label.setText("Searching for human pose...")

        # Show video
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(display_frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(), self.video_label.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            # Force a redraw of current frame
            display_frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = display_frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(display_frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.width(), self.video_label.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            
    def stop_processing(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.status_label.setText("Processing stopped")
        self.current_frame = None
        self.video_label.setText("Stopped")

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MocapApp()
    window.show()
    sys.exit(app.exec_())
