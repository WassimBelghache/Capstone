import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ------------------------------
# MediaPipe Setup
# ------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

leg_landmarks = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

landmark_names = {
    mp_pose.PoseLandmark.LEFT_HIP: "LEFT_HIP",
    mp_pose.PoseLandmark.LEFT_KNEE: "LEFT_KNEE", 
    mp_pose.PoseLandmark.LEFT_ANKLE: "LEFT_ANKLE",
    mp_pose.PoseLandmark.RIGHT_HIP: "RIGHT_HIP",
    mp_pose.PoseLandmark.RIGHT_KNEE: "RIGHT_KNEE",
    mp_pose.PoseLandmark.RIGHT_ANKLE: "RIGHT_ANKLE"
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
            coords[landmark_names[lm]] = np.array([x_px, y_px])
    return coords

def get_stable_body_coords(results, img_shape):
    """Get body coordinates with robust handling for hip joint issues"""
    h, w, _ = img_shape
    coords = {}
    
    if not results.pose_landmarks:
        return coords
    
    landmarks = results.pose_landmarks.landmark
    
    # Get hip positions
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w, 
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h])
    
    # Calculate hip distance - this is the key fix!
    hip_distance = np.linalg.norm(right_hip - left_hip)
    
    # If hips are too close (less than 10 pixels), use shoulder-based reference
    if hip_distance < 10:
        # Fallback: use shoulders to define body orientation
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])
        
        body_center = (left_shoulder + right_shoulder) / 2
        shoulder_vector = right_shoulder - left_shoulder
        
        if np.linalg.norm(shoulder_vector) > 10:
            body_x = shoulder_vector / np.linalg.norm(shoulder_vector)
        else:
            # If shoulders are also too close, use default orientation
            body_x = np.array([1, 0])
            
        body_y = np.array([-body_x[1], body_x[0]])
        
        # Recalculate hip positions relative to shoulder center
        left_hip_rel = left_hip - body_center
        right_hip_rel = right_hip - body_center
        
    else:
        # Normal case: use hips for body coordinate system
        body_center = (left_hip + right_hip) / 2
        hip_vector = right_hip - left_hip
        body_x = hip_vector / np.linalg.norm(hip_vector)
        body_y = np.array([-body_x[1], body_x[0]])
    
    # Convert all leg landmarks to body coordinates
    for lm in leg_landmarks:
        world_coord = np.array([landmarks[lm].x * w, landmarks[lm].y * h])
        
        # Transform to body coordinates
        rel_coord = world_coord - body_center
        body_coord_x = np.dot(rel_coord, body_x)
        body_coord_y = np.dot(rel_coord, body_y)
        
        coords[landmark_names[lm]] = np.array([body_coord_x, body_coord_y])
    
    return coords, hip_distance

def compute_stable_angles(coords, hip_distance):
    """Compute angles with protection against hip joint collapse"""
    angles = {}
    try:
        # If hips are too close, limit the angle calculations to prevent spikes
        stability_factor = min(1.0, hip_distance / 50.0)  # Normalize by expected hip distance
        
        # HIP AD/ABDUCTION
        if "LEFT_HIP" in coords and "LEFT_KNEE" in coords:
            thigh_vector = coords["LEFT_KNEE"] - coords["LEFT_HIP"]
            vertical_ref = np.array([0, -1])
            raw_angle = angle_between(thigh_vector, vertical_ref)
            # Apply stability factor to smooth extreme values when hips are close
            angles["LEFT_HIP_adduction"] = raw_angle * stability_factor
            
        if "RIGHT_HIP" in coords and "RIGHT_KNEE" in coords:
            thigh_vector = coords["RIGHT_KNEE"] - coords["RIGHT_HIP"]
            vertical_ref = np.array([0, -1])
            raw_angle = angle_between(thigh_vector, vertical_ref)
            angles["RIGHT_HIP_adduction"] = raw_angle * stability_factor

        # KNEE VALGUS - Apply similar stability
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

    except Exception as e:
        print(f"Angle computation error: {e}")
    return angles

def fix_video_rotation(frame, cap):
    try:
        rotation_code = None
        try:
            rotation_code = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        except:
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
        
    def stabilize_angles(self, new_angles, hip_distance):
        # If hips are too close, rely more on history
        trust_factor = min(1.0, hip_distance / 30.0)
        
        if self.previous_angles is None:
            self.previous_angles = new_angles
            # Initialize history
            for key in new_angles:
                self.angle_history[key] = [new_angles[key]]
            return new_angles
        
        stabilized = {}
        for key in new_angles:
            if key in self.previous_angles:
                # Dynamic smoothing based on hip distance
                current_smoothing = self.smoothing_factor * (1 - trust_factor) + 0.9 * trust_factor
                
                # Use median of recent history for more stability
                if key in self.angle_history:
                    self.angle_history[key].append(new_angles[key])
                    if len(self.angle_history[key]) > self.max_history:
                        self.angle_history[key].pop(0)
                    
                    # Use median filtering when hips are close
                    if hip_distance < 20:
                        historical_value = np.median(self.angle_history[key])
                    else:
                        historical_value = self.previous_angles[key]
                else:
                    historical_value = self.previous_angles[key]
                
                stabilized[key] = (current_smoothing * new_angles[key] + 
                                 (1 - current_smoothing) * historical_value)
            else:
                stabilized[key] = new_angles[key]
        
        self.previous_angles = stabilized
        return stabilized

# ------------------------------
# GUI Canvases
# ------------------------------
class AngleCanvas(FigureCanvas):
    """Hip Adduction Graph"""
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.tight_layout(pad=3.0)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.left_hip_data = []
        self.right_hip_data = []
        self.frame_idx = 0

    def update_plot(self, angles):
        self.frame_idx += 1
        self.left_hip_data.append(angles.get("LEFT_HIP_adduction", 0))
        self.right_hip_data.append(angles.get("RIGHT_HIP_adduction", 0))

        # Limit data length to prevent memory issues
        if len(self.left_hip_data) > 500:
            self.left_hip_data = self.left_hip_data[-500:]
            self.right_hip_data = self.right_hip_data[-500:]

        self.ax.cla()
        left_smoothed = smooth_data(self.left_hip_data)
        right_smoothed = smooth_data(self.right_hip_data)

        self.ax.plot(left_smoothed, label="Left Hip Adduction", color='r')
        self.ax.plot(right_smoothed, label="Right Hip Adduction", color='m')
        self.ax.set_xlabel("Frame", fontsize=12)
        self.ax.set_ylabel("Angle (deg)", fontsize=12)
        self.ax.set_title("Hip Adduction Over Time", fontsize=14)
        self.ax.legend(loc='upper right', fontsize=10)

        all_data = left_smoothed + right_smoothed
        if all_data:
            ymin = max(0, min(all_data) - 5)
            ymax = min(180, max(all_data) + 5)  # Cap at reasonable angle
            self.ax.set_ylim(ymin, ymax)

        self.draw()

class ValgusCanvas(FigureCanvas):
    """Knee Valgus Graph"""
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.tight_layout(pad=3.0)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.left_knee_data = []
        self.right_knee_data = []
        self.frame_idx = 0

    def update_plot(self, angles):
        self.frame_idx += 1
        self.left_knee_data.append(angles.get("LEFT_KNEE_valgus", 0))
        self.right_knee_data.append(angles.get("RIGHT_KNEE_valgus", 0))

        # Limit data length
        if len(self.left_knee_data) > 500:
            self.left_knee_data = self.left_knee_data[-500:]
            self.right_knee_data = self.right_knee_data[-500:]

        self.ax.cla()
        left_smoothed = smooth_data(self.left_knee_data)
        right_smoothed = smooth_data(self.right_knee_data)

        self.ax.plot(left_smoothed, label="Left Knee Valgus", color='b')
        self.ax.plot(right_smoothed, label="Right Knee Valgus", color='c')
        self.ax.set_xlabel("Frame", fontsize=12)
        self.ax.set_ylabel("Angle (deg)", fontsize=12)
        self.ax.set_title("Knee Valgus Over Time", fontsize=14)
        self.ax.legend(loc='upper right', fontsize=10)

        all_data = left_smoothed + right_smoothed
        if all_data:
            ymin = max(0, min(all_data) - 5)
            ymax = min(90, max(all_data) + 5)  # Cap at reasonable angle
            self.ax.set_ylim(ymin, ymax)

        self.draw()

# ------------------------------
# Main App
# ------------------------------
class MocapApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frontal Lower-Limb Motion Capture (Robust Version)")
        self.resize(1200, 600)

        main_layout = QtWidgets.QHBoxLayout(self)

        # Left panel: video
        left_panel = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid gray;")
        self.video_label.setText("Load a video to begin")
        left_panel.addWidget(self.video_label)
        self.load_btn = QtWidgets.QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        left_panel.addWidget(self.load_btn)

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

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.timer.stop()
            self.status_label.setText("Video finished")
            return

        frame = fix_video_rotation(frame, self.cap)
        self.current_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Get absolute coordinates for display
        abs_coords_2d = get_2d_joint_coords(results, frame.shape)
        
        display_frame = frame.copy()
        
        if results.pose_landmarks and len(abs_coords_2d) >= 4:
            self.detection_count += 1
            
            # Get stable body coordinates with hip distance monitoring
            body_coords, hip_distance = get_stable_body_coords(results, frame.shape)
            angles = compute_stable_angles(body_coords, hip_distance)
            
            # Advanced stabilization
            stabilized_angles = self.stabilizer.stabilize_angles(angles, hip_distance)
            
            # Draw skeleton with hip distance warning
            for name, pt in abs_coords_2d.items():
                cv2.circle(display_frame, tuple(pt.astype(int)), 8, (0, 255, 0), -1)
                cv2.putText(display_frame, name.split('_')[-1], 
                           tuple(pt.astype(int) + np.array([10, -10])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            bone_pairs = [
                ("LEFT_HIP", "RIGHT_HIP"),
                ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
                ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE")
            ]
            for j1, j2 in bone_pairs:
                if j1 in abs_coords_2d and j2 in abs_coords_2d:
                    color = (255, 0, 0)  # Normal blue
                    if j1 in ["LEFT_HIP", "RIGHT_HIP"] and j2 in ["LEFT_HIP", "RIGHT_HIP"]:
                        # Highlight hip line in red if hips are too close
                        if hip_distance < 20:
                            color = (0, 0, 255)  # Red warning
                    cv2.line(display_frame, tuple(abs_coords_2d[j1]), tuple(abs_coords_2d[j2]), color, 3)

            # Display angles and hip status
            y_offset = 40
            status_text = f"Hip distance: {hip_distance:.1f}px"
            color = (0, 255, 0) if hip_distance >= 20 else (0, 0, 255)
            cv2.putText(display_frame, status_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(display_frame, status_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 35
            
            for joint, angle in stabilized_angles.items():
                text = f"{joint}: {angle:.1f} deg"
                cv2.putText(display_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(display_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                y_offset += 35
            
            # Update graphs
            self.adduction_canvas.update_plot(stabilized_angles)
            self.valgus_canvas.update_plot(stabilized_angles)
            self.status_label.setText(f"Tracking: {self.detection_count} frames | Hip distance: {hip_distance:.1f}px")
            
        else:
            cv2.putText(display_frame, "Searching for pose...", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.status_label.setText("Searching for human pose...")

        # Show video
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(display_frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(),
                                      QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.update_frame()

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MocapApp()
    window.show()
    sys.exit(app.exec_())