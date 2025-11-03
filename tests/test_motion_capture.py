"""
test_motion_capture.py
-----------------
"""

import os
from src.motion_capture.data_processing import preprocess_trc
from src.motion_capture.markerless_model import MarkerlessEstimator
from src.visualization.plot_3d_motion import plot_marker_trajectories

def main():
    data_path = os.path.join("data", "sample_marker_data", "RUN.trc")
    df, markers = preprocess_trc(data_path)

    # 3D Visualization
    plot_marker_trajectories(df)

    # Basic ML Model Simulation
    model = MarkerlessEstimator()
    model.fit(df, marker="RHEA")

if __name__ == "__main__":
    main()
