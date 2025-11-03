import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from motion_capture.data_processing import preprocess_trc
from motion_capture.markerless_model import MarkerlessEstimator
from visualization.plot_3d_motion import plot_marker_trajectories, plot_simple_skeleton

def main():
    print("=== Drone Markerless Motion Capture - Data Analysis ===")
    
    # Use relative path from project root
    data_path = os.path.join("data", "sample_marker_data", "RUN.trc")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please check your file paths and run from project root directory")
        return
    
    try:
        # Load and preprocess data
        df, markers = preprocess_trc(data_path)
        
        print(f"\nData loaded successfully:")
        print(f"  - Frames: {len(df)}")
        print(f"  - Markers: {len(markers)}")
        print(f"  - Columns: {list(df.columns[:6])}...")  # Show first few columns
        
        # Basic statistics
        print(f"\nBasic statistics for RHEA marker:")
        if 'RHEA_X' in df.columns:
            for coord in ['X', 'Y', 'Z']:
                col = f'RHEA_{coord}'
                print(f"  {coord}: min={df[col].min():.1f}, max={df[col].max():.1f}, mean={df[col].mean():.1f}")
        
        # Visualizations
        print("\nGenerating visualizations...")
        plot_marker_trajectories(df)
        plot_simple_skeleton(df, frame_idx=0)
        
        # ML model (if we have enough data)
        if len(df) > 50:
            print("\nTraining basic ML model...")
            model = MarkerlessEstimator()
            scores = model.fit(df, marker="RHEA")
            
            if scores:
                print(f"Model training completed with average MAE: {sum(scores.values())/3:.2f}")
            else:
                print("Model training failed - check marker names")
        else:
            print(f"\nNot enough data for ML training (only {len(df)} frames)")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()