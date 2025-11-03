import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_marker_trajectories(df, markers=None, max_points=200):
    """
    Plots 3D marker trajectories with downsampling for large datasets
    """
    if markers is None:
        markers = ["RHEA", "LHEA", "C7", "STER", "LASI", "RASI", "LKNE", "RKNE", "LANK", "RANK"]
    
    # Downsample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        plot_df = df.iloc[::step]
        print(f"Downsampled to {len(plot_df)} points for visualization")
    else:
        plot_df = df
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(markers)))
    
    for i, marker in enumerate(markers):
        if all(f"{marker}_{coord}" in plot_df.columns for coord in ['X', 'Y', 'Z']):
            x = plot_df[f"{marker}_X"]
            y = plot_df[f"{marker}_Y"] 
            z = plot_df[f"{marker}_Z"]
            
            # Plot trajectory line
            ax.plot(x, y, z, color=colors[i], label=marker, alpha=0.7, linewidth=2)
            # Plot points
            ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=colors[i], s=50, marker='o')
            ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=colors[i], s=50, marker='s')
    
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position") 
    ax.set_zlabel("Z Position")
    ax.set_title("3D Marker Trajectories (Start=circle, End=square)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_simple_skeleton(df, frame_idx=0):
    """Plot a simple skeleton for one frame"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Simple skeleton connections
    skeleton_connections = [
        ('RHEA', 'LHEA'),  # Head width
        ('RHEA', 'C7'), ('LHEA', 'C7'),  # Neck
        ('C7', 'STER'),  # Upper spine
        ('STER', 'RASI'), ('STER', 'LASI'),  # Hips
        ('RASI', 'RKNE'), ('LASI', 'LKNE'),  # Thighs
        ('RKNE', 'RANK'), ('LKNE', 'LANK'),  # Shins
    ]
    
    # Plot markers
    plotted_markers = set()
    for start_marker, end_marker in skeleton_connections:
        for marker in [start_marker, end_marker]:
            if marker not in plotted_markers and all(f"{marker}_{coord}" in df.columns for coord in ['X', 'Y', 'Z']):
                x = df.iloc[frame_idx][f"{marker}_X"]
                y = df.iloc[frame_idx][f"{marker}_Y"]
                z = df.iloc[frame_idx][f"{marker}_Z"]
                ax.scatter(x, y, z, s=100, label=marker)
                plotted_markers.add(marker)
    
    # Plot connections
    for start_marker, end_marker in skeleton_connections:
        if all(f"{marker}_{coord}" in df.columns for coord in ['X', 'Y', 'Z'] for marker in [start_marker, end_marker]):
            start_x = df.iloc[frame_idx][f"{start_marker}_X"]
            start_y = df.iloc[frame_idx][f"{start_marker}_Y"]
            start_z = df.iloc[frame_idx][f"{start_marker}_Z"]
            
            end_x = df.iloc[frame_idx][f"{end_marker}_X"]
            end_y = df.iloc[frame_idx][f"{end_marker}_Y"]
            end_z = df.iloc[frame_idx][f"{end_marker}_Z"]
            
            ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 'b-', linewidth=3, alpha=0.8)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Skeleton - Frame {frame_idx}")
    if plotted_markers:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()