from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

from drone_mocap.io.video import get_video_meta, iter_frames
from drone_mocap.pose.mediapose import MediaPipePoseEstimator
from drone_mocap.filters.smoothing import savgol_smooth_xy
from drone_mocap.angles.saggital2D import joint_angles_sagittal
from drone_mocap.io.mocap_txt import read_mocap_angles_txt

def run_pipeline(
    video: Path,
    out_root: Path,
    mocap_txt: Path | None = None,
    visible_side: str = "right",
    max_frames: int = 0,
) -> Path:
    meta = get_video_meta(video)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "derived").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_dict = {
        "video": str(video),
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "width": meta.width,
        "height": meta.height,
        "visible_side": visible_side,
        "max_frames": max_frames,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_dict, indent=2))

    # Pose inference
    estimator = MediaPipePoseEstimator()
    xy_list = []
    vis_list = []
    frame_idx = []

    for i, frame in iter_frames(video, max_frames=max_frames):
        pr = estimator.predict(frame)
        xy_list.append(pr.xy)
        vis_list.append(pr.vis)
        frame_idx.append(i)

    estimator.close()

    xy = np.stack(xy_list, axis=0)   # (T,33,2)
    vis = np.stack(vis_list, axis=0) # (T,33)
    T = xy.shape[0]

    # Smooth
    xy_s = savgol_smooth_xy(xy, window=11, poly=2)

    # Compute angles per frame (visible side only for MVP)
    rows = []
    for t in range(T):
        ang = joint_angles_sagittal(xy_s[t], vis[t], visible_side=visible_side, min_vis=0.5)
        rows.append({
            "frame": frame_idx[t],
            "time_s": frame_idx[t] / meta.fps,
            f"{visible_side}_hip_deg": ang["hip"],
            f"{visible_side}_knee_deg": ang["knee"],
            f"{visible_side}_ankle_deg": ang["ankle"],
        })

    df_angles = pd.DataFrame(rows)

    # Save keypoints (optional big file)
    # Store as flattened columns for portability
    flat = {}
    for j in range(33):
        flat[f"j{j}_x"] = xy_s[:, j, 0]
        flat[f"j{j}_y"] = xy_s[:, j, 1]
        flat[f"j{j}_vis"] = vis[:, j]
    df_pose = pd.DataFrame(flat)
    df_pose.insert(0, "frame", frame_idx)
    df_pose.insert(1, "time_s", df_angles["time_s"].values)

    df_pose.to_parquet(out_dir / "derived" / "poses.parquet", index=False)
    df_angles.to_csv(out_dir / "derived" / "angles_sagittal.csv", index=False)
    df_angles.to_parquet(out_dir / "derived" / "angles_sagittal.parquet", index=False)

    # Optional MoCap load (no alignment yet in MVP)
    if mocap_txt:
        df_mocap = read_mocap_angles_txt(mocap_txt)
        df_mocap.to_parquet(out_dir / "raw" / "mocap_angles.parquet", index=False)

    summary = {
        "frames_processed": int(T),
        "outputs": {
            "poses_parquet": "derived/poses.parquet",
            "angles_csv": "derived/angles_sagittal.csv",
            "angles_parquet": "derived/angles_sagittal.parquet",
            "mocap_parquet": "raw/mocap_angles.parquet" if mocap_txt else None,
        }
    }
    (out_dir / "reports" / "summary.json").write_text(json.dumps(summary, indent=2))

    return out_dir
