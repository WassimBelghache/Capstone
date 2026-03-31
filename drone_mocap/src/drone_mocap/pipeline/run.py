from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Callable
import json
import numpy as np
import pandas as pd
import cv2

from drone_mocap.io.video import get_video_meta, iter_frames
from drone_mocap.pose.mediapose import MediaPipePoseEstimator, LM
from drone_mocap.filters.smoothing import butterworth_smooth_xy, median_smooth_angles
from drone_mocap.angles.saggital2D import joint_angles_sagittal, clamp_angle_series
from drone_mocap.io.mocap_txt import read_mocap_angles_txt
from drone_mocap.evaluation.compare_mocap import compare_video_to_mocap, save_compare_outputs
from drone_mocap.utils.scaling import PixelScaler

# ---------------------------------------------------------------------------
# Skeleton connectivity (MediaPipe Pose 33-keypoint topology)
# ---------------------------------------------------------------------------
_SKELETON_LINKS = [
    (11, 12), (11, 23), (12, 24), (23, 24),   # torso
    (23, 25), (24, 26),                         # upper legs
    (25, 27), (26, 28),                         # lower legs
    (27, 29), (28, 30), (29, 31), (30, 32),    # feet
    (11, 13), (12, 14), (13, 15), (14, 16),    # arms
]

# Color palette (BGR)
_CLR_HIGH   = (0, 220,   0)   # green  — vis > 0.8
_CLR_INTERP = (0, 220, 220)   # yellow — 0.3 ≤ vis ≤ 0.8
_CLR_LOW    = (0,  40, 220)   # red    — vis < 0.3


def _kp_color(vis_score: float) -> tuple[int, int, int]:
    if vis_score >= 0.8:
        return _CLR_HIGH
    if vis_score >= 0.3:
        return _CLR_INTERP
    return _CLR_LOW


def _draw_skeleton(
    frame: np.ndarray,
    xy: np.ndarray,
    vis: np.ndarray,
    angles: dict[str, float],
    visible_side: str,
) -> np.ndarray:
    """Draw color-coded skeleton and angle labels onto a single frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    prefix = "R" if visible_side.lower().startswith("r") else "L"

    for i, j in _SKELETON_LINKS:
        pi, pj = xy[i], xy[j]
        if not (np.all(np.isfinite(pi)) and np.all(np.isfinite(pj))):
            continue
        color = _kp_color((vis[i] + vis[j]) / 2.0)
        pt1 = (int(np.clip(pi[0], 0, w - 1)), int(np.clip(pi[1], 0, h - 1)))
        pt2 = (int(np.clip(pj[0], 0, w - 1)), int(np.clip(pj[1], 0, h - 1)))
        cv2.line(out, pt1, pt2, color, 2, cv2.LINE_AA)

    for idx in range(33):
        pt = xy[idx]
        if not np.all(np.isfinite(pt)):
            continue
        cx = int(np.clip(pt[0], 0, w - 1))
        cy = int(np.clip(pt[1], 0, h - 1))
        color = _kp_color(vis[idx])
        cv2.circle(out, (cx, cy), 4, color, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)

    label_map = {
        "knee":  LM[f"{prefix}_KNEE"],
        "hip":   LM[f"{prefix}_HIP"],
        "ankle": LM[f"{prefix}_ANKLE"],
    }
    for joint_name, kp_idx in label_map.items():
        val = angles.get(joint_name, np.nan)
        if not np.isfinite(val):
            continue
        pt = xy[kp_idx]
        if not np.all(np.isfinite(pt)):
            continue
        cx = int(np.clip(pt[0], 0, w - 1))
        cy = int(np.clip(pt[1], 0, h - 1))
        label = f"{joint_name[0].upper()}: {val:+.1f}"
        color = _kp_color(vis[kp_idx])
        cv2.putText(out, label, (cx + 6, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (cx + 6, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return out


def _write_diagnostic_video(
    video_path: Path,
    out_path: Path,
    xy_smooth: np.ndarray,
    vis_raw: np.ndarray,
    angles_per_frame: list[dict[str, float]],
    fps: float,
    visible_side: str,
    max_frames: int = 0,
) -> None:
    """Second pass through the source video: annotate each frame and write MP4."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for diagnostic pass: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok or t >= len(xy_smooth):
            break
        if max_frames and t >= max_frames:
            break
        annotated = _draw_skeleton(
            frame, xy_smooth[t], vis_raw[t], angles_per_frame[t], visible_side
        )
        writer.write(annotated)
        t += 1

    cap.release()
    writer.release()


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    video: Path,
    out_root: Path,
    mocap_txt: Path | None = None,
    visible_side: str = "right",
    max_frames: int = 0,
    cutoff_hz: float = 6.0,
    filter_order: int = 4,
    min_vis: float = 0.3,
    athlete_height_m: float | None = None,
    diagnostic_video: bool = True,
    on_progress: Callable[[int, int, "np.ndarray", "np.ndarray", "np.ndarray"], None] | None = None,
) -> Path:
    """
    Full sagittal markerless MoCap pipeline (v1.3.0).

    Args:
        video:             Path to input video.
        out_root:          Root output directory.
        mocap_txt:         Optional ground-truth MoCap file for comparison.
        visible_side:      "right" or "left" — which side faces the camera.
        max_frames:        0 = all frames; >0 limits for quick tests.
        cutoff_hz:         Butterworth low-pass cutoff in Hz.
        filter_order:      Butterworth order (default 4).
        min_vis:           Keypoint visibility anchor threshold (default 0.3).
        athlete_height_m:  Enables pixel→metre scaling from the athlete's height.
        diagnostic_video:  Write annotated overlay video (default True).
    """
    meta = get_video_meta(video)
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "derived").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    meta_dict = {
        "video": str(video), "fps": meta.fps, "frame_count": meta.frame_count,
        "width": meta.width, "height": meta.height, "visible_side": visible_side,
        "max_frames": max_frames, "cutoff_hz": cutoff_hz,
        "filter_order": filter_order, "min_vis": min_vis,
        "athlete_height_m": athlete_height_m,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_dict, indent=2))

    # ------------------------------------------------------------------
    # Pass 1: Pose inference
    # ------------------------------------------------------------------
    estimator = MediaPipePoseEstimator()
    xy_list, vis_list, frame_idx = [], [], []
    total_frames = meta.frame_count if not max_frames else min(max_frames, meta.frame_count)

    for i, frame in iter_frames(video, max_frames=max_frames):
        pr = estimator.predict(frame)
        xy_list.append(pr.xy)
        vis_list.append(pr.vis)
        frame_idx.append(i)
        if on_progress is not None:
            on_progress(i, total_frames, frame, pr.xy, pr.vis)

    estimator.close()

    xy  = np.stack(xy_list,  axis=0)   # (T, 33, 2)
    vis = np.stack(vis_list, axis=0)   # (T, 33)
    T   = xy.shape[0]

    # ------------------------------------------------------------------
    # Filtering: confidence-weighted spline → zero-phase Butterworth
    # ------------------------------------------------------------------
    xy_s = butterworth_smooth_xy(
        xy, vis,
        fps=meta.fps,
        cutoff_hz=cutoff_hz,
        order=filter_order,
        min_vis=min_vis,
    )

    # ------------------------------------------------------------------
    # Angle computation
    # ------------------------------------------------------------------
    rows: list[dict] = []
    angles_per_frame: list[dict[str, float]] = []

    for t in range(T):
        ang = joint_angles_sagittal(
            xy_s[t], vis[t],
            visible_side=visible_side,
            min_vis=min_vis,
        )
        angles_per_frame.append(ang)
        rows.append({
            "frame":                     frame_idx[t],
            "time_s":                    frame_idx[t] / meta.fps,
            f"{visible_side}_hip_deg":   ang["hip"],
            f"{visible_side}_knee_deg":  ang["knee"],
            f"{visible_side}_ankle_deg": ang["ankle"],
        })

    df_angles = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Anatomical clamping — replaces isolated outliers via rolling mean
    # ------------------------------------------------------------------
    angle_arrays = {
        "hip":   df_angles[f"{visible_side}_hip_deg"].to_numpy(float),
        "knee":  df_angles[f"{visible_side}_knee_deg"].to_numpy(float),
        "ankle": df_angles[f"{visible_side}_ankle_deg"].to_numpy(float),
    }
    clamped = clamp_angle_series(angle_arrays)

    # ------------------------------------------------------------------
    # Median smoothing — removes impulse noise / keypoint jitter
    # Applied after clamping so spikes are already reduced before median
    # Ankle uses kernel=13 (empirically optimal for sagittal jogging data)
    # ------------------------------------------------------------------
    smoothed = median_smooth_angles(clamped, kernel_sizes={
        "hip":   3,
        "knee":  3,
        "ankle": 13,
    })

    df_angles[f"{visible_side}_hip_deg"]   = smoothed["hip"]
    df_angles[f"{visible_side}_knee_deg"]  = smoothed["knee"]
    df_angles[f"{visible_side}_ankle_deg"] = smoothed["ankle"]

    # Sync smoothed values back into per-frame dicts for the overlay
    for t in range(T):
        angles_per_frame[t]["hip"]   = float(smoothed["hip"][t])
        angles_per_frame[t]["knee"]  = float(smoothed["knee"][t])
        angles_per_frame[t]["ankle"] = float(smoothed["ankle"][t])

    # ------------------------------------------------------------------
    # Optional: pixel → metre scaling + joint velocities
    # ------------------------------------------------------------------
    scaler: PixelScaler | None = None
    if athlete_height_m is not None:
        nose_idx = 0
        heel_idx = LM["R_HEEL"] if visible_side.lower().startswith("r") else LM["L_HEEL"]
        both_vis = (vis[:, nose_idx] >= min_vis) & (vis[:, heel_idx] >= min_vis)
        if both_vis.sum() > 5:
            px_heights = np.linalg.norm(
                xy_s[both_vis, nose_idx, :] - xy_s[both_vis, heel_idx, :], axis=1
            )
            scaler = PixelScaler.from_known_distance(
                float(np.median(px_heights)), athlete_height_m
            )

    if scaler is not None:
        pfx = "R" if visible_side.lower().startswith("r") else "L"
        joint_kp = {
            "hip":   LM[f"{pfx}_HIP"],
            "knee":  LM[f"{pfx}_KNEE"],
            "ankle": LM[f"{pfx}_ANKLE"],
        }
        vels = scaler.joint_velocities_m_per_s(xy_s, meta.fps, joint_kp)
        pd.DataFrame({
            "frame": frame_idx,
            "time_s": df_angles["time_s"].values,
            **{f"{k}_speed_m_s": v for k, v in vels.items()},
        }).to_csv(out_dir / "derived" / "velocities.csv", index=False)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    flat = {}
    for j in range(33):
        flat[f"j{j}_x"]   = xy_s[:, j, 0]
        flat[f"j{j}_y"]   = xy_s[:, j, 1]
        flat[f"j{j}_vis"] = vis[:, j]
    df_pose = pd.DataFrame(flat)
    df_pose.insert(0, "frame",  frame_idx)
    df_pose.insert(1, "time_s", df_angles["time_s"].values)

    df_pose.to_parquet(out_dir    / "derived" / "poses.parquet",            index=False)
    df_angles.to_csv(out_dir      / "derived" / "angles_sagittal.csv",      index=False)
    df_angles.to_parquet(out_dir  / "derived" / "angles_sagittal.parquet",  index=False)

    # ------------------------------------------------------------------
    # Optional MoCap comparison
    # ------------------------------------------------------------------
    if mocap_txt:
        df_mocap = read_mocap_angles_txt(mocap_txt)
        df_mocap.to_parquet(out_dir / "raw" / "mocap_angles.parquet", index=False)
        res = compare_video_to_mocap(df_angles, df_mocap, visible_side=visible_side)
        save_compare_outputs(res, out_dir / "reports")

    # ------------------------------------------------------------------
    # Pass 2: Diagnostic video overlay
    # ------------------------------------------------------------------
    if diagnostic_video:
        _write_diagnostic_video(
            video_path=video,
            out_path=out_dir / "diagnostic.mp4",
            xy_smooth=xy_s,
            vis_raw=vis,
            angles_per_frame=angles_per_frame,
            fps=meta.fps,
            visible_side=visible_side,
            max_frames=max_frames,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "version": "1.3.0",
        "frames_processed": int(T),
        "filter": {
            "type":           "butterworth_zerophase + median",
            "cutoff_hz":      cutoff_hz,
            "order":          filter_order,
            "min_vis_anchor": min_vis,
            "median_kernels": {"hip": 3, "knee": 3, "ankle": 13},
        },
        "scaling": {
            "px_per_meter":   scaler.px_per_meter if scaler else None,
            "athlete_height_m": athlete_height_m,
        },
        "outputs": {
            "poses_parquet":    "derived/poses.parquet",
            "angles_csv":       "derived/angles_sagittal.csv",
            "angles_parquet":   "derived/angles_sagittal.parquet",
            "velocities_csv":   "derived/velocities.csv" if scaler else None,
            "mocap_parquet":    "raw/mocap_angles.parquet" if mocap_txt else None,
            "diagnostic_video": "diagnostic.mp4" if diagnostic_video else None,
        },
    }
    (out_dir / "reports" / "summary.json").write_text(json.dumps(summary, indent=2))

    return out_dir