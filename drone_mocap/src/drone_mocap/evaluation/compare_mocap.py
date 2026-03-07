from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MocapCompareResult:
    axis_choice: dict
    sign_flip: dict
    metrics: dict
    compare_df: pd.DataFrame


def _interp_to(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    """1D linear interpolation with NaN-safe behavior."""
    mask = np.isfinite(x_src) & np.isfinite(y_src)
    if mask.sum() < 2:
        return np.full_like(x_tgt, np.nan, dtype=float)
    return np.interp(x_tgt, x_src[mask], y_src[mask]).astype(float)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])

def _shift_series(y: np.ndarray, k: int) -> np.ndarray:
    """Shift y by k frames (positive k delays the series). Pads with NaN."""
    out = np.full(y.shape, np.nan, dtype=float)
    y = y.astype(float, copy=False)
    if k == 0:
        return y.copy()
    if k > 0:
        out[k:] = y[:-k]
    else:
        kk = -k
        out[:-kk] = y[kk:]
    return out

def _best_ankle_transform(v: np.ndarray, m: np.ndarray) -> tuple[str, np.ndarray, float]:
    """
    Try common ankle transforms on the VIDEO signal and pick the one with best correlation to mocap.
    Returns (name, v_transformed, corr).
    """
    cands = {
        "raw": v,
        "neg": -v,
        "180_minus": 180.0 - v,
        "v_minus_90": v - 90.0,
        "90_minus_v": 90.0 - v,
    }
    best_name = "raw"
    best_v = v
    best_r = -np.inf
    for name, vv in cands.items():
        r = _corr(vv, m)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_name = name
            best_v = vv
    return best_name, best_v, float(best_r)



def best_shift_frames(v: np.ndarray, m: np.ndarray, max_shift: int = 60) -> tuple[int, float]:
    """Find integer frame shift maximizing correlation."""
    best_k = 0
    best_r = -np.inf
    for k in range(-max_shift, max_shift + 1):
        ms = _shift_series(m, k)
        r = _corr(v, ms)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_k = k
    return best_k, float(best_r)


def compare_video_to_mocap(
    video_angles: pd.DataFrame,
    mocap_angles: pd.DataFrame,
    visible_side: str = "right",
) -> MocapCompareResult:
    """
    video_angles: derived/angles_sagittal.parquet
      columns: time_s, right_knee_deg, right_hip_deg, right_ankle_deg (or left_*)
    mocap_angles: raw/mocap_angles.parquet
      columns: time, R_KNEE_Angle, R_KNEE_Angle_1, R_KNEE_Angle_2, ...
    """

    side_prefix = "R" if visible_side.lower().startswith("r") else "L"
    joints = ["HIP", "KNEE", "ANKLE"]

    # Video signals
    vtime = video_angles["time_s"].to_numpy(dtype=float)

    vcols = {
        "HIP": f"{visible_side}_hip_deg",
        "KNEE": f"{visible_side}_knee_deg",
        "ANKLE": f"{visible_side}_ankle_deg",
    }

    # MoCap time
    mtime = mocap_angles["time"].to_numpy(dtype=float)

    axis_choice: dict[str, str | None] = {}
    sign_flip: dict[str, bool] = {}
    metrics: dict[str, dict] = {}

    out = pd.DataFrame({"time_s": vtime})

    best_series_by_joint: dict[str, np.ndarray] = {}
    video_by_joint: dict[str, np.ndarray] = {}

    for j in joints:
        v = video_angles[vcols[j]].to_numpy(dtype=float)
        video_by_joint[j] = v
        out[f"video_{j.lower()}"] = v

        base = f"{side_prefix}_{j}_Angle"
        cand_cols = [base, f"{base}_1", f"{base}_2"]

        if any(c not in mocap_angles.columns for c in cand_cols):
            axis_choice[j] = None
            sign_flip[j] = False
            best_series_by_joint[j] = np.full_like(vtime, np.nan, dtype=float)
            out[f"mocap_{j.lower()}"] = np.nan
            out[f"error_{j.lower()}"] = np.nan
            metrics[j] = {"rmse": np.nan, "mae": np.nan, "corr": np.nan}
            continue

        best_col = None
        best_score = -np.inf
        best_flip = False
        best_series = None

        for c in cand_cols:
            m = mocap_angles[c].to_numpy(dtype=float)
            m_rs = _interp_to(mtime, m, vtime)

            r = _corr(v, m_rs)
            r_flip = _corr(v, -m_rs)

            if np.isfinite(r) and r > best_score:
                best_col, best_score, best_flip, best_series = c, r, False, m_rs
            if np.isfinite(r_flip) and r_flip > best_score:
                best_col, best_score, best_flip, best_series = c, r_flip, True, -m_rs

        axis_choice[j] = best_col
        sign_flip[j] = bool(best_flip)

        best_series_by_joint[j] = best_series
        out[f"mocap_{j.lower()}"] = best_series

    # ---- 2) Compute best integer frame shift using KNEE, then shift ALL MoCap series ----
    # Search window: +/- ~1 second in frames
    if "KNEE" in best_series_by_joint:
        v_knee = video_by_joint["KNEE"]
        m_knee = best_series_by_joint["KNEE"]
        dt = float(np.median(np.diff(vtime))) if len(vtime) > 2 else 0.0
        max_shift = int(round(3.0 / dt)) if dt > 0 else 60
        k, knee_r_at_k = best_shift_frames(v_knee, m_knee, max_shift=max_shift)
        metrics["_knee_corr_at_shift"] = float(knee_r_at_k)

    else:
        k = 0

    for j in joints:
        m_best = best_series_by_joint.get(j, None)
        if m_best is None:
            continue
        m_best_shifted = _shift_series(m_best, k)
        best_series_by_joint[j] = m_best_shifted
        out[f"mocap_{j.lower()}"] = m_best_shifted
        out[f"error_{j.lower()}"] = out[f"video_{j.lower()}"] - out[f"mocap_{j.lower()}"]

    # ---- 3) Compute metrics AFTER alignment ----
    for j in joints:
        v = video_by_joint.get(j, None)
        m_best = best_series_by_joint.get(j, None)
        if v is None or m_best is None:
            metrics[j] = {"rmse": np.nan, "mae": np.nan, "corr": np.nan}
            continue

        # --- ankle definition transform (video-side) ---
        if j == "ANKLE":
            tname, v_use, r0 = _best_ankle_transform(v, m_best)
            metrics["_ankle_transform"] = tname
            metrics["_ankle_corr_pre_metrics"] = float(r0)
            # overwrite in output dataframe so plots match what you evaluated
            out["video_ankle"] = v_use
            v = v_use

        msk = np.isfinite(v) & np.isfinite(m_best)
        if msk.sum() < 5:
            rmse = mae = corr = np.nan
        else:
            err = v[msk] - m_best[msk]
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            corr = _corr(v, m_best)

        metrics[j] = {"rmse": rmse, "mae": mae, "corr": corr}


    # record chosen time shift (frames) for debugging
    metrics["_time_shift_frames"] = int(k)

    return MocapCompareResult(
        axis_choice=axis_choice,
        sign_flip=sign_flip,
        metrics=metrics,
        compare_df=out,
    )



def save_compare_outputs(result: MocapCompareResult, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)

    result.compare_df.to_csv(reports_dir / "compare_mocap.csv", index=False)
    result.compare_df.to_parquet(reports_dir / "compare_mocap.parquet", index=False)

    payload = {
        "axis_choice": result.axis_choice,
        "sign_flip": result.sign_flip,
        "metrics": result.metrics,
    }
    (reports_dir / "metrics_mocap.json").write_text(json.dumps(payload, indent=2))
