"""
Kinematic comparison engine — video-derived angles vs. MoCap / SPT ground truth.

Key fixes over v1.1:
  • best_shift_frames now tries both signs of the reference signal to handle
    sign convention mismatches between systems.
  • Axis selection now picks the best axis+sign combination jointly rather
    than selecting axis first then sign separately.
  • save_compare_outputs writes a 3-subplot matplotlib figure alongside
    the existing CSV / JSON outputs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MocapCompareResult:
    axis_choice: dict[str, str | None]
    sign_flip: dict[str, bool]
    metrics: dict
    compare_df: pd.DataFrame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interp_to(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    """1-D linear interpolation with NaN-safe behavior."""
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
    """Shift y by k samples (positive k delays the series). Pads with NaN."""
    out = np.full(y.shape, np.nan, dtype=float)
    y = y.astype(float, copy=False)
    if k == 0:
        return y.copy()
    if k > 0:
        out[k:] = y[:-k]
    else:
        out[:k] = y[-k:]
    return out


def _best_ankle_transform(
    v: np.ndarray, m: np.ndarray
) -> tuple[str, np.ndarray, float]:
    """
    Try common ankle transforms on the video signal and pick the one with the
    best correlation to the reference.
    """
    cands = {
        "raw":        v,
        "neg":        -v,
        "180_minus":  180.0 - v,
        "v_minus_90": v - 90.0,
        "90_minus_v": 90.0 - v,
    }
    best_name, best_v, best_r = "raw", v, -np.inf
    for name, vv in cands.items():
        r = _corr(vv, m)
        if np.isfinite(r) and r > best_r:
            best_name, best_v, best_r = name, vv, r
    return best_name, best_v, float(best_r)


def best_shift_frames(
    v: np.ndarray, m: np.ndarray, max_shift: int = 60
) -> tuple[int, float]:
    """
    Return the integer frame shift that maximises cross-correlation.
    Tries both the original and sign-flipped reference signal to handle
    convention mismatches between systems.
    """
    best_k, best_r = 0, -np.inf
    for sign in [1.0, -1.0]:
        m_try = sign * m
        for k in range(-max_shift, max_shift + 1):
            r = _corr(v, _shift_series(m_try, k))
            if np.isfinite(r) and r > best_r:
                best_r, best_k = r, k
    return best_k, float(best_r)


# ---------------------------------------------------------------------------
# Fuzzy column matching
# ---------------------------------------------------------------------------

def _fuzzy_find_columns(
    mocap_cols: list[str],
    joint: str,
    side: str,
) -> list[str]:
    """
    Find MoCap DataFrame columns that likely contain the angle for the
    requested joint and side.
    """
    joint_token = joint.lower()
    side_tokens: set[str]
    if side.upper() == "R":
        side_tokens = {"right", "r"}
        exclude_tokens = {"left", "l"}
    else:
        side_tokens = {"left", "l"}
        exclude_tokens = {"right", "r"}

    candidates: list[str] = []
    for col in mocap_cols:
        tokens = set(re.split(r"[_\-\s]+", col.lower()))
        has_joint = joint_token in tokens
        has_side = bool(tokens & side_tokens)
        has_wrong_side = bool(tokens & exclude_tokens)
        if has_joint and has_side and not has_wrong_side:
            candidates.append(col)

    return candidates


# ---------------------------------------------------------------------------
# Primary comparison entry point
# ---------------------------------------------------------------------------

def compare_video_to_mocap(
    video_angles: pd.DataFrame,
    mocap_angles: pd.DataFrame,
    visible_side: str = "right",
) -> MocapCompareResult:
    """
    Align and compare video-derived kinematic angles to a MoCap / SPT reference.
    """
    side_prefix = "R" if visible_side.lower().startswith("r") else "L"
    joints = ["HIP", "KNEE", "ANKLE"]

    vtime = video_angles["time_s"].to_numpy(dtype=float)
    mtime = mocap_angles["time"].to_numpy(dtype=float)

    vcols = {
        "HIP":   f"{visible_side}_hip_deg",
        "KNEE":  f"{visible_side}_knee_deg",
        "ANKLE": f"{visible_side}_ankle_deg",
    }

    axis_choice: dict[str, str | None] = {}
    sign_flip: dict[str, bool] = {}
    metrics: dict = {}
    out = pd.DataFrame({"time_s": vtime})

    best_series_by_joint: dict[str, np.ndarray] = {}
    video_by_joint: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Step 1 & 2 combined: Find best axis + sign + shift for each joint.
    #         The knee is used as the reference for the global shift,
    #         but each joint's axis/sign is selected AFTER applying that shift.
    # ------------------------------------------------------------------

    # First pass: find the global shift using ALL joints combined
    # We do this by finding the best shift for each candidate and picking
    # the shift that works best for the knee (most periodic/clean signal)
    dt = float(np.median(np.diff(vtime))) if len(vtime) > 2 else 0.0
    max_shift = int(round(3.0 / dt)) if dt > 0 else 180

    # Find best shift from knee signal trying all axes and signs
    knee_cands = _fuzzy_find_columns(mocap_angles.columns.tolist(), "KNEE", side_prefix)
    best_global_shift = 0
    best_global_r = -np.inf

    for c in knee_cands:
        m_raw = mocap_angles[c].to_numpy(dtype=float)
        m_rs = _interp_to(mtime, m_raw, vtime)
        v_knee = video_angles[vcols["KNEE"]].to_numpy(dtype=float)
        for sign in [1.0, -1.0]:
            k, r = best_shift_frames(v_knee, sign * m_rs, max_shift=max_shift)
            if np.isfinite(r) and r > best_global_r:
                best_global_r = r
                best_global_shift = k

    k = best_global_shift
    metrics["_knee_corr_at_shift"] = float(best_global_r)

    # Second pass: for each joint, pick best axis+sign AFTER applying global shift
    for j in joints:
        v = video_angles[vcols[j]].to_numpy(dtype=float)
        video_by_joint[j] = v
        out[f"video_{j.lower()}"] = v

        cand_cols = _fuzzy_find_columns(mocap_angles.columns.tolist(), j, side_prefix)

        if not cand_cols:
            axis_choice[j] = None
            sign_flip[j] = False
            best_series_by_joint[j] = np.full_like(vtime, np.nan, dtype=float)
            out[f"mocap_{j.lower()}"] = np.nan
            out[f"error_{j.lower()}"] = np.nan
            metrics[j] = {"rmse": np.nan, "mae": np.nan, "corr": np.nan,
                          "matched_col": None}
            continue

        best_col: str | None = None
        best_score = -np.inf
        best_flip = False
        best_series: np.ndarray = np.full_like(vtime, np.nan, dtype=float)

        # Try every candidate column with both signs AFTER applying global shift
        for c in cand_cols:
            m_raw = mocap_angles[c].to_numpy(dtype=float)
            m_rs = _interp_to(mtime, m_raw, vtime)
            for flip, m_try in [(False, m_rs), (True, -m_rs)]:
                m_shifted = _shift_series(m_try, k)
                r = _corr(v, m_shifted)
                if np.isfinite(r) and r > best_score:
                    best_col, best_score, best_flip, best_series = c, r, flip, m_shifted

        axis_choice[j] = best_col
        sign_flip[j] = bool(best_flip)
        best_series_by_joint[j] = best_series
        out[f"mocap_{j.lower()}"] = best_series
        out[f"error_{j.lower()}"] = out[f"video_{j.lower()}"] - out[f"mocap_{j.lower()}"]

    # ------------------------------------------------------------------
    # Step 3: Compute final metrics after alignment.
    # ------------------------------------------------------------------
    for j in joints:
        v     = video_by_joint.get(j)
        m_best = best_series_by_joint.get(j)
        if v is None or m_best is None:
            metrics[j] = {"rmse": np.nan, "mae": np.nan, "corr": np.nan,
                          "matched_col": axis_choice.get(j)}
            continue

        # Ankle convention auto-transform
        if j == "ANKLE":
            tname, v_use, r0 = _best_ankle_transform(v, m_best)
            metrics["_ankle_transform"] = tname
            metrics["_ankle_corr_pre_metrics"] = float(r0)
            out["video_ankle"] = v_use
            v = v_use

        msk = np.isfinite(v) & np.isfinite(m_best)
        if msk.sum() < 5:
            rmse = mae = corr = np.nan
        else:
            err  = v[msk] - m_best[msk]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae  = float(np.mean(np.abs(err)))
            corr = _corr(v, m_best)

        metrics[j] = {
            "rmse": rmse,
            "mae":  mae,
            "corr": corr,
            "matched_col": axis_choice.get(j),
        }

    metrics["_time_shift_frames"] = int(k)

    return MocapCompareResult(
        axis_choice=axis_choice,
        sign_flip=sign_flip,
        metrics=metrics,
        compare_df=out,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _save_comparison_plot(result: MocapCompareResult, out_path: Path) -> None:
    """Write a 3-subplot figure comparing video angles to MoCap reference."""
    joints = ["hip", "knee", "ankle"]
    df = result.compare_df
    t = df["time_s"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Video vs. MoCap — Kinematic Comparison", fontsize=13, fontweight="bold")

    colors = {"video": "#2196F3", "mocap": "#F44336"}

    for ax, j in zip(axes, joints):
        v_col = f"video_{j}"
        m_col = f"mocap_{j}"
        v_sig = df[v_col].to_numpy(dtype=float) if v_col in df.columns else None
        m_sig = df[m_col].to_numpy(dtype=float) if m_col in df.columns else None

        if v_sig is not None:
            ax.plot(t, v_sig, color=colors["video"], lw=1.4,
                    label="Video (DroCap)", zorder=3)
        if m_sig is not None and not np.all(np.isnan(m_sig)):
            ax.plot(t, m_sig, color=colors["mocap"], lw=1.4,
                    linestyle="--", label="MoCap / SPT", zorder=2)

        jkey = j.upper()
        m = result.metrics.get(jkey, {})
        rmse = m.get("rmse", np.nan)
        corr = m.get("corr", np.nan)
        matched = m.get("matched_col") or axis_choice_str(result.axis_choice.get(jkey))
        info = f"RMSE={rmse:.2f}°  r={corr:.3f}  ref: {matched}"
        ax.set_title(f"{j.capitalize()}  —  {info}", fontsize=9, loc="left")
        ax.set_ylabel("Angle (°)")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def axis_choice_str(v: object) -> str:
    return str(v) if v is not None else "not found"


def save_compare_outputs(result: MocapCompareResult, reports_dir: Path) -> None:
    """Write all comparison artefacts to reports_dir."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    result.compare_df.to_csv(reports_dir / "compare_mocap.csv", index=False)
    result.compare_df.to_parquet(reports_dir / "compare_mocap.parquet", index=False)

    payload = {
        "axis_choice": result.axis_choice,
        "sign_flip":   result.sign_flip,
        "metrics":     result.metrics,
    }
    (reports_dir / "metrics_mocap.json").write_text(json.dumps(payload, indent=2))

    _save_comparison_plot(result, reports_dir / "comparison_plot.png")