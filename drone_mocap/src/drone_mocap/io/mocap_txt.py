"""
MoCap / SPT angle file reader.

Dispatch order:
  1.  SPT CSV  — file ends in .csv AND header contains 'timestamp_ms' or
                 flexion/extension column names.
  2.  Sponsor .txt — tab-separated with X/Y/Z sub-columns per joint,
                     frame numbers only (no time column), captured at a
                     known rate (default 250 Hz based on TRC metadata).
  3.  Legacy tab-separated MoCap .txt — original regex-based parser as fallback.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SPT CSV path
# ---------------------------------------------------------------------------

def _sniff_spt_csv(path: Path) -> bool:
    """Return True when the file looks like an SPT angles CSV."""
    if path.suffix.lower() != ".csv":
        return False
    try:
        header_cols = set(pd.read_csv(path, nrows=0).columns.str.lower())
        keywords = {"timestamp_ms", "flexion", "extension", "state"}
        return bool(keywords & header_cols)
    except Exception:
        return False


def _read_spt_csv(path: Path) -> pd.DataFrame:
    """
    Parse an SPT angles CSV into a normalised DataFrame.
    """
    df = pd.read_csv(path)

    if "state" in df.columns:
        df = df[df["state"].str.strip().str.lower() != "static"].copy()
        df = df.drop(columns=["state"])

    if "timestamp_ms" in df.columns:
        df["time"] = df["timestamp_ms"].astype(float) / 1000.0
        df = df.drop(columns=["timestamp_ms"])
    elif "time" not in df.columns:
        time_cands = [c for c in df.columns if re.search(r"time|timestamp", c, re.IGNORECASE)]
        if not time_cands:
            raise ValueError(f"SPT CSV '{path.name}' has no recognisable time column.")
        src = time_cands[0]
        df["time"] = pd.to_numeric(df[src], errors="coerce")
        if "ms" in src.lower():
            df["time"] = df["time"] / 1000.0
        if src != "time":
            df = df.drop(columns=[src])

    for col in [c for c in df.columns if c != "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop(columns=[c for c in df.columns if c != "time" and df[c].isna().all()])

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df[df["time"].notna()].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Sponsor .txt path  (X/Y/Z sub-columns, frame numbers, 250 Hz)
# ---------------------------------------------------------------------------

def _sniff_sponsor_txt(path: Path) -> bool:
    """
    Return True when the file matches the sponsor's angle export format.
    Line 1 (0-indexed) contains joint names like L_ANKLE_angle separated by tabs.
    Line 4 (0-indexed) contains ITEM  X  Y  Z  X  Y  Z ...
    """
    if path.suffix.lower() != ".txt":
        return False
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 5:
            return False
        # Line 1: joint names
        row1_tokens = set(lines[1].upper().split("\t"))
        has_joint = bool({"L_ANKLE_ANGLE", "L_KNEE_ANGLE", "L_HIP_ANGLE",
                          "R_ANKLE_ANGLE", "R_KNEE_ANGLE", "R_HIP_ANGLE"} & row1_tokens)
        # Line 4: ITEM X Y Z ...
        row4 = lines[4].strip().upper()
        has_item_xyz = row4.startswith("ITEM") and "\tX\t" in row4
        return has_joint and has_item_xyz
    except Exception:
        return False


def _read_sponsor_txt(path: Path, capture_hz: float = 250.0) -> pd.DataFrame:
    """
    Parse the sponsor's tab-separated angle .txt export.

    File structure (0-indexed lines):
      Line 0: file path metadata (ignored)
      Line 1: joint names — each repeated 3 times for X/Y/Z, tab-separated
              e.g. \\tL_ANKLE_angle\\tL_ANKLE_angle\\tL_ANKLE_angle\\tL_KNEE_angle...
      Line 2: component type (LINK_MODEL_BASED — ignored)
      Line 3: ORIGINAL tags (ignored)
      Line 4: ITEM\\tX\\tY\\tZ\\tX\\tY\\tZ... (sub-column labels)
      Line 5+: data — first column is frame number (1-based), then X/Y/Z values

    Each joint has 3 sub-columns (X, Y, Z Euler angles in degrees).
    All three axes are kept so the fuzzy matcher in compare_mocap.py can
    select the best one automatically.

    Time is computed as: time = (frame_number - 1) / capture_hz
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # --- Line 1: joint names (tab-separated, first token empty due to leading tab) ---
    joint_name_tokens = lines[1].split("\t")
    joint_names = [t.strip() for t in joint_name_tokens if t.strip()]

    # Build column names: assign _X, _Y, _Z suffix cycling through each joint
    axis_cycle = ["X", "Y", "Z"]
    col_names: list[str] = []
    axis_counters: dict[str, int] = {}
    for jname in joint_names:
        count = axis_counters.get(jname, 0)
        suffix = axis_cycle[count % 3]
        col_names.append(f"{jname}_{suffix}")
        axis_counters[jname] = count + 1

    # --- Lines 5+: data rows ---
    data_rows: list[list[float]] = []
    for ln in lines[5:]:
        s = ln.strip()
        if not s:
            continue
        fields = s.split("\t")
        try:
            row = [float(x) for x in fields if x.strip()]
        except ValueError:
            continue
        if row:
            data_rows.append(row)

    if not data_rows:
        raise ValueError(f"No data rows found in {path.name}")

    arr = np.array(data_rows, dtype=float)

    # First column is frame number (1-based)
    frame_nums = arr[:, 0]
    data_vals  = arr[:, 1:]

    # Trim or pad col_names to match actual data columns
    n_data_cols = data_vals.shape[1]
    if len(col_names) > n_data_cols:
        col_names = col_names[:n_data_cols]
    elif len(col_names) < n_data_cols:
        col_names += [f"col_{i}" for i in range(len(col_names), n_data_cols)]

    df = pd.DataFrame(data_vals, columns=col_names)

    # Compute time from frame number and capture rate
    df.insert(0, "time", (frame_nums - 1.0) / capture_hz)

    return df


# ---------------------------------------------------------------------------
# Legacy tab-separated MoCap .txt path (kept as fallback)
# ---------------------------------------------------------------------------

def _read_tab_separated(path: Path) -> pd.DataFrame:
    """Original regex-based parser for the lab's tab-separated .txt export."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError(f"MoCap txt looks too short: {path}")

    line2 = lines[1].strip()
    line2 = re.sub(r"(?<=[A-Za-z0-9_])(?=[LR]_[A-Z])", " ", line2)
    tokens = [t for t in re.split(r"\s+", line2) if t]

    rows: list[list] = []
    max_fields = 0
    for ln in lines[3:]:
        s = ln.strip()
        if not s:
            continue
        fields = re.split(r"\s+", s)
        row: list = []
        for x in fields:
            try:
                row.append(float(x))
            except ValueError:
                row.append(x)
        rows.append(row)
        if len(row) > max_fields:
            max_fields = len(row)

    for r in rows:
        if len(r) < max_fields:
            r.extend([float("nan")] * (max_fields - len(r)))

    df = pd.DataFrame(rows)
    ncols = df.shape[1]

    if len(tokens) == ncols:
        cols = tokens
    elif len(tokens) + 1 == ncols:
        cols = ["time"] + tokens
    else:
        if len(tokens) > 0 and ncols % len(tokens) == 0:
            rep = ncols // len(tokens)
            cols = [f"{name}_{k}" for name in tokens for k in range(rep)]
        else:
            cols = [f"col_{i}" for i in range(ncols)]

    df.columns = cols

    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    df.columns = new_cols

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df[df["time"].notna()].reset_index(drop=True)
    for c in [c for c in df.columns if c != "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def read_mocap_angles_txt(path: str | Path, capture_hz: float = 250.0) -> pd.DataFrame:
    """
    Unified MoCap / SPT angle file reader.

    Dispatches automatically between:
      • SPT CSV  (e.g. M_Treadmill_Jogging.angles.csv)
      • Sponsor .txt with X/Y/Z sub-columns and frame numbers (250 Hz)
      • Legacy tab-separated lab .txt export

    Args:
        path:        Path to the MoCap reference file.
        capture_hz:  Capture rate in Hz — only used for the sponsor .txt format.
                     Confirmed as 250 Hz from the TRC file metadata.

    Returns:
        DataFrame with a ``time`` column (seconds) and angle columns.
        For the sponsor .txt format, columns are named like:
          L_KNEE_angle_X, L_KNEE_angle_Y, L_KNEE_angle_Z,
          R_HIP_angle_X,  R_HIP_angle_Y,  R_HIP_angle_Z, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MoCap file not found: {path}")
    if _sniff_spt_csv(path):
        return _read_spt_csv(path)
    if _sniff_sponsor_txt(path):
        return _read_sponsor_txt(path, capture_hz=capture_hz)
    return _read_tab_separated(path)