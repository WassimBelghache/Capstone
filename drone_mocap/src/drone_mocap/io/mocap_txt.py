"""
MoCap / SPT angle file reader.

Dispatch order:
  1.  SPT CSV  — file ends in .csv AND header contains 'timestamp_ms' or
                 flexion/extension column names (M_Treadmill_Jogging.angles.csv).
  2.  Legacy tab-separated MoCap .txt — original regex-based parser as fallback.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# SPT CSV (Sport Product Testing) path
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

    Transformations:
      • timestamp_ms → time  (ms ÷ 1000 = seconds)
      • Rows where state == "static" are dropped (calibration frames).
      • Non-numeric columns (other than time) are dropped.

    Column names are preserved verbatim so the fuzzy matcher in
    compare_mocap.py can recognise patterns like RIGHT_KNEE_flexion.
    """
    df = pd.read_csv(path)

    # Drop static calibration frames
    if "state" in df.columns:
        df = df[df["state"].str.strip().str.lower() != "static"].copy()
        df = df.drop(columns=["state"])

    # Normalise time column
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

    # Coerce all non-time columns to numeric; drop columns that are entirely NaN
    for col in [c for c in df.columns if c != "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop(columns=[c for c in df.columns if c != "time" and df[c].isna().all()])

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df[df["time"].notna()].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Legacy tab-separated MoCap .txt path (original parser, kept as fallback)
# ---------------------------------------------------------------------------

def _read_tab_separated(path: Path) -> pd.DataFrame:
    """Original regex-based parser for the lab's tab-separated .txt export."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError(f"MoCap txt looks too short: {path}")

    line2 = lines[1].strip()
    # Fix glued tokens like "...R_KNEE_AngleL_HIP_Angle..."
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

    # Deduplicate column names
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
# Public entry point — name kept for backward compatibility with run.py
# ---------------------------------------------------------------------------

def read_mocap_angles_txt(path: str | Path) -> pd.DataFrame:
    """
    Unified MoCap / SPT angle file reader.

    Dispatches automatically between:
      • SPT CSV  (e.g. M_Treadmill_Jogging.angles.csv)
      • Legacy tab-separated lab .txt export

    Returns a DataFrame with a ``time`` column (seconds) and one or more
    angle columns in their original naming convention.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MoCap file not found: {path}")
    if _sniff_spt_csv(path):
        return _read_spt_csv(path)
    return _read_tab_separated(path)
