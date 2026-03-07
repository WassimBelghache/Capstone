from pathlib import Path
import re
import pandas as pd

def read_mocap_angles_txt(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError(f"MoCap txt looks too short: {path}")

    # Header lines
    line2 = lines[1].strip()  # labels
    # Fix glued token like "...R_KNEE_AngleL_HIP_Angle..."
    line2 = re.sub(r"(?<=[A-Za-z0-9_])(?=[LR]_[A-Z])", " ", line2)
    tokens = [t for t in re.split(r"\s+", line2) if t]

    # Data starts after 3 header lines
    data_lines = lines[3:]

    rows = []
    max_fields = 0
    for ln in data_lines:
        s = ln.strip()
        if not s:
            continue
        fields = re.split(r"\s+", s)
        # try numeric conversion; keep as string if not
        row = []
        for x in fields:
            try:
                row.append(float(x))
            except ValueError:
                row.append(x)
        rows.append(row)
        if len(row) > max_fields:
            max_fields = len(row)

    # Pad short rows with NaN so DataFrame is rectangular
    for r in rows:
        if len(r) < max_fields:
            r.extend([float("nan")] * (max_fields - len(r)))

    df = pd.DataFrame(rows)

    ncols = df.shape[1]

    # Build column names.
    # Common case: data has a leading time/frame column not present in tokens
    if len(tokens) == ncols:
        cols = tokens
    elif len(tokens) + 1 == ncols:
        cols = ["time"] + tokens
    else:
        # If columns are a multiple of label count, suffix each repeated dimension
        if len(tokens) > 0 and ncols % len(tokens) == 0:
            rep = ncols // len(tokens)
            cols = []
            for name in tokens:
                for k in range(rep):
                    cols.append(f"{name}_{k}")
        else:
            cols = [f"col_{i}" for i in range(ncols)]

    df.columns = cols
    # Remove any leftover non-data rows (e.g., "ORIGINAL", "ITEM")
    # Make duplicate column names unique by appending _0/_1/_2
    seen = {}
    new_cols = []
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
    
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
