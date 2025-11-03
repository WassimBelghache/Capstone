"""
trc_loader.py
-------------
Loads and parses Cortex .trc motion capture files.
"""

import pandas as pd

def load_trc(file_path: str) -> pd.DataFrame:
    """Load a .trc file and return a DataFrame with marker positions."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header lines (usually 6)
    data_start = 6
    headers = lines[data_start - 1].strip().split()
    df = pd.read_csv(file_path, sep=r"\s+", skiprows=data_start, names=headers)

    # Drop frame/time columns if they exist
    df.drop(columns=["Frame#", "Time"], inplace=True, errors="ignore")
    return df


def extract_markers(df: pd.DataFrame):
    """Return {marker_name: np.array([[x,y,z], ...])}"""
    markers = {}
    for col in df.columns[::3]:  # Every 3 columns = one marker
        marker = col.split("_")[0]
        markers[marker] = df[[f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]].values
    return markers
