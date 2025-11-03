import pandas as pd
import numpy as np

def load_trc(file_path: str):
    """
    Proper TRC file loader that handles the actual format
    """
    print(f"Loading TRC file: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the data header line (contains marker names)
    header_line_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Frame#'):
            header_line_idx = i
            break
    
    if header_line_idx is None:
        raise ValueError("Could not find header line in TRC file")
    
    # Parse marker names from header
    header_parts = lines[header_line_idx].strip().split('\t')
    marker_names = [name for name in header_parts[2:] if name]  # Skip Frame# and Time
    
    # Read data (skip header lines)
    data_lines = []
    for line in lines[header_line_idx + 1:]:
        if line.strip() and not line.startswith('EndOfFile'):
            data_lines.append(line.strip().split('\t'))
    
    # Create proper column names
    columns = ['Frame', 'Time']
    for marker in marker_names:
        columns.extend([f'{marker}_X', f'{marker}_Y', f'{marker}_Z'])
    
    # Create DataFrame
    df = pd.DataFrame(data_lines, columns=columns)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded {len(df)} frames with {len(marker_names)} markers")
    return df, marker_names

def extract_markers(df: pd.DataFrame):
    """Return dictionary of marker data"""
    markers = {}
    # Get all unique marker names from columns
    marker_names = list(set([col.split('_')[0] for col in df.columns if '_X' in col]))
    
    for marker in marker_names:
        if all(f'{marker}_{coord}' in df.columns for coord in ['X', 'Y', 'Z']):
            markers[marker] = df[[f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
    
    return markers