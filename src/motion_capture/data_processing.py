from src.utils.data_loader import load_trc, extract_markers

def preprocess_trc(file_path):
    """Load TRC and prepare structured marker data."""
    df, marker_names = load_trc(file_path)
    markers = extract_markers(df)
    return df, markers