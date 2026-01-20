def fix_rotation(frame: np.ndarray, cap: cv2.VideoCapture) -> np.ndarray:
    """
    Fix video rotation based on metadata.
    
    Returns:
        Rotated frame if needed
    """
    try:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return frame


def to_pixel_coords(landmark, width: int, height: int) -> np.ndarray:
    """
    Convert normalized landmark to pixel coordinates.
    
    Returns:
        Pixel coordinates as [x, y] array
    """
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

