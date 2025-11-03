import numpy as np

def compute_joint_angle(a, b, c):
    """Compute angle at point b (degrees) formed by points a-b-c."""
    a, b, c = map(np.array, (a, b, c))
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))