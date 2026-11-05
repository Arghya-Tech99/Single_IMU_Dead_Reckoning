"""
NASA Coding Guidelines Level Architecture: Math Utilities
Provides mathematical operations without hidden side-effects.
"""

import numpy as np
from scipy.spatial.transform import Rotation

def wrap(a: float) -> float:
    """Wraps an angle to the range [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def skew(v: np.ndarray) -> np.ndarray:
    """Returns the skew-symmetric matrix of a 3D vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

def quat_normalize(rot: Rotation) -> Rotation:
    """Normalizes a SciPy Rotation object safely."""
    q = rot.as_quat()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return rot
    return Rotation.from_quat(q / n)

def get_rpy_swing_safe(R: np.ndarray) -> np.ndarray:
    """Extracts Roll, Pitch, Yaw from a rotation matrix safely."""
    cth = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    theta = np.arctan2(-R[2, 0], cth)
    if cth < 1e-6:
        phi = np.arctan2(R[0, 1], R[1, 1])
        psi = 0.0
    else:
        phi = np.arctan2(R[2, 1], R[2, 2])
        psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([phi, theta, psi])
