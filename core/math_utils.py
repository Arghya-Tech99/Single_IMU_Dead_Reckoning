import numpy as np
from scipy.spatial.transform import Rotation

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

def quat_normalize(rot):
    q = rot.as_quat()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return rot
    return Rotation.from_quat(q / n)

def get_rpy_swing_safe(R):
    cth = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    theta = np.arctan2(-R[2, 0], cth)
    if cth < 1e-6:
        phi = np.arctan2(R[0, 1], R[1, 1])
        psi = 0.0
    else:
        phi = np.arctan2(R[2, 1], R[2, 2])
        psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([phi, theta, psi])



def force_sphere(points, radius=None):
    """
    Project each point onto a sphere of given radius centered at origin.
    If radius is None, use the average norm of the points.
    """
    import numpy as np
    points = np.asarray(points)
    norms = np.linalg.norm(points, axis=1)
    if radius is None:
        radius = np.mean(norms)
    with np.errstate(divide='ignore', invalid='ignore'):
        projected = points * (radius / norms[:, np.newaxis])
        projected[~np.isfinite(projected)] = 0.0
    return projected
