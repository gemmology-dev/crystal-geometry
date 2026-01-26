"""
Geometric transformations for crystal twinning.

Provides rotation matrices, reflection matrices, and common crystallographic
directions used in twin operations.
"""

import numpy as np


def rotation_matrix_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Create 3x3 rotation matrix from axis-angle representation.

    Uses Rodrigues' rotation formula.

    Args:
        axis: 3D rotation axis (will be normalized)
        angle_deg: Rotation angle in degrees

    Returns:
        3x3 rotation matrix
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.eye(3)
    axis = axis / norm

    angle_rad = np.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1 - c

    x, y, z = axis

    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def rotation_matrix_4x4(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Create 4x4 homogeneous rotation matrix.

    Args:
        axis: 3D rotation axis
        angle_deg: Rotation angle in degrees

    Returns:
        4x4 homogeneous transformation matrix
    """
    R = rotation_matrix_axis_angle(axis, angle_deg)
    result = np.eye(4)
    result[:3, :3] = R
    return result


def translation_matrix_4x4(offset: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous translation matrix.

    Args:
        offset: 3D translation vector

    Returns:
        4x4 homogeneous transformation matrix
    """
    result = np.eye(4)
    result[:3, 3] = offset
    return result


def reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Create 3x3 reflection matrix across plane with given normal.

    The reflection formula is: R = I - 2 * n * n^T
    where n is the unit normal.

    Args:
        normal: Normal vector to the reflection plane (will be normalized)

    Returns:
        3x3 reflection matrix
    """
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return np.eye(3)
    normal = normal / norm
    n = normal.reshape(3, 1)
    return np.eye(3) - 2 * (n @ n.T)


# Common crystallographic directions (normalized)
DIRECTIONS: dict[str, np.ndarray] = {
    "[100]": np.array([1.0, 0.0, 0.0]),
    "[010]": np.array([0.0, 1.0, 0.0]),
    "[001]": np.array([0.0, 0.0, 1.0]),
    "[110]": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
    "[111]": np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    "[-111]": np.array([-1.0, 1.0, 1.0]) / np.sqrt(3),
    "[1-11]": np.array([1.0, -1.0, 1.0]) / np.sqrt(3),
    "[11-1]": np.array([1.0, 1.0, -1.0]) / np.sqrt(3),
    "[1-10]": np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
    "[11-2]": np.array([1.0, 1.0, -2.0]) / np.sqrt(6),
    "[021]": np.array([0.0, 2.0, 1.0]) / np.sqrt(5),
}


def get_direction(name: str) -> np.ndarray:
    """Get a crystallographic direction by Miller-like notation.

    Args:
        name: Direction string like '[111]' or '[1-10]'

    Returns:
        Normalized 3D direction vector
    """
    if name in DIRECTIONS:
        return DIRECTIONS[name].copy()
    raise ValueError(f"Unknown direction: {name}. Available: {list(DIRECTIONS.keys())}")
