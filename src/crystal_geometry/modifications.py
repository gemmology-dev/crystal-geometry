"""
Crystal morphology modifications.

Provides functions to apply geometric modifications like elongation
and tapering to crystal geometry.
"""

from typing import Any

import numpy as np

# Axis name to vector mapping
AXIS_MAP: dict[str, np.ndarray] = {
    "a": np.array([1.0, 0.0, 0.0]),
    "b": np.array([0.0, 1.0, 0.0]),
    "c": np.array([0.0, 0.0, 1.0]),
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def apply_modifications(vertices: np.ndarray, modifications: list[Any]) -> np.ndarray:
    """Apply all modifications to vertex array.

    Processes modifications in order, applying each to the result
    of the previous modification.

    Args:
        vertices: Nx3 array of vertex positions
        modifications: List of Modification objects from CDL parser
                       Each has 'name' (or 'type') and 'params' attributes

    Returns:
        Nx3 array of modified vertex positions
    """
    result = vertices.copy()

    for mod in modifications:
        # Get modification name (support both 'name' and 'type' attributes)
        mod_name = getattr(mod, "name", None) or getattr(mod, "type", None)

        # Get parameters (support both 'params' and 'parameters' attributes)
        params = getattr(mod, "params", None) or getattr(mod, "parameters", {}) or {}

        if mod_name == "elongate":
            result = apply_elongation(result, params)
        elif mod_name == "taper":
            result = apply_taper(result, params)
        elif mod_name == "flatten":
            result = apply_flatten(result, params)
        elif mod_name == "twist":
            result = apply_twist(result, params)
        # Truncation is handled via form combination, not post-processing
        # Bevel requires adding new faces, not just vertex modification

    return result


def apply_elongation(vertices: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply elongation (scaling) along a specified axis.

    Elongation stretches or compresses the crystal along one axis.

    Args:
        vertices: Nx3 array of vertex positions
        params: Modification parameters. Supports two formats:
               - {'axis': 'c', 'ratio': 1.5}
               - {'c': 1.5} (shorthand)

    Returns:
        Nx3 array of modified vertices
    """
    # Parse parameters
    axis_name = params.get("axis", "c").lower()
    ratio = params.get("ratio", params.get(axis_name, 1.0))

    if axis_name not in AXIS_MAP:
        raise ValueError(f"Unknown axis: '{axis_name}'. Use a, b, c, x, y, or z.")

    axis = AXIS_MAP[axis_name]

    # Scale along axis: v' = v + (ratio - 1) * (v · axis) * axis
    # This preserves the position along perpendicular directions
    result = vertices.copy()
    projections = vertices @ axis
    result = vertices + (ratio - 1) * np.outer(projections, axis)

    return result


def apply_taper(vertices: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply tapering toward a specified direction.

    Tapering scales vertices perpendicular to an axis based on their
    position along that axis, creating a narrowing effect.

    Args:
        vertices: Nx3 array of vertex positions
        params: Modification parameters:
               - 'direction': '+c', '-c', '+a', etc. (direction of tapering)
               - 'factor': 0.7 (scale factor at the tapered end, < 1 for narrowing)

    Returns:
        Nx3 array of modified vertices
    """
    direction = params.get("direction", "+c")
    factor = params.get("factor", 0.8)

    # Parse direction string
    positive = not direction.startswith("-")
    axis_name = direction[-1].lower()

    if axis_name not in AXIS_MAP:
        raise ValueError(f"Unknown axis in direction: '{direction}'")

    axis = AXIS_MAP[axis_name]

    # Get extent along axis
    projections = vertices @ axis
    min_proj = projections.min()
    max_proj = projections.max()
    extent = max_proj - min_proj

    if extent < 1e-10:
        # No extent along axis, nothing to taper
        return vertices.copy()

    # Compute t: normalized position along axis (0 to 1)
    t = (projections - min_proj) / extent

    if not positive:
        # Taper in negative direction (scale down at min end)
        t = 1 - t

    # Scale factor varies from 1 at base to 'factor' at tip
    scales = 1 - (1 - factor) * t

    # Decompose into parallel and perpendicular components
    parallel = np.outer(projections, axis)
    perpendicular = vertices - parallel

    # Scale perpendicular component
    result = parallel + perpendicular * scales[:, np.newaxis]

    return result


def apply_flatten(vertices: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply flattening perpendicular to a specified axis.

    Flattening compresses the crystal perpendicular to an axis,
    making it tabular or plate-like.

    Args:
        vertices: Nx3 array of vertex positions
        params: Modification parameters:
               - 'axis': 'c' (axis perpendicular to the flattening direction)
               - 'ratio': 0.5 (compression ratio, < 1 for flattening)

    Returns:
        Nx3 array of modified vertices
    """
    axis_name = params.get("axis", "c").lower()
    ratio = params.get("ratio", 0.5)

    if axis_name not in AXIS_MAP:
        raise ValueError(f"Unknown axis: '{axis_name}'")

    axis = AXIS_MAP[axis_name]

    # Scale along axis (flattening compresses along the specified axis)
    result = vertices.copy()
    projections = vertices @ axis
    parallel = np.outer(projections, axis)
    perpendicular = vertices - parallel

    # Compress along axis
    result = parallel * ratio + perpendicular

    return result


def apply_twist(vertices: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply twist deformation around a specified axis.

    Twist rotates vertices around an axis with the angle proportional
    to their position along that axis. Uses vectorized Rodrigues' formula
    for optimal performance.

    Args:
        vertices: Nx3 array of vertex positions
        params: Modification parameters:
               - 'axis': 'c' (rotation axis)
               - 'angle': 30 (total twist angle in degrees)

    Returns:
        Nx3 array of modified vertices
    """
    axis_name = params.get("axis", "c").lower()
    total_angle = params.get("angle", 30.0)

    if axis_name not in AXIS_MAP:
        raise ValueError(f"Unknown axis: '{axis_name}'")

    axis = AXIS_MAP[axis_name]

    # Ensure contiguous array
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)

    # Get extent along axis
    projections = vertices @ axis
    min_proj = projections.min()
    max_proj = projections.max()
    extent = max_proj - min_proj

    if extent < 1e-10:
        return vertices.copy()

    # Compute t: normalized position along axis (0 to 1)
    t = (projections - min_proj) / extent

    # Convert to rotation angles (one per vertex)
    angles = np.radians(total_angle * t)

    # Vectorized Rodrigues' formula:
    # v' = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    c = np.cos(angles)[:, np.newaxis]  # (N, 1)
    s = np.sin(angles)[:, np.newaxis]  # (N, 1)

    # k × v for all vertices (vectorized cross product)
    k_cross_v = np.cross(axis.reshape(1, 3), vertices)  # (N, 3)

    # k · v for all vertices (vectorized dot product)
    k_dot_v = (vertices @ axis)[:, np.newaxis]  # (N, 1)

    # Apply Rodrigues' formula
    result = vertices * c + k_cross_v * s + axis.reshape(1, 3) * k_dot_v * (1 - c)

    return result
