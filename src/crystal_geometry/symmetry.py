"""
Crystal Symmetry Operations.

Implements the 32 crystallographic point groups and their symmetry operations.
Used to generate symmetry-equivalent faces from a single Miller index.
"""

from functools import lru_cache

import numpy as np

from .models import DEFAULT_LATTICE, LatticeParams

# =============================================================================
# Symmetry Operation Matrices
# =============================================================================

# Identity
E = np.eye(3)

# Inversion
I = -np.eye(3)


def Rz(n: int) -> np.ndarray:
    """n-fold rotation about z axis."""
    angle = 2 * np.pi / n
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rx(n: int) -> np.ndarray:
    """n-fold rotation about x axis."""
    angle = 2 * np.pi / n
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(n: int) -> np.ndarray:
    """n-fold rotation about y axis."""
    angle = 2 * np.pi / n
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Common rotations
C2z = Rz(2)  # 180 degree rotation about z
C2x = Rx(2)  # 180 degree rotation about x
C2y = Ry(2)  # 180 degree rotation about y
C3z = Rz(3)  # 120 degree rotation about z
C4z = Rz(4)  # 90 degree rotation about z
C6z = Rz(6)  # 60 degree rotation about z

# Mirrors
Mxy = np.diag([1, 1, -1])  # Mirror perpendicular to z
Mxz = np.diag([1, -1, 1])  # Mirror perpendicular to y
Myz = np.diag([-1, 1, 1])  # Mirror perpendicular to x

# Diagonal mirrors (for cubic system)
M110 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # Mirror across (110) plane

M1_10 = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])  # Mirror across (1-10) plane

# 3-fold rotation about [111]
C3_111 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

# 4-fold rotation about [100] (x-axis)
C4x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# 2-fold rotation about [110]
C2_110 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])


# =============================================================================
# Point Group Operations
# =============================================================================


def _generate_group(generators: list[np.ndarray], max_elements: int = 200) -> list[np.ndarray]:
    """Generate point group from generator matrices.

    Args:
        generators: List of generator matrices
        max_elements: Maximum number of elements to generate

    Returns:
        List of all unique group elements
    """
    group = [E.copy()]
    queue = list(generators)

    while queue and len(group) < max_elements:
        new_op = queue.pop(0)

        # Check if already in group
        is_new = True
        for existing in group:
            if np.allclose(new_op, existing, atol=1e-10):
                is_new = False
                break

        if is_new:
            group.append(new_op)
            # Generate new elements by multiplication
            for gen in generators:
                queue.append(new_op @ gen)
                queue.append(gen @ new_op)

    return group


# Cache for point group operations
_POINT_GROUP_CACHE: dict[str, list[np.ndarray]] = {}
_POINT_GROUP_ARRAY_CACHE: dict[str, np.ndarray] = {}  # Stacked (N,3,3) arrays


def get_point_group_operations(point_group: str) -> list[np.ndarray]:
    """Get symmetry operations for a point group.

    Args:
        point_group: Hermann-Mauguin symbol (e.g., 'm3m', '6/mmm')

    Returns:
        List of 3x3 rotation/reflection matrices
    """
    if point_group in _POINT_GROUP_CACHE:
        return _POINT_GROUP_CACHE[point_group]

    # Define generators for each point group
    generators_map = {
        # Cubic
        "m3m": [C4z, C3_111, I],
        "432": [C4z, C3_111],
        "-43m": [C4z @ I, C3_111],  # S4
        "m-3": [C2z, C3_111, I],
        "23": [C2z, C3_111],
        # Hexagonal
        "6/mmm": [C6z, C2x, Mxy],
        "622": [C6z, C2x],
        "6mm": [C6z, Mxz],
        "-6m2": [C3z, Mxy, Mxz],
        "6/m": [C6z, Mxy],
        "-6": [C3z, Mxy],
        "6": [C6z],
        # Trigonal
        "-3m": [C3z, C2x, I],
        "32": [C3z, C2x],
        "3m": [C3z, Mxz],
        "-3": [C3z, I],
        "3": [C3z],
        # Tetragonal
        "4/mmm": [C4z, C2x, Mxy],
        "422": [C4z, C2x],
        "4mm": [C4z, Mxz],
        "-42m": [C4z @ I, C2x],  # S4
        "4/m": [C4z, Mxy],
        "-4": [C4z @ I],  # S4
        "4": [C4z],
        # Orthorhombic
        "mmm": [C2z, C2x, I],
        "222": [C2z, C2x],
        "mm2": [C2z, Mxz],
        # Monoclinic
        "2/m": [C2y, Mxz],
        "2": [C2y],
        "m": [Mxz],
        # Triclinic
        "-1": [I],
        "1": [],
    }

    if point_group not in generators_map:
        raise ValueError(f"Unknown point group: {point_group}")

    generators = generators_map[point_group]
    if not generators:
        operations = [E.copy()]
    else:
        operations = _generate_group(generators)

    _POINT_GROUP_CACHE[point_group] = operations
    return operations


@lru_cache(maxsize=64)
def _get_reciprocal_lattice(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reciprocal lattice vectors (cached for performance).

    Args:
        a, b, c: Lattice parameters
        alpha, beta, gamma: Lattice angles in radians

    Returns:
        Tuple of (a*, b*, c*) reciprocal lattice vectors
    """
    # Direct lattice vectors
    a_vec = np.array([a, 0, 0])
    b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])

    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    c_vec = np.array([cx, cy, cz])

    # Volume
    V = np.dot(a_vec, np.cross(b_vec, c_vec))

    # Reciprocal lattice vectors
    a_star = np.cross(b_vec, c_vec) / V
    b_star = np.cross(c_vec, a_vec) / V
    c_star = np.cross(a_vec, b_vec) / V

    return a_star, b_star, c_star


def miller_to_normal(
    h: int, k: int, l: int, lattice: LatticeParams = DEFAULT_LATTICE
) -> np.ndarray:
    """Convert Miller index to face normal vector.

    For non-cubic systems, uses the reciprocal lattice (cached for performance).

    Args:
        h, k, l: Miller indices
        lattice: Lattice parameters

    Returns:
        Unit normal vector
    """
    # For cubic systems, Miller indices directly give the normal
    # For other systems, we need the reciprocal lattice
    if (
        lattice.a == lattice.b == lattice.c
        and lattice.alpha == lattice.beta == lattice.gamma == np.pi / 2
    ):
        # Cubic
        normal = np.array([h, k, l], dtype=float)
    else:
        # Non-cubic: use cached reciprocal lattice vectors
        a_star, b_star, c_star = _get_reciprocal_lattice(
            lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma
        )

        # Normal in Cartesian coordinates
        normal = h * a_star + k * b_star + l * c_star

    # Normalize
    norm = np.linalg.norm(normal)
    if norm > 0:
        return normal / norm
    return np.array([0, 0, 1])


def generate_equivalent_faces(
    h: int, k: int, l: int, point_group: str, lattice: LatticeParams = DEFAULT_LATTICE
) -> list[tuple[int, int, int]]:
    """Generate all symmetry-equivalent Miller indices.

    Args:
        h, k, l: Miller indices
        point_group: Hermann-Mauguin point group symbol
        lattice: Lattice parameters

    Returns:
        List of unique (h, k, l) tuples for equivalent faces
    """
    # Get cached stacked operations array, or build and cache it
    if point_group not in _POINT_GROUP_ARRAY_CACHE:
        operations = get_point_group_operations(point_group)
        _POINT_GROUP_ARRAY_CACHE[point_group] = np.array(operations)  # Shape: (N, 3, 3)

    operations_arr = _POINT_GROUP_ARRAY_CACHE[point_group]
    miller = np.array([h, k, l], dtype=np.float64)

    # Vectorized: apply all operations at once
    all_millers = operations_arr @ miller  # Shape: (N, 3)

    # Vectorized rounding to integers
    rounded = np.rint(all_millers).astype(np.int32)

    # Convert to set of tuples for uniqueness
    equivalent = {tuple(m) for m in rounded}

    return list(equivalent)


def get_lattice_for_system(system: str, c_ratio: float = 1.0) -> LatticeParams:
    """Get appropriate lattice parameters for a crystal system.

    Args:
        system: Crystal system name
        c_ratio: c/a ratio for non-cubic systems

    Returns:
        LatticeParams for the system
    """
    if system == "cubic":
        return LatticeParams.cubic()
    elif system == "hexagonal" or system == "trigonal":
        return LatticeParams.hexagonal(c_ratio)
    elif system == "tetragonal":
        return LatticeParams.tetragonal(c_ratio)
    else:
        # Orthorhombic, monoclinic, triclinic - use defaults
        return LatticeParams()
