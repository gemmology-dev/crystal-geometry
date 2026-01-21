"""
Crystal Geometry Engine.

Computes 3D crystal geometry from CDL descriptions.
Uses half-space intersection to combine crystal forms.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, cKDTree

from cdl_parser import CrystalDescription, parse_cdl

from .models import CrystalGeometry
from .symmetry import (
    generate_equivalent_faces,
    get_lattice_for_system,
    miller_to_normal,
)


def _find_interior_point(
    normals: np.ndarray,
    distances: np.ndarray
) -> np.ndarray | None:
    """Find interior point using linear programming (Chebyshev center).

    Finds the center of the largest ball that fits inside the polyhedron
    defined by the halfspaces. This is a robust way to find a strictly
    interior point.

    Args:
        normals: Nx3 array of unit normal vectors
        distances: N array of distances

    Returns:
        Interior point as 3-element array, or None if no solution
    """
    n_constraints = len(normals)

    # Maximize r subject to: n_i · x + r <= d_i
    # Variables: [x, y, z, r]
    # We minimize -r (to maximize r)
    c = np.array([0.0, 0.0, 0.0, -1.0])

    # Build constraint matrix: [n_x, n_y, n_z, ||n||] (||n|| = 1 for unit normals)
    A_ub = np.hstack([normals, np.ones((n_constraints, 1))])
    b_ub = distances

    # Bounds: x, y, z can be anything reasonable, r >= 0
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (1e-10, None)]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if result.success and result.x[3] > 1e-10:
            return result.x[:3]
    except Exception:
        pass

    return None


def _iterative_interior_point(
    normals: np.ndarray,
    distances: np.ndarray
) -> np.ndarray | None:
    """Find interior point by iterative shrinking.

    Fallback method when linear programming fails.

    Args:
        normals: Nx3 array of unit normal vectors
        distances: N array of distances

    Returns:
        Interior point or None
    """
    # Start at centroid of normals weighted by distances
    centroid = np.zeros(3)
    total_weight = 0.0

    for normal, dist in zip(normals, distances, strict=False):
        # Point on plane in direction of normal
        point = normal * dist
        centroid += point
        total_weight += 1.0

    if total_weight > 0:
        centroid /= total_weight

    # Check if centroid is inside all halfspaces
    for normal, dist in zip(normals, distances, strict=False):
        if np.dot(normal, centroid) > dist - 1e-10:
            # Not inside, try shrinking toward origin
            for scale in [0.5, 0.3, 0.1, 0.05, 0.01]:
                test_point = centroid * scale
                inside = True
                for n, d in zip(normals, distances, strict=False):
                    if np.dot(n, test_point) > d - 1e-10:
                        inside = False
                        break
                if inside:
                    return test_point

            # Try pure origin
            origin = np.zeros(3)
            inside = True
            for n, d in zip(normals, distances, strict=False):
                if np.dot(n, origin) > d - 1e-10:
                    inside = False
                    break
            if inside:
                return origin

            return None

    return centroid


def halfspace_intersection_3d(
    normals: list[np.ndarray],
    distances: list[float],
    interior_point: np.ndarray | None = None
) -> np.ndarray | None:
    """Compute intersection of half-spaces in 3D.

    Each half-space is defined by: normal . x <= distance

    Args:
        normals: List of unit normal vectors pointing outward
        distances: List of distances from origin to each plane
        interior_point: A point known to be inside the intersection

    Returns:
        Array of vertices, or None if intersection is empty/unbounded
    """
    normals_arr = np.array(normals)
    distances_arr = np.array(distances)

    if interior_point is None:
        # Try Chebyshev center first (most robust)
        interior_point = _find_interior_point(normals_arr, distances_arr)

        if interior_point is None:
            # Fallback to iterative method
            interior_point = _iterative_interior_point(normals_arr, distances_arr)

        if interior_point is None:
            # Last resort: try origin
            interior_point = np.array([0.0, 0.0, 0.0])

    # Build halfspace matrix for scipy
    # Format: [A | -b] where Ax <= b becomes Ax - b <= 0
    halfspaces = np.hstack([normals_arr, -distances_arr.reshape(-1, 1)])

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        return hs.intersections
    except Exception:
        # Try with scaled interior point
        try:
            scaled_point = interior_point * 0.5
            hs = HalfspaceIntersection(halfspaces, scaled_point)
            return hs.intersections
        except Exception:
            return None


def _deduplicate_vertices(
    vertices: np.ndarray,
    tolerance: float = 1e-8
) -> np.ndarray:
    """Remove duplicate vertices using KD-tree for O(n log n) performance.

    Args:
        vertices: Nx3 array of vertex positions
        tolerance: Distance threshold for considering vertices identical

    Returns:
        Array of unique vertices
    """
    if len(vertices) == 0:
        return vertices

    # Build KD-tree for efficient nearest neighbor queries
    tree = cKDTree(vertices)

    # Find clusters of nearby vertices
    unique_indices = []
    visited = set()

    for i in range(len(vertices)):
        if i in visited:
            continue

        # Find all vertices within tolerance
        neighbors = tree.query_ball_point(vertices[i], tolerance)

        # Mark all neighbors as visited
        for n in neighbors:
            visited.add(n)

        # Keep only the first vertex in this cluster
        unique_indices.append(i)

    return vertices[unique_indices]


def compute_face_vertices(
    vertices: np.ndarray,
    normal: np.ndarray,
    distance: float,
    tolerance: float = 1e-6
) -> list[int]:
    """Find vertices that lie on a face plane.

    Args:
        vertices: All vertices
        normal: Face normal
        distance: Distance from origin to face plane
        tolerance: Numerical tolerance

    Returns:
        List of vertex indices on this face, ordered counter-clockwise
    """
    # Find vertices on plane: normal . v = distance
    on_face = []
    for i, v in enumerate(vertices):
        d = np.dot(normal, v)
        if abs(d - distance) < tolerance:
            on_face.append(i)

    if len(on_face) < 3:
        return []

    # Order vertices counter-clockwise when viewed from outside
    center = np.mean(vertices[on_face], axis=0)

    # Create local coordinate system on face
    u = vertices[on_face[0]] - center
    u = u - np.dot(u, normal) * normal
    if np.linalg.norm(u) < tolerance:
        if len(on_face) > 1:
            u = vertices[on_face[1]] - center
            u = u - np.dot(u, normal) * normal
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(normal, u)

    # Compute angles
    angles = []
    for idx in on_face:
        vec = vertices[idx] - center
        angle = np.arctan2(np.dot(vec, v), np.dot(vec, u))
        angles.append((angle, idx))

    # Sort by angle
    angles.sort()
    return [idx for _, idx in angles]


def cdl_to_geometry(
    desc: CrystalDescription,
    c_ratio: float = 1.0
) -> CrystalGeometry:
    """Convert CDL description to 3D geometry.

    Args:
        desc: Parsed CDL description
        c_ratio: c/a ratio for non-cubic systems

    Returns:
        CrystalGeometry with vertices and faces
    """
    lattice = get_lattice_for_system(desc.system, c_ratio)

    # Collect all half-spaces from all forms
    normals = []
    distances = []
    face_form_indices = []
    face_millers = []

    for form_idx, form in enumerate(desc.forms):
        miller = form.miller.as_3index()
        h, k, l = miller

        # Generate all symmetry-equivalent faces
        equivalent = generate_equivalent_faces(h, k, l, desc.point_group, lattice)

        for eq_miller in equivalent:
            normal = miller_to_normal(*eq_miller, lattice)
            distance = form.scale

            normals.append(normal)
            distances.append(distance)
            face_form_indices.append(form_idx)
            face_millers.append(eq_miller)

    # Compute half-space intersection
    vertices = halfspace_intersection_3d(normals, distances)

    if vertices is None or len(vertices) < 4:
        raise ValueError("Failed to compute crystal geometry - no valid intersection")

    # Remove duplicate vertices using KD-tree for O(n log n) performance
    vertices = _deduplicate_vertices(vertices)

    # Build faces
    faces = []
    face_normals_list = []
    final_face_forms = []
    final_face_millers = []

    for i, (normal, distance) in enumerate(zip(normals, distances, strict=False)):
        face_verts = compute_face_vertices(vertices, normal, distance)
        if len(face_verts) >= 3:
            faces.append(face_verts)
            face_normals_list.append(normal)
            final_face_forms.append(face_form_indices[i])
            final_face_millers.append(face_millers[i])

    return CrystalGeometry(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals_list,
        face_forms=final_face_forms,
        face_millers=final_face_millers,
        forms=desc.forms
    )


def cdl_string_to_geometry(cdl: str, c_ratio: float = 1.0) -> CrystalGeometry:
    """Convenience function to convert CDL string directly to geometry.

    Args:
        cdl: CDL string like "cubic[m3m]:{111}@1.0 + {100}@1.3"
        c_ratio: c/a ratio for non-cubic systems

    Returns:
        CrystalGeometry
    """
    desc = parse_cdl(cdl)
    return cdl_to_geometry(desc, c_ratio)


def create_octahedron(scale: float = 1.0) -> CrystalGeometry:
    """Create a regular octahedron.

    Args:
        scale: Distance from origin to vertices

    Returns:
        CrystalGeometry for octahedron
    """
    return cdl_string_to_geometry(f"cubic[m3m]:{{111}}@{scale}")


def create_cube(scale: float = 1.0) -> CrystalGeometry:
    """Create a cube.

    Args:
        scale: Distance from origin to face centers

    Returns:
        CrystalGeometry for cube
    """
    return cdl_string_to_geometry(f"cubic[m3m]:{{100}}@{scale}")


def create_dodecahedron(scale: float = 1.0) -> CrystalGeometry:
    """Create a rhombic dodecahedron.

    Args:
        scale: Distance from origin to face centers

    Returns:
        CrystalGeometry for dodecahedron
    """
    return cdl_string_to_geometry(f"cubic[m3m]:{{110}}@{scale}")


def create_truncated_octahedron(
    octahedron_scale: float = 1.0,
    cube_scale: float = 1.3
) -> CrystalGeometry:
    """Create a truncated octahedron (cuboctahedron-like).

    Args:
        octahedron_scale: Scale for octahedron faces
        cube_scale: Scale for cube truncation

    Returns:
        CrystalGeometry
    """
    return cdl_string_to_geometry(
        f"cubic[m3m]:{{111}}@{octahedron_scale} + {{100}}@{cube_scale}"
    )
