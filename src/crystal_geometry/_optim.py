"""
Optimized core geometry operations.

This module consolidates performance-critical functions that were previously
duplicated across geometry.py and twins/generators.py. All functions use
vectorized operations and squared-distance comparisons for optimal performance.

These functions can be accelerated via native C++ implementations when available.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

from ._accel import prefer_native


def _spatial_hash_key(vertex: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    """Compute spatial hash key for a vertex.

    Args:
        vertex: 3D vertex position
        cell_size: Size of hash grid cells

    Returns:
        Tuple of (x, y, z) cell indices
    """
    return (
        int(np.floor(vertex[0] / cell_size)),
        int(np.floor(vertex[1] / cell_size)),
        int(np.floor(vertex[2] / cell_size)),
    )


@prefer_native
def find_interior_point(normals: np.ndarray, distances: np.ndarray) -> np.ndarray | None:
    """Find interior point using linear programming (Chebyshev center).

    Finds the center of the largest ball that fits inside the polyhedron
    defined by the halfspaces. This is a robust way to find a strictly
    interior point.

    Args:
        normals: Nx3 array of unit normal vectors (must be contiguous)
        distances: N array of distances (must be contiguous)

    Returns:
        Interior point as 3-element array, or None if no solution
    """
    # Ensure contiguous arrays for optimal performance
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)

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
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success and result.x[3] > 1e-10:
            return result.x[:3]
    except Exception:
        pass

    return None


def find_interior_point_iterative(normals: np.ndarray, distances: np.ndarray) -> np.ndarray | None:
    """Find interior point by iterative shrinking.

    Fallback method when linear programming fails.

    Args:
        normals: Nx3 array of unit normal vectors
        distances: N array of distances

    Returns:
        Interior point or None
    """
    # Ensure contiguous arrays
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)

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


@prefer_native
def deduplicate_vertices(vertices: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Remove duplicate vertices using spatial hashing for O(n) performance.

    Uses a hash grid to achieve approximately O(n) complexity by only
    checking nearby cells for potential duplicates.

    Args:
        vertices: Nx3 array of vertex positions
        tolerance: Distance threshold for considering vertices identical

    Returns:
        Array of unique vertices
    """
    if len(vertices) == 0:
        return vertices

    # Ensure contiguous array
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)

    # Use cell size slightly larger than tolerance to ensure we check all neighbors
    cell_size = tolerance * 2
    tolerance_sq = tolerance * tolerance

    # Hash table: key -> first vertex index in that cell
    seen: dict[tuple[int, int, int], int] = {}
    unique_indices: list[int] = []

    for i in range(len(vertices)):
        v = vertices[i]
        key = _spatial_hash_key(v, cell_size)

        # Check this cell and all 26 neighbors for potential duplicates
        found_duplicate = False
        for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
            neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
            if neighbor_key in seen:
                existing_idx = seen[neighbor_key]
                diff = v - vertices[existing_idx]
                # Use squared distance to avoid sqrt
                if diff @ diff < tolerance_sq:
                    found_duplicate = True
                    break

        if not found_duplicate:
            seen[key] = i
            unique_indices.append(i)

    return vertices[unique_indices]


@prefer_native
def compute_face_vertices(
    vertices: np.ndarray, normal: np.ndarray, distance: float, tolerance: float = 1e-6
) -> list[int]:
    """Find vertices that lie on a face plane.

    Uses vectorized operations for improved performance.

    Args:
        vertices: All vertices (Nx3 array)
        normal: Face normal (3-element array)
        distance: Distance from origin to face plane
        tolerance: Numerical tolerance

    Returns:
        List of vertex indices on this face, ordered counter-clockwise
    """
    # Ensure contiguous arrays
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    normal = np.ascontiguousarray(normal, dtype=np.float64)

    # Vectorized distance computation for all vertices
    projections = vertices @ normal
    on_face_mask = np.abs(projections - distance) < tolerance
    on_face_indices = np.where(on_face_mask)[0]

    if len(on_face_indices) < 3:
        return []

    # Get vertices on face
    on_face_verts = vertices[on_face_indices]

    # Order vertices counter-clockwise when viewed from outside
    center = np.mean(on_face_verts, axis=0)

    # Create local coordinate system on face
    u = on_face_verts[0] - center
    u = u - np.dot(u, normal) * normal
    u_norm = u @ u  # squared norm
    if u_norm < tolerance * tolerance:
        if len(on_face_indices) > 1:
            u = on_face_verts[1] - center
            u = u - np.dot(u, normal) * normal
            u_norm = u @ u
    u = u / (np.sqrt(u_norm) + 1e-10)
    v_axis = np.cross(normal, u)

    # Vectorized angle computation
    vecs = on_face_verts - center
    angles = np.arctan2(vecs @ v_axis, vecs @ u)

    # Sort by angle
    sorted_order = np.argsort(angles)
    return [int(on_face_indices[i]) for i in sorted_order]


@prefer_native
def halfspace_intersection(
    normals: np.ndarray, distances: np.ndarray, interior_point: np.ndarray | None = None
) -> np.ndarray | None:
    """Compute intersection of half-spaces in 3D.

    Each half-space is defined by: normal . x <= distance

    Args:
        normals: Nx3 array of unit normal vectors pointing outward
        distances: N array of distances from origin to each plane
        interior_point: A point known to be inside the intersection

    Returns:
        Array of vertices, or None if intersection is empty/unbounded
    """
    # Ensure contiguous arrays for optimal performance
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)

    if interior_point is None:
        # Try Chebyshev center first (most robust)
        interior_point = find_interior_point(normals, distances)

        if interior_point is None:
            # Fallback to iterative method
            interior_point = find_interior_point_iterative(normals, distances)

        if interior_point is None:
            # Last resort: try origin
            interior_point = np.array([0.0, 0.0, 0.0])

    # Build halfspace matrix for scipy
    # Format: [A | -b] where Ax <= b becomes Ax - b <= 0
    halfspaces = np.hstack([normals, -distances.reshape(-1, 1)])

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


def compute_vertices_direct(normals: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Compute vertices by finding triple-plane intersections.

    Fallback method when scipy fails.

    Args:
        normals: Nx3 array of normals
        distances: N-element array of distances

    Returns:
        Array of intersection vertices
    """
    # Ensure contiguous arrays
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)

    vertices: list[np.ndarray] = []
    n = len(normals)
    tolerance = 1e-6
    tolerance_sq = tolerance * tolerance

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                A = np.array([normals[i], normals[j], normals[k]])
                b = np.array([distances[i], distances[j], distances[k]])

                try:
                    if abs(np.linalg.det(A)) > 1e-10:
                        v = np.linalg.solve(A, b)

                        # Check if inside all halfspaces
                        inside = True
                        for ni, di in zip(normals, distances, strict=False):
                            if np.dot(ni, v) > di + tolerance:
                                inside = False
                                break

                        if inside:
                            # Check for duplicates using squared distance
                            is_dup = False
                            for existing in vertices:
                                diff = v - existing
                                if diff @ diff < tolerance_sq:
                                    is_dup = True
                                    break
                            if not is_dup:
                                vertices.append(v)
                except np.linalg.LinAlgError:
                    pass

    if vertices:
        return np.array(vertices)
    return np.array([]).reshape(0, 3)


def compute_all_face_vertices(
    vertices: np.ndarray, normals: np.ndarray, distances: np.ndarray, tolerance: float = 1e-5
) -> list[list[int]]:
    """Compute face vertex indices for all halfspaces.

    Uses vectorized operations for improved performance.

    Args:
        vertices: Mx3 array of vertices
        normals: Nx3 array of face normals
        distances: N-element array of distances
        tolerance: Tolerance for vertex-on-plane detection

    Returns:
        List of faces, each face is list of vertex indices (counter-clockwise)
    """
    # Ensure contiguous arrays
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)

    faces = []
    tolerance_sq = tolerance * tolerance

    for normal, distance in zip(normals, distances, strict=False):
        # Vectorized distance computation
        projections = vertices @ normal
        on_face_mask = np.abs(projections - distance) < tolerance
        on_face_indices = np.where(on_face_mask)[0]

        if len(on_face_indices) < 3:
            continue

        on_face_verts = vertices[on_face_indices]

        # Order vertices counter-clockwise when viewed from outside
        center = np.mean(on_face_verts, axis=0)

        # Create local coordinate system on face
        u = on_face_verts[0] - center
        u = u - np.dot(u, normal) * normal
        u_norm_sq = u @ u
        if u_norm_sq < tolerance_sq:
            if len(on_face_indices) > 1:
                u = on_face_verts[1] - center
                u = u - np.dot(u, normal) * normal
                u_norm_sq = u @ u
        u = u / (np.sqrt(u_norm_sq) + 1e-10)
        v_axis = np.cross(normal, u)

        # Vectorized angle computation
        vecs = on_face_verts - center
        angles = np.arctan2(vecs @ v_axis, vecs @ u)

        # Sort by angle
        sorted_order = np.argsort(angles)
        faces.append([int(on_face_indices[i]) for i in sorted_order])

    return faces
