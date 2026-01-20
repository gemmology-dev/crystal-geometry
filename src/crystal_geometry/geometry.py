"""
Crystal Geometry Engine.

Computes 3D crystal geometry from CDL descriptions.
Uses half-space intersection to combine crystal forms.
"""

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from typing import List, Optional, Tuple

from cdl_parser import CrystalDescription, CrystalForm, parse_cdl

from .models import CrystalGeometry, LatticeParams, DEFAULT_LATTICE
from .symmetry import (
    generate_equivalent_faces,
    get_lattice_for_system,
    miller_to_normal,
)


def halfspace_intersection_3d(
    normals: List[np.ndarray],
    distances: List[float],
    interior_point: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Compute intersection of half-spaces in 3D.

    Each half-space is defined by: normal . x <= distance

    Args:
        normals: List of unit normal vectors pointing outward
        distances: List of distances from origin to each plane
        interior_point: A point known to be inside the intersection

    Returns:
        Array of vertices, or None if intersection is empty/unbounded
    """
    if interior_point is None:
        interior_point = np.array([0.0, 0.0, 0.0])

    # Build halfspace matrix for scipy
    # Format: [A | -b] where Ax <= b becomes Ax - b <= 0
    halfspaces = []
    for normal, dist in zip(normals, distances):
        row = list(normal) + [-dist]
        halfspaces.append(row)

    halfspaces = np.array(halfspaces)

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        return hs.intersections
    except Exception:
        # Try with adjusted interior point
        try:
            interior_point = interior_point * 0.1
            hs = HalfspaceIntersection(halfspaces, interior_point)
            return hs.intersections
        except Exception:
            return None


def compute_face_vertices(
    vertices: np.ndarray,
    normal: np.ndarray,
    distance: float,
    tolerance: float = 1e-6
) -> List[int]:
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

    # Remove duplicate vertices
    unique_verts = []
    vert_map = {}
    tolerance = 1e-8

    for i, v in enumerate(vertices):
        found = False
        for j, uv in enumerate(unique_verts):
            if np.linalg.norm(v - uv) < tolerance:
                vert_map[i] = j
                found = True
                break
        if not found:
            vert_map[i] = len(unique_verts)
            unique_verts.append(v)

    vertices = np.array(unique_verts)

    # Build faces
    faces = []
    face_normals_list = []
    final_face_forms = []
    final_face_millers = []

    for i, (normal, distance) in enumerate(zip(normals, distances)):
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
