"""
Crystal Geometry Engine.

Computes 3D crystal geometry from CDL descriptions.
Uses half-space intersection to combine crystal forms.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

from cdl_parser import CrystalDescription, parse_cdl

from ._accel import prefer_native
from .models import CrystalGeometry, LatticeParams, TwinMetadata
from .symmetry import (
    generate_equivalent_faces,
    get_lattice_for_system,
    miller_to_normal,
)


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
def _find_interior_point(normals: np.ndarray, distances: np.ndarray) -> np.ndarray | None:
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


def _iterative_interior_point(normals: np.ndarray, distances: np.ndarray) -> np.ndarray | None:
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


@prefer_native
def halfspace_intersection_3d(
    normals: list[np.ndarray] | np.ndarray,
    distances: list[float] | np.ndarray,
    interior_point: np.ndarray | None = None,
) -> np.ndarray | None:
    """Compute intersection of half-spaces in 3D.

    Each half-space is defined by: normal . x <= distance

    Args:
        normals: List or array of unit normal vectors pointing outward
        distances: List or array of distances from origin to each plane
        interior_point: A point known to be inside the intersection

    Returns:
        Array of vertices, or None if intersection is empty/unbounded
    """
    # Ensure contiguous arrays for optimal performance
    normals_arr = np.ascontiguousarray(normals, dtype=np.float64)
    distances_arr = np.ascontiguousarray(distances, dtype=np.float64)

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


@prefer_native
def _deduplicate_vertices(vertices: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
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


def _build_halfspaces(
    desc: CrystalDescription, lattice: LatticeParams
) -> tuple[list[np.ndarray], list[float], list[int], list[tuple[int, int, int]]]:
    """Build halfspaces from CDL forms.

    Args:
        desc: Parsed CDL description
        lattice: Lattice vectors

    Returns:
        Tuple of (normals, distances, face_form_indices, face_millers)
    """
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

    return normals, distances, face_form_indices, face_millers


def _generate_base_geometry(
    normals: list[np.ndarray],
    distances: list[float],
    face_form_indices: list[int],
    face_millers: list[tuple[int, int, int]],
    forms: list,
) -> CrystalGeometry:
    """Generate base crystal geometry from halfspaces.

    Args:
        normals: Face normals
        distances: Face distances
        face_form_indices: Form index for each face
        face_millers: Miller indices for each face
        forms: Original form list

    Returns:
        CrystalGeometry
    """
    # Compute half-space intersection
    vertices = halfspace_intersection_3d(normals, distances)

    if vertices is None or len(vertices) < 4:
        raise ValueError("Failed to compute crystal geometry - no valid intersection")

    # Remove duplicate vertices
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
        forms=forms,
    )


def _generate_twinned_geometry(
    desc: CrystalDescription,
    normals: list[np.ndarray],
    distances: list[float],
    face_form_indices: list[int],
    face_millers: list[tuple[int, int, int]],
) -> CrystalGeometry:
    """Generate twinned crystal geometry.

    Args:
        desc: CDL description with twin specification
        normals: Base halfspace normals
        distances: Base halfspace distances
        face_form_indices: Form indices
        face_millers: Miller indices

    Returns:
        CrystalGeometry with twin metadata
    """
    from .twins import get_generator, get_twin_law

    twin_spec = desc.twin
    assert twin_spec is not None

    # Get the twin law or create custom one
    if twin_spec.law:
        twin_law = get_twin_law(twin_spec.law)
        twin_axis = twin_law.axis
        twin_angle = twin_law.angle
        render_mode = twin_law.render_mode
        twin_type = twin_law.twin_type
        law_name = twin_law.name
    else:
        # Custom twin law from axis/angle
        twin_axis = np.array(twin_spec.axis) if twin_spec.axis else np.array([1, 1, 1])
        twin_axis = twin_axis / np.linalg.norm(twin_axis)
        twin_angle = twin_spec.angle
        twin_type = twin_spec.twin_type
        render_mode = "dual_crystal"  # Default for custom
        law_name = "custom"

    # Get the appropriate generator
    generator = get_generator(render_mode)

    # Convert to numpy arrays
    normals_arr = np.array(normals)
    distances_arr = np.array(distances)

    # Build twin_info dict for generator
    twin_info = {
        "axis": twin_axis,
        "angle": twin_angle,
        "type": twin_type,
        "n_fold": twin_spec.count if twin_spec.count > 2 else None,
    }

    # Generate twinned geometry
    twin_result = generator.generate(
        twin_info=twin_info,
        normals=normals_arr,
        distances=distances_arr,
    )

    # Build face data
    faces = []
    face_normals_list = []
    final_face_forms = []
    final_face_millers = []
    component_ids = []

    # For unified geometry, faces are already computed with correct vertex indices
    if twin_result.render_mode == "unified" and twin_result.components:
        from .twins import rotation_matrix_axis_angle

        component = twin_result.components[0]
        # Unified twins have a single vertex set - safe to deduplicate
        all_vertices = _deduplicate_vertices(component.get_transformed_vertices())

        # Get metadata for face-to-form matching
        meta = twin_result.metadata
        n_components = meta.get("n_original_components", 2)

        for face in component.faces:
            if len(face) < 3:
                continue

            # Compute actual face normal from vertex positions
            face_verts_arr = all_vertices[face]
            v0, v1, v2 = face_verts_arr[0], face_verts_arr[1], face_verts_arr[2]
            computed_normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(computed_normal)
            if norm_len < 1e-10:
                continue
            computed_normal = computed_normal / norm_len

            # Match against all rotated versions of original normals
            best_form_idx = 0
            best_dot = -1.0
            best_comp_id = 0
            for comp_id in range(n_components):
                R = rotation_matrix_axis_angle(twin_axis, twin_angle * comp_id)
                for form_idx, orig_normal in enumerate(normals):
                    rotated = R @ np.array(orig_normal)
                    dot = abs(np.dot(rotated, computed_normal))
                    if dot > best_dot:
                        best_dot = dot
                        best_form_idx = form_idx
                        best_comp_id = comp_id

            # Use matched values
            faces.append(list(face))
            face_normals_list.append(computed_normal)
            final_face_forms.append(face_form_indices[best_form_idx])
            final_face_millers.append(face_millers[best_form_idx])
            component_ids.append(best_comp_id)
    else:
        # For dual/v-shaped/cyclic: concatenate component vertices without deduplication
        # (components are separate polyhedra that may overlap but don't share vertices)
        from .twins import rotation_matrix_axis_angle

        all_vertices_list = []
        vertex_offset = 0

        for comp_idx, component in enumerate(twin_result.components):
            comp_verts = component.get_transformed_vertices()
            all_vertices_list.append(comp_verts)

            # Compute rotation for this component (generators don't store it in transform)
            if render_mode == "v_shaped":
                # V-shaped uses reflection, not rotation
                R = np.eye(3)
            else:
                # Dual/cyclic: rotation by comp_idx * angle around twin axis
                R = rotation_matrix_axis_angle(twin_axis, twin_angle * comp_idx)

            # Use faces from the generator, match to forms
            for face in component.faces:
                if len(face) < 3:
                    continue
                # Compute actual face normal from vertices
                face_verts_arr = comp_verts[face]
                v0, v1, v2 = face_verts_arr[0], face_verts_arr[1], face_verts_arr[2]
                computed_normal = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(computed_normal)
                if norm_len < 1e-10:
                    continue
                computed_normal = computed_normal / norm_len

                # Match to original form by comparing normals
                best_form_idx = 0
                best_dot = -1.0
                for form_idx, orig_normal in enumerate(normals):
                    rotated = R @ np.array(orig_normal)
                    dot = abs(np.dot(rotated, computed_normal))
                    if dot > best_dot:
                        best_dot = dot
                        best_form_idx = form_idx

                # Offset face indices for combined vertex array
                offset_face = [idx + vertex_offset for idx in face]
                faces.append(offset_face)
                face_normals_list.append(computed_normal)
                final_face_forms.append(face_form_indices[best_form_idx])
                final_face_millers.append(face_millers[best_form_idx])
                component_ids.append(comp_idx)

            vertex_offset += len(comp_verts)

        # Concatenate all component vertices
        all_vertices = (
            np.vstack(all_vertices_list) if all_vertices_list else np.array([]).reshape(0, 3)
        )

    # Create twin metadata
    twin_metadata = TwinMetadata(
        twin_law=law_name,
        render_mode=render_mode,
        n_components=twin_result.n_components,
        twin_axis=tuple(twin_axis.tolist()),
        twin_angle=twin_angle,
        face_attribution=component_ids,
    )

    return CrystalGeometry(
        vertices=all_vertices,
        faces=faces,
        face_normals=face_normals_list,
        face_forms=final_face_forms,
        face_millers=final_face_millers,
        forms=desc.forms,
        component_ids=component_ids,
        twin_metadata=twin_metadata,
    )


def cdl_to_geometry(desc: CrystalDescription, c_ratio: float = 1.0) -> CrystalGeometry:
    """Convert CDL description to 3D geometry.

    Args:
        desc: Parsed CDL description
        c_ratio: c/a ratio for non-cubic systems

    Returns:
        CrystalGeometry with vertices and faces
    """
    lattice = get_lattice_for_system(desc.system, c_ratio)

    # Build halfspaces from forms
    normals, distances, face_form_indices, face_millers = _build_halfspaces(desc, lattice)

    # Generate geometry (twinned or base)
    if desc.twin is not None:
        geometry = _generate_twinned_geometry(
            desc, normals, distances, face_form_indices, face_millers
        )
    else:
        geometry = _generate_base_geometry(
            normals, distances, face_form_indices, face_millers, desc.forms
        )

    # Apply modifications if present
    if desc.modifications:
        from .modifications import apply_modifications

        geometry = CrystalGeometry(
            vertices=apply_modifications(geometry.vertices, desc.modifications),
            faces=geometry.faces,
            face_normals=geometry.face_normals,
            face_forms=geometry.face_forms,
            face_millers=geometry.face_millers,
            forms=geometry.forms,
            component_ids=geometry.component_ids,
            twin_metadata=geometry.twin_metadata,
        )

    return geometry


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
    octahedron_scale: float = 1.0, cube_scale: float = 1.3
) -> CrystalGeometry:
    """Create a truncated octahedron (cuboctahedron-like).

    Args:
        octahedron_scale: Scale for octahedron faces
        cube_scale: Scale for cube truncation

    Returns:
        CrystalGeometry
    """
    return cdl_string_to_geometry(f"cubic[m3m]:{{111}}@{octahedron_scale} + {{100}}@{cube_scale}")
