"""
Twin geometry generators.

Provides 5 generator classes for different twin rendering strategies:
- UnifiedGeometryGenerator: Single polyhedron via halfspace intersection
- DualCrystalGeometryGenerator: Two interpenetrating crystals
- VShapedGeometryGenerator: V-shaped contact twins
- CyclicGeometryGenerator: Cyclic twins with n-fold symmetry
- SingleCrystalGeometryGenerator: Internal twins (no external change)

Face Winding Convention:
- All faces use counter-clockwise vertex ordering when viewed from outside
- Face normals point outward (right-hand rule: CCW vertices → outward normal)
- When reflecting geometry across a plane, face vertex order must be reversed
  to maintain correct outward-facing normals
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .transforms import rotation_matrix_axis_angle


class CrystalComponent:
    """Represents a single crystal component with geometry and transform.

    Attributes:
        vertices: Nx3 array of vertex positions (in local coordinates)
        faces: List of faces, each face is list of vertex indices
        transform: 4x4 homogeneous transformation matrix
        component_id: Integer identifier for this component
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: list[list[int]],
        transform: np.ndarray | None = None,
        component_id: int = 0,
    ):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = faces
        self.transform = transform if transform is not None else np.eye(4)
        self.component_id = component_id

    def get_transformed_vertices(self) -> np.ndarray:
        """Apply transformation to vertices.

        Returns:
            Nx3 array of transformed vertex positions
        """
        if np.allclose(self.transform, np.eye(4)):
            return self.vertices

        # Apply 4x4 homogeneous transform
        n_verts = len(self.vertices)
        homogeneous = np.hstack([self.vertices, np.ones((n_verts, 1))])
        transformed = homogeneous @ self.transform.T
        return transformed[:, :3]


class TwinGeometry:
    """Container for generated twin geometry.

    Supports both unified (single polyhedron) and separate (multiple component)
    rendering modes.

    Attributes:
        components: List of CrystalComponent objects
        render_mode: 'unified', 'separate', or 'single_crystal'
        metadata: Additional rendering hints
    """

    def __init__(
        self,
        components: list[CrystalComponent],
        render_mode: str = "unified",
        metadata: dict[str, Any] | None = None,
    ):
        self.components = components
        self.render_mode = render_mode
        self.metadata = metadata or {}

    @property
    def n_components(self) -> int:
        """Number of crystal components."""
        return len(self.components)

    def get_all_vertices(self) -> np.ndarray:
        """Get all vertices from all components (transformed).

        Returns:
            Mx3 array of all vertices concatenated
        """
        all_verts = []
        for comp in self.components:
            all_verts.append(comp.get_transformed_vertices())
        if not all_verts:
            return np.array([]).reshape(0, 3)
        return np.vstack(all_verts)

    def get_all_faces(self) -> list[list[int]]:
        """Get all faces from all components with corrected vertex indices.

        Returns:
            List of faces with global vertex indices
        """
        all_faces = []
        vertex_offset = 0

        for comp in self.components:
            for face in comp.faces:
                all_faces.append([idx + vertex_offset for idx in face])
            vertex_offset += len(comp.vertices)

        return all_faces

    def get_face_attribution(self) -> np.ndarray:
        """Get component ID for each face.

        Returns:
            Array of component IDs, one per face
        """
        attribution = []
        for comp in self.components:
            attribution.extend([comp.component_id] * len(comp.faces))
        return np.array(attribution, dtype=np.int32)


class TwinGeometryGenerator(ABC):
    """Base class for twin geometry generation strategies."""

    @abstractmethod
    def generate(
        self, twin_info: dict[str, Any], normals: np.ndarray, distances: np.ndarray
    ) -> TwinGeometry:
        """Generate geometry for a twin type.

        Args:
            twin_info: Twin law configuration (dict with axis, angle, type, etc.)
            normals: Nx3 array of halfspace normals
            distances: N-element array of halfspace distances

        Returns:
            TwinGeometry with components and rendering metadata
        """
        pass


def _compute_halfspace_intersection(normals: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Compute vertices from halfspace intersection.

    Uses scipy's HalfspaceIntersection internally.

    Args:
        normals: Nx3 array of unit normal vectors
        distances: N-element array of distances

    Returns:
        Mx3 array of intersection vertices
    """
    from scipy.optimize import linprog
    from scipy.spatial import HalfspaceIntersection

    # Ensure contiguous arrays for optimal performance
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)

    n_constraints = len(normals)

    # Find interior point using Chebyshev center (linear programming)
    c = np.array([0.0, 0.0, 0.0, -1.0])
    A_ub = np.hstack([normals, np.ones((n_constraints, 1))])
    b_ub = distances
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (1e-10, None)]

    interior_point = None
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if result.success and result.x[3] > 1e-10:
            interior_point = result.x[:3]
    except Exception:
        pass

    # Fallback: use centroid from direct vertex enumeration
    if interior_point is None:
        fallback_verts = _compute_vertices_direct(normals, distances)
        if len(fallback_verts) > 0:
            interior_point = np.mean(fallback_verts, axis=0)
        else:
            # Last resort: origin (may fail, but we'll catch it)
            interior_point = np.array([0.0, 0.0, 0.0])

    # Build halfspace matrix for scipy
    halfspaces = np.hstack([normals, -distances.reshape(-1, 1)])

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        return hs.intersections
    except Exception:
        # Fallback: direct computation
        return _compute_vertices_direct(normals, distances)


def _compute_vertices_direct(normals: np.ndarray, distances: np.ndarray) -> np.ndarray:
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


def _compute_face_vertices(
    vertices: np.ndarray, normals: np.ndarray, distances: np.ndarray, tolerance: float = 1e-5
) -> list[list[int]]:
    """Compute face vertex indices for each halfspace.

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


class UnifiedGeometryGenerator(TwinGeometryGenerator):
    """Halfspace intersection approach - single unified polyhedron.

    Works for twins where constraint intersection produces correct shape:
    - Spinel law (macle)
    - Fluorite penetration
    - Albite, Manebach, Baveno contact twins
    """

    def generate(
        self, twin_info: dict[str, Any], normals: np.ndarray, distances: np.ndarray
    ) -> TwinGeometry:
        twin_axis = np.asarray(twin_info["axis"])
        twin_angle = twin_info["angle"]
        twin_type = twin_info.get("type", "contact")

        # Determine number of components
        if twin_type == "cyclic":
            n_components = int(round(360.0 / twin_angle))
        else:
            n_components = 2

        # Collect all halfspaces from all rotated orientations (pre-allocated)
        n_base = len(normals)
        n_halfspaces = n_components * n_base
        all_normals = np.empty((n_halfspaces, 3), dtype=np.float64)
        all_distances = np.empty(n_halfspaces, dtype=np.float64)

        idx = 0
        for comp_id in range(n_components):
            R = rotation_matrix_axis_angle(twin_axis, twin_angle * comp_id)
            # Vectorized rotation of all normals at once
            rotated = (R @ normals.T).T
            all_normals[idx : idx + n_base] = rotated
            all_distances[idx : idx + n_base] = distances
            idx += n_base

        # Compute unified intersection
        vertices = _compute_halfspace_intersection(all_normals, all_distances)

        if len(vertices) < 4:
            raise ValueError("Failed to compute unified twin geometry")

        # Compute faces
        faces = _compute_face_vertices(vertices, all_normals, all_distances)

        # Build face attribution by matching each face to its source halfspace
        # (faces may skip some halfspaces, so we match by geometric proximity)
        n_base = len(normals)
        final_attribution = []
        for face in faces:
            if len(face) < 3:
                final_attribution.append(0)
                continue

            # Compute face centroid
            face_verts = vertices[face]
            centroid = np.mean(face_verts, axis=0)

            # Find which halfspace this face lies on (centroid should be on the plane)
            best_hs_idx = 0
            best_dist = float("inf")
            for hs_idx, (hs_normal, hs_dist) in enumerate(
                zip(all_normals, all_distances, strict=False)
            ):
                # Distance from centroid to plane
                plane_dist = abs(np.dot(centroid, hs_normal) - hs_dist)
                if plane_dist < best_dist:
                    best_dist = plane_dist
                    best_hs_idx = hs_idx

            # Component ID is which rotation group this halfspace belongs to
            comp_id = best_hs_idx // n_base
            final_attribution.append(comp_id)

        component = CrystalComponent(
            vertices=vertices, faces=faces, transform=np.eye(4), component_id=0
        )

        return TwinGeometry(
            components=[component],
            render_mode="unified",
            metadata={
                "blend_mode": "single",
                "face_attribution": np.array(final_attribution, dtype=np.int32),
                "n_original_components": n_components,
                "twin_axis": twin_axis,
                "twin_angle": twin_angle,
            },
        )


class DualCrystalGeometryGenerator(TwinGeometryGenerator):
    """Two complete interpenetrating crystals (penetration twins).

    Required for twins where two distinct crystal volumes interpenetrate:
    - Staurolite 60°/90° (cross shapes)
    - Iron cross (pyrite)
    - Brazil twin (quartz)
    - Carlsbad twin
    """

    def generate(
        self, twin_info: dict[str, Any], normals: np.ndarray, distances: np.ndarray
    ) -> TwinGeometry:
        twin_axis = np.asarray(twin_info["axis"])
        twin_angle = twin_info["angle"]

        # Crystal 1: Complete crystal at origin
        verts1 = _compute_halfspace_intersection(normals, distances)
        if len(verts1) < 4:
            raise ValueError("Failed to compute crystal 1 geometry")
        faces1 = _compute_face_vertices(verts1, normals, distances)

        component1 = CrystalComponent(
            vertices=verts1, faces=faces1, transform=np.eye(4), component_id=0
        )

        # Crystal 2: Complete rotated crystal
        R = rotation_matrix_axis_angle(twin_axis, twin_angle)
        rotated_normals = normals @ R.T

        verts2 = _compute_halfspace_intersection(rotated_normals, distances)
        if len(verts2) < 4:
            raise ValueError("Failed to compute crystal 2 geometry")
        faces2 = _compute_face_vertices(verts2, rotated_normals, distances)

        component2 = CrystalComponent(
            vertices=verts2,
            faces=faces2,
            transform=np.eye(4),  # Already rotated via normals
            component_id=1,
        )

        return TwinGeometry(
            components=[component1, component2],
            render_mode="separate",
            metadata={
                "blend_mode": "overlay",
                "twin_axis": twin_axis,
                "twin_angle": twin_angle,
            },
        )


class VShapedGeometryGenerator(TwinGeometryGenerator):
    """V-shaped contact twins using reflection across composition plane.

    Required for contact twins forming V or re-entrant shapes:
    - Japan law quartz (84.5° V)
    - Gypsum swallow-tail

    The two crystal halves share an edge at the composition plane,
    creating a characteristic V or heart shape.
    """

    def generate(
        self, twin_info: dict[str, Any], normals: np.ndarray, distances: np.ndarray
    ) -> TwinGeometry:
        twin_axis = np.asarray(twin_info["axis"])
        twin_axis = twin_axis / np.linalg.norm(twin_axis)
        twin_angle = twin_info["angle"]

        # Composition plane passes through origin
        composition_offset = 0.0

        # Crystal 1: Clip full crystal at composition plane (keep positive side)
        all_normals1 = np.vstack([normals, -twin_axis.reshape(1, 3)])
        all_distances1 = np.append(distances, -composition_offset)

        verts1 = _compute_halfspace_intersection(all_normals1, all_distances1)
        if len(verts1) < 4:
            raise ValueError("Failed to compute V-shaped crystal 1 geometry")
        faces1 = _compute_face_vertices(verts1, all_normals1, all_distances1)

        # Crystal 2: Reflection of crystal 1 across composition plane
        # v' = v - 2*(v·n)*n
        verts2 = verts1 - 2 * np.outer(verts1 @ twin_axis, twin_axis)
        # Reverse face winding to maintain outward normals after reflection
        faces2 = [list(reversed(face)) for face in faces1]

        component1 = CrystalComponent(
            vertices=verts1, faces=faces1, transform=np.eye(4), component_id=0
        )

        component2 = CrystalComponent(
            vertices=verts2, faces=faces2, transform=np.eye(4), component_id=1
        )

        return TwinGeometry(
            components=[component1, component2],
            render_mode="separate",
            metadata={
                "blend_mode": "adjacent",
                "composition_plane": {"normal": twin_axis, "offset": composition_offset},
                "twin_axis": twin_axis,
                "twin_angle": twin_angle,
                "transform": "reflection",
            },
        )


class CyclicGeometryGenerator(TwinGeometryGenerator):
    """Cyclic twins with n-fold rotational symmetry.

    Can use either unified (intersection) or separate (multiple crystals)
    approach depending on configuration.
    """

    def __init__(self, use_unified: bool = True):
        """Initialize cyclic generator.

        Args:
            use_unified: If True, use halfspace intersection for unified shape.
                         If False, render separate crystal components.
        """
        self.use_unified = use_unified

    def generate(
        self, twin_info: dict[str, Any], normals: np.ndarray, distances: np.ndarray
    ) -> TwinGeometry:
        twin_axis = np.asarray(twin_info["axis"])
        twin_angle = twin_info["angle"]
        n_components = int(round(360.0 / twin_angle))

        if self.use_unified:
            # Use unified approach (halfspace intersection) with pre-allocated arrays
            n_base = len(normals)
            n_halfspaces = n_components * n_base
            all_normals = np.empty((n_halfspaces, 3), dtype=np.float64)
            all_distances = np.empty(n_halfspaces, dtype=np.float64)

            idx = 0
            for comp_id in range(n_components):
                R = rotation_matrix_axis_angle(twin_axis, twin_angle * comp_id)
                rotated = (R @ normals.T).T
                all_normals[idx : idx + n_base] = rotated
                all_distances[idx : idx + n_base] = distances
                idx += n_base

            vertices = _compute_halfspace_intersection(all_normals, all_distances)
            if len(vertices) < 4:
                raise ValueError("Failed to compute cyclic twin geometry")
            faces = _compute_face_vertices(vertices, all_normals, all_distances)

            # Build face attribution by matching faces to source halfspaces
            final_attribution = []
            for face in faces:
                if len(face) < 3:
                    final_attribution.append(0)
                    continue
                face_verts = vertices[face]
                centroid = np.mean(face_verts, axis=0)
                best_hs_idx = 0
                best_dist = float("inf")
                for hs_idx, (hs_normal, hs_dist) in enumerate(
                    zip(all_normals, all_distances, strict=False)
                ):
                    plane_dist = abs(np.dot(centroid, hs_normal) - hs_dist)
                    if plane_dist < best_dist:
                        best_dist = plane_dist
                        best_hs_idx = hs_idx
                final_attribution.append(best_hs_idx // n_base)

            component = CrystalComponent(
                vertices=vertices, faces=faces, transform=np.eye(4), component_id=0
            )

            return TwinGeometry(
                components=[component],
                render_mode="unified",
                metadata={
                    "blend_mode": "cyclic",
                    "face_attribution": np.array(final_attribution, dtype=np.int32),
                    "n_fold": n_components,
                    "twin_axis": twin_axis,
                    "twin_angle": twin_angle,
                },
            )
        else:
            # Render each component separately
            components = []
            for i in range(n_components):
                angle = twin_angle * i
                R = rotation_matrix_axis_angle(twin_axis, angle)
                rotated_normals = normals @ R.T

                verts = _compute_halfspace_intersection(rotated_normals, distances)
                if len(verts) < 4:
                    continue
                faces = _compute_face_vertices(verts, rotated_normals, distances)

                components.append(
                    CrystalComponent(
                        vertices=verts, faces=faces, transform=np.eye(4), component_id=i
                    )
                )

            return TwinGeometry(
                components=components,
                render_mode="separate",
                metadata={
                    "blend_mode": "cyclic",
                    "n_fold": n_components,
                    "twin_axis": twin_axis,
                    "twin_angle": twin_angle,
                },
            )


class SingleCrystalGeometryGenerator(TwinGeometryGenerator):
    """Single crystal generator - returns base habit without twin modifications.

    Used for twins where external morphology is identical to untwinned crystal:
    - Dauphine law quartz (180° about c-axis - internal/electrical twin only)

    These twins have atomic-level structural differences but no visible
    morphological change.
    """

    def generate(
        self, twin_info: dict[str, Any], normals: np.ndarray, distances: np.ndarray
    ) -> TwinGeometry:
        twin_axis = twin_info.get("axis", np.array([0, 0, 1]))
        twin_angle = twin_info.get("angle", 180)

        # Compute vertices from halfspace intersection
        vertices = _compute_halfspace_intersection(normals, distances)
        if len(vertices) < 4:
            raise ValueError("Failed to compute single crystal geometry")

        # Compute faces
        faces = _compute_face_vertices(vertices, normals, distances)

        component = CrystalComponent(
            vertices=vertices, faces=faces, transform=np.eye(4), component_id=0
        )

        return TwinGeometry(
            components=[component],
            render_mode="single_crystal",
            metadata={
                "blend_mode": "single",
                "description": "Internal/electrical twin - external morphology unchanged",
                "twin_axis": np.asarray(twin_axis),
                "twin_angle": twin_angle,
            },
        )


# Geometry generator registry
GEOMETRY_GENERATORS: dict[str, TwinGeometryGenerator] = {
    "unified": UnifiedGeometryGenerator(),
    "dual_crystal": DualCrystalGeometryGenerator(),
    "v_shaped": VShapedGeometryGenerator(),
    "cyclic": CyclicGeometryGenerator(use_unified=True),
    "cyclic_separate": CyclicGeometryGenerator(use_unified=False),
    "single_crystal": SingleCrystalGeometryGenerator(),
}


def register_generator(name: str, generator: TwinGeometryGenerator) -> None:
    """Register a new geometry generator.

    Args:
        name: Name for the generator (used as render_mode)
        generator: TwinGeometryGenerator instance
    """
    GEOMETRY_GENERATORS[name] = generator


def get_generator(render_mode: str) -> TwinGeometryGenerator:
    """Get the appropriate generator for a render mode.

    Args:
        render_mode: Rendering mode name ('unified', 'dual_crystal', etc.)

    Returns:
        TwinGeometryGenerator instance

    Raises:
        ValueError: If render_mode is not recognized
    """
    if render_mode not in GEOMETRY_GENERATORS:
        available = ", ".join(sorted(GEOMETRY_GENERATORS.keys()))
        raise ValueError(f"Unknown render mode: '{render_mode}'. Available: {available}")
    return GEOMETRY_GENERATORS[render_mode]


def list_generators() -> list[str]:
    """List all available geometry generator names.

    Returns:
        Sorted list of generator names
    """
    return sorted(GEOMETRY_GENERATORS.keys())
