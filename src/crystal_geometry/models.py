"""
Crystal Geometry Models.

Data classes for 3D crystal geometry representation.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cdl_parser import CrystalForm


@dataclass
class TwinMetadata:
    """Metadata for twinned crystal geometry.

    Attributes:
        twin_law: Name of the twin law used
        render_mode: Rendering strategy ('unified', 'separate', etc.)
        n_components: Number of crystal components
        twin_axis: Rotation axis as (x, y, z) tuple
        twin_angle: Rotation angle in degrees
        face_attribution: Optional array mapping faces to components
    """

    twin_law: str
    render_mode: str
    n_components: int
    twin_axis: tuple[float, float, float]
    twin_angle: float
    face_attribution: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "twin_law": self.twin_law,
            "render_mode": self.render_mode,
            "n_components": self.n_components,
            "twin_axis": self.twin_axis,
            "twin_angle": self.twin_angle,
            "face_attribution": self.face_attribution,
        }


@dataclass
class CrystalGeometry:
    """3D crystal geometry with vertices and faces.

    Attributes:
        vertices: Nx3 array of vertex positions
        faces: List of faces, each face is list of vertex indices (counter-clockwise)
        face_normals: Normal vector for each face
        face_forms: Which form each face belongs to (index into forms list)
        face_millers: Miller index (h, k, l) for each face
        forms: Original form definitions from CDL
        component_ids: Optional array of component IDs for each face (for twins)
        twin_metadata: Optional metadata for twinned crystals
    """

    vertices: np.ndarray  # Nx3 array of vertex positions
    faces: list[list[int]]  # List of faces, each face is list of vertex indices
    face_normals: list[np.ndarray]  # Normal vector for each face
    face_forms: list[int]  # Which form each face belongs to (index into forms list)
    face_millers: list[tuple[int, int, int]]  # Miller index for each face
    forms: list[CrystalForm] = field(default_factory=list)  # Original form definitions
    component_ids: list[int] | None = None  # Component ID for each face (for twins)
    twin_metadata: TwinMetadata | None = None  # Metadata for twinned crystals

    def get_edges(self) -> list[tuple[int, int]]:
        """Get all unique edges as vertex index pairs.

        Returns:
            List of (v1, v2) tuples where v1 < v2
        """
        edges: set[tuple[int, int]] = set()
        for face in self.faces:
            n = len(face)
            for i in range(n):
                v1, v2 = face[i], face[(i + 1) % n]
                edge = (min(v1, v2), max(v1, v2))
                edges.add(edge)
        return list(edges)

    def center(self) -> np.ndarray:
        """Get center of geometry (mean of all vertices)."""
        return np.mean(self.vertices, axis=0)

    def scale_to_unit(self) -> "CrystalGeometry":
        """Scale geometry to fit in unit sphere.

        Returns:
            New CrystalGeometry with vertices scaled to max distance of 1.0
        """
        max_dist = np.max(np.linalg.norm(self.vertices, axis=1))
        if max_dist > 0:
            new_verts = self.vertices / max_dist
        else:
            new_verts = self.vertices.copy()

        return CrystalGeometry(
            vertices=new_verts,
            faces=self.faces,
            face_normals=self.face_normals,
            face_forms=self.face_forms,
            face_millers=self.face_millers,
            forms=self.forms,
        )

    def translate(self, offset: np.ndarray) -> "CrystalGeometry":
        """Translate geometry by offset.

        Args:
            offset: 3D translation vector

        Returns:
            New translated CrystalGeometry
        """
        return CrystalGeometry(
            vertices=self.vertices + offset,
            faces=self.faces,
            face_normals=self.face_normals,
            face_forms=self.face_forms,
            face_millers=self.face_millers,
            forms=self.forms,
        )

    def rotate(self, matrix: np.ndarray) -> "CrystalGeometry":
        """Rotate geometry by rotation matrix.

        Args:
            matrix: 3x3 rotation matrix

        Returns:
            New rotated CrystalGeometry
        """
        new_verts = self.vertices @ matrix.T
        new_normals = [n @ matrix.T for n in self.face_normals]

        return CrystalGeometry(
            vertices=new_verts,
            faces=self.faces,
            face_normals=new_normals,
            face_forms=self.face_forms,
            face_millers=self.face_millers,
            forms=self.forms,
        )

    def euler_characteristic(self) -> int:
        """Compute Euler characteristic V - E + F.

        Returns:
            Euler characteristic (should be 2 for convex polyhedra)
        """
        V = len(self.vertices)
        E = len(self.get_edges())
        F = len(self.faces)
        return V - E + F

    def is_valid(self) -> bool:
        """Check if geometry is valid (Euler's formula holds).

        Returns:
            True if V - E + F = 2
        """
        return self.euler_characteristic() == 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "vertices": self.vertices.tolist(),
            "faces": self.faces,
            "face_normals": [n.tolist() for n in self.face_normals],
            "face_forms": self.face_forms,
            "face_millers": self.face_millers,
        }
        if self.component_ids is not None:
            result["component_ids"] = self.component_ids
        if self.twin_metadata is not None:
            result["twin_metadata"] = self.twin_metadata.to_dict()
        return result


@dataclass
class LatticeParams:
    """Crystal lattice parameters.

    Attributes:
        a, b, c: Lattice vector lengths
        alpha, beta, gamma: Angles between lattice vectors (in radians)
    """

    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    alpha: float = np.pi / 2  # 90 degrees
    beta: float = np.pi / 2
    gamma: float = np.pi / 2

    @classmethod
    def cubic(cls) -> "LatticeParams":
        """Create cubic lattice (a = b = c, all angles 90°)."""
        return cls(1.0, 1.0, 1.0, np.pi / 2, np.pi / 2, np.pi / 2)

    @classmethod
    def hexagonal(cls, c_ratio: float = 1.633) -> "LatticeParams":
        """Create hexagonal lattice (a = b, gamma = 120°)."""
        return cls(1.0, 1.0, c_ratio, np.pi / 2, np.pi / 2, 2 * np.pi / 3)

    @classmethod
    def tetragonal(cls, c_ratio: float = 1.0) -> "LatticeParams":
        """Create tetragonal lattice (a = b, all angles 90°)."""
        return cls(1.0, 1.0, c_ratio, np.pi / 2, np.pi / 2, np.pi / 2)


# Default lattice for cubic system
DEFAULT_LATTICE = LatticeParams.cubic()
