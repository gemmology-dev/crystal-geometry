"""
Base class for crystal habit geometry.

Provides the abstract interface that all habit classes implement.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class CrystalHabit(ABC):
    """Base class for crystal habit geometry generation.

    Crystal habits represent the characteristic external shape of a crystal,
    determined by the relative development of different crystal faces.

    Subclasses implement specific habit geometries (octahedron, cube, prism, etc.)
    by providing vertices and face definitions.

    Attributes:
        scale: Overall size multiplier
        params: Additional parameters specific to the habit
    """

    def __init__(self, scale: float = 1.0, **params: Any):
        """Initialize crystal habit.

        Args:
            scale: Size multiplier applied to all vertices
            **params: Habit-specific parameters (e.g., c_ratio for prisms)
        """
        self.scale = scale
        self.params = params
        self._vertices: np.ndarray | None = None
        self._faces: list[list[int]] | None = None

    @property
    def vertices(self) -> np.ndarray:
        """Get scaled vertex positions.

        Returns:
            Nx3 array of vertex positions
        """
        if self._vertices is None:
            self._vertices = self._compute_vertices()
        return self._vertices * self.scale

    @property
    def faces(self) -> list[list[int]]:
        """Get face vertex indices.

        Returns:
            List of faces, each face is a list of vertex indices
            in counter-clockwise order when viewed from outside
        """
        if self._faces is None:
            self._faces = self._compute_faces()
        return self._faces

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable habit name.

        Returns:
            Name string like 'Octahedron' or 'Hexagonal Prism'
        """
        pass

    @abstractmethod
    def _compute_vertices(self) -> np.ndarray:
        """Compute unscaled vertex positions.

        Returns:
            Nx3 array of vertex positions (before scaling)
        """
        pass

    @abstractmethod
    def _compute_faces(self) -> list[list[int]]:
        """Compute face definitions.

        Returns:
            List of faces, each face is a list of vertex indices
        """
        pass

    def get_face_vertices(self) -> list[np.ndarray]:
        """Get vertex coordinates for each face.

        Convenience method for direct rendering (e.g., Poly3DCollection).

        Returns:
            List of Mx3 arrays, each containing the vertices of one face
        """
        verts = self.vertices
        return [verts[face] for face in self.faces]

    def get_halfspaces(self) -> tuple[np.ndarray, np.ndarray]:
        """Get halfspace representation of the habit.

        Computes the normals and distances for halfspace intersection.
        Each face contributes one halfspace constraint: normal . x <= distance

        Returns:
            Tuple of (normals, distances):
            - normals: Nx3 array of unit normal vectors (pointing outward)
            - distances: N-element array of distances from origin
        """
        face_vertices = self.get_face_vertices()
        normals = []
        distances = []

        # Compute centroid for consistent outward normals
        all_verts = self.vertices
        centroid = np.mean(all_verts, axis=0)

        for face_verts in face_vertices:
            if len(face_verts) < 3:
                continue

            # Compute plane from first 3 vertices
            v0, v1, v2 = face_verts[:3]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)

            if norm_len < 1e-10:
                continue

            normal = normal / norm_len
            distance = np.dot(normal, v0)

            # Ensure normal points outward
            face_center = np.mean(face_verts, axis=0)
            if np.dot(normal, face_center - centroid) < 0:
                normal = -normal
                distance = -distance

            normals.append(normal)
            distances.append(distance)

        return np.array(normals), np.array(distances)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        if params_str:
            return f"{self.__class__.__name__}(scale={self.scale}, {params_str})"
        return f"{self.__class__.__name__}(scale={self.scale})"
