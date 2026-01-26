"""
Hexagonal crystal system habits.

Includes hexagonal prism and hexagonal bipyramid.
"""

from typing import Any

import numpy as np

from .base import CrystalHabit


class HexagonalPrism(CrystalHabit):
    """Hexagonal prism {10-10} habit.

    Common for: beryl (emerald, aquamarine), quartz, apatite, tourmaline
    6 rectangular prism faces + 2 hexagonal end faces
    """

    def __init__(self, scale: float = 1.0, c_ratio: float = 1.5, **params: Any) -> None:
        """Initialize hexagonal prism.

        Args:
            scale: Overall size multiplier
            c_ratio: Height to width ratio of the prism
        """
        super().__init__(scale, c_ratio=c_ratio, **params)
        self.c_ratio = c_ratio

    @property
    def name(self) -> str:
        return "Hexagonal Prism"

    def _compute_vertices(self) -> np.ndarray:
        # Hexagonal base vertices
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices
        r = 1.0
        h = self.c_ratio

        verts = []
        # Bottom hexagon
        for a in angles:
            verts.append([r * np.cos(a), r * np.sin(a), -h / 2])
        # Top hexagon
        for a in angles:
            verts.append([r * np.cos(a), r * np.sin(a), h / 2])

        return np.array(verts, dtype=np.float64)

    def _compute_faces(self) -> list[list[int]]:
        faces = []
        # Bottom face (reversed for outward normal)
        faces.append([5, 4, 3, 2, 1, 0])
        # Top face
        faces.append([6, 7, 8, 9, 10, 11])
        # Side faces
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([i, next_i, next_i + 6, i + 6])
        return faces


class HexagonalBipyramid(CrystalHabit):
    """Hexagonal bipyramid {10-11} habit.

    Common for: quartz crystals
    12 triangular faces meeting at two apices
    """

    def __init__(self, scale: float = 1.0, apex_ratio: float = 1.2, **params: Any) -> None:
        """Initialize hexagonal bipyramid.

        Args:
            scale: Overall size multiplier
            apex_ratio: Height of apex relative to equatorial radius
        """
        super().__init__(scale, apex_ratio=apex_ratio, **params)
        self.apex_ratio = apex_ratio

    @property
    def name(self) -> str:
        return "Hexagonal Bipyramid"

    def _compute_vertices(self) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        r = 1.0
        h = self.apex_ratio

        verts = []
        # Equatorial hexagon
        for a in angles:
            verts.append([r * np.cos(a), r * np.sin(a), 0.0])
        # Apex vertices
        verts.append([0.0, 0.0, h])  # Top apex (index 6)
        verts.append([0.0, 0.0, -h])  # Bottom apex (index 7)

        return np.array(verts, dtype=np.float64)

    def _compute_faces(self) -> list[list[int]]:
        faces = []
        # Upper pyramid faces (connecting to top apex at index 6)
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([i, next_i, 6])
        # Lower pyramid faces (connecting to bottom apex at index 7)
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([next_i, i, 7])
        return faces
