"""
Tetragonal crystal system habits.

Includes tetragonal prism and tetragonal bipyramid.
"""

from typing import Any

import numpy as np

from .base import CrystalHabit


class TetragonalPrism(CrystalHabit):
    """Tetragonal prism {100} habit.

    Common for: zircon, rutile, cassiterite
    4 rectangular prism faces + 2 square end faces
    """

    def __init__(self, scale: float = 1.0, c_ratio: float = 1.8, **params: Any) -> None:
        """Initialize tetragonal prism.

        Args:
            scale: Overall size multiplier
            c_ratio: Height to width ratio
        """
        super().__init__(scale, c_ratio=c_ratio, **params)
        self.c_ratio = c_ratio

    @property
    def name(self) -> str:
        return "Tetragonal Prism"

    def _compute_vertices(self) -> np.ndarray:
        a = 1.0
        h = self.c_ratio
        # Normalize to maintain reasonable aspect ratio
        s = 1.0 / np.sqrt(2)
        return np.array(
            [
                [-a * s, -a * s, -h / 2],
                [a * s, -a * s, -h / 2],
                [a * s, a * s, -h / 2],
                [-a * s, a * s, -h / 2],
                [-a * s, -a * s, h / 2],
                [a * s, -a * s, h / 2],
                [a * s, a * s, h / 2],
                [-a * s, a * s, h / 2],
            ],
            dtype=np.float64,
        )

    def _compute_faces(self) -> list[list[int]]:
        return [
            [3, 2, 1, 0],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 4, 7, 3],  # Left
            [1, 2, 6, 5],  # Right
        ]


class TetragonalBipyramid(CrystalHabit):
    """Tetragonal bipyramid {101} habit.

    Common for: zircon, scheelite
    8 isosceles triangular faces meeting at two apices
    """

    def __init__(self, scale: float = 1.0, apex_ratio: float = 1.5, **params: Any) -> None:
        """Initialize tetragonal bipyramid.

        Args:
            scale: Overall size multiplier
            apex_ratio: Height of apex relative to equatorial width
        """
        super().__init__(scale, apex_ratio=apex_ratio, **params)
        self.apex_ratio = apex_ratio

    @property
    def name(self) -> str:
        return "Tetragonal Bipyramid"

    def _compute_vertices(self) -> np.ndarray:
        a = 1.0
        h = self.apex_ratio
        return np.array(
            [
                # Equatorial square
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [-a, 0.0, 0.0],
                [0.0, -a, 0.0],
                # Apices
                [0.0, 0.0, h],
                [0.0, 0.0, -h],
            ],
            dtype=np.float64,
        )

    def _compute_faces(self) -> list[list[int]]:
        return [
            # Upper pyramid faces (apex at index 4)
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            # Lower pyramid faces (apex at index 5)
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ]
