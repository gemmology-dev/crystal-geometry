"""
Orthorhombic crystal system habits.

Includes orthorhombic prism with three unequal axes.
"""

from typing import Any

import numpy as np

from .base import CrystalHabit


class OrthorhombicPrism(CrystalHabit):
    """Orthorhombic prism habit (rectangular box with unequal sides).

    Common for: topaz, olivine (peridot), chrysoberyl, tanzanite
    6 rectangular faces with three different dimensions
    """

    def __init__(
        self, scale: float = 1.0, b_ratio: float = 1.2, c_ratio: float = 1.5, **params: Any
    ) -> None:
        """Initialize orthorhombic prism.

        Args:
            scale: Overall size multiplier
            b_ratio: b-axis to a-axis ratio
            c_ratio: c-axis to a-axis ratio
        """
        super().__init__(scale, b_ratio=b_ratio, c_ratio=c_ratio, **params)
        self.b_ratio = b_ratio
        self.c_ratio = c_ratio

    @property
    def name(self) -> str:
        return "Orthorhombic Prism"

    def _compute_vertices(self) -> np.ndarray:
        a = 1.0
        b = self.b_ratio
        c = self.c_ratio
        return (
            np.array(
                [
                    [-a, -b, -c],
                    [a, -b, -c],
                    [a, b, -c],
                    [-a, b, -c],
                    [-a, -b, c],
                    [a, -b, c],
                    [a, b, c],
                    [-a, b, c],
                ],
                dtype=np.float64,
            )
            / 2.0
        )

    def _compute_faces(self) -> list[list[int]]:
        return [
            [3, 2, 1, 0],  # Bottom (z = -c/2)
            [4, 5, 6, 7],  # Top (z = +c/2)
            [0, 1, 5, 4],  # Front (y = -b/2)
            [2, 3, 7, 6],  # Back (y = +b/2)
            [0, 4, 7, 3],  # Left (x = -a/2)
            [1, 2, 6, 5],  # Right (x = +a/2)
        ]
