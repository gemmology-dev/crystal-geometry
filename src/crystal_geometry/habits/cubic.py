"""
Cubic crystal system habits.

Includes octahedron, cube, and rhombic dodecahedron.
"""

import numpy as np

from .base import CrystalHabit


class Octahedron(CrystalHabit):
    """Regular octahedron {111} habit.

    Common for: diamond, spinel, magnetite, fluorite
    8 equilateral triangular faces, 6 vertices, 12 edges
    """

    @property
    def name(self) -> str:
        return "Octahedron"

    def _compute_vertices(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

    def _compute_faces(self) -> list[list[int]]:
        return [
            [0, 2, 4],
            [0, 4, 3],
            [0, 3, 5],
            [0, 5, 2],
            [1, 4, 2],
            [1, 3, 4],
            [1, 5, 3],
            [1, 2, 5],
        ]


class Cube(CrystalHabit):
    """Regular cube {100} habit.

    Common for: fluorite, pyrite, galena, halite
    6 square faces, 8 vertices, 12 edges
    """

    @property
    def name(self) -> str:
        return "Cube"

    def _compute_vertices(self) -> np.ndarray:
        # Vertices normalized to lie on unit sphere
        s = 1.0 / np.sqrt(3)
        return np.array(
            [
                [-s, -s, -s],
                [s, -s, -s],
                [s, s, -s],
                [-s, s, -s],
                [-s, -s, s],
                [s, -s, s],
                [s, s, s],
                [-s, s, s],
            ],
            dtype=np.float64,
        )

    def _compute_faces(self) -> list[list[int]]:
        return [
            [3, 2, 1, 0],  # Bottom (z = -s)
            [4, 5, 6, 7],  # Top (z = +s)
            [0, 1, 5, 4],  # Front (y = -s)
            [2, 3, 7, 6],  # Back (y = +s)
            [0, 4, 7, 3],  # Left (x = -s)
            [1, 2, 6, 5],  # Right (x = +s)
        ]


class Dodecahedron(CrystalHabit):
    """Rhombic dodecahedron {110} habit.

    Common for: garnet
    12 rhombic faces, 14 vertices, 24 edges
    """

    @property
    def name(self) -> str:
        return "Rhombic Dodecahedron"

    def _compute_vertices(self) -> np.ndarray:
        a = 1.0
        # 8 vertices from cube + 6 vertices on axes
        return (
            np.array(
                [
                    # Vertices from cube (scaled down)
                    [a, a, a],
                    [a, a, -a],
                    [a, -a, a],
                    [a, -a, -a],
                    [-a, a, a],
                    [-a, a, -a],
                    [-a, -a, a],
                    [-a, -a, -a],
                    # Vertices on axes (extended)
                    [2 * a, 0, 0],
                    [-2 * a, 0, 0],
                    [0, 2 * a, 0],
                    [0, -2 * a, 0],
                    [0, 0, 2 * a],
                    [0, 0, -2 * a],
                ],
                dtype=np.float64,
            )
            / 2.0
        )

    def _compute_faces(self) -> list[list[int]]:
        # 12 rhombic faces
        return [
            [8, 0, 10, 1],
            [8, 1, 13, 3],
            [8, 3, 11, 2],
            [8, 2, 12, 0],
            [9, 4, 12, 6],
            [9, 6, 11, 7],
            [9, 7, 13, 5],
            [9, 5, 10, 4],
            [10, 0, 12, 4],
            [12, 2, 11, 6],
            [11, 3, 13, 7],
            [13, 1, 10, 5],
        ]
