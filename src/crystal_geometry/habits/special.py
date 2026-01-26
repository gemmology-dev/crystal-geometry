"""
Special crystal habits with unique geometries.

Includes barrel (corundum), tabular, trapezohedron, quartz crystal with
rhombohedral terminations, and pyritohedron.
"""

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from .base import CrystalHabit


class Barrel(CrystalHabit):
    """Barrel habit (tapered hexagonal prism).

    Common for: corundum (ruby, sapphire)
    Hexagonal prism with tapered ends, giving a barrel-like appearance
    """

    def __init__(
        self, scale: float = 1.0, taper: float = 0.7, c_ratio: float = 1.2, **params: Any
    ) -> None:
        """Initialize barrel habit.

        Args:
            scale: Overall size multiplier
            taper: Top/bottom radius ratio (< 1 for tapering)
            c_ratio: Height to width ratio
        """
        super().__init__(scale, taper=taper, c_ratio=c_ratio, **params)
        self.taper = taper
        self.c_ratio = c_ratio

    @property
    def name(self) -> str:
        return "Barrel"

    def _compute_vertices(self) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        r_base = 1.0
        r_top = r_base * self.taper
        h = self.c_ratio

        verts = []
        # Bottom hexagon (larger)
        for a in angles:
            verts.append([r_base * np.cos(a), r_base * np.sin(a), -h / 2])
        # Top hexagon (smaller due to taper)
        for a in angles:
            verts.append([r_top * np.cos(a), r_top * np.sin(a), h / 2])

        return np.array(verts, dtype=np.float64)

    def _compute_faces(self) -> list[list[int]]:
        faces = []
        # Bottom face
        faces.append([5, 4, 3, 2, 1, 0])
        # Top face
        faces.append([6, 7, 8, 9, 10, 11])
        # Side faces (trapezoidal)
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([i, next_i, next_i + 6, i + 6])
        return faces


class Tabular(CrystalHabit):
    """Tabular habit (flattened plate-like crystal).

    Common for: feldspar, mica, barite
    Thin rectangular plate with dominant pinacoid faces
    """

    def __init__(self, scale: float = 1.0, thickness: float = 0.3, **params: Any) -> None:
        """Initialize tabular habit.

        Args:
            scale: Overall size multiplier
            thickness: Relative thickness (< 1 for thin plates)
        """
        super().__init__(scale, thickness=thickness, **params)
        self.thickness = thickness

    @property
    def name(self) -> str:
        return "Tabular"

    def _compute_vertices(self) -> np.ndarray:
        a = 1.0
        b = 0.8
        h = self.thickness
        return np.array(
            [
                [-a, -b, -h],
                [a, -b, -h],
                [a, b, -h],
                [-a, b, -h],
                [-a, -b, h],
                [a, -b, h],
                [a, b, h],
                [-a, b, h],
            ],
            dtype=np.float64,
        )

    def _compute_faces(self) -> list[list[int]]:
        return [
            [3, 2, 1, 0],  # Bottom (large pinacoid face)
            [4, 5, 6, 7],  # Top (large pinacoid face)
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 4, 7, 3],  # Left
            [1, 2, 6, 5],  # Right
        ]


class FeldsparTabular(CrystalHabit):
    """Monoclinic feldspar habit with asymmetric prism faces.

    Common for: orthoclase, plagioclase (albite), microcline

    This habit has asymmetric {110} and {-110} face development, which makes
    twin boundaries visible when the crystal is twinned. The standard symmetric
    tabular habit produces invisible twins for 180° rotation twins like Albite
    and Manebach because both halves end up in identical orientations.

    The geometry approximates monoclinic 2/m symmetry with:
    - {001} pinacoid (thin, dominant)
    - {010} side pinacoid
    - {110} prism face (smaller)
    - {-110} prism face (larger, asymmetric)
    """

    def __init__(
        self,
        scale: float = 1.0,
        thickness: float = 0.3,
        asymmetry: float = 0.3,
        **params: Any,
    ) -> None:
        """Initialize feldspar tabular habit.

        Args:
            scale: Overall size multiplier
            thickness: Relative thickness of the plate (z-dimension)
            asymmetry: Difference between {110} and {-110} face development.
                       Higher values make twins more visible. Default 0.3.
        """
        super().__init__(scale, thickness=thickness, asymmetry=asymmetry, **params)
        self.thickness = thickness
        self.asymmetry = asymmetry

    @property
    def name(self) -> str:
        return "Feldspar Tabular"

    def _compute_vertices(self) -> np.ndarray:
        h = self.thickness
        asym = self.asymmetry

        # Base dimensions for the tabular shape
        # The asymmetry creates different angles for {110} vs {-110}
        # This breaks the mirror symmetry that makes twins invisible

        # Front face vertices (y < 0)
        x_front = 1.0
        y_front = -0.8

        # Back face vertices (y > 0) - asymmetric
        x_back_left = -1.0 - asym  # {-110} extends further
        x_back_right = 1.0
        y_back = 0.8

        return np.array(
            [
                # Bottom face (z = -h)
                [-1.0, y_front, -h],  # 0: front-left
                [x_front, y_front, -h],  # 1: front-right
                [x_back_right, y_back, -h],  # 2: back-right
                [x_back_left, y_back, -h],  # 3: back-left (asymmetric)
                # Top face (z = +h)
                [-1.0, y_front, h],  # 4: front-left
                [x_front, y_front, h],  # 5: front-right
                [x_back_right, y_back, h],  # 6: back-right
                [x_back_left, y_back, h],  # 7: back-left (asymmetric)
            ],
            dtype=np.float64,
        )

    def _compute_faces(self) -> list[list[int]]:
        return [
            [3, 2, 1, 0],  # Bottom {001} pinacoid
            [4, 5, 6, 7],  # Top {001} pinacoid
            [0, 1, 5, 4],  # Front {010} face
            [2, 3, 7, 6],  # Back {0-10} face
            [0, 4, 7, 3],  # Left {-110} face (larger due to asymmetry)
            [1, 2, 6, 5],  # Right {110} face
        ]


class Trapezohedron(CrystalHabit):
    """Tetragonal trapezohedron habit.

    Common for: garnet
    24 trapezoidal faces in full form, simplified here to 8 faces
    """

    @property
    def name(self) -> str:
        return "Trapezohedron"

    def _compute_vertices(self) -> np.ndarray:
        a = 1.0
        h = 0.7
        return np.array(
            [
                # Equatorial square
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [-a, 0.0, 0.0],
                [0.0, -a, 0.0],
                # Upper square (rotated 45°)
                [a * 0.7, a * 0.7, h],
                [-a * 0.7, a * 0.7, h],
                [-a * 0.7, -a * 0.7, h],
                [a * 0.7, -a * 0.7, h],
                # Lower square (rotated 45°)
                [a * 0.7, a * 0.7, -h],
                [-a * 0.7, a * 0.7, -h],
                [-a * 0.7, -a * 0.7, -h],
                [a * 0.7, -a * 0.7, -h],
                # Apices
                [0.0, 0.0, a * 1.2],
                [0.0, 0.0, -a * 1.2],
            ],
            dtype=np.float64,
        )

    def _compute_faces(self) -> list[list[int]]:
        return [
            # Upper trapezohedral faces
            [0, 4, 12],
            [0, 12, 7],
            [1, 4, 12],
            [1, 12, 5],
            [2, 5, 12],
            [2, 12, 6],
            [3, 6, 12],
            [3, 12, 7],
            # Lower trapezohedral faces
            [0, 8, 13],
            [0, 13, 11],
            [1, 8, 13],
            [1, 13, 9],
            [2, 9, 13],
            [2, 13, 10],
            [3, 10, 13],
            [3, 13, 11],
            # Middle band
            [0, 4, 8],
            [4, 1, 8],
            [8, 1, 9],
            [1, 5, 9],
            [5, 2, 9],
            [9, 2, 10],
            [2, 6, 10],
            [6, 3, 10],
            [10, 3, 11],
            [3, 7, 11],
            [7, 0, 11],
            [11, 0, 8],
        ]


class QuartzCrystal(CrystalHabit):
    """Quartz crystal habit with hexagonal prism and pointed terminations.

    The classic quartz habit has:
    - 6 prism faces (m-faces) forming hexagonal body
    - 6 rhombohedral termination faces (r-faces) forming pointed ends

    Common for: rock crystal, amethyst, citrine, smoky quartz
    """

    def __init__(
        self,
        scale: float = 1.0,
        c_ratio: float = 2.5,
        termination_angle: float = 38.2,
        **params: Any,
    ) -> None:
        """Initialize quartz crystal habit.

        Args:
            scale: Overall size multiplier
            c_ratio: Height to width ratio of the prism body
            termination_angle: Angle of termination faces from horizontal (degrees).
                              Default 38.2° is typical for quartz r-faces.
        """
        super().__init__(scale, c_ratio=c_ratio, termination_angle=termination_angle, **params)
        self.c_ratio = c_ratio
        self.termination_angle = termination_angle

    @property
    def name(self) -> str:
        return "Quartz Crystal"

    def _compute_vertices(self) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices per ring
        r = 1.0

        # Calculate termination height based on angle
        term_height = r * np.tan(np.radians(self.termination_angle))

        # Prism body height
        body_half_height = self.c_ratio / 2

        verts = []

        # Top apex (index 0)
        verts.append([0.0, 0.0, body_half_height + term_height])

        # Upper hexagon (indices 1-6)
        for a in angles:
            verts.append([r * np.cos(a), r * np.sin(a), body_half_height])

        # Lower hexagon (indices 7-12)
        for a in angles:
            verts.append([r * np.cos(a), r * np.sin(a), -body_half_height])

        # Bottom apex (index 13)
        verts.append([0.0, 0.0, -(body_half_height + term_height)])

        return np.array(verts, dtype=np.float64)

    def _compute_faces(self) -> list[list[int]]:
        faces = []

        # Top termination faces (6 triangles from apex 0 to upper hexagon 1-6)
        for i in range(6):
            curr = i + 1
            next_v = (i + 1) % 6 + 1
            faces.append([0, curr, next_v])

        # Prism side faces (6 rectangles)
        for i in range(6):
            upper_curr = i + 1
            upper_next = (i + 1) % 6 + 1
            lower_curr = i + 7
            lower_next = (i + 1) % 6 + 7
            faces.append([upper_curr, lower_curr, lower_next, upper_next])

        # Bottom termination faces (6 triangles from lower hexagon to apex 13)
        for i in range(6):
            curr = i + 7
            next_v = (i + 1) % 6 + 7
            faces.append([13, next_v, curr])

        return faces


class Pyritohedron(CrystalHabit):
    """Pyritohedron (pentagonal dodecahedron) {210} habit.

    Characteristic habit of pyrite (fool's gold).
    12 irregular pentagonal faces, 30 edges, 20 vertices
    """

    def __init__(self, scale: float = 1.0, h: float = 0.618, **params: Any) -> None:
        """Initialize pyritohedron.

        Args:
            scale: Overall size multiplier
            h: Shape parameter (h ≈ 0.618 = 1/phi gives typical pyritohedron)
        """
        super().__init__(scale, h=h, **params)
        self.h = h

    @property
    def name(self) -> str:
        return "Pyritohedron"

    def _compute_vertices(self) -> np.ndarray:
        h = self.h

        # The 20 vertices of a pyritohedron from cyclic permutations
        verts_list: list[list[float]] = []

        # (0, ±1, ±h)
        verts_list.extend(
            [
                [0, 1, h],
                [0, 1, -h],
                [0, -1, h],
                [0, -1, -h],
            ]
        )

        # (±h, 0, ±1)
        verts_list.extend(
            [
                [h, 0, 1],
                [h, 0, -1],
                [-h, 0, 1],
                [-h, 0, -1],
            ]
        )

        # (±1, ±h, 0)
        verts_list.extend(
            [
                [1, h, 0],
                [1, -h, 0],
                [-1, h, 0],
                [-1, -h, 0],
            ]
        )

        # (0, ±h, ±1)
        verts_list.extend(
            [
                [0, h, 1],
                [0, -h, 1],
                [0, h, -1],
                [0, -h, -1],
            ]
        )

        # (±1, 0, ±h)
        verts_list.extend(
            [
                [1, 0, h],
                [1, 0, -h],
                [-1, 0, h],
                [-1, 0, -h],
            ]
        )

        verts = np.array(verts_list, dtype=np.float64)

        # Normalize to unit sphere
        max_r = np.max(np.linalg.norm(verts, axis=1))
        verts = verts / max_r

        return verts

    def _compute_faces(self) -> list[list[int]]:
        """Compute faces using convex hull."""
        verts = self._compute_vertices()

        try:
            hull = ConvexHull(verts)
        except Exception:
            # Fallback to simple triangulation
            return []

        # Group triangles by face normal
        face_groups: dict[tuple, list[set]] = {}

        for simplex in hull.simplices:
            v0, v1, v2 = verts[simplex]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm

            # Ensure outward normal
            center = (v0 + v1 + v2) / 3
            if np.dot(normal, center) < 0:
                normal = -normal

            # Round for grouping
            normal_key = tuple(np.round(normal, 4))

            if normal_key not in face_groups:
                face_groups[normal_key] = []
            face_groups[normal_key].append(set(simplex))

        # Merge triangles into polygons
        faces = []
        for normal_key, triangles in face_groups.items():
            all_verts_set: set[int] = set()
            for tri in triangles:
                all_verts_set.update(tri)

            vert_list = list(all_verts_set)
            if len(vert_list) < 3:
                continue

            # Sort vertices counter-clockwise
            center = np.mean(verts[vert_list], axis=0)
            normal = np.array(normal_key)

            # Create 2D coordinate system on face plane
            if abs(normal[0]) < 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            else:
                ref = np.array([0.0, 1.0, 0.0])
            u = np.cross(normal, ref)
            u = u / np.linalg.norm(u)
            v_axis = np.cross(normal, u)

            angles = []
            for idx in vert_list:
                pt = verts[idx] - center
                angle = np.arctan2(np.dot(pt, v_axis), np.dot(pt, u))
                angles.append(angle)

            sorted_indices = [x for _, x in sorted(zip(angles, vert_list, strict=False))]
            faces.append(sorted_indices)

        return faces
