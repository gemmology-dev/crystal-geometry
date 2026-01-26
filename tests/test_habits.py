"""Tests for the habit system."""

import numpy as np
import pytest

from crystal_geometry.habits import (
    HABIT_REGISTRY,
    Barrel,
    CrystalHabit,
    Cube,
    HexagonalPrism,
    Octahedron,
    Pyritohedron,
    QuartzCrystal,
    get_gemstone_habits,
    get_habit,
    list_habits,
)


class TestHabitRegistry:
    """Tests for habit registry functions."""

    def test_list_habits(self):
        """Should list all available habits."""
        habits = list_habits()
        assert len(habits) >= 14
        assert "octahedron" in habits
        assert "cube" in habits
        assert "quartz_crystal" in habits

    def test_get_habit_valid(self):
        """Should get valid habits."""
        habit = get_habit("octahedron")
        assert isinstance(habit, CrystalHabit)
        assert isinstance(habit, Octahedron)

    def test_get_habit_invalid(self):
        """Should raise for unknown habit."""
        with pytest.raises(ValueError, match="Unknown habit"):
            get_habit("nonexistent_habit")

    def test_get_habit_normalized_name(self):
        """Should normalize habit names."""
        # Spaces and hyphens should be converted to underscores
        habit1 = get_habit("hexagonal_prism")
        habit2 = get_habit("hexagonal-prism")
        habit3 = get_habit("hexagonal prism")
        assert type(habit1) is type(habit2) is type(habit3)

    def test_get_habit_with_scale(self):
        """Should apply scale parameter."""
        habit = get_habit("cube", scale=2.0)
        assert habit.scale == 2.0

        # Vertices should be scaled
        max_dist = np.max(np.linalg.norm(habit.vertices, axis=1))
        assert max_dist > 1.5  # Should be larger than default


class TestGemstoneHabits:
    """Tests for gemstone to habit mapping."""

    def test_get_gemstone_habits(self):
        """Should get habits for known gemstones."""
        habits = get_gemstone_habits("diamond")
        assert "octahedron" in habits

        habits = get_gemstone_habits("quartz")
        assert "quartz_crystal" in habits

    def test_get_gemstone_habits_default(self):
        """Should return default for unknown gemstones."""
        habits = get_gemstone_habits("unknown_gem")
        assert habits == ["cube"]


class TestBaseHabitInterface:
    """Tests for CrystalHabit base class interface."""

    @pytest.mark.parametrize("habit_name", list(HABIT_REGISTRY.keys()))
    def test_habit_has_name(self, habit_name):
        """All habits should have a name property."""
        habit = get_habit(habit_name)
        assert habit.name
        assert isinstance(habit.name, str)

    @pytest.mark.parametrize("habit_name", list(HABIT_REGISTRY.keys()))
    def test_habit_has_vertices(self, habit_name):
        """All habits should produce valid vertices."""
        habit = get_habit(habit_name)
        vertices = habit.vertices
        assert isinstance(vertices, np.ndarray)
        assert vertices.shape[1] == 3
        assert len(vertices) >= 4

    @pytest.mark.parametrize("habit_name", list(HABIT_REGISTRY.keys()))
    def test_habit_has_faces(self, habit_name):
        """All habits should produce valid faces."""
        habit = get_habit(habit_name)
        faces = habit.faces
        assert isinstance(faces, list)
        assert len(faces) >= 4

        # Each face should reference valid vertex indices
        n_verts = len(habit.vertices)
        for face in faces:
            assert len(face) >= 3
            for idx in face:
                assert 0 <= idx < n_verts

    @pytest.mark.parametrize("habit_name", list(HABIT_REGISTRY.keys()))
    def test_habit_get_face_vertices(self, habit_name):
        """get_face_vertices should return coordinate arrays."""
        habit = get_habit(habit_name)
        face_verts = habit.get_face_vertices()

        assert len(face_verts) == len(habit.faces)
        for fv in face_verts:
            assert isinstance(fv, np.ndarray)
            assert fv.shape[1] == 3

    @pytest.mark.parametrize("habit_name", list(HABIT_REGISTRY.keys()))
    def test_habit_get_halfspaces(self, habit_name):
        """get_halfspaces should return normals and distances."""
        habit = get_habit(habit_name)
        normals, distances = habit.get_halfspaces()

        assert isinstance(normals, np.ndarray)
        assert isinstance(distances, np.ndarray)
        assert normals.shape[1] == 3
        assert len(normals) == len(distances)

        # Normals should be approximately unit length
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)


class TestOctahedron:
    """Tests specific to Octahedron habit."""

    def test_octahedron_vertices(self):
        """Octahedron should have 6 vertices on axes."""
        habit = Octahedron()
        vertices = habit.vertices

        assert len(vertices) == 6

        # Vertices should be on the coordinate axes
        for v in vertices:
            non_zero = np.sum(np.abs(v) > 0.01)
            assert non_zero == 1  # Only one component is non-zero

    def test_octahedron_faces(self):
        """Octahedron should have 8 triangular faces."""
        habit = Octahedron()
        faces = habit.faces

        assert len(faces) == 8
        for face in faces:
            assert len(face) == 3


class TestCube:
    """Tests specific to Cube habit."""

    def test_cube_vertices(self):
        """Cube should have 8 vertices at corners."""
        habit = Cube()
        vertices = habit.vertices

        assert len(vertices) == 8

    def test_cube_faces(self):
        """Cube should have 6 square faces."""
        habit = Cube()
        faces = habit.faces

        assert len(faces) == 6
        for face in faces:
            assert len(face) == 4


class TestHexagonalPrism:
    """Tests specific to HexagonalPrism habit."""

    def test_hexagonal_prism_vertices(self):
        """Hexagonal prism should have 12 vertices."""
        habit = HexagonalPrism()
        vertices = habit.vertices

        assert len(vertices) == 12

    def test_hexagonal_prism_faces(self):
        """Hexagonal prism should have 8 faces (2 hexagon + 6 rectangle)."""
        habit = HexagonalPrism()
        faces = habit.faces

        assert len(faces) == 8

    def test_hexagonal_prism_c_ratio(self):
        """c_ratio should affect height."""
        habit1 = HexagonalPrism(c_ratio=1.0)
        habit2 = HexagonalPrism(c_ratio=2.0)

        z1 = np.max(habit1.vertices[:, 2])
        z2 = np.max(habit2.vertices[:, 2])

        assert z2 > z1


class TestQuartzCrystal:
    """Tests specific to QuartzCrystal habit."""

    def test_quartz_vertices(self):
        """Quartz crystal should have 14 vertices (2 apices + 2 hexagons)."""
        habit = QuartzCrystal()
        vertices = habit.vertices

        assert len(vertices) == 14

    def test_quartz_faces(self):
        """Quartz should have 18 faces (6 top + 6 side + 6 bottom)."""
        habit = QuartzCrystal()
        faces = habit.faces

        assert len(faces) == 18


class TestBarrel:
    """Tests specific to Barrel (corundum) habit."""

    def test_barrel_taper(self):
        """Barrel with taper should have smaller top."""
        habit = Barrel(taper=0.5)
        vertices = habit.vertices

        # Top vertices (6-11) should have smaller x, y than bottom
        bottom_r = np.mean(np.sqrt(vertices[:6, 0] ** 2 + vertices[:6, 1] ** 2))
        top_r = np.mean(np.sqrt(vertices[6:, 0] ** 2 + vertices[6:, 1] ** 2))

        assert top_r < bottom_r


class TestPyritohedron:
    """Tests specific to Pyritohedron habit."""

    def test_pyritohedron_faces(self):
        """Pyritohedron should have faces (may be triangulated by ConvexHull)."""
        habit = Pyritohedron()
        faces = habit.faces

        # ConvexHull triangulates the surface, so we get more than 12 faces
        # A pyritohedron has 12 pentagonal faces, but triangulation produces more
        assert len(faces) >= 12

        # Each face should have at least 3 vertices
        for face in faces:
            assert len(face) >= 3
