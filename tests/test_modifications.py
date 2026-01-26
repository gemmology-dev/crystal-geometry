"""Tests for the modifications system."""

import numpy as np

from crystal_geometry.modifications import (
    AXIS_MAP,
    apply_elongation,
    apply_flatten,
    apply_modifications,
    apply_taper,
    apply_twist,
)


class TestAxisMap:
    """Tests for axis name to vector mapping."""

    def test_axis_map_contents(self):
        """Should have all standard axes."""
        assert "a" in AXIS_MAP
        assert "b" in AXIS_MAP
        assert "c" in AXIS_MAP
        assert "x" in AXIS_MAP
        assert "y" in AXIS_MAP
        assert "z" in AXIS_MAP

    def test_axis_map_values(self):
        """Axes should be unit vectors along coordinate axes."""
        assert np.allclose(AXIS_MAP["x"], [1, 0, 0])
        assert np.allclose(AXIS_MAP["y"], [0, 1, 0])
        assert np.allclose(AXIS_MAP["z"], [0, 0, 1])


class TestApplyElongation:
    """Tests for elongation modification."""

    def test_elongation_identity(self):
        """Elongation with ratio 1.0 should not change vertices."""
        vertices = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        result = apply_elongation(vertices, {"axis": "c", "ratio": 1.0})
        assert np.allclose(result, vertices)

    def test_elongation_doubles_z(self):
        """Elongation along c with ratio 2 should double z coordinates."""
        vertices = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 2],
            ],
            dtype=np.float64,
        )

        result = apply_elongation(vertices, {"axis": "c", "ratio": 2.0})

        # X and Y should be unchanged
        assert np.allclose(result[:, 0], vertices[:, 0])
        assert np.allclose(result[:, 1], vertices[:, 1])

        # Z should be doubled
        assert np.allclose(result[:, 2], vertices[:, 2] * 2)

    def test_elongation_along_x(self):
        """Elongation along x/a axis."""
        vertices = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [2, 0, 0],
            ],
            dtype=np.float64,
        )

        result = apply_elongation(vertices, {"axis": "a", "ratio": 3.0})

        # Y and Z should be unchanged
        assert np.allclose(result[:, 1], vertices[:, 1])
        assert np.allclose(result[:, 2], vertices[:, 2])

        # X should be tripled
        assert np.allclose(result[:, 0], vertices[:, 0] * 3)

    def test_elongation_shorthand_format(self):
        """Elongation with shorthand params {'c': 1.5}."""
        vertices = np.array(
            [
                [0, 0, 1],
                [0, 0, 2],
            ],
            dtype=np.float64,
        )

        result = apply_elongation(vertices, {"c": 1.5})
        assert np.allclose(result[:, 2], vertices[:, 2] * 1.5)


class TestApplyTaper:
    """Tests for tapering modification."""

    def test_taper_positive_direction(self):
        """Taper toward +c should narrow at top."""
        vertices = np.array(
            [
                [1, 0, 0],  # Bottom, full width
                [-1, 0, 0],  # Bottom, full width
                [1, 0, 1],  # Top, should be tapered
                [-1, 0, 1],  # Top, should be tapered
            ],
            dtype=np.float64,
        )

        result = apply_taper(vertices, {"direction": "+c", "factor": 0.5})

        # Bottom vertices should be unchanged (at z=0)
        assert np.allclose(result[0], [1, 0, 0])
        assert np.allclose(result[1], [-1, 0, 0])

        # Top vertices should be scaled by factor 0.5 in x, y
        assert np.isclose(result[2, 0], 0.5)
        assert np.isclose(result[3, 0], -0.5)

    def test_taper_negative_direction(self):
        """Taper toward -c should narrow at bottom."""
        vertices = np.array(
            [
                [1, 0, 0],  # Bottom, should be tapered
                [-1, 0, 0],  # Bottom, should be tapered
                [1, 0, 1],  # Top, full width
                [-1, 0, 1],  # Top, full width
            ],
            dtype=np.float64,
        )

        result = apply_taper(vertices, {"direction": "-c", "factor": 0.5})

        # Top vertices should be unchanged
        assert np.isclose(result[2, 0], 1.0)
        assert np.isclose(result[3, 0], -1.0)

        # Bottom vertices should be tapered
        assert np.isclose(result[0, 0], 0.5)
        assert np.isclose(result[1, 0], -0.5)


class TestApplyFlatten:
    """Tests for flattening modification."""

    def test_flatten_halves_z(self):
        """Flatten with ratio 0.5 should halve z coordinates."""
        vertices = np.array(
            [
                [1, 1, 2],
                [0, 0, 4],
            ],
            dtype=np.float64,
        )

        result = apply_flatten(vertices, {"axis": "c", "ratio": 0.5})

        # X and Y should be unchanged
        assert np.allclose(result[:, 0], vertices[:, 0])
        assert np.allclose(result[:, 1], vertices[:, 1])

        # Z should be halved
        assert np.allclose(result[:, 2], vertices[:, 2] * 0.5)


class TestApplyTwist:
    """Tests for twist modification."""

    def test_twist_zero_angle(self):
        """Twist with 0 angle should not change vertices."""
        vertices = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
            ],
            dtype=np.float64,
        )

        result = apply_twist(vertices, {"axis": "c", "angle": 0.0})
        assert np.allclose(result, vertices)

    def test_twist_preserves_z(self):
        """Twist around c-axis should preserve z coordinates."""
        vertices = np.array(
            [
                [1, 0, 0],
                [1, 0, 0.5],
                [1, 0, 1],
            ],
            dtype=np.float64,
        )

        result = apply_twist(vertices, {"axis": "c", "angle": 90.0})

        # Z coordinates should be preserved
        assert np.allclose(result[:, 2], vertices[:, 2])


class TestApplyModifications:
    """Tests for the combined apply_modifications function."""

    def test_apply_multiple_modifications(self):
        """Should apply modifications in order."""
        vertices = np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [0, 0, 2],
            ],
            dtype=np.float64,
        )

        # Mock modification objects
        class Mod:
            def __init__(self, name, params):
                self.name = name
                self.params = params

        mods = [
            Mod("elongate", {"axis": "c", "ratio": 2.0}),
            Mod("taper", {"direction": "+c", "factor": 0.5}),
        ]

        result = apply_modifications(vertices, mods)

        # First elongation doubles z, then taper narrows toward +z
        # Z coordinates should be doubled
        assert result[2, 2] == 4.0  # 2 * 2 = 4

    def test_apply_empty_modifications(self):
        """Empty modification list should return copy of vertices."""
        vertices = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            dtype=np.float64,
        )

        result = apply_modifications(vertices, [])

        assert np.allclose(result, vertices)
        # Should be a copy, not same array
        assert result is not vertices

    def test_apply_unknown_modification(self):
        """Unknown modification types should be skipped."""
        vertices = np.array([[1, 2, 3]], dtype=np.float64)

        class Mod:
            name = "unknown_mod"
            params = {}

        result = apply_modifications(vertices, [Mod()])
        assert np.allclose(result, vertices)
