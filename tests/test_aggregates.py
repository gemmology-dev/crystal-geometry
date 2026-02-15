"""Tests for aggregate layout algorithms and geometry generation."""

import numpy as np
import pytest

from crystal_geometry.aggregates import (
    MAX_INSTANCES,
    generate_aggregate,
    layout_cluster,
    layout_druse,
    layout_epitaxial,
    layout_parallel,
    layout_radial,
    layout_random,
)
from crystal_geometry.geometry import cdl_string_to_geometry


def _make_base_geometry():
    """Create a simple octahedron as base geometry for testing."""
    return cdl_string_to_geometry("cubic[m3m]:{111}")


class TestLayoutAlgorithms:
    """Test each layout algorithm produces valid transforms."""

    def _validate_transforms(self, transforms, expected_count):
        """Common validation for transform lists."""
        assert len(transforms) == expected_count
        for T in transforms:
            assert T.shape == (4, 4)
            # Bottom row should be [0, 0, 0, 1]
            np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)
            # Rotation part should be orthogonal
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_parallel(self):
        transforms = layout_parallel(count=5, spacing=2.0)
        self._validate_transforms(transforms, 5)
        # All should be along z-axis (no rotation)
        for T in transforms:
            np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)
            assert T[0, 3] == 0.0 and T[1, 3] == 0.0  # Only z offset
        # Check spacing
        z_values = [T[2, 3] for T in transforms]
        diffs = np.diff(z_values)
        np.testing.assert_allclose(diffs, 2.0, atol=1e-10)

    def test_random(self):
        transforms = layout_random(count=10, bounding_radius=3.0)
        self._validate_transforms(transforms, 10)
        # All positions should be within bounding sphere
        for T in transforms:
            pos = T[:3, 3]
            assert np.linalg.norm(pos) <= 3.0 + 1e-6

    def test_radial(self):
        transforms = layout_radial(count=6, spread_radius=2.0)
        self._validate_transforms(transforms, 6)
        # All positions should be at spread_radius from origin (in xy-plane)
        for T in transforms:
            pos = T[:3, 3]
            assert abs(pos[2]) < 1e-10  # All on z=0 plane
            assert abs(np.linalg.norm(pos[:2]) - 2.0) < 1e-6

    def test_epitaxial(self):
        transforms = layout_epitaxial(count=8)
        self._validate_transforms(transforms, 8)

    def test_druse(self):
        transforms = layout_druse(count=10, hemisphere_radius=2.0)
        self._validate_transforms(transforms, 10)
        # All positions should be on the hemisphere
        for T in transforms:
            pos = T[:3, 3]
            assert abs(np.linalg.norm(pos) - 2.0) < 0.1

    def test_cluster(self):
        transforms = layout_cluster(count=8, dome_radius=2.0)
        self._validate_transforms(transforms, 8)
        # All positions should be on upper hemisphere
        for T in transforms:
            pos = T[:3, 3]
            assert pos[2] >= -0.1  # Upper hemisphere

    def test_deterministic_random_layout(self):
        t1 = layout_random(count=5, seed=42)
        t2 = layout_random(count=5, seed=42)
        for a, b in zip(t1, t2):
            np.testing.assert_array_equal(a, b)


class TestGenerateAggregate:
    """Test the full aggregate generation pipeline."""

    def test_basic_aggregate(self):
        base = _make_base_geometry()
        result = generate_aggregate(base, "parallel", count=3, spacing=2.0)

        assert result.component_ids is not None
        assert set(result.component_ids) == {0, 1, 2}
        assert len(result.faces) == len(base.faces) * 3
        assert result.aggregate_metadata is not None
        assert result.aggregate_metadata.arrangement == "parallel"
        assert result.aggregate_metadata.n_instances == 3

    def test_aggregate_metadata(self):
        base = _make_base_geometry()
        result = generate_aggregate(
            base, "random", count=5, spacing=1.5, orientation="aligned"
        )
        meta = result.aggregate_metadata
        assert meta is not None
        assert meta.arrangement == "random"
        assert meta.n_instances == 5
        assert meta.spacing == 1.5
        assert meta.orientation == "aligned"

    def test_aggregate_metadata_to_dict(self):
        base = _make_base_geometry()
        result = generate_aggregate(base, "parallel", count=2)
        d = result.aggregate_metadata.to_dict()
        assert d["arrangement"] == "parallel"
        assert d["n_instances"] == 2

    def test_count_capping(self):
        base = _make_base_geometry()
        result = generate_aggregate(base, "parallel", count=500)
        assert result.aggregate_metadata.n_instances == MAX_INSTANCES
        assert max(result.component_ids) == MAX_INSTANCES - 1

    def test_component_ids_per_face(self):
        base = _make_base_geometry()
        n_base_faces = len(base.faces)
        result = generate_aggregate(base, "radial", count=4)

        assert len(result.component_ids) == n_base_faces * 4
        # First n_base_faces should be component 0
        assert all(c == 0 for c in result.component_ids[:n_base_faces])
        # Last n_base_faces should be component 3
        assert all(c == 3 for c in result.component_ids[-n_base_faces:])

    def test_vertices_transformed(self):
        base = _make_base_geometry()
        result = generate_aggregate(base, "parallel", count=2, spacing=5.0)

        # Vertices of component 1 should be offset from component 0
        n_base_verts = len(base.vertices)
        comp0_center = np.mean(result.vertices[:n_base_verts], axis=0)
        comp1_center = np.mean(result.vertices[n_base_verts:2 * n_base_verts], axis=0)
        assert np.linalg.norm(comp1_center - comp0_center) > 1.0

    def test_unknown_arrangement_raises(self):
        base = _make_base_geometry()
        with pytest.raises(ValueError, match="Unknown arrangement"):
            generate_aggregate(base, "nonexistent", count=3)

    @pytest.mark.parametrize(
        "arrangement",
        ["parallel", "random", "radial", "epitaxial", "druse", "cluster"],
    )
    def test_all_arrangements_produce_valid_output(self, arrangement):
        base = _make_base_geometry()
        result = generate_aggregate(base, arrangement, count=4)
        assert len(result.faces) == len(base.faces) * 4
        assert result.component_ids is not None
        assert len(result.component_ids) == len(result.faces)
        assert np.all(np.isfinite(result.vertices))
