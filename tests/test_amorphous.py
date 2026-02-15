"""Tests for amorphous geometry generation."""

import numpy as np
import pytest

from crystal_geometry.amorphous import (
    generate_amorphous_shape,
    generate_botryoidal,
    generate_conchoidal,
    generate_mammillary,
    generate_massive,
    generate_nodular,
    generate_reniform,
    generate_stalactitic,
)


class TestAmorphousShapeGenerators:
    """Test each shape generator produces valid geometry."""

    def _validate_geometry(self, geom):
        """Common validation for amorphous geometry."""
        assert geom.is_amorphous is True
        assert len(geom.vertices) > 0
        assert len(geom.faces) > 0
        assert len(geom.face_normals) == len(geom.faces)
        assert len(geom.face_forms) == len(geom.faces)
        assert len(geom.face_millers) == len(geom.faces)
        assert geom.forms == []
        # All face_millers should be (0,0,0) for amorphous
        assert all(m == (0, 0, 0) for m in geom.face_millers)
        # Vertices should be finite
        assert np.all(np.isfinite(geom.vertices))

    def test_massive(self):
        geom = generate_massive()
        self._validate_geometry(geom)
        # Massive should be roughly spherical
        distances = np.linalg.norm(geom.vertices, axis=1)
        assert np.std(distances) < 0.5  # Not too irregular

    def test_botryoidal(self):
        geom = generate_botryoidal()
        self._validate_geometry(geom)
        # Should have multiple sphere groups → more vertices than a single sphere
        assert len(geom.vertices) > 100

    def test_reniform(self):
        geom = generate_reniform()
        self._validate_geometry(geom)
        # Reniform should be elongated in x
        x_range = np.ptp(geom.vertices[:, 0])
        z_range = np.ptp(geom.vertices[:, 2])
        assert x_range > z_range

    def test_stalactitic(self):
        geom = generate_stalactitic()
        self._validate_geometry(geom)
        # Stalactitic should be elongated along z
        z_range = np.ptp(geom.vertices[:, 2])
        x_range = np.ptp(geom.vertices[:, 0])
        assert z_range > x_range

    def test_mammillary(self):
        geom = generate_mammillary()
        self._validate_geometry(geom)
        # Mammillary should mostly be above z=0
        assert np.mean(geom.vertices[:, 2]) >= -0.3

    def test_nodular(self):
        geom = generate_nodular()
        self._validate_geometry(geom)
        # Nodular should be lumpy but roughly spherical
        distances = np.linalg.norm(geom.vertices, axis=1)
        assert np.std(distances) > 0.01  # Has some lumpiness

    def test_conchoidal(self):
        geom = generate_conchoidal()
        self._validate_geometry(geom)
        # Conchoidal is a clean sphere — should be regular
        distances = np.linalg.norm(geom.vertices, axis=1)
        assert np.std(distances) < 0.01  # Very regular


class TestAmorphousShapeRegistry:
    """Test the registry/dispatcher function."""

    @pytest.mark.parametrize(
        "shape",
        ["massive", "botryoidal", "reniform", "stalactitic", "mammillary", "nodular", "conchoidal"],
    )
    def test_generate_amorphous_shape(self, shape):
        geom = generate_amorphous_shape(shape)
        assert geom.is_amorphous is True
        assert len(geom.vertices) > 0
        assert len(geom.faces) > 0

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown amorphous shape"):
            generate_amorphous_shape("nonexistent")

    def test_custom_radius(self):
        geom = generate_amorphous_shape("conchoidal", radius=2.0)
        distances = np.linalg.norm(geom.vertices, axis=1)
        assert np.mean(distances) > 1.5  # Larger than default

    def test_deterministic_seed(self):
        g1 = generate_amorphous_shape("massive", seed=123)
        g2 = generate_amorphous_shape("massive", seed=123)
        np.testing.assert_array_equal(g1.vertices, g2.vertices)

    def test_different_seeds_differ(self):
        g1 = generate_amorphous_shape("massive", seed=1)
        g2 = generate_amorphous_shape("massive", seed=2)
        assert not np.allclose(g1.vertices, g2.vertices)
