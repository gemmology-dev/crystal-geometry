"""Tests for nested growth geometry generation."""

import numpy as np
import pytest

from cdl_parser.models import CrystalDescription, CrystalForm, FormGroup, MillerIndex, NestedGrowth
from crystal_geometry.geometry import cdl_to_geometry
from crystal_geometry.models import LatticeParams


class TestNestedGrowth:
    """Test nested growth geometry generation."""

    def _make_octahedron_form(self, scale=1.0):
        return CrystalForm(miller=MillerIndex(1, 1, 1), scale=scale)

    def _make_cube_form(self, scale=1.0):
        return CrystalForm(miller=MillerIndex(1, 0, 0), scale=scale)

    def test_scepter_crystal(self):
        """Test basic scepter crystal: octahedron > cube."""
        base = self._make_octahedron_form()
        over = self._make_cube_form()
        nested = NestedGrowth(base=base, overgrowth=over)

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[nested],
        )
        geom = cdl_to_geometry(desc)

        # Should have component_ids with at least 2 distinct values
        assert geom.component_ids is not None
        unique_ids = set(geom.component_ids)
        assert len(unique_ids) >= 2

        # Should have valid geometry
        assert len(geom.vertices) > 0
        assert len(geom.faces) > 0
        assert np.all(np.isfinite(geom.vertices))

    def test_component_separation(self):
        """Test that base and overgrowth have separate component IDs."""
        base = self._make_octahedron_form()
        over = self._make_cube_form()
        nested = NestedGrowth(base=base, overgrowth=over)

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[nested],
        )
        geom = cdl_to_geometry(desc)

        # Get faces per component
        comp_0_faces = [i for i, c in enumerate(geom.component_ids) if c == 0]
        comp_1_faces = [i for i, c in enumerate(geom.component_ids) if c == 1]
        assert len(comp_0_faces) > 0
        assert len(comp_1_faces) > 0

    def test_three_generation_nested(self):
        """Test 3 generations: oct > cube > oct (recursive nesting)."""
        inner_over = self._make_octahedron_form(scale=0.8)
        middle = NestedGrowth(
            base=self._make_cube_form(),
            overgrowth=inner_over,
        )
        outer = NestedGrowth(
            base=self._make_octahedron_form(),
            overgrowth=middle,
        )

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[outer],
        )
        geom = cdl_to_geometry(desc)

        # Should have component_ids with at least 3 distinct values
        assert geom.component_ids is not None
        unique_ids = set(geom.component_ids)
        assert len(unique_ids) >= 3

    def test_overgrowth_positioned_on_top(self):
        """Test that overgrowth is positioned on the termination face."""
        base = self._make_octahedron_form()
        over = self._make_cube_form()
        nested = NestedGrowth(base=base, overgrowth=over)

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[nested],
        )
        geom = cdl_to_geometry(desc)

        # The overgrowth vertices should be near the top of the base
        comp_1_faces = [i for i, c in enumerate(geom.component_ids) if c == 1]
        if comp_1_faces:
            # Get vertex indices for overgrowth faces
            over_vert_indices = set()
            for fi in comp_1_faces:
                over_vert_indices.update(geom.faces[fi])
            over_verts = geom.vertices[list(over_vert_indices)]
            over_center_z = np.mean(over_verts[:, 2])

            # Get vertex indices for base faces
            comp_0_faces = [i for i, c in enumerate(geom.component_ids) if c == 0]
            base_vert_indices = set()
            for fi in comp_0_faces:
                base_vert_indices.update(geom.faces[fi])
            base_verts = geom.vertices[list(base_vert_indices)]
            base_center_z = np.mean(base_verts[:, 2])

            # Overgrowth should be at or above base center
            assert over_center_z >= base_center_z - 0.5

    def test_nested_growth_with_formgroup(self):
        """Test nested growth where base is a FormGroup."""
        base_group = FormGroup(
            forms=[
                self._make_octahedron_form(),
                self._make_cube_form(scale=1.3),
            ]
        )
        over = self._make_octahedron_form(scale=0.5)
        nested = NestedGrowth(base=base_group, overgrowth=over)

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[nested],
        )
        geom = cdl_to_geometry(desc)

        assert geom.component_ids is not None
        assert len(set(geom.component_ids)) >= 2
        assert len(geom.vertices) > 0

    def test_nested_preserves_finite_vertices(self):
        """Ensure no NaN or Inf in nested growth output."""
        base = self._make_octahedron_form()
        over = self._make_cube_form()
        nested = NestedGrowth(base=base, overgrowth=over)

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[nested],
        )
        geom = cdl_to_geometry(desc)

        assert np.all(np.isfinite(geom.vertices))
        for normal in geom.face_normals:
            assert np.all(np.isfinite(normal))

    def test_face_data_lengths_consistent(self):
        """Check that all face-level arrays have the same length."""
        base = self._make_octahedron_form()
        over = self._make_cube_form()
        nested = NestedGrowth(base=base, overgrowth=over)

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[nested],
        )
        geom = cdl_to_geometry(desc)

        n_faces = len(geom.faces)
        assert len(geom.face_normals) == n_faces
        assert len(geom.face_forms) == n_faces
        assert len(geom.face_millers) == n_faces
        assert len(geom.component_ids) == n_faces
