"""
Test suite for crystal-geometry.

Tests geometry generation, symmetry operations, and model classes.
"""

import numpy as np
import pytest

from cdl_parser import parse_cdl
from crystal_geometry import (
    CrystalGeometry,
    LatticeParams,
    cdl_string_to_geometry,
    cdl_to_geometry,
    create_cube,
    create_dodecahedron,
    create_octahedron,
    create_truncated_octahedron,
    generate_equivalent_faces,
    get_point_group_operations,
    miller_to_normal,
)

# =============================================================================
# LatticeParams Tests
# =============================================================================


class TestLatticeParams:
    """Test LatticeParams dataclass."""

    def test_cubic_lattice(self):
        """Test cubic lattice creation."""
        lattice = LatticeParams.cubic()
        assert lattice.a == lattice.b == lattice.c == 1.0
        assert lattice.alpha == lattice.beta == lattice.gamma == np.pi / 2

    def test_hexagonal_lattice(self):
        """Test hexagonal lattice creation."""
        lattice = LatticeParams.hexagonal(c_ratio=1.5)
        assert lattice.a == lattice.b == 1.0
        assert lattice.c == 1.5
        assert lattice.gamma == pytest.approx(2 * np.pi / 3)

    def test_tetragonal_lattice(self):
        """Test tetragonal lattice creation."""
        lattice = LatticeParams.tetragonal(c_ratio=2.0)
        assert lattice.a == lattice.b == 1.0
        assert lattice.c == 2.0


# =============================================================================
# Symmetry Tests
# =============================================================================


class TestSymmetry:
    """Test symmetry operations."""

    def test_cubic_point_group_m3m(self):
        """Test m3m point group has 48 operations."""
        ops = get_point_group_operations("m3m")
        assert len(ops) == 48

    def test_cubic_point_group_432(self):
        """Test 432 point group has 24 operations."""
        ops = get_point_group_operations("432")
        assert len(ops) == 24

    def test_triclinic_point_group_1(self):
        """Test point group 1 has 1 operation (identity)."""
        ops = get_point_group_operations("1")
        assert len(ops) == 1

    def test_triclinic_point_group_minus1(self):
        """Test point group -1 has 2 operations."""
        ops = get_point_group_operations("-1")
        assert len(ops) == 2

    def test_equivalent_faces_octahedron(self):
        """Test {111} in m3m generates 8 faces."""
        faces = generate_equivalent_faces(1, 1, 1, "m3m")
        assert len(faces) == 8
        # All combinations of (+1, +1, +1)
        for h, k, l in faces:
            assert abs(h) == abs(k) == abs(l) == 1

    def test_equivalent_faces_cube(self):
        """Test {100} in m3m generates 6 faces."""
        faces = generate_equivalent_faces(1, 0, 0, "m3m")
        assert len(faces) == 6

    def test_equivalent_faces_dodecahedron(self):
        """Test {110} in m3m generates 12 faces."""
        faces = generate_equivalent_faces(1, 1, 0, "m3m")
        assert len(faces) == 12

    def test_miller_to_normal_cubic(self):
        """Test Miller to normal conversion for cubic."""
        normal = miller_to_normal(1, 1, 1)
        expected = np.array([1, 1, 1]) / np.sqrt(3)
        assert np.allclose(normal, expected, atol=1e-10)

    def test_miller_to_normal_100(self):
        """Test {100} normal."""
        normal = miller_to_normal(1, 0, 0)
        expected = np.array([1, 0, 0])
        assert np.allclose(normal, expected, atol=1e-10)


# =============================================================================
# Geometry Generation Tests
# =============================================================================


class TestGeometryGeneration:
    """Test geometry generation functions."""

    def test_create_octahedron(self):
        """Test creating octahedron."""
        geom = create_octahedron()
        assert isinstance(geom, CrystalGeometry)
        assert len(geom.vertices) == 6
        assert len(geom.faces) == 8

    def test_create_cube(self):
        """Test creating cube."""
        geom = create_cube()
        assert isinstance(geom, CrystalGeometry)
        assert len(geom.vertices) == 8
        assert len(geom.faces) == 6

    def test_create_dodecahedron(self):
        """Test creating dodecahedron."""
        geom = create_dodecahedron()
        assert isinstance(geom, CrystalGeometry)
        assert len(geom.faces) == 12

    def test_create_truncated_octahedron(self):
        """Test creating truncated octahedron."""
        geom = create_truncated_octahedron()
        assert isinstance(geom, CrystalGeometry)
        # Should have 8 + 6 = 14 faces
        assert len(geom.faces) == 14

    def test_cdl_string_to_geometry(self):
        """Test direct CDL string to geometry."""
        geom = cdl_string_to_geometry("cubic[m3m]:{111}")
        assert isinstance(geom, CrystalGeometry)
        assert len(geom.faces) == 8

    def test_cdl_to_geometry(self):
        """Test CDL description to geometry."""
        desc = parse_cdl("cubic[m3m]:{111}@1.0 + {100}@1.3")
        geom = cdl_to_geometry(desc)
        assert isinstance(geom, CrystalGeometry)
        assert len(geom.faces) == 14


# =============================================================================
# CrystalGeometry Tests
# =============================================================================


class TestCrystalGeometry:
    """Test CrystalGeometry dataclass."""

    def test_get_edges(self):
        """Test edge extraction."""
        geom = create_octahedron()
        edges = geom.get_edges()
        assert len(edges) == 12  # Octahedron has 12 edges

    def test_center(self):
        """Test center calculation."""
        geom = create_octahedron()
        center = geom.center()
        assert np.allclose(center, [0, 0, 0], atol=1e-10)

    def test_scale_to_unit(self):
        """Test scaling to unit sphere."""
        geom = create_octahedron(scale=2.0)
        scaled = geom.scale_to_unit()
        max_dist = np.max(np.linalg.norm(scaled.vertices, axis=1))
        assert max_dist == pytest.approx(1.0, abs=1e-10)

    def test_euler_characteristic(self):
        """Test Euler's formula V - E + F = 2."""
        geom = create_octahedron()
        assert geom.euler_characteristic() == 2

        geom = create_cube()
        assert geom.euler_characteristic() == 2

        geom = create_truncated_octahedron()
        assert geom.euler_characteristic() == 2

    def test_is_valid(self):
        """Test validity check."""
        geom = create_octahedron()
        assert geom.is_valid()

    def test_translate(self):
        """Test translation."""
        geom = create_octahedron()
        translated = geom.translate(np.array([1, 2, 3]))
        center = translated.center()
        assert np.allclose(center, [1, 2, 3], atol=1e-10)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        geom = create_octahedron()
        d = geom.to_dict()
        assert "vertices" in d
        assert "faces" in d
        assert "face_normals" in d
        assert len(d["vertices"]) == 6
        assert len(d["faces"]) == 8


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for real crystal forms."""

    def test_diamond_crystal(self):
        """Test diamond-like crystal (octahedron + cube truncation)."""
        # Scale 1.3 for cube ensures truncation (larger than octahedron's 1.0)
        geom = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.3")
        assert geom.is_valid()
        # Should have octahedron faces (8) + cube faces (6) = 14
        assert len(geom.faces) == 14

    def test_garnet_crystal(self):
        """Test garnet-like crystal (dodecahedron + trapezohedron)."""
        geom = cdl_string_to_geometry("cubic[m3m]:{110}@1.0 + {211}@0.6")
        assert geom.is_valid()
        # Dodecahedron (12) + trapezohedron (24) = 36 faces when combined
        # At minimum should have 24 faces (trapezohedron contributes more)
        assert len(geom.faces) >= 24

    def test_hexagonal_prism(self):
        """Test hexagonal prism."""
        geom = cdl_string_to_geometry("hexagonal[6/mmm]:{10-10}@1.0 + {0001}@0.5")
        assert geom.is_valid()
        # Prism faces (varies by symmetry) + 2 pinacoid faces
        # The {10-10} form in 6/mmm generates faces based on symmetry
        assert len(geom.faces) >= 8  # At least 6 prism + 2 pinacoid

    def test_tetragonal_dipyramid(self):
        """Test tetragonal dipyramid."""
        geom = cdl_string_to_geometry("tetragonal[4/mmm]:{101}")
        assert geom.is_valid()
        assert len(geom.faces) == 8  # 4 upper + 4 lower


# =============================================================================
# Twin Integration Tests
# =============================================================================


class TestTwinIntegration:
    """Test twin integration in cdl_to_geometry."""

    def test_spinel_twin(self):
        """Test spinel law twin (macle)."""
        from cdl_parser import CrystalDescription, CrystalForm, MillerIndex, TwinSpec

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[CrystalForm(miller=MillerIndex(1, 1, 1), scale=1.0)],
            twin=TwinSpec(law="spinel_law"),
        )

        geom = cdl_to_geometry(desc)
        # Twin geometry may not satisfy simple Euler (interpenetrating polyhedra)
        assert len(geom.vertices) >= 4
        assert len(geom.faces) >= 4
        assert geom.twin_metadata is not None
        assert geom.twin_metadata.twin_law == "Spinel Law (Macle)"
        assert geom.twin_metadata.render_mode == "contact_rotation"

    def test_brazil_twin(self):
        """Test Brazil twin (quartz) - uses cubic octahedron as proxy."""
        from cdl_parser import CrystalDescription, CrystalForm, MillerIndex, TwinSpec

        # Use cubic octahedron as a simple test case for Brazil twin
        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[CrystalForm(miller=MillerIndex(1, 1, 1), scale=1.0)],
            twin=TwinSpec(law="brazil"),
        )

        geom = cdl_to_geometry(desc)
        assert len(geom.vertices) >= 4
        assert len(geom.faces) >= 4
        assert geom.twin_metadata is not None
        assert "Brazil" in geom.twin_metadata.twin_law
        assert geom.twin_metadata.render_mode == "dual_crystal"

    def test_custom_twin(self):
        """Test custom twin axis/angle."""
        from cdl_parser import CrystalDescription, CrystalForm, MillerIndex, TwinSpec

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[CrystalForm(miller=MillerIndex(1, 1, 1), scale=1.0)],
            twin=TwinSpec(axis=(1, 1, 1), angle=180.0, twin_type="contact"),
        )

        geom = cdl_to_geometry(desc)
        assert len(geom.vertices) >= 4
        assert geom.twin_metadata is not None
        assert geom.twin_metadata.twin_law == "custom"

    def test_no_twin(self):
        """Test geometry without twin has no twin_metadata."""
        geom = cdl_string_to_geometry("cubic[m3m]:{111}")
        assert geom.is_valid()
        assert geom.twin_metadata is None


# =============================================================================
# Modification Integration Tests
# =============================================================================


class TestModificationIntegration:
    """Test modification integration in cdl_to_geometry."""

    def test_elongation(self):
        """Test elongation modifier."""
        from cdl_parser import CrystalDescription, CrystalForm, MillerIndex, Modification

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[CrystalForm(miller=MillerIndex(1, 0, 0), scale=1.0)],
            modifications=[Modification(type="elongate", params={"axis": "c", "ratio": 2.0})],
        )

        geom = cdl_to_geometry(desc)
        assert geom.is_valid()

        # Z coordinates should be doubled relative to unmodified
        z_range = geom.vertices[:, 2].max() - geom.vertices[:, 2].min()
        x_range = geom.vertices[:, 0].max() - geom.vertices[:, 0].min()
        # After elongation along c by 2x, z_range should be ~2x x_range
        assert z_range > x_range * 1.5

    def test_taper(self):
        """Test taper modifier on hexagonal prism."""
        from cdl_parser import CrystalDescription, CrystalForm, MillerIndex, Modification

        # Use hexagonal prism with correct Miller-Bravais notation: h+k+i=0
        # {10-10} -> h=1, k=0, i=-1, l=0
        desc = CrystalDescription(
            system="hexagonal",
            point_group="6/mmm",
            forms=[
                CrystalForm(miller=MillerIndex(1, 0, 0), scale=1.0),  # Use 3-index for simplicity
                CrystalForm(miller=MillerIndex(0, 0, 1), scale=1.5),
            ],
            modifications=[Modification(type="taper", params={"direction": "+c", "factor": 0.5})],
        )

        geom = cdl_to_geometry(desc)
        assert geom.is_valid()

        # Top vertices should have smaller x,y extent than bottom
        z_mid = (geom.vertices[:, 2].max() + geom.vertices[:, 2].min()) / 2
        top_verts = geom.vertices[geom.vertices[:, 2] > z_mid]
        bot_verts = geom.vertices[geom.vertices[:, 2] < z_mid]

        if len(top_verts) > 0 and len(bot_verts) > 0:
            top_extent = np.sqrt(top_verts[:, 0] ** 2 + top_verts[:, 1] ** 2).max()
            bot_extent = np.sqrt(bot_verts[:, 0] ** 2 + bot_verts[:, 1] ** 2).max()
            assert top_extent < bot_extent

    def test_multiple_modifications(self):
        """Test multiple modifications applied in sequence."""
        from cdl_parser import CrystalDescription, CrystalForm, MillerIndex, Modification

        desc = CrystalDescription(
            system="cubic",
            point_group="m3m",
            forms=[CrystalForm(miller=MillerIndex(1, 0, 0), scale=1.0)],
            modifications=[
                Modification(type="elongate", params={"axis": "c", "ratio": 2.0}),
                Modification(type="taper", params={"direction": "+c", "factor": 0.5}),
            ],
        )

        geom = cdl_to_geometry(desc)
        assert geom.is_valid()

    def test_no_modifications(self):
        """Test geometry without modifications."""
        geom = cdl_string_to_geometry("cubic[m3m]:{111}")
        assert geom.is_valid()
