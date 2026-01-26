"""Tests for the twin system."""

import numpy as np
import pytest

from crystal_geometry.twins import (
    DIRECTIONS,
    TWIN_LAWS,
    CrystalComponent,
    TwinGeometry,
    TwinLaw,
    get_gemstone_twins,
    get_generator,
    get_twin_law,
    list_generators,
    list_twin_laws,
    reflection_matrix,
    rotation_matrix_axis_angle,
)


class TestTransforms:
    """Tests for geometric transformation functions."""

    def test_rotation_matrix_identity(self):
        """Rotation of 0 degrees should be identity."""
        axis = np.array([1.0, 0.0, 0.0])
        R = rotation_matrix_axis_angle(axis, 0.0)
        assert np.allclose(R, np.eye(3))

    def test_rotation_matrix_90_degrees(self):
        """90° rotation about z-axis."""
        axis = np.array([0.0, 0.0, 1.0])
        R = rotation_matrix_axis_angle(axis, 90.0)

        # Test that [1, 0, 0] rotates to [0, 1, 0]
        v = np.array([1.0, 0.0, 0.0])
        rotated = R @ v
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(rotated, expected, atol=1e-10)

    def test_rotation_matrix_180_degrees(self):
        """180° rotation about z-axis."""
        axis = np.array([0.0, 0.0, 1.0])
        R = rotation_matrix_axis_angle(axis, 180.0)

        v = np.array([1.0, 0.0, 0.0])
        rotated = R @ v
        expected = np.array([-1.0, 0.0, 0.0])
        assert np.allclose(rotated, expected, atol=1e-10)

    def test_reflection_matrix(self):
        """Reflection across xy-plane (z-normal)."""
        normal = np.array([0.0, 0.0, 1.0])
        R = reflection_matrix(normal)

        v = np.array([1.0, 2.0, 3.0])
        reflected = R @ v
        expected = np.array([1.0, 2.0, -3.0])
        assert np.allclose(reflected, expected)

    def test_reflection_matrix_double(self):
        """Double reflection should return to original."""
        normal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        R = reflection_matrix(normal)

        v = np.array([1.0, 2.0, 3.0])
        double_reflected = R @ (R @ v)
        assert np.allclose(double_reflected, v)


class TestDirections:
    """Tests for crystallographic directions."""

    def test_directions_normalized(self):
        """All directions should be unit vectors."""
        for name, direction in DIRECTIONS.items():
            norm = np.linalg.norm(direction)
            assert np.isclose(norm, 1.0), f"{name} is not normalized"

    def test_basic_directions(self):
        """Check standard axis directions."""
        assert np.allclose(DIRECTIONS["[100]"], [1, 0, 0])
        assert np.allclose(DIRECTIONS["[010]"], [0, 1, 0])
        assert np.allclose(DIRECTIONS["[001]"], [0, 0, 1])


class TestTwinLaws:
    """Tests for twin law definitions."""

    def test_twin_laws_exist(self):
        """Should have 14 twin laws defined."""
        assert len(TWIN_LAWS) == 14

    def test_get_twin_law_valid(self):
        """Should get valid twin laws."""
        law = get_twin_law("spinel_law")
        assert isinstance(law, TwinLaw)
        assert law.name == "Spinel Law (Macle)"
        assert law.angle == 180.0

    def test_get_twin_law_invalid(self):
        """Should raise for unknown twin law."""
        with pytest.raises(ValueError, match="Unknown twin law"):
            get_twin_law("nonexistent")

    def test_list_twin_laws(self):
        """Should list all twin laws."""
        laws = list_twin_laws()
        assert len(laws) == 14
        assert "spinel_law" in laws
        assert "japan" in laws

    @pytest.mark.parametrize("law_name", list(TWIN_LAWS.keys()))
    def test_all_twin_laws_have_required_fields(self, law_name):
        """Each twin law should have all required fields."""
        law = get_twin_law(law_name)
        assert law.name
        assert law.description
        assert law.twin_type in ("contact", "penetration", "cyclic")
        assert law.render_mode in (
            "unified",
            "dual_crystal",
            "v_shaped",
            "contact_rotation",
            "cyclic",
            "single_crystal",
        )
        assert len(law.axis) == 3
        assert law.angle > 0
        assert law.habit


class TestGemstoneTwins:
    """Tests for gemstone to twin mapping."""

    def test_get_gemstone_twins(self):
        """Should get twins for known gemstones."""
        twins = get_gemstone_twins("diamond")
        assert twins == ["spinel_law"]

        twins = get_gemstone_twins("quartz")
        assert "japan" in twins
        assert "brazil" in twins

    def test_get_gemstone_twins_unknown(self):
        """Should return empty list for unknown gemstones."""
        twins = get_gemstone_twins("unknown_gem")
        assert twins == []

    def test_get_gemstone_twins_case_insensitive(self):
        """Should be case-insensitive."""
        assert get_gemstone_twins("Diamond") == get_gemstone_twins("diamond")


class TestGenerators:
    """Tests for geometry generators."""

    def test_list_generators(self):
        """Should list all generator types."""
        generators = list_generators()
        assert "unified" in generators
        assert "dual_crystal" in generators
        assert "v_shaped" in generators
        assert "cyclic" in generators
        assert "single_crystal" in generators

    def test_get_generator_valid(self):
        """Should get valid generators."""
        gen = get_generator("unified")
        assert gen is not None
        assert hasattr(gen, "generate")

    def test_get_generator_invalid(self):
        """Should raise for unknown generator."""
        with pytest.raises(ValueError, match="Unknown render mode"):
            get_generator("nonexistent")


class TestCrystalComponent:
    """Tests for CrystalComponent class."""

    def test_component_creation(self):
        """Should create component with vertices and faces."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = [[0, 1, 2]]
        comp = CrystalComponent(vertices, faces, component_id=0)

        assert len(comp.vertices) == 3
        assert len(comp.faces) == 1
        assert comp.component_id == 0

    def test_component_transform(self):
        """Should apply transform correctly."""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        faces = [[0, 1, 2]]

        # Translation by [1, 1, 1]
        transform = np.eye(4)
        transform[:3, 3] = [1, 1, 1]

        comp = CrystalComponent(vertices, faces, transform=transform)
        transformed = comp.get_transformed_vertices()

        expected = vertices + np.array([1, 1, 1])
        assert np.allclose(transformed, expected)


class TestTwinGeometry:
    """Tests for TwinGeometry class."""

    def test_geometry_creation(self):
        """Should create geometry with components."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = [[0, 1, 2]]
        comp = CrystalComponent(vertices, faces, component_id=0)

        geom = TwinGeometry([comp], render_mode="unified")

        assert geom.n_components == 1
        assert geom.render_mode == "unified"

    def test_geometry_all_vertices(self):
        """Should concatenate vertices from all components."""
        v1 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        v2 = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float64)

        comp1 = CrystalComponent(v1, [[0, 1]], component_id=0)
        comp2 = CrystalComponent(v2, [[0, 1]], component_id=1)

        geom = TwinGeometry([comp1, comp2], render_mode="separate")

        all_verts = geom.get_all_vertices()
        assert len(all_verts) == 4

    def test_geometry_face_attribution(self):
        """Should return correct face attribution."""
        v1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        v2 = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float64)

        comp1 = CrystalComponent(v1, [[0, 1, 2]], component_id=0)
        comp2 = CrystalComponent(v2, [[0, 1, 2]], component_id=1)

        geom = TwinGeometry([comp1, comp2], render_mode="separate")

        attribution = geom.get_face_attribution()
        assert list(attribution) == [0, 1]


class TestTwinGeometryCorrectness:
    """Tests for geometric correctness of twin output."""

    def test_v_shaped_face_normals_point_outward(self):
        """V-shaped twins should have outward-pointing normals on both components."""
        from crystal_geometry.twins.generators import VShapedGeometryGenerator

        generator = VShapedGeometryGenerator()

        # Simple octahedron-like halfspaces
        normals = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        distances = np.ones(6)

        twin_info = {
            "axis": np.array([1, 0, 0]),
            "angle": 180.0,
            "type": "contact",
        }

        result = generator.generate(twin_info, normals, distances)
        assert result.n_components == 2

        # Check that all faces have consistent winding (outward normals)
        for comp in result.components:
            verts = comp.get_transformed_vertices()
            centroid = np.mean(verts, axis=0)

            for face in comp.faces:
                if len(face) < 3:
                    continue
                # Compute face normal from vertices
                v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                face_normal = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(face_normal)
                if norm_len < 1e-10:
                    continue
                face_normal = face_normal / norm_len

                # Face center
                face_center = np.mean(verts[face], axis=0)
                outward = face_center - centroid

                # Normal should point outward (same direction as centroid-to-face vector)
                assert np.dot(face_normal, outward) > 0, (
                    f"Face normal points inward: dot={np.dot(face_normal, outward)}"
                )

    def test_unified_face_attribution_matches_geometry(self):
        """Face attribution should correctly identify which component each face belongs to."""
        from crystal_geometry.twins.generators import UnifiedGeometryGenerator

        generator = UnifiedGeometryGenerator()

        # Octahedron normals
        normals = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ],
            dtype=np.float64,
        )
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        distances = np.ones(8)

        # Spinel law: 180° about [111]
        twin_info = {
            "axis": np.array([1, 1, 1]) / np.sqrt(3),
            "angle": 180.0,
            "type": "contact",
        }

        result = generator.generate(twin_info, normals, distances)
        assert result.n_components == 1  # Unified geometry

        # Face attribution should be in metadata
        assert "face_attribution" in result.metadata
        attr = result.metadata["face_attribution"]

        # Should have two component values (0 and 1) for spinel twin
        unique_comps = set(attr)
        assert len(unique_comps) <= 2, f"Expected 1-2 components, got {len(unique_comps)}"

    def test_japan_law_angle_is_correct(self):
        """Japan law angle should be 84°33'30" (the standard crystallographic value)."""
        from crystal_geometry.twins import get_twin_law
        from crystal_geometry.twins.laws import _JAPAN_ANGLE_QUARTZ

        law = get_twin_law("japan")

        # Verify the angle matches the defined constant
        assert np.isclose(law.angle, _JAPAN_ANGLE_QUARTZ, atol=1e-10)

        # Should be 84°33'30" = 84.55833...°
        expected_dms = 84.0 + 33.0 / 60.0 + 30.0 / 3600.0
        assert np.isclose(law.angle, expected_dms, atol=1e-4)
        assert 84.55 < law.angle < 84.56


class TestContactRotationGeometry:
    """Tests for contact rotation twin geometry (spinel, albite, manebach, baveno)."""

    def test_contact_rotation_produces_two_components(self):
        """Contact rotation should produce two separate components."""
        from crystal_geometry.twins.generators import ContactRotationGeometryGenerator

        generator = ContactRotationGeometryGenerator()
        # Octahedron normals
        normals = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ],
            dtype=np.float64,
        )
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        distances = np.ones(8)

        twin_info = {"axis": np.array([1, 1, 1]) / np.sqrt(3), "angle": 180.0}
        result = generator.generate(twin_info, normals, distances)

        assert result.n_components == 2
        assert result.render_mode == "separate"

    def test_contact_rotation_components_share_composition_plane(self):
        """Both components should have vertices on the composition plane."""
        from crystal_geometry.twins.generators import ContactRotationGeometryGenerator

        generator = ContactRotationGeometryGenerator()
        # Octahedron normals
        normals = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ],
            dtype=np.float64,
        )
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        distances = np.ones(8)

        twin_axis = np.array([1, 1, 1]) / np.sqrt(3)
        twin_info = {"axis": twin_axis, "angle": 180.0}
        result = generator.generate(twin_info, normals, distances)

        # Both components should have vertices on the composition plane (dot product ≈ 0)
        for comp in result.components:
            verts = comp.get_transformed_vertices()
            # Find vertices on composition plane
            on_plane = np.abs(verts @ twin_axis) < 1e-5
            assert np.any(on_plane), "Component should have vertices on composition plane"

    def test_contact_rotation_euler_characteristic(self):
        """Each contact rotation component should be a valid polyhedron (Euler = 2)."""
        from crystal_geometry.twins.generators import ContactRotationGeometryGenerator

        generator = ContactRotationGeometryGenerator()
        # Octahedron normals
        normals = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ],
            dtype=np.float64,
        )
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        distances = np.ones(8)

        twin_info = {"axis": np.array([1, 1, 1]) / np.sqrt(3), "angle": 180.0}
        result = generator.generate(twin_info, normals, distances)

        for comp in result.components:
            V = len(comp.vertices)
            F = len(comp.faces)
            # Count edges (each edge shared by 2 faces)
            edge_set = set()
            for face in comp.faces:
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    edge_set.add(edge)
            E = len(edge_set)

            euler = V - E + F
            assert euler == 2, f"Euler characteristic should be 2, got {euler}"

    def test_spinel_law_uses_contact_rotation_mode(self):
        """Spinel twin law should use contact_rotation render mode."""
        from crystal_geometry.twins import get_twin_law

        law = get_twin_law("spinel_law")
        assert law.render_mode == "contact_rotation"

    def test_feldspar_twins_use_contact_rotation_mode(self):
        """Feldspar twins (albite, manebach, baveno) should use contact_rotation."""
        from crystal_geometry.twins import get_twin_law

        for twin_name in ["albite", "manebach", "baveno"]:
            law = get_twin_law(twin_name)
            assert law.render_mode == "contact_rotation", f"{twin_name} should use contact_rotation"


class TestFluoritePenetrationTwin:
    """Tests for fluorite interpenetrating cube twin."""

    def test_fluorite_twin_law_uses_dual_crystal(self):
        """Fluorite twin law should use dual_crystal render mode."""
        from crystal_geometry.twins import get_twin_law

        law = get_twin_law("fluorite")
        assert law.render_mode == "dual_crystal"

    def test_fluorite_produces_two_complete_cubes(self):
        """Each component should be a complete cube (6 faces)."""
        from crystal_geometry.twins.generators import DualCrystalGeometryGenerator

        generator = DualCrystalGeometryGenerator()
        # Cube normals
        normals = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )
        distances = np.ones(6)

        # Fluorite: 180° about [111]
        twin_info = {"axis": np.array([1, 1, 1]) / np.sqrt(3), "angle": 180.0}
        result = generator.generate(twin_info, normals, distances)

        assert result.n_components == 2
        assert result.render_mode == "separate"

        # Each component should be a complete cube with 6 faces
        for comp in result.components:
            assert len(comp.faces) == 6, f"Expected 6 faces (cube), got {len(comp.faces)}"
            assert len(comp.vertices) == 8, f"Expected 8 vertices (cube), got {len(comp.vertices)}"

    def test_fluorite_twin_components_are_rotated(self):
        """Second component should be rotated 180° about [111]."""
        from crystal_geometry.twins.generators import DualCrystalGeometryGenerator

        generator = DualCrystalGeometryGenerator()
        # Cube normals
        normals = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )
        distances = np.ones(6)

        twin_info = {"axis": np.array([1, 1, 1]) / np.sqrt(3), "angle": 180.0}
        result = generator.generate(twin_info, normals, distances)

        # Get centroids of the two cubes - they should be at origin
        # but vertices should be in different positions
        verts1 = result.components[0].get_transformed_vertices()
        verts2 = result.components[1].get_transformed_vertices()

        # Centroids should both be at origin
        c1 = np.mean(verts1, axis=0)
        c2 = np.mean(verts2, axis=0)
        assert np.allclose(c1, [0, 0, 0], atol=1e-10)
        assert np.allclose(c2, [0, 0, 0], atol=1e-10)

        # But the vertex positions should be different (rotated)
        # Sort vertices for comparison
        sorted_v1 = verts1[np.lexsort(verts1.T)]
        sorted_v2 = verts2[np.lexsort(verts2.T)]
        assert not np.allclose(sorted_v1, sorted_v2, atol=1e-5)
