"""
Crystal Geometry - 3D Crystal Geometry Engine.

Computes 3D crystal geometry from Crystal Description Language (CDL) strings.
Uses half-space intersection to combine crystal forms with point group symmetry.

Example:
    >>> from crystal_geometry import cdl_to_geometry, CrystalGeometry
    >>> from cdl_parser import parse_cdl
    >>>
    >>> desc = parse_cdl("cubic[m3m]:{111}@1.0 + {100}@1.3")
    >>> geom = cdl_to_geometry(desc)
    >>> print(len(geom.vertices), len(geom.faces))
    24 14

    >>> # Direct from string
    >>> from crystal_geometry import cdl_string_to_geometry
    >>> geom = cdl_string_to_geometry("cubic[m3m]:{111}")
"""

__version__ = "1.0.0"
__author__ = "Fabian Schuh"
__email__ = "fabian@gemmology.dev"

# Acceleration / Backend info
from ._accel import (
    get_backend,
    get_backend_info,
    get_num_threads,
    set_num_threads,
)

# Core geometry generation
from .geometry import (
    cdl_string_to_geometry,
    cdl_to_geometry,
    compute_face_vertices,
    create_cube,
    create_dodecahedron,
    create_octahedron,
    create_truncated_octahedron,
    halfspace_intersection_3d,
)

# Habit system
from .habits import (
    GEMSTONE_HABITS,
    HABIT_REGISTRY,
    Barrel,
    CrystalHabit,
    Cube,
    Dodecahedron,
    HexagonalBipyramid,
    HexagonalPrism,
    Octahedron,
    OrthorhombicPrism,
    Pyritohedron,
    QuartzCrystal,
    Tabular,
    TetragonalBipyramid,
    TetragonalPrism,
    Trapezohedron,
    get_gemstone_habits,
    get_habit,
    list_habits,
)

# Data classes
from .models import DEFAULT_LATTICE, CrystalGeometry, LatticeParams, TwinMetadata

# Modifications
from .modifications import (
    AXIS_MAP,
    apply_elongation,
    apply_flatten,
    apply_modifications,
    apply_taper,
    apply_twist,
)

# Symmetry operations
from .symmetry import (
    generate_equivalent_faces,
    get_lattice_for_system,
    get_point_group_operations,
    miller_to_normal,
)

# Twin system
from .twins import (
    DIRECTIONS,
    GEMSTONE_TWINS,
    GEOMETRY_GENERATORS,
    TWIN_LAWS,
    CrystalComponent,
    TwinGeometry,
    TwinGeometryGenerator,
    TwinLaw,
    get_gemstone_twins,
    get_generator,
    get_twin_law,
    list_generators,
    list_twin_laws,
    reflection_matrix,
    rotation_matrix_axis_angle,
)

__all__ = [
    # Version
    "__version__",
    # Core functions
    "cdl_to_geometry",
    "cdl_string_to_geometry",
    "halfspace_intersection_3d",
    "compute_face_vertices",
    # Convenience constructors
    "create_octahedron",
    "create_cube",
    "create_dodecahedron",
    "create_truncated_octahedron",
    # Data classes
    "CrystalGeometry",
    "LatticeParams",
    "DEFAULT_LATTICE",
    "TwinMetadata",
    # Symmetry
    "generate_equivalent_faces",
    "get_point_group_operations",
    "miller_to_normal",
    "get_lattice_for_system",
    # Twin system
    "TwinLaw",
    "TWIN_LAWS",
    "GEMSTONE_TWINS",
    "get_twin_law",
    "list_twin_laws",
    "get_gemstone_twins",
    "rotation_matrix_axis_angle",
    "reflection_matrix",
    "DIRECTIONS",
    "TwinGeometryGenerator",
    "TwinGeometry",
    "CrystalComponent",
    "GEOMETRY_GENERATORS",
    "get_generator",
    "list_generators",
    # Habit system
    "CrystalHabit",
    "HABIT_REGISTRY",
    "GEMSTONE_HABITS",
    "get_habit",
    "list_habits",
    "get_gemstone_habits",
    "Octahedron",
    "Cube",
    "Dodecahedron",
    "HexagonalPrism",
    "HexagonalBipyramid",
    "TetragonalPrism",
    "TetragonalBipyramid",
    "OrthorhombicPrism",
    "Barrel",
    "Tabular",
    "Trapezohedron",
    "QuartzCrystal",
    "Pyritohedron",
    # Modifications
    "apply_modifications",
    "apply_elongation",
    "apply_taper",
    "apply_flatten",
    "apply_twist",
    "AXIS_MAP",
    # Backend / Acceleration
    "get_backend",
    "get_backend_info",
    "get_num_threads",
    "set_num_threads",
]
