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

# Data classes
from .models import DEFAULT_LATTICE, CrystalGeometry, LatticeParams

# Symmetry operations
from .symmetry import (
    generate_equivalent_faces,
    get_lattice_for_system,
    get_point_group_operations,
    miller_to_normal,
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
    # Symmetry
    "generate_equivalent_faces",
    "get_point_group_operations",
    "miller_to_normal",
    "get_lattice_for_system",
]
