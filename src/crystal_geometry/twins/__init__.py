"""
Crystal Twinning Module.

Provides transformation matrices and geometry generators for common twin laws.
Supports 14 twin laws with 5 different rendering strategies.
"""

from .generators import (
    GEOMETRY_GENERATORS,
    CrystalComponent,
    CyclicGeometryGenerator,
    DualCrystalGeometryGenerator,
    SingleCrystalGeometryGenerator,
    TwinGeometry,
    TwinGeometryGenerator,
    UnifiedGeometryGenerator,
    VShapedGeometryGenerator,
    get_generator,
    list_generators,
    register_generator,
)
from .laws import (
    GEMSTONE_TWINS,
    TWIN_LAWS,
    TwinLaw,
    get_gemstone_twins,
    get_twin_law,
    list_twin_laws,
)
from .transforms import (
    DIRECTIONS,
    reflection_matrix,
    rotation_matrix_4x4,
    rotation_matrix_axis_angle,
    translation_matrix_4x4,
)

__all__ = [
    # Transform functions
    "rotation_matrix_axis_angle",
    "rotation_matrix_4x4",
    "translation_matrix_4x4",
    "reflection_matrix",
    "DIRECTIONS",
    # Twin law data
    "TwinLaw",
    "TWIN_LAWS",
    "GEMSTONE_TWINS",
    "get_twin_law",
    "list_twin_laws",
    "get_gemstone_twins",
    # Geometry generators
    "TwinGeometryGenerator",
    "UnifiedGeometryGenerator",
    "DualCrystalGeometryGenerator",
    "VShapedGeometryGenerator",
    "CyclicGeometryGenerator",
    "SingleCrystalGeometryGenerator",
    "GEOMETRY_GENERATORS",
    "get_generator",
    "list_generators",
    "register_generator",
    # Data classes
    "CrystalComponent",
    "TwinGeometry",
]
