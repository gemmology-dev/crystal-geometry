"""
Crystal Habit System.

Provides geometry definitions for common crystal habits (morphological forms).
Supports 14 habit classes with customizable parameters.
"""

from .base import CrystalHabit
from .cubic import Cube, Dodecahedron, Octahedron
from .hexagonal import HexagonalBipyramid, HexagonalPrism
from .orthorhombic import OrthorhombicPrism
from .registry import (
    GEMSTONE_HABITS,
    HABIT_REGISTRY,
    get_gemstone_habits,
    get_habit,
    list_habits,
)
from .special import Barrel, Pyritohedron, QuartzCrystal, Tabular, Trapezohedron
from .tetragonal import TetragonalBipyramid, TetragonalPrism

__all__ = [
    # Base class
    "CrystalHabit",
    # Cubic habits
    "Octahedron",
    "Cube",
    "Dodecahedron",
    # Hexagonal habits
    "HexagonalPrism",
    "HexagonalBipyramid",
    # Tetragonal habits
    "TetragonalPrism",
    "TetragonalBipyramid",
    # Orthorhombic habits
    "OrthorhombicPrism",
    # Special habits
    "Barrel",
    "Tabular",
    "Trapezohedron",
    "QuartzCrystal",
    "Pyritohedron",
    # Registry functions
    "HABIT_REGISTRY",
    "GEMSTONE_HABITS",
    "get_habit",
    "list_habits",
    "get_gemstone_habits",
]
