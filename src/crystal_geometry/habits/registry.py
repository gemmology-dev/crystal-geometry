"""
Crystal habit registry.

Provides a central registry of all habit classes and lookup functions.
"""

from typing import Any

from .base import CrystalHabit
from .cubic import Cube, Dodecahedron, Octahedron
from .hexagonal import HexagonalBipyramid, HexagonalPrism
from .orthorhombic import OrthorhombicPrism
from .special import Barrel, Pyritohedron, QuartzCrystal, Tabular, Trapezohedron
from .tetragonal import TetragonalBipyramid, TetragonalPrism

# Registry of available habits
HABIT_REGISTRY: dict[str, type[CrystalHabit]] = {
    # Cubic system
    "octahedron": Octahedron,
    "cube": Cube,
    "dodecahedron": Dodecahedron,
    "rhombic_dodecahedron": Dodecahedron,
    # Hexagonal system
    "hexagonal_prism": HexagonalPrism,
    "hexagonal_bipyramid": HexagonalBipyramid,
    # Tetragonal system
    "tetragonal_prism": TetragonalPrism,
    "tetragonal_bipyramid": TetragonalBipyramid,
    # Orthorhombic system
    "orthorhombic_prism": OrthorhombicPrism,
    # Special habits
    "barrel": Barrel,
    "tabular": Tabular,
    "trapezohedron": Trapezohedron,
    "quartz_crystal": QuartzCrystal,
    "pyritohedron": Pyritohedron,
}


# Gemstone to typical habits mapping
GEMSTONE_HABITS: dict[str, list[str]] = {
    "diamond": ["octahedron", "dodecahedron", "cube"],
    "ruby": ["barrel", "tabular", "hexagonal_prism"],
    "sapphire": ["barrel", "hexagonal_bipyramid", "tabular"],
    "corundum": ["barrel", "hexagonal_prism", "tabular"],
    "emerald": ["hexagonal_prism"],
    "aquamarine": ["hexagonal_prism"],
    "beryl": ["hexagonal_prism"],
    "quartz": ["quartz_crystal", "hexagonal_prism", "hexagonal_bipyramid"],
    "garnet": ["dodecahedron", "trapezohedron"],
    "spinel": ["octahedron"],
    "topaz": ["orthorhombic_prism"],
    "peridot": ["orthorhombic_prism", "tabular"],
    "chrysoberyl": ["tabular", "orthorhombic_prism"],
    "zircon": ["tetragonal_prism", "tetragonal_bipyramid"],
    "fluorite": ["cube", "octahedron"],
    "kunzite": ["orthorhombic_prism", "tabular"],
    "spodumene": ["orthorhombic_prism"],
    "tourmaline": ["hexagonal_prism"],
    "apatite": ["hexagonal_prism", "hexagonal_bipyramid"],
    "tanzanite": ["orthorhombic_prism"],
    "pyrite": ["pyritohedron", "cube"],
    "magnetite": ["octahedron"],
}


def get_habit(name: str, scale: float = 1.0, **params: Any) -> CrystalHabit:
    """Get a crystal habit instance by name.

    Args:
        name: Habit name (case-insensitive, spaces/hyphens converted to underscores)
        scale: Size multiplier for the habit
        **params: Additional parameters passed to the habit constructor

    Returns:
        Initialized CrystalHabit instance

    Raises:
        ValueError: If habit name is not recognized
    """
    # Normalize name
    normalized = name.lower().replace(" ", "_").replace("-", "_")

    if normalized not in HABIT_REGISTRY:
        available = ", ".join(sorted(HABIT_REGISTRY.keys()))
        raise ValueError(f"Unknown habit: '{name}'. Available: {available}")

    return HABIT_REGISTRY[normalized](scale=scale, **params)


def list_habits() -> list[str]:
    """List all available habit names.

    Returns:
        Sorted list of habit names
    """
    return sorted(HABIT_REGISTRY.keys())


def get_gemstone_habits(gemstone: str) -> list[str]:
    """Get list of typical habits for a gemstone.

    Args:
        gemstone: Gemstone name (case-insensitive)

    Returns:
        List of habit names, or ['cube'] if gemstone not found
    """
    return GEMSTONE_HABITS.get(gemstone.lower(), ["cube"])
