"""
Twin law definitions for crystal geometry.

Contains 14 common twin laws with their geometric parameters and rendering modes.
"""

from dataclasses import dataclass, field

import numpy as np

from .transforms import DIRECTIONS

# Japan twin angle: 84°33'30" = 84 + 33/60 + 30/3600 degrees
# This is the angle between c-axes of the two quartz individuals in a Japan twin.
# The twin plane is (11-22), and this angle is derived from the rhombohedral
# geometry of α-quartz with lattice parameters a = 4.913 Å, c = 5.405 Å.
_JAPAN_ANGLE_QUARTZ = 84.0 + 33.0 / 60.0 + 30.0 / 3600.0  # 84.55833...°


@dataclass(frozen=True)
class TwinLaw:
    """Definition of a crystallographic twin law.

    Attributes:
        name: Human-readable name of the twin law
        description: Detailed description of the twin operation
        twin_type: Type of twin ('contact', 'penetration', 'cyclic')
        render_mode: Rendering strategy ('unified', 'dual_crystal', 'v_shaped',
                     'cyclic', 'single_crystal')
        axis: Twin rotation/reflection axis (normalized)
        angle: Twin rotation angle in degrees
        habit: Default crystal habit for this twin
        habit_params: Optional parameters for the default habit
        examples: Example minerals exhibiting this twin
    """

    name: str
    description: str
    twin_type: str
    render_mode: str
    axis: np.ndarray
    angle: float
    habit: str
    habit_params: dict = field(default_factory=dict)
    examples: tuple[str, ...] = field(default_factory=tuple)

    def __hash__(self) -> int:
        """Make TwinLaw hashable by using immutable attributes."""
        return hash((self.name, self.twin_type, self.render_mode, self.angle, self.habit))


# Twin law definitions
# render_mode values:
#   'unified'          - Single polyhedron via halfspace intersection
#   'dual_crystal'     - Two complete interpenetrating crystals (fluorite, staurolite, iron cross)
#   'v_shaped'         - Two clipped halves with reflection (Japan, gypsum swallow-tail)
#   'contact_rotation' - Two clipped halves with rotation (spinel, albite, manebach, baveno)
#   'cyclic'           - Unified cyclic twin (trilling)
#   'single_crystal'   - External morphology unchanged (internal/electrical twins)

TWIN_LAWS: dict[str, TwinLaw] = {
    "spinel_law": TwinLaw(
        name="Spinel Law (Macle)",
        description="180° rotation about [111] with {111} composition plane",
        twin_type="contact",
        render_mode="contact_rotation",
        axis=DIRECTIONS["[111]"],
        angle=180.0,
        habit="octahedron",
        examples=("spinel", "diamond", "magnetite"),
    ),
    "iron_cross": TwinLaw(
        name="Iron Cross Twin",
        description="90° rotation about [001] (pyrite)",
        twin_type="penetration",
        render_mode="dual_crystal",
        axis=DIRECTIONS["[001]"],
        angle=90.0,
        habit="orthorhombic_prism",
        habit_params={"b_ratio": 0.5, "c_ratio": 2.5},
        examples=("pyrite",),
    ),
    "carlsbad": TwinLaw(
        name="Carlsbad Twin",
        description="180° rotation about [001] (feldspar)",
        twin_type="penetration",
        render_mode="dual_crystal",
        axis=DIRECTIONS["[001]"],
        angle=180.0,
        habit="orthorhombic_prism",
        examples=("orthoclase", "feldspar"),
    ),
    "albite": TwinLaw(
        name="Albite Twin",
        description="180° rotation about [010] with (010) composition plane",
        twin_type="contact",
        render_mode="contact_rotation",
        axis=DIRECTIONS["[010]"],
        angle=180.0,
        habit="feldspar_tabular",
        examples=("plagioclase", "albite"),
    ),
    "brazil": TwinLaw(
        name="Brazil Twin (Quartz)",
        description="180° rotation about [110] (optical twins, opposite handedness)",
        twin_type="penetration",
        render_mode="dual_crystal",
        axis=DIRECTIONS["[110]"],
        angle=180.0,
        habit="quartz_crystal",
        habit_params={"c_ratio": 2.5},
        examples=("quartz",),
    ),
    "dauphine": TwinLaw(
        name="Dauphine Twin (Quartz)",
        description="180° rotation about c-axis [001] (internal/electrical twin)",
        twin_type="penetration",
        render_mode="single_crystal",
        axis=DIRECTIONS["[001]"],
        angle=180.0,
        habit="quartz_crystal",
        habit_params={"c_ratio": 2.5},
        examples=("quartz",),
    ),
    "japan": TwinLaw(
        name="Japan Twin (Quartz)",
        description="Contact twin at 84°33'30\" angle (twin plane {11-22})",
        twin_type="contact",
        render_mode="v_shaped",
        axis=DIRECTIONS["[11-2]"],
        angle=_JAPAN_ANGLE_QUARTZ,
        habit="quartz_crystal",
        habit_params={"c_ratio": 2.5},
        examples=("quartz",),
    ),
    "trilling": TwinLaw(
        name="Trilling (Cyclic Twin)",
        description="Three crystals rotated 120° about c-axis",
        twin_type="cyclic",
        render_mode="cyclic",
        axis=DIRECTIONS["[001]"],
        angle=120.0,
        habit="tabular",
        examples=("chrysoberyl", "aragonite"),
    ),
    "fluorite": TwinLaw(
        name="Fluorite Penetration Twin",
        description="Two cubes interpenetrating along [111]",
        twin_type="penetration",
        render_mode="dual_crystal",
        axis=DIRECTIONS["[111]"],
        angle=180.0,
        habit="cube",
        examples=("fluorite",),
    ),
    "staurolite_60": TwinLaw(
        name="Staurolite 60° Twin",
        description="60° cross-shaped penetration twin",
        twin_type="penetration",
        render_mode="dual_crystal",
        axis=DIRECTIONS["[001]"],
        angle=60.0,
        habit="orthorhombic_prism",
        examples=("staurolite",),
    ),
    "staurolite_90": TwinLaw(
        name="Staurolite 90° Twin",
        description="90° cross-shaped penetration twin",
        twin_type="penetration",
        render_mode="dual_crystal",
        axis=DIRECTIONS["[001]"],
        angle=90.0,
        habit="orthorhombic_prism",
        examples=("staurolite",),
    ),
    "manebach": TwinLaw(
        name="Manebach Twin",
        description="180° rotation about [001] with (001) composition plane",
        twin_type="contact",
        render_mode="contact_rotation",
        axis=DIRECTIONS["[001]"],
        angle=180.0,
        habit="feldspar_tabular",
        examples=("orthoclase", "feldspar"),
    ),
    "baveno": TwinLaw(
        name="Baveno Twin",
        description="180° rotation about [021] with (021) composition plane",
        twin_type="contact",
        render_mode="contact_rotation",
        axis=DIRECTIONS["[021]"],
        angle=180.0,
        habit="feldspar_tabular",
        examples=("orthoclase", "feldspar"),
    ),
    "gypsum_swallow": TwinLaw(
        name="Gypsum Swallow-Tail Twin",
        description="Contact twin forming characteristic swallow-tail shape",
        twin_type="contact",
        render_mode="v_shaped",
        axis=DIRECTIONS["[100]"],
        angle=180.0,
        habit="tabular",
        examples=("gypsum",),
    ),
}


# Gemstone to twin law mapping
GEMSTONE_TWINS: dict[str, list[str]] = {
    "diamond": ["spinel_law"],
    "spinel": ["spinel_law"],
    "fluorite": ["fluorite"],
    "quartz": ["brazil", "dauphine", "japan"],
    "feldspar": ["carlsbad", "albite", "manebach", "baveno"],
    "orthoclase": ["carlsbad", "manebach", "baveno"],
    "plagioclase": ["albite"],
    "chrysoberyl": ["trilling"],
    "staurolite": ["staurolite_60", "staurolite_90"],
    "pyrite": ["iron_cross"],
    "aragonite": ["trilling"],
    "gypsum": ["gypsum_swallow"],
    "magnetite": ["spinel_law"],
}


def get_twin_law(name: str) -> TwinLaw:
    """Get a twin law by name.

    Args:
        name: Twin law name (e.g., 'spinel_law', 'japan')

    Returns:
        TwinLaw object

    Raises:
        ValueError: If twin law name is not recognized
    """
    if name not in TWIN_LAWS:
        available = ", ".join(sorted(TWIN_LAWS.keys()))
        raise ValueError(f"Unknown twin law: '{name}'. Available: {available}")
    return TWIN_LAWS[name]


def list_twin_laws() -> list[str]:
    """List all available twin law names.

    Returns:
        Sorted list of twin law names
    """
    return sorted(TWIN_LAWS.keys())


def get_gemstone_twins(gemstone: str) -> list[str]:
    """Get available twin laws for a gemstone.

    Args:
        gemstone: Gemstone name (case-insensitive)

    Returns:
        List of applicable twin law names, or empty list if none
    """
    return GEMSTONE_TWINS.get(gemstone.lower(), [])
