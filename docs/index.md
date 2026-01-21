# Crystal Geometry

**3D Crystal Geometry Engine** - Computes polyhedra from Crystal Description Language (CDL) strings using half-space intersection with point group symmetry.

Part of the [Gemmology Project](https://gemmology.dev).

## Overview

Crystal Geometry provides a robust 3D geometry engine for crystallographic visualization:

- **CDL Integration**: Parse Crystal Description Language strings to 3D geometry
- **Point Group Symmetry**: All 32 crystallographic point groups supported
- **Half-Space Intersection**: Robust geometry computation using scipy
- **7 Crystal Systems**: Cubic, tetragonal, orthorhombic, hexagonal, trigonal, monoclinic, triclinic
- **Miller Indices**: Full support for 3-index (hkl) and 4-index (hkil) notation

## Installation

```bash
pip install gemmology-crystal-geometry
```

### Requirements

- Python >= 3.10
- numpy >= 1.20.0
- scipy >= 1.7.0
- gemmology-cdl-parser >= 1.0.0

## Quick Start

```python
from crystal_geometry import cdl_string_to_geometry, create_octahedron

# Create geometry from CDL string
geom = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.3")
print(f"Vertices: {len(geom.vertices)}, Faces: {len(geom.faces)}")

# Use convenience constructors
octahedron = create_octahedron()
print(f"Octahedron has {len(octahedron.faces)} faces")
```

## Crystal Systems and Point Groups

| System | Point Groups |
|--------|-------------|
| Cubic | m3m, 432, -43m, m-3, 23 |
| Tetragonal | 4/mmm, 422, 4mm, -42m, 4/m, -4, 4 |
| Hexagonal | 6/mmm, 622, 6mm, -6m2, 6/m, -6, 6 |
| Trigonal | -3m, 32, 3m, -3, 3 |
| Orthorhombic | mmm, 222, mm2 |
| Monoclinic | 2/m, 2, m |
| Triclinic | -1, 1 |

## Core Concepts

### Half-Space Intersection

Crystal forms are computed by intersecting half-spaces defined by Miller indices. Each crystal face defines a half-space, and the polyhedron is the intersection of all half-spaces.

### Point Group Symmetry

Point group operations are applied to generate all equivalent faces from a single Miller index. For example, the {111} form in point group m3m generates 8 equivalent faces forming an octahedron.

### Lattice Parameters

Different crystal systems have different lattice parameters:

- **Cubic**: $a = b = c$, $\alpha = \beta = \gamma = 90°$
- **Tetragonal**: $a = b \neq c$, $\alpha = \beta = \gamma = 90°$
- **Hexagonal**: $a = b \neq c$, $\alpha = \beta = 90°$, $\gamma = 120°$

## Related Packages

- [cdl-parser](https://cdl-parser.gemmology.dev) - Crystal Description Language parser
- [mineral-database](https://mineral-database.gemmology.dev) - Mineral presets
- [crystal-renderer](https://crystal-renderer.gemmology.dev) - SVG/3D rendering

## License

MIT License - see [LICENSE](https://github.com/gemmology-dev/crystal-geometry/blob/main/LICENSE) for details.
