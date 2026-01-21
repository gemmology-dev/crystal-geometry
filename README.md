# Crystal Geometry

3D crystal geometry engine for crystallographic visualization. Computes polyhedra from Crystal Description Language (CDL) strings using half-space intersection with point group symmetry.

Part of the [Gemmology Project](https://gemmology.dev).

## Installation

```bash
pip install crystal-geometry
```

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

## Features

- **CDL Integration**: Parse Crystal Description Language strings to 3D geometry
- **Point Group Symmetry**: All 32 crystallographic point groups supported
- **Half-Space Intersection**: Robust geometry computation using scipy
- **7 Crystal Systems**: Cubic, tetragonal, orthorhombic, hexagonal, trigonal, monoclinic, triclinic
- **Miller Indices**: Full support for 3-index (hkl) and 4-index (hkil) notation

## Core API

### Geometry Generation

```python
from cdl_parser import parse_cdl
from crystal_geometry import cdl_to_geometry, cdl_string_to_geometry

# From CDL string directly
geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# From parsed description (more control)
desc = parse_cdl("cubic[m3m]:{111}@1.0 + {100}@1.3")
geom = cdl_to_geometry(desc, c_ratio=1.0)
```

### Convenience Constructors

```python
from crystal_geometry import (
    create_octahedron,
    create_cube,
    create_dodecahedron,
    create_truncated_octahedron,
)

# Regular polyhedra
octahedron = create_octahedron(scale=1.0)
cube = create_cube(scale=1.0)
dodecahedron = create_dodecahedron(scale=1.0)
truncated = create_truncated_octahedron(octahedron_scale=1.0, cube_scale=1.3)
```

### CrystalGeometry Class

```python
from crystal_geometry import CrystalGeometry

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Properties
geom.vertices      # Nx3 numpy array of vertex positions
geom.faces         # List of face vertex indices
geom.face_normals  # List of unit normal vectors
geom.face_forms    # Form index for each face
geom.face_millers  # Miller indices for each face

# Methods
geom.get_edges()           # Get unique edges as vertex pairs
geom.center()              # Compute centroid
geom.scale_to_unit()       # Scale to fit unit sphere
geom.translate(offset)     # Translate by vector
geom.euler_characteristic()  # V - E + F (should be 2)
geom.is_valid()            # Verify geometry integrity
geom.to_dict()             # Export to dictionary
```

### Symmetry Operations

```python
from crystal_geometry import (
    get_point_group_operations,
    generate_equivalent_faces,
    miller_to_normal,
    LatticeParams,
)

# Get symmetry operations for a point group
ops = get_point_group_operations('m3m')  # 48 operations
ops = get_point_group_operations('6/mmm')  # 24 operations

# Generate equivalent faces from one Miller index
faces = generate_equivalent_faces(1, 1, 1, 'm3m')  # 8 faces for {111}
faces = generate_equivalent_faces(1, 0, 0, 'm3m')  # 6 faces for {100}

# Convert Miller indices to normal vector
lattice = LatticeParams.cubic()
normal = miller_to_normal(1, 1, 1, lattice)

# Create lattice parameters
cubic = LatticeParams.cubic()
tetragonal = LatticeParams.tetragonal(c_ratio=1.5)
hexagonal = LatticeParams.hexagonal(c_ratio=1.2)
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

## Examples

### Diamond-like Crystal

```python
# Octahedron truncated by cube
geom = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.3")
assert len(geom.faces) == 14  # 8 octahedron + 6 cube
```

### Garnet-like Crystal

```python
# Dodecahedron with trapezohedron
geom = cdl_string_to_geometry("cubic[m3m]:{110}@1.0 + {211}@0.6")
```

### Hexagonal Prism

```python
# Prism with pinacoid termination
geom = cdl_string_to_geometry("hexagonal[6/mmm]:{10-10}@1.0 + {0001}@0.5")
```

### Quartz-like Crystal

```python
# Prism with rhombohedron
geom = cdl_string_to_geometry("trigonal[-3m]:{10-10}@1.0 + {10-11}@0.8")
```

## Requirements

- Python >= 3.10
- numpy >= 1.20.0
- scipy >= 1.7.0
- cdl-parser >= 1.0.0

## Documentation

See [crystal-geometry.gemmology.dev](https://crystal-geometry.gemmology.dev) for full documentation.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Packages

- [cdl-parser](https://github.com/gemmology-dev/cdl-parser) - Crystal Description Language parser
- [mineral-database](https://github.com/gemmology-dev/mineral-database) - Mineral preset database
- [crystal-renderer](https://github.com/gemmology-dev/crystal-renderer) - SVG/3D rendering
