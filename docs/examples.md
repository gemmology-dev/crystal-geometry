# Examples

## Basic Geometry Generation

### From CDL Strings

```python
from crystal_geometry import cdl_string_to_geometry

# Simple octahedron
geom = cdl_string_to_geometry("cubic[m3m]:{111}")
print(f"Vertices: {len(geom.vertices)}")  # 6
print(f"Faces: {len(geom.faces)}")        # 8

# Diamond-like truncated octahedron
geom = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.3")
print(f"Faces: {len(geom.faces)}")        # 14 (8 octahedron + 6 cube)
```

### Using Convenience Constructors

```python
from crystal_geometry import (
    create_octahedron,
    create_cube,
    create_dodecahedron,
    create_truncated_octahedron,
)

# Create standard polyhedra
octahedron = create_octahedron()
cube = create_cube()
dodecahedron = create_dodecahedron()

# Create truncated forms
truncated = create_truncated_octahedron(octahedron_scale=1.0, cube_scale=1.3)
```

## Crystal Systems

### Cubic System

```python
from crystal_geometry import cdl_string_to_geometry

# Octahedron (8 faces)
octahedron = cdl_string_to_geometry("cubic[m3m]:{111}")

# Cube (6 faces)
cube = cdl_string_to_geometry("cubic[m3m]:{100}")

# Dodecahedron (12 faces)
dodecahedron = cdl_string_to_geometry("cubic[m3m]:{110}")

# Diamond-like
diamond = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@0.3")

# Garnet-like (dodecahedron + trapezohedron)
garnet = cdl_string_to_geometry("cubic[m3m]:{110}@1.0 + {211}@0.6")
```

### Hexagonal System

```python
from crystal_geometry import cdl_string_to_geometry

# Hexagonal prism with pinacoid
beryl = cdl_string_to_geometry("hexagonal[6/mmm]:{10-10}@1.0 + {0001}@0.5")
print(f"Beryl faces: {len(beryl.faces)}")  # 6 prism + 2 pinacoid = 8
```

### Trigonal System

```python
from crystal_geometry import cdl_string_to_geometry

# Quartz prism with rhombohedron
quartz = cdl_string_to_geometry("trigonal[-3m]:{10-10}@1.0 + {10-11}@0.8")

# Simple rhombohedron
calcite = cdl_string_to_geometry("trigonal[-3m]:{10-11}")
```

### Tetragonal System

```python
from crystal_geometry import cdl_string_to_geometry

# Zircon-like prism with pyramid
zircon = cdl_string_to_geometry("tetragonal[4/mmm]:{100}@1.0 + {101}@0.8")
```

## Working with Geometry

### Accessing Geometry Data

```python
from crystal_geometry import cdl_string_to_geometry
import numpy as np

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Vertex positions (Nx3 array)
vertices = geom.vertices
print(f"First vertex: {vertices[0]}")

# Face indices (list of tuples)
faces = geom.faces
print(f"First face vertices: {faces[0]}")

# Face normals
normals = geom.face_normals
print(f"First face normal: {normals[0]}")

# Miller indices for each face
millers = geom.face_millers
print(f"First face Miller: {millers[0]}")
```

### Geometry Validation

```python
from crystal_geometry import cdl_string_to_geometry

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Check validity
if geom.is_valid():
    print("Geometry is valid")

# Euler characteristic (should be 2 for convex polyhedra)
euler = geom.euler_characteristic()
print(f"V - E + F = {euler}")  # Should print 2
```

### Transformations

```python
from crystal_geometry import cdl_string_to_geometry
import numpy as np

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Scale to fit unit sphere
geom.scale_to_unit()

# Translate
geom.translate(np.array([1.0, 0.0, 0.0]))

# Get centroid
center = geom.center()
print(f"Centroid: {center}")
```

### Edges

```python
from crystal_geometry import cdl_string_to_geometry

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Get unique edges
edges = geom.get_edges()
print(f"Number of edges: {len(edges)}")  # 12 for octahedron

for v1, v2 in edges[:3]:
    print(f"Edge: {v1} -> {v2}")
```

### Export to Dictionary

```python
from crystal_geometry import cdl_string_to_geometry
import json

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Export to dictionary (JSON-serializable)
data = geom.to_dict()
print(json.dumps(data, indent=2))
```

## Symmetry Operations

### Point Group Operations

```python
from crystal_geometry import get_point_group_operations
import numpy as np

# Get symmetry matrices for m3m (48 operations)
ops = get_point_group_operations('m3m')
print(f"m3m has {len(ops)} symmetry operations")

# Apply to a vector
v = np.array([1, 0, 0])
for i, op in enumerate(ops[:5]):
    v_transformed = op @ v
    print(f"Operation {i}: {v} -> {v_transformed}")
```

### Generate Equivalent Faces

```python
from crystal_geometry import generate_equivalent_faces

# Generate all {111} equivalent faces in m3m
faces = generate_equivalent_faces(1, 1, 1, 'm3m')
print(f"Number of {111} faces in m3m: {len(faces)}")  # 8

# Generate all {100} equivalent faces
faces = generate_equivalent_faces(1, 0, 0, 'm3m')
print(f"Number of {100} faces in m3m: {len(faces)}")  # 6
```

### Miller Index to Normal

```python
from crystal_geometry import miller_to_normal, LatticeParams
import numpy as np

# Cubic lattice
cubic = LatticeParams.cubic()
normal = miller_to_normal(1, 1, 1, cubic)
print(f"(111) normal in cubic: {normal}")

# Hexagonal lattice
hex_lattice = LatticeParams.hexagonal(c_ratio=1.2)
normal = miller_to_normal(1, 0, -1, 0, hex_lattice)  # (10-10) prism
print(f"(10-10) normal in hexagonal: {normal}")
```

## Integration with Other Packages

### With crystal-renderer

```python
from crystal_geometry import cdl_string_to_geometry
from crystal_renderer import generate_geometry_svg

# Generate geometry
geom = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.3")

# Render to SVG
generate_geometry_svg(
    geom.vertices,
    geom.faces,
    "crystal.svg",
    face_color='#81D4FA',
    edge_color='#0277BD'
)
```

### With mineral-database

```python
from mineral_database import get_preset
from cdl_parser import parse_cdl
from crystal_geometry import cdl_to_geometry

# Get preset CDL string
diamond = get_preset('diamond')
desc = parse_cdl(diamond['cdl'])

# Generate geometry with custom lattice parameters
geom = cdl_to_geometry(desc, c_ratio=1.0)
print(f"Diamond geometry: {len(geom.vertices)} vertices, {len(geom.faces)} faces")
```

## Advanced Usage

### Custom Lattice Parameters

```python
from crystal_geometry import cdl_to_geometry, LatticeParams
from cdl_parser import parse_cdl

# Parse CDL
desc = parse_cdl("tetragonal[4/mmm]:{100}@1.0 + {001}@0.5")

# Generate with custom c/a ratio
geom = cdl_to_geometry(desc, c_ratio=2.0)  # Elongated tetragonal
```

### Low-Level Half-Space Operations

```python
from crystal_geometry import HalfSpace, build_crystal
import numpy as np

# Define half-spaces manually
halfspaces = [
    HalfSpace(normal=np.array([1, 0, 0]), d=1.0),
    HalfSpace(normal=np.array([-1, 0, 0]), d=1.0),
    HalfSpace(normal=np.array([0, 1, 0]), d=1.0),
    HalfSpace(normal=np.array([0, -1, 0]), d=1.0),
    HalfSpace(normal=np.array([0, 0, 1]), d=1.0),
    HalfSpace(normal=np.array([0, 0, -1]), d=1.0),
]

# Build geometry
geometry = build_crystal(halfspaces)
print(f"Built cube: {len(geometry.faces)} faces")
```
