# API Reference

## Geometry Generation

### cdl_string_to_geometry

Generate geometry directly from a CDL string.

```python
from crystal_geometry import cdl_string_to_geometry

geom = cdl_string_to_geometry("cubic[m3m]:{111}")
```

::: crystal_geometry.cdl_string_to_geometry

### cdl_to_geometry

Generate geometry from a parsed CrystalDescription.

```python
from cdl_parser import parse_cdl
from crystal_geometry import cdl_to_geometry

desc = parse_cdl("cubic[m3m]:{111}@1.0 + {100}@1.3")
geom = cdl_to_geometry(desc, c_ratio=1.0)
```

::: crystal_geometry.cdl_to_geometry

## Convenience Constructors

### create_octahedron

```python
from crystal_geometry import create_octahedron

octahedron = create_octahedron(scale=1.0)
```

::: crystal_geometry.create_octahedron

### create_cube

```python
from crystal_geometry import create_cube

cube = create_cube(scale=1.0)
```

::: crystal_geometry.create_cube

### create_dodecahedron

```python
from crystal_geometry import create_dodecahedron

dodecahedron = create_dodecahedron(scale=1.0)
```

::: crystal_geometry.create_dodecahedron

### create_truncated_octahedron

```python
from crystal_geometry import create_truncated_octahedron

truncated = create_truncated_octahedron(octahedron_scale=1.0, cube_scale=1.3)
```

::: crystal_geometry.create_truncated_octahedron

## CrystalGeometry Class

The main geometry container class.

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

::: crystal_geometry.CrystalGeometry

## Symmetry Operations

### get_point_group_operations

Get symmetry operations for a crystallographic point group.

```python
from crystal_geometry import get_point_group_operations

ops = get_point_group_operations('m3m')  # 48 operations
ops = get_point_group_operations('6/mmm')  # 24 operations
```

::: crystal_geometry.get_point_group_operations

### generate_equivalent_faces

Generate equivalent faces from one Miller index using point group symmetry.

```python
from crystal_geometry import generate_equivalent_faces

faces = generate_equivalent_faces(1, 1, 1, 'm3m')  # 8 faces for {111}
faces = generate_equivalent_faces(1, 0, 0, 'm3m')  # 6 faces for {100}
```

::: crystal_geometry.generate_equivalent_faces

### miller_to_normal

Convert Miller indices to a unit normal vector.

```python
from crystal_geometry import miller_to_normal, LatticeParams

lattice = LatticeParams.cubic()
normal = miller_to_normal(1, 1, 1, lattice)
```

::: crystal_geometry.miller_to_normal

## Lattice Parameters

### LatticeParams

Class for defining crystal lattice parameters.

```python
from crystal_geometry import LatticeParams

# Factory methods for different crystal systems
cubic = LatticeParams.cubic()
tetragonal = LatticeParams.tetragonal(c_ratio=1.5)
hexagonal = LatticeParams.hexagonal(c_ratio=1.2)
orthorhombic = LatticeParams.orthorhombic(a=1.0, b=1.2, c=1.5)
monoclinic = LatticeParams.monoclinic(a=1.0, b=1.2, c=1.5, beta=100.0)
triclinic = LatticeParams.triclinic(a=1.0, b=1.2, c=1.5, alpha=80.0, beta=100.0, gamma=110.0)
```

::: crystal_geometry.LatticeParams

### get_lattice_for_system

Get default lattice parameters for a crystal system.

```python
from crystal_geometry import get_lattice_for_system

lattice = get_lattice_for_system('cubic')
lattice = get_lattice_for_system('hexagonal', c_ratio=1.2)
```

::: crystal_geometry.get_lattice_for_system

## Low-Level Functions

### halfspace_intersection_3d

Compute the convex polyhedron from half-space intersection.

```python
from crystal_geometry import halfspace_intersection_3d
import numpy as np

# Half-spaces as Ax <= b
A = np.array([...])  # Normals
b = np.array([...])  # Distances

vertices, faces = halfspace_intersection_3d(A, b)
```

::: crystal_geometry.halfspace_intersection_3d

### compute_face_vertices

Compute ordered vertices for each face of a polyhedron.

```python
from crystal_geometry import compute_face_vertices

face_vertices = compute_face_vertices(vertices, faces)
```

::: crystal_geometry.compute_face_vertices
