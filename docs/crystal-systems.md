# Crystal Systems Guide

A comprehensive guide to working with different crystal systems in crystal-geometry.

## Overview

Crystal-geometry supports all 7 crystal systems with their unique lattice parameters and symmetry operations. Understanding these systems is essential for generating accurate crystal morphologies.

## The 7 Crystal Systems

### Cubic System

**Lattice Parameters:** a = b = c, α = β = γ = 90°

The highest-symmetry system with three equal, perpendicular axes.

```python
from crystal_geometry import LatticeParams, cdl_string_to_geometry

# Cubic lattice (default)
lattice = LatticeParams.cubic()

# Generate cubic crystals
octahedron = cdl_string_to_geometry("cubic[m3m]:{111}")
cube = cdl_string_to_geometry("cubic[m3m]:{100}")
dodecahedron = cdl_string_to_geometry("cubic[m3m]:{110}")
```

**Common Minerals:** Diamond, garnet, fluorite, pyrite, spinel

**Point Groups:** m3m (48), 432 (24), -43m (24), m-3 (24), 23 (12)

### Tetragonal System

**Lattice Parameters:** a = b ≠ c, α = β = γ = 90°

Two equal axes perpendicular to a unique c-axis.

```python
from crystal_geometry import LatticeParams, cdl_string_to_geometry

# Tetragonal with c/a ratio of 1.5
lattice = LatticeParams.tetragonal(c_ratio=1.5)

# Zircon-like crystal
zircon = cdl_string_to_geometry("tetragonal[4/mmm]:{100}@1.0 + {101}@0.8")
```

**c/a Ratio Effects:**

| c/a Ratio | Crystal Appearance |
|-----------|-------------------|
| < 1.0 | Flattened (tabular) |
| = 1.0 | Equant |
| > 1.0 | Elongated (prismatic) |

**Common Minerals:** Zircon (c/a ≈ 0.64), rutile (c/a ≈ 0.64), cassiterite

### Hexagonal System

**Lattice Parameters:** a = b ≠ c, α = β = 90°, γ = 120°

Three equal axes at 120° in the basal plane, perpendicular to c.

```python
from crystal_geometry import LatticeParams, cdl_string_to_geometry

# Hexagonal with c/a ratio of 1.2
lattice = LatticeParams.hexagonal(c_ratio=1.2)

# Beryl-like crystal (prism + pinacoid)
beryl = cdl_string_to_geometry("hexagonal[6/mmm]:{10-10}@1.0 + {0001}@0.5")
```

**Miller-Bravais Notation:**

Hexagonal crystals use 4-index notation {hkil} where i = -(h+k):

```python
# Hexagonal prism: h=1, k=0, i=-1, l=0
"{10-10}"

# Basal pinacoid: h=0, k=0, i=0, l=1
"{0001}"

# Hexagonal dipyramid
"{10-11}"
```

**Common Minerals:** Beryl (c/a ≈ 0.50), apatite, graphite

### Trigonal System

**Lattice Parameters:** Same as hexagonal (a = b ≠ c, γ = 120°)

Shares hexagonal axes but has 3-fold rather than 6-fold symmetry.

```python
from crystal_geometry import cdl_string_to_geometry

# Quartz crystal (prism + rhombohedra)
quartz = cdl_string_to_geometry("trigonal[32]:{10-10}@1.0 + {10-11}@0.8 + {01-11}@0.8")

# Corundum (ruby/sapphire)
corundum = cdl_string_to_geometry("trigonal[-3m]:{10-10}@1.0 + {0001}@0.3 + {10-11}@0.5")
```

**Rhombohedra:**

| Form | Description |
|------|-------------|
| {10-11} | Positive rhombohedron (r) |
| {01-11} | Negative rhombohedron (z) |

**Common Minerals:** Quartz, calcite, corundum, tourmaline

### Orthorhombic System

**Lattice Parameters:** a ≠ b ≠ c, α = β = γ = 90°

Three unequal perpendicular axes.

```python
from crystal_geometry import cdl_string_to_geometry

# Topaz-like crystal
topaz = cdl_string_to_geometry("orthorhombic[mmm]:{110}@1.0 + {011}@0.6 + {001}@0.3")

# Olivine (peridot)
olivine = cdl_string_to_geometry("orthorhombic[mmm]:{010}@1.0 + {110}@0.8 + {021}@0.5")
```

**Common Minerals:** Topaz, olivine, aragonite, barite

### Monoclinic System

**Lattice Parameters:** a ≠ b ≠ c, α = γ = 90°, β ≠ 90°

One axis inclined to the other two.

```python
from crystal_geometry import cdl_string_to_geometry

# Orthoclase feldspar
orthoclase = cdl_string_to_geometry("monoclinic[2/m]:{010}@1.0 + {001}@0.8 + {110}@0.6")

# Gypsum
gypsum = cdl_string_to_geometry("monoclinic[2/m]:{010}@1.0 + {111}@0.5")
```

**Common Minerals:** Orthoclase, gypsum, kunzite, epidote

### Triclinic System

**Lattice Parameters:** a ≠ b ≠ c, α ≠ β ≠ γ ≠ 90°

Three unequal axes at oblique angles.

```python
from crystal_geometry import cdl_string_to_geometry

# Plagioclase feldspar
plagioclase = cdl_string_to_geometry("triclinic[-1]:{010}@1.0 + {001}@0.8 + {110}@0.6")
```

**Common Minerals:** Plagioclase feldspars, kyanite, turquoise

---

## Lattice Parameters in Practice

### Setting c/a Ratios

The c/a ratio significantly affects crystal shape:

```python
from crystal_geometry import cdl_to_geometry, LatticeParams
from cdl_parser import parse_cdl

# Parse CDL
desc = parse_cdl("tetragonal[4/mmm]:{100}@1.0 + {101}@0.8")

# Generate with different c/a ratios
geom_elongated = cdl_to_geometry(desc, c_ratio=2.0)  # Elongated
geom_equant = cdl_to_geometry(desc, c_ratio=1.0)     # Equant
geom_flattened = cdl_to_geometry(desc, c_ratio=0.5)  # Flattened

print(f"Elongated: {len(geom_elongated.faces)} faces")
print(f"Equant: {len(geom_equant.faces)} faces")
print(f"Flattened: {len(geom_flattened.faces)} faces")
```

### Real Mineral c/a Ratios

| Mineral | System | c/a Ratio |
|---------|--------|-----------|
| Zircon | Tetragonal | 0.64 |
| Rutile | Tetragonal | 0.64 |
| Beryl | Hexagonal | 0.50 |
| Apatite | Hexagonal | 0.73 |
| Quartz | Trigonal | 1.10 |
| Calcite | Trigonal | 0.85 |
| Corundum | Trigonal | 0.36 |

### Working with Hexagonal/Trigonal Systems

```python
from crystal_geometry import (
    cdl_string_to_geometry,
    get_lattice_for_system,
    miller_to_normal
)

# Get hexagonal lattice with custom c/a
lattice = get_lattice_for_system('hexagonal', c_ratio=0.5)

# Convert Miller-Bravais to normal vector
# For {10-10}: h=1, k=0, l=-1 (3-index equivalent)
normal = miller_to_normal(1, 0, -1, lattice)
print(f"Prism face normal: {normal}")
```

---

## Symmetry Operations

### Understanding Point Group Operations

Point groups define the symmetry operations for each crystal system:

```python
from crystal_geometry import get_point_group_operations

# Get all 48 operations for cubic m3m
ops = get_point_group_operations('m3m')
print(f"m3m has {len(ops)} symmetry operations")

# Get trigonal symmetry
ops_trig = get_point_group_operations('-3m')
print(f"-3m has {len(ops_trig)} symmetry operations")
```

### Generating Equivalent Faces

Symmetry operations generate all equivalent faces from one Miller index:

```python
from crystal_geometry import generate_equivalent_faces

# Cubic {111} generates 8 equivalent faces
faces_111 = generate_equivalent_faces(1, 1, 1, 'm3m')
print(f"Cubic {{111}}: {len(faces_111)} faces")

# Cubic {100} generates 6 equivalent faces
faces_100 = generate_equivalent_faces(1, 0, 0, 'm3m')
print(f"Cubic {{100}}: {len(faces_100)} faces")

# Hexagonal {10-10} generates 6 equivalent faces
faces_prism = generate_equivalent_faces(1, 0, -1, '6/mmm')
print(f"Hexagonal {{10-10}}: {len(faces_prism)} faces")
```

### Face Count by System

| Form | Cubic m3m | Tetragonal 4/mmm | Hexagonal 6/mmm | Trigonal -3m |
|------|-----------|------------------|-----------------|--------------|
| {100} | 6 | 4 | - | - |
| {111} | 8 | 8 | - | - |
| {110} | 12 | 4 | - | - |
| {10-10} | - | - | 6 | 6 |
| {0001} | - | - | 2 | 2 |
| {10-11} | - | - | 12 | 6 |

---

## Common Morphology Patterns

### Diamond Habit

```python
# Classic diamond octahedron with cube modification
diamond = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@0.3")
```

### Garnet Habit

```python
# Dodecahedron with trapezohedron (typical almandine)
garnet = cdl_string_to_geometry("cubic[m3m]:{110}@1.0 + {211}@0.6")
```

### Quartz Habit

```python
# Hexagonal prism with positive and negative rhombohedra
quartz = cdl_string_to_geometry("trigonal[32]:{10-10}@1.0 + {10-11}@0.8 + {01-11}@0.8")
```

### Beryl/Emerald Habit

```python
# Hexagonal prism with basal pinacoid
beryl = cdl_string_to_geometry("hexagonal[6/mmm]:{10-10}@1.0 + {0001}@0.5")
```

### Corundum (Ruby/Sapphire) Habit

```python
# Barrel-shaped with prism, pinacoid, and rhombohedron
corundum = cdl_string_to_geometry("trigonal[-3m]:{10-10}@1.0 + {0001}@0.3 + {10-11}@0.5")
```

---

## Troubleshooting

### Common Issues

**"Failed to compute crystal geometry"**

Usually indicates incompatible forms for the specified symmetry:

```python
# Wrong: Using cubic form with hexagonal system
# This will fail or produce unexpected results
wrong = cdl_string_to_geometry("hexagonal[6/mmm]:{111}")  # Don't do this

# Correct: Use appropriate hexagonal forms
correct = cdl_string_to_geometry("hexagonal[6/mmm]:{10-10}@1.0 + {0001}@0.5")
```

**Unexpected face counts**

Check that point group is appropriate for the form:

```python
from crystal_geometry import generate_equivalent_faces

# With full cubic symmetry
faces_full = generate_equivalent_faces(1, 1, 1, 'm3m')
print(f"m3m: {len(faces_full)} faces")  # 8

# With tetrahedral symmetry
faces_tet = generate_equivalent_faces(1, 1, 1, '-43m')
print(f"-43m: {len(faces_tet)} faces")  # 4 (only half)
```

**Geometry validation fails**

Use `is_valid()` to check geometry integrity:

```python
geom = cdl_string_to_geometry("cubic[m3m]:{111}")

if not geom.is_valid():
    print("Geometry has issues")
    print(f"Euler characteristic: {geom.euler_characteristic()}")  # Should be 2
```
