# Troubleshooting Guide

Common issues and solutions when working with crystal-geometry.

## Geometry Generation Errors

### "Failed to compute crystal geometry - no valid intersection"

**Cause:** The half-space intersection algorithm couldn't find a valid bounded polyhedron.

**Common reasons:**

1. **Incompatible forms** - Forms that don't produce a closed polyhedron
2. **Scale values too extreme** - Very large or very small scales
3. **System/form mismatch** - Using cubic forms in hexagonal system

**Solutions:**

```python
from crystal_geometry import cdl_string_to_geometry

# Problem: Single form without closure
# {100} alone in some systems may not produce bounded geometry
bad = cdl_string_to_geometry("orthorhombic[mmm]:{100}")  # May fail

# Solution: Add complementary forms
good = cdl_string_to_geometry("orthorhombic[mmm]:{100}@1.0 + {010}@1.0 + {001}@1.0")
```

### "ValueError: Interior point finding failed"

**Cause:** The algorithm couldn't find a point inside all half-spaces.

**Solutions:**

1. Check that forms are compatible
2. Adjust scale values to be closer to 1.0
3. Ensure forms intersect to create a bounded region

```python
# Problem: Extreme scale difference
bad = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@100.0")

# Solution: Use reasonable scale ratios (0.1 to 3.0)
good = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.5")
```

### Unexpected Number of Faces

**Cause:** Point group symmetry affects face count.

**Debug approach:**

```python
from crystal_geometry import generate_equivalent_faces, cdl_string_to_geometry

# Check expected face count
faces_111 = generate_equivalent_faces(1, 1, 1, 'm3m')
faces_100 = generate_equivalent_faces(1, 0, 0, 'm3m')
print(f"Expected: {len(faces_111)} + {len(faces_100)} = {len(faces_111) + len(faces_100)}")

# Generate and compare
geom = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.3")
print(f"Actual faces: {len(geom.faces)}")

# Note: Some faces may merge if scales make forms tangent
```

### Geometry Validation Fails

**Use built-in validation:**

```python
from crystal_geometry import cdl_string_to_geometry

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Check validity
if not geom.is_valid():
    print("Geometry issues detected")

# Euler characteristic should be 2 for convex polyhedra
euler = geom.euler_characteristic()
print(f"V - E + F = {euler}")  # Should print 2

if euler != 2:
    print("Warning: Non-convex or invalid geometry")
```

---

## Numerical Precision Issues

### Duplicate Vertices

**Cause:** Floating-point precision creates near-duplicate vertices.

**Solution:** The library automatically deduplicates with tolerance 1e-6:

```python
from crystal_geometry import cdl_string_to_geometry

geom = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.0")  # Forms tangent

# Vertices are automatically deduplicated
print(f"Unique vertices: {len(geom.vertices)}")
```

### Scale Sensitivity

**Extreme scales can cause numerical issues:**

```python
# Problematic: Very small scale
bad = cdl_string_to_geometry("cubic[m3m]:{111}@0.001")

# Better: Use scales closer to 1.0
good = cdl_string_to_geometry("cubic[m3m]:{111}@1.0")

# If you need small relative to another form:
geom = cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.001")  # Very slight truncation
```

### Face Normal Orientation

**Faces should have outward-pointing normals:**

```python
import numpy as np
from crystal_geometry import cdl_string_to_geometry

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Check normals point outward (dot with position should be positive)
for i, (face, normal) in enumerate(zip(geom.faces, geom.face_normals)):
    center = np.mean(geom.vertices[list(face)], axis=0)
    dot = np.dot(center, normal)
    if dot < 0:
        print(f"Face {i} has inward normal")
```

---

## Lattice Parameter Issues

### Non-Cubic Systems

**c/a ratio affects geometry significantly:**

```python
from crystal_geometry import cdl_to_geometry, LatticeParams
from cdl_parser import parse_cdl

desc = parse_cdl("tetragonal[4/mmm]:{100}@1.0 + {101}@0.8")

# Different c/a ratios produce different geometries
geom1 = cdl_to_geometry(desc, c_ratio=0.5)  # Flattened
geom2 = cdl_to_geometry(desc, c_ratio=1.0)  # Equant
geom3 = cdl_to_geometry(desc, c_ratio=2.0)  # Elongated

print(f"c/a=0.5: {len(geom1.vertices)} vertices")
print(f"c/a=1.0: {len(geom2.vertices)} vertices")
print(f"c/a=2.0: {len(geom3.vertices)} vertices")
```

### Hexagonal/Trigonal Systems

**Using 3-index vs 4-index notation:**

```python
# In crystal-geometry, use 3-index internally
# The parser handles 4-index (Miller-Bravais) conversion

# These are equivalent after parsing:
# CDL: {10-10} → internal: (1, 0, -1) for normal calculation
# CDL: {0001} → internal: (0, 0, 1)
```

---

## Performance Issues

### Large Numbers of Forms

**Many forms slow down computation:**

```python
from crystal_geometry import cdl_string_to_geometry
import time

# Simple: fast
start = time.time()
simple = cdl_string_to_geometry("cubic[m3m]:{111}")
print(f"Simple: {time.time() - start:.3f}s")

# Complex: slower
start = time.time()
complex_crystal = cdl_string_to_geometry(
    "cubic[m3m]:{111}@1.0 + {100}@1.2 + {110}@0.8 + {211}@0.5"
)
print(f"Complex: {time.time() - start:.3f}s")
```

**Recommendations:**

- Keep forms to 4-5 maximum for interactive use
- Pre-compute geometries for batch rendering
- Use simpler point groups when full symmetry isn't needed

### Memory Usage

**Large geometries consume memory:**

```python
import sys
from crystal_geometry import cdl_string_to_geometry

geom = cdl_string_to_geometry("cubic[m3m]:{321}")  # 48 faces!

# Check approximate memory
vertices_size = sys.getsizeof(geom.vertices)
faces_size = sum(sys.getsizeof(f) for f in geom.faces)
print(f"Vertices: {vertices_size} bytes")
print(f"Faces: {faces_size} bytes")
```

---

## Integration Issues

### With cdl-parser

**Ensure compatible versions:**

```python
from cdl_parser import parse_cdl, __version__ as parser_version
from crystal_geometry import __version__ as geom_version

print(f"cdl-parser: {parser_version}")
print(f"crystal-geometry: {geom_version}")

# Parse then generate
desc = parse_cdl("cubic[m3m]:{111}")
# desc contains CrystalDescription object
```

### With crystal-renderer

**Pass geometry correctly:**

```python
from crystal_geometry import cdl_string_to_geometry
from crystal_renderer import generate_geometry_svg

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Use vertices and faces properties
generate_geometry_svg(
    geom.vertices,
    geom.faces,
    "output.svg"
)
```

### With numpy

**Vertices are numpy arrays:**

```python
import numpy as np
from crystal_geometry import cdl_string_to_geometry

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Vertices is Nx3 numpy array
print(f"Vertices type: {type(geom.vertices)}")
print(f"Vertices shape: {geom.vertices.shape}")
print(f"Vertices dtype: {geom.vertices.dtype}")

# Safe operations
centroid = np.mean(geom.vertices, axis=0)
max_distance = np.max(np.linalg.norm(geom.vertices, axis=1))
```

---

## Debugging Tips

### Visualize Intermediate Results

```python
from crystal_geometry import (
    cdl_string_to_geometry,
    generate_equivalent_faces,
    miller_to_normal,
    LatticeParams
)

# Step 1: Check face generation
faces = generate_equivalent_faces(1, 1, 1, 'm3m')
print(f"Generated {len(faces)} equivalent faces for {{111}}")
for h, k, l in faces[:4]:
    print(f"  ({h}, {k}, {l})")

# Step 2: Check normal vectors
lattice = LatticeParams.cubic()
for h, k, l in faces[:4]:
    normal = miller_to_normal(h, k, l, lattice)
    print(f"  ({h},{k},{l}) → normal: {normal}")

# Step 3: Generate and inspect
geom = cdl_string_to_geometry("cubic[m3m]:{111}")
print(f"\nFinal geometry:")
print(f"  Vertices: {len(geom.vertices)}")
print(f"  Faces: {len(geom.faces)}")
print(f"  Valid: {geom.is_valid()}")
```

### Export for External Inspection

```python
from crystal_geometry import cdl_string_to_geometry
import json

geom = cdl_string_to_geometry("cubic[m3m]:{111}")

# Export to dict for inspection
data = geom.to_dict()
print(json.dumps(data, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))
```

### Check Euler Characteristic

For convex polyhedra, V - E + F = 2:

```python
geom = cdl_string_to_geometry("cubic[m3m]:{111}")

V = len(geom.vertices)
E = len(geom.get_edges())
F = len(geom.faces)

print(f"V={V}, E={E}, F={F}")
print(f"V - E + F = {V - E + F}")  # Should be 2
```
