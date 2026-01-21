# Crystal-Geometry Optimizations & Bug Fixes

**Status**: Pending Implementation
**Priority**: High (blocks hexagonal/trigonal systems)
**Date**: January 2026

---

## Critical Bug Fixes

### 1. Hexagonal/Trigonal Geometry Failure

**Issue**: `{10-10}` and similar Miller-Bravais indices fail half-space intersection.

```python
# This fails:
cdl_string_to_geometry("hexagonal[6/mmm]:{10-10}@1.0")
# ValueError: Failed to compute crystal geometry - no valid intersection
```

**Root Cause Analysis**:
- Interior point selection fails for hexagonal geometry normals
- Reciprocal lattice computation may produce malformed normals
- 4-index to 3-index conversion might lose precision

**Proposed Fix**:
```python
# In geometry.py, improve interior point selection:
def _find_interior_point(normals: List[np.ndarray], distances: List[float]) -> np.ndarray:
    """Find a point strictly inside all half-spaces."""
    # Current: uses origin (0,0,0) which may not be interior
    # Fix: Use centroid of bounding box or linear programming

    # Option 1: Chebyshev center (largest inscribed sphere)
    from scipy.optimize import linprog
    # Maximize r subject to: n_i · x + r ≤ d_i

    # Option 2: Iterative shrinking
    point = np.zeros(3)
    for _ in range(10):
        violations = [n @ point - d for n, d in zip(normals, distances)]
        if all(v < 0 for v in violations):
            return point
        point *= 0.9  # Shrink toward origin

    return point
```

**Files to Modify**:
- `src/crystal_geometry/geometry.py` (lines 22-61)

---

### 2. Point Group Operation Count Discrepancies

**Issue**: Two point groups generate incorrect operation counts:

| Point Group | Actual | Expected | Over-generation |
|-------------|--------|----------|-----------------|
| `4/m` | 8 | 4 | 2× |
| `6/m` | 12 | 6 | 2× |

**Root Cause**: Generator matrices include redundant operations.

**Current Generators** (symmetry.py):
```python
'4/m': [C4z, Mxy]      # Generates 8 instead of 4
'6/m': [C6z, C2x, Mxy] # Generates 12 instead of 6
```

**Proposed Fix**:
```python
# 4/m should be: C4 rotation + horizontal mirror
# The group is {E, C4, C2, C4^3, i, S4, sigma_h, S4^3} = 8 elements
# Wait - 4/m actually HAS 8 elements! The "expected 4" was wrong.

# Let me verify:
# 4/m = C4h in Schoenflies notation
# Elements: E, C4, C4^2=C2, C4^3, i, S4, σh, S4^3
# That's 8 elements, not 4!

# Similarly, 6/m = C6h has 12 elements:
# E, C6, C3, C2, C3^2, C6^5, i, S6, S3, σh, S3^5, S6^5

# CONCLUSION: The operation counts are CORRECT!
# The test expectations were wrong.
```

**Action**: Update test expectations, not the symmetry code.

**Files to Modify**:
- `tests/test_geometry.py` - Fix expected operation counts

---

## Performance Optimizations

### 3. Vertex Deduplication: O(n²) → O(n log n)

**Current Implementation** (geometry.py, ~line 168):
```python
# Naive pairwise comparison
unique = []
for v in vertices:
    is_dup = False
    for u in unique:
        if np.allclose(v, u, atol=1e-8):
            is_dup = True
            break
    if not is_dup:
        unique.append(v)
```

**Proposed Fix**: Use spatial indexing
```python
from scipy.spatial import cKDTree

def deduplicate_vertices(vertices: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Remove duplicate vertices using KD-tree."""
    if len(vertices) == 0:
        return vertices

    tree = cKDTree(vertices)
    # Find all pairs within tolerance
    pairs = tree.query_pairs(r=tol)

    # Build keep mask
    keep = np.ones(len(vertices), dtype=bool)
    for i, j in pairs:
        if keep[i] and keep[j]:
            keep[j] = False  # Remove duplicate

    return vertices[keep]
```

**Impact**: For 48 vertices, reduces from ~2300 comparisons to ~200.

---

### 4. Reciprocal Lattice Caching

**Current**: Recomputed for each form in non-cubic systems.

**Proposed Fix**:
```python
@lru_cache(maxsize=32)
def get_reciprocal_lattice(lattice: LatticeParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and cache reciprocal lattice vectors."""
    # Build direct lattice vectors
    a_vec = np.array([lattice.a, 0, 0])
    b_vec = np.array([
        lattice.b * np.cos(lattice.gamma),
        lattice.b * np.sin(lattice.gamma),
        0
    ])
    # ... compute c_vec ...

    # Volume
    V = np.dot(a_vec, np.cross(b_vec, c_vec))

    # Reciprocal vectors
    a_star = np.cross(b_vec, c_vec) / V
    b_star = np.cross(c_vec, a_vec) / V
    c_star = np.cross(a_vec, b_vec) / V

    return a_star, b_star, c_star
```

**Files to Modify**:
- `src/crystal_geometry/symmetry.py` (miller_to_normal function)

---

### 5. Configurable Numerical Tolerances

**Current**: Hardcoded magic numbers scattered across codebase.

```python
# Current scattered tolerances:
1e-8   # Vertex deduplication (geometry.py)
1e-6   # Vertex on plane (geometry.py)
1e-10  # Group element comparison (symmetry.py)
```

**Proposed Fix**: Centralized tolerance configuration
```python
# In models.py or new config.py
@dataclass
class GeometryConfig:
    vertex_tolerance: float = 1e-8
    plane_tolerance: float = 1e-6
    matrix_tolerance: float = 1e-10

    @classmethod
    def high_precision(cls) -> 'GeometryConfig':
        return cls(1e-12, 1e-10, 1e-14)

    @classmethod
    def fast(cls) -> 'GeometryConfig':
        return cls(1e-6, 1e-4, 1e-8)

# Usage:
config = GeometryConfig()
deduplicate_vertices(vertices, tol=config.vertex_tolerance)
```

---

## Type Safety Improvements

### 6. Fix 26 mypy Errors

**Categories of Errors**:

1. **Missing ndarray generics** (12 errors):
   ```python
   # Current
   def foo(arr: np.ndarray) -> np.ndarray:

   # Fixed
   from numpy.typing import NDArray
   def foo(arr: NDArray[np.float64]) -> NDArray[np.float64]:
   ```

2. **No scipy stubs** (8 errors):
   ```python
   # Add to pyproject.toml
   [tool.mypy]
   [[tool.mypy.overrides]]
   module = "scipy.*"
   ignore_missing_imports = true
   ```

3. **Incomplete return types** (6 errors):
   ```python
   # Current
   def get_edges(self):

   # Fixed
   def get_edges(self) -> Set[Tuple[int, int]]:
   ```

---

## CDL v2 Preparation

### 7. Modification Support Infrastructure

**New file**: `src/crystal_geometry/modifications.py`

```python
"""Morphological modification operations for crystal geometry."""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from .models import CrystalGeometry

@dataclass
class Modification:
    """Base class for geometry modifications."""
    pass

@dataclass
class Elongate(Modification):
    """Stretch geometry along an axis."""
    axis: Literal['a', 'b', 'c']
    ratio: float

@dataclass
class Truncate(Modification):
    """Cut geometry with additional plane."""
    miller: Tuple[int, int, int]
    depth: float

def apply_modification(geom: CrystalGeometry, mod: Modification) -> CrystalGeometry:
    """Apply a morphological modification to crystal geometry."""
    if isinstance(mod, Elongate):
        return _apply_elongate(geom, mod)
    elif isinstance(mod, Truncate):
        return _apply_truncate(geom, mod)
    else:
        raise ValueError(f"Unknown modification type: {type(mod)}")

def _apply_elongate(geom: CrystalGeometry, mod: Elongate) -> CrystalGeometry:
    """Scale vertices along specified axis."""
    axis_map = {'a': 0, 'b': 1, 'c': 2}
    axis_idx = axis_map[mod.axis]

    new_vertices = geom.vertices.copy()
    new_vertices[:, axis_idx] *= mod.ratio

    # Recalculate face normals after stretching
    new_normals = [_recalc_normal(new_vertices, face) for face in geom.faces]

    return CrystalGeometry(
        vertices=new_vertices,
        faces=geom.faces,
        face_normals=new_normals,
        face_forms=geom.face_forms,
        face_millers=geom.face_millers,
        forms=geom.forms
    )

def _apply_truncate(geom: CrystalGeometry, mod: Truncate) -> CrystalGeometry:
    """Add truncation plane and recompute geometry."""
    # This is more complex - needs to:
    # 1. Add new half-space constraint
    # 2. Recompute intersection
    # 3. Rebuild face topology
    raise NotImplementedError("Truncation requires geometry rebuild")
```

---

### 8. Twinning Support Infrastructure

**New file**: `src/crystal_geometry/twinning.py`

```python
"""Crystal twinning operations."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np
from .models import CrystalGeometry

# Twin law rotation matrices
TWIN_LAWS = {
    'spinel': {
        'axis': np.array([1, 1, 1]) / np.sqrt(3),
        'angle': 180.0,  # degrees
        'type': 'contact'
    },
    'brazil': {
        'axis': np.array([1, 0, 0]),
        'angle': 180.0,
        'type': 'penetration'
    },
    'japan': {
        'axis': np.array([1, 1, -2, 2]),  # Miller-Bravais
        'angle': 84.55,  # degrees
        'type': 'contact'
    },
    'dauphine': {
        'axis': np.array([0, 0, 1]),
        'angle': 180.0,
        'type': 'penetration'
    },
    # ... more laws
}

def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Create rotation matrix for axis-angle rotation."""
    angle = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def apply_twin(
    geom: CrystalGeometry,
    law: str,
    count: int = 2
) -> List[CrystalGeometry]:
    """Generate twinned crystal components."""
    if law not in TWIN_LAWS:
        raise ValueError(f"Unknown twin law: {law}")

    twin_info = TWIN_LAWS[law]
    R = rotation_matrix(twin_info['axis'], twin_info['angle'])

    components = [geom]  # Original

    for i in range(1, count):
        angle = twin_info['angle'] * i / (count - 1) if count > 1 else twin_info['angle']
        R_i = rotation_matrix(twin_info['axis'], angle)
        twinned = geom.rotate(R_i)
        components.append(twinned)

    return components
```

---

### 9. Aggregate Support Infrastructure

**New file**: `src/crystal_geometry/aggregates.py`

```python
"""Crystal aggregate assembly."""

from dataclasses import dataclass
from typing import List, Literal
import numpy as np
from .models import CrystalGeometry

@dataclass
class AggregateSpec:
    """Specification for crystal aggregate."""
    arrangement: Literal['parallel', 'random', 'radial', 'cluster', 'druse']
    count: int
    spacing: float = 1.0
    orientation_variance: float = 0.0  # 0 = perfect alignment, 1 = random

def create_aggregate(
    base: CrystalGeometry,
    spec: AggregateSpec
) -> List[CrystalGeometry]:
    """Generate aggregate of crystal individuals."""

    if spec.arrangement == 'parallel':
        return _parallel_aggregate(base, spec)
    elif spec.arrangement == 'radial':
        return _radial_aggregate(base, spec)
    elif spec.arrangement == 'cluster':
        return _cluster_aggregate(base, spec)
    elif spec.arrangement == 'druse':
        return _druse_aggregate(base, spec)
    else:
        raise ValueError(f"Unknown arrangement: {spec.arrangement}")

def _parallel_aggregate(base: CrystalGeometry, spec: AggregateSpec) -> List[CrystalGeometry]:
    """Create parallel-aligned crystal aggregate."""
    components = []
    for i in range(spec.count):
        offset = np.array([0, 0, i * spec.spacing])

        # Add small random rotation if variance > 0
        if spec.orientation_variance > 0:
            R = _random_rotation(spec.orientation_variance)
            crystal = base.rotate(R).translate(offset)
        else:
            crystal = base.translate(offset)

        components.append(crystal)

    return components

def _radial_aggregate(base: CrystalGeometry, spec: AggregateSpec) -> List[CrystalGeometry]:
    """Create radially-arranged crystal aggregate."""
    components = []
    for i in range(spec.count):
        angle = 2 * np.pi * i / spec.count
        R = rotation_matrix(np.array([0, 0, 1]), np.degrees(angle))
        offset = np.array([spec.spacing * np.cos(angle), spec.spacing * np.sin(angle), 0])

        crystal = base.rotate(R).translate(offset)
        components.append(crystal)

    return components
```

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 🔴 P0 | Fix hexagonal geometry | 2-3 days | Unblocks hex/trig systems |
| 🔴 P0 | Fix point group test expectations | 1 hour | Fixes false test failures |
| 🟡 P1 | Add type hints (mypy fixes) | 1 day | Code quality |
| 🟡 P1 | Vertex deduplication optimization | 2 hours | Performance |
| 🟢 P2 | Reciprocal lattice caching | 1 hour | Performance |
| 🟢 P2 | Configurable tolerances | 2 hours | Flexibility |
| 🔵 P3 | Modifications infrastructure | 1 week | CDL v2 |
| 🔵 P3 | Twinning infrastructure | 1 week | CDL v2 |
| 🔵 P3 | Aggregates infrastructure | 1 week | CDL v2 |

---

## Testing Checklist

After implementing fixes:

- [ ] All 29 existing tests pass
- [ ] `hexagonal[6/mmm]:{10-10}@1.0` generates valid geometry
- [ ] `trigonal[-3m]:{10-11}@1.0` generates valid geometry
- [ ] `4/m` and `6/m` operation counts match expectations
- [ ] mypy passes with zero errors
- [ ] No performance regression (benchmark before/after)
- [ ] New modification tests added
- [ ] New twinning tests added
- [ ] New aggregate tests added

---

*Document created: 2026-01-20*
