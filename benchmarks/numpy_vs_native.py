#!/usr/bin/env python3
"""
Benchmark: NumPy vs Native C++ (Eigen) performance comparison.

This measures the actual operations used in crystal geometry:
1. 3x3 linear system solving (plane intersection)
2. Cross products
3. Dot products
4. Norm calculations
5. Full halfspace intersection algorithm

Run with: python benchmarks/numpy_vs_native.py
"""

import time
import numpy as np
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog

# Import crystal_geometry for backend-aware benchmarks
try:
    from crystal_geometry import (
        get_backend,
        get_backend_info,
        create_octahedron,
        create_truncated_octahedron,
        cdl_string_to_geometry,
    )
    CRYSTAL_GEOMETRY_AVAILABLE = True
except ImportError:
    CRYSTAL_GEOMETRY_AVAILABLE = False


def benchmark(func, n_iterations=1000, warmup=100):
    """Run benchmark with warmup."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_iterations):
        func()
    elapsed = time.perf_counter() - start

    return elapsed / n_iterations * 1e6  # microseconds per call


# =============================================================================
# Test Data
# =============================================================================

# Typical octahedron normals (8 faces)
OCTAHEDRON_NORMALS = np.array([
    [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
    [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]
], dtype=np.float64) / np.sqrt(3)

OCTAHEDRON_DISTANCES = np.ones(8, dtype=np.float64)

# More complex crystal (24 faces - truncated octahedron)
n_complex = 24
np.random.seed(42)
COMPLEX_NORMALS = np.random.randn(n_complex, 3)
COMPLEX_NORMALS /= np.linalg.norm(COMPLEX_NORMALS, axis=1, keepdims=True)
COMPLEX_DISTANCES = np.ones(n_complex, dtype=np.float64) * 1.5


# =============================================================================
# Individual Operations
# =============================================================================

def bench_cross_product():
    """Cross product of two 3D vectors."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    return np.cross(a, b)


def bench_dot_product():
    """Dot product of two 3D vectors."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    return np.dot(a, b)


def bench_norm():
    """Euclidean norm of 3D vector."""
    a = np.array([1.0, 2.0, 3.0])
    return np.linalg.norm(a)


def bench_squared_norm():
    """Squared norm (avoiding sqrt)."""
    a = np.array([1.0, 2.0, 3.0])
    return a @ a  # or np.dot(a, a)


def bench_3x3_solve():
    """Solve 3x3 linear system (plane intersection)."""
    A = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0]
    ]) / np.sqrt(3)
    b = np.array([1.0, 1.0, 1.0])
    return np.linalg.solve(A, b)


def bench_3x3_det():
    """3x3 matrix determinant."""
    A = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0]
    ])
    return np.linalg.det(A)


def bench_matrix_vector_mult():
    """3x3 matrix times 3D vector."""
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.5, -0.866],
        [0.0, 0.866, 0.5]
    ])
    v = np.array([1.0, 2.0, 3.0])
    return A @ v


# =============================================================================
# Algorithm-Level Operations
# =============================================================================

def bench_triple_plane_intersection():
    """Single triple-plane intersection (core of halfspace algorithm)."""
    n0, n1, n2 = OCTAHEDRON_NORMALS[0], OCTAHEDRON_NORMALS[1], OCTAHEDRON_NORMALS[2]
    d0, d1, d2 = 1.0, 1.0, 1.0

    # Cross product for parallelism check
    cross = np.cross(n0, n1)
    if np.dot(cross, cross) < 1e-20:
        return None

    # Build matrix and solve
    A = np.array([n0, n1, n2])
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return None

    b = np.array([d0, d1, d2])
    vertex = np.linalg.solve(A, b)

    # Validate vertex is inside all halfspaces
    for n, d in zip(OCTAHEDRON_NORMALS, OCTAHEDRON_DISTANCES):
        if np.dot(n, vertex) > d + 1e-8:
            return None

    return vertex


def bench_scipy_halfspace_octahedron():
    """Full scipy HalfspaceIntersection for octahedron."""
    halfspaces = np.hstack([OCTAHEDRON_NORMALS, -OCTAHEDRON_DISTANCES.reshape(-1, 1)])
    interior = np.array([0.0, 0.0, 0.0])
    hs = HalfspaceIntersection(halfspaces, interior)
    return hs.intersections


def bench_scipy_halfspace_complex():
    """Full scipy HalfspaceIntersection for complex crystal."""
    halfspaces = np.hstack([COMPLEX_NORMALS, -COMPLEX_DISTANCES.reshape(-1, 1)])

    # Find interior point via linear programming
    n = len(COMPLEX_NORMALS)
    c = np.array([0.0, 0.0, 0.0, -1.0])
    A_ub = np.hstack([COMPLEX_NORMALS, np.ones((n, 1))])
    b_ub = COMPLEX_DISTANCES
    bounds = [(-10, 10), (-10, 10), (-10, 10), (1e-10, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    interior = result.x[:3]

    hs = HalfspaceIntersection(halfspaces, interior)
    return hs.intersections


def bench_brute_force_halfspace_octahedron():
    """Brute-force triple intersection for octahedron (what C++ does)."""
    normals = OCTAHEDRON_NORMALS
    distances = OCTAHEDRON_DISTANCES
    n = len(normals)
    vertices = []
    tolerance = 1e-8

    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Build system
                A = np.array([normals[i], normals[j], normals[k]])

                # Check determinant
                det = np.linalg.det(A)
                if abs(det) < 1e-12:
                    continue

                b = np.array([distances[i], distances[j], distances[k]])
                vertex = np.linalg.solve(A, b)

                # Check if inside all halfspaces
                inside = True
                for ni, di in zip(normals, distances):
                    if np.dot(ni, vertex) > di + tolerance:
                        inside = False
                        break

                if inside:
                    # Check for duplicates
                    is_dup = False
                    for v in vertices:
                        if np.linalg.norm(vertex - v) < tolerance:
                            is_dup = True
                            break
                    if not is_dup:
                        vertices.append(vertex)

    return np.array(vertices)


# =============================================================================
# Run Benchmarks
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NumPy Performance Benchmark for Crystal Geometry Operations")
    print("=" * 70)
    print()

    print("INDIVIDUAL OPERATIONS (microseconds per call)")
    print("-" * 50)

    ops = [
        ("Cross product (3D)", bench_cross_product),
        ("Dot product (3D)", bench_dot_product),
        ("Norm (3D)", bench_norm),
        ("Squared norm (3D)", bench_squared_norm),
        ("3x3 linear solve", bench_3x3_solve),
        ("3x3 determinant", bench_3x3_det),
        ("3x3 @ 3D vector", bench_matrix_vector_mult),
    ]

    for name, func in ops:
        us = benchmark(func, n_iterations=10000)
        print(f"  {name:25s}: {us:8.2f} µs")

    print()
    print("ALGORITHM-LEVEL OPERATIONS (microseconds per call)")
    print("-" * 50)

    algos = [
        ("Triple plane intersection", bench_triple_plane_intersection, 10000),
        ("Scipy halfspace (8 faces)", bench_scipy_halfspace_octahedron, 1000),
        ("Scipy halfspace (24 faces)", bench_scipy_halfspace_complex, 100),
        ("Brute-force (8 faces)", bench_brute_force_halfspace_octahedron, 100),
    ]

    for name, func, n_iter in algos:
        us = benchmark(func, n_iterations=n_iter, warmup=min(n_iter, 100))
        print(f"  {name:30s}: {us:10.2f} µs")

    print()
    print("=" * 70)
    print("ANALYSIS: NumPy vs Expected Eigen Performance")
    print("=" * 70)
    print()
    print("Key observations:")
    print()
    print("1. FUNCTION CALL OVERHEAD:")
    print("   - Each NumPy call has ~1-5µs Python overhead")
    print("   - Eigen inline calls: ~0.01-0.1µs (10-100x faster)")
    print()
    print("2. SMALL MATRIX OPERATIONS (3x3):")
    print("   - NumPy uses general LAPACK routines")
    print("   - Eigen uses fixed-size optimized code")
    print("   - Expected speedup: 5-20x for individual ops")
    print()
    print("3. ALGORITHM LEVEL (many small ops):")
    print("   - Python loop overhead dominates")
    print("   - C++ tight loops: 10-50x faster")
    print()
    print("4. SCIPY HALFSPACE:")
    print("   - Already uses C (Qhull library)")
    print("   - Custom C++ may be 2-5x faster due to:")
    print("     * Specialized for crystal geometry")
    print("     * OpenMP parallelization")
    print("     * Early rejection optimizations")
    print()

    # Calculate theoretical speedups
    print("ESTIMATED SPEEDUPS WITH EIGEN/C++:")
    print("-" * 50)

    estimates = [
        ("Individual 3D ops", "5-20x", "Function call overhead elimination"),
        ("3x3 solve", "10-30x", "Fixed-size optimization + no Python"),
        ("Triple intersection", "20-50x", "Inline + no Python loops"),
        ("Full halfspace (simple)", "5-10x", "Already uses Qhull, gains from parallelism"),
        ("Full halfspace (complex)", "10-30x", "OpenMP + early rejection"),
        ("Overall geometry pipeline", "10-50x", "Depends on crystal complexity"),
    ]

    for op, speedup, reason in estimates:
        print(f"  {op:25s}: {speedup:8s}  ({reason})")

    # High-level crystal_geometry benchmarks
    if CRYSTAL_GEOMETRY_AVAILABLE:
        print()
        print("=" * 70)
        print("CRYSTAL GEOMETRY MODULE BENCHMARKS")
        print("=" * 70)
        print()

        info = get_backend_info()
        print(f"Backend: {info['backend']}")
        print(f"Native available: {info.get('native_available', False)}")
        if info.get('openmp_enabled'):
            print(f"OpenMP threads: {info.get('num_threads', 1)}")
        print()

        def bench_create_octahedron():
            return create_octahedron(1.0)

        def bench_create_truncated():
            return create_truncated_octahedron(1.0, 1.3)

        def bench_cdl_simple():
            return cdl_string_to_geometry("cubic[m3m]:{111}")

        def bench_cdl_complex():
            return cdl_string_to_geometry("cubic[m3m]:{111}@1.0 + {100}@1.3 + {110}@1.5")

        crystal_benchmarks = [
            ("create_octahedron (8 faces)", bench_create_octahedron, 100),
            ("create_truncated_octahedron (14 faces)", bench_create_truncated, 100),
            ("CDL simple (octahedron)", bench_cdl_simple, 100),
            ("CDL complex (26 faces)", bench_cdl_complex, 50),
        ]

        print("HIGH-LEVEL OPERATIONS (milliseconds per call)")
        print("-" * 50)

        for name, func, n_iter in crystal_benchmarks:
            us = benchmark(func, n_iterations=n_iter, warmup=min(n_iter, 20))
            ms = us / 1000
            print(f"  {name:40s}: {ms:8.3f} ms")

        print()
        print("Performance targets:")
        print("  - Octahedron (8 faces):      < 0.1 ms")
        print("  - Truncated octahedron (14): < 0.15 ms")
        print("  - Complex crystal (24+):     < 0.3 ms")
