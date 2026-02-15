"""Parametric mesh generators for amorphous (non-crystalline) shapes.

Generates simple educational-quality meshes for amorphous materials like
opal, obsidian, and chalcedony. Each shape returns a CrystalGeometry
with is_amorphous=True and empty face_forms/face_millers.
"""

from __future__ import annotations

import numpy as np

from .models import CrystalGeometry


def _icosphere(subdivisions: int = 2, radius: float = 1.0) -> tuple[np.ndarray, list[list[int]]]:
    """Generate an icosphere by subdividing an icosahedron.

    Args:
        subdivisions: Number of subdivision iterations
        radius: Sphere radius

    Returns:
        Tuple of (vertices, faces)
    """
    # Golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedron vertices
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    vertices = np.array(verts, dtype=np.float64)
    # Normalise to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])

    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    # Subdivide
    for _ in range(subdivisions):
        mid_cache: dict[tuple[int, int], int] = {}
        new_faces = []
        for tri in faces:
            mids = []
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                key = (min(a, b), max(a, b))
                if key not in mid_cache:
                    mid = (vertices[a] + vertices[b]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    mid_cache[key] = len(vertices)
                    vertices = np.vstack([vertices, mid])
                mids.append(mid_cache[key])
            new_faces.append([tri[0], mids[0], mids[2]])
            new_faces.append([tri[1], mids[1], mids[0]])
            new_faces.append([tri[2], mids[2], mids[1]])
            new_faces.append([mids[0], mids[1], mids[2]])
        faces = new_faces

    vertices = vertices * radius
    return vertices, faces


def _compute_face_normals(
    vertices: np.ndarray, faces: list[list[int]]
) -> list[np.ndarray]:
    """Compute outward-pointing normal for each face."""
    normals = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(n)
        if norm_len > 1e-10:
            n = n / norm_len
        else:
            n = np.array([0.0, 0.0, 1.0])
        normals.append(n)
    return normals


def _make_amorphous_geometry(
    vertices: np.ndarray, faces: list[list[int]]
) -> CrystalGeometry:
    """Wrap raw mesh data into a CrystalGeometry with amorphous flags."""
    face_normals = _compute_face_normals(vertices, faces)
    return CrystalGeometry(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        face_forms=[0] * len(faces),
        face_millers=[(0, 0, 0)] * len(faces),
        forms=[],
        is_amorphous=True,
    )


def generate_massive(radius: float = 1.0, seed: int = 42) -> CrystalGeometry:
    """Generate a massive (irregular lump) shape via perturbed icosphere.

    Args:
        radius: Base radius
        seed: Random seed for reproducibility

    Returns:
        CrystalGeometry with is_amorphous=True
    """
    rng = np.random.default_rng(seed)
    vertices, faces = _icosphere(subdivisions=2, radius=radius)
    # Perturb vertices radially
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    directions = vertices / norms
    perturbation = rng.normal(0, 0.08 * radius, size=(len(vertices), 1))
    vertices = vertices + directions * perturbation
    return _make_amorphous_geometry(vertices, faces)


def generate_botryoidal(radius: float = 1.0, seed: int = 42) -> CrystalGeometry:
    """Generate a botryoidal (grape-like) shape from overlapping spheres.

    Args:
        radius: Overall radius
        seed: Random seed for reproducibility

    Returns:
        CrystalGeometry with is_amorphous=True
    """
    rng = np.random.default_rng(seed)
    n_spheres = rng.integers(3, 6)
    all_verts: list[np.ndarray] = []
    all_faces: list[list[int]] = []
    offset = 0

    for _ in range(n_spheres):
        sphere_r = radius * rng.uniform(0.3, 0.6)
        center = rng.uniform(-0.4, 0.4, size=3) * radius
        verts, faces = _icosphere(subdivisions=2, radius=sphere_r)
        verts = verts + center
        # Offset face indices
        offset_faces = [[idx + offset for idx in f] for f in faces]
        all_verts.append(verts)
        all_faces.extend(offset_faces)
        offset += len(verts)

    vertices = np.vstack(all_verts)
    return _make_amorphous_geometry(vertices, all_faces)


def generate_reniform(radius: float = 1.0, seed: int = 42) -> CrystalGeometry:
    """Generate a reniform (kidney-shaped) deformed ellipsoid.

    Args:
        radius: Overall radius
        seed: Random seed for reproducibility

    Returns:
        CrystalGeometry with is_amorphous=True
    """
    rng = np.random.default_rng(seed)
    vertices, faces = _icosphere(subdivisions=2, radius=radius)
    # Deform into ellipsoid, flattened on one side
    vertices[:, 0] *= 1.4  # elongate x
    vertices[:, 1] *= 0.8  # compress y
    vertices[:, 2] *= 0.6  # flatten z
    # Add kidney indentation on one side
    for i in range(len(vertices)):
        x = vertices[i, 0]
        if x > 0:
            indent = 0.15 * radius * np.exp(-((x / radius) ** 2) * 2)
            vertices[i, 1] -= indent
    # Small random perturbation
    vertices += rng.normal(0, 0.03 * radius, size=vertices.shape)
    return _make_amorphous_geometry(vertices, faces)


def generate_stalactitic(radius: float = 1.0, seed: int = 42) -> CrystalGeometry:
    """Generate a stalactitic (tapered cylinder) shape.

    Args:
        radius: Base radius of the cylinder
        seed: Random seed for reproducibility

    Returns:
        CrystalGeometry with is_amorphous=True
    """
    rng = np.random.default_rng(seed)
    n_rings = 12
    n_segments = 16
    height = radius * 3.0
    all_verts = []

    for ring_i in range(n_rings):
        t = ring_i / (n_rings - 1)  # 0 to 1, top to bottom
        z = height * (0.5 - t)
        # Taper: wider at top, narrower at bottom
        r = radius * (1.0 - 0.7 * t)
        for seg_i in range(n_segments):
            angle = 2 * np.pi * seg_i / n_segments
            x = r * np.cos(angle) + rng.normal(0, 0.02 * radius)
            y = r * np.sin(angle) + rng.normal(0, 0.02 * radius)
            all_verts.append([x, y, z])

    vertices = np.array(all_verts, dtype=np.float64)

    # Build faces connecting rings
    faces: list[list[int]] = []
    for ring_i in range(n_rings - 1):
        for seg_i in range(n_segments):
            next_seg = (seg_i + 1) % n_segments
            a = ring_i * n_segments + seg_i
            b = ring_i * n_segments + next_seg
            c = (ring_i + 1) * n_segments + next_seg
            d = (ring_i + 1) * n_segments + seg_i
            faces.append([a, b, c, d])

    # Cap top
    top_center_idx = len(vertices)
    top_center = np.mean(vertices[:n_segments], axis=0)
    vertices = np.vstack([vertices, top_center])
    for seg_i in range(n_segments):
        next_seg = (seg_i + 1) % n_segments
        faces.append([top_center_idx, seg_i, next_seg])

    # Cap bottom (point)
    bottom_idx = len(vertices)
    bottom_point = np.array([0.0, 0.0, -height * 0.5 - radius * 0.1])
    vertices = np.vstack([vertices, bottom_point])
    base_ring_start = (n_rings - 1) * n_segments
    for seg_i in range(n_segments):
        next_seg = (seg_i + 1) % n_segments
        faces.append([bottom_idx, base_ring_start + next_seg, base_ring_start + seg_i])

    return _make_amorphous_geometry(vertices, faces)


def generate_mammillary(radius: float = 1.0, seed: int = 42) -> CrystalGeometry:
    """Generate a mammillary (hemisphere/dome) shape.

    Args:
        radius: Hemisphere radius
        seed: Random seed for reproducibility

    Returns:
        CrystalGeometry with is_amorphous=True
    """
    rng = np.random.default_rng(seed)
    vertices, faces = _icosphere(subdivisions=2, radius=radius)
    # Keep only the upper hemisphere
    keep_mask = vertices[:, 2] >= -0.05 * radius
    # Remap vertex indices
    idx_map = np.full(len(vertices), -1, dtype=int)
    new_idx = 0
    for i in range(len(vertices)):
        if keep_mask[i]:
            idx_map[i] = new_idx
            new_idx += 1
    new_verts = vertices[keep_mask]

    # Filter and remap faces
    new_faces = []
    for face in faces:
        if all(keep_mask[v] for v in face):
            new_faces.append([idx_map[v] for v in face])

    # Add a flat base
    base_verts = new_verts[new_verts[:, 2] < 0.1 * radius]
    if len(base_verts) > 2:
        # Add a center point for the base
        center_idx = len(new_verts)
        center = np.array([0.0, 0.0, 0.0])
        new_verts = np.vstack([new_verts, center])

        # Find boundary vertices (close to z=0)
        boundary = []
        for i in range(len(new_verts) - 1):
            if abs(new_verts[i, 2]) < 0.1 * radius:
                boundary.append(i)

        if len(boundary) >= 3:
            # Sort boundary by angle
            angles = np.arctan2(
                new_verts[boundary, 1], new_verts[boundary, 0]
            )
            boundary = [boundary[i] for i in np.argsort(angles)]
            for i in range(len(boundary)):
                j = (i + 1) % len(boundary)
                new_faces.append([center_idx, boundary[j], boundary[i]])

    # Small perturbation
    new_verts += rng.normal(0, 0.02 * radius, size=new_verts.shape)
    return _make_amorphous_geometry(new_verts, new_faces)


def generate_nodular(radius: float = 1.0, seed: int = 42) -> CrystalGeometry:
    """Generate a nodular (lumpy ellipsoid with noise) shape.

    Args:
        radius: Base radius
        seed: Random seed for reproducibility

    Returns:
        CrystalGeometry with is_amorphous=True
    """
    rng = np.random.default_rng(seed)
    vertices, faces = _icosphere(subdivisions=2, radius=radius)
    # Make it slightly ellipsoidal
    vertices[:, 0] *= rng.uniform(0.9, 1.2)
    vertices[:, 1] *= rng.uniform(0.8, 1.1)
    vertices[:, 2] *= rng.uniform(0.7, 1.0)
    # Add larger bumps
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    directions = vertices / norms
    perturbation = rng.normal(0, 0.12 * radius, size=(len(vertices), 1))
    vertices = vertices + directions * perturbation
    return _make_amorphous_geometry(vertices, faces)


def generate_conchoidal(radius: float = 1.0, seed: int = 42) -> CrystalGeometry:
    """Generate a conchoidal (smooth sphere fallback) shape.

    Used as a simple fallback for conchoidal fracture surfaces.

    Args:
        radius: Sphere radius
        seed: Random seed (unused, included for API consistency)

    Returns:
        CrystalGeometry with is_amorphous=True
    """
    vertices, faces = _icosphere(subdivisions=2, radius=radius)
    return _make_amorphous_geometry(vertices, faces)


# Registry mapping shape names to generators
AMORPHOUS_GENERATORS: dict[str, type[None] | None] = {}  # Placeholder for type

_SHAPE_GENERATORS = {
    "massive": generate_massive,
    "botryoidal": generate_botryoidal,
    "reniform": generate_reniform,
    "stalactitic": generate_stalactitic,
    "mammillary": generate_mammillary,
    "nodular": generate_nodular,
    "conchoidal": generate_conchoidal,
}


def generate_amorphous_shape(
    shape: str, radius: float = 1.0, seed: int = 42
) -> CrystalGeometry:
    """Generate geometry for a named amorphous shape.

    Args:
        shape: Shape name ('massive', 'botryoidal', 'reniform', etc.)
        radius: Base radius for the shape
        seed: Random seed for reproducibility

    Returns:
        CrystalGeometry with is_amorphous=True

    Raises:
        ValueError: If shape name is not recognised
    """
    generator = _SHAPE_GENERATORS.get(shape)
    if generator is None:
        valid = ", ".join(sorted(_SHAPE_GENERATORS))
        raise ValueError(f"Unknown amorphous shape '{shape}'. Valid shapes: {valid}")
    return generator(radius=radius, seed=seed)
