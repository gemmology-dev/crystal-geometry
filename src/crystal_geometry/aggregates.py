"""Aggregate layout algorithms for crystal aggregates.

Provides 6 spatial arrangement algorithms that place multiple copies of a
base crystal geometry into aggregate formations. Each algorithm produces
a list of 4x4 affine transforms which are applied to the base geometry.
"""

from __future__ import annotations

import numpy as np

from .models import AggregateMetadata, CrystalGeometry

# Maximum number of instances in an aggregate (performance guard)
MAX_INSTANCES = 200


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from axis and angle (radians)."""
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a random 3x3 rotation matrix."""
    # QR decomposition of random matrix gives uniform rotation
    z = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(z)
    # Ensure proper rotation (det = +1)
    d = np.diag(np.sign(np.diag(r)))
    q = q @ d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Combine 3x3 rotation and 3D translation into 4x4 affine transform."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def layout_parallel(count: int, spacing: float = 1.0, seed: int = 42) -> list[np.ndarray]:
    """Parallel layout: translate along c-axis with even spacing.

    Args:
        count: Number of instances
        spacing: Distance between instances along z-axis
        seed: Random seed (unused, deterministic layout)

    Returns:
        List of 4x4 affine transform matrices
    """
    transforms = []
    # Center the arrangement around origin
    total_length = (count - 1) * spacing
    start_z = -total_length / 2.0
    for i in range(count):
        z = start_z + i * spacing
        transforms.append(_make_transform(np.eye(3), np.array([0.0, 0.0, z])))
    return transforms


def layout_random(count: int, bounding_radius: float = 2.0, seed: int = 42) -> list[np.ndarray]:
    """Random layout: random rotation and position within bounding sphere.

    Args:
        count: Number of instances
        bounding_radius: Radius of bounding sphere for positions
        seed: Random seed for reproducibility

    Returns:
        List of 4x4 affine transform matrices
    """
    rng = np.random.default_rng(seed)
    transforms = []
    for _ in range(count):
        rot = _random_rotation(rng)
        # Random position inside sphere (uniform in volume)
        r = bounding_radius * rng.uniform(0, 1) ** (1.0 / 3.0)
        direction = rng.standard_normal(3)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        pos = direction * r
        transforms.append(_make_transform(rot, pos))
    return transforms


def layout_radial(count: int, spread_radius: float = 2.0, seed: int = 42) -> list[np.ndarray]:
    """Radial layout: evenly rotated around origin, translated outward.

    Args:
        count: Number of instances
        spread_radius: Distance from origin to each instance
        seed: Random seed (unused, deterministic layout)

    Returns:
        List of 4x4 affine transform matrices
    """
    transforms = []
    for i in range(count):
        angle = 2.0 * np.pi * i / count
        rot = _rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), angle)
        pos = np.array([spread_radius * np.cos(angle), spread_radius * np.sin(angle), 0.0])
        transforms.append(_make_transform(rot, pos))
    return transforms


def layout_epitaxial(
    count: int, host_normals: np.ndarray | None = None, seed: int = 42
) -> list[np.ndarray]:
    """Epitaxial layout: placed on host face normals.

    If host_normals are not provided, uses evenly-distributed directions
    on a sphere.

    Args:
        count: Number of instances
        host_normals: Optional Nx3 array of host face normal directions
        seed: Random seed for reproducibility

    Returns:
        List of 4x4 affine transform matrices
    """
    rng = np.random.default_rng(seed)
    transforms = []

    if host_normals is not None and len(host_normals) >= count:
        # Use provided face normals, pick evenly-spaced subset
        indices = np.linspace(0, len(host_normals) - 1, count, dtype=int)
        normals = host_normals[indices]
    else:
        # Generate Fibonacci sphere points for even distribution
        normals = _fibonacci_sphere(count)

    for normal in normals:
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        # Position along normal
        pos = normal * 1.5
        # Align z-axis with normal
        rot = _align_z_to_direction(normal)
        # Small random rotation around the normal
        angle = rng.uniform(0, 2 * np.pi)
        spin = _rotation_matrix_from_axis_angle(normal, angle)
        rot = spin @ rot
        transforms.append(_make_transform(rot, pos))
    return transforms


def layout_druse(count: int, hemisphere_radius: float = 2.0, seed: int = 42) -> list[np.ndarray]:
    """Druse layout: crystals pointing outward on a hemisphere surface.

    Args:
        count: Number of instances
        hemisphere_radius: Radius of the hemisphere base
        seed: Random seed for reproducibility

    Returns:
        List of 4x4 affine transform matrices
    """
    rng = np.random.default_rng(seed)
    transforms = []
    # Use Fibonacci hemisphere for even distribution
    points = _fibonacci_sphere(count * 2)
    # Keep upper hemisphere only
    upper = points[points[:, 2] >= 0]
    if len(upper) < count:
        upper = points[:count]
    upper = upper[:count]

    for point in upper:
        normal = point / (np.linalg.norm(point) + 1e-10)
        pos = normal * hemisphere_radius
        rot = _align_z_to_direction(normal)
        # Small random spin
        angle = rng.uniform(0, 2 * np.pi)
        spin = _rotation_matrix_from_axis_angle(normal, angle)
        rot = spin @ rot
        transforms.append(_make_transform(rot, pos))
    return transforms


def layout_cluster(count: int, dome_radius: float = 2.0, seed: int = 42) -> list[np.ndarray]:
    """Cluster layout: random rotation and position on a dome.

    Args:
        count: Number of instances
        dome_radius: Radius of the dome
        seed: Random seed for reproducibility

    Returns:
        List of 4x4 affine transform matrices
    """
    rng = np.random.default_rng(seed)
    transforms = []
    for _ in range(count):
        rot = _random_rotation(rng)
        # Random position on upper hemisphere
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi / 2)
        x = dome_radius * np.sin(phi) * np.cos(theta)
        y = dome_radius * np.sin(phi) * np.sin(theta)
        z = dome_radius * np.cos(phi)
        pos = np.array([x, y, z])
        transforms.append(_make_transform(rot, pos))
    return transforms


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately evenly-spaced points on a sphere."""
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    indices = np.arange(n, dtype=np.float64)
    theta = 2.0 * np.pi * indices / golden_ratio
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _align_z_to_direction(direction: np.ndarray) -> np.ndarray:
    """Create a rotation matrix that aligns [0,0,1] to the given direction."""
    z = direction / (np.linalg.norm(direction) + 1e-10)
    # Find a vector not parallel to z
    if abs(z[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-10)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# Layout registry
_LAYOUT_ALGORITHMS = {
    "parallel": layout_parallel,
    "random": layout_random,
    "radial": layout_radial,
    "epitaxial": layout_epitaxial,
    "druse": layout_druse,
    "cluster": layout_cluster,
}


def generate_aggregate(
    base_geometry: CrystalGeometry,
    arrangement: str,
    count: int,
    spacing: float | None = None,
    orientation: str | None = None,
    seed: int = 42,
) -> CrystalGeometry:
    """Generate an aggregate from a base crystal geometry.

    Creates multiple copies of the base geometry arranged according to the
    specified layout algorithm, and concatenates them into a single
    CrystalGeometry with component_ids tracking individual crystals.

    Args:
        base_geometry: The crystal geometry to replicate
        arrangement: Layout algorithm name ('parallel', 'random', etc.)
        count: Number of instances (capped at MAX_INSTANCES)
        spacing: Optional spacing parameter for layout algorithms
        orientation: Optional orientation mode (currently unused, reserved)
        seed: Random seed for reproducibility

    Returns:
        CrystalGeometry with concatenated instances and component_ids

    Raises:
        ValueError: If arrangement name is not recognised
    """
    layout_fn = _LAYOUT_ALGORITHMS.get(arrangement)
    if layout_fn is None:
        valid = ", ".join(sorted(_LAYOUT_ALGORITHMS))
        raise ValueError(f"Unknown arrangement '{arrangement}'. Valid: {valid}")

    # Cap instance count
    count = min(count, MAX_INSTANCES)

    # Get transforms
    kwargs: dict[str, float | int] = {"count": count, "seed": seed}
    if spacing is not None and arrangement == "parallel":
        kwargs["spacing"] = spacing
    transforms = layout_fn(**kwargs)  # type: ignore[arg-type]

    # Apply transforms to base geometry and concatenate
    all_verts: list[np.ndarray] = []
    all_faces: list[list[int]] = []
    all_normals: list[np.ndarray] = []
    all_forms: list[int] = []
    all_millers: list[tuple[int, int, int]] = []
    component_ids: list[int] = []
    vertex_offset = 0

    for comp_id, transform in enumerate(transforms):
        rot = transform[:3, :3]
        translation = transform[:3, 3]

        # Transform vertices
        new_verts = base_geometry.vertices @ rot.T + translation
        all_verts.append(new_verts)

        # Offset face indices
        for face in base_geometry.faces:
            all_faces.append([idx + vertex_offset for idx in face])

        # Transform normals (rotation only)
        for normal in base_geometry.face_normals:
            all_normals.append(rot @ normal)

        # Copy form data
        all_forms.extend(base_geometry.face_forms)
        all_millers.extend(base_geometry.face_millers)
        component_ids.extend([comp_id] * len(base_geometry.faces))

        vertex_offset += len(base_geometry.vertices)

    vertices = np.vstack(all_verts) if all_verts else np.zeros((0, 3))

    return CrystalGeometry(
        vertices=vertices,
        faces=all_faces,
        face_normals=all_normals,
        face_forms=all_forms,
        face_millers=all_millers,
        forms=base_geometry.forms,
        component_ids=component_ids,
        is_amorphous=base_geometry.is_amorphous,
        aggregate_metadata=AggregateMetadata(
            arrangement=arrangement,
            n_instances=count,
            spacing=spacing,
            orientation=orientation,
        ),
    )
