/**
 * @file geometry.cpp
 * @brief Implementation of core geometry operations.
 */

#include "geometry.hpp"

namespace crystal_geometry {

std::pair<Vector3, bool> find_interior_point(
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances
) {
    // Simple approach: try centroid of halfspace intersection points
    // weighted by distances, then verify it's inside all halfspaces

    Eigen::Index n = normals.rows();
    Vector3 centroid = Vector3::Zero();
    Scalar total_weight = 0.0;

    for (Eigen::Index i = 0; i < n; ++i) {
        Vector3 normal = normals.row(i);
        Scalar dist = distances(i);
        // Point on plane in direction of normal
        centroid += normal * dist;
        total_weight += 1.0;
    }

    if (total_weight > 0) {
        centroid /= total_weight;
    }

    // Check if centroid is inside all halfspaces
    if (is_inside_all_halfspaces(centroid, normals, distances, 1e-10)) {
        return {centroid, true};
    }

    // Try scaling toward origin
    for (Scalar scale : {0.5, 0.3, 0.1, 0.05, 0.01}) {
        Vector3 test_point = centroid * scale;
        if (is_inside_all_halfspaces(test_point, normals, distances, 1e-10)) {
            return {test_point, true};
        }
    }

    // Try pure origin
    Vector3 origin = Vector3::Zero();
    if (is_inside_all_halfspaces(origin, normals, distances, 1e-10)) {
        return {origin, true};
    }

    return {Vector3::Zero(), false};
}

std::pair<Vector3, bool> find_interior_point_iterative(
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances
) {
    // Same as find_interior_point - consolidated implementation
    return find_interior_point(normals, distances);
}

bool is_inside_all_halfspaces(
    const Vector3& point,
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances,
    Scalar tolerance
) {
    for (Eigen::Index i = 0; i < normals.rows(); ++i) {
        Vector3 normal = normals.row(i);
        Scalar dist = distances(i);
        if (normal.dot(point) > dist - tolerance) {
            return false;
        }
    }
    return true;
}

HalfspaceResult halfspace_intersection(
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances,
    Scalar tolerance
) {
    HalfspaceResult result;
    result.success = false;

    Eigen::Index n = normals.rows();
    if (n < 4) {
        result.error_message = "Need at least 4 halfspaces for bounded polyhedron";
        return result;
    }

    const Scalar tolerance_sq = tolerance * tolerance;
    std::vector<Vector3> vertices;
    vertices.reserve(static_cast<std::size_t>(n * n));  // Upper bound estimate

    // Spatial hash for deduplication
    SpatialHashGrid grid(tolerance * 2);
    grid.set_tolerance(tolerance);
    std::size_t vertex_count = 0;

    // Triple-plane intersection loop
    // This is O(n³) but each iteration is very fast in C++
#ifdef CRYSTAL_USE_OPENMP
    std::mutex vertices_mutex;
    #pragma omp parallel for schedule(dynamic)
    for (Eigen::Index i = 0; i < n - 2; ++i) {
        for (Eigen::Index j = i + 1; j < n - 1; ++j) {
            for (Eigen::Index k = j + 1; k < n; ++k) {
#else
    for (Eigen::Index i = 0; i < n - 2; ++i) {
        for (Eigen::Index j = i + 1; j < n - 1; ++j) {
            for (Eigen::Index k = j + 1; k < n; ++k) {
#endif
                // Build 3x3 system from three plane normals
                Matrix3 A;
                A.row(0) = normals.row(i);
                A.row(1) = normals.row(j);
                A.row(2) = normals.row(k);

                // Check if planes are not parallel (determinant check)
                Scalar det = A.determinant();
                if (std::abs(det) < DETERMINANT_THRESHOLD) {
                    continue;
                }

                // Solve Ax = b for intersection point
                Vector3 b(distances(i), distances(j), distances(k));
                Vector3 vertex = A.fullPivLu().solve(b);

                // Verify solution is valid
                if ((A * vertex - b).squaredNorm() > tolerance_sq) {
                    continue;
                }

                // Check if inside all halfspaces
                bool inside = true;
                for (Eigen::Index m = 0; m < n; ++m) {
                    Vector3 nm = normals.row(m);
                    if (nm.dot(vertex) > distances(m) + tolerance) {
                        inside = false;
                        break;
                    }
                }

                if (inside) {
#ifdef CRYSTAL_USE_OPENMP
                    std::lock_guard<std::mutex> lock(vertices_mutex);
#endif
                    // Check for duplicates via spatial hash
                    if (grid.try_add(vertex, vertex_count)) {
                        vertices.push_back(vertex);
                        vertex_count++;
                    }
                }
            }
        }
    }

    if (vertices.size() < 4) {
        result.error_message = "Failed to find at least 4 vertices";
        return result;
    }

    // Convert to Eigen matrix
    result.vertices.resize(static_cast<Eigen::Index>(vertices.size()), 3);
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        result.vertices.row(static_cast<Eigen::Index>(i)) = vertices[i];
    }

    result.success = true;
    return result;
}

std::vector<int64_t> compute_face_vertices(
    const Eigen::Ref<const RowMatrixX3>& vertices,
    const Vector3& normal,
    Scalar distance,
    Scalar tolerance
) {
    std::vector<int64_t> on_face;
    const Scalar tolerance_sq = tolerance * tolerance;

    // Find vertices on the face plane
    for (Eigen::Index i = 0; i < vertices.rows(); ++i) {
        Vector3 v = vertices.row(i);
        Scalar d = normal.dot(v);
        if (std::abs(d - distance) < tolerance) {
            on_face.push_back(static_cast<int64_t>(i));
        }
    }

    if (on_face.size() < 3) {
        return {};
    }

    // Compute center of face vertices
    Vector3 center = Vector3::Zero();
    for (int64_t idx : on_face) {
        center += vertices.row(static_cast<Eigen::Index>(idx));
    }
    center /= static_cast<Scalar>(on_face.size());

    // Build local coordinate system on face
    Vector3 first_vert = vertices.row(static_cast<Eigen::Index>(on_face[0]));
    Vector3 u = first_vert - center;
    u = u - u.dot(normal) * normal;

    if (u.squaredNorm() < tolerance_sq) {
        if (on_face.size() > 1) {
            Vector3 second_vert = vertices.row(static_cast<Eigen::Index>(on_face[1]));
            u = second_vert - center;
            u = u - u.dot(normal) * normal;
        }
    }
    u.normalize();
    Vector3 v_axis = normal.cross(u);

    // Compute angles and sort
    std::vector<std::pair<Scalar, int64_t>> angles;
    angles.reserve(on_face.size());

    for (int64_t idx : on_face) {
        Vector3 vert = vertices.row(static_cast<Eigen::Index>(idx));
        Vector3 vec = vert - center;
        Scalar angle = std::atan2(vec.dot(v_axis), vec.dot(u));
        angles.emplace_back(angle, idx);
    }

    std::sort(angles.begin(), angles.end());

    // Extract sorted indices
    std::vector<int64_t> result;
    result.reserve(angles.size());
    for (const auto& [angle, idx] : angles) {
        result.push_back(idx);
    }

    return result;
}

std::vector<std::vector<int64_t>> compute_all_face_vertices(
    const Eigen::Ref<const RowMatrixX3>& vertices,
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances,
    Scalar tolerance
) {
    std::vector<std::vector<int64_t>> faces;
    faces.reserve(static_cast<std::size_t>(normals.rows()));

    for (Eigen::Index i = 0; i < normals.rows(); ++i) {
        Vector3 normal = normals.row(i);
        Scalar distance = distances(i);

        std::vector<int64_t> face = compute_face_vertices(
            vertices, normal, distance, tolerance
        );

        if (!face.empty()) {
            faces.push_back(std::move(face));
        }
    }

    return faces;
}

}  // namespace crystal_geometry
