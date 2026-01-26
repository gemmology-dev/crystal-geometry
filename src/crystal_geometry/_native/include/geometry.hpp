/**
 * @file geometry.hpp
 * @brief Core geometry operations for crystal polyhedra.
 */

#ifndef CRYSTAL_GEOMETRY_GEOMETRY_HPP
#define CRYSTAL_GEOMETRY_GEOMETRY_HPP

#include "config.hpp"
#include "spatial_hash.hpp"

namespace crystal_geometry {

/**
 * @brief Result of halfspace intersection computation.
 */
struct HalfspaceResult {
    RowMatrixX3 vertices;
    bool success;
    std::string error_message;
};

/**
 * @brief Find interior point using Chebyshev center method.
 *
 * Finds the center of the largest ball that fits inside the polyhedron
 * defined by the halfspaces. Uses a simple iterative approach.
 *
 * @param normals Nx3 matrix of unit normal vectors
 * @param distances N-element vector of distances
 * @return std::pair<Vector3, bool> Interior point and success flag
 */
std::pair<Vector3, bool> find_interior_point(
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances
);

/**
 * @brief Find interior point by iterative shrinking (fallback method).
 *
 * @param normals Nx3 matrix of unit normal vectors
 * @param distances N-element vector of distances
 * @return std::pair<Vector3, bool> Interior point and success flag
 */
std::pair<Vector3, bool> find_interior_point_iterative(
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances
);

/**
 * @brief Compute halfspace intersection by finding triple-plane intersections.
 *
 * For each combination of three planes, solves the 3x3 linear system to find
 * the intersection point. Points are validated to be inside all halfspaces.
 *
 * This is the core O(n³) algorithm that benefits most from C++ acceleration
 * due to tight loops and many small matrix operations.
 *
 * @param normals Nx3 matrix of unit normal vectors (row-major)
 * @param distances N-element vector of distances from origin
 * @param tolerance Numerical tolerance for deduplication
 * @return HalfspaceResult Result containing vertices or error
 */
HalfspaceResult halfspace_intersection(
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances,
    Scalar tolerance = DEFAULT_TOLERANCE
);

/**
 * @brief Compute vertices of a face given plane equation.
 *
 * Finds all vertices that lie on a plane (normal · v = distance) and
 * orders them counter-clockwise when viewed from outside.
 *
 * @param vertices Mx3 matrix of all vertices
 * @param normal Face normal vector
 * @param distance Distance from origin to face plane
 * @param tolerance Numerical tolerance for on-plane detection
 * @return std::vector<int64_t> Vertex indices in counter-clockwise order
 */
std::vector<int64_t> compute_face_vertices(
    const Eigen::Ref<const RowMatrixX3>& vertices,
    const Vector3& normal,
    Scalar distance,
    Scalar tolerance = 1e-6
);

/**
 * @brief Compute all face vertex lists for a set of halfspaces.
 *
 * For each halfspace (plane), finds vertices on that plane and orders
 * them counter-clockwise.
 *
 * @param vertices Mx3 matrix of all vertices
 * @param normals Nx3 matrix of face normals
 * @param distances N-element vector of distances
 * @param tolerance Numerical tolerance
 * @return std::vector<std::vector<int64_t>> List of face vertex index lists
 */
std::vector<std::vector<int64_t>> compute_all_face_vertices(
    const Eigen::Ref<const RowMatrixX3>& vertices,
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances,
    Scalar tolerance = 1e-5
);

/**
 * @brief Check if a point is inside all halfspaces.
 *
 * @param point Point to check
 * @param normals Nx3 matrix of normals
 * @param distances N-element vector of distances
 * @param tolerance Tolerance for boundary
 * @return true if point is strictly inside all halfspaces
 */
bool is_inside_all_halfspaces(
    const Vector3& point,
    const Eigen::Ref<const RowMatrixX3>& normals,
    const Eigen::Ref<const VectorX>& distances,
    Scalar tolerance = 1e-10
);

}  // namespace crystal_geometry

#endif  // CRYSTAL_GEOMETRY_GEOMETRY_HPP
