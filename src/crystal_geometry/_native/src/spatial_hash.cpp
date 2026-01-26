/**
 * @file spatial_hash.cpp
 * @brief Implementation of spatial hashing for vertex deduplication.
 */

#include "spatial_hash.hpp"

namespace crystal_geometry {

RowMatrixX3 deduplicate_vertices(
    const Eigen::Ref<const RowMatrixX3>& vertices,
    Scalar tolerance
) {
    if (vertices.rows() == 0) {
        return RowMatrixX3(0, 3);
    }

    SpatialHashGrid grid(tolerance * 2);
    grid.set_tolerance(tolerance);

    std::vector<std::size_t> unique_indices;
    unique_indices.reserve(vertices.rows());

    for (Eigen::Index i = 0; i < vertices.rows(); ++i) {
        Vector3 v = vertices.row(i);
        if (grid.try_add(v, static_cast<std::size_t>(i))) {
            unique_indices.push_back(static_cast<std::size_t>(i));
        }
    }

    // Build result matrix
    RowMatrixX3 result(unique_indices.size(), 3);
    for (std::size_t i = 0; i < unique_indices.size(); ++i) {
        result.row(static_cast<Eigen::Index>(i)) = vertices.row(
            static_cast<Eigen::Index>(unique_indices[i])
        );
    }

    return result;
}

}  // namespace crystal_geometry
