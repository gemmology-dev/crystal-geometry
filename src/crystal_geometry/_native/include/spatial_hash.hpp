/**
 * @file spatial_hash.hpp
 * @brief Spatial hashing for O(n) vertex deduplication.
 */

#ifndef CRYSTAL_GEOMETRY_SPATIAL_HASH_HPP
#define CRYSTAL_GEOMETRY_SPATIAL_HASH_HPP

#include "config.hpp"

namespace crystal_geometry {

/**
 * @brief Integer 3D cell key for spatial hashing.
 */
struct CellKey {
    int64_t x, y, z;

    bool operator==(const CellKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * @brief Hash function for CellKey.
 */
struct CellKeyHash {
    std::size_t operator()(const CellKey& k) const {
        // FNV-1a inspired hash combining
        std::size_t h = 14695981039346656037ULL;
        h ^= static_cast<std::size_t>(k.x);
        h *= 1099511628211ULL;
        h ^= static_cast<std::size_t>(k.y);
        h *= 1099511628211ULL;
        h ^= static_cast<std::size_t>(k.z);
        h *= 1099511628211ULL;
        return h;
    }
};

/**
 * @brief Compute the cell key for a vertex position.
 *
 * @param vertex The 3D vertex position
 * @param cell_size The size of each hash cell
 * @return CellKey The integer cell coordinates
 */
inline CellKey spatial_hash_key(const Vector3& vertex, Scalar cell_size) {
    return {
        static_cast<int64_t>(std::floor(vertex.x() / cell_size)),
        static_cast<int64_t>(std::floor(vertex.y() / cell_size)),
        static_cast<int64_t>(std::floor(vertex.z() / cell_size))
    };
}

/**
 * @brief Spatial hash grid for efficient vertex deduplication.
 *
 * Uses a hash map with cell keys to achieve O(n) average-case deduplication
 * by only checking vertices in neighboring cells.
 */
class SpatialHashGrid {
public:
    /**
     * @brief Construct a new Spatial Hash Grid.
     *
     * @param cell_size Size of each hash cell (should be ~2x tolerance)
     */
    explicit SpatialHashGrid(Scalar cell_size = DEFAULT_TOLERANCE * 2)
        : cell_size_(cell_size)
        , tolerance_sq_(cell_size * cell_size / 4)  // (cell_size/2)^2
    {}

    /**
     * @brief Try to add a vertex to the grid.
     *
     * @param vertex The vertex to add
     * @param index The index to store if vertex is unique
     * @return true if vertex was added (is unique)
     * @return false if vertex is a duplicate
     */
    bool try_add(const Vector3& vertex, std::size_t index) {
        CellKey key = spatial_hash_key(vertex, cell_size_);

        // Check this cell and 26 neighbors
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    CellKey neighbor{key.x + dx, key.y + dy, key.z + dz};
                    auto it = grid_.find(neighbor);
                    if (it != grid_.end()) {
                        // Check if vertex is within tolerance of existing
                        const auto& existing = vertices_[it->second];
                        Scalar dist_sq = (vertex - existing).squaredNorm();
                        if (dist_sq < tolerance_sq_) {
                            return false;  // Duplicate found
                        }
                    }
                }
            }
        }

        // No duplicate found, add to grid
        grid_[key] = index;
        vertices_.push_back(vertex);
        return true;
    }

    /**
     * @brief Get all unique vertices.
     */
    const std::vector<Vector3>& vertices() const { return vertices_; }

    /**
     * @brief Get number of unique vertices.
     */
    std::size_t size() const { return vertices_.size(); }

    /**
     * @brief Clear the grid.
     */
    void clear() {
        grid_.clear();
        vertices_.clear();
    }

    /**
     * @brief Set the tolerance for deduplication.
     *
     * @param tolerance Distance threshold for considering vertices identical
     */
    void set_tolerance(Scalar tolerance) {
        cell_size_ = tolerance * 2;
        tolerance_sq_ = tolerance * tolerance;
    }

private:
    Scalar cell_size_;
    Scalar tolerance_sq_;
    std::unordered_map<CellKey, std::size_t, CellKeyHash> grid_;
    std::vector<Vector3> vertices_;
};

/**
 * @brief Deduplicate vertices using spatial hashing.
 *
 * @param vertices Nx3 matrix of vertices (row-major)
 * @param tolerance Distance threshold for considering vertices identical
 * @return RowMatrixX3 Matrix of unique vertices
 */
RowMatrixX3 deduplicate_vertices(
    const Eigen::Ref<const RowMatrixX3>& vertices,
    Scalar tolerance = DEFAULT_TOLERANCE
);

}  // namespace crystal_geometry

#endif  // CRYSTAL_GEOMETRY_SPATIAL_HASH_HPP
