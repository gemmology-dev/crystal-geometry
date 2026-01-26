/**
 * @file config.hpp
 * @brief Configuration macros and common includes for crystal geometry native module.
 */

#ifndef CRYSTAL_GEOMETRY_CONFIG_HPP
#define CRYSTAL_GEOMETRY_CONFIG_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <tuple>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <limits>

// Version information
#define CRYSTAL_GEOMETRY_VERSION_MAJOR 1
#define CRYSTAL_GEOMETRY_VERSION_MINOR 0
#define CRYSTAL_GEOMETRY_VERSION_PATCH 0
#define CRYSTAL_GEOMETRY_VERSION "1.0.0"

// OpenMP support (optional)
#ifdef CRYSTAL_USE_OPENMP
    #include <omp.h>
    #include <mutex>
#endif

namespace crystal_geometry {

// Type aliases for clarity and consistency
using Scalar = double;
using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;
using VectorX = Eigen::VectorXd;
using MatrixX = Eigen::MatrixXd;

// Row-major matrices for compatibility with NumPy (C-contiguous)
using RowMatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrix3X = Eigen::Matrix<Scalar, 3, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixX3 = Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;

// Numerical tolerances
constexpr Scalar DEFAULT_TOLERANCE = 1e-8;
constexpr Scalar DETERMINANT_THRESHOLD = 1e-10;
constexpr Scalar BOUNDS_LIMIT = 10.0;

// Thread control
#ifdef CRYSTAL_USE_OPENMP
inline int get_num_threads() {
    return omp_get_max_threads();
}

inline void set_num_threads(int n) {
    if (n <= 0) {
        omp_set_num_threads(omp_get_num_procs());
    } else {
        omp_set_num_threads(n);
    }
}
#else
inline int get_num_threads() { return 1; }
inline void set_num_threads(int) {}
#endif

}  // namespace crystal_geometry

#endif  // CRYSTAL_GEOMETRY_CONFIG_HPP
