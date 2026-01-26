/**
 * @file bindings.cpp
 * @brief Python bindings for crystal geometry native module.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "config.hpp"
#include "geometry.hpp"
#include "spatial_hash.hpp"

namespace py = pybind11;
using namespace crystal_geometry;

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native C++ acceleration module for crystal geometry operations";

    // Version information
    m.attr("__version__") = CRYSTAL_GEOMETRY_VERSION;

    // Thread control
    m.def("get_num_threads", &get_num_threads,
          "Get number of threads for parallel operations");
    m.def("set_num_threads", &set_num_threads,
          "Set number of threads (0 for auto)",
          py::arg("n") = 0);

    // Core geometry functions
    m.def("_find_interior_point",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> normals,
             py::array_t<double, py::array::c_style | py::array::forcecast> distances)
              -> py::object {
              // Map to Eigen
              auto normals_buf = normals.request();
              auto distances_buf = distances.request();

              if (normals_buf.ndim != 2 || normals_buf.shape[1] != 3) {
                  throw std::runtime_error("normals must be Nx3 array");
              }
              if (distances_buf.ndim != 1) {
                  throw std::runtime_error("distances must be 1D array");
              }

              Eigen::Map<const RowMatrixX3> normals_map(
                  static_cast<double*>(normals_buf.ptr),
                  normals_buf.shape[0], 3
              );
              Eigen::Map<const VectorX> distances_map(
                  static_cast<double*>(distances_buf.ptr),
                  distances_buf.shape[0]
              );

              auto [point, success] = find_interior_point(normals_map, distances_map);

              if (success) {
                  py::array_t<double> result(3);
                  auto result_buf = result.request();
                  double* result_ptr = static_cast<double*>(result_buf.ptr);
                  result_ptr[0] = point.x();
                  result_ptr[1] = point.y();
                  result_ptr[2] = point.z();
                  return result;
              } else {
                  return py::none();
              }
          },
          "Find interior point using Chebyshev center method",
          py::arg("normals"), py::arg("distances"));

    m.def("_deduplicate_vertices",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
             double tolerance) -> py::array_t<double> {
              auto vertices_buf = vertices.request();

              if (vertices_buf.ndim != 2 || vertices_buf.shape[1] != 3) {
                  throw std::runtime_error("vertices must be Nx3 array");
              }

              Eigen::Map<const RowMatrixX3> vertices_map(
                  static_cast<double*>(vertices_buf.ptr),
                  vertices_buf.shape[0], 3
              );

              RowMatrixX3 result = deduplicate_vertices(vertices_map, tolerance);

              py::array_t<double> output({result.rows(), Eigen::Index(3)});
              auto output_buf = output.request();
              double* output_ptr = static_cast<double*>(output_buf.ptr);

              for (Eigen::Index i = 0; i < result.rows(); ++i) {
                  for (Eigen::Index j = 0; j < 3; ++j) {
                      output_ptr[i * 3 + j] = result(i, j);
                  }
              }

              return output;
          },
          "Deduplicate vertices using spatial hashing (O(n) complexity)",
          py::arg("vertices"), py::arg("tolerance") = DEFAULT_TOLERANCE);

    // Named to match Python function for @prefer_native decorator
    m.def("halfspace_intersection_3d",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> normals,
             py::array_t<double, py::array::c_style | py::array::forcecast> distances,
             py::object interior_point_arg,
             double tolerance) -> py::object {
              auto normals_buf = normals.request();
              auto distances_buf = distances.request();

              if (normals_buf.ndim != 2 || normals_buf.shape[1] != 3) {
                  throw std::runtime_error("normals must be Nx3 array");
              }
              if (distances_buf.ndim != 1) {
                  throw std::runtime_error("distances must be 1D array");
              }

              Eigen::Map<const RowMatrixX3> normals_map(
                  static_cast<double*>(normals_buf.ptr),
                  normals_buf.shape[0], 3
              );
              Eigen::Map<const VectorX> distances_map(
                  static_cast<double*>(distances_buf.ptr),
                  distances_buf.shape[0]
              );

              HalfspaceResult result = halfspace_intersection(
                  normals_map, distances_map, tolerance
              );

              if (!result.success) {
                  return py::none();
              }

              // Convert result to numpy array
              py::array_t<double> output({result.vertices.rows(), Eigen::Index(3)});
              auto output_buf = output.request();
              double* output_ptr = static_cast<double*>(output_buf.ptr);

              for (Eigen::Index i = 0; i < result.vertices.rows(); ++i) {
                  for (Eigen::Index j = 0; j < 3; ++j) {
                      output_ptr[i * 3 + j] = result.vertices(i, j);
                  }
              }

              return output;
          },
          "Compute halfspace intersection vertices",
          py::arg("normals"),
          py::arg("distances"),
          py::arg("interior_point") = py::none(),
          py::arg("tolerance") = DEFAULT_TOLERANCE);

    m.def("compute_face_vertices",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
             py::array_t<double, py::array::c_style | py::array::forcecast> normal,
             double distance,
             double tolerance) -> std::vector<int64_t> {
              auto vertices_buf = vertices.request();
              auto normal_buf = normal.request();

              if (vertices_buf.ndim != 2 || vertices_buf.shape[1] != 3) {
                  throw std::runtime_error("vertices must be Nx3 array");
              }
              if (normal_buf.ndim != 1 || normal_buf.shape[0] != 3) {
                  throw std::runtime_error("normal must be 3-element array");
              }

              Eigen::Map<const RowMatrixX3> vertices_map(
                  static_cast<double*>(vertices_buf.ptr),
                  vertices_buf.shape[0], 3
              );

              double* normal_ptr = static_cast<double*>(normal_buf.ptr);
              Vector3 normal_vec(normal_ptr[0], normal_ptr[1], normal_ptr[2]);

              return compute_face_vertices(vertices_map, normal_vec, distance, tolerance);
          },
          "Find and order vertices on a face plane",
          py::arg("vertices"),
          py::arg("normal"),
          py::arg("distance"),
          py::arg("tolerance") = 1e-6);

    m.def("compute_all_face_vertices",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
             py::array_t<double, py::array::c_style | py::array::forcecast> normals,
             py::array_t<double, py::array::c_style | py::array::forcecast> distances,
             double tolerance) -> std::vector<std::vector<int64_t>> {
              auto vertices_buf = vertices.request();
              auto normals_buf = normals.request();
              auto distances_buf = distances.request();

              Eigen::Map<const RowMatrixX3> vertices_map(
                  static_cast<double*>(vertices_buf.ptr),
                  vertices_buf.shape[0], 3
              );
              Eigen::Map<const RowMatrixX3> normals_map(
                  static_cast<double*>(normals_buf.ptr),
                  normals_buf.shape[0], 3
              );
              Eigen::Map<const VectorX> distances_map(
                  static_cast<double*>(distances_buf.ptr),
                  distances_buf.shape[0]
              );

              return compute_all_face_vertices(
                  vertices_map, normals_map, distances_map, tolerance
              );
          },
          "Compute face vertex lists for all halfspaces",
          py::arg("vertices"),
          py::arg("normals"),
          py::arg("distances"),
          py::arg("tolerance") = 1e-5);
}
