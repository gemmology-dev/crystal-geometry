# CMakeLists.txt for crystal_geometry native module

cmake_minimum_required(VERSION 3.15)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen3 (no version requirement - both 3.x and 5.x are compatible)
find_package(Eigen3 REQUIRED CONFIG)

# Option for OpenMP
option(USE_OPENMP "Enable OpenMP for parallel computation" OFF)

# Source files
set(NATIVE_SOURCES
    src/spatial_hash.cpp
    src/geometry.cpp
    src/bindings.cpp
)

# Create the Python module
pybind11_add_module(_native ${NATIVE_SOURCES})

# Include directories
target_include_directories(_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Eigen3
target_link_libraries(_native PRIVATE Eigen3::Eigen)

# C++ standard
target_compile_features(_native PRIVATE cxx_std_17)

# Compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_native PRIVATE
        -O3
        -march=native
        -ffast-math
    )
elseif(MSVC)
    target_compile_options(_native PRIVATE
        /O2
        /fp:fast
    )
endif()

# OpenMP support
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_native PRIVATE CRYSTAL_USE_OPENMP)
endif()

# Installation
install(TARGETS _native DESTINATION crystal_geometry)
