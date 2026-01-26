# CMakeLists.txt for crystal-geometry with native C++ acceleration
#
# This project provides optional C++ acceleration for performance-critical
# operations in the crystal geometry engine.
#
# Build options:
#   -DUSE_OPENMP=ON    Enable OpenMP parallelization
#   -DSKIP_BUILD=ON    Skip native module (pure Python fallback)
#
# Usage:
#   pip install -e ".[dev]"                      # Standard build
#   CMAKE_ARGS="-DUSE_OPENMP=ON" pip install -e ".[dev]"  # With OpenMP
#   CMAKE_ARGS="-DSKIP_BUILD=ON" pip install -e ".[dev]"  # Pure Python only

cmake_minimum_required(VERSION 3.15)
project(crystal_geometry_native
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Native C++ acceleration for crystal geometry"
)

# Option to skip building native module entirely (for pure Python fallback)
option(SKIP_BUILD "Skip building native C++ module" OFF)
if(SKIP_BUILD)
    message(STATUS "SKIP_BUILD=ON: Skipping native C++ module build")
    message(STATUS "Package will use pure Python implementation")
    return()
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# macOS Homebrew support - set Eigen3_DIR directly for reliable detection
# Homebrew's Eigen cmake config isn't in CMAKE_PREFIX_PATH by default
if(APPLE AND NOT Eigen3_DIR)
    # Homebrew ARM64 (Apple Silicon)
    if(EXISTS "/opt/homebrew/opt/eigen/share/eigen3/cmake/Eigen3Config.cmake")
        set(Eigen3_DIR "/opt/homebrew/opt/eigen/share/eigen3/cmake")
    # Homebrew Intel
    elseif(EXISTS "/usr/local/opt/eigen/share/eigen3/cmake/Eigen3Config.cmake")
        set(Eigen3_DIR "/usr/local/opt/eigen/share/eigen3/cmake")
    endif()
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages (will fail gracefully if not found)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(WARNING "pybind11 not found - native module will not be built")
    message(STATUS "Install pybind11: pip install pybind11")
    return()
endif()

# Note: No version requirement - Homebrew's Eigen 5.x uses SameMajorVersion which
# rejects the 3.3 requirement. Both Eigen 3.x and 5.x are compatible with our code.
find_package(Eigen3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    message(WARNING "Eigen3 not found - native module will not be built")
    message(STATUS "Install Eigen3:")
    message(STATUS "  Ubuntu: sudo apt-get install libeigen3-dev")
    message(STATUS "  macOS:  brew install eigen")
    message(STATUS "  Windows: choco install eigen")
    return()
endif()

# Option for OpenMP
option(USE_OPENMP "Enable OpenMP for parallel computation" OFF)
if(USE_OPENMP)
    find_package(OpenMP QUIET)
    if(NOT OpenMP_FOUND)
        message(WARNING "OpenMP not found - building without parallelization")
        set(USE_OPENMP OFF)
    endif()
endif()

# Add native module subdirectory
add_subdirectory(src/crystal_geometry/_native)

# Print configuration summary
message(STATUS "")
message(STATUS "Crystal Geometry Native Module Configuration")
message(STATUS "============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Eigen3: ${Eigen3_VERSION}")
message(STATUS "pybind11: ${pybind11_VERSION}")
message(STATUS "OpenMP: ${USE_OPENMP}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
