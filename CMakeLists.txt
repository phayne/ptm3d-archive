cmake_minimum_required(VERSION 3.10)
project(europa_thermal_simulator VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add custom cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# AVX support options
option(ENABLE_AVX2 "Enable AVX2 SIMD instructions" ON)
option(ENABLE_AVX512 "Enable AVX512 instructions" ON)

# Check for AVX2 support
include(CheckCXXCompilerFlag)
if(ENABLE_AVX2)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            set(AVX2_FLAGS "/arch:AVX2")
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            set(AVX2_FLAGS "-mavx2")
        endif()
    endif()
    
    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "AVX2 instructions enabled")
        add_definitions(-DAVX2_ENABLED)
    else()
        message(WARNING "AVX2 instructions requested but not supported by compiler")
    endif()
endif()

# Check for AVX-512 support
if(ENABLE_AVX512)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
        if(COMPILER_SUPPORTS_AVX512)
            set(AVX512_FLAGS "/arch:AVX512")
        endif()
    else()
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        if(COMPILER_SUPPORTS_AVX512)
            set(AVX512_FLAGS "-mavx512f")
        endif()
    endif()
    
    if(COMPILER_SUPPORTS_AVX512)
        message(STATUS "AVX512 instructions enabled")
        add_definitions(-DAVX512_ENABLED)
        set(SIMD_FLAGS "${AVX512_FLAGS}")
    else()
        message(STATUS "AVX512 instructions not supported, falling back to AVX2 if available")
    endif()
endif()

# If AVX-512 not available, use AVX2
if(NOT DEFINED SIMD_FLAGS AND COMPILER_SUPPORTS_AVX2)
    set(SIMD_FLAGS "${AVX2_FLAGS}")
endif()

# Apply SIMD flags to global compiler flags
if(SIMD_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SIMD_FLAGS}")
endif()

# Optimization flags
option(USE_AGGRESSIVE_OPTIMIZATIONS "Enable aggressive optimizations" ON)

# Set optimization flags based on build type and platform
set(OPT_FLAGS "")

if(USE_AGGRESSIVE_OPTIMIZATIONS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(OPT_FLAGS "${OPT_FLAGS} -O3 -ffast-math -ftree-vectorize")
        
        # Add Apple Silicon specific optimizations
        if(APPLE)
            execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(ARCH STREQUAL "arm64")
                set(OPT_FLAGS "${OPT_FLAGS} -mcpu=apple-m1")
                message(STATUS "Apple Silicon detected, using optimized flags")
            endif()
        endif()
    elseif(MSVC)
        set(OPT_FLAGS "${OPT_FLAGS} /O2 /fp:fast")
    endif()
    
    message(STATUS "Using aggressive optimizations: ${OPT_FLAGS}")
endif()

# Apply optimization flags to global compiler flags
if(OPT_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}")
endif()

# Add this near the top with your other options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

# Configure OpenMP for macOS
if(APPLE)
    # Special handling for macOS with libomp from Homebrew
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
endif()

# Find OpenMP with the above settings
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, parallel support enabled")
endif()

# Check if CFITSIO is already installed through conda
if(DEFINED ENV{CONDA_PREFIX})
    set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

# Embree Configuration
find_package(embree 4 QUIET)
if(NOT embree_FOUND)
    # Try with pkgconfig if CMake module doesn't work
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(EMBREE embree4)
    endif()
    
    # If still not found, set manual paths for Mac
    if(NOT EMBREE_FOUND AND APPLE)
        set(EMBREE_INCLUDE_DIRS "/usr/local/include" CACHE PATH "Path to Embree headers")
        set(EMBREE_LIBRARIES "/usr/local/lib/libembree4.dylib" CACHE FILEPATH "Embree library")
    endif()
endif()

# Include Embree in all targets that need it
include_directories(${EMBREE_INCLUDE_DIRS})

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# SPICE Configuration - Allow user to specify custom paths
set(CSPICE_ROOT "/Users/paha3326/software/cspice" CACHE PATH "Path to CSPICE installation")
set(CSPICE_INCLUDE_DIR "${CSPICE_ROOT}/include" CACHE PATH "Path to CSPICE headers")
set(CSPICE_LIBRARY_DIR "${CSPICE_ROOT}/lib" CACHE PATH "Path to CSPICE libraries")

# Check if the CSPICE library exists
find_library(CSPICE_LIBRARY
    NAMES cspice
    PATHS ${CSPICE_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

if(NOT CSPICE_LIBRARY)
    message(FATAL_ERROR "CSPICE library not found. Please set CSPICE_ROOT to your CSPICE installation directory.")
endif()

# Kernels directory
set(SPICE_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/kernels" CACHE PATH "Path to SPICE kernel files")

# Create a configuration file for runtime access to kernels path
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.h
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CSPICE_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include  # For the generated config.h
)

# Link directories
link_directories(
    ${CSPICE_LIBRARY_DIR}
)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python ${Python_VERSION}")

# Make sure we use the Python from the active environment
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Using conda environment: $ENV{CONDA_PREFIX}")
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
endif()

# Find pybind11
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    # Look for pybind11 installed via pip
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
        message(STATUS "Found pybind11: ${PYBIND11_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Could not find pybind11. Please install with pip install pybind11")
    endif()
endif()

# Add the subdirectories for the main project
add_subdirectory(src/cpp/constants)
add_subdirectory(src/cpp/maps)
add_subdirectory(src/cpp/mesh)
add_subdirectory(src/cpp/thermal)
add_subdirectory(src/cpp/illumination)
add_subdirectory(src/cpp/main)
add_subdirectory(src/cpp/tools)

# Enable testing
# enable_testing()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(src/python/bindings)
endif()
