#
# CMakeLists.txt - CMake build configuration for the PTM3D
# Planetary Thermophysical Model 3D
# Paul O. Hayne, University of Colorado Boulder
#

cmake_minimum_required(VERSION 3.10)

# ------------------------------------------------------------------------------
# Cross-platform compiler selection (before project())
# Honors, in order: -DCMAKE_*_COMPILER, env CC/CXX, PTM_PREFERRED_CXX,
# then auto-detects a sensible default per platform.
# To switch compilers in an existing build dir, clear CMake cache first.
# Usage examples:
#   cmake -S . -B build -DPTM_PREFERRED_CXX=clang
#   CC=gcc CXX=g++ cmake -S . -B build
# ------------------------------------------------------------------------------
# Skip override if a Visual Studio generator is used (MSVC selected by generator)
if(NOT DEFINED CMAKE_CXX_COMPILER
   AND NOT CMAKE_GENERATOR MATCHES "Visual Studio"
   AND NOT MSVC)
  # If env CXX/CC provided, let CMake honor them
  if(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
    set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE FILEPATH "C++ compiler" FORCE)
  endif()
  if(DEFINED ENV{CC} AND NOT "$ENV{CC}" STREQUAL "")
    set(CMAKE_C_COMPILER "$ENV{CC}" CACHE FILEPATH "C compiler" FORCE)
  endif()

  # Optional preference (lowercased)
  if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(PTM_PREFERRED_CXX "$ENV{PTM_PREFERRED_CXX}" CACHE STRING
        "Preferred C++ compiler (clang|gcc|msvc|icx). Leave empty for autodetect.")
    string(TOLOWER "${PTM_PREFERRED_CXX}" PTM_PREFERRED_CXX_LC)

    if(PTM_PREFERRED_CXX_LC STREQUAL "clang")
      find_program(_PTM_CLANGXX NAMES clang++ clang++-18 clang++-17 clang++)
      find_program(_PTM_CLANGC  NAMES clang clang-18 clang-17 clang)
      if(_PTM_CLANGXX)
        set(CMAKE_CXX_COMPILER "${_PTM_CLANGXX}" CACHE FILEPATH "C++ compiler" FORCE)
      endif()
      if(_PTM_CLANGC)
        set(CMAKE_C_COMPILER  "${_PTM_CLANGC}"  CACHE FILEPATH "C compiler"  FORCE)
      endif()
    elseif(PTM_PREFERRED_CXX_LC STREQUAL "gcc" OR PTM_PREFERRED_CXX_LC STREQUAL "g++")
      find_program(_PTM_GXX NAMES g++ g++-13 g++-12 g++)
      find_program(_PTM_GCC NAMES gcc gcc-13 gcc-12 gcc)
      if(_PTM_GXX)
        set(CMAKE_CXX_COMPILER "${_PTM_GXX}" CACHE FILEPATH "C++ compiler" FORCE)
      endif()
      if(_PTM_GCC)
        set(CMAKE_C_COMPILER  "${_PTM_GCC}" CACHE FILEPATH "C compiler" FORCE)
      endif()
    elseif(PTM_PREFERRED_CXX_LC STREQUAL "icx" OR PTM_PREFERRED_CXX_LC STREQUAL "icpx")
      # oneAPI (new): icx/icpx
      find_program(_PTM_ICXX NAMES icpx icx)
      find_program(_PTM_ICC NAMES icx icc)
      if(_PTM_ICXX)
        set(CMAKE_CXX_COMPILER "${_PTM_ICXX}" CACHE FILEPATH "C++ compiler" FORCE)
      endif()
      if(_PTM_ICC)
        set(CMAKE_C_COMPILER  "${_PTM_ICC}"  CACHE FILEPATH "C compiler"  FORCE)
      endif()
    endif()
  endif()

  # Final fallback autodetect if still unset
  if(NOT DEFINED CMAKE_CXX_COMPILER)
    if(WIN32)
      # Prefer LLVM clang-cl for Ninja/Makefile generators on Windows if present
      find_program(_PTM_CLANGCL NAMES clang-cl)
      if(_PTM_CLANGCL)
        set(CMAKE_CXX_COMPILER "${_PTM_CLANGCL}" CACHE FILEPATH "C++ compiler" FORCE)
        # Use clang for C if available
        find_program(_PTM_CLANG NAMES clang)
        if(_PTM_CLANG)
          set(CMAKE_C_COMPILER "${_PTM_CLANG}" CACHE FILEPATH "C compiler" FORCE)
        endif()
      endif()
    endif()

    # POSIX: prefer clang++, then g++
    if(NOT DEFINED CMAKE_CXX_COMPILER)
      find_program(_PTM_CLANGXX NAMES clang++ clang++-18 clang++-17 clang++)
      find_program(_PTM_GXX     NAMES g++ g++-13 g++-12 g++)
      if(APPLE AND _PTM_CLANGXX)
        set(CMAKE_CXX_COMPILER "${_PTM_CLANGXX}" CACHE FILEPATH "C++ compiler" FORCE)
        find_program(_PTM_CLANG NAMES clang)
        if(_PTM_CLANG)
          set(CMAKE_C_COMPILER "${_PTM_CLANG}" CACHE FILEPATH "C compiler" FORCE)
        endif()
      elseif(_PTM_CLANGXX)
        set(CMAKE_CXX_COMPILER "${_PTM_CLANGXX}" CACHE FILEPATH "C++ compiler" FORCE)
        find_program(_PTM_CLANG NAMES clang)
        if(_PTM_CLANG)
          set(CMAKE_C_COMPILER "${_PTM_CLANG}" CACHE FILEPATH "C compiler" FORCE)
        endif()
      elseif(_PTM_GXX)
        set(CMAKE_CXX_COMPILER "${_PTM_GXX}" CACHE FILEPATH "C++ compiler" FORCE)
        find_program(_PTM_GCC NAMES gcc gcc-13 gcc-12 gcc)
        if(_PTM_GCC)
          set(CMAKE_C_COMPILER "${_PTM_GCC}" CACHE FILEPATH "C compiler" FORCE)
        endif()
      endif()
    endif()
  endif()
endif()

project(planetary_thermophysical_model VERSION 1.1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add custom cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# SIMD Support Detection and Configuration
# ============================================================================
option(ENABLE_AVX2 "Enable AVX2 SIMD instructions" ON)
option(ENABLE_AVX512 "Enable AVX512 instructions" ON)

include(CheckCXXCompilerFlag)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(X86_64_ARCHITECTURE TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(ARM_ARCHITECTURE TRUE)
    message(STATUS "ARM architecture detected - SIMD optimizations will use NEON if available")
endif()

# Initialize SIMD support variables
set(COMPILER_SUPPORTS_AVX512 FALSE CACHE INTERNAL "AVX512 support")
set(COMPILER_SUPPORTS_AVX2 FALSE CACHE INTERNAL "AVX2 support")

# Only check for AVX on x86_64
if(X86_64_ARCHITECTURE)
    # Check for AVX2 support
    if(ENABLE_AVX2)
        if(MSVC)
            check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                # For MSVC, use single flag
                set(AVX2_FLAGS "/arch:AVX2")
                add_compile_options(/arch:AVX2)
            endif()
        else()
            # Check AVX2 support
            check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2_FLAG)
            if(COMPILER_SUPPORTS_AVX2_FLAG)
                set(COMPILER_SUPPORTS_AVX2 TRUE CACHE INTERNAL "AVX2 support")
                # CRITICAL FIX: Set AVX2_FLAGS as a list, not a string
                set(AVX2_FLAGS -mavx2 -mfma)  # No quotes - this creates a list
                add_compile_options(-mavx2 -mfma)
                add_definitions(-DAVX2_ENABLED)
                message(STATUS "AVX2 instructions enabled")
            endif()
        endif()
    endif()

    # Check for AVX-512 support on x86_64
    # Note: Some Intel Macs do support AVX-512, so we allow detection
    # Only exclude ARM-based Macs (Apple Silicon)
    if(ENABLE_AVX512 AND NOT ARM_ARCHITECTURE)
        if(MSVC)
            check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
            if(COMPILER_SUPPORTS_AVX512)
                set(AVX512_FLAGS "/arch:AVX512")
                add_compile_options(/arch:AVX512)
                add_definitions(-DAVX512_ENABLED)
            endif()
        else()
            # Check each AVX-512 flag separately
            check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
            check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
            
            if(COMPILER_SUPPORTS_AVX512F AND COMPILER_SUPPORTS_AVX512DQ)
                set(COMPILER_SUPPORTS_AVX512 TRUE CACHE INTERNAL "AVX512 support")
                # CRITICAL FIX: Set AVX512_FLAGS as a list, not a string
                set(AVX512_FLAGS -mavx512f -mavx512dq)  # No quotes - this creates a list
                add_compile_options(-mavx512f)
                add_compile_options(-mavx512dq)
                add_definitions(-DAVX512_ENABLED -D__AVX512F__ -D__AVX512DQ__)
                message(STATUS "AVX-512 instructions enabled")
            endif()
        endif()
    endif()
elseif(ARM_ARCHITECTURE)
    # ARM NEON is typically enabled by default on ARM64
    add_definitions(-DARM_NEON_ENABLED)
    message(STATUS "ARM NEON SIMD will be used if available")
endif()

# Export for subdirectories (already set as CACHE INTERNAL above)

# ============================================================================
# General Optimization Flags
# ============================================================================
option(USE_AGGRESSIVE_OPTIMIZATIONS "Enable aggressive optimizations" ON)

if(USE_AGGRESSIVE_OPTIMIZATIONS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        add_compile_options(-O3)
        add_compile_options(-ffast-math)
        add_compile_options(-ftree-vectorize)
        add_compile_options(-funroll-loops)
        
        # Architecture-specific optimizations
        if(APPLE)
            execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(ARCH STREQUAL "arm64")
                # Apple Silicon optimizations
                add_compile_options(-mcpu=apple-m1)
                message(STATUS "Apple Silicon detected, using M1 optimizations")
            elseif(X86_64_ARCHITECTURE)
                # Intel Mac - use native optimizations (includes AVX-512 if available)
                add_compile_options(-march=native)
                message(STATUS "Intel Mac detected, using native optimizations")
            endif()
        elseif(X86_64_ARCHITECTURE)
            # Linux/Unix x86-64
            add_compile_options(-march=native)
        endif()
        
        message(STATUS "Aggressive optimizations enabled")
    elseif(MSVC)
        add_compile_options(/O2 /fp:fast)
        message(STATUS "MSVC optimizations enabled")
    endif()
endif()

# ============================================================================
# OpenMP Configuration
# ============================================================================
# Configure OpenMP for macOS
if(APPLE)
    # Special handling for macOS with libomp from Homebrew or MacPorts
    # Try Homebrew first
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    # Suppress duplicate library warnings from Apple's ld
    foreach(var CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
        if(NOT "${${var}}" MATCHES "no_warn_duplicate_libraries")
            set(${var} "${${var}} -Wl,-no_warn_duplicate_libraries")
        endif()
    endforeach()
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, parallel support enabled")
    # Don't use CMAKE_CXX_FLAGS, use target-specific linking instead
endif()

# ============================================================================
# External Dependencies
# ============================================================================

# CFITSIO Configuration
if(DEFINED ENV{CONDA_PREFIX})
    set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    message(STATUS "Using conda environment: $ENV{CONDA_PREFIX}")
endif()

find_package(CFITSIO REQUIRED)
message(STATUS "Found CFITSIO: ${CFITSIO_LIBRARIES}")

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

if(EMBREE_FOUND OR embree_FOUND)
    message(STATUS "Embree found")
else()
    message(FATAL_ERROR "Embree not found. Please install Embree 4")
endif()

# Include Embree in all targets that need it
include_directories(${EMBREE_INCLUDE_DIRS})

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

# SPICE Configuration
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
message(STATUS "Found CSPICE: ${CSPICE_LIBRARY}")

# Set CSPICE_LIBRARIES for compatibility
set(CSPICE_LIBRARIES ${CSPICE_LIBRARY})

# Kernels directory
set(SPICE_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/kernels" CACHE PATH "Path to SPICE kernel files")

# Albedo maps directory
set(PTM_ALBEDO_MAPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/albedo_maps" CACHE PATH "Path to albedo map files")

# Project data directory (parent of kernels and albedo_maps)
set(PTM_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Path to PTM data directory")

# ============================================================================
# Configuration File Generation
# ============================================================================
# Create a configuration file for runtime access to kernels path
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.h
)

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CSPICE_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include  # For the generated config.h
)

# Link directories
link_directories(
    ${CSPICE_LIBRARY_DIR}
)

# ============================================================================
# yaml-cpp Configuration - Robust macOS/Linux Solution
# ============================================================================

set(YAML_CPP_FOUND FALSE)
set(YAML_CPP_INCLUDE_DIRS "")
set(YAML_CPP_LIBRARIES "")

# Method 1: Try pkg-config first (most reliable)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(YAML_CPP QUIET yaml-cpp)
    if(YAML_CPP_FOUND)
        message(STATUS "Found yaml-cpp via pkg-config")
        message(STATUS "  Include: ${YAML_CPP_INCLUDE_DIRS}")
        message(STATUS "  Library: ${YAML_CPP_LIBRARIES}")
    endif()
endif()

# Method 2: Try find_package
if(NOT YAML_CPP_FOUND)
    find_package(yaml-cpp QUIET CONFIG)
    if(yaml-cpp_FOUND)
        set(YAML_CPP_FOUND TRUE)
        # For modern CMake targets
        if(TARGET yaml-cpp)
            get_target_property(YAML_CPP_INCLUDE_DIRS yaml-cpp INTERFACE_INCLUDE_DIRECTORIES)
            if(NOT YAML_CPP_INCLUDE_DIRS)
                set(YAML_CPP_INCLUDE_DIRS "")
            endif()
            # Use the target directly for linking
            set(YAML_CPP_LIBRARIES yaml-cpp)
            message(STATUS "Found yaml-cpp via find_package (target)")
        else()
            message(WARNING "yaml-cpp found but target not available")
            set(YAML_CPP_FOUND FALSE)
        endif()
    endif()
endif()

# Method 3: Manual detection for Homebrew (macOS)
if(NOT YAML_CPP_FOUND AND APPLE)
    message(STATUS "Trying manual yaml-cpp detection for Homebrew...")
    
    # Determine Homebrew prefix
    execute_process(
        COMMAND brew --prefix
        OUTPUT_VARIABLE HOMEBREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(HOMEBREW_PREFIX)
        set(HOMEBREW_PATHS ${HOMEBREW_PREFIX})
    else()
        # Fallback to common paths
        set(HOMEBREW_PATHS /opt/homebrew /usr/local)
    endif()
    
    # Search for header and library
    foreach(PREFIX ${HOMEBREW_PATHS})
        if(EXISTS "${PREFIX}/include/yaml-cpp/yaml.h")
            set(YAML_CPP_INCLUDE_DIRS "${PREFIX}/include")
            
            # Look for library (try .dylib first, then .a)
            find_library(YAML_CPP_LIBRARY_PATH
                NAMES yaml-cpp
                PATHS "${PREFIX}/lib"
                NO_DEFAULT_PATH
            )
            
            if(YAML_CPP_LIBRARY_PATH)
                set(YAML_CPP_FOUND TRUE)
                set(YAML_CPP_LIBRARIES "${YAML_CPP_LIBRARY_PATH}")
                message(STATUS "Found yaml-cpp (Homebrew manual)")
                message(STATUS "  Include: ${YAML_CPP_INCLUDE_DIRS}")
                message(STATUS "  Library: ${YAML_CPP_LIBRARIES}")
                break()
            endif()
        endif()
    endforeach()
endif()

# Method 4: Check conda environment
if(NOT YAML_CPP_FOUND AND DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    if(EXISTS "${CONDA_PREFIX}/include/yaml-cpp/yaml.h")
        set(YAML_CPP_INCLUDE_DIRS "${CONDA_PREFIX}/include")
        
        find_library(YAML_CPP_LIBRARY_PATH
            NAMES yaml-cpp
            PATHS "${CONDA_PREFIX}/lib"
            NO_DEFAULT_PATH
        )
        
        if(YAML_CPP_LIBRARY_PATH)
            set(YAML_CPP_FOUND TRUE)
            set(YAML_CPP_LIBRARIES "${YAML_CPP_LIBRARY_PATH}")
            message(STATUS "Found yaml-cpp (Conda)")
            message(STATUS "  Include: ${YAML_CPP_INCLUDE_DIRS}")
            message(STATUS "  Library: ${YAML_CPP_LIBRARIES}")
        endif()
    endif()
endif()

# Set compile definition if found
if(YAML_CPP_FOUND)
    add_compile_definitions(HAVE_YAML_CPP)
else()
    message(WARNING "========================================")
    message(WARNING "yaml-cpp not found!")
    message(WARNING "Config file support will be disabled.")
    message(WARNING "To install:")
    message(WARNING "  macOS: brew install yaml-cpp")
    message(WARNING "  conda: conda install -c conda-forge yaml-cpp")
    message(WARNING "========================================")
endif()

# ============================================================================
# Python Support (Optional)
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    # Find Python
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Found Python ${Python_VERSION}")
    
    # Make sure we use the Python from the active environment
    if(DEFINED ENV{CONDA_PREFIX})
        set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    endif()
    
    # Look for pybind11
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_IMPORT_RESULT
    )
    
    if(PYBIND11_IMPORT_RESULT EQUAL 0)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG REQUIRED)
            message(STATUS "Found pybind11: ${PYBIND11_INCLUDE_DIR}")
        endif()
    else()
        message(WARNING "pybind11 not found. Python bindings will not be built.")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()

# ============================================================================
# GPU Acceleration (Optional)
# ============================================================================
option(ENABLE_GPU "Enable GPU acceleration for thermal solver" OFF)
set(GPU_BACKEND "AUTO" CACHE STRING "GPU backend: AUTO, HIP, CUDA, or NONE")

if(ENABLE_GPU)
    message(STATUS "GPU acceleration enabled (backend: ${GPU_BACKEND})")
endif()

# ============================================================================
# Subdirectories
# ============================================================================
# Add the subdirectories for the main project
add_subdirectory(src/cpp/constants)
add_subdirectory(src/cpp/maps)
add_subdirectory(src/cpp/mesh)
add_subdirectory(src/cpp/thermal)      # This will pick up thermal_adaptive_simd.cpp
add_subdirectory(src/cpp/orbital)      # Keplerian propagator, body-fixed frame

# GPU module (must be before illumination/radiation/main to allow linking)
if(ENABLE_GPU)
    add_subdirectory(src/cpp/gpu)
endif()

add_subdirectory(src/cpp/illumination)
add_subdirectory(src/cpp/radiation)
add_subdirectory(src/cpp/fileio)
add_subdirectory(src/cpp/tools)

add_subdirectory(src/cpp/main)
add_subdirectory(src/cpp/exosphere)
add_subdirectory(src/cpp/viewshed)
add_subdirectory(src/cpp/ice)          # Ice table depth solver

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(src/python/bindings)
endif()

# ============================================================================
# Testing
# ============================================================================
option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Installation
# ============================================================================
install(DIRECTORY data/kernels 
    DESTINATION share/ptm_thermal
    PATTERN "*.bsp"
    PATTERN "*.tls"
    PATTERN "*.pck"
)

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "===========================================================")
message(STATUS "    Planetary Thermophysical Model Configuration Summary")
message(STATUS "===========================================================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Architecture:      ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")
message(STATUS "SIMD Support:")
if(X86_64_ARCHITECTURE)
    if(COMPILER_SUPPORTS_AVX512)
        message(STATUS "  AVX-512:       ENABLED")
    else()
        message(STATUS "  AVX-512:       Not available")
    endif()
    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "  AVX2:          ENABLED")
    else()
        message(STATUS "  AVX2:          Not available")
    endif()
elseif(ARM_ARCHITECTURE)
    message(STATUS "  ARM NEON:      Available")
else()
    message(STATUS "  SIMD:          Not available (unsupported architecture)")
endif()
message(STATUS "")
message(STATUS "External Libraries:")
message(STATUS "  OpenMP:        Found")
message(STATUS "  Eigen3:        Found")
message(STATUS "  CSPICE:        ${CSPICE_LIBRARY}")
message(STATUS "  Embree:        ${EMBREE_LIBRARIES}")
message(STATUS "  CFITSIO:       ${CFITSIO_LIBRARIES}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  Optimizations: ${USE_AGGRESSIVE_OPTIMIZATIONS}")
message(STATUS "  Python:        ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Tests:         ${BUILD_TESTS}")
message(STATUS "===========================================================")
message(STATUS "")

# Print a concise compiler summary after languages are enabled
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Compiler path: ${CMAKE_CXX_COMPILER}")