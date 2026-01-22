# CMakeLists.txt for Viewshed module
# Place this file in: src/cpp/viewshed/CMakeLists.txt

# ============================================================================
# Viewshed Library
# ============================================================================

# Source files
set(VIEWSHED_SOURCES
    viewshed_analyzer.cpp
)

# Header files
set(VIEWSHED_HEADERS
    viewshed_analyzer.h
)

# Create the viewshed library
add_library(viewshed STATIC ${VIEWSHED_SOURCES})

# Include directories
target_include_directories(viewshed PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${EMBREE_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CSPICE_INCLUDE_DIR}
)

# Link dependencies - using libraries from parent CMakeLists.txt
target_link_libraries(viewshed PUBLIC
    illumination_model   # FIX: correct target name
    mesh_utils           # FIX: correct target name
    thermal
    constants
    ${EMBREE_LIBRARIES}
    ${CSPICE_LIBRARIES}
)

# OpenMP support (inherited from parent)
if(OpenMP_CXX_FOUND)
    target_link_libraries(viewshed PUBLIC OpenMP::OpenMP_CXX)
endif()

# HDF5 support (optional, disabled by default to avoid link issues on macOS/Conda)
option(VIEWSHED_ENABLE_HDF5 "Enable HDF5 export in viewshed module" OFF)

if(VIEWSHED_ENABLE_HDF5)
    find_package(HDF5 COMPONENTS CXX QUIET)
    if(HDF5_FOUND)
        target_include_directories(viewshed PRIVATE ${HDF5_INCLUDE_DIRS})
        target_compile_definitions(viewshed PRIVATE HAS_HDF5)

        if(TARGET HDF5::CXX)
            target_link_libraries(viewshed PUBLIC HDF5::CXX)
        elseif(HDF5_CXX_LIBRARIES)
            target_link_libraries(viewshed PUBLIC ${HDF5_CXX_LIBRARIES})
        endif()

        if(TARGET HDF5::HDF5)
            target_link_libraries(viewshed PUBLIC HDF5::HDF5)
        elseif(HDF5_LIBRARIES)
            target_link_libraries(viewshed PUBLIC ${HDF5_LIBRARIES})
        endif()

        # Fallbacks for macOS/Conda
        set(_H5_HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/opt/homebrew/lib")
        if(NOT TARGET HDF5::CXX AND NOT HDF5_CXX_LIBRARIES)
            find_library(HDF5_CXX_LIB NAMES hdf5_cpp PATHS ${_H5_HINTS})
            find_library(HDF5_C_LIB  NAMES hdf5      PATHS ${_H5_HINTS})
            if(HDF5_CXX_LIB AND HDF5_C_LIB)
                target_link_libraries(viewshed PUBLIC ${HDF5_CXX_LIB} ${HDF5_C_LIB})
            endif()
        endif()
        find_library(SZIP_LIB szip PATHS ${_H5_HINTS})
        if(SZIP_LIB)
            target_link_libraries(viewshed PUBLIC ${SZIP_LIB})
        endif()
        find_package(ZLIB QUIET)
        if(ZLIB_FOUND)
            target_link_libraries(viewshed PUBLIC ZLIB::ZLIB)
        else()
            find_library(ZLIB_LIB z PATHS ${_H5_HINTS})
            if(ZLIB_LIB)
                target_link_libraries(viewshed PUBLIC ${ZLIB_LIB})
            endif()
        endif()

        message(STATUS "HDF5 support enabled for viewshed module")
    else()
        message(WARNING "HDF5 requested but not found; disabling HAS_HDF5")
    endif()
else()
    message(STATUS "Viewshed HDF5 disabled (set -DVIEWSHED_ENABLE_HDF5=ON to enable)")
endif()

# GDAL support for GeoTIFF (optional)
find_package(GDAL QUIET)
if(GDAL_FOUND)
    target_include_directories(viewshed PRIVATE ${GDAL_INCLUDE_DIRS})
    target_link_libraries(viewshed PUBLIC ${GDAL_LIBRARIES})
    target_compile_definitions(viewshed PRIVATE HAS_GDAL)
    message(STATUS "GDAL support enabled for viewshed module")
endif()

# SIMD optimizations (inherited from parent)
if(COMPILER_SUPPORTS_AVX512)
    target_compile_options(viewshed PRIVATE ${AVX512_FLAGS})
    target_compile_definitions(viewshed PRIVATE USE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2)
    target_compile_options(viewshed PRIVATE ${AVX2_FLAGS})
    target_compile_definitions(viewshed PRIVATE USE_AVX2)
endif()

# ============================================================================
# Viewshed Executable for PRISM Dark Horse
# ============================================================================

# Resolve example source path robustly
set(DARK_HORSE_SOURCE "")
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dark_horse_viewshed_example.cpp)
    set(DARK_HORSE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/dark_horse_viewshed_example.cpp)
elseif(EXISTS ${CMAKE_SOURCE_DIR}/examples/viewshed/dark_horse_viewshed_example.cpp)
    set(DARK_HORSE_SOURCE ${CMAKE_SOURCE_DIR}/examples/viewshed/dark_horse_viewshed_example.cpp)
else()
    message(FATAL_ERROR "dark_horse_viewshed_example.cpp not found in either:
      ${CMAKE_CURRENT_SOURCE_DIR} or ${CMAKE_SOURCE_DIR}/examples/viewshed")
endif()

# Build the Dark Horse analysis executable
add_executable(dark_horse_viewshed
    ${DARK_HORSE_SOURCE}
)

# Use keyword signature consistently
target_link_libraries(dark_horse_viewshed
    PRIVATE
    viewshed
    illumination_model   # FIX
    mesh_utils           # FIX
    thermal
    constants
    fileio
    ${EMBREE_LIBRARIES}
    ${CSPICE_LIBRARIES}
    ${CFITSIO_LIBRARIES}
)

# Enable VTK reader support for thermal VTU/PVD parsing
option(DARK_HORSE_ENABLE_VTK "Enable VTK reader in dark_horse_viewshed" ON)
if(DARK_HORSE_ENABLE_VTK)
    # Try to find VTK libraries commonly needed for VTU/PVD IO
    find_package(VTK QUIET COMPONENTS IOXML CommonDataModel CommonCore)
    if(VTK_FOUND)
        target_compile_definitions(dark_horse_viewshed PRIVATE PTM_USE_VTK_READER)
        if(TARGET VTK::IOXML)
            target_link_libraries(dark_horse_viewshed PRIVATE VTK::IOXML)
        endif()
        if(TARGET VTK::CommonDataModel)
            target_link_libraries(dark_horse_viewshed PRIVATE VTK::CommonDataModel)
        endif()
        if(TARGET VTK::CommonCore)
            target_link_libraries(dark_horse_viewshed PRIVATE VTK::CommonCore)
        endif()
        # Some distros require additional VTK modules; link if available
        if(TARGET VTK::IOCore)
            target_link_libraries(dark_horse_viewshed PRIVATE VTK::IOCore)
        endif()
        if(TARGET VTK::CommonSystem)
            target_link_libraries(dark_horse_viewshed PRIVATE VTK::CommonSystem)
        endif()
        message(STATUS "VTK support enabled for dark_horse_viewshed")
    else()
        message(WARNING "VTK requested for dark_horse_viewshed but not found; VTU/PVD thermal parsing will be disabled")
    endif()
endif()

# Propagate optional defs/includes to the example executables only when enabled
if(VIEWSHED_ENABLE_HDF5 AND HDF5_FOUND)
    target_compile_definitions(dark_horse_viewshed PRIVATE HAS_HDF5)
    target_include_directories(dark_horse_viewshed PRIVATE ${HDF5_INCLUDE_DIRS})
    if(TARGET HDF5::CXX)
        target_link_libraries(dark_horse_viewshed PRIVATE HDF5::CXX)
    elseif(HDF5_CXX_LIBRARIES)
        target_link_libraries(dark_horse_viewshed PRIVATE ${HDF5_CXX_LIBRARIES})
    endif()
    if(TARGET HDF5::HDF5)
        target_link_libraries(dark_horse_viewshed PRIVATE HDF5::HDF5)
    elseif(HDF5_LIBRARIES)
        target_link_libraries(dark_horse_viewshed PRIVATE ${HDF5_LIBRARIES})
    endif()
    find_library(HDF5_CXX_LIB NAMES hdf5_cpp PATHS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/opt/homebrew/lib")
    find_library(HDF5_C_LIB  NAMES hdf5      PATHS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/opt/homebrew/lib")
    if(HDF5_CXX_LIB AND HDF5_C_LIB)
        target_link_libraries(dark_horse_viewshed PRIVATE ${HDF5_CXX_LIB} ${HDF5_C_LIB})
    endif()
    if(APPLE AND DEFINED ENV{CONDA_PREFIX})
        target_link_options(dark_horse_viewshed PRIVATE "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
    endif()
endif()

# Optional: place the binary in the top-level bin
set_target_properties(dark_horse_viewshed
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ============================================================================
# Optional: Viewshed CLI tool
# ============================================================================

option(BUILD_VIEWSHED_CLI "Build viewshed command-line tool" ON)
if(BUILD_VIEWSHED_CLI AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/viewshed_cli.cpp)
    add_executable(viewshed_cli viewshed_cli.cpp)
    target_link_libraries(viewshed_cli
        PRIVATE
        viewshed
        illumination_model
        mesh_utils
        thermal
        constants
        fileio
        ${EMBREE_LIBRARIES}
        ${CSPICE_LIBRARIES}
        ${CFITSIO_LIBRARIES}
    )
    if(VIEWSHED_ENABLE_HDF5 AND HDF5_FOUND)
        target_compile_definitions(viewshed_cli PRIVATE HAS_HDF5)
        target_include_directories(viewshed_cli PRIVATE ${HDF5_INCLUDE_DIRS})
        if(TARGET HDF5::CXX)
            target_link_libraries(viewshed_cli PRIVATE HDF5::CXX)
        elseif(HDF5_CXX_LIBRARIES)
            target_link_libraries(viewshed_cli PRIVATE ${HDF5_CXX_LIBRARIES})
        endif()
        if(TARGET HDF5::HDF5)
            target_link_libraries(viewshed_cli PRIVATE HDF5::HDF5)
        elseif(HDF5_LIBRARIES)
            target_link_libraries(viewshed_cli PRIVATE ${HDF5_LIBRARIES})
        endif()
        find_library(HDF5_CXX_LIB NAMES hdf5_cpp PATHS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/opt/homebrew/lib")
        find_library(HDF5_C_LIB  NAMES hdf5      PATHS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/opt/homebrew/lib")
        if(HDF5_CXX_LIB AND HDF5_C_LIB)
            target_link_libraries(viewshed_cli PRIVATE ${HDF5_CXX_LIB} ${HDF5_C_LIB})
        endif()
        if(APPLE AND DEFINED ENV{CONDA_PREFIX})
            target_link_options(viewshed_cli PRIVATE "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
        endif()
    endif()
    install(TARGETS viewshed_cli RUNTIME DESTINATION bin)
endif()

# ============================================================================
# Installation
# ============================================================================

# Install library
install(TARGETS viewshed
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install headers
install(FILES ${VIEWSHED_HEADERS}
    DESTINATION include/ptm/viewshed
)

# Install Dark Horse executable
install(TARGETS dark_horse_viewshed
    RUNTIME DESTINATION bin
)

# ============================================================================
# Testing
# ============================================================================

if(BUILD_TESTS)
    # Add viewshed-specific tests here
    # add_executable(test_viewshed test_viewshed.cpp)
    # target_link_libraries(test_viewshed viewshed)
    # add_test(NAME ViewshedTest COMMAND test_viewshed)
endif()

message(STATUS "Viewshed module configured")
message(STATUS "  HDF5 support: ${HDF5_FOUND}")
message(STATUS "  GDAL support: ${GDAL_FOUND}")