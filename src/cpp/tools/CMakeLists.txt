# Add the map processor utility
add_executable(map_processor map_processor.cpp)
target_link_libraries(map_processor 
    spatial_maps
    mesh_utils
    constants
)

# Add the albedo map generator tool
add_executable(albedo_map_generator albedo_map_generator.cpp)
target_link_libraries(albedo_map_generator
    mesh_utils
    spatial_maps
    constants
    ${OpenMP_CXX_LIBRARIES}
)

# Lunar property-map generator (Diviner H-par raster -> per-face icosphere maps)
add_executable(lunar_property_map_generator lunar_property_map_generator.cpp)
target_link_libraries(lunar_property_map_generator
    mesh_utils
    constants
)
set_target_properties(lunar_property_map_generator
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Try to find GDAL with hints to the conda environment
# Prefer active conda env if available; fall back to europa_viz env path
set(CONDA_PREFIX_DIR "$ENV{CONDA_PREFIX}")
set(CONDA_ENV_PATH "$ENV{HOME}/anaconda3/envs/europa_viz")

set(_GDAL_HINTS "")
if(CONDA_PREFIX_DIR AND EXISTS "${CONDA_PREFIX_DIR}")
    list(APPEND _GDAL_HINTS
        ${CONDA_PREFIX_DIR}
        ${CONDA_PREFIX_DIR}/lib
        ${CONDA_PREFIX_DIR}/include
    )
endif()
if(CONDA_ENV_PATH AND EXISTS "${CONDA_ENV_PATH}")
    list(APPEND _GDAL_HINTS
        ${CONDA_ENV_PATH}
        ${CONDA_ENV_PATH}/lib
        ${CONDA_ENV_PATH}/include
    )
endif()

find_package(GDAL QUIET
    HINTS
    ${_GDAL_HINTS}
    PATH_SUFFIXES gdal
)

# If GDAL was found but include directory wasn't, set it manually
if(GDAL_FOUND)
    # Normalize include dir: prefer Conda include when library is in Conda
    if(NOT GDAL_INCLUDE_DIR AND DEFINED GDAL_INCLUDE_DIRS)
        set(GDAL_INCLUDE_DIR "${GDAL_INCLUDE_DIRS}")
    endif()

    if(CONDA_PREFIX_DIR AND GDAL_LIBRARIES AND GDAL_LIBRARIES MATCHES "${CONDA_PREFIX_DIR}")
        set(GDAL_INCLUDE_DIR "${CONDA_PREFIX_DIR}/include" CACHE PATH "GDAL include dir" FORCE)
        message(STATUS "GDAL library in conda env; forcing GDAL_INCLUDE_DIR = ${GDAL_INCLUDE_DIR}")
    elseif(CONDA_ENV_PATH AND GDAL_LIBRARIES AND GDAL_LIBRARIES MATCHES "${CONDA_ENV_PATH}")
        set(GDAL_INCLUDE_DIR "${CONDA_ENV_PATH}/include" CACHE PATH "GDAL include dir" FORCE)
        message(STATUS "GDAL library in europa_viz env; forcing GDAL_INCLUDE_DIR = ${GDAL_INCLUDE_DIR}")
    endif()

    # If GDAL_LIBRARIES is just the name, resolve to a full path within hints
    if(GDAL_LIBRARIES STREQUAL "gdal" OR GDAL_LIBRARIES STREQUAL "GDAL::GDAL")
        find_library(GDAL_LIBRARY_FILE
            NAMES gdal
            HINTS ${_GDAL_HINTS}
            PATHS ${_GDAL_HINTS}
        )
        if(GDAL_LIBRARY_FILE)
            set(GDAL_LIBRARIES "${GDAL_LIBRARY_FILE}" CACHE FILEPATH "GDAL library" FORCE)
            message(STATUS "Resolved GDAL library: ${GDAL_LIBRARIES}")
        endif()
    endif()
endif()

# Only build dem_mesh_generator if GDAL is properly configured
if(GDAL_FOUND AND GDAL_LIBRARIES)
    message(STATUS "GDAL found: Building dem_mesh_generator")
    message(STATUS "GDAL include dir: ${GDAL_INCLUDE_DIR}")
    message(STATUS "GDAL library: ${GDAL_LIBRARIES}")

    add_executable(dem_mesh_generator dem_mesh_generator.cpp)
    target_include_directories(dem_mesh_generator PRIVATE
        ${GDAL_INCLUDE_DIR}
    )
    target_link_libraries(dem_mesh_generator
        PRIVATE
        mesh_utils
        spatial_maps
        constants
        ${GDAL_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
    )
    # Ensure runtime finds conda libs on macOS (use rpath)
    if(APPLE AND CONDA_PREFIX_DIR)
        target_link_options(dem_mesh_generator PRIVATE "-Wl,-rpath,${CONDA_PREFIX_DIR}/lib")
    endif()

    # Compiler flag to silence the infinity warning
    target_compile_options(dem_mesh_generator PRIVATE -Wno-nan-infinity-disabled)

    set_target_properties(dem_mesh_generator
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
else()
    message(WARNING "GDAL not properly configured: dem_mesh_generator will not be built")
    message(STATUS "  GDAL_FOUND: ${GDAL_FOUND}")
    message(STATUS "  GDAL_LIBRARIES: ${GDAL_LIBRARIES}")
endif()

# Equilibration large-dt accuracy test
add_executable(test_equilibration_dt test_equilibration_dt.cpp)
target_link_libraries(test_equilibration_dt
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_equilibration_dt
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Chunked-run statistics merge tool. Combines per-chunk raw statistics
# (statistics/_raw/) from N chunks into a single set of merged stats
# (max/avg/min) plus the Schorghofer ice-stability depth recomputed
# from the merged ⟨E(z)⟩ accumulator.
add_executable(ptm_merge_chunks ptm_merge_chunks.cpp)
target_link_libraries(ptm_merge_chunks
    PRIVATE
    ice_table
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(ptm_merge_chunks
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add the crater mesh generator tool (no external dependencies)
add_executable(crater_mesh_generator crater_mesh_generator.cpp)
set_target_properties(crater_mesh_generator
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Set output directory for the other tools
set_target_properties(map_processor albedo_map_generator
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

cmake_minimum_required(VERSION 3.10)

# Collect tool sources
set(TOOL_SOURCES
    config_parser.cpp
)

# Create tools library
add_library(ptm_tools STATIC ${TOOL_SOURCES})

# Include directories
target_include_directories(ptm_tools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link yaml-cpp if available
if(YAML_CPP_FOUND)
    # Add include directories
    if(YAML_CPP_INCLUDE_DIRS)
        target_include_directories(ptm_tools PUBLIC ${YAML_CPP_INCLUDE_DIRS})
        message(STATUS "ptm_tools: Added include dirs: ${YAML_CPP_INCLUDE_DIRS}")
    endif()
    
    # THE FIX: Find the actual library file, don't use just the name
    if(YAML_CPP_LIBRARIES STREQUAL "yaml-cpp")
        # pkg-config returned just the name, find the actual file
        find_library(YAML_CPP_LIBRARY_PATH
            NAMES yaml-cpp
            PATHS
                ${YAML_CPP_LIBRARY_DIRS}
                ${YAML_CPP_LIBDIR}
                /usr/local/lib
                /opt/homebrew/lib
                $ENV{CONDA_PREFIX}/lib
            NO_DEFAULT_PATH
        )
        
        if(YAML_CPP_LIBRARY_PATH)
            target_link_libraries(ptm_tools PUBLIC ${YAML_CPP_LIBRARY_PATH})
            message(STATUS "ptm_tools: Linked yaml-cpp (resolved): ${YAML_CPP_LIBRARY_PATH}")
        else()
            message(WARNING "ptm_tools: Could not find yaml-cpp library file")
        endif()
    elseif(TARGET yaml-cpp)
        # It's a CMake target
        target_link_libraries(ptm_tools PUBLIC yaml-cpp)
        message(STATUS "ptm_tools: Linked yaml-cpp (target)")
    elseif(YAML_CPP_LIBRARIES)
        # It's already a path
        target_link_libraries(ptm_tools PUBLIC ${YAML_CPP_LIBRARIES})
        message(STATUS "ptm_tools: Linked yaml-cpp (path): ${YAML_CPP_LIBRARIES}")
    else()
        message(WARNING "ptm_tools: yaml-cpp found but cannot determine how to link")
    endif()
else()
    message(STATUS "ptm_tools: Building without yaml-cpp")
endif()

# Suppress yaml-cpp infinity warnings on macOS
if(APPLE AND YAML_CPP_FOUND)
    target_compile_options(ptm_tools PRIVATE -Wno-nan-infinity-disabled)
endif()

# Set C++ standard
set_target_properties(ptm_tools PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(ptm_tools PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()

message(STATUS "Configured ptm_tools library")