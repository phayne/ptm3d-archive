# CMakeLists.txt for Camera Simulation module
# Place this file in: src/cpp/camera/CMakeLists.txt

# ============================================================================
# Camera Simulation Library
# ============================================================================

set(CAMERA_SOURCES
    camera_model.cpp
    photometric_model.cpp
    camera_simulator.cpp
    ptm_output_reader.cpp
    trajectory_reader.cpp
    camera_yaml.cpp
)

set(CAMERA_HEADERS
    camera_model.h
    photometric_model.h
    camera_simulator.h
    ptm_output_reader.h
    trajectory_reader.h
    camera_yaml.h
)

add_library(camera_sim STATIC ${CAMERA_SOURCES})

target_include_directories(camera_sim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${EMBREE_INCLUDE_DIRS}
    ${CSPICE_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)

target_link_libraries(camera_sim PUBLIC
    mesh_utils
    constants
    ${EMBREE_LIBRARIES}
    ${CSPICE_LIBRARIES}
    ${CFITSIO_LIBRARIES}
)

# Link yaml-cpp for the camera YAML definition file loader.
# HAVE_YAML_CPP is defined globally by the project-level CMake when yaml-cpp
# is found; camera_yaml.cpp compiles to a stub when the flag is absent.
# (Mirrors the pattern used in src/cpp/tools/CMakeLists.txt — when
# YAML_CPP_LIBRARIES is just the bare name "yaml-cpp", we need to resolve
# it to a full path via find_library so linking succeeds.)
if(YAML_CPP_FOUND)
    if(YAML_CPP_INCLUDE_DIRS)
        target_include_directories(camera_sim PUBLIC ${YAML_CPP_INCLUDE_DIRS})
    endif()
    if(YAML_CPP_LIBRARIES STREQUAL "yaml-cpp")
        find_library(YAML_CPP_LIBRARY_PATH_CAMERA
            NAMES yaml-cpp
            HINTS
                ${YAML_CPP_LIBRARY_DIRS}
                ${YAML_CPP_LIBDIR}
                /opt/homebrew/lib
                /usr/local/lib
                $ENV{CONDA_PREFIX}/lib
        )
        if(YAML_CPP_LIBRARY_PATH_CAMERA)
            target_link_libraries(camera_sim PUBLIC ${YAML_CPP_LIBRARY_PATH_CAMERA})
        endif()
    elseif(TARGET yaml-cpp)
        target_link_libraries(camera_sim PUBLIC yaml-cpp)
    elseif(YAML_CPP_LIBRARIES)
        target_link_libraries(camera_sim PUBLIC ${YAML_CPP_LIBRARIES})
    endif()
endif()

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(camera_sim PUBLIC OpenMP::OpenMP_CXX)
endif()

# SIMD optimizations (inherited from parent)
if(COMPILER_SUPPORTS_AVX512)
    target_compile_options(camera_sim PRIVATE ${AVX512_FLAGS})
    target_compile_definitions(camera_sim PRIVATE USE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2)
    target_compile_options(camera_sim PRIVATE ${AVX2_FLAGS})
    target_compile_definitions(camera_sim PRIVATE USE_AVX2)
endif()

# ============================================================================
# Camera Simulator Executable
# ============================================================================

add_executable(ptm_camera_sim camera_sim_main.cpp)

target_link_libraries(ptm_camera_sim PRIVATE
    camera_sim
    mesh_utils
    constants
    ${EMBREE_LIBRARIES}
    ${CSPICE_LIBRARIES}
    ${CFITSIO_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ptm_camera_sim PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(ptm_camera_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS camera_sim
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${CAMERA_HEADERS}
    DESTINATION include/ptm/camera
)

install(TARGETS ptm_camera_sim
    RUNTIME DESTINATION bin
)

message(STATUS "Camera simulation module configured")
