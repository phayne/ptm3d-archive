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
)

set(CAMERA_HEADERS
    camera_model.h
    photometric_model.h
    camera_simulator.h
    ptm_output_reader.h
    trajectory_reader.h
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
