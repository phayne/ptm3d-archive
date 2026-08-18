# CMakeLists.txt for the flux-cube module
#
# ptm_flux_cube_geom is the ONE-TIME, CPU-ONLY geometry build (placement +
# hemisphere view-factor rows). It takes no GPU lock and opens no OpenCL
# context, so it is safe to run concurrently with production GPU windows.
#
# The engine-side post-solve pass lives in flux_cube_pass.cpp and is linked
# into ptm_thermal_sim; it LOADS a cache and never builds one.

# Engine-side post-solve pass. LOADS a cache, never builds one. No Embree and
# no illumination-model dependency: ray casting and the primary-body physics
# are injected by the caller, so the module is unit-testable against synthetic
# geometry and cannot acquire a second spelling of an engine formula.
add_library(flux_cubes STATIC
    flux_cube_pass.cpp
)
target_include_directories(flux_cubes PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/cpp
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flux_cubes PUBLIC OpenMP::OpenMP_CXX)
endif()

# Unit test for the pass: synthetic cavity + plane geometry, closed-form
# expectations, and negative controls that must fail.
add_executable(test_flux_cube_pass test_flux_cube_pass.cpp)
target_link_libraries(test_flux_cube_pass PRIVATE flux_cubes)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_flux_cube_pass PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(test_flux_cube_pass PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Standalone geometry builder. Embree only; deliberately NOT linked against the
# illumination model, whose scene lifetime is tied to a running simulation.
add_executable(ptm_flux_cube_geom flux_cube_geom.cpp)
target_link_libraries(ptm_flux_cube_geom PRIVATE
    mesh_utils
    ${EMBREE_LIBRARIES}
)
target_include_directories(ptm_flux_cube_geom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${EMBREE_INCLUDE_DIRS}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ptm_flux_cube_geom PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(ptm_flux_cube_geom PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
