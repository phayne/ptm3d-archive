# CMakeLists.txt for the flux-cube module
#
# ptm_flux_cube_geom is the ONE-TIME, CPU-ONLY geometry build (placement +
# hemisphere view-factor rows). It takes no GPU lock and opens no OpenCL
# context, so it is safe to run concurrently with production GPU windows.
#
# The engine-side post-solve pass lives in flux_cube_pass.cpp and is linked
# into ptm_thermal_sim; it LOADS a cache and never builds one.

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
