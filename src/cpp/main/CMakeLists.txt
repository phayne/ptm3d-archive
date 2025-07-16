# Main Thermal Simulation Executable
add_executable(ptm_thermal_sim
    ptm_thermal_sim.cpp  # Keep original filename until the file is renamed
)

# Include directories
target_include_directories(ptm_thermal_sim
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CSPICE_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
)

# Link dependencies with PRIVATE to avoid transitive dependencies
target_link_libraries(ptm_thermal_sim
    PRIVATE
    mesh_utils
    illumination_model
    constants
    thermal
    ${CSPICE_LIBRARIES}
    ${EMBREE_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    spatial_maps
    ${CFITSIO_LIBRARIES}
)

set_target_properties(ptm_thermal_sim
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
