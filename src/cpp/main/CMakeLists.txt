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
    ${CMAKE_SOURCE_DIR}/src/cpp/fileio
)

# Link dependencies with PRIVATE to avoid transitive dependencies
target_link_libraries(ptm_thermal_sim
    PRIVATE
    mesh_utils
    illumination_model
    constants
    thermal
    ice_table
    fileio
    radiation
    flux_cubes
    ${CSPICE_LIBRARIES}
    ${EMBREE_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    spatial_maps
    ${CFITSIO_LIBRARIES}
    ptm_tools
)

set_target_properties(ptm_thermal_sim
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ============================================================================
# VTK Reader Support
# ============================================================================
option(PTM_USE_VTK_READER "Use VTK C++ reader for mesh import (replicates PyVista)" ON)

if(PTM_USE_VTK_READER)
  find_package(VTK COMPONENTS
    CommonCore
    CommonDataModel
    IOLegacy
    IOXML
    FiltersCore
    FiltersGeneral
    FiltersGeometry          # <-- required for vtkGeometryFilter
  QUIET)

  if(VTK_FOUND)
    message(STATUS "VTK found; enabling VTK-based mesh reader")
    add_compile_definitions(PTM_USE_VTK_READER)
    target_include_directories(ptm_thermal_sim PRIVATE ${VTK_INCLUDE_DIRS})
    target_link_libraries(ptm_thermal_sim PRIVATE ${VTK_LIBRARIES})
    # Ensure VTK module autoinit for this executable
    vtk_module_autoinit(TARGETS ptm_thermal_sim MODULES ${VTK_LIBRARIES})
  else()
    message(WARNING "VTK not found; PTM_USE_VTK_READER disabled")
  endif()
endif()