# Mesh Library
add_library(mesh_utils
    mesh_utils.cpp
    lateral_adjacency.cpp   # C1c lateral conduction adjacency (unused unless lateral is enabled)
)

# Include directories
target_include_directories(mesh_utils
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link dependencies
target_link_libraries(mesh_utils
    PUBLIC
    constants
    thermal
    spatial_maps
    ${EMBREE_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Ellipsoid mesh geometry gate (see test_ellipsoid_mesh.cpp for what it proves)
add_executable(test_ellipsoid_mesh
    test_ellipsoid_mesh.cpp
)
target_link_libraries(test_ellipsoid_mesh
    PRIVATE
    mesh_utils
)
set_target_properties(test_ellipsoid_mesh
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Enable VTK reader in this library too
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
    target_compile_definitions(mesh_utils PRIVATE PTM_USE_VTK_READER)
    target_include_directories(mesh_utils PRIVATE ${VTK_INCLUDE_DIRS})
    target_link_libraries(mesh_utils PRIVATE ${VTK_LIBRARIES})
    # Autoinit for this shared library (needed on some platforms)
    vtk_module_autoinit(TARGETS mesh_utils MODULES ${VTK_LIBRARIES})
  else()
    message(WARNING "VTK not found for mesh_utils; PTM_USE_VTK_READER disabled for this target")
  endif()
endif()