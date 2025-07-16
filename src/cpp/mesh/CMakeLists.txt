# Mesh Library
add_library(mesh_utils
    mesh_utils.cpp
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