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

# Set output directory for the tools
set_target_properties(map_processor albedo_map_generator
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)