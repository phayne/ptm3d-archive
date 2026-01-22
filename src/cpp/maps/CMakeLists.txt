# Make sure to find CFITSIO
find_package(CFITSIO REQUIRED)

# Print found CFITSIO info
message(STATUS "Found CFITSIO: ${CFITSIO_LIBRARIES}")
message(STATUS "CFITSIO include directories: ${CFITSIO_INCLUDE_DIRS}")

# Spatial mapping library
add_library(spatial_maps
    spatial_map.cpp
    fits_reader.cpp
    albedo_manager.cpp
)

# Include directories
target_include_directories(spatial_maps
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CFITSIO_INCLUDE_DIRS}  # Use plural include_dirs
)

# Link dependencies
target_link_libraries(spatial_maps
    PUBLIC
    ${CFITSIO_LIBRARIES}  # Make CFITSIO link public
)

# Add installation rules if needed
install(TARGETS spatial_maps
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)