# Illumination Model Library

# Add all source files
file(GLOB SOURCES *.cpp)
file(GLOB HEADERS *.h *.hpp)

# Create the illumination model library
add_library(illumination_model
    illumination_model.cpp
)

# Include directories
target_include_directories(illumination_model
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CSPICE_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
)

# Link dependencies
target_link_libraries(illumination_model
    PUBLIC
    constants
    mesh_utils
    ${CSPICE_LIBRARY}
    ${EMBREE_LIBRARIES}  # Add this line
    OpenMP::OpenMP_CXX
)