# Constants Library
add_library(constants
    constants.cpp
)

# Include directories
target_include_directories(constants
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)