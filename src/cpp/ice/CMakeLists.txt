add_library(ice_table
    ice_table.cpp
)

target_include_directories(ice_table
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ice_table
    PUBLIC
    ${OpenMP_CXX_LIBRARIES}
)

set_target_properties(ice_table PROPERTIES
    OUTPUT_NAME ice_table
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
