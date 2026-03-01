# Orbital mechanics library
# Provides Keplerian orbit propagation and body-fixed frame rotation
# for generic planetary body support

add_library(orbital
    kepler_propagator.cpp
    body_fixed_frame.cpp
)

target_include_directories(orbital PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link to thermal for ThermophysicalProperties (header-only dependency)
target_link_libraries(orbital PUBLIC
    thermal
)

if(OpenMP_FOUND)
    target_link_libraries(orbital PRIVATE OpenMP::OpenMP_CXX)
endif()
