cmake_minimum_required(VERSION 3.10)

# Collect source files (excluding bindings)
file(GLOB THERMAL_SOURCES "*.cpp")
list(REMOVE_ITEM THERMAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/thermal_model_bindings.cpp")

add_library(thermal ${THERMAL_SOURCES})

# Include directories
target_include_directories(thermal 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(thermal
    PUBLIC
    spatial_maps  # Add this to link with the spatial_maps library
)

# Set up compilation flags
target_compile_features(thermal PUBLIC cxx_std_14)

# Apply AVX optimizations to the thermal library specifically
if(COMPILER_SUPPORTS_AVX512)
    target_compile_options(thermal PRIVATE ${AVX512_FLAGS})
    target_compile_definitions(thermal PRIVATE AVX512_ENABLED)
elseif(COMPILER_SUPPORTS_AVX2)
    target_compile_options(thermal PRIVATE ${AVX2_FLAGS})
    target_compile_definitions(thermal PRIVATE AVX2_ENABLED)
endif()

# Apply additional optimization flags for thermal calculations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(thermal PRIVATE -O3 -ffast-math -ftree-vectorize)
    
    # Enable interprocedural optimization if supported
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED)
    if(IPO_SUPPORTED)
        message(STATUS "Interprocedural optimization enabled for thermal library")
        set_target_properties(thermal PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# Add this to the thermal module's CMakeLists.txt
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nan-infinity-disabled")

# For Python bindings - create a separate target
if(BUILD_PYTHON_BINDINGS)
    # Make sure the module directory exists
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python_modules)
    
    # Create the Python module
    pybind11_add_module(thermal_model_cpp thermal_model_bindings.cpp)
    
    # Link with the thermal library
    target_link_libraries(thermal_model_cpp PRIVATE thermal)
    
    # Apply the same optimizations to the Python module
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(thermal_model_cpp PRIVATE ${AVX2_FLAGS})
        target_compile_definitions(thermal_model_cpp PRIVATE AVX2_ENABLED)
    endif()
    
    # Make sure the bindings can find the headers
    target_include_directories(thermal_model_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cpp
        ${CMAKE_SOURCE_DIR}/src/cpp/constants
    )
    
    # Set output directory for the Python module
    set_target_properties(thermal_model_cpp
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_modules"
    )
    
    # Print information about the module
    add_custom_command(TARGET thermal_model_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Thermal model Python module built at: $<TARGET_FILE:thermal_model_cpp>"
    )
endif()