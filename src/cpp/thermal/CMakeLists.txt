cmake_minimum_required(VERSION 3.10)

# Collect source files (excluding bindings and backup files)
file(GLOB THERMAL_SOURCES "*.cpp")
list(REMOVE_ITEM THERMAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/thermal_model_bindings.cpp")
# Remove any backup/temp files (e.g., -MacL4864.cpp patterns)
list(FILTER THERMAL_SOURCES EXCLUDE REGEX ".*-Mac[A-Z][0-9]+\\.cpp$")

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


# C1c lateral operator: the ONLY thermal source compiled with OpenMP (its loops are per-element
# independent and its reductions use fixed chunks, so results do not depend on the thread count).
# The rest of libthermal keeps its existing flags.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    separate_arguments(_lat_omp_flags UNIX_COMMAND "${OpenMP_CXX_FLAGS}")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/lateral_conduction.cpp
        PROPERTIES COMPILE_OPTIONS "${_lat_omp_flags}")
    # The runtime library only -- NOT the OpenMP::OpenMP_CXX target, whose interface compile
    # options would switch OpenMP on for every other thermal source.
    target_link_libraries(thermal PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

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

# Add optimized adaptive header to installation
install(FILES 
    thermal_model.h
    thermal_adaptive.h
    thermal_adaptive_optimized.h 
    DESTINATION include/thermal
)

# Apply additional optimization flags for thermal calculations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # -fno-finite-math-only MUST follow -ffast-math: without it, finite-math-only
    # makes std::isnan/std::isfinite guards in the thermal solver no-ops.
    target_compile_options(thermal PRIVATE -O3 -ffast-math -fno-finite-math-only -ftree-vectorize)
    
    # Interprocedural optimization (LTO) — native/production profile only.
    # Under clang, INTERPROCEDURAL_OPTIMIZATION emits LLVM-bitcode objects into
    # libthermal.a; GNU ld (the Linux default) cannot read bitcode without the
    # LLVM/gold plugin and fails the final link with "file format not
    # recognized". Apple ld handles it via libLTO, which is why macOS never hit
    # it. Gate LTO to PTM_ARCH_NATIVE=ON so the native build keeps it
    # (byte-identical to before), while the portable/CPU profile
    # (PTM_ARCH_NATIVE=OFF, used by CI) links with stock ld.
    if(PTM_ARCH_NATIVE)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT IPO_SUPPORTED)
        if(IPO_SUPPORTED)
            message(STATUS "Interprocedural optimization enabled for thermal library")
            set_target_properties(thermal PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    else()
        message(STATUS "PTM_ARCH_NATIVE=OFF: LTO disabled for thermal library "
                       "(portable link; GNU ld cannot read clang LTO bitcode)")
    endif()
endif()

# Suppress specific warnings for thermal calculations
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