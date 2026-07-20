cmake_minimum_required(VERSION 3.10)

# Create fileio library
add_library(fileio STATIC
    memory_mapped_io.cpp
)

# Set include directories for this library
target_include_directories(fileio PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(fileio PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "fileio: Linking with OpenMP")
else()
    message(WARNING "fileio: OpenMP not found, building without parallel support")
endif()

# Platform-specific settings
if(UNIX AND NOT APPLE)
    # Linux specific - for mmap, msync, etc.
    target_link_libraries(fileio PUBLIC rt)
    message(STATUS "fileio: Linking with rt library (Linux)")
elseif(APPLE)
    # macOS doesn't need special libraries for mmap
    message(STATUS "fileio: Building for macOS (no special libs needed for mmap)")
endif()

# Enable SIMD optimizations if available
if(DEFINED SIMD_FLAGS)
    target_compile_options(fileio PRIVATE ${SIMD_FLAGS})
    message(STATUS "fileio: Using SIMD flags: ${SIMD_FLAGS}")
endif()

# Add optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(fileio PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()

# Set C++ standard
set_target_properties(fileio PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export for use in other targets
set(FILEIO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

message(STATUS "Configured fileio library for memory-mapped I/O")