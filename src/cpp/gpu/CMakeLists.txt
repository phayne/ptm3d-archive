# GPU acceleration module for PTM thermal model
# Supports HIP (AMD/NVIDIA), CUDA, and OpenCL backends

cmake_minimum_required(VERSION 3.10)

# This file is only included when ENABLE_GPU is ON (from root CMakeLists.txt)

# Determine GPU backend
if(NOT DEFINED GPU_BACKEND)
    set(GPU_BACKEND "AUTO" CACHE STRING "GPU backend: AUTO, HIP, CUDA, OPENCL, or NONE")
endif()

set(GPU_FOUND FALSE)
set(GPU_BACKEND_USED "NONE")

# Auto-detect GPU backend if requested
if(GPU_BACKEND STREQUAL "AUTO" OR GPU_BACKEND STREQUAL "HIP")
    # Try to find HIP first (works for both AMD and NVIDIA)
    find_package(hip QUIET)
    if(hip_FOUND)
        set(GPU_FOUND TRUE)
        set(GPU_BACKEND_USED "HIP")
        message(STATUS "Found HIP: ${hip_VERSION}")
    endif()
endif()

if(NOT GPU_FOUND AND (GPU_BACKEND STREQUAL "AUTO" OR GPU_BACKEND STREQUAL "CUDA"))
    # Try CUDA
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(GPU_FOUND TRUE)
        set(GPU_BACKEND_USED "CUDA")
        message(STATUS "Found CUDA: ${CMAKE_CUDA_COMPILER_VERSION}")
    endif()
endif()

# Try OpenCL (works on macOS, Linux, Windows)
if(NOT GPU_FOUND AND (GPU_BACKEND STREQUAL "AUTO" OR GPU_BACKEND STREQUAL "OPENCL"))
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(GPU_FOUND TRUE)
        set(GPU_BACKEND_USED "OPENCL")
        message(STATUS "Found OpenCL: ${OpenCL_VERSION_STRING}")
    endif()
endif()

if(NOT GPU_FOUND)
    message(STATUS "No GPU backend found - GPU acceleration disabled")
    message(STATUS "  To enable GPU support, install:")
    message(STATUS "    - HIP (ROCm) for AMD GPUs on Linux")
    message(STATUS "    - CUDA toolkit for NVIDIA GPUs")
    message(STATUS "    - OpenCL SDK (works on macOS with AMD GPUs)")
    # Create stub library and return
    add_library(ptm_gpu INTERFACE)
    target_compile_definitions(ptm_gpu INTERFACE PTM_GPU_ENABLED=0)
    return()
endif()

message(STATUS "GPU backend: ${GPU_BACKEND_USED}")

# Create the GPU library based on backend
if(GPU_BACKEND_USED STREQUAL "HIP")
    # HIP-based GPU library
    # Note: For HIP, we compile .cpp files with hipcc
    set(GPU_SOURCES
        thermal_gpu.cpp
        thermal_kernels.cpp
    )

    # Set HIP as the language for these files (compiles with hipcc)
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)

    add_library(ptm_gpu ${GPU_SOURCES})

    target_link_libraries(ptm_gpu
        PUBLIC
        hip::device
    )

    target_compile_definitions(ptm_gpu
        PUBLIC
        PTM_GPU_ENABLED=1
        PTM_GPU_HIP=1
    )

    # HIP compile options
    target_compile_options(ptm_gpu PRIVATE
        -fgpu-rdc  # Relocatable device code for separate compilation
        -O3        # Optimization
    )

elseif(GPU_BACKEND_USED STREQUAL "CUDA")
    # CUDA-based GPU library
    # For CUDA, kernel file needs .cu extension
    set(GPU_SOURCES
        thermal_gpu.cpp
    )

    set(CUDA_SOURCES
        thermal_kernels.cpp  # Will be renamed or we use .cu
    )

    # Rename .cpp to .cu for CUDA compilation
    configure_file(thermal_kernels.cpp thermal_kernels.cu COPYONLY)
    list(APPEND CUDA_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/thermal_kernels.cu)

    add_library(ptm_gpu ${GPU_SOURCES} ${CUDA_SOURCES})

    target_compile_definitions(ptm_gpu
        PUBLIC
        PTM_GPU_ENABLED=1
        PTM_GPU_CUDA=1
    )

    # CUDA compile options
    set_target_properties(ptm_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 14
    )

    # Architecture flags - auto-detect or use common ones
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86" CACHE STRING "CUDA architectures")
    endif()
    set_target_properties(ptm_gpu PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

elseif(GPU_BACKEND_USED STREQUAL "OPENCL")
    # OpenCL-based GPU library
    # Works on macOS with AMD GPUs, Linux, Windows
    set(GPU_SOURCES
        thermal_opencl.cpp
        radiosity_opencl.cpp
        bvh_opencl.cpp
    )

    add_library(ptm_gpu ${GPU_SOURCES})

    target_link_libraries(ptm_gpu
        PUBLIC
        OpenCL::OpenCL
    )

    target_compile_definitions(ptm_gpu
        PUBLIC
        PTM_GPU_ENABLED=1
        PTM_GPU_OPENCL=1
    )

    # OpenCL compile options
    target_compile_options(ptm_gpu PRIVATE
        -O3
    )

    # Copy kernel files to build directory for runtime loading (optional)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/thermal_kernels.cl
        ${CMAKE_CURRENT_BINARY_DIR}/thermal_kernels.cl
        COPYONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/radiosity_kernels.cl
        ${CMAKE_CURRENT_BINARY_DIR}/radiosity_kernels.cl
        COPYONLY
    )

    # Install kernel files
    install(FILES
        thermal_kernels.cl
        radiosity_kernels.cl
        DESTINATION share/ptm/kernels
    )
endif()

# Common settings for all backends
target_include_directories(ptm_gpu
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with thermal library for shared types
target_link_libraries(ptm_gpu
    PUBLIC
    thermal
)

# Install headers
install(FILES
    gpu_config.h
    gpu_memory.h
    gpu_utils.h
    thermal_gpu.h
    thermal_kernels.h
    DESTINATION include/gpu
)

# Install OpenCL-specific headers if using OpenCL
if(GPU_BACKEND_USED STREQUAL "OPENCL")
    install(FILES
        opencl_config.h
        thermal_opencl.h
        radiosity_opencl.h
        bvh_opencl.h
        DESTINATION include/gpu
    )
endif()
