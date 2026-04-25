# CMakeLists.txt for radiation exchange module
# Handles view factor computation and radiosity solving for thermal re-radiation

add_library(radiation STATIC
    view_factor.cpp
    radiation_exchange.cpp
    face_selection.cpp
    hmatrix_view_factor.cpp
    cluster_tree.cpp
)

target_include_directories(radiation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${EMBREE_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Link libraries (OpenMP::OpenMP_CXX handles compile flags automatically)
target_link_libraries(radiation PUBLIC
    mesh_utils
    ${EMBREE_LIBRARIES}
)

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(radiation PUBLIC OpenMP::OpenMP_CXX)
endif()

# GPU support for radiosity solver
if(ENABLE_GPU AND TARGET ptm_gpu)
    target_link_libraries(radiation PUBLIC ptm_gpu)
    target_compile_definitions(radiation PUBLIC PTM_GPU_OPENCL)
endif()

# SIMD flags for optimized computations
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(radiation PRIVATE ${AVX2_FLAGS})
endif()

if(COMPILER_SUPPORTS_AVX512)
    target_compile_options(radiation PRIVATE ${AVX512_FLAGS})
endif()

# Test executable for bowl crater validation
add_executable(test_bowl_crater test_bowl_crater.cpp)
target_link_libraries(test_bowl_crater PRIVATE
    radiation
    mesh_utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_bowl_crater PRIVATE OpenMP::OpenMP_CXX)
endif()
# Link GPU library for OpenCL radiosity test
if(ENABLE_GPU AND TARGET ptm_gpu)
    target_link_libraries(test_bowl_crater PRIVATE ptm_gpu)
    target_compile_definitions(test_bowl_crater PRIVATE PTM_GPU_OPENCL)
endif()
set_target_properties(test_bowl_crater PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Benchmark executable for H-matrix vs sparse comparison
add_executable(benchmark_hmatrix benchmark_hmatrix.cpp)
target_link_libraries(benchmark_hmatrix PRIVATE
    radiation
    mesh_utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_hmatrix PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(benchmark_hmatrix PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Benchmark executable: CPU Embree packets vs GPU OpenCL BVH shadow rays
add_executable(benchmark_gpu_visibility benchmark_gpu_visibility.cpp)
target_link_libraries(benchmark_gpu_visibility PRIVATE
    radiation
    mesh_utils
    ${EMBREE_LIBRARIES}
)
target_include_directories(benchmark_gpu_visibility PRIVATE
    ${EMBREE_INCLUDE_DIRS}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_gpu_visibility PRIVATE OpenMP::OpenMP_CXX)
endif()
if(ENABLE_GPU AND TARGET ptm_gpu)
    target_link_libraries(benchmark_gpu_visibility PRIVATE ptm_gpu)
    target_compile_definitions(benchmark_gpu_visibility PRIVATE PTM_GPU_OPENCL)
endif()
# Match radiation library's SIMD flags so the 16-wide packet path compiles.
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(benchmark_gpu_visibility PRIVATE ${AVX2_FLAGS})
endif()
if(COMPILER_SUPPORTS_AVX512)
    target_compile_options(benchmark_gpu_visibility PRIVATE ${AVX512_FLAGS})
endif()
set_target_properties(benchmark_gpu_visibility PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
