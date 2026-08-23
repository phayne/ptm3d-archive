# CMakeLists.txt for radiation exchange module
# Handles view factor computation and radiosity solving for thermal re-radiation

add_library(radiation STATIC
    view_factor.cpp
    nearfield_quadrature.cpp
    radiation_exchange.cpp
    face_selection.cpp
    hmatrix_view_factor.cpp
    cluster_tree.cpp
    mc_view_factor.cpp
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

# Two-band radiosity PTM-08 severity test: Gauss-Seidel vs Jacobi on a synthetic
# concave enclosure with Σ_j F_ij > 1 (vf_row_norm_ active), visible Bond albedo A.
# Regression fixture: exits nonzero while GS omits the row-norm, 0 once A1 lands.
add_executable(test_two_band_rownorm test_two_band_rownorm.cpp)
target_link_libraries(test_two_band_rownorm PRIVATE
    radiation
    mesh_utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_two_band_rownorm PRIVATE OpenMP::OpenMP_CXX)
endif()
if(ENABLE_GPU AND TARGET ptm_gpu)
    target_link_libraries(test_two_band_rownorm PRIVATE ptm_gpu)
    target_compile_definitions(test_two_band_rownorm PRIVATE PTM_GPU_OPENCL)
endif()
set_target_properties(test_two_band_rownorm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# STAGE 0 of the flux cubes: the host-side reconstruction of the terrain
# radiosity B_vis/B_ir must equal the solver's own converged B to 1e-12
# relative, with negative controls that are REQUIRED to fail. Falsifies
# design decision D3 (reconstruct on the host, no GPU readback).
add_executable(test_flux_cube_identity test_flux_cube_identity.cpp)
target_link_libraries(test_flux_cube_identity PRIVATE
    radiation
    mesh_utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_flux_cube_identity PRIVATE OpenMP::OpenMP_CXX)
endif()
if(ENABLE_GPU AND TARGET ptm_gpu)
    target_link_libraries(test_flux_cube_identity PRIVATE ptm_gpu)
    target_compile_definitions(test_flux_cube_identity PRIVATE PTM_GPU_OPENCL)
endif()
set_target_properties(test_flux_cube_identity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Closed-form gate on the PRODUCTION in-engine view-factor path. test_bowl_crater
# builds its view factors from its own inline centroid kernel and never calls
# ViewFactorCalculator, which is why FC-08 (a 2x far-field inflation) lived for
# months under a green run of it. This target drives the real Embree-occluded
# ViewFactorCalculator against the Ingersoll spherical cap and a coaxial
# parallel-disk pair, and carries its own fault-injection negative controls.
add_executable(test_vf_production test_vf_production.cpp)
target_link_libraries(test_vf_production PRIVATE
    radiation
    mesh_utils
    ${EMBREE_LIBRARIES}
)
target_include_directories(test_vf_production PRIVATE ${EMBREE_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_vf_production PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(test_vf_production PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Ingersoll bowl-crater validation driven by the PRODUCTION radiation module
# (Embree-occluded ViewFactorCalculator + RadiositySolver). Writes .npy outputs.
add_executable(bowl_crater_ptm bowl_crater_ptm.cpp)
target_link_libraries(bowl_crater_ptm PRIVATE
    radiation
    mesh_utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bowl_crater_ptm PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(bowl_crater_ptm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Dump per-face VF row sums (terrain view factor) from a saved H-matrix, to
# compare model floor->wall coupling against the geometric DEM ground truth.
add_executable(vf_rowsum_dump vf_rowsum_dump.cpp)
target_link_libraries(vf_rowsum_dump PRIVATE radiation mesh_utils)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vf_rowsum_dump PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(vf_rowsum_dump PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Emit a correctly-dimensioned, current-format (HMAT/v2) EMPTY base .vf (nnz=0)
# via the production serializer, so a rolling-VF site can satisfy the VF-load
# gate without the multi-GB build (overridden by PTM_HMATRIX_VF_REPLACE). See
# BUGS PTM-24 (stale MFFV/v1 base -> silent recompute wedge).
add_executable(make_empty_vf make_empty_vf.cpp)
target_link_libraries(make_empty_vf PRIVATE radiation mesh_utils)
if(OpenMP_CXX_FOUND)
    target_link_libraries(make_empty_vf PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(make_empty_vf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Apply a saved H-matrix to an arbitrary per-face vector (y = F*v), to diagnose
# WHERE a face's view-factor coupling points (e.g. floor -> warm rim).
add_executable(vf_apply vf_apply.cpp)
target_link_libraries(vf_apply PRIVATE radiation mesh_utils)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vf_apply PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(vf_apply PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Concrete floor->warm pairwise examination (visibility + cos_i facet/macroscopic).
add_executable(vf_examine_pairs vf_examine_pairs.cpp)
target_link_libraries(vf_examine_pairs PRIVATE radiation mesh_utils ${EMBREE_LIBRARIES})
target_include_directories(vf_examine_pairs PRIVATE ${EMBREE_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(vf_examine_pairs PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(vf_examine_pairs PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Ground-truth VF by cosine-weighted hemisphere Monte-Carlo ray casting (Embree).
add_executable(vf_hemisphere_mc vf_hemisphere_mc.cpp)
target_link_libraries(vf_hemisphere_mc PRIVATE radiation mesh_utils ${EMBREE_LIBRARIES})
target_include_directories(vf_hemisphere_mc PRIVATE ${EMBREE_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(vf_hemisphere_mc PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(vf_hemisphere_mc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Geometric ground-truth floor indirect FLUX (cosine MC weighted by eps*sigma*Tmax^4),
# to decide whether the floor-center coupling deficit is a VF under-count or physical.
add_executable(vf_mc_flux vf_mc_flux.cpp)
target_link_libraries(vf_mc_flux PRIVATE radiation mesh_utils ${EMBREE_LIBRARIES})
target_include_directories(vf_mc_flux PRIVATE ${EMBREE_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(vf_mc_flux PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(vf_mc_flux PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Single-face VF visibility probe (robust-mode toggle) for the fine-mesh
# floor->far-rim coupling-collapse diagnosis.
add_executable(vf_face_probe vf_face_probe.cpp)
target_link_libraries(vf_face_probe PRIVATE radiation mesh_utils ${EMBREE_LIBRARIES})
target_include_directories(vf_face_probe PRIVATE ${EMBREE_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(vf_face_probe PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(vf_face_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Standalone validator for the Smith (1967) beamed-emission chi factor
# (header-only math in smith_beaming.h; no mesh/Embree/SPICE dependency).
add_executable(test_beam_chi test_beam_chi.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_beam_chi PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(test_beam_chi PROPERTIES
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

# Grazing-Sun shadow-ray artifact check (lift sweep)
add_executable(illum_check illum_check.cpp)
target_link_libraries(illum_check PRIVATE radiation mesh_utils ${EMBREE_LIBRARIES})
target_include_directories(illum_check PRIVATE ${EMBREE_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(illum_check PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(illum_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
