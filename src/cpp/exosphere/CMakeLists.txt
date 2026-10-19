# CMakeLists.txt for Exosphere module - Fixed for macOS/AppleClang

cmake_minimum_required(VERSION 3.12)

# =====================================================================================
# Core library: physics + PTM integration
# =====================================================================================
add_library(exosphere_core STATIC
    exosphere_physics.cpp
    ptm_exosphere_integration.cpp
    ptm_exosphere_albedo_response.cpp
    sputtering_model.cpp
)

# Back-compat alias so existing target_link_libraries(... exosphere) still works
add_library(exosphere ALIAS exosphere_core)

target_include_directories(exosphere_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(exosphere_core
    PUBLIC
    constants
    mesh_utils
    Eigen3::Eigen
)

# =====================================================================================
# SIMD and OpenMP for core lib
# =====================================================================================
include(CheckCXXCompilerFlag)

if(APPLE)
    message(STATUS "Configuring Exosphere for macOS/AppleClang")
endif()

# x86 ISA opt-ins only. X86_64_ARCHITECTURE / ARM_ARCHITECTURE come from the
# root CMakeLists.txt. On arm64 these probes are skipped outright: -mavx* is
# meaningless there, and the root selects the ISA once (-mcpu=apple-m1).
if(X86_64_ARCHITECTURE)
# AVX2 (native/production profile only; PTM_ARCH_NATIVE=OFF keeps x86-64 baseline)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2 AND PTM_ARCH_NATIVE)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        target_compile_options(exosphere_core PRIVATE -mavx2 -mfma)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(exosphere_core PRIVATE -mavx2 -mfma -mavx)
    else()
        target_compile_options(exosphere_core PRIVATE -mavx2 -mfma)
    endif()
    target_compile_definitions(exosphere_core PRIVATE AVX2_ENABLED)
    message(STATUS "AVX2 support enabled for exosphere_core")
endif()

# AVX-512 (native/production profile only; PTM_ARCH_NATIVE=OFF keeps x86-64 baseline)
CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512 AND PTM_ARCH_NATIVE)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        target_compile_options(exosphere_core PRIVATE
            -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl)
    else()
        target_compile_options(exosphere_core PRIVATE
            -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl)
    endif()
    target_compile_definitions(exosphere_core PRIVATE AVX512_ENABLED)
    message(STATUS "AVX512 support enabled for exosphere_core")
endif()
endif()  # X86_64_ARCHITECTURE

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(exosphere_core PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(exosphere_core PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(exosphere_core PUBLIC ${OpenMP_omp_LIBRARY})
        else()
            find_library(LIBOMP_LIB NAMES omp libomp
                PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib /usr/local/lib)
            if(LIBOMP_LIB)
                target_link_libraries(exosphere_core PUBLIC ${LIBOMP_LIB})
                message(STATUS "Found libomp: ${LIBOMP_LIB}")
            else()
                message(WARNING "OpenMP library not found; OpenMP may not link on macOS")
            endif()
        endif()
        if(OpenMP_CXX_INCLUDE_DIRS)
            target_include_directories(exosphere_core PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
        else()
            target_include_directories(exosphere_core PRIVATE
                /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
        endif()
    else()
        target_link_libraries(exosphere_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
    message(STATUS "OpenMP support enabled for exosphere_core")
else()
    message(WARNING "OpenMP not found; parallel regions will be disabled")
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # -fno-finite-math-only MUST follow -ffast-math: finite-math-only otherwise
    # turns std::isnan/std::isfinite guards into no-ops (see top-level CMakeLists).
    # -march=native/-xHost/-mtune=native and /arch:AVX2 tune to the build host and
    # SIGILL on a different target CPU -- gate them to PTM_ARCH_NATIVE=ON. The
    # portable profile (PTM_ARCH_NATIVE=OFF) emits generic baseline codegen only.
    if(PTM_ARCH_NATIVE AND ARM_ARCHITECTURE)
        # arm64: NO per-target -march=native. The root already applies the one
        # project-wide CPU selection (-mcpu=apple-m1 on Apple Silicon); adding
        # -march=native here gave exosphere_core a second -march/-mcpu token
        # set, which the root's Eigen-ABI uniformity assertion rejects
        # (FATAL_ERROR at configure: "built across 2 different Eigen ABIs").
        target_compile_options(exosphere_core PRIVATE -O3 -ffast-math -fno-finite-math-only)
        message(STATUS "exosphere_core: arm64, project-wide CPU flags only (no -march=native)")
    elseif(PTM_ARCH_NATIVE)
        if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
            target_compile_options(exosphere_core PRIVATE -O3 -ffast-math -fno-finite-math-only -march=native)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            target_compile_options(exosphere_core PRIVATE -O3 -march=native -mtune=native -ffast-math -fno-finite-math-only)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(exosphere_core PRIVATE -O3 -march=native -ffast-math -fno-finite-math-only)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            target_compile_options(exosphere_core PRIVATE -O3 -xHost -fp-model=fast)
        elseif(MSVC)
            target_compile_options(exosphere_core PRIVATE /O2 /arch:AVX2 /fp:fast)
        endif()
    else()
        if(MSVC)
            target_compile_options(exosphere_core PRIVATE /O2 /fp:fast)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            target_compile_options(exosphere_core PRIVATE -O3 -fp-model=fast)
        else()
            target_compile_options(exosphere_core PRIVATE -O3 -ffast-math -fno-finite-math-only)
        endif()
    endif()
endif()

# =====================================================================================
# CLI executable
# =====================================================================================
add_executable(ptm_exosphere
    ptm_exosphere_main.cpp
)

target_include_directories(ptm_exosphere PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${CMAKE_SOURCE_DIR}/src/cpp/fileio
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(ptm_exosphere
    PRIVATE
    exosphere_core
    thermal          # --albedo-response diurnal: the thermal engine's depth grid
    mesh_utils
    fileio
    constants
    Eigen3::Eigen
)

# macOS OpenMP flags on the CLI as well
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(ptm_exosphere PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(ptm_exosphere PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(ptm_exosphere PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(ptm_exosphere PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(ptm_exosphere PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Uniform output location and filename
set_target_properties(ptm_exosphere
    PROPERTIES
    OUTPUT_NAME "ptm_exosphere_main"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_compile_features(ptm_exosphere PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(ptm_exosphere PRIVATE stdc++fs)
endif()

# =====================================================================================
# Optional test
# =====================================================================================
if(BUILD_TESTS)
    add_executable(test_exosphere_physics
        test_transport_physics.cpp
    )
    target_include_directories(test_exosphere_physics PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cpp
        ${CMAKE_SOURCE_DIR}/src/cpp/constants
        ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(test_exosphere_physics
        PRIVATE
        exosphere_core
        constants
        Eigen3::Eigen
    )
    if(OpenMP_CXX_FOUND)
        if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
            if(OpenMP_CXX_FLAGS)
                string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
                target_compile_options(test_exosphere_physics PRIVATE ${OMP_FLAGS_LIST})
            else()
                target_compile_options(test_exosphere_physics PRIVATE -Xpreprocessor -fopenmp)
            endif()
            if(OpenMP_omp_LIBRARY)
                target_link_libraries(test_exosphere_physics PRIVATE ${OpenMP_omp_LIBRARY})
            elseif(LIBOMP_LIB)
                target_link_libraries(test_exosphere_physics PRIVATE ${LIBOMP_LIB})
            endif()
        else()
            target_link_libraries(test_exosphere_physics PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
    set_target_properties(test_exosphere_physics
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Cutoff benchmark test (always build)
add_executable(test_cutoff_benchmark
    test_cutoff_benchmark.cpp
)
target_include_directories(test_cutoff_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_cutoff_benchmark
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_cutoff_benchmark PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_cutoff_benchmark PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_cutoff_benchmark PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_cutoff_benchmark PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_cutoff_benchmark PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_cutoff_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Real data smooth kernel test
add_executable(test_smooth_kernel_real_data
    test_smooth_kernel_real_data.cpp
)
target_include_directories(test_smooth_kernel_real_data PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_smooth_kernel_real_data
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_smooth_kernel_real_data PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_smooth_kernel_real_data PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_smooth_kernel_real_data PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_smooth_kernel_real_data PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_smooth_kernel_real_data PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_smooth_kernel_real_data
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Pentagon bias analysis test
add_executable(test_pentagon_bias
    test_pentagon_bias.cpp
)
target_include_directories(test_pentagon_bias PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_pentagon_bias
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_pentagon_bias PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_pentagon_bias PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_pentagon_bias PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_pentagon_bias PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_pentagon_bias PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_pentagon_bias
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Equilibrium bias analysis test
add_executable(test_equilibrium_bias
    test_equilibrium_bias.cpp
)
target_include_directories(test_equilibrium_bias PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_equilibrium_bias
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_equilibrium_bias PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_equilibrium_bias PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_equilibrium_bias PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_equilibrium_bias PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_equilibrium_bias PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_equilibrium_bias
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# EXO-01 / EXO-02 regression test (always build; self-contained, exit code 0 = pass)
add_executable(test_bessel_k0_and_areas
    test_bessel_k0_and_areas.cpp
)
target_include_directories(test_bessel_k0_and_areas PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_bessel_k0_and_areas
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_bessel_k0_and_areas PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_bessel_k0_and_areas PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_bessel_k0_and_areas PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_bessel_k0_and_areas PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_bessel_k0_and_areas PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_bessel_k0_and_areas
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --albedo-coupling temperature-refresh test (always build; self-contained, exit code 0 = pass)
add_executable(test_albedo_refresh
    test_albedo_refresh.cpp
)
target_include_directories(test_albedo_refresh PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_albedo_refresh
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_albedo_refresh PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_albedo_refresh PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_albedo_refresh PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_albedo_refresh PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_albedo_refresh PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_albedo_refresh
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --albedo-response diurnal unit test (always build; header-only code under test,
# exit code 0 = pass): the diurnal-shape response against an independent nonlinear
# periodic column, and AlbedoRefreshTracker::updateWith.
add_executable(test_diurnal_albedo_response
    test_diurnal_albedo_response.cpp
)
target_include_directories(test_diurnal_albedo_response PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
)
target_compile_features(test_diurnal_albedo_response PRIVATE cxx_std_17)
set_target_properties(test_diurnal_albedo_response
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --ice-fraction-from-albedo=hapke-intimate unit test (always build; header-only
# code under test, exit code 0 = pass; optional arg: per-face Bond albedo .bin)
add_executable(test_ice_fraction_from_albedo
    test_ice_fraction_from_albedo.cpp
)
target_include_directories(test_ice_fraction_from_albedo PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
)
target_compile_features(test_ice_fraction_from_albedo PRIVATE cxx_std_17)
set_target_properties(test_ice_fraction_from_albedo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# =====================================================================================
# Install and summary
# =====================================================================================
install(TARGETS ptm_exosphere
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES exosphere_physics.h ptm_exosphere_integration.h distance_cache.h spherical_index.h sputtering_model.h ice_crystallization_model.h ice_layer_model.h multitimescale_evolution.h hapke_mixing.h albedo_refresh.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ptm/exosphere
)

message(STATUS "")
message(STATUS "Exosphere Module Configuration:")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  AVX2: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "  AVX512: ${COMPILER_SUPPORTS_AVX512}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "  OpenMP lib: ${OpenMP_omp_LIBRARY}")
endif()
message(STATUS "")
# Multiscale snapshot timing (PTM-636): snapshot L is the state after exactly L
# epochs and time_steps.txt lists every snapshot (exit code 0 = pass).
add_executable(test_multiscale_snapshot_time
    test_multiscale_snapshot_time.cpp
)
target_include_directories(test_multiscale_snapshot_time PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_multiscale_snapshot_time
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_multiscale_snapshot_time PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_multiscale_snapshot_time PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_multiscale_snapshot_time PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_multiscale_snapshot_time PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_multiscale_snapshot_time PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_multiscale_snapshot_time
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Sputter mass accounting (PTM-642) and transport-kernel options (PTM-643):
# stoichiometry, fate closure, deposition rescale, end-to-end audit (exit code 0 = pass).
add_executable(test_sputter_mass_accounting
    test_sputter_mass_accounting.cpp
)
target_include_directories(test_sputter_mass_accounting PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_sputter_mass_accounting
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_sputter_mass_accounting PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_sputter_mass_accounting PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_sputter_mass_accounting PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_sputter_mass_accounting PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_sputter_mass_accounting PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_sputter_mass_accounting
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# PTM-649 sputtering parameter set (derivations, tau(T), normalization, ion maps)
add_executable(test_sputter_parameters
    test_sputter_parameters.cpp
)
target_include_directories(test_sputter_parameters PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_sputter_parameters
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_sputter_parameters PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_sputter_parameters PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_sputter_parameters PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_sputter_parameters PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_sputter_parameters PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_sputter_parameters
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# PTM-644 super-step deposition consistency (unit + ico3 end to end)
add_executable(test_superstep_deposition
    test_superstep_deposition.cpp
)
target_include_directories(test_superstep_deposition PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_superstep_deposition
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_superstep_deposition PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_superstep_deposition PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_superstep_deposition PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_superstep_deposition PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_superstep_deposition PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_superstep_deposition
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Lag-layer diffusion barrier (unit: analytic, veneer, mass, t = 0 invariance)
add_executable(test_lag_barrier
    test_lag_barrier.cpp
)
target_include_directories(test_lag_barrier PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_lag_barrier
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_lag_barrier PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_lag_barrier PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_lag_barrier PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_lag_barrier PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_lag_barrier PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_lag_barrier
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Gardened lag + sputter split (PTM-669; Costello cross-check, Spencer, d*, mass)
add_executable(test_lag_gardening
    test_lag_gardening.cpp
)
target_include_directories(test_lag_gardening PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_lag_gardening
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_lag_gardening PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_lag_gardening PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_lag_gardening PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_lag_gardening PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_lag_gardening PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_lag_gardening
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Fixed-temperature sputter transport operator vs the direct kernel (PTM-671)
add_executable(test_sputter_operator
    test_sputter_operator.cpp
)
target_include_directories(test_sputter_operator PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/constants
    ${CMAKE_SOURCE_DIR}/src/cpp/mesh
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_sputter_operator
    PRIVATE
    exosphere_core
    constants
    mesh_utils
    Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
    if(APPLE AND (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang"))
        if(OpenMP_CXX_FLAGS)
            string(REPLACE " " ";" OMP_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
            target_compile_options(test_sputter_operator PRIVATE ${OMP_FLAGS_LIST})
        else()
            target_compile_options(test_sputter_operator PRIVATE -Xpreprocessor -fopenmp)
        endif()
        if(OpenMP_omp_LIBRARY)
            target_link_libraries(test_sputter_operator PRIVATE ${OpenMP_omp_LIBRARY})
        elseif(LIBOMP_LIB)
            target_link_libraries(test_sputter_operator PRIVATE ${LIBOMP_LIB})
        endif()
    else()
        target_link_libraries(test_sputter_operator PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
set_target_properties(test_sputter_operator
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
