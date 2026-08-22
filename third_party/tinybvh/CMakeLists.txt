# Vendored single-header BVH library (jbikker/tinybvh).
# Used for the CWBVH8 builder; see README_VENDORED.md.
add_library(tinybvh INTERFACE)
target_include_directories(tinybvh INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# NO ISA flags here, deliberately. This target used to carry
# `target_compile_options(tinybvh INTERFACE -mavx2 -mfma)` so that the TU
# defining TINYBVH_IMPLEMENTATION (cwbvh_builder.cpp) would get AVX2+FMA. On an
# INTERFACE library that flag is not scoped to that TU: it propagates through
# ptm_gpu (which links tinybvh PUBLIC) to radiation, illumination_model,
# ptm_thermal_sim and everything downstream, while thermal/, ice/, maps/,
# tools/, fileio/ and mesh/ keep the project baseline ISA.
#
# A split ISA is not merely a performance oddity, it is an ABI split. Eigen
# derives EIGEN_MAX_ALIGN_BYTES from the enabled ISA (16 baseline / 32 AVX /
# 64 AVX-512), and on macOS EIGEN_MALLOC_ALREADY_ALIGNED is 1 only at 16. So
# baseline TUs allocate Eigen storage with plain malloc()/free() while AVX TUs
# use handmade_aligned_malloc()/handmade_aligned_free() (an over-allocating,
# offset-pointer scheme). The two are not interchangeable, the linker keeps one
# copy of each shared Eigen weak symbol, and the mismatch shows up as an
# invalid free -> malloc_report -> SIGABRT far from the flag that caused it.
#
# The ISA now comes from the project-wide selection in the root CMakeLists.txt
# (PTM_ARCH_NATIVE / ENABLE_AVX2 / ENABLE_AVX512), which applies the same flags
# to every target. tinybvh autodetects __AVX2__ and falls back to a scalar
# builder when it is absent, so PTM_ARCH_NATIVE=OFF simply gets the scalar
# CWBVH build -- which is what "portable x86-64 baseline" is supposed to mean.
# The root CMakeLists.txt asserts that uniformity at configure time; see its
# "ISA uniformity assertion" section (_ptm_effective_isa).
