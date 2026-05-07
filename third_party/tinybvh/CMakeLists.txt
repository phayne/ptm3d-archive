# Vendored single-header BVH library (jbikker/tinybvh).
# Used for the CWBVH8 builder; see README_VENDORED.md.
add_library(tinybvh INTERFACE)
target_include_directories(tinybvh INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Enable AVX2 + FMA on the TU that defines TINYBVH_IMPLEMENTATION
# (cwbvh_builder.cpp). tinybvh autodetects __AVX2__ and falls back to
# scalar otherwise. Cascade Lake / Ice Lake / Mac Pro Xeon supports AVX2.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavx2 _has_mavx2)
check_cxx_compiler_flag(-mfma  _has_mfma)
if(_has_mavx2 AND _has_mfma)
    target_compile_options(tinybvh INTERFACE -mavx2 -mfma)
endif()
