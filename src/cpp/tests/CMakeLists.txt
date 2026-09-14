# CMakeLists.txt for the standalone cross-module tests.
#
# ⛔ THIS DIRECTORY WAS NOT IN THE BUILD BEFORE 2026-09-14. Three sources sat
# here (test_gpu_guardrails.cpp, test_machine_profile.cpp,
# test_riseeps_occlusion.cpp) with no CMake target anywhere in the tree; the
# only thing that ever compiled one is a hand-written `c++` line in INSTALL.md
# section for test_machine_profile. They are deliberately LEFT as they are here
# -- adding targets for three tests nobody has compiled in months is a separate
# change with its own gating -- but the gap is real and is filed as
# PTM-TESTS-DIR-UNWIRED in the ledger.

# The shadow-depth falsifier: synthetic meshes, closed-form umbral ceilings, no
# SPICE, no GPU, no Embree, no engine. Links only the pass, which is why the
# pass takes its ray casting as a callback.
add_executable(test_shadow_depth test_shadow_depth.cpp)
target_link_libraries(test_shadow_depth PRIVATE shadow_depth)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_shadow_depth PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(test_shadow_depth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
