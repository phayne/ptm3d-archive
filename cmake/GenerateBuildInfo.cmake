# GenerateBuildInfo.cmake -- resolve the engine's build identity into
# ptm_build_info.h. Run with `cmake -P`, both at configure time (so the header
# exists before the first dependency scan) and as a pre-build step (so a commit
# made between builds is not reported as the previous one).
#
# Inputs, all -D on the command line:
#   PTM_BUILD_INFO_IN   the .h.in template
#   PTM_BUILD_INFO_OUT  the header to write
#   PTM_SRC_DIR         repository root, for `git rev-parse`
#   PTM_VERSION         project version
#   PTM_BUILD_TYPE      CMAKE_BUILD_TYPE
#   PTM_GPU_LABEL       "ON" or "OFF"
#   PTM_COMPILER        compiler id + version
#   PTM_EMBREE_LIB      resolved Embree library path (results-affecting; see
#                       the note in ptm_build_info.h.in)
#   PTM_EMBREE_MAX_ISA  that Embree's EMBREE_MAX_ISA, or "unknown"
#
# configure_file() writes only when the content differs, which is the whole
# point: a no-op rebuild must not perturb the executable's md5.
#
# A checkout with no git, or an exported tarball, is not an error -- the SHA
# becomes "unknown" and the run says "unknown" rather than something plausible
# and wrong.

set(PTM_GIT_SHA "unknown")

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PTM_SRC_DIR}/.git")
    # EXISTS matches both a .git directory and the .git FILE that a git
    # worktree uses, so this works in the worktrees this project develops in.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=10 HEAD
        WORKING_DIRECTORY "${PTM_SRC_DIR}"
        OUTPUT_VARIABLE _ptm_sha
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _ptm_rc)
    if(_ptm_rc EQUAL 0 AND NOT _ptm_sha STREQUAL "")
        set(PTM_GIT_SHA "${_ptm_sha}")
    endif()
endif()

configure_file("${PTM_BUILD_INFO_IN}" "${PTM_BUILD_INFO_OUT}" @ONLY)
