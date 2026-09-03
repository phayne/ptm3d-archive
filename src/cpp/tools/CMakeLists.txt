# Add the map processor utility
add_executable(map_processor map_processor.cpp)
target_link_libraries(map_processor 
    spatial_maps
    mesh_utils
    constants
)

# Add the albedo map generator tool
add_executable(albedo_map_generator albedo_map_generator.cpp)
target_link_libraries(albedo_map_generator
    mesh_utils
    spatial_maps
    constants
    ${OpenMP_CXX_LIBRARIES}
)

# Lunar property-map generator (Diviner H-par raster -> per-face icosphere maps)
add_executable(lunar_property_map_generator lunar_property_map_generator.cpp)
target_link_libraries(lunar_property_map_generator
    mesh_utils
    constants
)
set_target_properties(lunar_property_map_generator
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Try to find GDAL with hints to the conda environment
# Prefer active conda env if available; fall back to europa_viz env path
set(CONDA_PREFIX_DIR "$ENV{CONDA_PREFIX}")
set(CONDA_ENV_PATH "$ENV{HOME}/anaconda3/envs/europa_viz")

set(_GDAL_HINTS "")
if(CONDA_PREFIX_DIR AND EXISTS "${CONDA_PREFIX_DIR}")
    list(APPEND _GDAL_HINTS
        ${CONDA_PREFIX_DIR}
        ${CONDA_PREFIX_DIR}/lib
        ${CONDA_PREFIX_DIR}/include
    )
endif()
if(CONDA_ENV_PATH AND EXISTS "${CONDA_ENV_PATH}")
    list(APPEND _GDAL_HINTS
        ${CONDA_ENV_PATH}
        ${CONDA_ENV_PATH}/lib
        ${CONDA_ENV_PATH}/include
    )
endif()

find_package(GDAL QUIET
    HINTS
    ${_GDAL_HINTS}
    PATH_SUFFIXES gdal
)

# If GDAL was found but include directory wasn't, set it manually
if(GDAL_FOUND)
    # Normalize the library variable BEFORE anything reads it.
    #
    # CMake's bundled FindGDAL module sets GDAL_LIBRARIES (plural). GDAL's own
    # installed CMake config package -- which Homebrew and many conda builds
    # ship, and which find_package prefers when present -- sets only
    # GDAL_LIBRARY (singular) plus the GDAL::GDAL imported target, leaving
    # GDAL_LIBRARIES EMPTY. The `GDAL_FOUND AND GDAL_LIBRARIES` gate below then
    # evaluated false on a machine that has a perfectly good GDAL, and every
    # GDAL tool was skipped with no error -- the configure log said
    # "GDAL_FOUND: 1" on the line above the skip. Close that hole here.
    if(NOT GDAL_LIBRARIES)
        if(GDAL_LIBRARY)
            set(GDAL_LIBRARIES "${GDAL_LIBRARY}")
            message(STATUS "GDAL config package: using GDAL_LIBRARY = ${GDAL_LIBRARIES}")
        elseif(TARGET GDAL::GDAL)
            # ASK THE TARGET WHERE IT IS. Taking the literal string
            # "GDAL::GDAL" here sent the resolution below off to `find_library`,
            # which searched the conda hints first and came back with a
            # DIFFERENT GDAL installation from the one find_package had just
            # located -- conda's 3.5 library against Homebrew's 3.12 headers.
            # The imported target knows its own file.
            get_target_property(_GDAL_IMP GDAL::GDAL IMPORTED_LOCATION)
            if(NOT _GDAL_IMP)
                foreach(_cfg RELEASE RELWITHDEBINFO MINSIZEREL DEBUG NOCONFIG)
                    get_target_property(_GDAL_IMP GDAL::GDAL
                                        IMPORTED_LOCATION_${_cfg})
                    if(_GDAL_IMP)
                        break()
                    endif()
                endforeach()
            endif()
            if(_GDAL_IMP AND EXISTS "${_GDAL_IMP}")
                set(GDAL_LIBRARIES "${_GDAL_IMP}")
                message(STATUS "GDAL config package: GDAL::GDAL resolves to "
                               "${GDAL_LIBRARIES}")
            else()
                set(GDAL_LIBRARIES GDAL::GDAL)
                message(STATUS "GDAL config package: using imported target GDAL::GDAL")
            endif()
        endif()
    endif()

    # RESOLVE THE LIBRARY TO A REAL PATH FIRST. This block used to run AFTER
    # the include-dir normalization below, and the ordering was the whole bug:
    # on a machine with both Homebrew and conda GDAL, `find_package` returns
    # the imported target `GDAL::GDAL` with GDAL_LIBRARY empty, so the branch
    # above sets GDAL_LIBRARIES to the literal string "GDAL::GDAL". The
    # include-dir rules then asked "does GDAL_LIBRARIES look like a conda
    # path?" -- of a string that is not a path at all -- and did nothing. Only
    # afterwards did this find_library replace it with CONDA'S libgdal, by
    # which point the include dir was settled (in fact EMPTY). The tools then
    # compiled against Homebrew's headers, which are on the default search
    # path, and linked conda's older library:
    #
    #   Undefined symbols for architecture x86_64:
    #     "GDALDataset::GetGeoTransform(double*) const"
    #
    # Measured 2026-09-02 on a fresh configure of origin/main ab7f2f9 in a
    # shell with CONDA_PREFIX=.../envs/europa_viz. It fails the WHOLE default
    # target build, which is what `./rebuild.sh` and `cmake --build build -j`
    # both do.
    if(GDAL_LIBRARIES STREQUAL "gdal" OR GDAL_LIBRARIES STREQUAL "GDAL::GDAL")
        find_library(GDAL_LIBRARY_FILE
            NAMES gdal
            HINTS ${_GDAL_HINTS}
            PATHS ${_GDAL_HINTS}
        )
        if(GDAL_LIBRARY_FILE)
            set(GDAL_LIBRARIES "${GDAL_LIBRARY_FILE}" CACHE FILEPATH "GDAL library" FORCE)
            message(STATUS "Resolved GDAL library: ${GDAL_LIBRARIES}")
        endif()
    endif()

    # Normalize include dir, from the library that was actually resolved.
    if(NOT GDAL_INCLUDE_DIR AND DEFINED GDAL_INCLUDE_DIRS)
        set(GDAL_INCLUDE_DIR "${GDAL_INCLUDE_DIRS}")
    endif()

    if(CONDA_PREFIX_DIR AND GDAL_LIBRARIES AND GDAL_LIBRARIES MATCHES "${CONDA_PREFIX_DIR}")
        set(GDAL_INCLUDE_DIR "${CONDA_PREFIX_DIR}/include" CACHE PATH "GDAL include dir" FORCE)
        message(STATUS "GDAL library in conda env; forcing GDAL_INCLUDE_DIR = ${GDAL_INCLUDE_DIR}")
    elseif(CONDA_ENV_PATH AND GDAL_LIBRARIES AND GDAL_LIBRARIES MATCHES "${CONDA_ENV_PATH}")
        set(GDAL_INCLUDE_DIR "${CONDA_ENV_PATH}/include" CACHE PATH "GDAL include dir" FORCE)
        message(STATUS "GDAL library in europa_viz env; forcing GDAL_INCLUDE_DIR = ${GDAL_INCLUDE_DIR}")
    elseif(GDAL_LIBRARIES AND IS_ABSOLUTE "${GDAL_LIBRARIES}")
        # The general case, which is what the two conda rules above are two
        # instances of: headers come from the SAME installation prefix as the
        # library. <prefix>/lib/libgdal.dylib -> <prefix>/include.
        get_filename_component(_GDAL_PREFIX "${GDAL_LIBRARIES}" DIRECTORY)
        get_filename_component(_GDAL_PREFIX "${_GDAL_PREFIX}" DIRECTORY)
        if(EXISTS "${_GDAL_PREFIX}/include/gdal_priv.h")
            set(GDAL_INCLUDE_DIR "${_GDAL_PREFIX}/include" CACHE PATH "GDAL include dir" FORCE)
            message(STATUS "GDAL headers taken from the resolved library's own "
                           "prefix: ${GDAL_INCLUDE_DIR}")
        endif()
    endif()
endif()

# Only build the GDAL-dependent DEM tools if GDAL is properly configured.
# dem_mesh_generator and dtm_to_ply_delaunay both need the GDAL C++ headers and
# library; the rest of PTM3D does not, so configure must still succeed without
# GDAL (with the warning at the bottom of this block naming exactly which
# targets were skipped).
#
# GDAL_INCLUDE_DIR IS PART OF THE GATE. Without it the tools still compiled --
# against whatever gdal_priv.h happened to sit on the default include path,
# which on this machine is a DIFFERENT GDAL from the one being linked. An
# empty include directory is not "use the default"; it is "we do not know
# which headers these are", and this build has no business guessing.
if(GDAL_FOUND AND GDAL_LIBRARIES AND GDAL_INCLUDE_DIR
        AND EXISTS "${GDAL_INCLUDE_DIR}/gdal_priv.h")
    message(STATUS "GDAL found: Building dem_mesh_generator, dtm_to_ply_delaunay")
    message(STATUS "GDAL include dir: ${GDAL_INCLUDE_DIR}")
    message(STATUS "GDAL library: ${GDAL_LIBRARIES}")

    add_executable(dem_mesh_generator dem_mesh_generator.cpp)
    # BEFORE: the project adds -I/usr/local/include at directory scope (Eigen,
    # yaml-cpp, Embree all live there on macOS), and on a machine with a
    # Homebrew GDAL that directory ALSO holds a gdal_priv.h. Without BEFORE it
    # wins over the GDAL installation this target was told to use, and the
    # object file is compiled against one GDAL and linked against another.
    target_include_directories(dem_mesh_generator BEFORE PRIVATE
        ${GDAL_INCLUDE_DIR}
    )
    target_link_libraries(dem_mesh_generator
        PRIVATE
        mesh_utils
        spatial_maps
        constants
        ${GDAL_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
    )
    # Ensure runtime finds conda libs on macOS (use rpath)
    if(APPLE AND CONDA_PREFIX_DIR)
        target_link_options(dem_mesh_generator PRIVATE "-Wl,-rpath,${CONDA_PREFIX_DIR}/lib")
    endif()

    # Compiler flag to silence the infinity warning
    target_compile_options(dem_mesh_generator PRIVATE -Wno-nan-infinity-disabled)

    set_target_properties(dem_mesh_generator
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # DTM -> PLY converter with true 2D Delaunay triangulation. This is the
    # binary the GUI dashboard's "Convert" and "ROI" actions invoke
    # (gui/dem_core/dem_runner.py). Header-only CDT comes from third_party/;
    # the source includes it as "CDT/include/CDT.h", so third_party/ itself is
    # the include root.
    add_executable(dtm_to_ply_delaunay dtm_to_ply_delaunay.cpp)
    # BEFORE, for the reason given on dem_mesh_generator above.
    target_include_directories(dtm_to_ply_delaunay BEFORE PRIVATE
        ${GDAL_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/third_party
    )
    # IEEE-754 semantics are REQUIRED here, so override the project-wide
    # -ffast-math for this target. CDT's predicates.h is Shewchuk's adaptive
    # exact-arithmetic orient2d/incircle: its error bounds assume strict
    # round-to-nearest and no reassociation or FMA contraction. Under
    # -ffast-math the compiler is free to break both, which silently corrupts
    # the triangulation on the near-degenerate triangles that ROI thinning
    # produces -- exactly the case this tool exists for. With these flags the
    # output is byte-identical to the pre-port standalone build.
    # (Target options land after the directory-scope ones on the command line,
    # so these win. Gated on the compiler family that understands them.)
    target_compile_options(dtm_to_ply_delaunay PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-fast-math>
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>
    )
    target_link_libraries(dtm_to_ply_delaunay
        PRIVATE
        ${GDAL_LIBRARIES}
    )
    if(APPLE AND CONDA_PREFIX_DIR)
        target_link_options(dtm_to_ply_delaunay PRIVATE "-Wl,-rpath,${CONDA_PREFIX_DIR}/lib")
    endif()
    set_target_properties(dtm_to_ply_delaunay
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
else()
    message(WARNING
        "GDAL not properly configured. SKIPPING these targets: "
        "dem_mesh_generator, dtm_to_ply_delaunay. "
        "The GUI dashboard's DEM conversion / ROI extraction actions will "
        "report a missing executable until GDAL is installed and CMake is "
        "re-run. Install GDAL (conda-forge recommended on macOS: "
        "'conda install -c conda-forge gdal') -- see INSTALL.md, 'DEM / DTM tools'.")
    message(STATUS "  GDAL_FOUND: ${GDAL_FOUND}")
    message(STATUS "  GDAL_LIBRARIES: ${GDAL_LIBRARIES}")
    message(STATUS "  GDAL_INCLUDE_DIR: ${GDAL_INCLUDE_DIR}")
    if(GDAL_FOUND AND GDAL_LIBRARIES AND NOT GDAL_INCLUDE_DIR)
        message(STATUS "  -- the library was found and the HEADERS were not. "
                       "Building against the default include path would mix a "
                       "second GDAL's headers with this library; refusing.")
    endif()
endif()

# Conductivity-model anchor tests (Hayne 2017 vs Martinez & Siegler 2021).
# Header-only under test, so no link against `thermal` is required -- which
# also means this test cannot be silently satisfied by a stale library.
add_executable(test_conductivity_model test_conductivity_model.cpp)
set_target_properties(test_conductivity_model
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Primary-body finite-disk geometry gate (PTM-41 Lambert-sphere photometry,
# PTM-44 fractional visibility). Header-only under test -- same reasoning as
# test_conductivity_model above: no library to go stale underneath it.
add_executable(test_primary_disk test_primary_disk.cpp)
set_target_properties(test_primary_disk
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Pass-stride RRE extrapolation gate. Must reproduce the reference
# implementation the GO decision was measured against
# (eq_compare_2026-08-08/code/modeext.py::_rre) to machine precision, on
# rank-deficient inputs as well as well-conditioned ones. Header-only under
# test -- same reasoning as test_conductivity_model: no library can go stale
# underneath it.
add_executable(test_eq_extrapolation test_eq_extrapolation.cpp)
set_target_properties(test_eq_extrapolation
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Depth-grid / layer-property dump. Prints the arrays the solver ACTUALLY
# builds (thermal::Profile z/dz/kc/rho), so an offline flux audit does not have
# to reconstruct them -- a reconstruction wrong by a few percent is
# indistinguishable from an energy-conservation defect, which is exactly what
# blocked the 2026-08-08 offset investigation. No GPU, no mesh, no simulation:
# safe to run while a production job owns the GPUs.
add_executable(dump_thermal_grid dump_thermal_grid.cpp)
target_link_libraries(dump_thermal_grid PRIVATE thermal ${OpenMP_CXX_LIBRARIES})
set_target_properties(dump_thermal_grid
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Equilibration large-dt accuracy test
add_executable(test_equilibration_dt test_equilibration_dt.cpp)
target_link_libraries(test_equilibration_dt
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_equilibration_dt
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Scalar surface-Newton convergence gate (PTM-116). Links `thermal` so it
# measures the SHIPPING surfTemp against a float64 bisection root -- a stale
# libthermal would be caught by the emulation-fidelity self-check inside.
add_executable(test_surface_newton test_surface_newton.cpp)
target_link_libraries(test_surface_newton
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_surface_newton
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Shared spin-up seed gate. The 46.7 K four-way seed divergence was found by
# hand because NOTHING compared any two solver paths' initial conditions; this
# pins the properties every path's seed must have.
add_executable(test_eq_seed test_eq_seed.cpp)
target_link_libraries(test_eq_seed
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_eq_seed
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Lit-face seed anchor gate. The two limits of the lumped-substrate anchor are
# BOUNDARY CONDITIONS, not statistics, so they are unit tests here; the zero-
# swing byte-identity and every regime fallback are pinned alongside them.
add_executable(test_eq_anchor test_eq_anchor.cpp)
target_link_libraries(test_eq_anchor
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_eq_anchor
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Thermal-state checkpoint gate (BUGS.md PTM-76). Pins the 'PTMT' format, the
# bit-identical round trip, the single float32 rounding at the write, and the
# refusal on every malformed or mismatched input. No GPU required.
add_executable(test_state_io test_state_io.cpp)
target_link_libraries(test_state_io
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_state_io
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Equilibration spin-up estimator gate. Checks tau1 against the analytic
# quarter-wave slab, the ln(1/target) scaling of n_req, and that an
# unreachable target returns NO cycle count.
add_executable(test_eq_spinup test_eq_spinup.cpp)
target_link_libraries(test_eq_spinup
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_eq_spinup
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Retrospective equilibration certificate, run against a BANKED checkpoint.
# Links `thermal` so the depth grid and the property model are the ENGINE's,
# and `ptm_tools` only for yaml-cpp (--config). No mesh, no view factors, no
# SPICE, no GPU: safe to run while a production job owns the devices.
add_executable(ptm_eq_certify ptm_eq_certify.cpp)
target_link_libraries(ptm_eq_certify
    PRIVATE
    thermal
    ptm_tools
    ${OpenMP_CXX_LIBRARIES}
)
if(YAML_CPP_FOUND)
    target_compile_definitions(ptm_eq_certify PRIVATE HAVE_YAML_CPP)
endif()
set_target_properties(ptm_eq_certify
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Retrospective equilibration certificate gate. Built as REDUCTIONS -- a
# demonstrably settled column and a demonstrably unsettled one, both from
# closed form -- plus the arrest discrimination the quantity exists for: a
# frozen-but-unsettled column that every increment test calls converged must
# still FAIL here. Links `thermal` so it scores the shipping grid and property
# model; no GPU, no mesh, no simulation.
add_executable(test_eq_certify test_eq_certify.cpp)
target_link_libraries(test_eq_certify
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_eq_certify
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Chunked-run statistics merge tool. Combines per-chunk raw statistics
# (statistics/_raw/) from N chunks into a single set of merged stats
# (max/avg/min) plus the Schorghofer ice-stability depth recomputed
# from the merged ⟨E(z)⟩ accumulator.
add_executable(ptm_merge_chunks ptm_merge_chunks.cpp)
target_link_libraries(ptm_merge_chunks
    PRIVATE
    ice_table
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(ptm_merge_chunks
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add the crater mesh generator tool (no external dependencies)
add_executable(crater_mesh_generator crater_mesh_generator.cpp)
set_target_properties(crater_mesh_generator
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Set output directory for the other tools
set_target_properties(map_processor albedo_map_generator
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

cmake_minimum_required(VERSION 3.10)

# Collect tool sources
set(TOOL_SOURCES
    config_parser.cpp
    run_provenance.cpp
    solver_disclosure.cpp
)

# Create tools library
add_library(ptm_tools STATIC ${TOOL_SOURCES})

# run_provenance.cpp includes the generated ptm_build_info.h.
add_dependencies(ptm_tools ptm_build_info)
target_include_directories(ptm_tools PRIVATE ${CMAKE_BINARY_DIR}/include)

# Include directories
target_include_directories(ptm_tools
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link yaml-cpp if available
if(YAML_CPP_FOUND)
    # Add include directories
    if(YAML_CPP_INCLUDE_DIRS)
        target_include_directories(ptm_tools PUBLIC ${YAML_CPP_INCLUDE_DIRS})
        message(STATUS "ptm_tools: Added include dirs: ${YAML_CPP_INCLUDE_DIRS}")
    endif()
    
    # THE FIX: Find the actual library file, don't use just the name
    if(YAML_CPP_LIBRARIES STREQUAL "yaml-cpp")
        # pkg-config returned just the name, find the actual file
        find_library(YAML_CPP_LIBRARY_PATH
            NAMES yaml-cpp
            PATHS
                ${YAML_CPP_LIBRARY_DIRS}
                ${YAML_CPP_LIBDIR}
                /usr/local/lib
                /opt/homebrew/lib
                $ENV{CONDA_PREFIX}/lib
            NO_DEFAULT_PATH
        )
        
        if(YAML_CPP_LIBRARY_PATH)
            target_link_libraries(ptm_tools PUBLIC ${YAML_CPP_LIBRARY_PATH})
            message(STATUS "ptm_tools: Linked yaml-cpp (resolved): ${YAML_CPP_LIBRARY_PATH}")
        else()
            message(WARNING "ptm_tools: Could not find yaml-cpp library file")
        endif()
    elseif(TARGET yaml-cpp)
        # It's a CMake target
        target_link_libraries(ptm_tools PUBLIC yaml-cpp)
        message(STATUS "ptm_tools: Linked yaml-cpp (target)")
    elseif(YAML_CPP_LIBRARIES)
        # It's already a path
        target_link_libraries(ptm_tools PUBLIC ${YAML_CPP_LIBRARIES})
        message(STATUS "ptm_tools: Linked yaml-cpp (path): ${YAML_CPP_LIBRARIES}")
    else()
        message(WARNING "ptm_tools: yaml-cpp found but cannot determine how to link")
    endif()
else()
    message(STATUS "ptm_tools: Building without yaml-cpp")
endif()

# Suppress yaml-cpp infinity warnings on macOS
if(APPLE AND YAML_CPP_FOUND)
    target_compile_options(ptm_tools PRIVATE -Wno-nan-infinity-disabled)
endif()

# Set C++ standard
set_target_properties(ptm_tools PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(ptm_tools PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()

message(STATUS "Configured ptm_tools library")
# Depth-grid convergence gate. Refines m and requires the solution to converge
# with dz == the grid's own spacing, and to FAIL to converge with the removed
# 1 mm dz clamp re-created — the falsifier for that clamp. Links `thermal` so
# it exercises the shipping Profile constructor.
add_executable(test_grid_convergence test_grid_convergence.cpp)
target_link_libraries(test_grid_convergence
    PRIVATE
    thermal
    ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(test_grid_convergence
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
