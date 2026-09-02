#!/bin/bash
# Clean rebuild script for PTM3D.
#
# WHAT THIS PRODUCES, AND WHY IT CHANGED (2026-09-02)
# ---------------------------------------------------
# Until today this script built a CPU-ONLY binary. It never passed
# -DPTM_ENABLE_GPU, so CMake's default (OFF, CMakeLists.txt:792 via
# CMakeLists.txt:787) applied -- and because step 2 does `rm -rf build/*`, it
# also DESTROYED the GPU-enabled cache of any existing build before configuring
# a CPU one in its place. A machine whose `build/` had been configured with
# -DPTM_ENABLE_GPU=ON came back CPU-only from a plain `./rebuild.sh`, and every
# config with `thermal.use_gpu: true` then refused to start. The crew's standing
# advice was "do not use rebuild.sh", which is not a fix.
#
# Now:
#   * GPU is ON by default. Turning it off takes an explicit --cpu-only.
#   * The existing build/CMakeCache.txt is READ BEFORE the directory is cleaned
#     and its settings are carried into the fresh configure, so a clean rebuild
#     reproduces the configuration that was there rather than replacing it.
#   * Every -D actually passed is echoed.
#   * The finished binary is INTERROGATED for the GPU core, and the script
#     FAILS if the GPU was asked for and is not there. That check is not
#     decoration: `src/cpp/gpu/CMakeLists.txt:49-58` substitutes a stub
#     INTERFACE library and prints a STATUS note when no backend is found, so
#     -DPTM_ENABLE_GPU=ON on a machine without OpenCL produces a CPU binary and
#     a zero exit status.
#
# It still differs from INSTALL.md section 3 in two ways, both stated at the
# end of the run:
#   * -DPTM_ARCH_NATIVE defaults ON, so the binary is tuned to this CPU
#     (-march=native) and is not portable. --portable turns that off.
#   * It builds the DEFAULT target set, not just ptm_thermal_sim.
#
# The documented equivalent of the default behavior here:
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#         -DPTM_ENABLE_GPU=ON -DGPU_BACKEND=OPENCL -DCSPICE_ROOT="$CSPICE_ROOT"
#   cmake --build build -j

set -e  # Exit on error

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
WANT_GPU=ON
WANT_NATIVE=""          # empty = inherit from the cache, else ON/OFF
FRESH=false
GPU_EXPLICIT=false
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: ./rebuild.sh [--cpu-only] [--gpu] [--portable] [--fresh] [--help]

  --cpu-only   build WITHOUT the GPU core (OpenCL). The default is GPU ON;
               this is the only way to get a CPU-only binary from this script,
               and it is stated in the summary at the end.
  --gpu        build WITH the GPU core (the default; accepted so an invocation
               can say so out loud).
  --portable   -DPTM_ARCH_NATIVE=OFF: a portable x86-64 baseline binary
               instead of one tuned to this machine's CPU.
  --fresh      ignore any existing build/CMakeCache.txt instead of carrying
               its settings forward.
  --dry-run    decide the configuration, print the exact cmake command it
               would run, and stop. Deletes nothing and builds nothing.
  --help       this text.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu-only|--no-gpu) WANT_GPU=OFF; GPU_EXPLICIT=true ;;
        --gpu)               WANT_GPU=ON;  GPU_EXPLICIT=true ;;
        --portable)          WANT_NATIVE=OFF ;;
        --native)            WANT_NATIVE=ON ;;
        --fresh)             FRESH=true ;;
        --dry-run|-n)        DRY_RUN=true ;;
        --help|-h)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

echo "=== PTM3D Clean Rebuild ==="
echo ""

# Check if we're in project root
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Must be run from project root directory"
    exit 1
fi

# CMake is not always on PATH on this machine (it lives in /usr/local/bin and
# some shells here do not have it). Resolve it once, loudly.
CMAKE_BIN="${CMAKE:-}"
if [ -z "$CMAKE_BIN" ]; then
    CMAKE_BIN="$(command -v cmake || true)"
fi
if [ -z "$CMAKE_BIN" ] && [ -x /usr/local/bin/cmake ]; then
    CMAKE_BIN=/usr/local/bin/cmake
fi
if [ -z "$CMAKE_BIN" ]; then
    echo "Error: cmake not found. Set \$CMAKE to its full path." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# 0. Inherit the existing configuration, BEFORE the directory is cleaned
# --------------------------------------------------------------------------
# The point of reading this first: step 3 deletes the cache, so anything not
# recovered here is silently replaced by a default. That is exactly how a
# GPU-enabled build became a CPU-only one.
cache_get() {   # cache_get <VAR> -> value on stdout, empty if absent
    [ -f build/CMakeCache.txt ] || return 0
    sed -n "s|^$1:[A-Z]*=\(.*\)$|\1|p" build/CMakeCache.txt | head -1
}

# Cache entries that are CARRIED FORWARD verbatim. Each one changes what the
# binary is, and none of them has a safe default:
#
#   BUILD_SHARED_LIBS   whether the project libraries are .dylib or .a, which
#                       decides whether the executable's md5 is a complete
#                       identity for the engine (it is not, when shared).
#   embree_DIR /        ⛔ CHANGES RESULTS. Embree's SIMD kernels make
#   EMBREE_ROOT         different floating-point association choices, so
#                       grazing-incidence occlusion verdicts flip between
#                       ISAs -- measured 2026-08-15 on an SSE2 -> multi-ISA
#                       swap of the identical Embree 4.3.3 source: a lunar
#                       polar DEM patch moved 75 of 260 output files. This
#                       machine's canonical build uses a hand-built
#                       ~/opt/embree-4.3.3-avx512 that a fresh configure does
#                       NOT find; it takes the Homebrew one instead, silently.
#   BUILD_PYTHON_BINDINGS   which targets exist afterwards.
CARRY_FORWARD="BUILD_SHARED_LIBS embree_DIR EMBREE_ROOT BUILD_PYTHON_BINDINGS"

echo "0. Reading the existing build configuration..."
INHERITED=""
CARRIED_ARGS=()
if [ "$FRESH" = true ]; then
    echo "   --fresh: any existing build/CMakeCache.txt will be ignored"
elif [ -f build/CMakeCache.txt ]; then
    C_GPU="$(cache_get PTM_ENABLE_GPU)"
    [ -z "$C_GPU" ] && C_GPU="$(cache_get ENABLE_GPU)"
    C_NATIVE="$(cache_get PTM_ARCH_NATIVE)"
    C_TYPE="$(cache_get CMAKE_BUILD_TYPE)"
    C_CSPICE="$(cache_get CSPICE_ROOT)"
    C_BACKEND="$(cache_get GPU_BACKEND)"
    echo "   found build/CMakeCache.txt:"
    echo "     PTM_ENABLE_GPU = ${C_GPU:-<unset>}"
    echo "     PTM_ARCH_NATIVE= ${C_NATIVE:-<unset>}"
    echo "     CMAKE_BUILD_TYPE=${C_TYPE:-<unset>}"
    echo "     CSPICE_ROOT    = ${C_CSPICE:-<unset>}"
    INHERITED="yes"
    # An explicit flag on the command line wins over the cache; otherwise the
    # cache wins over this script's own default, so a rebuild reproduces the
    # build that was there.
    if [ "$GPU_EXPLICIT" = false ] && [ -n "$C_GPU" ]; then
        WANT_GPU="$C_GPU"
    fi
    if [ -z "$WANT_NATIVE" ] && [ -n "$C_NATIVE" ]; then
        WANT_NATIVE="$C_NATIVE"
    fi
    [ -n "$C_TYPE" ] && BUILD_TYPE="$C_TYPE"
    [ -n "$C_CSPICE" ] && CSPICE_ROOT_ARG="$C_CSPICE"
    [ -n "$C_BACKEND" ] && GPU_BACKEND_ARG="$C_BACKEND"
    for key in $CARRY_FORWARD; do
        val="$(cache_get "$key")"
        if [ -n "$val" ]; then
            CARRIED_ARGS+=("-D$key=$val")
            echo "     $key = $val  [carried forward]"
        fi
    done
else
    echo "   no existing build/CMakeCache.txt; using this script's defaults"
fi
BUILD_TYPE="${BUILD_TYPE:-Release}"
WANT_NATIVE="${WANT_NATIVE:-ON}"
# OPENCL rather than AUTO: this is the backend on this platform, and naming it
# makes a machine with a half-installed HIP/CUDA fail here instead of picking
# a backend nobody asked for.
GPU_BACKEND_ARG="${GPU_BACKEND_ARG:-OPENCL}"
CSPICE_ROOT_ARG="${CSPICE_ROOT_ARG:-${CSPICE_ROOT:-${CSPICE_DIR:-}}}"

echo ""
echo "   This build will be:"
echo "     GPU core     : $WANT_GPU  (backend $GPU_BACKEND_ARG)$(
        [ "$GPU_EXPLICIT" = true ] && echo '  [from the command line]' ||
        { [ -n "$INHERITED" ] && echo '  [inherited from build/CMakeCache.txt]' ||
          echo '  [this script default]'; })"
echo "     CPU tuning   : PTM_ARCH_NATIVE=$WANT_NATIVE"
echo "     Build type   : $BUILD_TYPE"
if [ "$WANT_GPU" = "OFF" ]; then
    echo ""
    echo "   ⚠ CPU-ONLY BUILD. A config with thermal.use_gpu: true will refuse"
    echo "     to start with FATAL [gpu]. Re-run without --cpu-only for GPU."
fi

# --------------------------------------------------------------------------
# 1. GPU prerequisites -- fail here, not after 20 minutes of compiling
# --------------------------------------------------------------------------
if [ "$WANT_GPU" = "ON" ]; then
    echo ""
    echo "1. Checking GPU prerequisites (OpenCL)..."
    OPENCL_OK=false
    if [ -d /System/Library/Frameworks/OpenCL.framework ]; then
        OPENCL_OK=true
        echo "   ✓ OpenCL.framework (macOS)"
    else
        for h in /usr/include/CL/cl.h /usr/local/include/CL/cl.h \
                 "${CONDA_PREFIX:-/nonexistent}/include/CL/cl.h"; do
            if [ -f "$h" ]; then OPENCL_OK=true; echo "   ✓ $h"; break; fi
        done
    fi
    if [ "$OPENCL_OK" = false ]; then
        echo "   ✗ No OpenCL headers or framework found." >&2
        echo "" >&2
        echo "     A GPU build was requested and cannot be produced here." >&2
        echo "     CMake would NOT stop for this: src/cpp/gpu/CMakeLists.txt" >&2
        echo "     substitutes a stub library, prints a STATUS note and exits 0," >&2
        echo "     which is how a 'GPU build' becomes a CPU binary in silence." >&2
        echo "     Install an OpenCL SDK, or run ./rebuild.sh --cpu-only to say" >&2
        echo "     out loud that you want the CPU core." >&2
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# 2. yaml-cpp
# --------------------------------------------------------------------------
echo ""
echo "2. Detecting yaml-cpp installation..."
YAML_FOUND=false
YAML_INCLUDE=""
YAML_LIB=""

# Check Homebrew
if command -v brew &> /dev/null; then
    BREW_PREFIX=$(brew --prefix 2>/dev/null || echo "")
    if [ -n "$BREW_PREFIX" ]; then
        if [ -f "$BREW_PREFIX/include/yaml-cpp/yaml.h" ]; then
            YAML_INCLUDE="$BREW_PREFIX/include"
            # Find library
            if [ -f "$BREW_PREFIX/lib/libyaml-cpp.dylib" ]; then
                YAML_LIB="$BREW_PREFIX/lib/libyaml-cpp.dylib"
                YAML_FOUND=true
                echo "   ✓ Found yaml-cpp (Homebrew)"
            elif [ -f "$BREW_PREFIX/lib/libyaml-cpp.a" ]; then
                YAML_LIB="$BREW_PREFIX/lib/libyaml-cpp.a"
                YAML_FOUND=true
                echo "   ✓ Found yaml-cpp (Homebrew - static)"
            fi
        fi
    fi
fi

# Check conda if Homebrew didn't work
if [ "$YAML_FOUND" = false ] && [ -n "$CONDA_PREFIX" ]; then
    if [ -f "$CONDA_PREFIX/include/yaml-cpp/yaml.h" ]; then
        YAML_INCLUDE="$CONDA_PREFIX/include"
        if [ -f "$CONDA_PREFIX/lib/libyaml-cpp.dylib" ]; then
            YAML_LIB="$CONDA_PREFIX/lib/libyaml-cpp.dylib"
            YAML_FOUND=true
            echo "   ✓ Found yaml-cpp (Conda)"
        elif [ -f "$CONDA_PREFIX/lib/libyaml-cpp.a" ]; then
            YAML_LIB="$CONDA_PREFIX/lib/libyaml-cpp.a"
            YAML_FOUND=true
            echo "   ✓ Found yaml-cpp (Conda - static)"
        fi
    fi
fi

if [ "$YAML_FOUND" = false ]; then
    echo "   ✗ yaml-cpp not found!"
    echo ""
    echo "   To install:"
    echo "     Homebrew: brew install yaml-cpp"
    echo "     Conda:    conda install -c conda-forge yaml-cpp"
    echo ""
    read -p "   Continue without yaml-cpp? (config support will be disabled) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "     Include: $YAML_INCLUDE"
    echo "     Library: $YAML_LIB"
fi

# --------------------------------------------------------------------------
# 3. Clean
# --------------------------------------------------------------------------
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "--dry-run: stopping here. Nothing was deleted and nothing was built."
    echo "The configure this would run:"
    DRY_ARGS=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE" "-DPTM_ENABLE_GPU=$WANT_GPU")
    [ "$WANT_GPU" = "ON" ] && DRY_ARGS+=("-DGPU_BACKEND=$GPU_BACKEND_ARG")
    DRY_ARGS+=("-DPTM_ARCH_NATIVE=$WANT_NATIVE")
    [ -n "$CSPICE_ROOT_ARG" ] && DRY_ARGS+=("-DCSPICE_ROOT=$CSPICE_ROOT_ARG")
    [ "$YAML_FOUND" = true ] && DRY_ARGS+=("-DYAML_CPP_INCLUDE_DIR=$YAML_INCLUDE" \
                                          "-DYAML_CPP_LIBRARY=$YAML_LIB")
    [ ${#CARRIED_ARGS[@]} -gt 0 ] && DRY_ARGS+=("${CARRIED_ARGS[@]}")
    echo "  $CMAKE_BIN -S . -B build ${DRY_ARGS[*]}"
    exit 0
fi

echo ""
echo "3. Cleaning build directory..."
if [ -d "build" ]; then
    rm -rf build/*
    echo "   ✓ Build directory cleaned (its settings were read in step 0)"
else
    mkdir -p build
    echo "   ✓ Build directory created"
fi

# --------------------------------------------------------------------------
# 4. Configure
# --------------------------------------------------------------------------
echo ""
echo "4. Configuring CMake..."

CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE")
CMAKE_ARGS+=("-DPTM_ENABLE_GPU=$WANT_GPU")
if [ "$WANT_GPU" = "ON" ]; then
    CMAKE_ARGS+=("-DGPU_BACKEND=$GPU_BACKEND_ARG")
fi
CMAKE_ARGS+=("-DPTM_ARCH_NATIVE=$WANT_NATIVE")
if [ -n "$CSPICE_ROOT_ARG" ]; then
    CMAKE_ARGS+=("-DCSPICE_ROOT=$CSPICE_ROOT_ARG")
fi
if [ "$YAML_FOUND" = true ]; then
    CMAKE_ARGS+=("-DYAML_CPP_INCLUDE_DIR=$YAML_INCLUDE")
    CMAKE_ARGS+=("-DYAML_CPP_LIBRARY=$YAML_LIB")
fi
if [ ${#CARRIED_ARGS[@]} -gt 0 ]; then
    CMAKE_ARGS+=("${CARRIED_ARGS[@]}")
fi

echo "   Every option this configure passes:"
for a in "${CMAKE_ARGS[@]}"; do echo "     $a"; done
echo ""
echo "   Running: $CMAKE_BIN -S . -B build ${CMAKE_ARGS[*]}"
"$CMAKE_BIN" -S . -B build "${CMAKE_ARGS[@]}"
echo "   ✓ CMake configuration successful"

# --------------------------------------------------------------------------
# 5. Build
# --------------------------------------------------------------------------
echo ""
echo "5. Building..."
NCORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
echo "   Using $NCORES cores"

set +e
"$CMAKE_BIN" --build build -j"$NCORES" 2>&1 | tee build/build.log
BUILD_RC=${PIPESTATUS[0]}
set -e
if [ "$BUILD_RC" -ne 0 ]; then
    echo ""
    echo "   ✗ Build failed. Check build/build.log for details"
    exit 1
fi
echo ""
echo "   ✓ Build successful"

# --------------------------------------------------------------------------
# 6. Interrogate what was actually produced
# --------------------------------------------------------------------------
echo ""
echo "6. Checking the binary..."
EXE=build/bin/ptm_thermal_sim
if [ ! -f "$EXE" ]; then
    echo "   ✗ Executable not found at $EXE" >&2
    exit 1
fi
echo "   ✓ Executable found: $EXE"

# THE GPU GATE. Two independent probes, because each can be wrong on its own:
# the linkage says what was compiled in, and the engine's own answer says what
# it believes it can do. A build with the GPU core but no hardware prints "No
# OpenCL GPU devices detected", which is NOT a build failure and must not be
# treated as one -- the compile-time signature is the "Built without OpenCL
# support" line (ptm_thermal_sim.cpp:13347).
GPU_LINKED=unknown
if command -v otool &> /dev/null; then
    if otool -L "$EXE" | grep -qi 'OpenCL'; then GPU_LINKED=yes; else GPU_LINKED=no; fi
elif command -v ldd &> /dev/null; then
    if ldd "$EXE" | grep -qi 'OpenCL'; then GPU_LINKED=yes; else GPU_LINKED=no; fi
fi
GPU_REPORT="$("$EXE" --list-gpu-devices 2>&1 || true)"
if echo "$GPU_REPORT" | grep -q 'Built without OpenCL support'; then
    GPU_COMPILED=no
else
    GPU_COMPILED=yes
fi
echo "   OpenCL linked into the executable : $GPU_LINKED"
echo "   engine reports the GPU core       : $GPU_COMPILED"
echo "$GPU_REPORT" | grep -E 'GPU [0-9]+:|OpenCL GPU device|without OpenCL' || true

if [ "$WANT_GPU" = "ON" ] && [ "$GPU_COMPILED" != "yes" ]; then
    echo "" >&2
    echo "   ✗ A GPU BUILD WAS REQUESTED AND THIS BINARY HAS NO GPU CORE." >&2
    echo "     The configure step accepted -DPTM_ENABLE_GPU=ON and the build" >&2
    echo "     succeeded, so nothing before this point failed. Look for" >&2
    echo "     'No GPU backend found' in build/build.log." >&2
    echo "     Not shipping this as a GPU build." >&2
    exit 1
fi
if [ "$WANT_GPU" = "ON" ] && [ "$GPU_LINKED" = "no" ]; then
    echo "" >&2
    echo "   ✗ The engine reports a GPU core but the executable does not link" >&2
    echo "     OpenCL. The two probes disagree; do not trust this binary." >&2
    exit 1
fi

# ⛔ EMBREE IS PART OF THE PHYSICS. Different SIMD kernels make different
# floating-point association choices, so occlusion verdicts at grazing
# incidence can flip between ISAs. Which one this binary got is stated, not
# assumed -- a rebuild that quietly swapped it is a results change.
#
# The CACHE value, not otool's: a shared build records `@rpath/libembree4.4
# .dylib`, which names no installation at all. What the build was pointed at
# is the answer.
echo "   Embree: $(cache_get embree_DIR)"
if command -v otool &> /dev/null; then
    echo "           linked as $(otool -L "$EXE" | grep -i embree \
                                 | awk '{print $1}' | head -1)"
fi

if command -v otool &> /dev/null; then
    if otool -L "$EXE" | grep -q yaml-cpp; then
        echo "   ✓ yaml-cpp is linked"
    else
        echo "   ℹ yaml-cpp not linked (config support disabled)"
    fi
else
    echo "   ℹ link check skipped (otool not available on this platform)"
fi

if "$EXE" --help &> /dev/null; then
    echo "   ✓ Help command works"
else
    echo "   ⚠ Help command failed"
fi

if [ "$YAML_FOUND" = true ]; then
    # A directory, not `mktemp -t X` + ".yaml": that spelling leaves the
    # mktemp'd file behind and only deletes the name with the suffix on it.
    TMPDIR_T="$(mktemp -d -t ptm_rebuild)"
    if "$EXE" --generate-template "$TMPDIR_T/template.yaml" &> /dev/null; then
        echo "   ✓ Template generation works"
    else
        echo "   ⚠ Template generation failed (yaml-cpp may not be properly linked)"
    fi
    rm -rf "$TMPDIR_T"
fi

# --------------------------------------------------------------------------
echo ""
echo "=== Build Complete ==="
echo ""
echo "Executable: $(pwd)/$EXE"
if command -v md5 &> /dev/null; then
    echo "md5:        $(md5 -q "$EXE")"
fi
echo "GPU acceleration: $WANT_GPU (backend $GPU_BACKEND_ARG, verified in step 6)"
if [ "$WANT_NATIVE" = "ON" ]; then
    echo "Portability: tuned to this CPU (-march=native). Use --portable for a"
    echo "             portable x86-64 baseline binary."
else
    echo "Portability: portable x86-64 baseline (-DPTM_ARCH_NATIVE=OFF)"
fi
if [ "$YAML_FOUND" = true ]; then
    echo "YAML config support: ENABLED"
    echo ""
    echo "Try:"
    echo "  ./$EXE --generate-template"
    echo "  ./$EXE --config myconfig.yaml"
else
    echo "YAML config support: DISABLED"
    echo ""
    echo "All command-line arguments still work normally."
fi
echo ""
