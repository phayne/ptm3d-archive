#!/bin/bash
# Clean rebuild script for PTM3D.
#
# THIS SCRIPT OWNS A CANONICAL CONFIGURATION AND ALWAYS CONVERGES TO IT.
# ---------------------------------------------------------------------
# It does NOT take settings from build/CMakeCache.txt. That was tried on
# 2026-09-02 and it is the wrong design: the cache on this workstation had been
# POISONED by the older version of this very script -- GPU OFF, static
# libraries, and the SSE2 Homebrew Embree -- and a script that faithfully
# preserves the cache faithfully preserves the poison. Inheriting a
# configuration is only correct if something guarantees the configuration was
# right, and nothing does.
#
# So: the canonical block below is the answer, every time. The cache is still
# READ, but only to DIFF against canonical and to PRINT what is being
# corrected, because converging in silence is exactly as bad as preserving in
# silence -- you would have no way to tell that your build just changed.
#
# The only things that beat canonical are explicit flags: --cpu-only,
# --portable, --embree <dir>. Nothing else does.
#
# ===================== THE CANONICAL CONFIGURATION =====================
#   CMAKE_BUILD_TYPE   Release
#   PTM_ENABLE_GPU     ON        OpenCL. --cpu-only to turn it off.
#   GPU_BACKEND        OPENCL    named, not AUTO, so a half-installed
#                                HIP/CUDA fails here instead of being picked.
#   PTM_ARCH_NATIVE    ON        -march=native. --portable for a portable
#                                x86-64 baseline binary.
#   BUILD_SHARED_LIBS  ON        matches the production build. With static
#                                libraries the executable's md5 is a complete
#                                identity and with shared ones it is not, so
#                                this must not drift silently between builds.
#   CSPICE_ROOT        auto-detected: $CSPICE_ROOT, $CSPICE_DIR, then the same
#                                prefix list CMakeLists.txt:529 searches.
#   EMBREE_ROOT        auto-detected and RANKED BY MEASURED ISA -- see below.
#                                --embree <prefix> for an explicit one.
# =======================================================================
#
# ⛔ WHY EMBREE IS AUTO-DETECTED BY MEASUREMENT, NOT BY NAME
# Embree compiles one BVH traversal kernel per SIMD ISA and dispatches at run
# time to the widest it was BUILT with. Homebrew's bottle is built
# EMBREE_MAX_ISA=SSE2 so it runs SSE2 traversal on an AVX-512 Xeon; ray
# traversal is ~83 % of the illumination phase, and a multi-ISA rebuild of the
# identical 4.3.3 source measured -14.8 % on the calib120 illumination phase
# (2026-08-15). It also CHANGES RESULTS: the same swap moved 75 of 260 output
# files on a lunar polar DEM patch.
#
# Every candidate prefix is asked what it actually is -- `EMBREE_MAX_ISA` is
# recorded in each installation's own lib/cmake/embree-*/embree-config.cmake --
# and the widest wins. A directory named "…-avx512" that is really an SSE2
# build therefore loses, and a good multi-ISA build in an unnamed prefix wins;
# neither would be true of a name match.
#
# This script builds the DEFAULT target set, not just ptm_thermal_sim, and it
# DELETES the contents of ./build. The documented equivalent of the canonical
# configure is in INSTALL.md section 3.

set -e  # Exit on error

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
WANT_GPU=ON
WANT_NATIVE=ON
EMBREE_OVERRIDE=""
FRESH=false
DRY_RUN=false
GPU_SOURCE="the canonical configuration"
NATIVE_SOURCE="the canonical configuration"

usage() {
    cat <<'EOF'
Usage: ./rebuild.sh [--cpu-only] [--portable] [--embree DIR] [--fresh]
                    [--dry-run] [--help]

This script always converges to ONE canonical configuration (GPU ON, shared
libraries, -march=native, the widest-ISA Embree it can find). It never takes
settings from build/CMakeCache.txt; it only reports where the cache differed.

  --cpu-only     build WITHOUT the GPU core (OpenCL). The ONLY way to get a
                 CPU-only binary from this script.
  --portable     -DPTM_ARCH_NATIVE=OFF: a portable x86-64 baseline binary
                 instead of one tuned to this machine's CPU.
  --embree DIR   use this Embree install PREFIX (the directory holding
                 include/embree4 and lib/cmake/embree-<version>) instead of
                 the auto-detected one.
  --fresh        accepted and documented for compatibility; it is now very
                 nearly a no-op, because the cache is never a source of
                 settings. All it does is skip the "what your cache had"
                 comparison. The build directory is cleaned either way.
  --dry-run      resolve the configuration, print the exact cmake command,
                 and stop. Deletes nothing and builds nothing.
  --help         this text.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu-only|--no-gpu) WANT_GPU=OFF; GPU_SOURCE="--cpu-only on the command line" ;;
        --gpu)               WANT_GPU=ON;  GPU_SOURCE="--gpu on the command line" ;;
        --portable)          WANT_NATIVE=OFF; NATIVE_SOURCE="--portable on the command line" ;;
        --native)            WANT_NATIVE=ON;  NATIVE_SOURCE="--native on the command line" ;;
        --embree)
            shift
            [ $# -gt 0 ] || { echo "--embree needs a directory" >&2; exit 2; }
            EMBREE_OVERRIDE="$1"
            ;;
        --fresh)             FRESH=true ;;
        --dry-run|-n)        DRY_RUN=true ;;
        --help|-h)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

echo "=== PTM3D Clean Rebuild (canonical configuration) ==="
echo ""

if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Must be run from project root directory" >&2
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

BUILD_TYPE=Release
GPU_BACKEND_ARG=OPENCL

# --------------------------------------------------------------------------
# 1. Resolve the canonical configuration
# --------------------------------------------------------------------------
echo "1. Resolving the canonical configuration..."

# -- CSPICE ----------------------------------------------------------------
# The same order CMakeLists.txt:522-536 uses, deliberately: if the two ever
# disagreed, the script would print one prefix and the build would use
# another. When nothing is found here, NO -DCSPICE_ROOT is passed and CMake's
# own identical search runs, ending in its actionable FATAL_ERROR.
CSPICE_ROOT_ARG=""
CSPICE_SOURCE=""
if [ -n "${CSPICE_ROOT:-}" ]; then
    CSPICE_ROOT_ARG="$CSPICE_ROOT"; CSPICE_SOURCE="\$CSPICE_ROOT"
elif [ -n "${CSPICE_DIR:-}" ]; then
    CSPICE_ROOT_ARG="$CSPICE_DIR"; CSPICE_SOURCE="\$CSPICE_DIR"
else
    for cand in /usr/local/cspice /opt/cspice "$HOME/cspice" \
                "$HOME/software/cspice"; do
        if [ -d "$cand/lib" ]; then
            CSPICE_ROOT_ARG="$cand"; CSPICE_SOURCE="auto-detected"; break
        fi
    done
fi
if [ -n "$CSPICE_ROOT_ARG" ]; then
    echo "   CSPICE : $CSPICE_ROOT_ARG  [$CSPICE_SOURCE]"
else
    echo "   CSPICE : not found by this script; leaving it to CMake's own"
    echo "            search, which fails with an actionable message if absent"
fi

# -- Embree ----------------------------------------------------------------
# `EMBREE_MAX_ISA` out of each installation's own config package. Anything
# this cannot read is ranked "unknown" and loses to anything it can.
embree_isa_of() {          # embree_isa_of <prefix> -> ISA on stdout, "" if none
    local cfg
    cfg="$(ls "$1"/lib/cmake/embree-*/embree-config.cmake 2>/dev/null | head -1)"
    [ -n "$cfg" ] || return 0
    [ -d "$1/include/embree4" ] || return 0
    sed -n 's/^[[:space:]]*SET(EMBREE_MAX_ISA[[:space:]]\{1,\}\([A-Za-z0-9]*\).*/\1/p' \
        "$cfg" | head -1
}

isa_rank() {
    case "$1" in
        AVX512*) echo 5 ;;
        AVX2)    echo 4 ;;
        AVX)     echo 3 ;;
        SSE42)   echo 2 ;;
        SSE2)    echo 1 ;;
        *)       echo 0 ;;
    esac
}

EMBREE_ROOT_ARG=""
EMBREE_ISA=""
EMBREE_SOURCE=""
ENV_EMBREE_LOST=""

if [ -n "$EMBREE_OVERRIDE" ]; then
    EMBREE_ROOT_ARG="$EMBREE_OVERRIDE"
    EMBREE_ISA="$(embree_isa_of "$EMBREE_ROOT_ARG")"
    EMBREE_SOURCE="--embree on the command line"
    if [ -z "$EMBREE_ISA" ]; then
        echo "   ⚠ --embree $EMBREE_ROOT_ARG does not look like an Embree install"
        echo "     prefix (no include/embree4 + lib/cmake/embree-*). Passing it"
        echo "     anyway, because you asked for it by name; CMake will refuse if"
        echo "     it is wrong."
        EMBREE_ISA="unknown"
    fi
else
    # Where to LOOK. The ranking below decides which one wins, so this list
    # only has to be broad, not ordered by quality.
    EMBREE_CANDIDATES=()
    for g in "$HOME"/opt/embree* /opt/embree* "$HOME"/software/embree*; do
        [ -d "$g" ] && EMBREE_CANDIDATES+=("$g")
    done
    [ -n "${EMBREE_ROOT:-}" ] && EMBREE_CANDIDATES+=("$EMBREE_ROOT")
    if [ -n "${embree_DIR:-}" ]; then
        # embree_DIR points at .../lib/cmake/embree-<v>; the prefix is 3 up.
        EMBREE_CANDIDATES+=("$(cd "$embree_DIR/../../.." 2>/dev/null && pwd)")
    fi
    [ -n "${CONDA_PREFIX:-}" ] && EMBREE_CANDIDATES+=("$CONDA_PREFIX")
    EMBREE_CANDIDATES+=("/usr/local/opt/embree" "/usr/local" "/opt/homebrew")

    echo "   Embree candidates (ISA read from each install's own"
    echo "   lib/cmake/embree-*/embree-config.cmake):"
    best_rank=-1
    seen_cfg=""
    for cand in "${EMBREE_CANDIDATES[@]}"; do
        [ -n "$cand" ] && [ -d "$cand" ] || continue
        isa="$(embree_isa_of "$cand")"
        [ -n "$isa" ] || continue
        # De-duplicate: /usr/local and /usr/local/opt/embree are one Homebrew
        # install reached by two names.
        cfg_real="$(cd "$(dirname "$(ls "$cand"/lib/cmake/embree-*/embree-config.cmake | head -1)")" && pwd -P)"
        case " $seen_cfg " in *" $cfg_real "*) continue ;; esac
        seen_cfg="$seen_cfg $cfg_real"
        rank="$(isa_rank "$isa")"
        echo "     max ISA $(printf '%-7s' "$isa") $cand"
        if [ "$rank" -gt "$best_rank" ]; then
            best_rank="$rank"; EMBREE_ROOT_ARG="$cand"; EMBREE_ISA="$isa"
        fi
    done
    if [ -n "$EMBREE_ROOT_ARG" ]; then
        EMBREE_SOURCE="auto-detected (widest ISA)"
    fi
    # An env var that was overruled must not be overruled in silence.
    for envname in EMBREE_ROOT embree_DIR; do
        eval "envval=\${$envname:-}"
        if [ -n "$envval" ]; then
            case "$EMBREE_ROOT_ARG" in
                "$envval"|"$envval"/*) ;;
                *) ENV_EMBREE_LOST="$ENV_EMBREE_LOST $envname=$envval" ;;
            esac
        fi
    done
fi

if [ -n "$EMBREE_ROOT_ARG" ]; then
    echo "   Embree : $EMBREE_ROOT_ARG  (max ISA ${EMBREE_ISA:-unknown})  [$EMBREE_SOURCE]"
else
    echo "   Embree : none found; leaving it to CMake's own search"
fi
if [ -n "$ENV_EMBREE_LOST" ]; then
    echo ""
    echo "   ⚠ NOT using the Embree named in your environment:$ENV_EMBREE_LOST"
    echo "     A wider-ISA installation was found and preferred. Pass"
    echo "     --embree <prefix> if you specifically want the other one."
fi
# ⛔ The SSE2 warning, printed by THIS SCRIPT. CMake prints one too, but it
# scrolls past in the middle of a long configure; a choice that costs ~15 % of
# the illumination phase has to be visible where the choice was made.
if [ "$EMBREE_ISA" = "SSE2" ]; then
    echo ""
    if [ -n "$EMBREE_OVERRIDE" ]; then
        echo "   ⚠ THE EMBREE YOU NAMED IS AN SSE2-ONLY BUILD (Homebrew's default)."
    else
        echo "   ⚠ THE ONLY EMBREE FOUND IS AN SSE2-ONLY BUILD (Homebrew's default)."
    fi
    echo "     On this x86-64 host it will run SSE2 BVH4 traversal, which is"
    echo "     ~83 % of the illumination phase; a multi-ISA rebuild of the same"
    echo "     Embree version measured -14.8 % on the calib120 illumination"
    echo "     phase (2026-08-15). See INSTALL.md, \"Embree ISA (performance)\","
    echo "     for the rebuild recipe, then re-run this script -- it will find"
    echo "     the new build and prefer it automatically."
fi

echo ""
echo "   This build will be:"
echo "     GPU core        : $WANT_GPU  (backend $GPU_BACKEND_ARG)  [$GPU_SOURCE]"
echo "     CPU tuning      : PTM_ARCH_NATIVE=$WANT_NATIVE  [$NATIVE_SOURCE]"
echo "     Shared libraries: BUILD_SHARED_LIBS=ON  [the canonical configuration]"
echo "     Build type      : $BUILD_TYPE  [the canonical configuration]"
if [ "$WANT_GPU" = "OFF" ]; then
    echo ""
    echo "   ⚠ CPU-ONLY BUILD, because $GPU_SOURCE. A config with"
    echo "     thermal.use_gpu: true will refuse to start with FATAL [gpu]."
fi

# --------------------------------------------------------------------------
# 2. What the existing cache had -- REPORTED, never used
# --------------------------------------------------------------------------
cache_get() {   # cache_get <VAR> -> value on stdout, empty if absent
    [ -f build/CMakeCache.txt ] || return 0
    sed -n "s|^$1:[A-Z]*=\(.*\)$|\1|p" build/CMakeCache.txt | head -1
}

echo ""
echo "2. Comparing with the existing build/CMakeCache.txt..."
if [ "$FRESH" = true ]; then
    echo "   --fresh: not reading the old cache. (It would not have been used"
    echo "   for anything but this comparison in any case.)"
elif [ ! -f build/CMakeCache.txt ]; then
    echo "   no existing cache; nothing to compare"
else
    corrections=0
    report_diff() {   # report_diff <label> <cached> <canonical>
        local cached="$2"
        if [ -n "$cached" ] && [ "$cached" != "$3" ]; then
            echo "   CORRECTING  $1: your cache had '$cached', this build uses '$3'"
            corrections=$((corrections + 1))
        fi
    }
    cached_gpu="$(cache_get PTM_ENABLE_GPU)"
    [ -z "$cached_gpu" ] && cached_gpu="$(cache_get ENABLE_GPU)"
    report_diff "PTM_ENABLE_GPU  " "$cached_gpu" "$WANT_GPU"
    report_diff "PTM_ARCH_NATIVE " "$(cache_get PTM_ARCH_NATIVE)" "$WANT_NATIVE"
    report_diff "BUILD_SHARED_LIBS" "$(cache_get BUILD_SHARED_LIBS)" "ON"
    report_diff "CMAKE_BUILD_TYPE" "$(cache_get CMAKE_BUILD_TYPE)" "$BUILD_TYPE"
    cached_embree="$(cache_get embree_DIR)"
    if [ -n "$cached_embree" ] && [ -n "$EMBREE_ROOT_ARG" ]; then
        case "$cached_embree" in
            "$EMBREE_ROOT_ARG"/*) ;;
            *) echo "   CORRECTING  Embree          : your cache had"
               echo "               '$cached_embree',"
               echo "               this build uses '$EMBREE_ROOT_ARG' (ISA $EMBREE_ISA)"
               corrections=$((corrections + 1)) ;;
        esac
    fi
    if [ "$corrections" -eq 0 ]; then
        echo "   the existing cache already matched this configuration"
    else
        echo ""
        echo "   $corrections setting(s) corrected. These are NOT inherited: this"
        echo "   script owns the configuration and converges to it every time."
    fi
fi

# --------------------------------------------------------------------------
# 3. GPU prerequisites -- fail here, not after 20 minutes of compiling
# --------------------------------------------------------------------------
if [ "$WANT_GPU" = "ON" ]; then
    echo ""
    echo "3. Checking GPU prerequisites (OpenCL)..."
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
        echo "     A GPU build is canonical and cannot be produced here." >&2
        echo "     CMake would NOT stop for this: src/cpp/gpu/CMakeLists.txt" >&2
        echo "     substitutes a stub library, prints a STATUS note and exits 0," >&2
        echo "     which is how a 'GPU build' becomes a CPU binary in silence." >&2
        echo "     Install an OpenCL SDK, or run ./rebuild.sh --cpu-only to say" >&2
        echo "     out loud that you want the CPU core." >&2
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# 4. yaml-cpp
# --------------------------------------------------------------------------
echo ""
echo "4. Detecting yaml-cpp installation..."
YAML_FOUND=false
YAML_INCLUDE=""
YAML_LIB=""

if command -v brew &> /dev/null; then
    BREW_PREFIX=$(brew --prefix 2>/dev/null || echo "")
    if [ -n "$BREW_PREFIX" ]; then
        if [ -f "$BREW_PREFIX/include/yaml-cpp/yaml.h" ]; then
            YAML_INCLUDE="$BREW_PREFIX/include"
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

if [ "$YAML_FOUND" = false ] && [ -n "${CONDA_PREFIX:-}" ]; then
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
# Assemble the configure line
# --------------------------------------------------------------------------
CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE")
CMAKE_ARGS+=("-DPTM_ENABLE_GPU=$WANT_GPU")
if [ "$WANT_GPU" = "ON" ]; then
    CMAKE_ARGS+=("-DGPU_BACKEND=$GPU_BACKEND_ARG")
fi
CMAKE_ARGS+=("-DPTM_ARCH_NATIVE=$WANT_NATIVE")
CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=ON")
if [ -n "$CSPICE_ROOT_ARG" ]; then
    CMAKE_ARGS+=("-DCSPICE_ROOT=$CSPICE_ROOT_ARG")
fi
if [ -n "$EMBREE_ROOT_ARG" ]; then
    # EMBREE_ROOT, not embree_DIR: CMakeLists.txt:367-384 treats it as the
    # documented knob, puts it ahead of CMAKE_PREFIX_PATH, and clears any
    # cached embree_DIR that would otherwise outrank it.
    CMAKE_ARGS+=("-DEMBREE_ROOT=$EMBREE_ROOT_ARG")
fi
if [ "$YAML_FOUND" = true ]; then
    CMAKE_ARGS+=("-DYAML_CPP_INCLUDE_DIR=$YAML_INCLUDE")
    CMAKE_ARGS+=("-DYAML_CPP_LIBRARY=$YAML_LIB")
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "--dry-run: stopping here. Nothing was deleted and nothing was built."
    echo "The configure this would run:"
    echo "  $CMAKE_BIN -S . -B build ${CMAKE_ARGS[*]}"
    exit 0
fi

# --------------------------------------------------------------------------
# 5. Clean
# --------------------------------------------------------------------------
echo ""
echo "5. Cleaning build directory..."
if [ -d "build" ]; then
    rm -rf build/*
    echo "   ✓ Build directory cleaned"
else
    mkdir -p build
    echo "   ✓ Build directory created"
fi

# --------------------------------------------------------------------------
# 6. Configure
# --------------------------------------------------------------------------
echo ""
echo "6. Configuring CMake..."
echo "   Every option this configure passes:"
for a in "${CMAKE_ARGS[@]}"; do echo "     $a"; done
echo ""
echo "   Running: $CMAKE_BIN -S . -B build ${CMAKE_ARGS[*]}"
"$CMAKE_BIN" -S . -B build "${CMAKE_ARGS[@]}"
echo "   ✓ CMake configuration successful"

# --------------------------------------------------------------------------
# 7. Build
# --------------------------------------------------------------------------
echo ""
echo "7. Building..."
NCORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
echo "   Using $NCORES cores"

set +e
"$CMAKE_BIN" --build build -j"$NCORES" 2>&1 | tee build/build.log
BUILD_RC=${PIPESTATUS[0]}
set -e
if [ "$BUILD_RC" -ne 0 ]; then
    echo ""
    echo "   ✗ Build failed. Check build/build.log for details" >&2
    exit 1
fi
echo ""
echo "   ✓ Build successful"

# --------------------------------------------------------------------------
# 8. Interrogate what was actually produced
# --------------------------------------------------------------------------
echo ""
echo "8. Checking the binary..."
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
if [ "$WANT_GPU" = "OFF" ] && [ "$GPU_COMPILED" = "yes" ]; then
    echo "" >&2
    echo "   ✗ --cpu-only was given and this binary HAS the GPU core." >&2
    exit 1
fi

# The Embree the build was pointed at, and how the executable reached it. A
# shared build records '@rpath/libembree4.4.dylib', which names no
# installation at all -- so the configured prefix is the answer.
echo "   Embree: ${EMBREE_ROOT_ARG:-<CMake default>} (max ISA ${EMBREE_ISA:-unknown})"
if command -v otool &> /dev/null; then
    echo "           linked as $(otool -L "$EXE" | grep -i embree \
                                 | awk '{print $1}' | head -1)"
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
    echo "            (BUILD_SHARED_LIBS=ON, so this md5 is NOT a complete"
    echo "             identity for the engine -- the project libraries are"
    echo "             separate .dylib files with their own hashes.)"
fi
if [ "$WANT_GPU" = "ON" ]; then
    echo "GPU acceleration: ON (backend $GPU_BACKEND_ARG, verified in step 8)"
else
    echo "GPU acceleration: OFF -- CPU-only, because $GPU_SOURCE (verified in step 8)"
fi
echo "Embree:           ${EMBREE_ROOT_ARG:-<CMake default>}, max ISA ${EMBREE_ISA:-unknown}"
if [ "$WANT_NATIVE" = "ON" ]; then
    echo "Portability: tuned to this CPU (-march=native). Use --portable for a"
    echo "             portable x86-64 baseline binary."
else
    echo "Portability: portable x86-64 baseline (-DPTM_ARCH_NATIVE=OFF)"
fi
if [ "$YAML_FOUND" = true ]; then
    echo "YAML config support: ENABLED"
else
    echo "YAML config support: DISABLED"
fi
echo ""
