# Embree ISA and illumination performance


A stock package-manager Embree is usually compiled SSE2-only, and on a modern x86 CPU that
costs measurable wall-clock time in PTM3D. Embree compiles a separate BVH traversal kernel per SIMD
instruction set, fixed at *its* build time by `EMBREE_MAX_ISA`, and dispatches at run time to the
widest one it was built with. A redistributable package has to run on every CPU it might land on,
so it is built for the lowest common denominator: Homebrew's `embree` 4.3.3 bottle is
`EMBREE_MAX_ISA=SSE2`.

Ray traversal is roughly 83% of PTM3D's illumination phase, so the choice matters.
Measured on the project's Mac Pro (Xeon W-3265M, Cascade Lake, AVX-512) on 2026-08-15, against
the identical Embree 4.3.3 source rebuilt multi-ISA:

| | SSE2-only (Homebrew) | multi-ISA (`AVX512`) |
|---|---|---|
| `rtcOccluded1`, traversal micro-benchmark | 1.00x | 1.22–1.45x |
| illumination phase, `calib120` run, end to end | 13.12 s | **12.20 s (−7.0%)** |

The end-to-end figure is 18 paired interleaved repeats across two sessions (18/18 favoring
multi-ISA; t = 6.0 and 13.8). It is quoted with its limits: the thermal phase, which casts no
rays and therefore cannot be affected, was measured alongside as a negative control and came out
null (t = −0.8 and −1.0), which is what makes the illumination difference believable. A longer
709-step run on the same machine could not resolve the difference at all (n = 6, sd 45 s);
the control was equally contaminated there, so that is machine contention, not evidence against
the effect. Expect a few percent off the illumination phase, not a large speedup; the
traversal micro-benchmark ratio is not an end-to-end ratio.

## How to tell which ISA you are running

Three independent checks, cheapest first:

1. At run time, every run prints one line to stderr before the first ray is cast:
   ```
   [Embree] 4.3.3  traversal kernels: AVX-512 (16-wide)  (native packets 4=yes 8=yes 16=yes)
   ```
   An SSE2-only library on an AVX host prints `SSE (4-wide)` and an explicit `[Embree] WARNING:`
   naming the cost. For Embree's own definitive target report, set `PTM_EMBREE_VERBOSE=1`.
   (This line reads Embree's native packet widths, which Embree gates on `EMBREE_RAY_PACKETS`
   as well as on the ISA; hence `-DEMBREE_RAY_PACKETS=ON` in the recipe below. A library built
   with packets off reports that it cannot tell, rather than guessing "SSE2".)
2. At configure time, CMake names the library it took and the kernels it carries:
   ```
   -- Embree: /Users/you/opt/embree-4.3.3-avx512/lib/libembree4.4.dylib
   -- Embree: version 4.3.3, max ISA AVX512, kernels: SSE2 SSE42 AVX AVX2 AVX512
   ```
   A `EMBREE_MAX_ISA=SSE2` library on an x86-64 host raises a CMake warning.
3. From the library itself: count the wide-register instructions; an SSE2 build has none:
   ```bash
   otool -tV /usr/local/Cellar/embree/4.3.3/lib/libembree4.4.dylib | grep -c ymm   # -> 0
   otool -tV "$HOME/opt/embree-4.3.3-avx512/lib/libembree4.4.dylib" | grep -c zmm  # -> nonzero
   # Linux: objdump -d <lib> | grep -c ymm
   ```

After the fact, every run's `provenance.txt` records which Embree produced it:
```
embree_library=/Users/you/opt/embree-4.3.3-avx512/lib/libembree4.4.dylib
embree_max_isa=AVX512
```
This is not redundant with `engine_binary_md5`. Embree is a dynamic library, so upgrading it (or
rebuilding the same prefix at a different `EMBREE_MAX_ISA`) changes results while leaving the
executable's checksum untouched.

## Building a multi-ISA Embree

This does not replace or modify the package-manager install; it sits alongside it in its own
prefix, and nothing uses it until you pass `-DEMBREE_ROOT`.

```bash
curl -sSL -o embree-4.3.3.tar.gz \
  https://github.com/embree/embree/archive/refs/tags/v4.3.3.tar.gz
tar xf embree-4.3.3.tar.gz

cmake -S embree-4.3.3 -B embree-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/embree-4.3.3-avx512" \
  -DEMBREE_MAX_ISA=AVX512 \
  -DEMBREE_RAY_PACKETS=ON \
  -DEMBREE_TASKING_SYSTEM=TBB \
  -DTBB_ROOT=/usr/local/opt/tbb \
  -DEMBREE_ISPC_SUPPORT=OFF \
  -DEMBREE_TUTORIALS=OFF
cmake --build embree-build -j
cmake --install embree-build
```

Notes: `EMBREE_MAX_ISA=AVX512` is a *maximum*, not a requirement: the build still emits SSE2
through AVX2 kernels and picks at run time, so the result stays correct on older CPUs.
`EMBREE_RAY_PACKETS=ON` is Embree's default and is kept explicit because the run-time ISA check
above reads packet widths. `EMBREE_ISPC_SUPPORT=OFF` and `EMBREE_TUTORIALS=OFF` only skip
components PTM3D does not use.
`TBB_ROOT` is macOS/Homebrew-specific; on Linux drop it if TBB is on the default search path.
Build time is ~25 min on 24 cores. Use a prefix that a scratch/worktree cleanup will not delete.

Then point PTM3D at it:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DEMBREE_ROOT="$HOME/opt/embree-4.3.3-avx512" \
  -DCSPICE_ROOT="$CSPICE_ROOT"
```

`EMBREE_ROOT` may also be an environment variable; `-Dembree_ROOT=` and `-DCMAKE_PREFIX_PATH=`
are equivalent CMake-native spellings. If `EMBREE_ROOT` is set but the search still resolves
elsewhere, the configure stops rather than silently building against the other library;
clear the CMake cache and check that `$EMBREE_ROOT/lib/cmake/embree-<version>/` exists.

> Treat an Embree swap as a results change, not just a speed change. Different ISA kernels
> make different floating-point association choices, so grazing-incidence occlusion verdicts can
> flip. Measured on the SSE2 to multi-ISA swap: Europa (Pwyll, ico6) and Bennu byte-identical, but
> a lunar polar DEM patch moved 75 of 260 output files with a worst face-frame |ΔT| of 0.0091 K.
> That is far inside any physical tolerance, but re-validate rather than assume.


