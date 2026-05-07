# tinybvh (vendored)

This directory contains a vendored copy of [jbikker/tinybvh](https://github.com/jbikker/tinybvh),
a single-header MIT-licensed BVH library by Jacco Bikker. PTM3D uses it for
the CWBVH8 (Ylitie 2017) builder backing the GPU shadow-ray tracer.

## Provenance

- Upstream: <https://github.com/jbikker/tinybvh>
- Vendored commit: `4735685f64d8b2bb0ad2c6c4643f291bf58774ad`
  ("Added Bvh::BuildAABB function", main branch, 2026-03-31).
- File present:
  - `tiny_bvh.h` — single-header library (~8.9 K LOC).
  - `LICENSE` — original MIT license (Jacco Bikker, 2024).
- Build flag: tinybvh implementations are gated by
  `#define TINYBVH_IMPLEMENTATION`. We define this in exactly one TU
  (`src/cpp/gpu/cwbvh_builder.cpp`).

## Why tinybvh

We need a CWBVH8 builder matching the Ylitie et al. 2017 layout
(80 B/node, fp32 origin + fp8 quantized children). tinybvh's `BVH8_CWBVH`
is the only actively-maintained free implementation; alternatives
(Embree, OptiX) either are licensed for non-redistribution or are tied
to vendor SDKs not available on macOS+AMD.

We use **only the builder** (CPU-side, called once at scene init).
The OpenCL traversal kernel is custom — see `src/cpp/gpu/cwbvh_kernels.cl`.

## How to refresh

```sh
cd third_party/tinybvh
curl -L -o tiny_bvh.h https://raw.githubusercontent.com/jbikker/tinybvh/main/tiny_bvh.h
curl -L -o LICENSE    https://raw.githubusercontent.com/jbikker/tinybvh/main/LICENSE
# Update vendored commit hash above with:
curl -s https://api.github.com/repos/jbikker/tinybvh/commits/main | grep '"sha"' | head -1
# Then rebuild + run the CWBVH unit test (Phase B Gate B1) to confirm
# the upgrade didn't break the BVH8_CWBVH API or on-disk format.
```

## Notes

- AVX2 / FMA: tinybvh autodetects `__AVX2__` and falls back to scalar.
  Mac Pro Xeon W-3265M (Cascade Lake) supports AVX2; pass `-mavx2 -mfma`
  to clang for max throughput on the CPU build phase.
- Threading: tinybvh uses `std::thread` for parallel SAH split eval on
  meshes >50 K triangles. No OpenMP/TBB conflict with PTM3D's existing
  parallelism.
- Disk format: `BVH8_CWBVH::Save()/Load()` writes a versioned binary;
  PTM3D wraps these in `CWBVHBuilder::save/load`. Version mismatches
  invalidate the cache (the IlluminationModel rebuilds from scratch).

Do not edit `tiny_bvh.h` directly. Patches should be sent upstream and
re-vendored via the refresh procedure above.
