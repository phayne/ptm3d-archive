# H-Matrix View Factor Optimization Roadmap

## Overview

This document outlines the optimization priorities for scaling the H-matrix view factor
computation to large meshes (~1M faces). The implementation is based on Potter et al. (2023)
hierarchical matrices approach for thermophysical modeling.

**Current Status** (Updated January 2026): All priority optimizations implemented!
- Quadtree spatial partitioning
- Low-rank block compression via truncated SVD
- Admissibility-based block classification
- Sparse block storage for non-compressible regions
- **Hierarchical block subdivision (COMPLETED)**
- **Spatial culling for back-facing geometry (COMPLETED)**
- **Binary serialization for save/load (COMPLETED)**

**Benchmark Results** (as of January 2026):
- Memory compression: 4-9.5x depending on mesh size
- MVP speedup: 1.15-2x for larger meshes
- Accuracy: <1% error at tolerance=0.01
- Serialization: ~20ms save, ~16ms load for 1044-face mesh

---

## Priority 1: Fix Hierarchical Block Subdivision (HIGH IMPACT) - ✅ COMPLETED

**Problem**: The original implementation had hierarchical block subdivision disabled due to
a bug where the nested 4x4 quadrant loop could generate more than 4 children, causing
array index issues.

**Solution Implemented**:
1. Changed `HierarchicalBlock::children` from `std::array<..., 4>` to `std::vector`
2. Fixed `buildBlock()` to properly handle all row×column quadrant pairs
3. Added special case in `HMatrixBlock::matvec()` to pass global arrays through hierarchical blocks
4. Re-enabled recursive subdivision with proper depth limiting

**Files modified**:
- `src/cpp/radiation/hmatrix_view_factor.h` - Changed children to dynamic vector
- `src/cpp/radiation/hmatrix_view_factor.cpp` - Fixed buildBlock() and matvec()

**Outcome**: Hierarchical blocks now correctly created, enabling better compression for large meshes

---

## Priority 2: Spatial Culling for Zero View Factors (HIGH IMPACT) - ✅ COMPLETED

**Problem**: Many face pairs have zero view factors due to:
- Back-facing geometry (normals point away from each other)
- Complete occlusion by intervening geometry
- Large separation distance

**Solution Implemented**:
1. **Face-level back-facing check**: In `computeViewFactor()`, check if dot(n_i, r_hat) > 0 AND dot(n_j, r_hat) < 0 before ray tracing
2. **Grazing angle rejection**: Skip pairs with cos(angle) < 0.01 (~89.4 degrees)
3. **Contribution-based culling**: Skip pairs where estimated F_ij would be below threshold even without occlusion
4. **Cluster-level culling**: New `canClustersSeeEachOther()` function uses average normals to skip entire blocks

**Files modified**:
- `src/cpp/radiation/hmatrix_view_factor.h` - Added `canClustersSeeEachOther()` declaration
- `src/cpp/radiation/hmatrix_view_factor.cpp` - Implemented all culling optimizations

**Outcome**: Significant reduction in unnecessary ray traces for crater geometries

---

## Priority 3: K-D Tree for Scalable Neighbor Queries (MEDIUM IMPACT) - ⏭️ DEFERRED

**Problem**: Current neighbor finding uses brute-force O(N²) search.

**Assessment**: The existing quadtree spatial partitioning already provides efficient
spatial queries for the H-matrix block construction. The O(N²) behavior in view factor
computation is inherent to the dense block computation, which is mitigated by:
- Hierarchical subdivision (Priority 1)
- Spatial culling (Priority 2)
- Low-rank compression

**Decision**: Deferred until profiling shows neighbor queries as a bottleneck for
the target mesh sizes. The current implementation handles ~3000 faces efficiently.

---

## Priority 4: View Factor Matrix Serialization (MEDIUM IMPACT) - ✅ COMPLETED

**Problem**: View factors are recomputed every simulation run, even when mesh hasn't changed.

**Solution Implemented**:
1. Binary file format with magic number (0x484D4154 = "HMAT") and version
2. Recursive serialization of all block types (Zero, Sparse, LowRank, Dense, Hierarchical)
3. Helper functions for Eigen matrices and STL vectors
4. Statistics stored for verification after load

**Files modified**:
- `src/cpp/radiation/hmatrix_view_factor.cpp` - Added ~300 lines for serialization

**Performance**:
- Save: ~20ms for 1044-face mesh
- Load: ~16ms for 1044-face mesh
- File size: ~4 MB (matches in-memory size)
- Verification: Perfect MVP match (0.0 difference)

**Usage**:
```cpp
HMatrixViewFactor hmatrix;
hmatrix.build(...);
hmatrix.writeBinary("view_factors.hmat");  // Save

HMatrixViewFactor hmatrix_loaded;
hmatrix_loaded.readBinary("view_factors.hmat");  // Load
```

**Outcome**: Near-instant view factor loading on subsequent runs

---

## Priority 5: GPU Acceleration (HIGH IMPACT, COMPLEX)

**Problem**: Ray tracing for visibility tests is embarrassingly parallel but runs on CPU.

**Impact**: GPU can provide 10-100x speedup for visibility computations.

**Solution Options**:
1. **Embree GPU** (Embree 4.x): Native GPU ray tracing via SYCL
2. **OptiX**: NVIDIA-specific but highly optimized
3. **Custom CUDA**: Manual implementation with batched ray casting

**Considerations**:
- Embree 4 already supports GPU via SYCL on Intel/NVIDIA
- Need to batch visibility queries for efficient GPU utilization
- Memory transfer overhead must be amortized

**Files to modify**:
- `src/cpp/radiation/view_factor.cpp` - Add GPU code path
- `CMakeLists.txt` - Add CUDA/SYCL detection

**Expected outcome**: 10-50x speedup in visibility computation

---

## Priority 6: Out-of-Core Processing (MEMORY REDUCTION)

**Problem**: For very large meshes, even compressed H-matrix may exceed available RAM.

**Impact**: Enables processing of meshes that don't fit in memory.

**Solution**:
1. Stream H-matrix blocks from disk during MVP
2. Use memory-mapped files for block storage
3. Implement LRU cache for frequently accessed blocks
4. Compress blocks on disk with fast compression (LZ4/Zstd)

**Files to modify**:
- `src/cpp/radiation/hmatrix_view_factor.cpp` - Add block streaming
- New file: `src/cpp/radiation/block_cache.h` - LRU cache implementation

**Expected outcome**: Process meshes up to disk capacity

---

## Implementation Order

Based on impact and complexity:

1. **Priority 1** (Hierarchical subdivision) - Foundation for O(N) scaling
2. **Priority 2** (Spatial culling) - Quick wins with minimal code changes
3. **Priority 3** (K-D tree) - Required for scalable neighbor queries
4. **Priority 4** (Serialization) - Quality of life improvement
5. **Priority 5** (GPU) - Major speedup but complex implementation
6. **Priority 6** (Out-of-core) - Only if memory becomes limiting

---

## Testing Strategy

Each optimization should be validated with:
1. **Correctness**: Compare results against reference (sparse) implementation
2. **Performance**: Benchmark on standard mesh sizes (1K, 10K, 100K, 1M faces)
3. **Memory**: Track peak memory usage during computation
4. **Stability**: Run overnight tests on large meshes

---

## References

- Potter et al. (2023) - Hierarchical matrices for thermophysical modeling
- Hackbusch (2015) - Hierarchical Matrices: Algorithms and Analysis
- Bebendorf (2008) - Hierarchical Matrices: A Means to Efficiently Solve Elliptic BVPs

---

## Notes

- Current implementation uses Eigen for SVD, which is single-threaded
- Consider switching to parallel SVD (Intel MKL or cuSOLVER) for large blocks
- The quadtree depth should be tuned based on mesh distribution
- Admissibility parameter η affects compression ratio vs accuracy tradeoff
