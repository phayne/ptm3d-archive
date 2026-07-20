// ice_table.h - Ice stability depth solver (C++ implementation)
//
// Provides a high-throughput Newton solver for ice table depth using
// mean (Tavg) and maximum (Tmax) surface temperatures, mirroring the
// Python vectorized implementation in ptm_ice/ice_table_vectorized.py.
//
// Public API (namespace ice):
//   std::vector<double> computeIceDepths(const std::vector<double>& Tavg,
//                                        const std::vector<double>& Tmax,
//                                        double J0,
//                                        double delta = 1e-4,
//                                        double z0 = 0.0,
//                                        double l = 100e-6,
//                                        double zs = 0.1,
//                                        double Q = 0.02,
//                                        double k = 1e-3,
//                                        int    max_iter = 200,
//                                        double z_min = 0.0,
//                                        double z_max = 5.0,
//                                        std::vector<int>* iterations = nullptr,
//                                        bool   use_openmp = true);
//
// The solver enforces physical guard rails, sets NaN where Tavg > Tmax or
// no convergence within max_iter, and supports optional iteration counts.
//
// Notes:
//  - Temperatures are assumed in Kelvin.
//  - Returned depths are in meters.
//  - Input vectors must be equal length; no reallocation of inputs.
//  - Implementation minimizes temporaries; per-element Newton iteration.
//  - Optional OpenMP parallelism; guarded by compilation flags.
//
// Future integration steps (not implemented here):
//  - Call after thermal simulation to append ice depth to statistics.
//  - Stream processing for very large face counts (>1e7) if needed.

#ifndef ICE_TABLE_H
#define ICE_TABLE_H

#include <vector>
#include <cstddef>

namespace ice {

// ---------------------------------------------------------------------------
// Schorghofer & Taylor (2007) depth-resolved ice-stability model
//
// The thermal solver fills a per-face × per-layer accumulator of E(T(z,t))
// during the integration window (E in kg/m^2/s, the Hertz–Knudsen
// sublimation rate). Post-solve, computeIceStabilityDepths() finds for each
// face the smallest depth z* such that
//
//     J(z) = μ ℓ ⟨E(z)⟩ / (2 z) ≤ J₀
//
// where ⟨E(z)⟩ is the time-averaged sublimation rate at depth z over the
// integration window. Returns one float per face: the stability depth in
// meters. Faces where ice is unstable to the bottom of the model domain
// receive z_max; faces with ⟨E⟩ already ≤ J₀ at z = 0 receive 0.
//
// The accumulator buffer is layer-major (sum_E[layer * n_faces + face])
// to match the GPU OpenCL kernel's natural coalesced-write layout.
// ---------------------------------------------------------------------------
struct IceStabilityAccumulator {
    // sum_E[layer * n_faces + face] — running sum of E(T_layer(t)) across
    // all integration timesteps. Allocated only when ice-stability is
    // enabled. float32 to match GPU buffer; ~1 GB at 4.8M faces × 50 layers.
    std::vector<float>  sum_E;
    int                 n_faces  = 0;
    int                 n_layers = 0;
    // n_samples is set by the host once the integration loop is done.
    // (Each substep adds one E sample per layer per face on the GPU path,
    // or one sample per output step on the CPU path — host knows which.)
    int                 n_samples = 0;
    // Native depth grid (m), copied from the thermal Profile after the solve.
    std::vector<double> z;

    void allocate(int faces, int layers) {
        n_faces  = faces;
        n_layers = layers;
        sum_E.assign(static_cast<std::size_t>(faces) * layers, 0.0f);
        z.assign(layers, 0.0);
        n_samples = 0;
    }
    bool ready() const {
        return !sum_E.empty() && n_samples > 0
            && static_cast<std::size_t>(n_faces) * n_layers == sum_E.size()
            && static_cast<int>(z.size()) == n_layers;
    }
};

// Compute per-face ice-stability depths from a finalized accumulator.
// Returns a vector of length acc.n_faces with depths in meters
// (NaN if inputs are invalid).
//   J0_kgm2s : sublimation flux threshold (default 3e-14 ≈ 1 mm/Ga)
//   l_m      : grain hop length / pore diameter (default 75 μm)
//   z_max_m  : cap stability depth at this value (default = max depth in acc.z)
std::vector<float>
computeIceStabilityDepths(const IceStabilityAccumulator& acc,
                          double J0_kgm2s = 3.0e-14,
                          double l_m      = 75.0e-6,
                          double z_max_m  = -1.0,
                          bool   use_openmp = true);


// Compute ice stability depths given mean and max temperatures.
// See header comment for parameter definitions.
std::vector<double> computeIceDepths(const std::vector<double>& Tavg,
                                     const std::vector<double>& Tmax,
                                     double J0,
                                     double delta = 1e-4,
                                     double z0 = 0.0,
                                     double l = 100e-6,
                                     double zs = 0.1,
                                     double Q = 0.02,
                                     double k = 1e-3,
                                     int    max_iter = 200,
                                     double z_min = 0.0,
                                     double z_max = 5.0,
                                     std::vector<int>* iterations = nullptr,
                                     bool   use_openmp = true);

// Pointer/stride variant for direct buffer operation (in-place output).
// Returns 0 on success, non-zero on error.
int computeIceDepthsRaw(const double* Tavg,
                        const double* Tmax,
                        std::size_t N,
                        double J0,
                        double* depths_out,
                        int* iterations_out = nullptr,
                        double delta = 1e-4,
                        double z0 = 0.0,
                        double l = 100e-6,
                        double zs = 0.1,
                        double Q = 0.02,
                        double k = 1e-3,
                        int    max_iter = 200,
                        double z_min = 0.0,
                        double z_max = 5.0,
                        bool   use_openmp = true);

} // namespace ice

#endif // ICE_TABLE_H
