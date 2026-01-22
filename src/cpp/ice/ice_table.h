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
