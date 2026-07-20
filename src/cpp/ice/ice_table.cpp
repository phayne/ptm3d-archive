// ice_table.cpp - Implementation of ice stability depth solver

#include "ice_table.h"
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ice {

namespace {
    // Physical constants (matching Python implementation)
    constexpr double MU  = 2.99e-26;      // kg (water molecule mass)
    constexpr double KB  = 1.38e-23;      // J/K Boltzmann
    constexpr double TT  = 273.16;        // K triple-point temp
    constexpr double PT  = 611.657;       // Pa triple-point pressure
    constexpr double L   = 2.834e6;       // J/kg enthalpy of sublimation
    constexpr double A   = 6.21e23;       // Pre-factor ( (2*pi*kb*mu)^-0.5 )

    inline double water_psat(double T) {
        // pv = PT * exp(-L*MU/KB * (1/T - 1/TT))
        return PT * std::exp(-L * MU / KB * (1.0 / T - 1.0 / TT));
    }
}

int computeIceDepthsRaw(const double* Tavg,
                        const double* Tmax,
                        std::size_t N,
                        double J0,
                        double* depths_out,
                        int* iterations_out,
                        double delta,
                        double z0,
                        double l,
                        double zs,
                        double Q,
                        double k,
                        int    max_iter,
                        double z_min,
                        double z_max,
                        bool   use_openmp) {
    if(!Tavg || !Tmax || !depths_out) return 1;
    if(N == 0) return 0;

    const double nanv = std::numeric_limits<double>::quiet_NaN();

    // Parallel loop (OpenMP optional)
    #ifdef _OPENMP
    if(use_openmp) {
        #pragma omp parallel for schedule(static)
        for(std::size_t i = 0; i < N; ++i) {
            double Ta = Tavg[i];
            double Tx = Tmax[i];
            if(Ta > Tx) {
                depths_out[i] = nanv;
                if(iterations_out) iterations_out[i] = 0;
                continue;
            }
            double z = z0;
            double dT = Tx - Ta;
            int iter = 0;
            double dz = delta * 10.0;
            while(std::fabs(dz) > delta && iter < max_iter) {
                // Temperature at depth
                double T = Ta + dT * std::exp(-z / zs) + (Q / k) * z;
                // Guard against non-physical T
                if(!(T > 0.0) || std::isnan(T)) { z = nanv; break; }
                double pv = water_psat(T);
                double f  = 2.0 * z * J0 / (MU * l) - A * pv * std::pow(T, -0.5);
                double F4 = -dT / zs * std::exp(-z / zs) + Q / k; // dTdz
                double F3 = L * MU / KB * std::pow(T, -2.0) * pv * F4; // dpdz
                double fp = 2.0 * J0 / (MU * l)
                          - A * std::pow(T, -0.5) * F3
                          - 0.5 * pv * A * std::pow(T, -1.5) * F4;
                if(std::fabs(fp) < 1e-14) { z = nanv; break; }
                dz = -f / fp;
                z += dz;
                if(z < z_min) z = z_min;
                if(z > z_max) z = z_max;
                ++iter;
            }
            if(std::fabs(dz) > delta) {
                z = nanv; // did not converge
            }
            depths_out[i] = z;
            if(iterations_out) iterations_out[i] = iter;
        }
        return 0;
    }
    #endif

    // Serial fallback
    for(std::size_t i = 0; i < N; ++i) {
        double Ta = Tavg[i];
        double Tx = Tmax[i];
        if(Ta > Tx) {
            depths_out[i] = nanv;
            if(iterations_out) iterations_out[i] = 0;
            continue;
        }
        double z = z0;
        double dT = Tx - Ta;
        int iter = 0;
        double dz = delta * 10.0;
        while(std::fabs(dz) > delta && iter < max_iter) {
            double T = Ta + dT * std::exp(-z / zs) + (Q / k) * z;
            if(!(T > 0.0) || std::isnan(T)) { z = nanv; break; }
            double pv = water_psat(T);
            double f  = 2.0 * z * J0 / (MU * l) - A * pv * std::pow(T, -0.5);
            double F4 = -dT / zs * std::exp(-z / zs) + Q / k;
            double F3 = L * MU / KB * std::pow(T, -2.0) * pv * F4;
            double fp = 2.0 * J0 / (MU * l)
                      - A * std::pow(T, -0.5) * F3
                      - 0.5 * pv * A * std::pow(T, -1.5) * F4;
            if(std::fabs(fp) < 1e-14) { z = nanv; break; }
            dz = -f / fp;
            z += dz;
            if(z < z_min) z = z_min;
            if(z > z_max) z = z_max;
            ++iter;
        }
        if(std::fabs(dz) > delta) {
            z = nanv;
        }
        depths_out[i] = z;
        if(iterations_out) iterations_out[i] = iter;
    }
    return 0;
}

std::vector<double> computeIceDepths(const std::vector<double>& Tavg,
                                     const std::vector<double>& Tmax,
                                     double J0,
                                     double delta,
                                     double z0,
                                     double l,
                                     double zs,
                                     double Q,
                                     double k,
                                     int    max_iter,
                                     double z_min,
                                     double z_max,
                                     std::vector<int>* iterations,
                                     bool   use_openmp) {
    if(Tavg.size() != Tmax.size()) {
        return std::vector<double>();
    }
    std::size_t N = Tavg.size();
    std::vector<double> depths(N);
    int* it_ptr = nullptr;
    if(iterations) {
        iterations->assign(N, 0);
        it_ptr = iterations->data();
    }
    computeIceDepthsRaw(Tavg.data(), Tmax.data(), N, J0, depths.data(), it_ptr,
                        delta, z0, l, zs, Q, k, max_iter, z_min, z_max, use_openmp);
    return depths;
}

// ---------------------------------------------------------------------------
// Schorghofer & Taylor (2007) depth-resolved ice-stability solver
// ---------------------------------------------------------------------------
std::vector<float>
computeIceStabilityDepths(const IceStabilityAccumulator& acc,
                          double J0_kgm2s,
                          double l_m,
                          double z_max_m,
                          bool   use_openmp) {
    std::vector<float> out;
    if (!acc.ready()) return out;
    const int F = acc.n_faces;
    const int L = acc.n_layers;
    const double inv_N = 1.0 / static_cast<double>(acc.n_samples);
    const double prefactor = 0.5 * MU * l_m / J0_kgm2s;  // J(z) = μℓ⟨E⟩/(2z) ⇒ z* = (μℓ/2J₀)·⟨E⟩
    const double z_cap = (z_max_m > 0.0) ? z_max_m : acc.z.back();
    out.assign(static_cast<std::size_t>(F), 0.0f);

    auto solve_face = [&](int f) {
        // Build ⟨E(z_i)⟩ on the fly; layer-major access.
        // Walk from surface down: stability depth is the smallest z such
        // that z ≥ μℓ⟨E(z)⟩ / (2 J₀).  Since ⟨E⟩ falls off rapidly with
        // depth (T decreases) and z grows monotonically, the function
        //     g(z) = z − (μℓ/2J₀)·⟨E(z)⟩
        // is monotonically increasing. We bracket between successive
        // layers and refine with log-linear interpolation in ⟨E⟩.

        // Quick check: ⟨E⟩ at surface already small enough?
        double E0 = acc.sum_E[0 * F + f] * inv_N;
        if (!(E0 > 0.0)) { out[f] = 0.0f; return; }
        double rhs0 = prefactor * E0;          // required depth to balance flux at z=0+
        if (rhs0 <= acc.z[0]) {
            out[f] = static_cast<float>(acc.z[0]);
            return;
        }
        // Walk down layers
        for (int i = 1; i < L; ++i) {
            double Ei = acc.sum_E[static_cast<std::size_t>(i) * F + f] * inv_N;
            double rhs_i = (Ei > 0.0) ? prefactor * Ei : 0.0;
            // crossing condition: at i, z_i ≥ rhs_i (i.e. g(z_i) ≥ 0)
            if (acc.z[i] >= rhs_i) {
                // crossed between i-1 and i. Log-linear interpolate ⟨E⟩
                // (i.e. linear in log) and use bisection-style refinement.
                double zlo = acc.z[i-1], zhi = acc.z[i];
                double Elo = acc.sum_E[static_cast<std::size_t>(i-1) * F + f] * inv_N;
                double Ehi = Ei;
                // 8 bisection iterations is plenty given E falls off
                // exponentially with z.
                for (int k = 0; k < 8; ++k) {
                    double zmid = 0.5 * (zlo + zhi);
                    // log-linear interp of E
                    double t = (zmid - acc.z[i-1]) / std::max(1e-30, acc.z[i] - acc.z[i-1]);
                    double logE_lo = std::log(std::max(1e-300, Elo));
                    double logE_hi = std::log(std::max(1e-300, Ehi));
                    double logE = logE_lo + t * (logE_hi - logE_lo);
                    double Em = std::exp(logE);
                    double rhs_m = prefactor * Em;
                    if (zmid >= rhs_m) zhi = zmid;
                    else               zlo = zmid;
                }
                double zsol = 0.5 * (zlo + zhi);
                if (zsol < 0.0)        zsol = 0.0;
                if (zsol > z_cap)      zsol = z_cap;
                out[f] = static_cast<float>(zsol);
                return;
            }
        }
        // ⟨E⟩ never falls low enough → ice unstable to the bottom of the model
        out[f] = static_cast<float>(z_cap);
    };

    #ifdef _OPENMP
    if (use_openmp) {
        #pragma omp parallel for schedule(static)
        for (int f = 0; f < F; ++f) solve_face(f);
        return out;
    }
    #endif
    for (int f = 0; f < F; ++f) solve_face(f);
    return out;
}

} // namespace ice
