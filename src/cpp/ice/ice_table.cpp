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

} // namespace ice
