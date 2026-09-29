/*
 * Diurnal-shape temperature response to an albedo change (--albedo-response diurnal).
 *
 * WHY. The exosphere does not re-solve the thermal model when the surface albedo
 * changes. The legacy law (albedo_refresh.h) multiplies the whole diurnal curve
 * by ((1 - A_new) / (1 - A_ref))^(1/4), which is exact only in radiative
 * equilibrium (zero thermal inertia). With thermal inertia the response is
 * concentrated at the daytime peak and damped at night. Measured against true
 * thermal re-solves (Europa albedo-map BASE phase 0, 50 kyr to 1 Myr), the uniform
 * rescale leaves darkened faces ~0.6 K too cold at peak and ~0.8 K too warm at
 * night (brightened faces the mirror image) and undercounts the Hertz-Knudsen
 * sublimation by -14.1% time-averaged.
 *
 * WHAT. An albedo change multiplies the ABSORBED SOLAR flux of the thermal run by
 *     r = (1 - A_new) / (1 - A_ref),   Q(t) = Q_ir(t) + r * Fs(t),
 * where Fs(t) = (1 - A_thermal) * (F_direct + F_reflected) is what the thermal run
 * absorbed and Q_ir (Jupiter's infrared, absorbed with the emissivity) does not
 * depend on albedo. The periodic temperature is expanded in eps = r - 1:
 *     T(t) = T0(t) + eps S1(t) + eps^2/2 S2(t) + O(eps^3),
 * where S1, S2 are the first- and second-order periodic responses of the same
 * one-dimensional conduction column the thermal engine integrates (massless
 * surface node with emissivity eps_ir, Hayne 2017 k(T) = kc(z)(1 + R350 T^3),
 * cp(T) polynomial, bottom flux Qb). S1 solves the tangent-linear problem forced
 * by Fs; S2 solves the same linear problem forced by the second derivatives of the
 * radiation, conduction and enthalpy terms evaluated on S1. Both are the EXACT
 * periodic solutions (monodromy of the linear stepper, one dense solve), so no
 * spin-up is involved.
 *
 * The result is applied in the log variable xi = ln r:
 *     ln T = ln T0 + xi g1 + xi^2/2 g2,   g1 = S1/T0,  g2 = (S1 + S2)/T0 - g1^2,
 * which is the same second-order expansion (eps = e^xi - 1) re-expressed so that
 * the radiative-equilibrium limit (g1 = 1/4, g2 = 0) is reproduced EXACTLY for any
 * r. The legacy rescale is therefore the special case g1 = 1/4, g2 = 0.
 *
 * The base state T0(z, t) the linearization needs is reconstructed per face by
 * driving the same column with the thermal run's own surface temperature
 * (Dirichlet) for `base_cycles` periods; only the material properties are read
 * from it, so the slow deep mode need not be converged (a 1 K deep error moves
 * cp by ~1%).
 *
 * Header-only so the unit test exercises the exact code the engine runs.
 */

#ifndef PTM_EXOSPHERE_DIURNAL_ALBEDO_RESPONSE_H
#define PTM_EXOSPHERE_DIURNAL_ALBEDO_RESPONSE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ptm {
namespace exosphere {

/** The thermal engine's column for one body (all faces share it on Europa). */
struct ThermalColumn {
    std::vector<double> z;        // node depths [m], z[0] = 0
    std::vector<double> kc;       // lattice conductivity at the nodes [W/m/K]
    std::vector<double> rho;      // density at the nodes [kg/m^3]
    double emissivity = 0.9;      // infrared emissivity (surface balance)
    double sigma = 5.670374419e-8;
    double Qb = 0.0;              // bottom heat flux [W/m^2]
    double R350 = 0.0;            // k = kc (1 + R350 T^3)
    double cp0 = 0.0;             // cp = cp0 + sum_j cpCoeff[j] T^(j+1)
    std::vector<double> cpCoeff;
    double period_s = 0.0;        // forcing period (the output window is one period)

    size_t layers() const { return z.size(); }

    double cp(double T) const {
        double r = 0.0, p = T;
        for (double c : cpCoeff) { r += c * p; p *= T; }
        return cp0 + r;
    }
    /** d^2 H / dT^2 = d cp / dT. */
    double dcp(double T) const {
        double r = 0.0, p = 1.0;
        for (size_t j = 0; j < cpCoeff.size(); ++j) { r += cpCoeff[j] * static_cast<double>(j + 1) * p; p *= T; }
        return r;
    }

    /** Empty string if usable, else why not. */
    std::string validate() const {
        const size_t n = z.size();
        if (n < 4) return "the column needs at least 4 nodes";
        if (kc.size() != n || rho.size() != n) return "kc/rho sizes differ from z";
        if (z[0] != 0.0) return "z[0] must be 0";
        for (size_t i = 0; i + 1 < n; ++i)
            if (!(z[i + 1] > z[i])) return "z must increase strictly";
        for (size_t i = 0; i < n; ++i)
            if (!(kc[i] > 0.0) || !(rho[i] > 0.0)) return "kc and rho must be positive";
        if (!(emissivity > 0.0) || !(period_s > 0.0)) return "emissivity and period must be positive";
        return "";
    }
};

struct DiurnalResponseOptions {
    int substeps = 4;       // conduction substeps per output interval
    int base_cycles = 3;    // Dirichlet base-state reconstruction periods
};

/** Per-face diagnostics. */
struct DiurnalResponseFaceInfo {
    bool ok = true;
    double max_abs_s1 = 0.0;     // max_t |S1| [K per unit eps]
    double min_g1 = 0.0, max_g1 = 0.0;
    // Surface energy balance of the thermal run's own T0 on this column:
    // res(t) = Fs + Qir + k dT/dz|0 - eps sigma T0^4 at the output epochs (W/m^2).
    // Small only if the column (grid, k, cp, emissivity) and the absorbed flux
    // match what the thermal run integrated.
    double balance_rms = 0.0;
    double balance_scale = 0.0;   // max_t max(Fs + Qir, eps sigma T0^4)
};

/** Reusable per-thread scratch. */
struct DiurnalResponseWorkspace {
    std::vector<double> Tb;                  // base T at every substep and node, (nsteps + 1) x N
    std::vector<double> A, B, H, C, dcp, Fuu, Fuv;   // per-substep coefficients
    std::vector<double> X, Xn;               // multi-column state, N x ncol
    std::vector<double> S1t;                 // S1 at every substep and node
    std::vector<double> fden, fcc, flo;      // tridiagonal factors per substep level
    std::vector<double> dz, wv, T, Told, a, b, c, d, cc, dd, Y, Z, IM, b1, b2;
    std::vector<int> piv;
};

namespace detail {

/** In-place dense LU with partial pivoting of the n x n row-major M. Returns false if singular. */
inline bool denseFactor(std::vector<double>& M, size_t n, std::vector<int>& piv) {
    piv.resize(n);
    for (size_t k = 0; k < n; ++k) {
        size_t p = k;
        double best = std::abs(M[k * n + k]);
        for (size_t i = k + 1; i < n; ++i) {
            const double v = std::abs(M[i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        if (!(best > 1e-300)) return false;
        piv[k] = static_cast<int>(p);
        if (p != k) {
            for (size_t j = 0; j < n; ++j) std::swap(M[k * n + j], M[p * n + j]);
        }
        for (size_t i = k + 1; i < n; ++i) {
            const double f = M[i * n + k] / M[k * n + k];
            M[i * n + k] = f;
            for (size_t j = k + 1; j < n; ++j) M[i * n + j] -= f * M[k * n + j];
        }
    }
    return true;
}

inline void denseLuSolve(const std::vector<double>& LU, const std::vector<int>& piv,
                         std::vector<double>& b, size_t n) {
    for (size_t k = 0; k < n; ++k) {
        if (static_cast<size_t>(piv[k]) != k) std::swap(b[k], b[static_cast<size_t>(piv[k])]);
        for (size_t i = k + 1; i < n; ++i) b[i] -= LU[i * n + k] * b[k];
    }
    for (size_t k = n; k-- > 0;) {
        for (size_t j = k + 1; j < n; ++j) b[k] -= LU[k * n + j] * b[j];
        b[k] /= LU[k * n + k];
    }
}

}  // namespace detail

/**
 * First- and second-order periodic responses of one face.
 *
 * T0[nt], Fs[nt]: the thermal run's surface temperature and absorbed solar flux
 * at nt output epochs uniformly spaced over ONE period (sample t at t P / nt).
 * Qir[nt] (may be null): the albedo-independent absorbed flux (Jupiter's
 * infrared); it enters only the energy-balance diagnostic, not S1 or S2.
 * On return S1[nt], S2[nt] hold the surface responses at the same epochs.
 * Returns false (and info->ok = false) only if the periodic system is singular
 * or the base state is not finite; the caller must then say so.
 */
inline bool computeDiurnalAlbedoResponse(const ThermalColumn& col, const double* T0, const double* Fs,
                                         const double* Qir, int nt, const DiurnalResponseOptions& opt,
                                         DiurnalResponseWorkspace& w, double* S1, double* S2,
                                         DiurnalResponseFaceInfo* info = nullptr) {
    const size_t N = col.layers();
    const size_t Ni = N - 2;                         // interior (state) nodes
    const size_t E = N - 1;                          // interfaces
    const size_t ns = static_cast<size_t>(std::max(1, opt.substeps));
    const size_t ntu = static_cast<size_t>(nt);
    const size_t nsteps = ntu * ns;
    const size_t L = nsteps + 1;                     // substep levels of one period
    const double dt = col.period_s / static_cast<double>(nsteps);
    const double es = col.emissivity * col.sigma;
    DiurnalResponseFaceInfo local;
    DiurnalResponseFaceInfo& inf = info ? *info : local;
    inf = DiurnalResponseFaceInfo();

    w.dz.resize(E); w.wv.assign(N, 0.0);
    for (size_t i = 0; i < E; ++i) w.dz[i] = col.z[i + 1] - col.z[i];
    for (size_t i = 1; i + 1 < N; ++i) w.wv[i] = 0.5 * (w.dz[i - 1] + w.dz[i]);
    const double* dz = w.dz.data();
    const double* wv = w.wv.data();

    // Periodic linear interpolation of a per-epoch series at substep level s.
    auto interp = [&](const double* v, size_t s) -> double {
        const size_t sm = s % nsteps, q = sm / ns, r = sm % ns;
        const double f = static_cast<double>(r) / static_cast<double>(ns);
        const double a = v[q], b = v[(q + 1) % ntu];
        return a + f * (b - a);
    };
    auto kOf = [&](size_t i, double T) { return col.kc[i] * (1.0 + col.R350 * T * T * T); };

    // Thomas solve of (a, b, c; d) over N rows into x (scratch cc, dd).
    auto thomas = [&](const double* a, const double* b, const double* c, const double* d, double* x) {
        double* cc = w.cc.data();
        double* dd = w.dd.data();
        cc[0] = c[0] / b[0]; dd[0] = d[0] / b[0];
        for (size_t i = 1; i < N; ++i) {
            const double den = b[i] - a[i] * cc[i - 1];
            cc[i] = c[i] / den;
            dd[i] = (d[i] - a[i] * dd[i - 1]) / den;
        }
        x[N - 1] = dd[N - 1];
        for (size_t i = N - 1; i-- > 0;) x[i] = dd[i] - cc[i] * x[i + 1];
    };
    w.a.resize(N); w.b.resize(N); w.c.resize(N); w.d.resize(N); w.cc.resize(N); w.dd.resize(N);
    w.T.resize(N); w.Told.resize(N);

    // ---------------------------------------------------------------------------
    // 1. Base state: Dirichlet surface = T0(t), CN interior with lagged properties
    //    (the engine's scheme), bottom flux Qb. Keep the LAST period.
    // ---------------------------------------------------------------------------
    w.Tb.assign(L * N, 0.0);
    {
        double mean = 0.0;
        for (size_t t = 0; t < ntu; ++t) mean += T0[t];
        mean /= static_cast<double>(ntu);
        for (size_t i = 0; i < N; ++i) w.T[i] = mean + col.Qb * col.z[i] / col.kc[i];
        w.T[0] = interp(T0, 0);
    }
    const int cycles = std::max(1, opt.base_cycles);
    for (int cyc = 0; cyc < cycles; ++cyc) {
        const bool keep = (cyc == cycles - 1);
        if (keep) std::copy(w.T.begin(), w.T.end(), w.Tb.begin());
        for (size_t s = 0; s < nsteps; ++s) {
            w.Told = w.T;
            const double* To = w.Told.data();
            w.a[0] = 0.0; w.b[0] = 1.0; w.c[0] = 0.0; w.d[0] = interp(T0, s + 1);
            for (size_t i = 1; i + 1 < N; ++i) {
                const double klo = kOf(i - 1, To[i - 1]) / dz[i - 1];
                const double khi = kOf(i, To[i]) / dz[i];
                const double m = wv[i] * col.rho[i] * col.cp(To[i]) / dt;
                w.a[i] = -0.5 * klo;
                w.b[i] = m + 0.5 * (klo + khi);
                w.c[i] = -0.5 * khi;
                w.d[i] = m * To[i] + 0.5 * (klo * (To[i - 1] - To[i]) + khi * (To[i + 1] - To[i]));
            }
            const double kb = kOf(N - 2, To[N - 2]) / dz[N - 2];
            w.a[N - 1] = -kb; w.b[N - 1] = kb; w.c[N - 1] = 0.0; w.d[N - 1] = col.Qb;
            thomas(w.a.data(), w.b.data(), w.c.data(), w.d.data(), w.T.data());
            if (keep) std::copy(w.T.begin(), w.T.end(), w.Tb.begin() + (s + 1) * N);
        }
    }
    for (double v : w.Tb) {
        if (!std::isfinite(v) || !(v > 0.0)) { inf.ok = false; return false; }
    }

    // Energy-balance diagnostic at the output epochs.
    {
        double ss = 0.0, scale = 0.0;
        for (size_t t = 0; t < ntu; ++t) {
            const double* Tz = &w.Tb[t * ns * N];
            const double q = Fs[t] + (Qir ? Qir[t] : 0.0);
            const double res = q + kOf(0, Tz[0]) * (Tz[1] - Tz[0]) / dz[0] - es * Tz[0] * Tz[0] * Tz[0] * Tz[0];
            ss += res * res;
            scale = std::max(scale, std::max(q, es * Tz[0] * Tz[0] * Tz[0] * Tz[0]));
        }
        inf.balance_rms = std::sqrt(ss / static_cast<double>(ntu));
        inf.balance_scale = scale;
    }

    // ---------------------------------------------------------------------------
    // 2. Per-level linearization coefficients on the base state.
    //    F_i = kc_i (1 + R u^3)(v - u)/dz_i,  u = T_i, v = T_{i+1}  (i = 0..N-2)
    //    dF_i = A_i th_i + B_i th_{i+1};  F''_i[s,s] = Fuu_i s_i^2 + 2 Fuv_i s_i s_{i+1}
    //    and the tridiagonal factors of the (level-s+1) step matrix, shared by
    //    every column and every march.
    // ---------------------------------------------------------------------------
    w.A.resize(L * E); w.B.resize(L * E); w.Fuu.resize(L * E); w.Fuv.resize(L * E);
    w.C.resize(L * N); w.dcp.resize(L * N); w.H.resize(L);
    w.fden.resize(L * N); w.fcc.resize(L * N); w.flo.resize(L * N);
    for (size_t s = 0; s < L; ++s) {
        const double* Tz = &w.Tb[s * N];
        for (size_t i = 0; i < E; ++i) {
            const double u = Tz[i], v = Tz[i + 1], R = col.R350, kc = col.kc[i] / dz[i];
            const double g = 1.0 + R * u * u * u;
            w.A[s * E + i] = kc * (3.0 * R * u * u * (v - u) - g);
            w.B[s * E + i] = kc * g;
            w.Fuu[s * E + i] = kc * (6.0 * R * u * (v - u) - 6.0 * R * u * u);
            w.Fuv[s * E + i] = kc * 3.0 * R * u * u;
        }
        for (size_t i = 0; i < N; ++i) {
            w.C[s * N + i] = col.cp(Tz[i]);
            w.dcp[s * N + i] = col.dcp(Tz[i]);
        }
        w.H[s] = 4.0 * es * Tz[0] * Tz[0] * Tz[0];
        // Rows at level s: 0 surface (H - A0) th0 - B0 th1; interior CN; N-1 bottom.
        const double* A = &w.A[s * E];
        const double* B = &w.B[s * E];
        double* lo = &w.flo[s * N];
        double* den = &w.fden[s * N];
        double* cc = &w.fcc[s * N];
        double up;
        lo[0] = 0.0; den[0] = w.H[s] - A[0]; up = -B[0]; cc[0] = up / den[0];
        for (size_t i = 1; i < N; ++i) {
            double di;
            if (i + 1 < N) {
                const double m = wv[i] * col.rho[i] / dt;
                lo[i] = 0.5 * A[i - 1];
                di = m * w.C[s * N + i] - 0.5 * (A[i] - B[i - 1]);
                up = -0.5 * B[i];
            } else {
                lo[i] = A[N - 2]; di = B[N - 2]; up = 0.0;
            }
            den[i] = di - lo[i] * cc[i - 1];
            cc[i] = up / den[i];
        }
    }

    // ---------------------------------------------------------------------------
    // 3. The linear stepper for ncol columns stored node-major, X[i * ncol + k].
    //    Column-specific forcing (level s, column k):
    //      surf(s, k)    surface-row source      bot(s, k)     bottom-row source
    //      flux(s, i, k) extra interface flux    ent(s, i, k)  extra enthalpy
    // ---------------------------------------------------------------------------
    auto completeBoundary = [&](double* X, size_t ncol, size_t s, auto&& surf, auto&& bot) {
        const double* A = &w.A[s * E];
        const double* B = &w.B[s * E];
        for (size_t k = 0; k < ncol; ++k) {
            X[k] = (surf(s, k) + B[0] * X[ncol + k]) / (w.H[s] - A[0]);
            X[(N - 1) * ncol + k] = (bot(s, k) - A[N - 2] * X[(N - 2) * ncol + k]) / B[N - 2];
        }
    };
    auto step = [&](std::vector<double>& Xv, size_t ncol, size_t s, auto&& surf, auto&& bot,
                    auto&& flux, auto&& ent) {
        const double* X = Xv.data();
        const size_t s1 = s + 1;
        const double* A0 = &w.A[s * E];
        const double* B0 = &w.B[s * E];
        const double* C0 = &w.C[s * N];
        const double* lo = &w.flo[s1 * N];
        const double* den = &w.fden[s1 * N];
        const double* cc = &w.fcc[s1 * N];
        w.Xn.resize(N * ncol);
        double* Xn = w.Xn.data();
        for (size_t k = 0; k < ncol; ++k) Xn[k] = surf(s1, k) / den[0];
        for (size_t i = 1; i + 1 < N; ++i) {
            const double m = wv[i] * col.rho[i] / dt;
            for (size_t k = 0; k < ncol; ++k) {
                const double th_m = X[(i - 1) * ncol + k], th = X[i * ncol + k], th_p = X[(i + 1) * ncol + k];
                const double dFhi = A0[i] * th + B0[i] * th_p;
                const double dFlo = A0[i - 1] * th_m + B0[i - 1] * th;
                const double r = m * (C0[i] * th + ent(s, i, k) - ent(s1, i, k))
                                 + 0.5 * (dFhi - dFlo)
                                 + 0.5 * (flux(s, i, k) - flux(s, i - 1, k) + flux(s1, i, k) - flux(s1, i - 1, k));
                Xn[i * ncol + k] = (r - lo[i] * Xn[(i - 1) * ncol + k]) / den[i];
            }
        }
        for (size_t k = 0; k < ncol; ++k)
            Xn[(N - 1) * ncol + k] = (bot(s1, k) - lo[N - 1] * Xn[(N - 2) * ncol + k]) / den[N - 1];
        for (size_t i = N - 1; i-- > 0;)
            for (size_t k = 0; k < ncol; ++k) Xn[i * ncol + k] -= cc[i] * Xn[(i + 1) * ncol + k];
        Xv.swap(w.Xn);
    };
    auto zero2 = [](size_t, size_t) { return 0.0; };
    auto zero3 = [](size_t, size_t, size_t) { return 0.0; };

    // ---------------------------------------------------------------------------
    // 4. S1: monodromy (Ni homogeneous columns) + one particular column forced by Fs.
    // ---------------------------------------------------------------------------
    const size_t nc = Ni + 1;
    w.X.assign(N * nc, 0.0);
    for (size_t k = 0; k < Ni; ++k) w.X[(k + 1) * nc + k] = 1.0;
    auto surf1 = [&](size_t s, size_t k) { return (k == Ni) ? interp(Fs, s) : 0.0; };
    completeBoundary(w.X.data(), nc, 0, surf1, zero2);
    for (size_t s = 0; s < nsteps; ++s) step(w.X, nc, s, surf1, zero2, zero3, zero3);
    w.IM.resize(Ni * Ni); w.b1.resize(Ni); w.b2.resize(Ni);
    for (size_t i = 0; i < Ni; ++i) {
        for (size_t k = 0; k < Ni; ++k) w.IM[i * Ni + k] = ((i == k) ? 1.0 : 0.0) - w.X[(i + 1) * nc + k];
        w.b1[i] = w.X[(i + 1) * nc + Ni];
    }
    if (!detail::denseFactor(w.IM, Ni, w.piv)) { inf.ok = false; return false; }
    detail::denseLuSolve(w.IM, w.piv, w.b1, Ni);   // periodic S1 interior at t = 0

    // ---------------------------------------------------------------------------
    // 5. March S1 from its periodic state, storing the trajectory; then S2: the
    //    particular solution (zero state) forced by the second-order terms, its
    //    periodic state, and the output march.
    // ---------------------------------------------------------------------------
    w.S1t.resize(L * N);
    auto surfS = [&](size_t s, size_t) { return interp(Fs, s); };
    w.Y.assign(N, 0.0);
    for (size_t i = 0; i < Ni; ++i) w.Y[i + 1] = w.b1[i];
    completeBoundary(w.Y.data(), 1, 0, surfS, zero2);
    std::copy(w.Y.begin(), w.Y.end(), w.S1t.begin());
    for (size_t s = 0; s < nsteps; ++s) {
        step(w.Y, 1, s, surfS, zero2, zero3, zero3);
        std::copy(w.Y.begin(), w.Y.end(), w.S1t.begin() + (s + 1) * N);
    }
    const double* S1t = w.S1t.data();
    auto F2 = [&](size_t s, size_t i) {
        const double u = S1t[s * N + i], v = S1t[s * N + i + 1];
        return w.Fuu[s * E + i] * u * u + 2.0 * w.Fuv[s * E + i] * u * v;
    };
    auto surf2 = [&](size_t s, size_t) {
        const double u = S1t[s * N], Ts = w.Tb[s * N];
        return F2(s, 0) - 12.0 * es * Ts * Ts * u * u;
    };
    auto bot2 = [&](size_t s, size_t) { return -F2(s, N - 2); };
    auto flux2 = [&](size_t s, size_t i, size_t) { return F2(s, i); };
    auto ent2 = [&](size_t s, size_t i, size_t) { const double u = S1t[s * N + i]; return w.dcp[s * N + i] * u * u; };
    w.Z.assign(N, 0.0);
    completeBoundary(w.Z.data(), 1, 0, surf2, bot2);
    for (size_t s = 0; s < nsteps; ++s) step(w.Z, 1, s, surf2, bot2, flux2, ent2);
    for (size_t i = 0; i < Ni; ++i) w.b2[i] = w.Z[i + 1];
    detail::denseLuSolve(w.IM, w.piv, w.b2, Ni);   // periodic S2 interior at t = 0
    std::fill(w.Z.begin(), w.Z.end(), 0.0);
    for (size_t i = 0; i < Ni; ++i) w.Z[i + 1] = w.b2[i];
    completeBoundary(w.Z.data(), 1, 0, surf2, bot2);
    S2[0] = w.Z[0];
    for (size_t s = 0; s + 1 < nsteps; ++s) {
        step(w.Z, 1, s, surf2, bot2, flux2, ent2);
        if ((s + 1) % ns == 0) S2[(s + 1) / ns] = w.Z[0];
    }
    for (size_t t = 0; t < ntu; ++t) S1[t] = S1t[t * ns * N];

    inf.min_g1 = 1e300; inf.max_g1 = -1e300;
    for (size_t t = 0; t < ntu; ++t) {
        if (!std::isfinite(S1[t]) || !std::isfinite(S2[t])) { inf.ok = false; return false; }
        inf.max_abs_s1 = std::max(inf.max_abs_s1, std::abs(S1[t]));
        const double g = S1[t] / T0[t];
        inf.min_g1 = std::min(inf.min_g1, g);
        inf.max_g1 = std::max(inf.max_g1, g);
    }
    return true;
}

/** Log-variable coefficients from the responses (see the header comment). */
inline void diurnalLogCoefficients(const double* T0, const double* S1, const double* S2, int nt,
                                   float* g1, float* g2) {
    for (int t = 0; t < nt; ++t) {
        const double a = S1[t] / T0[t];
        g1[t] = static_cast<float>(a);
        g2[t] = static_cast<float>((S1[t] + S2[t]) / T0[t] - a * a);
    }
}

/** T(t) for the absorbed-solar ratio r = (1 - A_new)/(1 - A_ref). */
inline double diurnalResponseTemperature(double T0, float g1, float g2, double r) {
    const double xi = std::log(r);
    return T0 * std::exp(xi * (static_cast<double>(g1) + 0.5 * xi * static_cast<double>(g2)));
}

}  // namespace exosphere
}  // namespace ptm

#endif  // PTM_EXOSPHERE_DIURNAL_ALBEDO_RESPONSE_H
