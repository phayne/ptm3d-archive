// Unit tests for --albedo-response diurnal (diurnal_albedo_response.h) and the
// AlbedoRefreshTracker::updateWith() hook the engine drives it through.
// Self-contained; exit code 0 = all checks pass.
//
//  (a) Against an INDEPENDENT reference: a fully nonlinear Crank-Nicolson column
//      (Newton on every step, enthalpy form, massless radiating surface) spun up
//      to a periodic state. The response is built from the reference's own
//      eps = 0 surface curve and compared with the reference re-solved at
//      absorbed-solar ratios r = 0.6 ... 1.5 (albedo changes of -0.25 ... +0.2 at
//      A = 0.5). The uniform 1/4-power rescale is scored on the same cases to show
//      each check discriminates.
//  (b) S1 against a centered finite difference of the reference.
//  (c) Radiative-equilibrium limit (vanishing conduction): g1 -> 1/4, g2 -> 0.
//  (d) Zero insolation: S1 = S2 = 0 exactly.
//  (e) The energy-balance diagnostic is small on the matching column and large
//      when the column's conductivity is doubled.
//  (f) updateWith(): same trigger, face selection and baseline bookkeeping as
//      update(); a face whose apply() refuses keeps its baseline and counts as
//      singular.

#include "diurnal_albedo_response.h"
#include "albedo_refresh.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace exo = ptm::exosphere;

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value = 0.0, double limit = 0.0) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-78s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

// Europa-like column: geometric grid from 3.76 mm to 0.53 m, the engine's Europa
// stratification law, cp = 7.49 T, Hayne 2017 chi = 2.7.
static exo::ThermalColumn europaLikeColumn(double kc_scale = 1.0) {
    exo::ThermalColumn c;
    double z = 0.0, dz = 0.00376;
    c.z.push_back(0.0);
    while (z < 0.53) { z += dz; c.z.push_back(z); dz *= 1.18; }
    for (double zi : c.z) {
        const double df = std::exp(-zi / 0.07);
        c.kc.push_back(kc_scale * (0.05 + (0.0096 - 0.05) * df));
        c.rho.push_back(700.0 + (500.0 - 700.0) * df);
    }
    c.emissivity = 0.9;
    c.Qb = 0.03;
    c.R350 = 2.7 / (350.0 * 350.0 * 350.0);
    c.cp0 = 0.0;
    c.cpCoeff = {7.49};
    c.period_s = 306720.0;
    return c;
}

// ---------------------------------------------------------------------------
// Independent reference: nonlinear CN in enthalpy form, surface balance at the
// new level, Newton per step (tridiagonal Jacobian), periodic by long spin-up.
// ---------------------------------------------------------------------------
static std::vector<double> referencePeriodic(const exo::ThermalColumn& c, const std::vector<double>& Fs,
                                             const std::vector<double>& Qir, double r, int ns, int cycles) {
    const size_t N = c.layers(), nt = Fs.size(), nsteps = nt * static_cast<size_t>(ns);
    const double dt = c.period_s / static_cast<double>(nsteps), es = c.emissivity * c.sigma;
    std::vector<double> dz(N - 1), wv(N, 0.0);
    for (size_t i = 0; i + 1 < N; ++i) dz[i] = c.z[i + 1] - c.z[i];
    for (size_t i = 1; i + 1 < N; ++i) wv[i] = 0.5 * (dz[i - 1] + dz[i]);
    auto H = [&](double T) { return c.cp0 * T + 7.49 * T * T / 2.0; };
    auto F = [&](size_t i, const std::vector<double>& T) {
        return c.kc[i] * (1.0 + c.R350 * T[i] * T[i] * T[i]) * (T[i + 1] - T[i]) / dz[i];
    };
    auto Q = [&](size_t s) {
        const size_t sm = s % nsteps, q = sm / static_cast<size_t>(ns);
        const double f = static_cast<double>(sm % static_cast<size_t>(ns)) / ns;
        const double a = r * Fs[q] + Qir[q], b = r * Fs[(q + 1) % nt] + Qir[(q + 1) % nt];
        return a + f * (b - a);
    };
    std::vector<double> T(N, 90.0), To(N), R(N), lo(N), di(N), up(N), cc(N), dd(N), dx(N), out(nt);
    for (size_t i = 0; i < N; ++i) T[i] = 90.0 + c.Qb * c.z[i] / c.kc[i];
    for (int cyc = 0; cyc < cycles; ++cyc) {
        for (size_t s = 0; s < nsteps; ++s) {
            if (cyc == cycles - 1 && s % static_cast<size_t>(ns) == 0) out[s / static_cast<size_t>(ns)] = T[0];
            To = T;
            std::vector<double> Fo(N - 1);
            for (size_t i = 0; i + 1 < N; ++i) Fo[i] = F(i, To);
            const double q1 = Q(s + 1);
            for (int it = 0; it < 30; ++it) {
                // residuals and Jacobian rows
                const double k0 = c.kc[0] * (1.0 + c.R350 * T[0] * T[0] * T[0]);
                R[0] = q1 + F(0, T) - es * T[0] * T[0] * T[0] * T[0];
                const double dFu0 = c.kc[0] * (3.0 * c.R350 * T[0] * T[0] * (T[1] - T[0]) / dz[0]) - k0 / dz[0];
                lo[0] = 0.0; di[0] = dFu0 - 4.0 * es * T[0] * T[0] * T[0]; up[0] = k0 / dz[0];
                for (size_t i = 1; i + 1 < N; ++i) {
                    const double m = wv[i] * c.rho[i] / dt;
                    R[i] = m * (H(T[i]) - H(To[i])) - 0.5 * ((F(i, T) - F(i - 1, T)) + (Fo[i] - Fo[i - 1]));
                    const double ki = c.kc[i] * (1.0 + c.R350 * T[i] * T[i] * T[i]);
                    const double km = c.kc[i - 1] * (1.0 + c.R350 * T[i - 1] * T[i - 1] * T[i - 1]);
                    const double dFi_du = c.kc[i] * 3.0 * c.R350 * T[i] * T[i] * (T[i + 1] - T[i]) / dz[i] - ki / dz[i];
                    const double dFm_du = c.kc[i - 1] * 3.0 * c.R350 * T[i - 1] * T[i - 1] * (T[i] - T[i - 1]) / dz[i - 1] - km / dz[i - 1];
                    lo[i] = -0.5 * (-dFm_du);
                    di[i] = m * (c.cp(T[i])) - 0.5 * (dFi_du - km / dz[i - 1]);
                    up[i] = -0.5 * (ki / dz[i]);
                }
                const size_t j = N - 2;
                const double kj = c.kc[j] * (1.0 + c.R350 * T[j] * T[j] * T[j]);
                R[N - 1] = F(j, T) - c.Qb;
                lo[N - 1] = c.kc[j] * 3.0 * c.R350 * T[j] * T[j] * (T[j + 1] - T[j]) / dz[j] - kj / dz[j];
                di[N - 1] = kj / dz[j]; up[N - 1] = 0.0;
                // solve J dx = -R
                cc[0] = up[0] / di[0]; dd[0] = -R[0] / di[0];
                for (size_t i = 1; i < N; ++i) {
                    const double den = di[i] - lo[i] * cc[i - 1];
                    cc[i] = up[i] / den; dd[i] = (-R[i] - lo[i] * dd[i - 1]) / den;
                }
                dx[N - 1] = dd[N - 1];
                for (size_t i = N - 1; i-- > 0;) dx[i] = dd[i] - cc[i] * dx[i + 1];
                double mx = 0.0;
                for (size_t i = 0; i < N; ++i) { T[i] += dx[i]; mx = std::max(mx, std::abs(dx[i])); }
                if (mx < 1e-12) break;
            }
        }
    }
    return out;
}

int main() {
    const int nt = 288, ns_ref = 4, cycles = 200;
    const exo::ThermalColumn col = europaLikeColumn();
    check(col.validate().empty(), "test column validates");

    // Europa-like forcing: sub-solar peak ~26 W/m^2 absorbed at A = 0.5, a 2.5 h
    // eclipse-like notch, and 0.13 W/m^2 of absorbed Jupiter infrared.
    std::vector<double> Fs(nt), Qir(nt, 0.13), Fz(nt, 0.0);
    for (int t = 0; t < nt; ++t) {
        const double ph = 2.0 * M_PI * t / nt;
        Fs[t] = 26.0 * std::max(0.0, std::cos(ph - 0.3));
        if (t >= 30 && t < 40) Fs[t] = 0.0;
    }
    const std::vector<double> T0 = referencePeriodic(col, Fs, Qir, 1.0, ns_ref, cycles);
    std::printf("  reference eps=0: T min %.3f max %.3f K\n", *std::min_element(T0.begin(), T0.end()),
                *std::max_element(T0.begin(), T0.end()));

    exo::DiurnalResponseOptions opt;
    exo::DiurnalResponseWorkspace ws;
    exo::DiurnalResponseFaceInfo inf;
    std::vector<double> S1(nt), S2(nt);
    const bool ok = exo::computeDiurnalAlbedoResponse(col, T0.data(), Fs.data(), Qir.data(), nt, opt, ws,
                                                      S1.data(), S2.data(), &inf);
    check(ok && inf.ok, "response computed");
    std::vector<float> g1(nt), g2(nt);
    exo::diurnalLogCoefficients(T0.data(), S1.data(), S2.data(), nt, g1.data(), g2.data());
    std::printf("  g1 range [%.4f, %.4f]; energy balance rms/scale %.3e\n", inf.min_g1, inf.max_g1,
                inf.balance_rms / inf.balance_scale);
    check(inf.max_g1 > 0.26 && inf.min_g1 < 0.2, "g1 departs from the uniform 1/4 (peak up, night down)",
          inf.max_g1 - inf.min_g1, 0.06);

    // (a) re-solves at several absorbed-solar ratios. The Europa albedo-map BASE run
    // spans dA = -0.233 ... +0.102 by 1 Myr, i.e. r ~ 0.80 ... 1.47 at A ~ 0.5: the
    // operating range, held to 0.05 K. r = 0.6 lies outside it; there the remainder
    // is third order in xi = ln r, so its error may grow no faster than |xi|^3
    // relative to r = 0.8 (the same side of r = 1).
    double err_08 = 0.0, err_06 = 0.0;
    for (double r : {0.6, 0.8, 1.2, 1.5}) {
        const std::vector<double> Tr = referencePeriodic(col, Fs, Qir, r, ns_ref, cycles);
        double e_diur = 0.0, e_resc = 0.0;
        const double tr_max = *std::max_element(Tr.begin(), Tr.end());
        double td_max = 0.0, tq_max = 0.0;
        for (int t = 0; t < nt; ++t) {
            const double Td = exo::diurnalResponseTemperature(T0[t], g1[t], g2[t], r);
            const double Tq = T0[t] * std::pow(r, 0.25);
            e_diur = std::max(e_diur, std::abs(Td - Tr[t]));
            e_resc = std::max(e_resc, std::abs(Tq - Tr[t]));
            td_max = std::max(td_max, Td); tq_max = std::max(tq_max, Tq);
        }
        const double pk_d = std::abs(td_max - tr_max), pk_r = std::abs(tq_max - tr_max);
        if (r == 0.8) err_08 = e_diur;
        if (r == 0.6) err_06 = e_diur;
        char nm[128];
        std::snprintf(nm, sizeof nm, "r=%.2f: 1/4-power rescale error is >= 10x larger (discriminates)", r);
        check(e_resc > 10.0 * e_diur, nm, e_resc, 10.0 * e_diur);
        if (r == 0.6) continue;   // outside the operating range: scored by the xi^3 check below
        std::snprintf(nm, sizeof nm, "r=%.2f: diurnal max|T - T_resolve| over the day", r);
        check(e_diur < 0.05, nm, e_diur, 0.05);
        std::snprintf(nm, sizeof nm, "r=%.2f: diurnal peak-T error (rescale %.3f K)", r, pk_r);
        check(pk_d < 0.05, nm, pk_d, 0.05);
    }
    {
        const double ratio = std::pow(std::log(0.6) / std::log(0.8), 3.0);
        check(err_06 < ratio * err_08, "r=0.60 (outside range): error grows no faster than |xi|^3 from r=0.80",
              err_06, ratio * err_08);
    }

    // (b) S1 vs centered finite difference of the reference
    {
        const double h = 0.01;
        const std::vector<double> Tp = referencePeriodic(col, Fs, Qir, 1.0 + h, ns_ref, cycles);
        const std::vector<double> Tm = referencePeriodic(col, Fs, Qir, 1.0 - h, ns_ref, cycles);
        double e = 0.0, smax = 0.0, e2 = 0.0, s2max = 0.0;
        for (int t = 0; t < nt; ++t) {
            e = std::max(e, std::abs((Tp[t] - Tm[t]) / (2 * h) - S1[t]));
            e2 = std::max(e2, std::abs((Tp[t] - 2 * T0[t] + Tm[t]) / (h * h) - S2[t]));
            smax = std::max(smax, std::abs(S1[t]));
            s2max = std::max(s2max, std::abs(S2[t]));
        }
        check(e < 0.01 * smax, "S1 matches the reference's centered difference (1% of max|S1|)", e, 0.01 * smax);
        check(e2 < 0.05 * s2max, "S2 matches the reference's second difference (5% of max|S2|)", e2, 0.05 * s2max);
    }

    // (c) radiative-equilibrium limit: conduction 1e-6 of Europa's
    {
        exo::ThermalColumn c0 = europaLikeColumn(1e-6);
        c0.Qb = 0.0;
        std::vector<double> Tq(nt), Qz(nt, 0.0), s1(nt), s2(nt);
        for (int t = 0; t < nt; ++t) Tq[t] = std::pow(std::max(Fs[t], 1e-3) / (c0.emissivity * c0.sigma), 0.25);
        std::vector<double> Fq(nt);
        for (int t = 0; t < nt; ++t) Fq[t] = std::max(Fs[t], 1e-3);
        exo::DiurnalResponseFaceInfo i0;
        exo::computeDiurnalAlbedoResponse(c0, Tq.data(), Fq.data(), Qz.data(), nt, opt, ws, s1.data(), s2.data(), &i0);
        std::vector<float> a(nt), b(nt);
        exo::diurnalLogCoefficients(Tq.data(), s1.data(), s2.data(), nt, a.data(), b.data());
        double e1 = 0.0, e2 = 0.0;
        for (int t = 0; t < nt; ++t) {
            if (Fs[t] < 1.0) continue;   // lit samples
            e1 = std::max(e1, std::abs(a[t] - 0.25));
            e2 = std::max(e2, std::abs(static_cast<double>(b[t])));
        }
        check(e1 < 2e-3, "radiative limit: g1 -> 1/4 on lit samples", e1, 2e-3);
        check(e2 < 2e-3, "radiative limit: g2 -> 0 on lit samples", e2, 2e-3);
    }

    // (d) zero insolation
    {
        std::vector<double> Tn = referencePeriodic(col, Fz, Qir, 1.0, ns_ref, 50), s1(nt), s2(nt);
        exo::computeDiurnalAlbedoResponse(col, Tn.data(), Fz.data(), Qir.data(), nt, opt, ws, s1.data(), s2.data());
        double m = 0.0;
        for (int t = 0; t < nt; ++t) m = std::max({m, std::abs(s1[t]), std::abs(s2[t])});
        check(m == 0.0, "zero insolation: S1 = S2 = 0 exactly", m, 0.0);
    }

    // (e) energy-balance diagnostic discriminates a wrong column
    {
        const double good = inf.balance_rms / inf.balance_scale;
        exo::DiurnalResponseFaceInfo ib;
        std::vector<double> s1(nt), s2(nt);
        exo::computeDiurnalAlbedoResponse(europaLikeColumn(2.0), T0.data(), Fs.data(), Qir.data(), nt, opt, ws,
                                          s1.data(), s2.data(), &ib);
        const double bad = ib.balance_rms / ib.balance_scale;
        check(good < 0.015, "energy balance on the matching column below the engine's 0.015 limit", good, 0.015);
        check(bad > 0.05, "energy balance with 2x conductivity well above it", bad, 0.05);
    }

    // (f) updateWith(): trigger and bookkeeping as update()
    {
        exo::AlbedoRefreshTracker ta, tb;
        const std::vector<double> seed = {0.5, 0.6, 0.4, 0.99999999999};
        ta.seed(seed); tb.seed(seed);
        std::vector<std::vector<double>> Ta(4, std::vector<double>(3, 100.0)), Tb = Ta;
        std::vector<double> cur = seed;
        size_t calls = 0;
        bool same = true;
        for (int pass = 0; pass < 12; ++pass) {
            cur[0] -= 0.004; cur[2] += 0.001; cur[3] -= 0.0;   // face 3 never moves
            const exo::AlbedoRefreshResult ra = ta.update(cur, Ta, 0.01);
            const exo::AlbedoRefreshResult rb = tb.updateWith(cur, 4, 0.01, 0.0, [&](size_t f, double a_new) {
                ++calls;
                const double s = std::pow((1.0 - a_new) / (1.0 - seed[f]), 0.25);
                for (double& T : Tb[f]) T = 100.0 * s;
                return true;
            });
            same = same && ra.fired == rb.fired && ra.faces_rescaled == rb.faces_rescaled &&
                   ta.baseline() == tb.baseline();
        }
        double dT = 0.0;
        for (size_t f = 0; f < 4; ++f)
            for (size_t t = 0; t < 3; ++t) dT = std::max(dT, std::abs(Ta[f][t] - Tb[f][t]));
        check(same && calls > 0, "updateWith fires, selects and re-bases exactly as update");
        check(dT < 1e-11, "non-cumulative 1/4 law through updateWith == cumulative update()", dT, 1e-11);
        exo::AlbedoRefreshTracker tc;
        tc.seed({0.5, 0.5});
        std::vector<double> c2 = {0.52, 0.53};
        const exo::AlbedoRefreshResult rc = tc.updateWith(c2, 2, 0.01, 0.0, [&](size_t f, double) { return f == 0; });
        check(rc.fired && rc.faces_rescaled == 1 && rc.faces_singular == 1 && tc.baseline()[1] == 0.5 &&
                  tc.baseline()[0] == 0.52,
              "a refused face keeps its baseline and counts as singular");
    }

    std::printf("\n%d/%d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
