#include "eq_certificate.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

namespace thermal {

bool eqCertificateEnabled() {
    // DEFAULT ON. This is report-only -- it changes no solver state and the
    // statistics files are byte-identical either way -- so the default is that
    // every run tells the user what its spin-up bought. PTM_EQ_CERTIFICATE=0
    // is the opt-out (for a byte-identical LOG, e.g. a log-diffing harness).
    static const bool on = [] {
        const char* e = std::getenv("PTM_EQ_CERTIFICATE");
        const bool off = (e && *e && std::strcmp(e, "0") == 0);
        if (off) {
            std::cout << "[Thermal] PTM_EQ_CERTIFICATE=0: equilibration spin-up "
                         "estimate suppressed." << std::endl;
        }
        return !off;
    }();
    return on;
}

namespace {

// Solve a tridiagonal system (Thomas). sub/diag/sup are n long; sub[0] and
// sup[n-1] are ignored. Returns false on a zero pivot.
bool thomas(const std::vector<double>& sub, std::vector<double> diag,
            const std::vector<double>& sup, std::vector<double> rhs,
            std::vector<double>& out) {
    const int n = static_cast<int>(diag.size());
    if (n < 2) return false;
    for (int i = 1; i < n; ++i) {
        if (std::abs(diag[i - 1]) < 1e-300) return false;
        const double m = sub[i] / diag[i - 1];
        diag[i] -= m * sup[i - 1];
        rhs[i]  -= m * rhs[i - 1];
    }
    out.assign(n, 0.0);
    if (std::abs(diag[n - 1]) < 1e-300) return false;
    out[n - 1] = rhs[n - 1] / diag[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        out[i] = (rhs[i] - sup[i] * out[i + 1]) / diag[i];
    }
    return true;
}

// Build the symmetric conductance matrix G and the lumped capacity C for a
// column with a Robin top and a Neumann bottom.
bool buildGC(const double* z, const double* T, const double* k,
             const double* rho, const double* cp, int n,
             double eps, double sigma,
             std::vector<double>& sub, std::vector<double>& diag,
             std::vector<double>& sup, std::vector<double>& C) {
    if (n < 3 || !z || !T || !k || !rho || !cp) return false;
    std::vector<double> g(n - 1, 0.0);   // interface conductances [W/m^2/K]
    for (int i = 0; i < n - 1; ++i) {
        const double dz = z[i + 1] - z[i];
        if (!(dz > 0.0)) return false;
        g[i] = k[i] / dz;                // k at the SHALLOWER node
    }
    sub.assign(n, 0.0); diag.assign(n, 0.0); sup.assign(n, 0.0); C.assign(n, 0.0);

    const double T0 = std::max(T[0], 1.0);
    const double h_rad = 4.0 * eps * sigma * T0 * T0 * T0;

    diag[0] = h_rad + g[0];
    sup[0]  = -g[0];
    C[0]    = rho[0] * cp[0] * 0.5 * (z[1] - z[0]);
    for (int i = 1; i < n - 1; ++i) {
        sub[i]  = -g[i - 1];
        diag[i] =  g[i - 1] + g[i];
        sup[i]  = -g[i];
        C[i]    = rho[i] * cp[i] * 0.5 * ((z[i] - z[i - 1]) + (z[i + 1] - z[i]));
    }
    sub[n - 1]  = -g[n - 2];
    diag[n - 1] =  g[n - 2];
    C[n - 1]    = rho[n - 1] * cp[n - 1] * 0.5 * (z[n - 1] - z[n - 2]);
    for (int i = 0; i < n; ++i) if (!(C[i] > 0.0)) return false;
    return true;
}

}  // namespace

double eqColumnTau1(const double* z, const double* T, const double* k,
                    const double* rho, const double* cp, int n,
                    double eps, double sigma) {
    std::vector<double> sub, diag, sup, C;
    if (!buildGC(z, T, k, rho, cp, n, eps, sigma, sub, diag, sup, C)) return -1.0;

    // Inverse iteration on G x = lambda C x: x <- G^{-1} (C x), normalized.
    // Three iterations (measured rel. err. 6.2e-6). The slowest mode is close
    // to uniform, so start there.
    std::vector<double> x(n, 1.0), rhs(n), y;
    for (int it = 0; it < 3; ++it) {
        for (int i = 0; i < n; ++i) rhs[i] = C[i] * x[i];
        if (!thomas(sub, diag, sup, rhs, y)) return -1.0;
        double nrm = 0.0;
        for (int i = 0; i < n; ++i) nrm = std::max(nrm, std::abs(y[i]));
        if (!(nrm > 0.0) || !std::isfinite(nrm)) return -1.0;
        for (int i = 0; i < n; ++i) x[i] = y[i] / nrm;
    }
    // Rayleigh quotient in the C inner product: lambda = xGx / xCx.
    double xGx = 0.0, xCx = 0.0;
    for (int i = 0; i < n; ++i) {
        double gx = diag[i] * x[i];
        if (i > 0)     gx += sub[i] * x[i - 1];
        if (i < n - 1) gx += sup[i] * x[i + 1];
        xGx += x[i] * gx;
        xCx += x[i] * C[i] * x[i];
    }
    if (!(xGx > 0.0) || !(xCx > 0.0)) return -1.0;
    const double lambda = xGx / xCx;           // [1/s]
    if (!(lambda > 0.0) || !std::isfinite(lambda)) return -1.0;
    return 1.0 / lambda;
}

EqSpinupResult eqEstimateSpinup(const EqSpinupInput& in) {
    EqSpinupResult r;
    if (in.n < 3 || !(in.P > 0.0) || !(in.target_K > 0.0)) return r;

    r.tau1 = eqColumnTau1(in.z, in.T, in.k, in.rho, in.cp, in.n, in.eps, in.sigma);
    if (!(r.tau1 > 0.0)) return r;
    r.tau1_cycles = r.tau1 / in.P;

    // E0 -- the seed's own departure from the conductive steady profile it must
    // develop, ANCHORED AT THE SEED'S OWN SURFACE. Anchoring instead at the
    // global (Qb/(eps*sigma))^0.25 would charge a lit column ~250 K for
    // insolation it legitimately carries, and hugely overestimate n_req. This
    // measures the thing spin-up actually has to do: grow the interior
    // gradient. Stated in the banner, because it IS an assumption.
    double Tref = in.T[0], e0 = 0.0;
    for (int i = 1; i < in.n; ++i) {
        const double dz = in.z[i] - in.z[i - 1];
        const double kk = (in.k[i - 1] > 0.0) ? in.k[i - 1] : 1e-6;
        Tref += in.Qb * dz / kk;
        e0 = std::max(e0, std::abs(in.T[i] - Tref));
    }
    r.E0 = e0;

    if (in.precision_floor_K > 0.0 && in.target_K < in.precision_floor_K) {
        r.unreachable = true;
        r.valid = true;
        return r;                       // deliberately NO cycle count
    }
    r.n_req = (r.E0 > in.target_K)
            ? r.tau1_cycles * std::log(r.E0 / in.target_K)
            : 0.0;
    r.valid = true;
    return r;
}

EqSpinupCensus::EqSpinupCensus(double P, double target_K, double precision_floor_K,
                               double seconds_per_cycle_estimate)
    : P_(P), target_K_(target_K), floor_K_(precision_floor_K),
      sec_per_cycle_(seconds_per_cycle_estimate) {}

void EqSpinupCensus::add(const EqSpinupResult& r, bool zero_forcing) {
    std::lock_guard<std::mutex> lk(m_);
    Bin& b = zero_forcing ? zero_ : forced_;
    ++b.n;
    // A column the estimator could not score must be COUNTED, not dropped. A
    // silently skipped face is how an estimate quietly becomes a sample of the
    // faces that happened to work.
    if (!r.valid) { ++b.invalid; return; }
    if (r.unreachable) { ++b.unreachable; return; }
    b.n_req.push_back(r.n_req);
    b.tau1_cycles.push_back(r.tau1_cycles);
    b.E0.push_back(r.E0);
}

namespace {
double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = static_cast<size_t>(q * (v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}
}  // namespace

void EqSpinupCensus::line(std::ostream& os, const char* label, const Bin& b,
                          double sec_per_cycle) {
    if (b.n == 0) {
        os << "    " << std::left << std::setw(16) << label
           << " (no faces)\n";
        return;
    }
    os << "    " << std::left << std::setw(16) << label << std::right
       << " n=" << std::setw(9) << b.n;
    if (b.invalid == b.n) {
        os << "   NOT SCORED (all faces; degenerate column)\n";
        return;
    }
    if (b.unreachable + b.invalid == b.n) {
        os << "   TARGET UNREACHABLE on this path (all scorable faces)\n";
        return;
    }
    const double med = pct(b.n_req, 0.50), p95 = pct(b.n_req, 0.95);
    os << "   tau1 = " << std::fixed << std::setprecision(2)
       << pct(b.tau1_cycles, 0.50) << " cycles"
       << "   E0 = " << std::setprecision(3) << pct(b.E0, 0.50) << " K"
       << "   cycles needed: median " << std::setprecision(1) << med
       << ", p95 " << p95;
    if (sec_per_cycle > 0.0) {
        os << "  (~" << std::setprecision(1)
           << (p95 * sec_per_cycle / 60.0) << " min at p95)";
    }
    if (b.unreachable > 0) os << "   [" << b.unreachable << " UNREACHABLE]";
    if (b.invalid > 0)     os << "   [" << b.invalid << " NOT SCORED]";
    os << "\n";
}

bool EqSpinupCensus::report(std::ostream& os, double n_configured) const {
    std::lock_guard<std::mutex> lk(m_);
    if (zero_.n == 0 && forced_.n == 0) return false;
    const long long unreachable = zero_.unreachable + forced_.unreachable;

    os << "\n=== Equilibration spin-up estimate (report-only; nothing is changed) ===\n"
       << "  Target accuracy      : " << target_K_ << " K"
       << "   (thermal.equilibration_target_K)\n"
       << "  Spin-up this run does: " << n_configured << " cycles"
          "   (analytic seed + thermal.equilibration_orbits)\n"
       << "  Model: n_req = (tau1/P) * ln(E0/target), with tau1 the slowest\n"
       << "  eigenvalue of the ACTUAL discrete column (Robin top, Neumann\n"
       << "  bottom, 3 inverse iterations) and E0 the seed's departure from the\n"
       << "  conductive steady profile ANCHORED AT ITS OWN SURFACE. That anchor\n"
       << "  is an assumption: it prices growing the interior gradient, not\n"
       << "  reaching an unknown final surface temperature.\n"
       << "  Reported per population -- the all-face median hides the lit\n"
       << "  faces behind a frozen dark majority (measured: median drift over\n"
       << "  50->100 lunations is 0.0 K all-face but -6.12 K on lit faces).\n"
       << "  CALIBRATION, so you know what n_req is worth: this is a SINGLE-MODE\n"
          "  model -- the seed's error is assumed to decay as exp(-t/tau1).\n"
          "  Checked once against a measured spin-up, on the production 50-layer\n"
          "  lunar grid: a lit column needs 339 cycles to reach 0.002 K by this\n"
          "  estimate against 150-200 integrated lunations measured, i.e. 1.9x\n"
          "  CONSERVATIVE -- it over-warns rather than leaving you short. Pinned\n"
          "  by build/bin/test_eq_spinup gate 4. tau1 is a property of THIS grid\n"
          "  and state: never compare a number from one grid against a\n"
          "  measurement from another (that error is what BUGS.md PTM-219 was).\n";
    line(os, "dark  (Qs=0)", zero_,  sec_per_cycle_);
    line(os, "lit   (Qs>0)", forced_, sec_per_cycle_);

    if (unreachable > 0) {
        os << "  *** TARGET UNREACHABLE on " << unreachable << " face(s). ***\n"
           << "      The requested " << target_K_ << " K is below this solver "
              "path's arithmetic floor of " << floor_K_ << " K. No cycle count\n"
              "      reaches it -- the limit is arithmetic, not physics, so a\n"
              "      larger equilibration_cycles would only cost time.\n"
              "      Remedy: raise thermal.equilibration_target_K above "
           << floor_K_ << " K, or run a solver path that does not carry that\n"
              "      floor (the float32 arrest is a GPU-path property; the CPU\n"
              "      scalar path is float64 and does not have it).\n";
    }
    const double worst = std::max(pct(zero_.n_req, 0.95), pct(forced_.n_req, 0.95));
    if (n_configured > 0.0 && worst > n_configured) {
        os << "  NOTE: the " << n_configured << " cycles this run integrates is "
              "BELOW the p95 requirement (" << std::fixed << std::setprecision(1)
           << worst << ").\n        Report-only at this stage -- the run proceeds.\n";
    }
    if (!(sec_per_cycle_ > 0.0)) {
        os << "  Wall-clock is not shown: nothing measured a seconds-per-cycle for\n"
              "  this run, and a made-up timing constant would be worse than none.\n";
    }
    os << "=======================================================================\n";
    return true;
}

// ===========================================================================
// RETROSPECTIVE certificate
// ===========================================================================

bool eqInterfaceFlux(const double* z, const double* T, const double* k, int n,
                     double* q_out) {
    if (n < 3 || !z || !T || !k || !q_out) return false;
    for (int j = 0; j < n - 1; ++j) {
        const double dz = z[j + 1] - z[j];
        if (!(dz > 0.0)) return false;
        q_out[j] = k[j] * (T[j + 1] - T[j]) / dz;   // k at the SHALLOWER node
    }
    return true;
}

double eqQbFromState(const double* z, const double* T, const double* k, int n) {
    if (n < 3 || !z || !T || !k) return -1.0;
    const double dz = z[n - 1] - z[n - 2];
    if (!(dz > 0.0)) return -1.0;
    // botTemp(): T[n-1] = T[n-2] + Qb*dz/k[n-1]. k at the DEEPER node.
    return k[n - 1] * (T[n - 1] - T[n - 2]) / dz;
}

double eqGridSkinDepth(const double* z, int n, int grid_b) {
    if (n < 2 || !z || grid_b <= 0) return -1.0;
    const double z_max = z[n - 1];
    if (!(z_max > 0.0)) return -1.0;
    return z_max / static_cast<double>(grid_b);
}

EqCut eqChooseCut(const double* z, int n, double delta_ref) {
    EqCut c;
    if (n < 4 || !z || !(delta_ref > 0.0)) return c;
    const double z_max = z[n - 1];
    if (!(z_max > 0.0)) return c;
    const double five = 5.0 * delta_ref;
    const double half = 0.5 * z_max;
    c.z_target = std::min(five, half);
    c.clamped  = (half < five);

    // First interface at or below the target. Interfaces 0 .. n-2 exist;
    // n-2 is the bottom-BC identity and is excluded by construction.
    int j = 0;
    while (j < n - 2 && z[j] < c.z_target) ++j;
    if (j > n - 3) j = n - 3;
    if (j < 1)     j = 1;
    c.j       = j;
    c.z       = z[j];
    c.n_delta = c.z / delta_ref;
    c.shallow = (c.n_delta < 2.0);
    return c;
}

const char* eqVerdictName(EqVerdict v) {
    switch (v) {
        case EqVerdict::Pass:          return "PASS";
        case EqVerdict::Fail:          return "FAIL";
        case EqVerdict::Indeterminate: return "INDETERMINATE";
        default:                       return "NOT SCORED";
    }
}

namespace {

// One bracket -> one verdict. E_lo/E_hi already carry BOTH uncertainties (the
// un-averaged periodic part of the flux, and an unknown harmonic mean of h).
EqVerdict verdictFromBracket(double e_lo, double e_hi, double target) {
    if (!(e_hi >= e_lo) || !std::isfinite(e_lo)) return EqVerdict::Indeterminate;
    if (std::isfinite(e_hi) && e_hi <= target)   return EqVerdict::Pass;
    if (e_lo > target)                           return EqVerdict::Fail;
    return EqVerdict::Indeterminate;
}

}  // namespace

EqCertifyResult eqCertifyColumn(const EqCertifyInput& in) {
    EqCertifyResult r;
    if (in.n < 4 || !in.z || !in.T || !in.k || !in.rho || !in.cp) return r;
    if (!(in.target_K > 0.0) || !(in.P_ref > 0.0)) return r;

    const int n = in.n;

    // ---- Qb: measured out of the state unless the caller insists -----------
    r.Qb_from_state = eqQbFromState(in.z, in.T, in.k, n);
    r.Qb_used = (in.Qb >= 0.0) ? in.Qb : r.Qb_from_state;
    if (!(r.Qb_used >= 0.0)) return r;

    // ---- cut ---------------------------------------------------------------
    // No fallback. Two defensible skin depths exist for the same column (the
    // grid's own, and the state's surface kappa) and they differ by ~3x on a
    // cold column, which moves the cut by twenty layers. Picking one silently
    // would be a substitution the caller never sees, so an unset delta_ref is
    // a refusal.
    if (!(in.delta_ref > 0.0)) return r;
    r.cut = eqChooseCut(in.z, n, in.delta_ref);
    if (r.cut.j < 1) return r;

    // ---- the residual ------------------------------------------------------
    if (in.qbar) {
        r.q_cut = in.qbar[r.cut.j];
    } else {
        const double dz = in.z[r.cut.j + 1] - in.z[r.cut.j];
        if (!(dz > 0.0)) return r;
        r.q_cut = in.k[r.cut.j] * (in.T[r.cut.j + 1] - in.T[r.cut.j]) / dz;
    }
    r.r_cut = r.q_cut - r.Qb_used;

    // ---- capacity below the cut, and tau1 ----------------------------------
    // Nodes strictly below the cut, EXCLUDING the bottom ghost node (n-1):
    // that node is set algebraically by the bottom BC and carries no capacity
    // in the solver, so counting it would inflate C_below.
    double C = 0.0;
    for (int i = r.cut.j + 1; i < n - 1; ++i) {
        const double dz_up   = in.z[i]     - in.z[i - 1];
        const double dz_down = in.z[i + 1] - in.z[i];
        C += in.rho[i] * in.cp[i] * 0.5 * (dz_up + dz_down);
    }
    r.C_below = C;
    if (!(C > 0.0)) return r;
    r.dTdt_below = -r.r_cut / C;      // SIGN: C dTbar/dt = Qb - qbar = -r

    r.tau1 = (in.tau1 > 0.0)
           ? in.tau1
           : eqColumnTau1(in.z, in.T, in.k, in.rho, in.cp, n, in.eps, in.sigma);

    // ---- flux bracket ------------------------------------------------------
    const double r_abs = std::abs(r.r_cut);
    double r_lo, r_hi;
    if (in.q_is_cycle_mean || in.no_periodic_forcing) {
        r_lo = r_hi = r_abs;
    } else if (in.q_phase_bound > 0.0) {
        r_lo = std::max(0.0, r_abs - in.q_phase_bound);
        r_hi = r_abs + in.q_phase_bound;
    } else {
        r_lo = 0.0;
        r_hi = std::numeric_limits<double>::infinity();
    }

    // ---- the format's own floor --------------------------------------------
    // A float32 checkpoint quantizes the two temperatures the cut gradient is
    // built from, so |r_c| cannot be resolved below k*(ulp(T_j)+ulp(T_j+1))/dz
    // no matter how well the run converged.
    if (in.T_is_float32) {
        const int    j  = r.cut.j;
        const double dz = in.z[j + 1] - in.z[j];
        const auto ulp32 = [](double T) {
            const float f = static_cast<float>(std::abs(T));
            return static_cast<double>(std::nextafterf(f, HUGE_VALF) - f);
        };
        r.q_quant = in.k[j] * (ulp32(in.T[j]) + ulp32(in.T[j + 1])) / dz;
    }

    // ---- observable 1: diurnal maximum -------------------------------------
    if (in.T_max > 0.0) {
        const double h = 4.0 * in.eps * in.sigma * in.T_max * in.T_max * in.T_max;
        if (h > 0.0) {
            r.E_floor = r.q_quant / h;
            r.E_max_lo = r_lo / h;
            r.E_max_hi = r_hi / h;
            r.v_max = verdictFromBracket(r.E_max_lo, r.E_max_hi, in.target_K);
        }
    }

    // ---- observable 1b: diurnal minimum ------------------------------------
    // The same conversion at the OTHER end of the cycle, where the surface is
    // radiatively slackest and therefore moves furthest per watt. This is the
    // binding observable for cold-trap work and it is the one the design note
    // left out; see the note on EqCertifyInput::T_min.
    if (in.T_min > 0.0) {
        const double h = 4.0 * in.eps * in.sigma * in.T_min * in.T_min * in.T_min;
        if (h > 0.0) {
            r.E_min_lo = r_lo / h;
            r.E_min_hi = r_hi / h;
            r.v_min = verdictFromBracket(r.E_min_lo, r.E_min_hi, in.target_K);
        }
    }

    // ---- observable 2: diurnal mean ----------------------------------------
    // The harmonic cycle mean when the caller has it; otherwise the rigorous
    // bracket h_lower <= h_harmonic <= h_upper. min(h) <= harmonic mean <=
    // arithmetic mean <= max(h), so 4 eps sigma T_min^3 and 4 eps sigma T_max^3
    // are both valid ends -- and the ARITHMETIC mean is never substituted for
    // the harmonic one, because that error runs to 8.8x in the direction that
    // hides the problem.
    {
        double h_lo = 0.0, h_hi = 0.0;      // bounds ON h
        if (in.h_harmonic > 0.0) {
            h_lo = h_hi = in.h_harmonic;
        } else if (in.h_lower > 0.0 && in.h_upper >= in.h_lower) {
            h_lo = in.h_lower;
            h_hi = in.h_upper;
        }
        if (h_lo > 0.0) {
            r.E_mean_lo = r_lo / h_hi;      // smallest E: small r, stiff surface
            r.E_mean_hi = r_hi / h_lo;
            r.v_mean = verdictFromBracket(r.E_mean_lo, r.E_mean_hi, in.target_K);
        }
    }

    // ---- observable 3: deep products ---------------------------------------
    // UNCALIBRATED (2.1-3.8x high against the exactly-known zero-forcing fixed
    // point). Reported so a deep-product consumer has a number; deliberately
    // biased high, so a PASS here is meaningful and a FAIL is a prompt to
    // measure rather than a verdict.
    if (r.tau1 > 0.0) {
        r.E_deep_lo = r_lo * r.tau1 / C;
        r.E_deep_hi = r_hi * r.tau1 / C;
        r.v_deep = verdictFromBracket(r.E_deep_lo, r.E_deep_hi, in.target_K);
    }

    r.valid = true;
    return r;
}

// ---------------------------------------------------------------------------
// Census
// ---------------------------------------------------------------------------

EqCertifyCensus::EqCertifyCensus(double target_K, double P_ref_seconds)
    : target_K_(target_K), P_ref_(P_ref_seconds) {}

void EqCertifyCensus::add(const EqCertifyResult& r, bool zero_forcing) {
    std::lock_guard<std::mutex> lk(m_);
    Bin& b = zero_forcing ? zero_ : forced_;
    ++b.n;
    // A column that could not be scored is COUNTED, never dropped -- the same
    // rule as the prospective census, for the same reason.
    if (!r.valid) { ++b.invalid; return; }
    if (r.cut.clamped) ++b.cut_clamped;
    if (r.cut.shallow) ++b.cut_shallow;
    b.r_abs.push_back(std::abs(r.r_cut));
    b.Qb_state.push_back(r.Qb_from_state);

    if (r.E_floor > 0.0) b.E_floor.push_back(r.E_floor);
    struct { Obs* o; EqVerdict v; double e; } rows[4] = {
        {&b.mx,   r.v_max,   r.E_max_hi},
        {&b.mn,   r.v_min,   r.E_min_hi},
        {&b.mean, r.v_mean,  r.E_mean_hi},
        {&b.deep, r.v_deep,  r.E_deep_hi},
    };
    for (const auto& row : rows) {
        switch (row.v) {
            case EqVerdict::Pass:          ++row.o->pass;  break;
            case EqVerdict::Fail:          ++row.o->fail;  break;
            case EqVerdict::Indeterminate: ++row.o->indet; break;
            default:                       ++row.o->not_scored; continue;
        }
        if (std::isfinite(row.e)) row.o->E_hi.push_back(row.e);
    }
}

EqVerdict EqCertifyCensus::worstVerdict() const {
    std::lock_guard<std::mutex> lk(m_);
    long long fail = 0, indet = 0, pass = 0;
    for (const Bin* b : {&zero_, &forced_}) {
        for (const Obs* o : {&b->mx, &b->mn, &b->mean, &b->deep}) {
            fail += o->fail; indet += o->indet; pass += o->pass;
        }
    }
    if (fail  > 0) return EqVerdict::Fail;
    if (indet > 0) return EqVerdict::Indeterminate;
    if (pass  > 0) return EqVerdict::Pass;
    return EqVerdict::NotScored;
}

void EqCertifyCensus::obsLine(std::ostream& os, const char* label, const Obs& o,
                              double target_K) {
    const long long tot = o.pass + o.fail + o.indet + o.not_scored;
    os << "      " << std::left << std::setw(22) << label << std::right;
    if (tot == 0) { os << " (no faces)\n"; return; }
    if (o.not_scored == tot) {
        os << " NOT SCORED (the observable's input was not supplied)\n";
        return;
    }
    os << " pass " << std::setw(9) << o.pass
       << "  FAIL " << std::setw(9) << o.fail
       << "  indet " << std::setw(9) << o.indet;
    if (o.not_scored > 0) os << "  n/s " << o.not_scored;
    if (!o.E_hi.empty()) {
        os << "   E(K): p50 " << std::scientific << std::setprecision(3)
           << pct(o.E_hi, 0.50) << "  p95 " << pct(o.E_hi, 0.95)
           << "  max " << pct(o.E_hi, 1.00) << std::defaultfloat;
    }
    os << "   [target " << target_K << " K]\n";
}

void EqCertifyCensus::line(std::ostream& os, const char* label, const Bin& b,
                           double target_K) {
    os << "    " << std::left << std::setw(16) << label << std::right
       << " n=" << b.n;
    if (b.n == 0) { os << "  (no faces)\n"; return; }
    if (b.invalid == b.n) {
        os << "   NOT SCORED (all faces; degenerate column)\n";
        return;
    }
    if (!b.r_abs.empty()) {
        os << "   |q_c - Qb| (W/m^2): p50 " << std::scientific
           << std::setprecision(3) << pct(b.r_abs, 0.50)
           << "  p95 " << pct(b.r_abs, 0.95) << std::defaultfloat;
    }
    if (!b.Qb_state.empty()) {
        os << "   Qb in the state: " << std::scientific << std::setprecision(9)
           << pct(b.Qb_state, 0.50) << " W/m^2" << std::defaultfloat;
    }
    if (b.invalid > 0) os << "   [" << b.invalid << " NOT SCORED]";
    os << "\n";
    obsLine(os, "diurnal maximum",  b.mx,   target_K);
    obsLine(os, "diurnal MINIMUM",  b.mn,   target_K);
    obsLine(os, "diurnal mean",     b.mean, target_K);
    obsLine(os, "deep (uncalib.)",  b.deep, target_K);
}

bool EqCertifyCensus::report(std::ostream& os, double cycles_used) const {
    std::lock_guard<std::mutex> lk(m_);
    if (zero_.n == 0 && forced_.n == 0) return false;

    os << "\n=== Equilibration certificate (retrospective; did it settle?) ===\n"
       << "  Target accuracy : " << target_K_ << " K"
          "   (thermal.equilibration_target_K)\n";
    if (cycles_used >= 0.0) {
        os << "  Cycles used     : " << cycles_used << "\n";
    }
    os << "  Quantity: r_c = qbar_c - Qb, the CYCLE-MEAN upward interface heat\n"
          "  flux at the cut, against the geothermal flux the state itself\n"
          "  carries. A conservation law on a spatial gradient, so an ARRESTED\n"
          "  column cannot pass it by standing still -- which is exactly what\n"
          "  every increment-based convergence test does pass.\n"
          "  E = |r_c| / h, h = 4*eps*sigma*T^3 for the surface observables\n"
          "  (HARMONIC cycle mean for the diurnal mean; the arithmetic mean is\n"
          "  up to 8.8x optimistic) and E_deep = |r_c|*tau1/C_below.\n"
          "  Ranges are brackets, not error bars: a snapshot of a forced column\n"
          "  does not contain a cycle mean, and an unknown harmonic mean is\n"
          "  bracketed rather than substituted. INDETERMINATE means the data\n"
          "  supplied cannot decide, which is a different thing from a FAIL.\n";
    const long long clamped = zero_.cut_clamped + forced_.cut_clamped;
    const long long shallow = zero_.cut_shallow + forced_.cut_shallow;
    line(os, "dark  (Qs=0)", zero_,  target_K_);
    line(os, "lit   (Qs>0)", forced_, target_K_);

    std::vector<double> floors = zero_.E_floor;
    floors.insert(floors.end(), forced_.E_floor.begin(), forced_.E_floor.end());
    if (!floors.empty()) {
        const double f95 = pct(floors, 0.95);
        os << "  float32 checkpoint floor: E >= " << std::scientific
           << std::setprecision(2) << pct(floors, 0.50) << " K (p50), "
           << f95 << " K (p95)" << std::defaultfloat << "\n"
              "        The 'PTMT' temperature block is float32 on every solver "
              "path, so the cut\n        gradient is quantized. Nothing below "
              "this is resolvable FROM A BANKED STATE\n        however well the "
              "run converged -- it is the format's limit, not the physics'.\n";
        if (target_K_ < f95) {
            os << "  *** The requested " << target_K_ << " K target is BELOW that "
                  "floor on at least 5% of faces. ***\n      Those verdicts are "
                  "about float32, not about equilibration. Certify inline\n      "
                  "(float64) or raise the target.\n";
        }
    }
    if (clamped > 0) {
        os << "  NOTE: on " << clamped << " face(s) the cut hit the z_max/2 bound "
              "before 5 skin depths\n        of the reference period, so the "
              "periodic wave is attenuated less than\n        e^-5 there and "
              "|r_c| carries more un-averaged forcing than the rule intends.\n";
    }
    if (shallow > 0) {
        os << "  *** " << shallow << " face(s) got a cut shallower than TWO skin "
              "depths. ***\n      At that depth the residual is dominated by the "
              "forcing itself, not by\n      equilibration error; treat those "
              "rows as uninformative, and deepen\n      the column (thermal_b) "
              "if you need them scored.\n";
    }
    os << "==================================================================\n";
    return true;
}

}  // namespace thermal
