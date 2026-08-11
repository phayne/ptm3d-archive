#include "eq_certificate.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
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

}  // namespace thermal
