#include "eq_seed.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "thermal_model.h"   // ABSOLUTE_ZERO_K, MAX_PHYSICAL_TEMP_K

namespace thermal {

bool parseEqSeedMode(const std::string& s, EqSeedMode& out) {
    std::string v;
    v.reserve(s.size());
    for (char c : s) {
        if (c == '_' || c == '-' || c == ' ') continue;
        v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (v.empty() || v == "physical" || v == "default" || v == "perface") {
        out = EqSeedMode::Physical;
        return true;
    }
    if (v == "legacy" || v == "historical") {
        out = EqSeedMode::Legacy;
        return true;
    }
    return false;
}

EqSeedResult eqSeedColumn(const EqSeedInput& in, double* T_out) {
    EqSeedResult r;

    // --- surface anchor: eps*sigma*T0^4 = <F_abs> + Qb ----------------------
    double F  = in.mean_absorbed_flux;
    double Qb = in.Qb;
    if (!(F  >= 0.0)) F  = 0.0;   // also catches NaN
    if (!(Qb >= 0.0)) Qb = 0.0;

    const double denom = in.emissivity * in.sigma;
    double T0 = (denom > 0.0) ? std::sqrt(std::sqrt((F + Qb) / denom)) : 0.0;

    // Lower guard only, and it is physics, not tuning: cp(T) is a polynomial
    // fit (Moon cp0 = -3.6125, Europa cp = 7.49*T with cp0 = 0), so a 0 K node
    // gives cp <= 0 and an infinite or negative diffusivity. Every other
    // temperature site in the solver already floors at ABSOLUTE_ZERO_K; the
    // seed did not. Reachable: a Qb = 0 body (rockox_generic-test.yaml) with a
    // face whose mean absorbed flux is exactly 0.
    if (!(T0 > ABSOLUTE_ZERO_K)) {
        T0 = ABSOLUTE_ZERO_K;
        r.clamped_low = true;
    }
    // NO upper clamp. The old seeds clamped at 400 K; Mercury's subsolar
    // CYCLE MEAN is ~468 K and its peak is 633 K, so that clamp was a silent
    // 68-233 K substitution on a body the engine lets you select. A radiative
    // equilibrium temperature is physical by construction -- but it is worth
    // saying out loud when it exceeds the CPU solver's own MAX_PHYSICAL_TEMP_K
    // clamp, because that clamp will then truncate the run.
    r.above_500K = (T0 > MAX_PHYSICAL_TEMP_K);

    r.T_surface = T0;
    r.T_deep    = T0;

    if (in.n <= 0 || in.z == nullptr) {
        if (T_out && in.n > 0) {
            for (int i = 0; i < in.n; ++i) T_out[i] = T0;
        }
        return r;
    }
    if (T_out) T_out[0] = T0;

    // --- conductive steady gradient: dT/dz = Qb / k(T, z) -------------------
    // Forward integration with k evaluated at the SHALLOWER node, which is the
    // same upwind choice the CN interface conductances make, so the seed is
    // consistent with the stencil that will step it.
    double T = T0;
    for (int i = 1; i < in.n; ++i) {
        const double dz = in.z[i] - in.z[i - 1];
        double k;
        if (in.cond == ConductivityModel::Martinez2021) {
            const double rho_i = in.rho ? in.rho[i - 1] : 0.0;
            k = martinezK(T, rho_i);
        } else {
            const double kc_i = in.kc ? in.kc[i - 1] : 0.0;
            k = hayneK(T, kc_i, in.R350);
        }
        if (k > 0.0 && dz > 0.0) {
            T += Qb * dz / k;
        }
        // k <= 0 (a Martinez density below 1004 kg/m^3, or a zero kc) would
        // give an infinite gradient. Hold the profile instead of producing
        // Inf; martinezDensityIsValid() is the caller's loud guard for that
        // configuration.
        if (T_out) T_out[i] = T;
    }
    r.T_deep = T;
    return r;
}

double analyticSeedCycles(EqSeedMode mode, double explicit_cycles,
                          double planet_day_s, double planet_year_s,
                          int nyearseq) {
    if (explicit_cycles >= 0.0) return explicit_cycles;
    if (mode == EqSeedMode::Physical) return 0.0;
    if (!(planet_day_s > 0.0)) return 0.0;
    return static_cast<double>(nyearseq) *
           std::round(planet_year_s / planet_day_s);
}

// ---------------------------------------------------------------------------
// EqSeedRecorder
// ---------------------------------------------------------------------------
EqSeedRecorder& EqSeedRecorder::instance() {
    static EqSeedRecorder r;
    return r;
}

EqSeedRecorder::EqSeedRecorder() {
    const char* p = std::getenv("PTM_DUMP_SEED");
    if (p && *p != '\0') {
        enabled_ = true;
        path_    = p;
        std::cerr << "[eq-seed] PTM_DUMP_SEED=" << path_
                  << " -- the per-face seed will be written for cross-path"
                     " comparison." << std::endl;
    }
}

void EqSeedRecorder::reserve(std::size_t n_faces) {
    if (!enabled_) return;
    if (rows_.size() < n_faces) rows_.resize(n_faces);
}

void EqSeedRecorder::record(std::size_t face, const EqSeedInput& in,
                            const EqSeedResult& res) {
    if (!enabled_) return;
    if (face >= rows_.size()) return;   // reserve() was not called; drop, quietly
    Row& r   = rows_[face];
    r.mean_flux = in.mean_absorbed_flux;
    r.Qb        = in.Qb;
    r.T_surf    = res.T_surface;
    r.T_deep    = res.T_deep;
    r.T_lowinertia = in.mean_inst_radeq_T;
    r.set       = true;
}

void EqSeedRecorder::write(const std::string& tag) const {
    if (!enabled_ || rows_.empty()) return;
    std::ofstream fh(path_);
    if (!fh) {
        std::cerr << "[eq-seed] WARNING: cannot open '" << path_
                  << "' for writing; NO seed dump produced." << std::endl;
        return;
    }
    fh << std::setprecision(17);
    fh << "# PTM3D shared equilibration seed, path=" << tag << "\n";
    fh << "face,mean_absorbed_flux_Wm2,Qb_Wm2,T_surface_K,T_deep_K,"
          "T_lowinertia_K\n";
    std::size_t written = 0;
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        if (!rows_[i].set) continue;
        fh << i << ',' << rows_[i].mean_flux << ',' << rows_[i].Qb << ','
           << rows_[i].T_surf << ',' << rows_[i].T_deep << ','
           << rows_[i].T_lowinertia << '\n';
        ++written;
    }
    std::cerr << "[eq-seed] wrote " << written << " seed rows to " << path_
              << std::endl;
}

}  // namespace thermal
