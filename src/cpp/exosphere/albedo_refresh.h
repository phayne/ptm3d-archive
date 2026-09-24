/*
 * Albedo -> temperature refresh bookkeeping for --albedo-coupling.
 *
 * The exosphere does not re-solve the thermal model when the surface albedo
 * changes. It rescales the thermal-run temperatures by the radiative-equilibrium
 * factor T_new = T_ref * ((1 - A_new) / (1 - A_ref))^(1/4). For that to be
 * right, A_ref must be the albedo the CURRENT temperature array corresponds to:
 * the thermal run's own per-face albedo until the first refresh, and the albedo
 * of the last refresh after that.
 *
 * AlbedoRefreshTracker keeps that per-face baseline. The refresh fires when
 * max_f |A_current(f) - A_ref(f)| exceeds the threshold, so slow drift
 * accumulates until it fires instead of being measured pass-to-pass (where a
 * drift below the threshold per pass never fires). When it fires, EVERY face
 * with a nonzero deviation is rescaled relative to its own baseline and the
 * baseline moves to the current albedo, so no sub-threshold drift is lost.
 *
 * Header-only so the unit test exercises the exact code the engine runs.
 */

#ifndef PTM_EXOSPHERE_ALBEDO_REFRESH_H
#define PTM_EXOSPHERE_ALBEDO_REFRESH_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace ptm {
namespace exosphere {

/** Result of one AlbedoRefreshTracker::update() call. */
struct AlbedoRefreshResult {
    bool fired = false;            // threshold exceeded, temperatures rescaled
    double max_deviation = 0.0;    // max_f |A_current - A_ref| BEFORE the update
    size_t worst_face = 0;         // face attaining max_deviation
    size_t faces_rescaled = 0;     // faces whose temperatures were multiplied
    size_t faces_singular = 0;     // faces skipped because 1 - A_ref ~ 0 (baseline kept)
};

class AlbedoRefreshTracker {
public:
    /** The radiative-equilibrium temperature factor from A_ref to A_new. */
    static double scaleFactor(double a_ref, double a_new) {
        return std::pow((1.0 - a_new) / (1.0 - a_ref), 0.25);
    }

    /** Seed the baseline with the albedo the temperature array was computed with. */
    void seed(const std::vector<double>& albedo_of_temperatures) {
        baseline_ = albedo_of_temperatures;
    }

    bool seeded() const { return !baseline_.empty(); }
    const std::vector<double>& baseline() const { return baseline_; }

    /** max_f |current(f) - baseline(f)|, and the face attaining it. */
    double maxDeviation(const std::vector<double>& current, size_t* worst_face = nullptr) const {
        double dmax = 0.0;
        size_t imax = 0;
        const size_t n = std::min(current.size(), baseline_.size());
        for (size_t f = 0; f < n; ++f) {
            const double d = std::abs(current[f] - baseline_[f]);
            if (d > dmax) { dmax = d; imax = f; }
        }
        if (worst_face) *worst_face = imax;
        return dmax;
    }

    /**
     * If maxDeviation(current) > threshold, rescale temperatures[f][*] for every
     * face with a deviation above rescale_min (default 0: every face with a
     * nonzero deviation) by scaleFactor(baseline[f], current[f]) and
     * set baseline[f] = current[f]. Otherwise change nothing (the deviation keeps
     * accumulating against the unchanged baseline).
     *
     * A face with 1 - baseline < 1e-10 cannot be rescaled; it keeps its
     * baseline and is counted in faces_singular (the caller must report it).
     */
    AlbedoRefreshResult update(const std::vector<double>& current,
                               std::vector<std::vector<double>>& temperatures,
                               double threshold, double rescale_min = 0.0) {
        AlbedoRefreshResult r;
        r.max_deviation = maxDeviation(current, &r.worst_face);
        if (!(r.max_deviation > threshold)) return r;
        r.fired = true;
        const size_t n = std::min({current.size(), baseline_.size(), temperatures.size()});
        size_t rescaled = 0, singular = 0;
        #pragma omp parallel for reduction(+:rescaled,singular)
        for (size_t f = 0; f < n; ++f) {
            if (current[f] == baseline_[f]) continue;
            // PTM-657 face-local refresh: faces at or below rescale_min keep their baseline.
            if (rescale_min > 0.0 && !(std::abs(current[f] - baseline_[f]) > rescale_min)) continue;
            if (1.0 - baseline_[f] < 1e-10) { ++singular; continue; }
            const double s = scaleFactor(baseline_[f], current[f]);
            for (double& T : temperatures[f]) T *= s;
            baseline_[f] = current[f];
            ++rescaled;
        }
        r.faces_rescaled = rescaled;
        r.faces_singular = singular;
        return r;
    }

private:
    std::vector<double> baseline_;
};

/** Per-face comparison of the thermal-run albedo with the t = 0 composition albedo. */
struct AlbedoMismatchReport {
    size_t n_faces = 0;
    size_t n_over = 0;            // faces with |A_comp - A_thermal| > tolerance
    double max_abs = 0.0;
    double mean_signed = 0.0;     // mean of (A_comp - A_thermal) over the compared faces
    size_t worst_face = 0;
    double worst_thermal = 0.0;
    double worst_composition = 0.0;
};

inline AlbedoMismatchReport compareThermalAndCompositionAlbedo(
    const std::vector<double>& thermal, const std::vector<double>& composition,
    double tolerance, const std::vector<char>* exclude = nullptr) {
    AlbedoMismatchReport rep;
    const size_t n = std::min(thermal.size(), composition.size());
    double sum = 0.0;
    for (size_t f = 0; f < n; ++f) {
        if (exclude && f < exclude->size() && (*exclude)[f]) continue;
        const double d = composition[f] - thermal[f];
        ++rep.n_faces;
        sum += d;
        if (std::abs(d) > tolerance) ++rep.n_over;
        if (std::abs(d) > rep.max_abs) {
            rep.max_abs = std::abs(d);
            rep.worst_face = f;
            rep.worst_thermal = thermal[f];
            rep.worst_composition = composition[f];
        }
    }
    rep.mean_signed = rep.n_faces ? sum / static_cast<double>(rep.n_faces) : 0.0;
    return rep;
}

/**
 * Print a loud WARNING if any face differs by more than `tolerance`, followed by
 * `consequence` (what the engine does about it; newline-terminated lines).
 * Returns true if the warning was printed.
 */
inline bool warnThermalCompositionMismatch(const AlbedoMismatchReport& rep, double tolerance,
                                           const std::string& what, const std::string& consequence,
                                           std::ostream& os,
                                           const std::string& other_name = "composition") {
    if (rep.n_over == 0) return false;
    const std::ios_base::fmtflags flags = os.flags();
    const std::streamsize prec = os.precision();
    os << std::defaultfloat << "\n********************************************************************************\n"
       << "WARNING: t = 0 albedo MISMATCH (" << what << "): the " << other_name << " albedo\n"
       << "  differs from the albedo the thermal run used by more than " << tolerance << " on "
       << rep.n_over << " of " << rep.n_faces << " faces.\n"
       << std::setprecision(6)
       << "  max |A_" << other_name << " - A_thermal| = " << rep.max_abs << " at face " << rep.worst_face
       << " (thermal " << rep.worst_thermal << ", " << other_name << " " << rep.worst_composition << ")\n"
       << "  mean (A_" << other_name << " - A_thermal) = " << rep.mean_signed << "\n"
       << consequence
       << "********************************************************************************\n";
    os.flags(flags);
    os.precision(prec);
    return true;
}

}  // namespace exosphere
}  // namespace ptm

#endif  // PTM_EXOSPHERE_ALBEDO_REFRESH_H
