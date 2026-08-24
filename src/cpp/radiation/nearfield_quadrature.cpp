#include "nearfield_quadrature.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace radiation {

void nearFieldQuadOrderClampWarning(int order) {
    std::cerr << "[view-factor] WARNING: nearfield_quadrature_order = " << order
              << " is outside 1.." << kMaxNearFieldQuadOrder
              << "; CLAMPING. The near-field view factors below are NOT the ones "
                 "this configuration asked for." << std::endl;
}

void reportNearFieldBoundClamps(const char* builder, int order) {
    const long long n = detail::g_nearfield_bound_clamps.exchange(0);
    const int worst = detail::g_nearfield_bound_worst_ratio_ppm.exchange(0);
    if (n == 0) return;
    std::cerr << "\n  *** NEAR-FIELD VIEW-FACTOR BOUND CLAMP (" << builder << "): " << n
              << " pair(s) exceeded A_i F_ij <= min(A_i, A_j) and were clamped."
              << "\n      Worst overshoot " << (worst / 10000.0) << "% above the bound."
              << "\n      Cause: the near-field quadrature diverges on a concave crease"
              << "\n      sharper than roughly 50 degrees (order " << order
              << " here). Either the mesh has folded"
              << "\n      or near-degenerate adjacent facets, or it genuinely has such"
              << "\n      creases and radiation.nearfield_quadrature_order must be raised."
              << "\n      The clamped entries are BOUNDED but not accurate. ***\n"
              << std::endl;
}

bool visibilityRayLegacyRequested() {
    static const bool on = [] {
        const char* e = std::getenv("PTM_VF_RAY_LEGACY");
        const bool want = (e && std::string(e) == "1");
        if (want) {
            std::cerr << "[view-factor] WARNING: PTM_VF_RAY_LEGACY=1 -- occlusion rays use "
                         "the retired absolute-meter margins (0.5 m, auto-visible below "
                         "0.1 m). On a sub-meter mesh that skips the occlusion test "
                         "entirely. This is a control, not a production setting."
                      << std::endl;
        }
        return want;
    }();
    return on;
}

bool directedStorageFloorRequested() {
    static const bool on = [] {
        const char* e = std::getenv("PTM_VF_DIRECTED_FLOOR");
        const bool want = (e && std::string(e) == "1");
        if (want) {
            std::cerr << "[view-factor] WARNING: PTM_VF_DIRECTED_FLOOR=1 -- the storage "
                         "floor is applied to the DIRECTED F_ij again, so an "
                         "area-asymmetric pair can be stored one-sided. Control only."
                      << std::endl;
        }
        return want;
    }();
    return on;
}

bool legacyPrecullRequested() {
    static const bool on = [] {
        const char* e = std::getenv("PTM_HMATRIX_PRECULL_LEGACY");
        const bool want = (e && std::string(e) == "1");
        if (want) {
            std::cerr << "[view-factor] WARNING: PTM_HMATRIX_PRECULL_LEGACY=1 -- near "
                         "pairs are pre-culled on their M1 estimate again, which can "
                         "delete a pair the refined kernel would keep. Control only."
                      << std::endl;
        }
        return want;
    }();
    return on;
}

void reportVisibilityUnresolvable(const char* builder) {
    const long long n = detail::g_visibility_unresolvable.exchange(0);
    if (n == 0) return;
    std::cerr << "\n  *** OCCLUSION RAY NOT RESOLVABLE IN FLOAT32 (" << builder << "): "
              << n << " pair(s)."
              << "\n      The margin needed to survive the float32 cast of the ray origin"
              << "\n      is larger than the source facet itself, so the segment carries no"
              << "\n      information. The scene's absolute coordinates are too big for its"
              << "\n      facets: recenter the mesh (the engine's radiation scene does this"
              << "\n      automatically; vf_dump_sparse and the harnesses do not). Those"
              << "\n      pairs were recorded VISIBLE. ***\n" << std::endl;
}

}  // namespace radiation
