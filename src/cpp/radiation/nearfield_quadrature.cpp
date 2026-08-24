#include "nearfield_quadrature.h"

#include <iostream>

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

}  // namespace radiation
