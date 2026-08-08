// dump_thermal_grid — print the engine's OWN depth grid and layer properties.
//
// WHY THIS EXISTS. The 2026-08-08 offset investigation needed to audit the
// discrete interface flux q_i = k_{i+1/2}(T_i - T_{i+1})/dz_i on a banked state.
// Every attempt to do that from outside the engine has to RECONSTRUCT z, kc and
// rho, and a reconstruction that is wrong by a few percent is indistinguishable
// from an energy-conservation defect. That ambiguity blocked the audit:
// a transcription of the kernel stencil failed its own control (it implied
// 1e-4 K/step on states that are provable float32 fixed points), so the measured
// 9.2 % flux deficit could not be attributed between "real non-conservation"
// and "my grid is wrong".
//
// This tool removes the ambiguity by printing the arrays the solver actually
// builds — `thermal::Profile`'s z/dz/kc/rho/kappa — for one face. It runs the
// same construction path the GPU driver uses
// (`thermal::Planet(name)` + `applyConfiguredThermo` + `thermal::Profile`),
// touches NO GPU, loads no mesh and runs no simulation, so it can be run safely
// while a production job owns the GPUs.
//
// Usage:
//   dump_thermal_grid [body] [m] [n] [b] [T_uniform_K]
// Defaults reproduce the moon3d chunk_000 configuration:
//   dump_thermal_grid Moon 10 15 32 30.0
//
// Output is CSV on stdout: i,z_m,dz_m,kc,rho,k_at_T,kappa
#include "../thermal/thermal_model.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    const std::string body = (argc > 1) ? argv[1] : "Moon";
    const int    m  = (argc > 2) ? std::atoi(argv[2]) : 10;
    const int    n  = (argc > 3) ? std::atoi(argv[3]) : 15;
    const int    b  = (argc > 4) ? std::atoi(argv[4]) : 32;
    const double Tu = (argc > 5) ? std::atof(argv[5]) : 30.0;

    thermal::Planet planet(body);
    thermal::Config config;
    config.m = m;
    config.n = n;
    config.b = b;

    thermal::Profile prof(planet, 0.0, config, /*verbose=*/false);

    std::fprintf(stderr,
                 "# body=%s m=%d n=%d b=%d  ADAPTIVE_GRID=%d  nlayers=%d\n"
                 "# planet: ks=%g kd=%g rhos=%g rhod=%g H=%g cp0=%g chi=%g Qb=%g eps=%g day=%g\n"
                 "# NOTE thermal_n is only consulted by the LEGACY non-adaptive grid path\n",
                 body.c_str(), m, n, b, static_cast<int>(config.ADAPTIVE_GRID),
                 prof.nlayers, planet.ks, planet.kd, planet.rhos, planet.rhod,
                 planet.H, planet.cp0, config.chi, planet.Qb, planet.emissivity,
                 planet.day);

    const auto& z   = prof.z();
    const auto& kc  = prof.kc();
    const auto& rho = prof.rho();
    const double R350 = config.getR350();

    std::printf("i,z_m,dz_m,kc,rho,k_at_T,kappa_at_T\n");
    for (int i = 0; i < prof.nlayers; ++i) {
        const double dz = (i + 1 < prof.nlayers) ? (z[i + 1] - z[i]) : 0.0;
        const double k  = kc[i] * (1.0 + R350 * Tu * Tu * Tu);
        // cp uses the same representative-polynomial form as the kernel.
        double cp = planet.cp0, Tp = Tu;
        for (size_t j = 0; j < planet.cpCoeff.size(); ++j) { cp += planet.cpCoeff[j] * Tp; Tp *= Tu; }
        if (cp <= 0.0) cp = 800.0;
        std::printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    i, z[i], dz, kc[i], rho[i], k, k / (rho[i] * cp));
    }
    return 0;
}
