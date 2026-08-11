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
//   dump_thermal_grid [body] [m] [n] [b] [T_uniform_K] [--key=value ...]
// Defaults reproduce the moon3d chunk_000 configuration:
//   dump_thermal_grid Moon 10 15 32 30.0
//
// Optional overrides let this tool reproduce a GENERIC-BODY column, whose
// properties come from the config's `generic_body.surface:` block and so are
// not reachable through thermal::Planet's name lookup. Recognized keys:
//   --ks= --kd= --rhos= --rhod= --cp0= --H= --day= --eps= --chi= --adaptive=
// e.g. the shipped asteroid quickstart (examples/quickstart_asteroid_diurnal.yaml):
//   dump_thermal_grid Moon 10 15 15 200 --ks=0.008333333 --kd=0.008333333 \
//       --rhos=1500 --rhod=1500 --cp0=800 --H=0.02 --day=18000 --eps=0.95 --chi=0
//
// Output is CSV on stdout:
//   i,z_m,dz_from_z_m,dz_solver_m,dz_ratio,kc,rho,k_at_T,kappa_at_T
#include "../thermal/thermal_model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    // Property overrides. An UNRECOGNIZED --key is a hard error rather than a
    // silent skip: a typo'd override would otherwise dump the wrong body's grid
    // under the right-looking command line.
    for (int a = 6; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg.rfind("--", 0) != 0) {
            std::fprintf(stderr, "FATAL: expected --key=value, got '%s'\n", argv[a]);
            return 2;
        }
        const size_t eq = arg.find('=');
        if (eq == std::string::npos) {
            std::fprintf(stderr, "FATAL: expected --key=value, got '%s'\n", argv[a]);
            return 2;
        }
        const std::string key = arg.substr(2, eq - 2);
        // strtod, not atof: atof silently stops at the first unparsable
        // character, so a shell that failed to word-split the argument list
        // would hand this loop one giant token and get the FIRST override
        // applied and the rest dropped without a word. (Observed; it produced
        // a plausible-looking grid for the wrong body.)
        const char* vbeg = arg.c_str() + eq + 1;
        char* vend = nullptr;
        const double val = std::strtod(vbeg, &vend);
        if (vend == vbeg || *vend != '\0') {
            std::fprintf(stderr,
                         "FATAL: --%s= expects one number, got '%s'"
                         " (unsplit argument list?)\n", key.c_str(), vbeg);
            return 2;
        }
        if      (key == "ks")   planet.ks   = val;
        else if (key == "kd")   planet.kd   = val;
        else if (key == "rhos") planet.rhos = val;
        else if (key == "rhod") planet.rhod = val;
        else if (key == "cp0")  { planet.cp0 = val; planet.cpCoeff.clear(); }
        else if (key == "H")    planet.H    = val;
        else if (key == "day")  planet.day  = val;
        else if (key == "eps")  planet.emissivity = val;
        else if (key == "chi")  config.chi  = val;
        else if (key == "adaptive") config.ADAPTIVE_GRID = (val != 0.0);
        else {
            std::fprintf(stderr, "FATAL: unrecognized override '--%s'\n", key.c_str());
            return 2;
        }
    }

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

    // Two SEPARATE thicknesses, deliberately both printed.
    //   dz_from_z_m  = z[i+1] - z[i], the geometry the depth grid states, and
    //                  the one the GPU path re-derives (thermal_gpu.cpp:319).
    //   dz_solver_m  = Profile::dz()[i], the geometry the CPU conduction
    //                  stencil actually integrates (g1/g2, surfTemp, botTemp).
    // They are equal only when no thickness floor has intervened. Printing the
    // z-difference alone (the original form of this tool) cannot see a dz-only
    // floor and would report a grid the CPU solver does not use.
    std::printf("i,z_m,dz_from_z_m,dz_solver_m,dz_ratio,kc,rho,k_at_T,kappa_at_T\n");
    for (int i = 0; i < prof.nlayers; ++i) {
        const double dz = (i + 1 < prof.nlayers) ? (z[i + 1] - z[i]) : 0.0;
        const double dzs = (i + 1 < prof.nlayers) ? prof.dz()[i] : 0.0;
        const double k  = kc[i] * (1.0 + R350 * Tu * Tu * Tu);
        // cp uses the same representative-polynomial form as the kernel.
        double cp = planet.cp0, Tp = Tu;
        for (size_t j = 0; j < planet.cpCoeff.size(); ++j) { cp += planet.cpCoeff[j] * Tp; Tp *= Tu; }
        if (cp <= 0.0) cp = 800.0;
        std::printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    i, z[i], dz, dzs, (dz > 0.0 ? dzs / dz : 1.0),
                    kc[i], rho[i], k, k / (rho[i] * cp));
    }
    return 0;
}
