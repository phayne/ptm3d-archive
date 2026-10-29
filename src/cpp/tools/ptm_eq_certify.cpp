// ptm_eq_certify -- RETROSPECTIVE equilibration certificate, run against a
// BANKED state. Answers "did the column actually settle?" for a run that has
// already finished, without re-running the simulation.
//
// WHY THIS IS A SEPARATE TOOL. The question is asked after the fact, usually
// about a run that took hours and whose binary is already gone. It needs the
// banked 'PTMT' checkpoint and the statistics next to it, and nothing else --
// no mesh, no view factors, no illumination, no SPICE, no GPU. It is safe to
// run while a production job owns the devices.
//
// THE QUANTITY (see thermal/eq_certificate.h for the derivation):
//     r_c = qbar_c - Qb   at z_c = min(5*delta_ref, z_max/2),  E = |r_c| / h
// a conservation law on a spatial gradient, so a float32-ARRESTED column --
// one that has stopped changing without having converged -- cannot pass it by
// standing still, which is exactly what every increment-based convergence test
// lets it do.
//
// WHAT IT WILL NOT DO
//   * It will not call a snapshot a cycle mean. One state of a forced column
//     does not contain the cycle mean of its interface flux, and the answer
//     for those faces is INDETERMINATE until either several states spanning a
//     cycle are supplied (--state repeated) or a phase bound can be measured
//     from statistics that cover at least one reference period.
//   * It will not substitute the arithmetic mean of 4*eps*sigma*T^3 for the
//     harmonic one (up to 8.8x optimistic). Without a per-step surface series
//     it brackets [4 eps sigma T_min^3, 4 eps sigma T_max^3] and reports the
//     bracket.
//   * It will not read the dead `thermophysical:` scalars out of a config.
//     On a built-in body `ks`, `kd`, `H`, `rhos`, `rhod`, `cp0`, `Qb`, `chi`,
//     `albedo` and `emissivity` under that block are read by NOTHING
//     -- the run used the compiled body defaults regardless
//     of what the file says. Honoring them here would reconstruct a column the
//     engine never integrated. --config lists them and ignores them, out loud.
//   * It will not take Qb from the config either. `geothermal_heat_flow` was
//     inert on every solver entry point for the whole moon3d campaign,
// so Qb is read out of the STATE's own bottom boundary
//     condition and the configured value is printed beside it for comparison.
//
// USAGE
//   ptm_eq_certify --state <file.bin> [--state <file.bin> ...]
//                  [--config <run config.yaml>]
//                  [--body Moon] [--m 10] [--n 15] [--b 32]
//                  [--ks=..] [--kd=..] [--rhos=..] [--rhod=..] [--cp0=..]
//                  [--cpCoeff=c1,c2,..] [--H=..] [--day=..] [--year=..]
//                  [--eps=..] [--chi=..] [--adaptive=0|1] [--Qb=..]
//                  [--period-ref diurnal|annual|<seconds>]
//                  [--delta-ref <m>]
//                  [--stats <statistics dir>] [--window-seconds <s>]
//                  [--target-K 0.1]
//                  [--json out.json] [--csv out.csv] [--csv-stride N]
//
// Exit status: 0 = every scored face PASSED. 2 = at least one FAIL.
// 3 = no FAIL but at least one INDETERMINATE. 1 = the tool refused to run.
// A nonzero status is a result, not a crash.

#include "../thermal/eq_certificate.h"
#include "../thermal/thermal_model.h"
#include "../thermal/thermal_state_io.h"

#ifdef HAVE_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

const double kDay = 86400.0;

[[noreturn]] void refuse(const std::string& why) {
    std::cerr << "\nREFUSED: " << why << "\n";
    std::exit(1);
}

// --------------------------------------------------------------------------
// statistics/<name>.bin -- uint32 N followed by N float32. The same on-disk
// convention as every other statistics file the engine writes.
// --------------------------------------------------------------------------
bool readStatsBin(const std::string& path, std::vector<float>& out,
                  std::string& err) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { err = "cannot open '" + path + "'"; return false; }
    std::uint32_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!is) { err = "'" + path + "' is shorter than its 4-byte header"; return false; }
    out.resize(n);
    is.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(n) * sizeof(float));
    if (!is) { err = "short read on '" + path + "'"; return false; }
    return true;
}

// time_steps.txt: comment lines, then "<elapsed seconds>  <UTC>". The window is
// last minus first.
double windowSecondsFromTimeSteps(const std::string& path) {
    std::ifstream is(path);
    if (!is) return -1.0;
    std::string line;
    double first = 0.0, last = 0.0;
    bool have_first = false;
    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        char* end = nullptr;
        const double v = std::strtod(line.c_str(), &end);
        if (end == line.c_str()) continue;
        if (!have_first) { first = v; have_first = true; }
        last = v;
    }
    return have_first ? (last - first) : -1.0;
}

struct Options {
    std::vector<std::string> states;
    std::string config_path, stats_dir, json_path, csv_path;
    std::string body = "Moon";
    int    m = 10, n = 15, b = 32;
    int    adaptive = -1;            // -1 = leave the Config default
    double ks = -1, kd = -1, rhos = -1, rhod = -1, cp0 = -1e300, H = -1;
    double day = -1, year = -1, eps = -1, chi = -1e300, Qb = -1;
    bool   have_cp_coeff = false;
    std::vector<double> cp_coeff;
    std::string period_ref = "";     // "", "diurnal", "annual", or a number
    double delta_ref = -1.0;
    double window_s = -1.0;
    double target_K = 0.1;
    int    csv_stride = 1;
};

double parseNumber(const std::string& key, const std::string& val) {
    const char* b = val.c_str();
    char* e = nullptr;
    const double v = std::strtod(b, &e);
    if (e == b || *e != '\0')
        refuse("--" + key + " expects one number, got '" + val + "'");
    return v;
}

std::vector<double> parseCsv(const std::string& s) {
    std::vector<double> out;
    std::size_t i = 0;
    while (i <= s.size()) {
        const std::size_t j = s.find(',', i);
        const std::string tok = s.substr(i, (j == std::string::npos ? s.size() : j) - i);
        if (!tok.empty()) out.push_back(parseNumber("cpCoeff", tok));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}

// --------------------------------------------------------------------------
// --config. Reads ONLY the keys the engine actually acts on, and refuses on
// every configuration whose column is not a single reference profile. See the
// header note: honoring the decorative `thermophysical:` scalars would rebuild
// a column the engine never integrated.
// --------------------------------------------------------------------------
void applyConfigFile(Options& o) {
#ifndef HAVE_YAML_CPP
    (void)o;
    refuse("--config needs yaml-cpp, and this binary was built without it. "
           "Pass the properties explicitly instead (--body/--m/--n/--b/--ks=...).");
#else
    YAML::Node root;
    try { root = YAML::LoadFile(o.config_path); }
    catch (const std::exception& e) {
        refuse(std::string("cannot parse '") + o.config_path + "': " + e.what());
    }

    if (root["generic_body"]) {
        refuse("this config uses a `generic_body:` block. Its property mapping "
               "lives in ptm_thermal_sim's applyConfiguredThermo and is not "
               "mirrored here; a second-best copy of it would be exactly the "
               "kind of silent substitution this tool exists to avoid. Pass "
               "--ks= --kd= --rhos= --rhod= --cp0= --H= --day= --eps= --chi= "
               "explicitly (the same overrides dump_thermal_grid takes).");
    }

    const YAML::Node sim = root["simulation"];
    if (sim && sim["body"]) o.body = sim["body"].as<std::string>();

    const YAML::Node th = root["thermal"];
    if (th) {
        if (th["thermal_m"]) o.m = th["thermal_m"].as<int>();
        if (th["thermal_n"]) o.n = th["thermal_n"].as<int>();
        if (th["thermal_b"]) o.b = th["thermal_b"].as<int>();
        auto nonEmpty = [&](const char* key) {
            return th[key] && !th[key].as<std::string>().empty();
        };
        if (nonEmpty("h_parameter_map") || nonEmpty("thermal_inertia_map"))
            refuse("this config loads a per-face property map (h_parameter_map "
                   "or thermal_inertia_map), so the run had a DIFFERENT column "
                   "on every face and one reference profile cannot represent "
                   "it. Per-face maps are not supported by this tool yet.");
        if (th["use_default_property_map"] &&
            th["use_default_property_map"].as<bool>())
            refuse("this config sets use_default_property_map: true, which "
                   "auto-loads a per-face H map. Same reason as above.");
        if (th["geothermal_heat_flow"]) {
            const double v = th["geothermal_heat_flow"].as<double>();
            std::cout << "[config] thermal.geothermal_heat_flow = " << v
                      << " W/m^2 -- NOT used. Qb is read out of the state's own\n"
                         "         bottom boundary condition and the two are "
                         "compared below (: this key was inert on\n"
                         "         every solver entry point for the whole moon3d "
                         "campaign).\n";
        }
    }

    const YAML::Node tp = root["thermophysical"];
    if (tp) {
        if (tp["constant_thermal_inertia"] &&
            tp["constant_thermal_inertia"].as<double>() > 0.0)
            refuse("this config uses constant_thermal_inertia > 0, which "
                   "silently zeroes chi and collapses kd->ks / rhod->rhos "
                   "inside the engine . Reproducing that here "
                   "would duplicate the rule; pass the resulting properties "
                   "explicitly instead.");
        if (tp["conductivity_model"] &&
            tp["conductivity_model"].as<std::string>() != "hayne2017")
            refuse("this config selects conductivity_model '" +
                   tp["conductivity_model"].as<std::string>() +
                   "'. Only hayne2017 is wired through this tool.");
        // The LIVE overrides (applyConfiguredThermo's kScalarOverrides).
        if (tp["ks_override"] && tp["ks_override"].as<double>() > 0.0)
            o.ks = tp["ks_override"].as<double>();
        if (tp["kd_override"] && tp["kd_override"].as<double>() > 0.0)
            o.kd = tp["kd_override"].as<double>();
        if (tp["emissivity_override"] && tp["emissivity_override"].as<double>() > 0.0)
            o.eps = tp["emissivity_override"].as<double>();

        // The DEAD ones. Named, not honored -- the engine ignored them too.
        static const char* const kDead[] = {
            "ks", "kd", "H", "rhos", "rhod", "cp0", "Qb", "chi", "albedo",
            "emissivity"
        };
        std::vector<std::string> present;
        for (const char* key : kDead) if (tp[key]) present.push_back(key);
        if (!present.empty()) {
            std::cout << "[config] IGNORED (read by nothing in the engine either "
                         "-- ): thermophysical.";
            for (std::size_t i = 0; i < present.size(); ++i)
                std::cout << (i ? ", " : "") << present[i];
            std::cout << "\n         The run used the compiled " << o.body
                      << " defaults for these, and so does this audit. Use the\n"
                         "         *_override keys, or pass --ks= --kd= ... to "
                         "this tool, if that is wrong.\n";
        }
    }
#endif
}

Options parseArgs(int argc, char** argv) {
    Options o;
    auto need = [&](int& i, const char* what) -> std::string {
        if (i + 1 >= argc) refuse(std::string(what) + " expects a value");
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        // --key=value form (matches dump_thermal_grid).
        const std::size_t eq = a.find('=');
        if (a.rfind("--", 0) == 0 && eq != std::string::npos) {
            const std::string k = a.substr(2, eq - 2);
            const std::string v = a.substr(eq + 1);
            if      (k == "ks")        o.ks   = parseNumber(k, v);
            else if (k == "kd")        o.kd   = parseNumber(k, v);
            else if (k == "rhos")      o.rhos = parseNumber(k, v);
            else if (k == "rhod")      o.rhod = parseNumber(k, v);
            else if (k == "cp0")       o.cp0  = parseNumber(k, v);
            else if (k == "H")         o.H    = parseNumber(k, v);
            else if (k == "day")       o.day  = parseNumber(k, v);
            else if (k == "year")      o.year = parseNumber(k, v);
            else if (k == "eps")       o.eps  = parseNumber(k, v);
            else if (k == "chi")       o.chi  = parseNumber(k, v);
            else if (k == "Qb")        o.Qb   = parseNumber(k, v);
            else if (k == "adaptive")  o.adaptive = static_cast<int>(parseNumber(k, v));
            else if (k == "m")         o.m = static_cast<int>(parseNumber(k, v));
            else if (k == "n")         o.n = static_cast<int>(parseNumber(k, v));
            else if (k == "b")         o.b = static_cast<int>(parseNumber(k, v));
            else if (k == "target-K")  o.target_K = parseNumber(k, v);
            else if (k == "delta-ref") o.delta_ref = parseNumber(k, v);
            else if (k == "window-seconds") o.window_s = parseNumber(k, v);
            else if (k == "csv-stride") o.csv_stride = static_cast<int>(parseNumber(k, v));
            else if (k == "cpCoeff")   { o.cp_coeff = parseCsv(v); o.have_cp_coeff = true; }
            else if (k == "period-ref") o.period_ref = v;
            else if (k == "state")     o.states.push_back(v);
            else if (k == "config")    o.config_path = v;
            else if (k == "stats")     o.stats_dir = v;
            else if (k == "json")      o.json_path = v;
            else if (k == "csv")       o.csv_path  = v;
            else if (k == "body")      o.body = v;
            else refuse("unrecognized option '--" + k + "'");
            continue;
        }
        if      (a == "--state")   o.states.push_back(need(i, "--state"));
        else if (a == "--config")  o.config_path = need(i, "--config");
        else if (a == "--stats")   o.stats_dir   = need(i, "--stats");
        else if (a == "--json")    o.json_path   = need(i, "--json");
        else if (a == "--csv")     o.csv_path    = need(i, "--csv");
        else if (a == "--body")    o.body        = need(i, "--body");
        else if (a == "--m")       o.m = static_cast<int>(parseNumber("m", need(i, "--m")));
        else if (a == "--n")       o.n = static_cast<int>(parseNumber("n", need(i, "--n")));
        else if (a == "--b")       o.b = static_cast<int>(parseNumber("b", need(i, "--b")));
        else if (a == "--target-K") o.target_K = parseNumber("target-K", need(i, "--target-K"));
        else if (a == "--delta-ref") o.delta_ref = parseNumber("delta-ref", need(i, "--delta-ref"));
        else if (a == "--window-seconds") o.window_s = parseNumber("window-seconds", need(i, "--window-seconds"));
        else if (a == "--csv-stride") o.csv_stride = static_cast<int>(parseNumber("csv-stride", need(i, "--csv-stride")));
        else if (a == "--period-ref") o.period_ref = need(i, "--period-ref");
        else if (a == "--help" || a == "-h") {
            std::cout << "See the header comment of src/cpp/tools/ptm_eq_certify.cpp.\n";
            std::exit(0);
        }
        else refuse("unrecognized argument '" + a + "'");
    }
    if (o.states.empty())
        refuse("no --state given. This tool audits a BANKED 'PTMT' checkpoint; "
               "it has nothing to do without one.");
    if (o.csv_stride < 1) refuse("--csv-stride must be >= 1");
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    Options o = parseArgs(argc, argv);
    if (!o.config_path.empty()) applyConfigFile(o);

    // ---- the column the engine would have built ----------------------------
    thermal::Planet planet(o.body);
    thermal::Config config;
    config.m = o.m; config.n = o.n; config.b = o.b;
    if (o.adaptive >= 0) config.ADAPTIVE_GRID = (o.adaptive != 0);
    if (o.ks   > 0)      planet.ks   = o.ks;
    if (o.kd   > 0)      planet.kd   = o.kd;
    if (o.rhos > 0)      planet.rhos = o.rhos;
    if (o.rhod > 0)      planet.rhod = o.rhod;
    if (o.cp0  > -1e299) planet.cp0  = o.cp0;
    if (o.H    > 0)      planet.H    = o.H;
    if (o.day  > 0)      planet.day  = o.day;
    if (o.year > 0)      planet.year = o.year;
    if (o.eps  > 0)      planet.emissivity = o.eps;
    if (o.chi  > -1e299) config.chi  = o.chi;
    if (o.have_cp_coeff) planet.cpCoeff = o.cp_coeff;

    thermal::Profile ref(planet, 0.0, config, /*verbose=*/false);
    const int n_layers = ref.nlayers;
    std::vector<double> z(ref.z().begin(), ref.z().begin() + n_layers);
    const double z_max = z[n_layers - 1];

    // ---- reference period + skin depth -------------------------------------
    double P_ref = 0.0;
    std::string P_ref_label;
    if (o.period_ref.empty() || o.period_ref == "auto") {
        if (planet.year > planet.day) { P_ref = planet.year; P_ref_label = "annual (auto)"; }
        else                          { P_ref = planet.day;  P_ref_label = "diurnal (auto)"; }
    } else if (o.period_ref == "annual")  { P_ref = planet.year; P_ref_label = "annual"; }
    else if (o.period_ref == "diurnal")   { P_ref = planet.day;  P_ref_label = "diurnal"; }
    else { P_ref = parseNumber("period-ref", o.period_ref); P_ref_label = "explicit"; }
    if (!(P_ref > 0.0)) refuse("the reference period resolved to a non-positive value");

    const double delta0 = thermal::eqGridSkinDepth(z.data(), n_layers, config.b);
    double delta_ref = (o.delta_ref > 0.0)
                     ? o.delta_ref
                     : delta0 * std::sqrt(P_ref / planet.day);
    if (!(delta_ref > 0.0)) refuse("could not resolve a reference skin depth");

    std::printf("=== ptm_eq_certify -- retrospective equilibration certificate ===\n");
    std::printf("body=%s  m=%d n=%d b=%d  adaptive=%d  n_layers=%d  z_max=%.6f m\n",
                o.body.c_str(), config.m, config.n, config.b,
                static_cast<int>(config.ADAPTIVE_GRID), n_layers, z_max);
    std::printf("planet: ks=%g kd=%g rhos=%g rhod=%g H=%g cp0=%g chi=%g eps=%g\n"
                "        day=%.6g s  year=%.6g s  Qb(compiled)=%g W/m^2\n",
                planet.ks, planet.kd, planet.rhos, planet.rhod, planet.H,
                planet.cp0, config.chi, planet.emissivity,
                planet.day, planet.year, planet.Qb);
    std::printf("reference period: %s, P_ref = %.6g s (%.3f days)\n",
                P_ref_label.c_str(), P_ref, P_ref / kDay);
    std::printf("skin depth: grid delta_0 = z_max/b = %.6f m -> delta_ref = %.6f m%s\n",
                delta0, delta_ref, (o.delta_ref > 0.0) ? "  (--delta-ref)" : "");

    // ---- states ------------------------------------------------------------
    std::vector<std::vector<float>> states;
    std::size_t n_faces = 0;
    for (const std::string& path : o.states) {
        // Shape is discovered from the first file, then enforced on the rest.
        std::uint32_t magic = 0, ff = 0, fl = 0;
        {
            std::ifstream is(path, std::ios::binary);
            if (!is) refuse("cannot open state '" + path + "'");
            is.read(reinterpret_cast<char*>(&magic), 4);
            is.read(reinterpret_cast<char*>(&ff), 4);
            is.read(reinterpret_cast<char*>(&fl), 4);
            if (!is) refuse("'" + path + "' is shorter than the 12-byte PTMT header");
        }
        if (magic != thermal::state_io::kMagic)
            refuse("'" + path + "' is not a PTMT checkpoint");
        if (static_cast<int>(fl) != n_layers) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                "'%s' has %u layers but the column reconstructed from these "
                "properties has %d. The checkpoint format does NOT record its "
                "own depth grid (thermal_state_io.h), so this is the only shape "
                "check available -- and it just failed. Fix --m/--n/--b/--body "
                "or the property overrides; do not proceed.",
                path.c_str(), fl, n_layers);
            refuse(buf);
        }
        if (n_faces == 0) n_faces = ff;
        else if (ff != n_faces) refuse("'" + path + "' disagrees on the face count");

        std::vector<float> buf;
        std::string err;
        if (!thermal::state_io::read(path, n_faces, n_layers, buf, err))
            refuse(err);
        states.push_back(std::move(buf));
        std::printf("state: %s  (%zu faces x %d layers)\n",
                    path.c_str(), n_faces, n_layers);
    }
    const int n_states = static_cast<int>(states.size());

    // ---- statistics --------------------------------------------------------
    std::vector<float> t_max, t_min, solar_max;
    bool have_T = false, have_solar = false;
    if (!o.stats_dir.empty()) {
        std::string err;
        have_T = readStatsBin(o.stats_dir + "/temperature_max.bin", t_max, err) &&
                 readStatsBin(o.stats_dir + "/temperature_min.bin", t_min, err);
        if (!have_T)
            std::printf("stats: temperature_max/min NOT read (%s) -- the surface\n"
                        "       observables cannot be scored without them.\n", err.c_str());
        have_solar = readStatsBin(o.stats_dir + "/direct_solar_max.bin", solar_max, err);
        if (have_T && (t_max.size() != n_faces || t_min.size() != n_faces))
            refuse("statistics face count does not match the state's");
        if (have_solar && solar_max.size() != n_faces) have_solar = false;
        if (o.window_s <= 0.0) {
            // ../time_steps.txt sits next to the statistics directory.
            const std::size_t slash = o.stats_dir.find_last_of('/');
            const std::string parent = (slash == std::string::npos)
                                     ? std::string(".")
                                     : o.stats_dir.substr(0, slash);
            o.window_s = windowSecondsFromTimeSteps(parent + "/time_steps.txt");
        }
    }
    const bool window_covers_P_ref = (o.window_s > 0.0 && o.window_s >= P_ref);
    if (o.window_s > 0.0) {
        std::printf("stats window: %.6g s (%.3f days) = %.4f reference periods%s\n",
                    o.window_s, o.window_s / kDay, o.window_s / P_ref,
                    window_covers_P_ref ? "" : "   *** SHORTER THAN P_ref ***");
    }

    // ---- how the cycle mean is established, said before any number ---------
    // Three routes, in decreasing strength. Whichever applies is reported per
    // face; none of them is assumed.
    std::printf("\ncycle-mean provenance:\n");
    if (n_states > 1) {
        std::printf("  %d states supplied -> qbar is their mean and the "
                    "half-range across them\n"
                    "  is the measured phase bound. This is only a cycle mean "
                    "if the states\n  actually span the reference period; the "
                    "tool cannot verify that and\n  does not claim it.\n", n_states);
    } else {
        std::printf("  ONE state supplied -> qbar is a SNAPSHOT.\n");
    }
    if (have_T && window_covers_P_ref) {
        std::printf("  Phase bound from statistics: "
                    "A_q <= Gamma*sqrt(omega_ref)*(T_max-T_min)/2, attenuated by\n"
                    "  exp(-z_c/delta_ref). Conservative -- it charges the whole "
                    "window swing to\n  the reference harmonic, and the diurnal "
                    "part of that swing is attenuated\n  far harder than "
                    "exp(-z_c/delta_ref).\n");
    } else if (have_T) {
        std::printf("  Phase bound NOT established: the statistics window is "
                    "shorter than one\n  reference period, so a swing larger "
                    "than the observed one cannot be ruled\n  out. Faces that "
                    "did not move at all over the window are still treated as\n"
                    "  unforced, and counted separately below.\n");
    } else {
        std::printf("  No statistics -> no phase bound and no surface "
                    "observables.\n");
    }

    // ---- per-face certification -------------------------------------------
    thermal::EqCertifyCensus census(o.target_K, P_ref);
    const double sigma = ptm::kStefanBoltzmann;
    const double omega_ref = 2.0 * M_PI / P_ref;

    long long n_steady = 0, n_pass = 0, n_fail = 0, n_indet = 0, n_ns = 0;
    std::vector<double> csv_rows;    // flattened, written after the loop
    const std::size_t kCsvCols = 10;

    std::ofstream csv;
    if (!o.csv_path.empty()) {
        csv.open(o.csv_path);
        if (!csv) refuse("cannot open '" + o.csv_path + "' for writing");
        csv << "face,j_cut,z_cut_m,Qb_state_Wm2,q_cut_Wm2,r_cut_Wm2,"
               "dTdt_below_K_per_s,E_max_hi_K,E_mean_hi_K,verdict_max\n";
    }

    #pragma omp parallel
    {
        // One Profile per thread, reused. Constructing 500k of them would cost
        // more than the audit; reusing one keeps the ENGINE's property model in
        // the loop, which a hand-rolled k(T)/cp(T) would not.
        thermal::Profile prof(planet, 0.0, config, /*verbose=*/false);
        std::vector<double> T(n_layers), kk(n_layers), rr(n_layers), cc(n_layers);
        std::vector<double> q(n_layers - 1), qsum(n_layers - 1);
        std::vector<double> qmin(n_layers - 1), qmax(n_layers - 1);
        std::vector<double> local_csv;
        long long l_steady = 0, l_pass = 0, l_fail = 0, l_indet = 0, l_ns = 0;

        #pragma omp for schedule(static)
        for (long long f = 0; f < static_cast<long long>(n_faces); ++f) {
            std::fill(qsum.begin(), qsum.end(), 0.0);
            std::fill(qmin.begin(), qmin.end(),  1e300);
            std::fill(qmax.begin(), qmax.end(), -1e300);

            for (int s = 0; s < n_states; ++s) {
                const std::vector<float>& st = states[s];
                for (int l = 0; l < n_layers; ++l)
                    T[l] = static_cast<double>(st[static_cast<std::size_t>(l) * n_faces + f]);
                for (int l = 0; l < n_layers; ++l) prof.T()[l] = T[l];
                prof.update_cp();
                prof.update_k();
                if (!thermal::eqInterfaceFlux(z.data(), T.data(), prof.k().data(),
                                              n_layers, q.data()))
                    continue;
                for (int j = 0; j < n_layers - 1; ++j) {
                    qsum[j] += q[j];
                    qmin[j] = std::min(qmin[j], q[j]);
                    qmax[j] = std::max(qmax[j], q[j]);
                }
            }
            // Properties/T left in `prof` are the LAST state's -- that is the
            // state the column is certified AS, and it is the one tau1, C_below
            // and Qb_from_state are read from.
            for (int l = 0; l < n_layers; ++l) {
                kk[l] = prof.k()[l];
                rr[l] = prof.rho()[l];
                cc[l] = prof.cp()[l];
            }
            for (int j = 0; j < n_layers - 1; ++j) qsum[j] /= n_states;

            thermal::EqCertifyInput in;
            in.z = z.data(); in.T = T.data(); in.k = kk.data();
            in.rho = rr.data(); in.cp = cc.data();
            in.n = n_layers;
            in.qbar = qsum.data();
            in.Qb = (o.Qb >= 0.0) ? o.Qb : -1.0;
            in.eps = planet.emissivity;
            in.sigma = sigma;
            in.P_ref = P_ref;
            in.delta_ref = delta_ref;
            in.target_K = o.target_K;
            in.T_is_float32 = true;   // it came out of a PTMT checkpoint

            bool dark = false;
            double swing = -1.0;
            if (have_T) {
                const double tmx = t_max[f], tmn = t_min[f];
                if (tmx > 0.0) {
                    in.T_max   = tmx;
                    in.h_upper = 4.0 * in.eps * sigma * tmx * tmx * tmx;
                    if (tmn > 0.0) {
                        in.T_min   = tmn;
                        in.h_lower = 4.0 * in.eps * sigma * tmn * tmn * tmn;
                    }
                    swing = tmx - tmn;
                }
            }
            dark = have_solar ? !(solar_max[f] > 0.0f)
                              : (swing == 0.0);

            // Provenance, in decreasing strength.
            if (n_states > 1) {
                // Half-range across the supplied states, at the cut. Measured,
                // not modeled -- but it is only a bound on the periodic content
                // the samples actually resolve.
                const thermal::EqCut cut = thermal::eqChooseCut(z.data(), n_layers, delta_ref);
                if (cut.j >= 0)
                    in.q_phase_bound = 0.5 * (qmax[cut.j] - qmin[cut.j]);
            }
            if (swing == 0.0) {
                // The surface did not move by a single float32 bit over the
                // window. Treated as unforced. When the window is shorter than
                // P_ref this cannot exclude a longer period, and those faces
                // are counted so the assumption is visible.
                in.no_periodic_forcing = true;
                if (!window_covers_P_ref) ++l_steady;
            } else if (swing > 0.0 && window_covers_P_ref) {
                const double Gamma = std::sqrt(kk[0] * rr[0] * cc[0]);
                const thermal::EqCut cut = thermal::eqChooseCut(z.data(), n_layers, delta_ref);
                if (cut.j >= 0) {
                    const double bound = Gamma * std::sqrt(omega_ref) * 0.5 * swing *
                                         std::exp(-cut.z / delta_ref);
                    in.q_phase_bound = std::max(in.q_phase_bound, bound);
                }
            }

            const thermal::EqCertifyResult r = thermal::eqCertifyColumn(in);
            census.add(r, dark);
            switch (r.v_max) {
                case thermal::EqVerdict::Pass:          ++l_pass;  break;
                case thermal::EqVerdict::Fail:          ++l_fail;  break;
                case thermal::EqVerdict::Indeterminate: ++l_indet; break;
                default:                                ++l_ns;    break;
            }

            if (csv.is_open() && (f % o.csv_stride) == 0) {
                local_csv.push_back(static_cast<double>(f));
                local_csv.push_back(r.cut.j);
                local_csv.push_back(r.cut.z);
                local_csv.push_back(r.Qb_from_state);
                local_csv.push_back(r.q_cut);
                local_csv.push_back(r.r_cut);
                local_csv.push_back(r.dTdt_below);
                local_csv.push_back(r.E_max_hi);
                local_csv.push_back(r.E_mean_hi);
                local_csv.push_back(static_cast<double>(static_cast<int>(r.v_max)));
            }
        }
        #pragma omp critical
        {
            n_steady += l_steady; n_pass += l_pass; n_fail += l_fail;
            n_indet += l_indet; n_ns += l_ns;
            csv_rows.insert(csv_rows.end(), local_csv.begin(), local_csv.end());
        }
    }

    if (csv.is_open()) {
        // Sorted by face so the file is reproducible regardless of scheduling.
        std::vector<std::size_t> order(csv_rows.size() / kCsvCols);
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return csv_rows[a * kCsvCols] < csv_rows[b * kCsvCols];
        });
        static const char* const kV[] = {"NOT_SCORED", "PASS", "FAIL", "INDETERMINATE"};
        for (std::size_t i : order) {
            const double* rrow = &csv_rows[i * kCsvCols];
            csv << static_cast<long long>(rrow[0]) << ','
                << static_cast<int>(rrow[1]) << ',';
            csv.precision(10);
            csv << rrow[2] << ',' << rrow[3] << ',' << rrow[4] << ',' << rrow[5]
                << ',' << rrow[6] << ',' << rrow[7] << ',' << rrow[8] << ','
                << kV[std::min(3, std::max(0, static_cast<int>(rrow[9])))] << '\n';
        }
        csv.close();
        std::printf("wrote %s (%zu rows, stride %d)\n", o.csv_path.c_str(),
                    csv_rows.size() / kCsvCols, o.csv_stride);
    }

    census.report(std::cout, /*cycles_used=*/-1.0);

    if (n_steady > 0) {
        std::printf("\n*** %lld face(s) were treated as UNFORCED because their "
                    "surface temperature did\n    not move by one float32 bit "
                    "over the statistics window -- but that window is\n    "
                    "%.4f reference periods, so a longer-period forcing is not "
                    "excluded by it.\n    Their verdicts rest on that "
                    "assumption. Point --stats at a window covering at\n    "
                    "least one P_ref (the MERGED statistics of a chunked run) "
                    "to remove it.\n", n_steady, o.window_s / P_ref);
    }
    std::printf("\ndiurnal-maximum roll-up: %lld PASS, %lld FAIL, %lld "
                "INDETERMINATE, %lld not scored\n", n_pass, n_fail, n_indet, n_ns);

    if (!o.json_path.empty()) {
        std::ofstream js(o.json_path);
        if (!js) refuse("cannot open '" + o.json_path + "' for writing");
        js.precision(12);
        js << "{\n"
           << "  \"body\": \"" << o.body << "\",\n"
           << "  \"n_faces\": " << n_faces << ",\n"
           << "  \"n_layers\": " << n_layers << ",\n"
           << "  \"z_max_m\": " << z_max << ",\n"
           << "  \"grid_delta0_m\": " << delta0 << ",\n"
           << "  \"P_ref_s\": " << P_ref << ",\n"
           << "  \"delta_ref_m\": " << delta_ref << ",\n"
           << "  \"target_K\": " << o.target_K << ",\n"
           << "  \"n_states\": " << n_states << ",\n"
           << "  \"stats_window_s\": " << o.window_s << ",\n"
           << "  \"window_covers_P_ref\": " << (window_covers_P_ref ? "true" : "false") << ",\n"
           << "  \"steadiness_assumed_beyond_window\": " << n_steady << ",\n"
           << "  \"diurnal_maximum\": {\"pass\": " << n_pass << ", \"fail\": " << n_fail
           << ", \"indeterminate\": " << n_indet << ", \"not_scored\": " << n_ns << "}\n"
           << "}\n";
        std::printf("wrote %s\n", o.json_path.c_str());
    }

    if (n_fail  > 0) return 2;
    if (n_indet > 0) return 3;
    return 0;
}
