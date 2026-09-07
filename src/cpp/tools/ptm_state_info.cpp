// ptm_state_info -- "what is this checkpoint?"
//
// A banked PTMT thermal state (thermal/thermal_state_io.h) is 12 bytes of
// header and a raw float32 block. ptm_eq_certify SCORES one; nothing printed
// what it IS. Every identity guard the format may grow later, and every
// restore refusal it already has, is undebuggable without a way to look at the
// file -- so this tool exists before those guards do.
//
// WHY C++ AND NOT PYTHON. The format is trivially readable from Python, and
// that is exactly the problem: a Python reader would be a SECOND
// implementation of a format whose own header comment records that being
// duplicated three times was the original defect (BUGS.md PTM-76). This tool
// calls state_io::probe() and state_io::read() -- the engine's own code -- so
// it cannot drift from what a restore will actually accept. The optional
// --body grid reconstruction has the same argument in a stronger form: it
// builds thermal::Profile, so the depth grid it reports is the one the solver
// would build, not a re-derivation of the formula.
//
// WHAT IT CANNOT TELL YOU, and says so out loud. The v1 format records the
// SHAPE and nothing else: no depth grid, no epoch, no body, no thermophysics,
// no mesh identity, no coupling state. Those are BUGS.md PTM-STATE-GRID-BLIND
// and PTM-STATE-NO-EPOCH. A PTMT v2 file closes them by carrying an identity
// trailer, which this tool prints in full and whose CRC32 it verifies over the
// whole file. For a v1 file the questions are still unanswered, and this tool
// prints them as unanswered rather than omitting them, because a report that is
// silent about what it did not check reads exactly like one that checked.
//
// No mesh, no view factors, no SPICE, no GPU: safe to run while a production
// job owns the devices.

#include "../thermal/thermal_state_io.h"
#include "../thermal/thermal_model.h"
#include "../thermal/eq_certificate.h"
#include "../thermal/conductivity_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

const char* kUsage =
"NAME\n"
"    ptm_state_info -- describe and validate a PTM3D thermal-state checkpoint\n"
"\n"
"SYNOPSIS\n"
"    ptm_state_info [options] <state.ptmt> [state2.ptmt ...]\n"
"\n"
"DESCRIPTION\n"
"    Prints the header, dimensions and structural validity of one or more\n"
"    banked PTMT thermal states, plus per-layer temperature statistics and the\n"
"    non-finite / unphysical counts that tell an all-zero cold-start column\n"
"    (BUGS.md PTM-78) apart from a real integrated one.\n"
"\n"
"    A PTMT v2 file carries an IDENTITY TRAILER -- the depth grid, the epoch,\n"
"    the body, the grid parameters, a mesh geometry hash, a physics hash and\n"
"    the coupling flags -- and a CRC32 over the whole file. This tool prints\n"
"    the trailer, verifies the checksum, and (given --body) cross-checks the\n"
"    depth grid the file RECORDS against the grid those settings would BUILD.\n"
"\n"
"    The v1 format records n_faces and n_layers AND NOTHING ELSE. It carries no\n"
"    depth grid, no epoch, no body and no thermophysical properties, so on a v1\n"
"    file this tool cannot verify that a checkpoint belongs to the run you mean\n"
"    to restore it into. It names each of those unanswered questions explicitly\n"
"    rather than leaving them out of the report.\n"
"\n"
"    Given a body and grid parameters (--body/--m/--n/--b and the property\n"
"    overrides), it additionally reconstructs the depth grid the ENGINE would\n"
"    build for those settings and reports two numbers that no other tool\n"
"    prints: the grid skin depth delta_0 = z_max/b, and Qb_from_state -- the\n"
"    geothermal flux read back out of the state's own bottom boundary\n"
"    condition. Qb_from_state is MEASURED from the column, not configured, so\n"
"    it is the way to tell whether a state was integrated with the Qb its\n"
"    config claims (BUGS.md PTM-161: `geothermal_heat_flow` was inert on every\n"
"    solver entry point for a whole campaign).\n"
"\n"
"OPTIONS\n"
"    --per-layer            Print the full per-layer statistics table.\n"
"    --json                 Emit one JSON object per file instead of text.\n"
"    -h, --help             This message.\n"
"\n"
"  Grid reconstruction (all optional; supply --body to enable it)\n"
"    --body <name>          Moon, Europa, Mercury, ... (enables the grid block)\n"
"    --m <int> --n <int> --b <int>\n"
"                           Depth-grid parameters. Defaults m=10 n=15 b=15,\n"
"                           which is thermal::Config's own default b.\n"
"    --adaptive <0|1>       Override Config::ADAPTIVE_GRID.\n"
"    --ks --kd --rhos --rhod --cp0 --H --day --chi --eps\n"
"                           Property overrides, same meaning as ptm_eq_certify.\n"
"    --cp-coeff a,b,...     Heat-capacity polynomial coefficients.\n"
"\n"
"EXIT STATUS\n"
"    0   every file named is a structurally valid PTMT checkpoint\n"
"    1   at least one file is missing, malformed, truncated, carries a trailer\n"
"        that fails its own checksum or layout, or (with a grid) disagrees with\n"
"        the reconstructed layer count or depth grid\n"
"    2   usage error\n"
"\n"
"EXAMPLES\n"
"    ptm_state_info out/_thermal_state/thermal_state_after_chunk_000.bin\n"
"    ptm_state_info --per-layer --body Moon --b 15 state.ptmt\n"
"    ptm_state_info --json out/_thermal_state/*.bin\n"
"\n"
"SEE ALSO\n"
"    ptm_eq_certify(1) -- scores a banked state for equilibration.\n"
"    src/cpp/thermal/thermal_state_io.h -- the format, and its known gaps.\n";

[[noreturn]] void usageError(const std::string& why) {
    std::fprintf(stderr, "ptm_state_info: %s\n\nRun with --help.\n", why.c_str());
    std::exit(2);
}

double parseNumber(const std::string& key, const std::string& val) {
    const char* b = val.c_str();
    char* e = nullptr;
    const double v = std::strtod(b, &e);
    if (e == b || *e != '\0')
        usageError("--" + key + " expects one number, got '" + val + "'");
    return v;
}

long parseInt(const std::string& key, const std::string& val) {
    const char* b = val.c_str();
    char* e = nullptr;
    const long v = std::strtol(b, &e, 10);
    if (e == b || *e != '\0')
        usageError("--" + key + " expects one integer, got '" + val + "'");
    return v;
}

struct Options {
    std::vector<std::string> files;
    bool per_layer = false;
    bool json = false;
    // Grid reconstruction (inactive until --body is given).
    bool   want_grid = false;
    std::string body = "Moon";
    int    m = 10, n = 15, b = 15;
    int    adaptive = -1;
    double ks = -1, kd = -1, rhos = -1, rhod = -1, cp0 = -1e300, H = -1;
    double day = -1, eps = -1, chi = -1e300;
    bool   have_cp_coeff = false;
    std::vector<double> cp_coeff;
};

std::vector<double> parseCsv(const std::string& s) {
    std::vector<double> out;
    std::size_t i = 0;
    while (i <= s.size()) {
        const std::size_t j = s.find(',', i);
        const std::string tok =
            s.substr(i, (j == std::string::npos ? s.size() : j) - i);
        if (!tok.empty()) out.push_back(parseNumber("cp-coeff", tok));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* key) -> std::string {
            if (i + 1 >= argc) usageError(std::string(key) + " needs a value");
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { std::fputs(kUsage, stdout); std::exit(0); }
        else if (a == "--per-layer")  o.per_layer = true;
        else if (a == "--json")       o.json = true;
        else if (a == "--body")   { o.body = value("--body"); o.want_grid = true; }
        else if (a == "--m")      o.m = static_cast<int>(parseInt("m", value("--m")));
        else if (a == "--n")      o.n = static_cast<int>(parseInt("n", value("--n")));
        else if (a == "--b")      o.b = static_cast<int>(parseInt("b", value("--b")));
        else if (a == "--adaptive")
            o.adaptive = static_cast<int>(parseInt("adaptive", value("--adaptive")));
        else if (a == "--ks")     o.ks   = parseNumber("ks",   value("--ks"));
        else if (a == "--kd")     o.kd   = parseNumber("kd",   value("--kd"));
        else if (a == "--rhos")   o.rhos = parseNumber("rhos", value("--rhos"));
        else if (a == "--rhod")   o.rhod = parseNumber("rhod", value("--rhod"));
        else if (a == "--cp0")    o.cp0  = parseNumber("cp0",  value("--cp0"));
        else if (a == "--H")      o.H    = parseNumber("H",    value("--H"));
        else if (a == "--day")    o.day  = parseNumber("day",  value("--day"));
        else if (a == "--eps")    o.eps  = parseNumber("eps",  value("--eps"));
        else if (a == "--chi")    o.chi  = parseNumber("chi",  value("--chi"));
        else if (a == "--cp-coeff") {
            o.cp_coeff = parseCsv(value("--cp-coeff"));
            o.have_cp_coeff = true;
        }
        else if (!a.empty() && a[0] == '-') usageError("unknown option '" + a + "'");
        else o.files.push_back(a);
    }
    if (o.files.empty())
        usageError("no checkpoint named. This tool answers 'what is this file'; "
                   "it has nothing to do without one.");
    if (o.b <= 0) usageError("--b must be positive (z_max = b * delta_0)");
    return o;
}

// ---------------------------------------------------------------------------
// Statistics over a layer-major block. Deliberately reports the counts that
// distinguish a real integrated column from a cold-start artifact:
//   n_nonfinite  -- a NaN in a checkpoint poisons every face it restores into
//   n_below_cmb  -- below 2.7 K is unphysical (thermal_model.h ABSOLUTE_ZERO_K)
//   n_exact_zero -- the PTM-78 signature: an all-zero column banked because the
//                   single-phase branch never filled the post-eq hand-off
// ---------------------------------------------------------------------------
struct Stats {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double mean = 0.0;
    std::uint64_t n = 0, n_nonfinite = 0, n_below_cmb = 0, n_exact_zero = 0;
};

Stats statsOf(const float* v, std::size_t n) {
    Stats s;
    long double sum = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(v[i]);
        if (!std::isfinite(x)) { ++s.n_nonfinite; continue; }
        if (x == 0.0f) ++s.n_exact_zero;
        if (x < thermal::ABSOLUTE_ZERO_K) ++s.n_below_cmb;
        if (x < s.min) s.min = x;
        if (x > s.max) s.max = x;
        sum += x;
        ++s.n;
    }
    s.mean = (s.n > 0) ? static_cast<double>(sum / static_cast<long double>(s.n))
                       : 0.0;
    if (s.n == 0) { s.min = 0.0; s.max = 0.0; }
    return s;
}

double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const std::size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

// ---------------------------------------------------------------------------
// The reconstructed grid, when --body was given. Everything here is the
// ENGINE's own code -- thermal::Profile for z/kc/rho, conductivityK() for the
// k(T) law, eqGridSkinDepth()/eqQbFromState() for the two derived numbers --
// so nothing in this tool re-derives a formula the solver owns.
// ---------------------------------------------------------------------------
struct Grid {
    bool built = false;
    int n_layers = 0;
    double z_max = 0.0, delta0 = 0.0, Qb_compiled = 0.0, R350 = 0.0;
    thermal::ConductivityModel cond = thermal::ConductivityModel::Hayne2017;
    std::vector<double> z, kc, rho;
};

Grid buildGrid(const Options& o) {
    Grid g;
    if (!o.want_grid) return g;
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
    if (o.eps  > 0)      planet.emissivity = o.eps;
    if (o.chi  > -1e299) config.chi  = o.chi;
    if (o.have_cp_coeff) planet.cpCoeff = o.cp_coeff;

    thermal::Profile ref(planet, 0.0, config, /*verbose=*/false);
    g.built    = true;
    g.n_layers = ref.nlayers;
    g.z.assign(ref.z().begin(), ref.z().begin() + g.n_layers);
    g.kc.assign(ref.kc().begin(), ref.kc().begin() + g.n_layers);
    g.rho.assign(ref.rho().begin(), ref.rho().begin() + g.n_layers);
    g.z_max  = g.z[g.n_layers - 1];
    g.delta0 = thermal::eqGridSkinDepth(g.z.data(), g.n_layers, config.b);
    g.Qb_compiled = planet.Qb;
    g.R350 = config.getR350();
    g.cond = config.conductivity_model;
    return g;
}

}  // namespace

int main(int argc, char** argv) {
    const Options o = parseArgs(argc, argv);
    const Grid grid = buildGrid(o);

    int bad = 0;
    for (std::size_t fi = 0; fi < o.files.size(); ++fi) {
        const std::string& path = o.files[fi];

        thermal::state_io::FileInfo info;
        std::string err;
        if (!thermal::state_io::probe(path, info, err)) {
            ++bad;
            if (o.json)
                std::printf("{\"path\":\"%s\",\"valid\":false,\"error\":\"%s\"}\n",
                            jsonEscape(path).c_str(), jsonEscape(err).c_str());
            else
                std::fprintf(stderr, "REFUSED  %s\n         %s\n",
                             path.c_str(), err.c_str());
            continue;
        }

        // The identity trailer, if the file has one. readTrailer() streams the
        // whole file to verify its CRC32, so a length-correct but zero-filled
        // cloud-sync placeholder -- which probe() and the magic check both pass
        // -- is caught here and nowhere else.
        thermal::state_io::Trailer tr;
        std::string tr_err;
        const auto tr_status = thermal::state_io::readTrailer(path, tr, tr_err);
        const bool have_tr =
            (tr_status == thermal::state_io::TrailerStatus::Ok);
        if (tr_status == thermal::state_io::TrailerStatus::Malformed) ++bad;

        std::vector<float> T;
        if (!thermal::state_io::read(path, info.n_faces, info.n_layers, T, err)) {
            ++bad;
            if (o.json)
                std::printf("{\"path\":\"%s\",\"valid\":false,\"error\":\"%s\"}\n",
                            jsonEscape(path).c_str(), jsonEscape(err).c_str());
            else
                std::fprintf(stderr, "REFUSED  %s\n         %s\n",
                             path.c_str(), err.c_str());
            continue;
        }

        const std::size_t NF = info.n_faces, NL = info.n_layers;
        const Stats all  = statsOf(T.data(), T.size());
        const Stats surf = statsOf(T.data(), NF);                    // layer 0
        const Stats bot  = statsOf(T.data() + (NL - 1) * NF, NF);    // layer NL-1

        // The grid, if one was reconstructed, must agree on the layer count.
        // This is the ONLY shape check the format makes available, and it is
        // the same one ptm_eq_certify refuses on.
        bool grid_ok = grid.built &&
                       (static_cast<std::size_t>(grid.n_layers) == NL);
        double qb_min = 0, qb_med = 0, qb_max = 0;
        std::uint64_t qb_degenerate = 0;
        if (grid_ok) {
            std::vector<double> qb;
            qb.reserve(NF);
            std::vector<double> col(NL), k(NL);
            for (std::size_t f = 0; f < NF; ++f) {
                bool finite = true;
                for (std::size_t l = 0; l < NL; ++l) {
                    col[l] = static_cast<double>(T[l * NF + f]);
                    if (!std::isfinite(col[l])) { finite = false; break; }
                    k[l] = thermal::conductivityK(grid.cond, col[l], grid.kc[l],
                                                  grid.rho[l], grid.R350);
                }
                const double q = finite
                    ? thermal::eqQbFromState(grid.z.data(), col.data(), k.data(),
                                             static_cast<int>(NL))
                    : -1.0;
                if (std::isfinite(q)) qb.push_back(q); else ++qb_degenerate;
            }
            if (!qb.empty()) {
                qb_min = *std::min_element(qb.begin(), qb.end());
                qb_max = *std::max_element(qb.begin(), qb.end());
                qb_med = median(qb);
            }
        }

        // The depth-grid cross-check the trailer makes possible: what the file
        // RECORDS against what --body/--m/--n/--b would BUILD. Computed HERE,
        // above the output branch, so it reaches the exit status in --json mode
        // too -- a script asking this question is exactly the caller who uses
        // --json, and a check that only runs in the human report is a check a
        // script cannot have.
        // The layer-count mismatch counted toward the exit status ONLY in the
        // text branch, so `--json --body Moon --b 32` on a 23-layer file exited
        // 0 while the same command without --json exited 1. That predates this
        // change (the merge-base binary does it too) and is fixed here rather
        // than left as a promise the usage text now makes and the tool does not
        // keep. --json is the form a script uses; a check a script cannot have
        // is not a check.
        if (grid.built && !grid_ok) ++bad;

        bool     tr_grid_checked  = false;
        bool     tr_grid_disagree = false;
        std::size_t tr_grid_first_bad = 0;
        double   tr_grid_worst_rel = 0.0;
        if (have_tr && grid.built && !tr.z.empty() &&
            tr.z.size() == static_cast<std::size_t>(grid.n_layers)) {
            tr_grid_checked = true;
            tr_grid_first_bad = tr.z.size();
            for (std::size_t l = 0; l < tr.z.size(); ++l) {
                const double d = std::fabs(tr.z[l] - grid.z[l]);
                const double s = std::fabs(grid.z[l]) > 0 ? std::fabs(grid.z[l]) : 1.0;
                if (d / s > tr_grid_worst_rel) tr_grid_worst_rel = d / s;
                if (tr_grid_first_bad == tr.z.size() && d > 1e-12 * s)
                    tr_grid_first_bad = l;
            }
            tr_grid_disagree = (tr_grid_first_bad < tr.z.size());
            if (tr_grid_disagree) ++bad;
        }

        if (o.json) {
            std::printf("{\"path\":\"%s\",\"valid\":true,\"format\":\"%s\","
                        "\"magic\":\"0x%08X\",\"n_faces\":%u,\"n_layers\":%u,"
                        "\"values\":%llu,\"bytes_expected\":%llu,"
                        "\"bytes_actual\":%llu,\"bytes_extra\":%llu,"
                        "\"T_min\":%.9g,\"T_max\":%.9g,\"T_mean\":%.9g,"
                        "\"n_nonfinite\":%llu,\"n_below_cmb\":%llu,"
                        "\"n_exact_zero\":%llu,"
                        "\"surface_min\":%.9g,\"surface_max\":%.9g,"
                        "\"surface_mean\":%.9g,\"bottom_min\":%.9g,"
                        "\"bottom_max\":%.9g,\"bottom_mean\":%.9g",
                        jsonEscape(path).c_str(),
                        have_tr ? "PTMT v2"
                                : (tr_status == thermal::state_io::TrailerStatus::Absent
                                       ? "PTMT v1" : "PTMT v1 + UNREADABLE TRAILER"),
                        info.magic,
                        info.n_faces, info.n_layers,
                        (unsigned long long)info.values,
                        (unsigned long long)info.bytes_expected,
                        (unsigned long long)info.bytes_actual,
                        (unsigned long long)info.bytes_extra,
                        all.min, all.max, all.mean,
                        (unsigned long long)all.n_nonfinite,
                        (unsigned long long)all.n_below_cmb,
                        (unsigned long long)all.n_exact_zero,
                        surf.min, surf.max, surf.mean,
                        bot.min, bot.max, bot.mean);
            if (grid.built)
                std::printf(",\"grid_layer_count_matches\":%s",
                            grid_ok ? "true" : "false");
            if (grid_ok)
                std::printf(",\"grid_body\":\"%s\",\"grid_b\":%d,"
                            "\"grid_z_max_m\":%.9g,\"grid_delta0_m\":%.9g,"
                            "\"Qb_compiled\":%.9g,\"Qb_from_state_min\":%.9g,"
                            "\"Qb_from_state_median\":%.9g,"
                            "\"Qb_from_state_max\":%.9g,"
                            "\"Qb_from_state_degenerate\":%llu",
                            jsonEscape(o.body).c_str(), o.b, grid.z_max,
                            grid.delta0, grid.Qb_compiled,
                            qb_min, qb_med, qb_max,
                            (unsigned long long)qb_degenerate);
            if (have_tr) {
                // EVERY float64 trailer field may be NaN -- that is the
                // format's documented "not recorded" -- and `nan` is not valid
                // JSON. So every one of them goes through jnum(), which emits
                // `null` instead. Guarding only some of them (which an earlier
                // draft of this function did) produces output that parses on
                // the files you happened to test and fails on the rest.
                auto jnum = [](const char* key, double v) {
                    if (std::isfinite(v)) std::printf(",\"%s\":%.17g", key, v);
                    else                  std::printf(",\"%s\":null", key);
                };
                std::printf(",\"trailer\":true,\"trailer_version\":%u,"
                            "\"trailer_bytes\":%u,\"crc32\":\"0x%08X\","
                            "\"depth_grid_in_file\":%s,"
                            "\"file_grid_m\":%d,\"file_grid_n\":%d,"
                            "\"file_grid_b\":%d,"
                            "\"config_identity_hash\":\"0x%016llX\","
                            "\"body_in_file\":\"%s\","
                            "\"engine_git_sha\":\"%s\",\"flags\":%u,"
                            "\"post_coupling_complete\":%s,"
                            "\"all_faces_patched\":%s,"
                            "\"coupling_select_dt_positive\":%s,"
                            "\"batch_completion_partial\":%s",
                            tr.version_read, tr.bytes_read, tr.crc32_stored,
                            tr.z.empty() ? "false" : "true",
                            tr.grid_m, tr.grid_n, tr.grid_b,
                            (unsigned long long)tr.config_identity_hash,
                            jsonEscape(tr.body_name).c_str(),
                            jsonEscape(tr.engine_git_sha).c_str(), tr.flags,
                            (tr.flags & thermal::state_io::flag::kPostCouplingComplete) ? "true" : "false",
                            (tr.flags & thermal::state_io::flag::kAllFacesPatched) ? "true" : "false",
                            (tr.flags & thermal::state_io::flag::kCouplingSelectDtPositive) ? "true" : "false",
                            (tr.flags & thermal::state_io::flag::kBatchCompletionPartial) ? "true" : "false");
                jnum("file_z_min",         tr.z.empty() ? NAN : tr.z.front());
                jnum("file_z_max",         tr.z.empty() ? NAN : tr.z.back());
                jnum("file_refgrid_H",     tr.refgrid_H);
                jnum("file_mesh_geomhash", tr.mesh_geomhash);
                jnum("epoch_et_s",         tr.et_seconds);
                jnum("output_cadence_s",   tr.output_cadence_s);
                jnum("w0_mean",            tr.w0_mean);
                jnum("w0_p05",             tr.w0_p05);
                jnum("w0_p95",             tr.w0_p95);
                if (tr_grid_checked)
                    std::printf(",\"grid_matches_reconstruction\":%s,"
                                "\"grid_worst_rel_diff\":%.17g",
                                tr_grid_disagree ? "false" : "true",
                                tr_grid_worst_rel);
                else
                    std::printf(",\"grid_matches_reconstruction\":null,"
                                "\"grid_worst_rel_diff\":null");
            } else {
                std::printf(",\"trailer\":false,\"epoch_et_s\":null,"
                            "\"depth_grid_in_file\":false,"
                            "\"body_in_file\":null,\"mesh_id_in_file\":null,"
                            "\"config_identity_hash\":null,"
                            "\"post_coupling_flag_in_file\":null");
                if (tr_status == thermal::state_io::TrailerStatus::Malformed)
                    std::printf(",\"trailer_error\":\"%s\"",
                                jsonEscape(tr_err).c_str());
            }
            std::printf("}\n");
            continue;
        }

        if (fi) std::printf("\n");
        std::printf("=== %s ===\n", path.c_str());
        // Print the magic MOST-SIGNIFICANT-BYTE FIRST, which is the reading
        // that spells the format's name. The four bytes on disk are the
        // reverse ("TMTP") because the file is little-endian, and printing
        // THAT is how this project once invented a nonexistent "MFFV" format
        // out of VFFM's magic. Say both, once, so nobody has to guess.
        std::printf("format        %-8s magic 0x%08X = '%c%c%c%c' "
                    "(on disk, little-endian: '%c%c%c%c')\n",
                    have_tr ? "PTMT v2" : "PTMT v1",
                    info.magic,
                    static_cast<char>((info.magic >> 24) & 0xFF),
                    static_cast<char>((info.magic >> 16) & 0xFF),
                    static_cast<char>((info.magic >> 8) & 0xFF),
                    static_cast<char>(info.magic & 0xFF),
                    static_cast<char>(info.magic & 0xFF),
                    static_cast<char>((info.magic >> 8) & 0xFF),
                    static_cast<char>((info.magic >> 16) & 0xFF),
                    static_cast<char>((info.magic >> 24) & 0xFF));
        std::printf("shape         %u faces x %u layers = %llu float32 values\n",
                    info.n_faces, info.n_layers,
                    (unsigned long long)info.values);
        std::printf("layout        LAYER-MAJOR, T[l * n_faces + f]\n");
        std::printf("length        %llu bytes; header rule 12 + F*L*4 = %llu",
                    (unsigned long long)info.bytes_actual,
                    (unsigned long long)info.bytes_expected);
        if (info.bytes_extra == 0) std::printf("  -- EXACT (v1 body, no trailer)\n");
        else if (have_tr)
            std::printf("  + %llu-byte v2 identity trailer\n",
                        (unsigned long long)info.bytes_extra);
        else
            std::printf("  -- %llu TRAILING BYTES that are NOT a readable\n"
                        "              PTMT v2 trailer\n",
                        (unsigned long long)info.bytes_extra);

        if (tr_status == thermal::state_io::TrailerStatus::Malformed)
            std::printf("\nTRAILER REFUSED  %s\n"
                        "  A restore would refuse this file. Its temperature "
                        "block is still\n"
                        "  reported below, because reading it is how you find "
                        "out what was lost.\n",
                        tr_err.c_str());

        std::printf("\ntemperature   min %.4f  mean %.4f  max %.4f K\n",
                    all.min, all.mean, all.max);
        std::printf("  layer 0     min %.4f  mean %.4f  max %.4f K\n",
                    surf.min, surf.mean, surf.max);
        std::printf("  layer %-5u min %.4f  mean %.4f  max %.4f K  (deepest)\n",
                    info.n_layers - 1, bot.min, bot.mean, bot.max);
        std::printf("  non-finite  %llu    below the %.1f K CMB floor  %llu"
                    "    exactly 0 K  %llu\n",
                    (unsigned long long)all.n_nonfinite,
                    thermal::ABSOLUTE_ZERO_K,
                    (unsigned long long)all.n_below_cmb,
                    (unsigned long long)all.n_exact_zero);
        if (all.n_exact_zero > 0)
            std::printf("  NOTE: exact zeros are the BUGS.md PTM-78 signature -- a\n"
                        "        checkpoint banked from a column that was never filled.\n");
        if (all.n_nonfinite > 0)
            std::printf("  NOTE: a non-finite value poisons every face it restores into.\n");

        if (o.per_layer) {
            std::printf("\n  layer %s      min        mean         max\n",
                        grid_ok ? "      z (m)  " : "");
            for (std::size_t l = 0; l < NL; ++l) {
                const Stats s = statsOf(T.data() + l * NF, NF);
                std::printf("  %5zu ", l);
                if (grid_ok) std::printf("%11.6f  ", grid.z[l]);
                std::printf("%10.4f  %10.4f  %10.4f\n", s.min, s.mean, s.max);
            }
        }

        // ---- the reconstructed grid, and what it lets us measure ----------
        if (grid.built && !grid_ok) {
            // `++bad` already counted above the output branch, so --json gets it.
            std::printf("\nGRID MISMATCH  body=%s m=%d n=%d b=%d builds %d layers; "
                        "this file has %u.\n"
                        "               The format does NOT record its own depth "
                        "grid, so the layer\n"
                        "               count is the only cross-check available -- "
                        "and it just failed.\n"
                        "               Fix --body/--m/--n/--b or the property "
                        "overrides.\n",
                        o.body.c_str(), o.m, o.n, o.b, grid.n_layers,
                        info.n_layers);
        } else if (grid_ok) {
            std::printf("\ngrid (RECONSTRUCTED from --body %s, m=%d n=%d b=%d --"
                        " NOT read from the file)\n",
                        o.body.c_str(), o.m, o.n, o.b);
            std::printf("  z_max       %.6f m       delta_0 = z_max/b = %.6f m\n",
                        grid.z_max, grid.delta0);
            std::printf("  %s\n",
                        thermal::state_io::gridFingerprint(grid.z.data(), NL).c_str());
            std::printf("  Qb_compiled       %.6g W/m^2   (this body's built-in value)\n",
                        grid.Qb_compiled);
            std::printf("  Qb_from_state     min %.6g  median %.6g  max %.6g W/m^2\n",
                        qb_min, qb_med, qb_max);
            if (qb_degenerate)
                std::printf("                    %llu face(s) gave no usable value\n",
                            (unsigned long long)qb_degenerate);
            std::printf("  Qb_from_state is MEASURED out of the state's own bottom BC\n"
                        "  (T[n-1] = T[n-2] + Qb*dz/k[n-1]). If it disagrees with the Qb\n"
                        "  the config claims, the config is not what was integrated --\n"
                        "  that is BUGS.md PTM-161, which was inert for a whole campaign.\n");
        }

        // ---- the identity trailer, when the file carries one ---------------
        if (have_tr) {
            std::printf("\nIDENTITY TRAILER (READ FROM THE FILE -- not reconstructed)\n");
            std::printf("  trailer       version %u, %u bytes, CRC32 0x%08X VERIFIED\n"
                        "                over the whole file, so a length-correct "
                        "zero-filled\n"
                        "                cloud placeholder cannot pass as this "
                        "checkpoint.\n",
                        tr.version_read, tr.bytes_read, tr.crc32_stored);
            if (std::isfinite(tr.et_seconds))
                std::printf("  epoch         ET %.6f s  (the epoch this column is "
                            "valid at)\n", tr.et_seconds);
            else
                std::printf("  epoch         not recorded\n");
            if (std::isfinite(tr.output_cadence_s))
                std::printf("  cadence       %.6f s  (the writing run's output "
                            "cadence; one\n"
                            "                cadence is the widest epoch gap a "
                            "restore treats as a\n"
                            "                stale hand-off rather than a "
                            "different window)\n",
                            tr.output_cadence_s);
            std::printf("  body          %s\n",
                        tr.body_name.empty() ? "(not recorded)"
                                             : tr.body_name.c_str());
            // refgrid_H < 0 is the engine's "no per-face H map was loaded, so
            // the grid was built on the body's own H" sentinel, not a depth.
            // Printing it as "-1 m" reads like a measurement.
            if (std::isfinite(tr.refgrid_H) && tr.refgrid_H > 0.0)
                std::printf("  grid          m=%d n=%d b=%d   refgrid_H %.9g m "
                            "(median H of a loaded per-face map)\n",
                            tr.grid_m, tr.grid_n, tr.grid_b, tr.refgrid_H);
            else
                std::printf("  grid          m=%d n=%d b=%d   refgrid_H unset "
                            "(no per-face H map; the body's own H set the grid)\n",
                            tr.grid_m, tr.grid_n, tr.grid_b);
            if (!tr.z.empty())
                std::printf("  depth grid    %s\n",
                            thermal::state_io::gridFingerprint(
                                tr.z.data(), tr.z.size()).c_str());
            std::printf("  mesh geomhash %.17g\n", tr.mesh_geomhash);
            std::printf("  physics hash  0x%016llX  (resolved thermophysical "
                        "scalars + grid\n"
                        "                parameters ONLY -- NOT per-face albedo/"
                        "TI/H maps, NOT a\n"
                        "                per-face Qb mask, NOT view factors. "
                        "See\n"
                        "                src/cpp/thermal/state_identity.h.)\n",
                        (unsigned long long)tr.config_identity_hash);
            std::printf("  engine        %s\n",
                        tr.engine_git_sha.empty() ? "(not recorded)"
                                                  : tr.engine_git_sha.c_str());
            std::printf("  flags         0x%08X  post_coupling=%s "
                        "all_faces_patched=%s\n"
                        "                coupling_select_dt>0=%s "
                        "batch_partial=%s\n",
                        tr.flags,
                        (tr.flags & thermal::state_io::flag::kPostCouplingComplete) ? "YES" : "no",
                        (tr.flags & thermal::state_io::flag::kAllFacesPatched) ? "YES" : "no",
                        (tr.flags & thermal::state_io::flag::kCouplingSelectDtPositive) ? "YES" : "no",
                        (tr.flags & thermal::state_io::flag::kBatchCompletionPartial) ? "YES" : "no");
            if (tr.flags & thermal::state_io::flag::kBatchCompletionPartial)
                std::printf("  NOTE: batch_partial is set. A restore REFUSES "
                            "this file; it covers\n"
                            "        only the faces one device had finished.\n");
            std::printf("  W(0) swing    NOT RECORDED by this engine version, on "
                        "purpose. It is\n"
                        "                an input to the warm-start "
                        "admissibility rule, and a field\n"
                        "                that is a measurement on some solver "
                        "paths and a\n"
                        "                coincidence on others is worse than "
                        "an absent one.\n");

            // The one cross-check this tool can make that nothing else does:
            // the grid the file RECORDS against the grid the given --body/--m/
            // --n/--b would BUILD. A disagreement here is PTM-STATE-GRID-BLIND
            // caught in the act. Computed above the output branch so --json
            // gets it too; only the rendering lives here.
            if (tr_grid_checked && tr_grid_disagree) {
                std::printf("\n  GRID DISAGREEMENT: the file's own z[] and the "
                            "grid --body %s\n"
                            "  m=%d n=%d b=%d builds differ from node %zu "
                            "(%.17g vs %.17g m,\n"
                            "  worst relative %.3g). A restore under these "
                            "settings would REFUSE.\n",
                            o.body.c_str(), o.m, o.n, o.b, tr_grid_first_bad,
                            tr.z[tr_grid_first_bad], grid.z[tr_grid_first_bad],
                            tr_grid_worst_rel);
            } else if (tr_grid_checked) {
                std::printf("\n  the file's own z[] MATCHES the grid --body %s "
                            "m=%d n=%d b=%d builds\n"
                            "  (worst relative difference %.3g).\n",
                            o.body.c_str(), o.m, o.n, o.b, tr_grid_worst_rel);
            }
        }

        // ---- what this file CANNOT answer ---------------------------------
        if (!have_tr) {
            std::printf("\nNOT RECORDED BY THE v1 FORMAT (each one is an open ledger row):\n"
                        "  depth grid z[]      no  -- a restore onto a different grid with the\n"
                        "                             same layer count is accepted in silence\n"
                        "                             (PTM-STATE-GRID-BLIND)\n"
                        "  epoch / time        no  -- a wrong-epoch restore is undetectable\n"
                        "                             (PTM-STATE-NO-EPOCH)\n"
                        "  body, properties    no      mesh identity      no\n"
                        "  post-coupling flag  no  -- whether this state is pre- or\n"
                        "                             post-radiation-coupling (PTM-57)\n"
                        "  Re-bank this state on the current engine to get a v2 identity\n"
                        "  trailer, which records all of the above.\n");
        } else {
            std::printf("\nSTILL NOT RECORDED, even by v2:\n"
                        "  per-face maps       no  -- albedo / thermal-inertia / H maps and a\n"
                        "                             per-face Qb mask change the physics face\n"
                        "                             by face and are outside the physics hash\n"
                        "  view factors        no  -- radiation coupling identity is not banked\n"
                        "  W(0) surface swing  no  -- see the note above\n");
        }
        if (!grid.built)
            std::printf("  (pass --body to reconstruct the grid the engine WOULD build and\n"
                        "   get delta_0 and Qb_from_state; it is still a reconstruction,\n"
                        "   not something this file records.)\n");
    }

    return (bad == 0) ? 0 : 1;
}
