// test_flux_cube_pass.cpp
// ---------------------------------------------------------------------------
// Unit test for the flux-cube post-solve pass, on SYNTHETIC geometry whose
// answers are closed forms. No mesh, no Embree, no SPICE, no illumination
// model -- the pass takes its ray casting and primary-body physics as
// callbacks, so the test supplies deterministic ones and knows the truth.
//
// What each arm can falsify:
//
//  A. SEALED NADIR (one partner face, F = 1, no indirect flux)
//       Q_terrain_ir  == eps*sigma*T^4          exactly
//       Q_terrain_vis == A_src * Q_dir_face     exactly
//     Falsifies the reconstruction wiring, the row dot, and the component
//     assignment. This is the "905 W/m^2 under a lander" anchor from the design.
//
//  B. INDIRECT TERM. Same cell, now with a non-zero absorbed indirect flux:
//       Q_terrain_ir == eps*sigma*T^4 + (1-eps)*(Q_ind_ir/eps)
//     Falsifies dropping or mis-weighting the reflected part -- the term a
//     naive implementation forgets, and the one that carries PSR heat.
//
//  C. SHADOW PROBE. A synthetic occluder blocks the four rim probes but not the
//     center. V_sun must be 1/5 and the penumbral flag must be SET. An
//     implementation that probes only the center reports V_sun = 1 and a clear
//     flag, and this arm fails it.
//
//  D. DIRECT-SOLAR ORIENTATION. With the Sun along +U, the zenith face must see
//     S*V_sun and the nadir face exactly 0. Falsifies a sign flip or a face
//     mis-assignment that arm A cannot see (arm A has no direct beam).
//
//  E. TIME-AXIS REFUSAL (negative control). A time axis whose length breaks
//     equilibration_steps + n_steps == total_time_steps MUST be refused. If it
//     is accepted, cube rows silently stop lining up with time_steps.txt, which
//     is the PTM-250/251 defect class.
//
//  F. CACHE-KEY REFUSAL (negative control). A cache built for a different mesh
//     MUST be refused, never rebuilt or applied anyway.
//
//  G. GRAY-PATH REFUSAL (negative control). Missing band-separated series MUST
//     be refused rather than emitting zeros or a merged column.
//
// Exit code = number of failures.
// ---------------------------------------------------------------------------

#include "flux_cube_format.h"
#include "flux_cube_pass.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ptm::fluxcubes;

namespace {

constexpr double SIGMA = 5.670374419e-8;

// Write a one-cube cache whose NADIR patch sees exactly one partner face with
// F = 1 (a sealed cell) and whose other five faces are pure sky.
std::string writeCache(const std::string& dir, int32_t partner_face,
                       double geomhash, uint64_t n_faces_core,
                       double height, double edge) {
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/synthetic.fcg";

    FcgHeader h{};
    std::memcpy(h.magic, "FCG1", 4);
    h.version = 1;
    h.n_cubes = 1;
    h.n_faces_per_cube = kNumFaces;
    h.total_nnz = 1;
    h.geomhash_core = geomhash;
    h.n_faces_core = n_faces_core;
    h.ring_hash = 0;
    h.center_lat_deg = 0.0; h.center_lon_deg = 0.0;
    h.spacing_m = 25.0; h.radius_m = 0.0;
    h.height_m = height; h.edge_m = edge;
    h.n_rays = 16384; h.sampler_id = kSamplerShirleyChiuCP; h.seed = 0;
    h.truncate_rel = 0.0;
    h.height_factor = 0.1; h.edge_factor = 0.1; h.local_median_edge_m = height / 0.1;
    // Site frame: U = +x, N = +z, E = N x U = +y (right-handed, E x N = U).
    h.frame_origin[0] = 1.0e6; h.frame_origin[1] = 0; h.frame_origin[2] = 0;
    h.frame_up[0]    = 1; h.frame_up[1]    = 0; h.frame_up[2]    = 0;
    h.frame_north[0] = 0; h.frame_north[1] = 0; h.frame_north[2] = 1;
    h.frame_east[0]  = 0; h.frame_east[1]  = 1; h.frame_east[2]  = 0;
    h.frame_max_tilt_deg = 0.0;
    const std::string json = "{\"schema\":1,\"tool\":\"test_flux_cube_pass\"}";
    h.json_len = static_cast<uint32_t>(json.size());

    CubeRecord cr{};
    cr.grid_i = 0; cr.grid_j = 0; cr.x_m = 0; cr.y_m = 0;
    cr.center[0] = 1.0e6 + height; cr.center[1] = 0; cr.center[2] = 0;
    cr.face_id = partner_face;
    cr.slope_deg = 0; cr.aspect_deg = 0; cr.area_m2 = 1.0f; cr.elev_m = 0;
    cr.height_achieved_m = height;

    std::vector<PatchRecord> pr(kNumFaces);
    for (int f = 0; f < kNumFaces; ++f) {
        pr[f] = PatchRecord{};
        pr[f].row_offset = 0;
        pr[f].row_len = 0;
        pr[f].f_sky = 1.0f;
    }
    // Nadir: one partner, F = 1, no sky.
    pr[kFaceNadir].row_offset = 0;
    pr[kFaceNadir].row_len = 1;
    pr[kFaceNadir].f_sky = 0.0f;
    pr[kFaceNadir].f_max = 1.0f;
    pr[kFaceNadir].n_faces_50pct = 1;

    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(&h), sizeof(h));
    o.write(reinterpret_cast<const char*>(&cr), sizeof(cr));
    o.write(reinterpret_cast<const char*>(pr.data()),
            static_cast<std::streamsize>(pr.size() * sizeof(PatchRecord)));
    const int32_t col = partner_face;
    const float val = 1.0f;
    o.write(reinterpret_cast<const char*>(&col), sizeof(col));
    o.write(reinterpret_cast<const char*>(&val), sizeof(val));
    o.write(json.data(), static_cast<std::streamsize>(json.size()));
    o.close();
    return path;
}

struct Fcb {
    FcbHeader h{};
    std::vector<double> flux;
    std::vector<double> t_face;
    std::vector<uint8_t> penumbral;
};

bool readFcb(const std::string& path, Fcb& f) {
    std::ifstream i(path, std::ios::binary);
    if (!i) return false;
    i.read(reinterpret_cast<char*>(&f.h), sizeof(f.h));
    if (std::memcmp(f.h.magic, "FCB1", 4) != 0) return false;
    const size_t n = static_cast<size_t>(f.h.n_cubes) * f.h.n_faces_per_cube *
                     f.h.n_components * f.h.n_steps;
    f.flux.resize(n);
    i.read(reinterpret_cast<char*>(f.flux.data()),
           static_cast<std::streamsize>(n * sizeof(double)));
    f.t_face.resize(static_cast<size_t>(f.h.n_cubes) * f.h.n_steps);
    i.read(reinterpret_cast<char*>(f.t_face.data()),
           static_cast<std::streamsize>(f.t_face.size() * sizeof(double)));
    f.penumbral.resize(static_cast<size_t>(f.h.n_cubes) * f.h.n_steps);
    i.read(reinterpret_cast<char*>(f.penumbral.data()),
           static_cast<std::streamsize>(f.penumbral.size()));
    return static_cast<bool>(i);
}

double at(const Fcb& f, int cube, int face, int comp, int step) {
    return f.flux[((static_cast<size_t>(cube) * f.h.n_faces_per_cube + face) *
                   f.h.n_components + comp) * f.h.n_steps + step];
}

// ---- the run fixture -------------------------------------------------------
struct Fixture {
    static constexpr size_t NF = 4;      // synthetic mesh faces
    static constexpr int EQ = 3;         // equilibration steps
    static constexpr int NS = 2;         // output steps
    static constexpr int TT = EQ + NS;
    int32_t partner = 2;
    double eps = 0.95, albedo = 0.20, T = 360.0;
    double q_dir_face = 0.0, q_ind_ir = 0.0, q_ind_vis = 0.0;
    double S = 1361.0;
    std::array<double, 3> sun{{0, 0, 0}};
    bool block_rim_only = false;
    bool block_all = false;

    std::vector<double> times{0.0, 21600.0};
    std::vector<std::string> tstr{"2028-08-03T22:21:00.000", "2028-08-04T04:21:00.000"};
    std::vector<double> direct;                      // flat [f*TT + t]
    std::vector<std::vector<double>> temps, isol, iir;
    std::vector<std::array<double, 3>> sunpos, pripos;

    void build() {
        direct.assign(NF * TT, 0.0);
        temps.assign(NF, std::vector<double>(TT, T));
        isol.assign(NF, std::vector<double>(TT, q_ind_vis));
        iir.assign(NF, std::vector<double>(TT, q_ind_ir));
        for (size_t f = 0; f < NF; ++f)
            for (int t = 0; t < TT; ++t) direct[f * TT + t] = q_dir_face;
        sunpos.assign(TT, sun);
        pripos.assign(TT, {{0, 0, 0}});
    }

    ptm::fluxcubes::PassInputs inputs(const std::string& cache, const std::string& out,
                                      double geomhash) {
        PassInputs in;
        in.cache_file = cache;
        in.geomhash_core = geomhash;
        in.n_faces_core = NF;
        in.times = &times;
        in.time_strings = &tstr;
        in.equilibration_steps = EQ;
        in.total_time_steps = TT;
        in.direct_solar = &direct;
        in.surface_temps = &temps;
        in.indirect_solar_series = &isol;
        in.indirect_ir_series = &iir;
        in.emissivity.assign(NF, eps);
        in.bond_albedo.assign(NF, albedo);
        in.sun_positions = &sunpos;
        in.primary_positions = &pripos;
        in.primary_radius_km = 0.0;
        const double Sv = S;
        in.solar_const = [Sv](int) { return Sv; };
        // No incidence-albedo function: A_src falls back to the Bond value, and
        // the pass records that in the header rather than assuming it silently.
        const bool rim = block_rim_only, all = block_all;
        // Origin x == the cube center x (1e6 + h) is the CENTER probe; the four
        // rim probes are displaced in y/z. Blocking exactly the displaced ones
        // is what makes arm C discriminate a 1-ray probe from a 5-ray one.
        in.occluded = [rim, all](const std::array<double, 3>& o,
                                 const std::array<double, 3>&) {
            if (all) return true;
            if (!rim) return false;
            return (std::fabs(o[1]) > 1e-12 || std::fabs(o[2]) > 1e-12);
        };
        in.output_dir = out;
        in.write_binary = true;
        in.write_csv = false;
        in.shadow_probe_rays = 5;
        return in;
    }
};

int check(const char* what, double got, double want, double tol, int& fails) {
    const double d = std::fabs(got - want);
    const bool ok = d <= tol;
    std::printf("  %-46s got %14.6f  want %14.6f  |d| %9.2e  %s\n",
                what, got, want, d, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
    return ok;
}

}  // namespace

int main() {
    std::printf("\n=== flux-cube post-solve pass: unit test on synthetic geometry ===\n");
    int fails = 0;
    const std::string base = "output/flux_cubes/unit";
    const double GH = 12345.678;

    // ---------------------------------------------------------------- ARM A
    {
        std::printf("\n--- ARM A: sealed nadir, no indirect flux ---\n");
        Fixture fx;
        fx.q_dir_face = 500.0;   // the FACE under the cube is sunlit
        fx.build();
        const std::string dir = base + "/armA";
        const std::string cache = writeCache(dir, fx.partner, GH, Fixture::NF, 0.108, 0.108);
        auto in = fx.inputs(cache, dir, GH);
        std::string err;
        if (!runFluxCubePass(in, err)) { std::printf("  FAIL: %s\n", err.c_str()); ++fails; }
        else {
            Fcb f;
            if (!readFcb(dir + "/flux_cubes/flux_cubes.fcb", f)) { std::printf("  FAIL: no .fcb\n"); ++fails; }
            else {
                const double want_ir = fx.eps * SIGMA * std::pow(fx.T, 4);
                const double want_vis = fx.albedo * fx.q_dir_face;   // A_src falls back to Bond
                check("nadir terrain_ir  = eps*sigma*T^4", at(f, 0, kFaceNadir, kCompTerrainIR, 0),
                      want_ir, 1e-9, fails);
                check("nadir terrain_vis = A*Q_dir_face", at(f, 0, kFaceNadir, kCompTerrainVis, 0),
                      want_vis, 1e-9, fails);
                check("zenith terrain_ir = 0 (pure sky)", at(f, 0, kFaceZenith, kCompTerrainIR, 0),
                      0.0, 1e-12, fails);
                check("T_face reported", f.t_face[0], fx.T, 1e-12, fails);
                std::printf("  (the design's lander-underside anchor: %.1f W/m^2 at T = %.1f K)\n",
                            want_ir, fx.T);
            }
        }
    }

    // ---------------------------------------------------------------- ARM B
    {
        std::printf("\n--- ARM B: the reflected-IR term (1-eps)*Q_ind_ir/eps ---\n");
        Fixture fx;
        fx.q_ind_ir = 40.0;
        fx.q_ind_vis = 7.0;
        fx.q_dir_face = 500.0;
        fx.build();
        const std::string dir = base + "/armB";
        const std::string cache = writeCache(dir, fx.partner, GH, Fixture::NF, 0.108, 0.108);
        auto in = fx.inputs(cache, dir, GH);
        std::string err;
        if (!runFluxCubePass(in, err)) { std::printf("  FAIL: %s\n", err.c_str()); ++fails; }
        else {
            Fcb f;
            readFcb(dir + "/flux_cubes/flux_cubes.fcb", f);
            const double want_ir = fx.eps * SIGMA * std::pow(fx.T, 4)
                                 + (1.0 - fx.eps) * (fx.q_ind_ir / fx.eps);
            const double want_vis = fx.albedo * fx.q_dir_face
                                  + fx.albedo * (fx.q_ind_vis / (1.0 - fx.albedo));
            const double naive_ir = fx.eps * SIGMA * std::pow(fx.T, 4);
            check("nadir terrain_ir  with reflected term", at(f, 0, kFaceNadir, kCompTerrainIR, 0),
                  want_ir, 1e-9, fails);
            check("nadir terrain_vis with scattered term", at(f, 0, kFaceNadir, kCompTerrainVis, 0),
                  want_vis, 1e-9, fails);
            std::printf("  (dropping the reflected term would give %.6f, off by %.4f W/m^2 -- "
                        "so this arm discriminates)\n", naive_ir, want_ir - naive_ir);
            if (std::fabs(want_ir - naive_ir) < 1e-6) {
                std::printf("  FAIL: arm B is not discriminating (the two spellings agree)\n");
                ++fails;
            }
        }
    }

    // ---------------------------------------------------------------- ARM C+D
    {
        std::printf("\n--- ARM C/D: 5-ray shadow probe, penumbral flag, and orientation ---\n");
        Fixture fx;
        fx.sun = {{1, 0, 0}};        // Sun straight up along +U
        fx.block_rim_only = true;    // only the four rim probes are occluded
        fx.build();
        const std::string dir = base + "/armC";
        const std::string cache = writeCache(dir, fx.partner, GH, Fixture::NF, 0.108, 0.108);
        auto in = fx.inputs(cache, dir, GH);
        std::string err;
        if (!runFluxCubePass(in, err)) { std::printf("  FAIL: %s\n", err.c_str()); ++fails; }
        else {
            Fcb f;
            readFcb(dir + "/flux_cubes/flux_cubes.fcb", f);
            // V_sun = 1 clear of 5 => zenith sees S/5; a center-only probe would
            // report the full S and no penumbral flag.
            check("zenith direct = V_sun*S, V_sun = 1/5", at(f, 0, kFaceZenith, kCompDirect, 0),
                  fx.S / 5.0, 1e-9, fails);
            check("nadir direct = 0 (Sun overhead)", at(f, 0, kFaceNadir, kCompDirect, 0),
                  0.0, 1e-12, fails);
            check("north direct = 0 (normal perp. to Sun)", at(f, 0, kFaceNorth, kCompDirect, 0),
                  0.0, 1e-12, fails);
            const bool pen = f.penumbral[0] != 0;
            std::printf("  %-46s got %14s  want %14s  %s\n", "penumbral flag SET",
                        pen ? "set" : "clear", "set", pen ? "PASS" : "FAIL");
            if (!pen) ++fails;
        }
    }

    // ------------------------------------------------------- NEGATIVE CONTROLS
    std::printf("\n--- NEGATIVE CONTROLS (each MUST be refused) ---\n");
    {
        Fixture fx; fx.build();
        const std::string dir = base + "/armE";
        const std::string cache = writeCache(dir, fx.partner, GH, Fixture::NF, 0.108, 0.108);
        auto in = fx.inputs(cache, dir, GH);
        in.total_time_steps = Fixture::TT + 1;    // breaks eq + n_steps == total
        std::string err;
        const bool ran = runFluxCubePass(in, err);
        std::printf("  %-46s %s\n", "E: inconsistent time axis",
                    ran ? "FAIL (accepted!)" : "PASS (refused)");
        if (ran) ++fails; else std::printf("      reason: %.140s\n", err.c_str());
    }
    {
        Fixture fx; fx.build();
        const std::string dir = base + "/armF";
        const std::string cache = writeCache(dir, fx.partner, GH, Fixture::NF, 0.108, 0.108);
        auto in = fx.inputs(cache, dir, GH * 2.0);   // wrong mesh identity
        std::string err;
        const bool ran = runFluxCubePass(in, err);
        std::printf("  %-46s %s\n", "F: cache built for a different mesh",
                    ran ? "FAIL (accepted!)" : "PASS (refused)");
        if (ran) ++fails; else std::printf("      reason: %.140s\n", err.c_str());
    }
    {
        Fixture fx; fx.build();
        const std::string dir = base + "/armG";
        const std::string cache = writeCache(dir, fx.partner, GH, Fixture::NF, 0.108, 0.108);
        auto in = fx.inputs(cache, dir, GH);
        in.indirect_ir_series = nullptr;             // the gray path
        std::string err;
        const bool ran = runFluxCubePass(in, err);
        std::printf("  %-46s %s\n", "G: gray path (no band-separated series)",
                    ran ? "FAIL (accepted!)" : "PASS (refused)");
        if (ran) ++fails; else std::printf("      reason: %.140s\n", err.c_str());
    }
    {
        Fixture fx; fx.build();
        const std::string dir = base + "/armH";
        const std::string cache = writeCache(dir, fx.partner, GH, Fixture::NF, 0.108, 0.108);
        auto in = fx.inputs(cache, dir, GH);
        in.occluded = nullptr;                       // no ray tracer
        std::string err;
        const bool ran = runFluxCubePass(in, err);
        std::printf("  %-46s %s\n", "H: no ray-occlusion function",
                    ran ? "FAIL (accepted!)" : "PASS (refused)");
        if (ran) ++fails; else std::printf("      reason: %.140s\n", err.c_str());
    }
    {
        Fixture fx; fx.eps = 1e-6; fx.build();       // eps below the guard
        const std::string dir = base + "/armI";
        const std::string cache = writeCache(dir, fx.partner, GH, Fixture::NF, 0.108, 0.108);
        auto in = fx.inputs(cache, dir, GH);
        std::string err;
        const bool ran = runFluxCubePass(in, err);
        std::printf("  %-46s %s\n", "I: unreconstructible face (eps -> 0)",
                    ran ? "FAIL (accepted!)" : "PASS (refused)");
        if (ran) ++fails; else std::printf("      reason: %.140s\n", err.c_str());
    }

    std::printf("\n==== flux-cube pass unit test: %d failure(s) ====\n\n", fails);
    return fails;
}
