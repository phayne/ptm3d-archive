// shadow_depth_pass.cpp -- see shadow_depth_pass.h for the design rationale.
#include "shadow_depth_pass.h"

#include "solar_disk_visibility.h"
#include "../fileio/frame_naming.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <array>

namespace ptm {
namespace shadowdepth {
namespace {

using Vec3 = std::array<double, 3>;

inline double len3(const Vec3& a) {
    return std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}
inline Vec3 unit3(const Vec3& a) {
    const double l = len3(a);
    return (l > 0.0) ? Vec3{a[0]/l, a[1]/l, a[2]/l} : Vec3{0.0, 0.0, 0.0};
}
inline double dot3(const Vec3& a, const Vec3& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Per-face search state, level-synchronous. Every face at a given rung is
// evaluated in ONE batched trace; that is the whole reason this is not a
// per-face loop calling a single-ray probe.
enum class Status : uint8_t {
    kMarching,     // still climbing the rungs, everything so far umbral
    kBisecting,    // bracket found, refining
    kLit,          // clear at the surface -> 0.0
    kSentinel,     // umbral at every rung up to h_max
    kDone
};

struct FaceState {
    Status st = Status::kMarching;
    double lo = 0.0;   // greatest height verified UMBRAL
    double hi = 0.0;   // least height verified CLEAR
};

// Write <dir>/<name>_<NNNN>.bin in the engine's frame convention:
// uint32 face count, then float32[F].
bool writeFrame(const std::string& path, const std::vector<float>& v) {
    std::ofstream o(path, std::ios::binary);
    if (!o) return false;
    const uint32_t n = static_cast<uint32_t>(v.size());
    o.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    o.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
    return static_cast<bool>(o);
}

bool writeRawSum(const std::string& path, const std::vector<double>& v) {
    std::ofstream o(path, std::ios::binary);
    if (!o) return false;
    const uint32_t n = static_cast<uint32_t>(v.size());
    o.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    o.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(double)));
    return static_cast<bool>(o);
}

bool writeRawCount(const std::string& path, const std::vector<uint32_t>& v) {
    std::ofstream o(path, std::ios::binary);
    if (!o) return false;
    const uint32_t n = static_cast<uint32_t>(v.size());
    o.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    o.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(uint32_t)));
    return static_cast<bool>(o);
}

// Same naming rule as every other per-step product (BUGS.md PTM-664): width
// from the window's epoch count, minimum six digits.
std::string frameIndex(int k, int n_steps) {
    return ptm::frameIndexString(k, n_steps);
}

}  // namespace

bool runShadowDepthPass(const PassInputs& in, std::string& err) {
    // ------------------------------------------------------- preconditions
    if (!in.occluded_batch) {
        err = "shadow depth: no batched ray-occlusion function was supplied. Refusing "
              "to run: treating every probe as unshadowed would report a clear sky "
              "0.0 m at every permanently shadowed face, which is the exact opposite "
              "of the truth and indistinguishable from a working product.";
        return false;
    }
    if (!in.face_centers || !in.face_normals || in.n_faces == 0) {
        err = "shadow depth: no mesh geometry supplied (face_centers / face_normals).";
        return false;
    }
    if (in.face_centers->size() != in.n_faces || in.face_normals->size() != in.n_faces) {
        err = "shadow depth: face_centers/face_normals length does not match n_faces.";
        return false;
    }
    if (!in.times || !in.time_strings) {
        err = "shadow depth: no time axis supplied.";
        return false;
    }
    if (in.times->size() != in.time_strings->size()) {
        err = "shadow depth: times and time_strings differ in length.";
        return false;
    }
    const int n_steps = static_cast<int>(in.times->size());
    if (n_steps <= 0) {
        err = "shadow depth: the output time axis is empty.";
        return false;
    }
    // The same invariant flux_cube_pass asserts, for the same reason: the output
    // epoch grid divides the COMBINED equilibration+output span, so re-deriving
    // it is PTM-250/251. If these do not add up, the rows this pass writes do not
    // line up with time_steps.txt and nothing downstream would notice.
    //
    // ⛔ THE RELATION IS `<=`, NOT `==`, AND THAT IS NOT A LOOSENING.
    // A CHUNKED run sets thermal.checkpoint_lookahead_steps: 1 -- the chunk
    // integrates one grid point PAST its exported window so the checkpoint it
    // hands the next chunk is the state at that chunk's FIRST epoch. The extra
    // point is integrated and never exported, so total_time_steps is
    // equilibration + output + lookahead. Requiring equality refuses every
    // chunked window: MEASURED on a 2-chunk crater80N run, 118 + 59 != 178.
    // (flux_cube_pass.cpp:265 still carries the `!=` form and therefore still
    // refuses a chunked window -- filed, not fixed here.)
    //
    // What must hold is that the exported epochs are the FIRST n_steps after
    // equilibration, which is what the absolute-step indexing below assumes.
    if (in.equilibration_steps + n_steps > in.total_time_steps) {
        std::ostringstream m;
        m << "shadow depth: inconsistent time axis -- equilibration_steps ("
          << in.equilibration_steps << ") + output steps (" << n_steps
          << ") EXCEEDS total_time_steps (" << in.total_time_steps
          << "). The per-epoch frames would index past the end of the ephemeris "
             "and silently stop lining up with time_steps.txt.";
        err = m.str();
        return false;
    }
    {
        const int extra = in.total_time_steps - in.equilibration_steps - n_steps;
        if (extra > 0) {
            // Say it out loud. One extra step is the documented chunk
            // lookahead; more than one is a config this pass has never been
            // shown correct for, and it must not pass in silence.
            std::cout << "[shadow-depth] time axis: " << in.equilibration_steps
                      << " equilibration + " << n_steps << " output = "
                      << (in.equilibration_steps + n_steps) << " of "
                      << in.total_time_steps << " integrated steps; the extra "
                      << extra << " step(s) are"
                      << (extra == 1 ? " the chunk checkpoint lookahead"
                                     : " NOT the single-step chunk lookahead -- "
                                       "confirm what they are")
                      << " and are integrated but not exported.\n";
            if (extra > 1) {
                std::cerr << "[shadow-depth] WARNING: " << extra << " integrated steps "
                             "are neither equilibration nor output. The exported epochs "
                             "are assumed to be the FIRST " << n_steps
                          << " after equilibration; if that is not true for this run, "
                             "every frame is mislabeled in time.\n";
            }
        }
    }
    if (!in.sun_positions ||
        static_cast<int>(in.sun_positions->size()) < in.total_time_steps) {
        err = "shadow depth: sun_positions is missing or shorter than total_time_steps.";
        return false;
    }
    if (!(in.solar_radius_km > 0.0)) {
        err = "shadow depth: solar_radius_km is zero. The umbra is DEFINED by the "
              "finite solar disk -- with a point sun this pass would compute the "
              "penumbral ceiling instead, a different and smaller number at every "
              "face, with nothing to distinguish the two products. Refusing.";
        return false;
    }
    if (!(in.h_max_m > 0.0)) { err = "shadow depth: h_max_m must be positive."; return false; }
    if (in.march_steps < 1)  { err = "shadow depth: march_steps must be >= 1."; return false; }
    if (in.bisect_rounds < 0){ err = "shadow depth: bisect_rounds must be >= 0."; return false; }

    // ⛔ THE tnear GATE. See the header. Both halves are checked, and the
    // resolved value is PRINTED, because a number nobody sees is a number
    // nobody can catch.
    if (!in.scene_has_offset) {
        std::ostringstream m;
        m << "shadow depth: the ray scene has NO origin offset applied. The batched "
             "occlusion path derives its ray tnear from the distance to the scene "
             "origin (max(1e-6*r, 0.01)); without the offset r is the body radius "
             "(~1.737e6 m on the Moon) and tnear is ~1.74 m, so every ray would skip "
             "the first 1.74 m of terrain and a 0-"
          << in.h_max_m << " m vertical search would return confident garbage with no "
             "error anywhere. Refusing rather than producing it.";
        err = m.str();
        return false;
    }
    if (!(in.resolved_tnear_m >= 0.0)) {
        err = "shadow depth: the caller did not report the resolved ray tnear. It is "
              "the one quantity that silently invalidates this whole product, so it "
              "is required, not defaulted.";
        return false;
    }
    const double tnear_limit = in.h_max_m / 100.0;
    if (in.resolved_tnear_m > tnear_limit) {
        std::ostringstream m;
        m << "shadow depth: the resolved ray tnear is " << in.resolved_tnear_m
          << " m, above the limit " << tnear_limit << " m (h_max/100) for a 0-"
          << in.h_max_m << " m search. Rays that skip their first "
          << in.resolved_tnear_m << " m cannot resolve a ceiling at that scale. "
             "Refusing.";
        err = m.str();
        return false;
    }

    const size_t F = in.n_faces;
    const double resolution_m =
        in.h_max_m / (static_cast<double>(in.march_steps) *
                      std::pow(2.0, static_cast<double>(in.bisect_rounds)));

    std::cout << "[shadow-depth] " << F << " faces x " << n_steps
              << " output epoch(s); h_max = " << in.h_max_m << " m, "
              << in.march_steps << " march rungs + " << in.bisect_rounds
              << " bisections -> final bracket " << resolution_m
              << " m (reported value is the bracket midpoint, +-"
              << 0.5 * resolution_m << " m)\n"
              << "[shadow-depth] scene origin offset APPLIED; resolved ray tnear = "
              << in.resolved_tnear_m << " m (limit " << tnear_limit << " m)\n";
    {
        // The ray budget, up front. This pass is the most expensive optional
        // product in the tree and an operator should learn that before it runs,
        // not from the wall clock.
        const double worst_rungs = static_cast<double>(in.march_steps + 1 + in.bisect_rounds);
        const double worst = static_cast<double>(F) * n_steps * worst_rungs *
                             static_cast<double>(ptm::disk::PRIMARY_DISK_PROBES);
        std::cout << "[shadow-depth] ray budget: up to " << std::setprecision(4)
                  << worst << " probe rays (" << ptm::disk::PRIMARY_DISK_PROBES
                  << " per rung), plus escalation where the probe splits. Faces "
                     "clear at the surface cost ONE rung.\n";
    }

    // Per-face vertical (body radial) and the per-face radius, computed once.
    std::vector<Vec3> up(F);
    size_t n_degenerate = 0;
    for (size_t i = 0; i < F; ++i) {
        up[i] = unit3((*in.face_centers)[i]);
        if (up[i][0] == 0.0 && up[i][1] == 0.0 && up[i][2] == 0.0) ++n_degenerate;
    }
    if (n_degenerate > 0) {
        std::ostringstream m;
        m << "shadow depth: " << n_degenerate << " face centroid(s) sit exactly at the "
             "body center, so the radial vertical is undefined there. The mesh frame is "
             "expected to be body-centered (the engine measures a face's radius as "
             "|face_center|). Refusing rather than substituting the face normal, which "
             "is a different quantity and would be reported under the same name.";
        err = m.str();
        return false;
    }

    const double km_per_mesh = in.mesh_to_km;
    if (!(km_per_mesh > 0.0)) { err = "shadow depth: mesh_to_km must be positive."; return false; }
    const double mesh_per_km = 1.0 / km_per_mesh;

    // ----------------------------------------------------- accumulators
    std::vector<double> sum_depth(F, 0.0);
    std::vector<uint32_t> cnt_depth(F, 0u);           // resolved AND nonzero epochs
    std::vector<float> max_depth(F, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> min_depth(F, std::numeric_limits<float>::quiet_NaN());
    std::vector<double> sum_lit(F, 0.0), sum_unres(F, 0.0);
    std::vector<uint32_t> cnt_evaluated(F, 0u);

    // Consistency-gate counters, over the whole window. COUNTED, not sampled.
    uint64_t gate_n_evaluated = 0;
    uint64_t gate_v_direct_pos_depth_nonzero = 0;  // direct>0 && depth != 0
    uint64_t gate_v_depth_pos_direct_pos = 0;      // depth>0  && direct > 0
    uint64_t gate_v_zero_cospos_direct_zero = 0;   // depth==0 && cos>0 && direct==0
    uint64_t gate_n_zero_cos_nonpositive = 0;      // the legitimate-difference set
    uint64_t mono_checked = 0, mono_failures = 0;
    // Attribution for the check-3 violations (depth==0 & cos_inc>0 but the
    // engine reports no direct flux): how much of the solar disk this pass's
    // probe actually saw at the surface, as a histogram over the clear-probe
    // count. A disagreement concentrated at 1-2 clear probes out of 5 is a LIMB
    // SLIVER -- this ladder puts two probes exactly on the solar limb in the
    // vertical plane, while the engine's direct-solar sampler is a concentric
    // point set that does not reach the limb -- and is bounded by the solar
    // angular radius. A disagreement spread across the histogram is not, and
    // would be a defect. Reported either way; the gate cannot say "5 violations"
    // and leave the reader with nothing to act on.
    std::array<uint64_t, 6> gate_c3_clear_hist{};
    std::array<uint64_t, 6> gate_c1_clear_hist{};

    std::vector<float> depth(F);
    std::vector<FaceState> state(F);

    // Scratch, reused across rungs so a 2.4 M-face mesh does not reallocate
    // 21 times per epoch.
    std::vector<size_t> active, split_idx;
    std::vector<Vec3> origins, dirs, probe_o;
    std::vector<uint8_t> blocked;
    std::vector<ptm::disk::DiskFrame> frames;
    std::vector<int> first_clear_rung(F, -1);
    std::vector<uint8_t> surface_clear(F, 0);

    const int nprobe = static_cast<int>(ptm::disk::PRIMARY_DISK_PROBES);
    uint64_t rays_cast = 0;

    const std::string frames_dir = in.output_dir + "/shadow_depth";
    if (in.write_frames) {
        try { std::filesystem::create_directories(frames_dir); }
        catch (const std::exception& e) {
            err = std::string("shadow depth: cannot create ") + frames_dir + ": " + e.what();
            return false;
        }
        // Frames of the other index width are a previous run's (BUGS.md PTM-664).
        const long long stale = ptm::removeOtherWidthFrames(
            frames_dir, "shadow_depth", ptm::frameIndexWidth(n_steps));
        if (stale > 0) {
            std::cerr << "[shadow-depth] NOTE: removed " << stale << " stale shadow_depth_*.bin "
                         "frame(s) of a different index width from " << frames_dir
                      << "/ (a previous run's; BUGS.md PTM-664)." << std::endl;
        }
    }

    // Deterministic monotonicity sample: a fixed stride over faces and epochs,
    // so the audit covers the same population on a re-run and two runs can be
    // compared face for face.
    const int mono_ep_stride = (in.monotonicity_epochs > 0)
        ? std::max(1, n_steps / in.monotonicity_epochs) : 0;
    const size_t mono_f_stride = (in.monotonicity_faces > 0)
        ? std::max<size_t>(1, F / static_cast<size_t>(in.monotonicity_faces)) : 0;

    // ------------------------------------------------------------- epochs
    for (int k = 0; k < n_steps; ++k) {
        const int at = in.equilibration_steps + k;   // ABSOLUTE step
        const Vec3 sun_km = (*in.sun_positions)[static_cast<size_t>(at)];
        // Sun position in MESH units. The engine does this conversion explicitly
        // and so does this pass; see the units note in the header.
        const Vec3 sun_mesh = {sun_km[0] * mesh_per_km,
                               sun_km[1] * mesh_per_km,
                               sun_km[2] * mesh_per_km};

        for (size_t i = 0; i < F; ++i) {
            state[i] = FaceState{};
            depth[i] = std::numeric_limits<float>::quiet_NaN();
        }

        const bool mono_epoch = (mono_ep_stride > 0) && (k % mono_ep_stride == 0);

        // The rung evaluator: for every face in `active`, is the WHOLE solar
        // disk occluded at that face's height? Fills `umbral`, one byte per
        // active face. Returns false when the injected caster broke its
        // contract -- a SHORT return would otherwise be read as "not blocked"
        // and would light up the terrain silently.
        auto evaluateRung = [&](const std::vector<size_t>& faces,
                                const std::vector<double>& heights,
                                std::vector<uint8_t>& umbral,
                                std::string& rerr,
                                bool escalate_on_blocked,
                                std::vector<uint8_t>* clear_probes = nullptr) -> bool {
            const size_t n = faces.size();
            umbral.assign(n, 0);
            if (n == 0) return true;

            frames.resize(n);
            probe_o.resize(n);

            // Pass 1: the PRIMARY_DISK_PROBES directions, every face, ONE
            // batched trace.
            origins.resize(n * static_cast<size_t>(nprobe));
            dirs.resize(n * static_cast<size_t>(nprobe));
            #pragma omp parallel for schedule(static)
            for (long long jj = 0; jj < static_cast<long long>(n); ++jj) {
                const size_t j = static_cast<size_t>(jj);
                const size_t i = faces[j];
                const Vec3& fc = (*in.face_centers)[i];
                const double h = std::max(heights[j], in.surface_offset_m);
                const Vec3 o = {fc[0] + up[i][0]*h, fc[1] + up[i][1]*h, fc[2] + up[i][2]*h};
                const Vec3 rel = {sun_mesh[0] - o[0], sun_mesh[1] - o[1], sun_mesh[2] - o[2]};
                const double d_mesh = len3(rel);
                // The engine's own solar angular radius: asin(R_sun/distance),
                // both in km, recomputed per face-epoch from the REAL distance
                // rather than a 1 AU constant (illumination_model.cpp:3538,
                // :4362).
                const double d_km = d_mesh * km_per_mesh;
                double sr = (d_km > 0.0) ? (in.solar_radius_km / d_km) : 1.0;
                if (sr > 1.0) sr = 1.0;
                // basis_ref = the local vertical, so probe offsets (0, +-1) land
                // exactly on the upper and lower solar limb in the vertical
                // plane. Those are the two rays a height search is decided by.
                frames[j].set(unit3(rel), std::asin(sr), up[i]);
                probe_o[j] = o;
                for (int q = 0; q < nprobe; ++q) {
                    origins[j * nprobe + q] = o;
                    dirs[j * nprobe + q] = frames[j].probe(static_cast<uint32_t>(q));
                }
            }
            blocked.clear();
            in.occluded_batch(origins, dirs, blocked);
            rays_cast += origins.size();
            if (blocked.size() != origins.size()) {
                rerr = "shadow depth: the injected batched ray caster returned a result "
                       "vector of the wrong length. It must fill `blocked` with exactly "
                       "one byte per ray; a short return would be read as 'not blocked' "
                       "and would silently light up the terrain.";
                return false;
            }

            split_idx.clear();
            for (size_t j = 0; j < n; ++j) {
                int clear = 0;
                for (int q = 0; q < nprobe; ++q)
                    if (!blocked[j * nprobe + q]) ++clear;
                if (clear_probes) (*clear_probes)[faces[j]] = static_cast<uint8_t>(clear);
                // ⭐ A CLEAR PROBE IS PROOF, AND NOTHING MAY OVERTURN IT.
                // The question here is BOOLEAN -- is any part of the solar disk
                // visible -- so one probe ray that reaches the Sun settles it.
                // The first version of this pass escalated a SPLIT probe to the
                // 128-sample set and then took `any_clear` from the escalation
                // alone, which could report UMBRAL for a face whose probe had
                // already found the Sun: the escalation point set does not
                // contain the probe directions, and at the solar limb it does
                // not come close to them. MEASURED on the 80N crater A/B run
                // (12,800 faces x 118 epochs): 2 of the 5 check-1 gate
                // violations were exactly that self-contradiction. A fraction
                // needs the escalation; a boolean does not.
                if (clear > 0) { umbral[j] = 0; continue; }
                umbral[j] = 1;
                if (escalate_on_blocked) split_idx.push_back(j);
            }
            if (split_idx.empty()) return true;

            // Escalation of the BLOCKED side -- the only side where more samples
            // can change a boolean answer. Five rays cannot prove a 0.53 deg
            // disk carries no clear sliver; they can only fail to find one.
            //
            // It runs only at the rungs whose answer is FINAL -- the surface
            // (is this face lit at h = 0?) and the search ceiling (is this face
            // unresolved?) -- because those two carry the product's semantics,
            // 0.0 and +inf. Escalating every rung would multiply a permanently
            // shadowed face's ray budget by ~26x to move a bracket by less than
            // the height equivalent of the solar angular radius. The cost of
            // that choice is stated in the manifest rather than hidden: an
            // intermediate rung is decided by the probe alone, so a reported
            // ceiling can be biased HIGH by at most the height a theta_sun
            // rotation of the limb ray spans at the local occluder distance.
            const uint32_t N = static_cast<uint32_t>(std::max(8, in.solar_disk_samples));
            origins.resize(split_idx.size() * N);
            dirs.resize(split_idx.size() * N);
            #pragma omp parallel for schedule(static)
            for (long long qq = 0; qq < static_cast<long long>(split_idx.size()); ++qq) {
                const size_t q = static_cast<size_t>(qq);
                const size_t j = split_idx[q];
                for (uint32_t t = 0; t < N; ++t) {
                    origins[q * N + t] = probe_o[j];
                    dirs[q * N + t] = frames[j].sample(t, N);
                }
            }
            blocked.clear();
            in.occluded_batch(origins, dirs, blocked);
            rays_cast += origins.size();
            if (blocked.size() != origins.size()) {
                rerr = "shadow depth: the injected batched ray caster returned a result "
                       "vector of the wrong length during disk escalation.";
                return false;
            }
            for (size_t q = 0; q < split_idx.size(); ++q) {
                for (uint32_t t = 0; t < N; ++t) {
                    if (!blocked[q * N + t]) { umbral[split_idx[q]] = 0; break; }
                }
            }
            return true;
        };

        // ---- MARCH ----------------------------------------------------
        // `first_clear_rung` records where each face first went clear; the
        // monotonicity audit is the only consumer. Reset per epoch.
        std::fill(first_clear_rung.begin(), first_clear_rung.end(), -1);
        // How many of the probe rays were clear AT THE SURFACE. Kept for the
        // consistency gate, which otherwise cannot say WHY it disagreed with
        // direct_solar -- and "5 violations" with no attribution is a number
        // nobody can act on.
        std::fill(surface_clear.begin(), surface_clear.end(), uint8_t(0));
        std::vector<uint8_t> umbral;
        std::vector<double> heights;

        for (int r = 0; r <= in.march_steps; ++r) {
            const double h = in.h_max_m * static_cast<double>(r) /
                             static_cast<double>(in.march_steps);
            active.clear();
            for (size_t i = 0; i < F; ++i)
                if (state[i].st == Status::kMarching) active.push_back(i);
            if (active.empty()) break;
            heights.assign(active.size(), h);
            // Escalate the blocked side only where the answer is final: the
            // surface rung decides "lit" (0.0) and the top rung decides
            // "unresolved" (+inf).
            const bool escalate = (r == 0) || (r == in.march_steps);
            if (!evaluateRung(active, heights, umbral, err, escalate,
                              (r == 0) ? &surface_clear : nullptr)) return false;
            for (size_t j = 0; j < active.size(); ++j) {
                const size_t i = active[j];
                if (umbral[j]) {
                    state[i].lo = h;
                    if (r == in.march_steps) state[i].st = Status::kSentinel;
                } else {
                    first_clear_rung[i] = r;
                    if (r == 0) {
                        state[i].st = Status::kLit;
                    } else {
                        state[i].hi = h;
                        state[i].st = Status::kBisecting;
                    }
                }
            }
        }

        // ---- BISECT ---------------------------------------------------
        for (int b = 0; b < in.bisect_rounds; ++b) {
            active.clear();
            for (size_t i = 0; i < F; ++i)
                if (state[i].st == Status::kBisecting) active.push_back(i);
            if (active.empty()) break;
            heights.resize(active.size());
            for (size_t j = 0; j < active.size(); ++j)
                heights[j] = 0.5 * (state[active[j]].lo + state[active[j]].hi);
            if (!evaluateRung(active, heights, umbral, err, /*escalate=*/false)) return false;
            for (size_t j = 0; j < active.size(); ++j) {
                const size_t i = active[j];
                if (umbral[j]) state[i].lo = heights[j];
                else           state[i].hi = heights[j];
            }
        }

        for (size_t i = 0; i < F; ++i) {
            switch (state[i].st) {
                case Status::kLit:      depth[i] = 0.0f; break;
                case Status::kSentinel: depth[i] = kUnresolved; break;
                case Status::kBisecting:
                    depth[i] = static_cast<float>(0.5 * (state[i].lo + state[i].hi));
                    break;
                default:
                    depth[i] = std::numeric_limits<float>::quiet_NaN();
                    break;
            }
        }

        // ---- MONOTONICITY AUDIT (sampled, reported, never zeroed) ------
        //
        // The march stops at the first clear rung, so nothing above it has been
        // looked at. An overhang can make a higher rung umbral again, which
        // would mean the reported ceiling is the bottom of a stack rather than
        // the top of one. That is DEFENSIBLE -- it is the contiguous column a
        // person standing there is inside -- but it has to be measured, not
        // assumed away, because a bisection over a non-monotone predicate is
        // exactly the failure this ladder was chosen to avoid.
        if (mono_epoch && mono_f_stride > 0) {
            std::vector<size_t> mf;
            for (size_t i = 0; i < F; i += mono_f_stride)
                if (first_clear_rung[i] >= 0 && first_clear_rung[i] < in.march_steps)
                    mf.push_back(i);
            for (int r = 1; r <= in.march_steps; ++r) {
                active.clear();
                for (size_t i : mf) if (first_clear_rung[i] < r) active.push_back(i);
                if (active.empty()) continue;
                const double h = in.h_max_m * static_cast<double>(r) /
                                 static_cast<double>(in.march_steps);
                heights.assign(active.size(), h);
                if (!evaluateRung(active, heights, umbral, err, /*escalate=*/false)) return false;
                for (size_t j = 0; j < active.size(); ++j) {
                    ++mono_checked;
                    if (umbral[j]) ++mono_failures;
                }
            }
        }

        // ---- CONSISTENCY GATE vs the engine's own direct_solar ---------
        if (in.direct_solar) {
            const size_t stride = static_cast<size_t>(in.total_time_steps);
            for (size_t i = 0; i < F; ++i) {
                const float d = depth[i];
                if (std::isnan(d)) continue;
                const size_t idx = i * stride + static_cast<size_t>(at);
                if (idx >= in.direct_solar->size()) continue;
                const double q = (*in.direct_solar)[idx];
                const Vec3 rel = {sun_mesh[0] - (*in.face_centers)[i][0],
                                  sun_mesh[1] - (*in.face_centers)[i][1],
                                  sun_mesh[2] - (*in.face_centers)[i][2]};
                const double cos_inc = dot3((*in.face_normals)[i], unit3(rel));
                ++gate_n_evaluated;
                if (q > 0.0 && d != 0.0f) {
                    ++gate_v_direct_pos_depth_nonzero;
                    ++gate_c1_clear_hist[std::min<size_t>(surface_clear[i], 5)];
                }
                if (d > 0.0f && q > 0.0)                  ++gate_v_depth_pos_direct_pos;
                if (d == 0.0f && cos_inc > 0.0 && !(q > 0.0)) {
                    ++gate_v_zero_cospos_direct_zero;
                    ++gate_c3_clear_hist[std::min<size_t>(surface_clear[i], 5)];
                }
                if (d == 0.0f && cos_inc <= 0.0)          ++gate_n_zero_cos_nonpositive;
            }
        }

        // ---- accumulate + write the frame ------------------------------
        for (size_t i = 0; i < F; ++i) {
            const float d = depth[i];
            if (std::isnan(d)) continue;
            ++cnt_evaluated[i];
            if (d == 0.0f) { sum_lit[i] += 1.0; continue; }
            if (std::isinf(d)) { sum_unres[i] += 1.0; continue; }
            sum_depth[i] += d;
            ++cnt_depth[i];
            if (std::isnan(max_depth[i]) || d > max_depth[i]) max_depth[i] = d;
            if (std::isnan(min_depth[i]) || d < min_depth[i]) min_depth[i] = d;
        }

        if (in.write_frames) {
            const std::string fp = frames_dir + "/shadow_depth_" + frameIndex(k, n_steps) + ".bin";
            if (!writeFrame(fp, depth)) {
                err = "shadow depth: cannot write " + fp;
                return false;
            }
        }
    }

    // ------------------------------------------------------- statistics
    //
    // avg/max/min are over RESOLVED, NONZERO epochs only, and NaN where a face
    // had none. Folding the 0.0 epochs in would make the mean of a face that is
    // lit for half the window read as half its real ceiling, and folding the
    // +inf epochs in would make it +inf. The fractions carry those two
    // populations instead, which is what makes the three fields readable
    // together.
    std::vector<float> avg(F, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> lit_frac(F, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> unres_frac(F, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < F; ++i) {
        if (cnt_depth[i] > 0)
            avg[i] = static_cast<float>(sum_depth[i] / static_cast<double>(cnt_depth[i]));
        if (cnt_evaluated[i] > 0) {
            lit_frac[i]   = static_cast<float>(sum_lit[i]   / static_cast<double>(cnt_evaluated[i]));
            unres_frac[i] = static_cast<float>(sum_unres[i] / static_cast<double>(cnt_evaluated[i]));
        }
    }

    const std::string stats_dir = in.output_dir + "/statistics";
    try { std::filesystem::create_directories(stats_dir); }
    catch (const std::exception& e) {
        err = std::string("shadow depth: cannot create ") + stats_dir + ": " + e.what();
        return false;
    }
    bool ok = true;
    ok &= writeFrame(stats_dir + "/shadow_depth_avg.bin", avg);
    ok &= writeFrame(stats_dir + "/shadow_depth_max.bin", max_depth);
    ok &= writeFrame(stats_dir + "/shadow_depth_min.bin", min_depth);
    ok &= writeFrame(stats_dir + "/shadow_lit_frac.bin", lit_frac);
    ok &= writeFrame(stats_dir + "/shadow_unresolved_frac.bin", unres_frac);
    if (!ok) { err = "shadow depth: a statistics write failed."; return false; }

    // Chunk merge needs sum+count for every field whose merge is not a max.
    // `max` merges as a max of maxima and needs nothing; `avg` and BOTH
    // fractions do not, and without these side files a chunked window would
    // silently ship the last chunk's fraction. Budgeted here, not later.
    if (in.write_raw_stats) {
        const std::string raw_dir = stats_dir + "/_raw";
        try { std::filesystem::create_directories(raw_dir); }
        catch (const std::exception& e) {
            err = std::string("shadow depth: cannot create ") + raw_dir + ": " + e.what();
            return false;
        }
        std::vector<uint32_t> ones_evaluated = cnt_evaluated;
        bool rok = true;
        rok &= writeRawSum  (raw_dir + "/shadow_depth_sum.bin",   sum_depth);
        rok &= writeRawCount(raw_dir + "/shadow_depth_count.bin", cnt_depth);
        rok &= writeRawSum  (raw_dir + "/shadow_lit_frac_sum.bin",   sum_lit);
        rok &= writeRawCount(raw_dir + "/shadow_lit_frac_count.bin", ones_evaluated);
        rok &= writeRawSum  (raw_dir + "/shadow_unresolved_frac_sum.bin",   sum_unres);
        rok &= writeRawCount(raw_dir + "/shadow_unresolved_frac_count.bin", ones_evaluated);
        // min is not merge-reconstructible from a max-of-maxima rule, but it IS
        // a min-of-minima, which ptm_merge_chunks already does for every
        // datatype. Nothing extra is needed for it.
        if (!rok) { err = "shadow depth: a raw statistics write failed."; return false; }
    }

    // --------------------------------------------------------- manifest
    //
    // Written LAST, deliberately: its presence certifies that every map beside
    // it is complete. A reader that finds the maps and no manifest is looking
    // at an interrupted run, and should say so rather than average it.
    {
        std::ostringstream js;
        js << std::setprecision(17);
        js << "{\"ptm3d_shadow_depth_schema\":1,"
           << "\"quantity\":\"maximum height above the local surface at which the Sun "
              "is not even partially visible (the UMBRAL ceiling)\","
           << "\"units\":\"meters\","
           << "\"vertical_convention\":\"body_radial\","
           << "\"vertical_note\":\"height is measured along the radial from the body "
              "center through the face centroid, NOT along the face normal; this is the "
              "same vertical ptm_flux_cube_geom places its cubes along\","
           << "\"h_max_m\":" << in.h_max_m << ","
           << "\"march_steps\":" << in.march_steps << ","
           << "\"march_rung_m\":" << (in.h_max_m / in.march_steps) << ","
           << "\"bisect_rounds\":" << in.bisect_rounds << ","
           << "\"final_bracket_m\":" << resolution_m << ","
           << "\"reported_value\":\"midpoint of the final bracket; the value is within "
              "+-final_bracket_m/2 of the transition the search found\","
           << "\"search\":\"march to the first clear rung from the surface upward, then "
              "bisect inside that bracket. Pure bisection is NOT used: it assumes the "
              "blocked-below/clear-above transition is single-valued and an overhang "
              "breaks that\","
           << "\"surface_offset_m\":" << in.surface_offset_m << ","
           << "\"sentinel_encoding\":{\"lit_at_surface\":\"0.0\","
              "\"unresolved\":\"+inf (float32 0x7F800000): no clear height at or below "
              "h_max_m. This is the MAJORITY value inside a permanently shadowed region "
              "by design; it is not a failure and no number is invented for it\","
              "\"not_evaluated\":\"NaN\"},"
           << "\"solar_disk\":{\"model\":\"finite_disk\",\"solar_radius_km\":"
           << in.solar_radius_km
           << ",\"angular_radius\":\"asin(solar_radius_km / distance), recomputed per "
              "face-epoch from the real Sun distance -- the engine's own formula "
              "(illumination_model.cpp:3538, :4362)\","
              "\"ladder\":\"probe " << ptm::disk::PRIMARY_DISK_PROBES
           << " directions (center + 4 limb points). ANY clear probe means the disk "
              "is not fully occluded and settles the rung -- the question is boolean, "
              "so one ray that reaches the Sun is proof and the escalation may not "
              "overturn it. A unanimously BLOCKED probe escalates to "
           << in.solar_disk_samples
           << " Hammersley/Shirley-Chiu disk samples, but ONLY at the two rungs whose "
              "answer is final: the surface (which decides 0.0) and the search ceiling "
              "(which decides +inf). Intermediate march rungs and the bisection are "
              "decided by the probe alone, so a reported ceiling can be biased HIGH by "
              "at most the height a theta_sun rotation of the limb ray spans at the "
              "local occluder distance\","
              "\"basis\":\"the disk basis is oriented on the local vertical, so two of "
              "the four limb probes land exactly on the upper and lower solar limb in "
              "the vertical plane -- the pair a height search is decided by\"},"
           << "\"tnear_resolved_m\":" << in.resolved_tnear_m << ","
           << "\"tnear_limit_m\":" << tnear_limit << ","
           << "\"scene_origin_offset_applied\":true,"
           << "\"epoch_window\":{\"n_output_steps\":" << n_steps
           << ",\"equilibration_steps\":" << in.equilibration_steps
           << ",\"total_time_steps\":" << in.total_time_steps
           << ",\"first_time\":\"" << (*in.time_strings).front()
           << "\",\"last_time\":\"" << (*in.time_strings).back()
           << "\",\"time_axis_source\":\"time_steps.txt\"},"
           << "\"statistics_scope\":\"avg/max/min are over RESOLVED, NONZERO epochs "
              "only; NaN where a face had none. shadow_lit_frac and "
              "shadow_unresolved_frac are over all EVALUATED epochs\","
           << "\"rays_cast\":" << rays_cast << ","
           << "\"consistency_gate\":";
        if (in.direct_solar) {
            js << "{\"ran\":true,\"scope\":\"every output epoch, every face; counted "
                  "not sampled\","
               << "\"n_face_epochs\":" << gate_n_evaluated << ","
               << "\"violations_direct_positive_implies_depth_zero\":"
               << gate_v_direct_pos_depth_nonzero << ","
               << "\"violations_depth_positive_implies_direct_zero\":"
               << gate_v_depth_pos_direct_pos << ","
               << "\"violations_depth_zero_and_cos_inc_positive_implies_direct_positive\":"
               << gate_v_zero_cospos_direct_zero << ","
               << "\"n_depth_zero_and_cos_inc_nonpositive\":"
               << gate_n_zero_cos_nonpositive << ","
               << "\"violation_attribution\":{"
               << "\"surface_clear_probe_histogram_check1\":[";
            for (size_t b2 = 0; b2 < 6; ++b2) js << (b2 ? "," : "") << gate_c1_clear_hist[b2];
            js << "],\"surface_clear_probe_histogram_check3\":[";
            for (size_t b2 = 0; b2 < 6; ++b2) js << (b2 ? "," : "") << gate_c3_clear_hist[b2];
            js << "],\"probe_rays\":" << ptm::disk::PRIMARY_DISK_PROBES
               << ",\"how_to_read\":\"index k = the number of this pass's probe rays "
                  "that were CLEAR at the surface. A check-3 population concentrated at "
                  "k = 1-2 is a limb sliver: two of this ladder's probes sit exactly on "
                  "the solar limb in the vertical plane, and the engine's direct-solar "
                  "sampler is a concentric point set that does not reach the limb, so at "
                  "a terrain shadow edge the limb ray is clear while every interior "
                  "sample is blocked. That difference is bounded by the solar angular "
                  "radius. A population spread across k is NOT that, and is a defect\"},"
               << "\"legitimate_difference_note\":\"the cos_inc <= 0 population is where "
                  "this field and direct_solar legitimately differ: a facet tilted away "
                  "from the Sun under a clear sky has zero direct flux while a person "
                  "standing on it is in sunlight. Its size is reported, not suppressed -- "
                  "it is the quantitative statement of the definition this product uses\"}";
        } else {
            js << "{\"ran\":false,\"reason\":\"no direct_solar field was supplied to the "
                  "pass, so nothing was compared. This is NOT a pass\"}";
        }
        js << ",\"monotonicity_audit\":{\"face_epoch_rungs_checked\":" << mono_checked
           << ",\"failures\":" << mono_failures
           << ",\"note\":\"a failure is a rung ABOVE the first clear rung that is umbral "
              "again -- an overhang. The reported ceiling is then the top of the "
              "contiguous umbral column from the surface, which is the quantity a person "
              "standing there is inside, and not the global maximum umbral height. "
              "Counted, never zeroed\"},"
           << "\"provenance\":" << (in.provenance_json.empty() ? "null" : in.provenance_json)
           << "}";
        const std::string mp = stats_dir + "/shadow_depth.json";
        std::ofstream o(mp);
        if (!o) { err = "shadow depth: cannot write " + mp; return false; }
        o << js.str() << "\n";
        if (!o.good()) { err = "shadow depth: failed writing " + mp; return false; }
    }

    std::cout << "[shadow-depth] " << rays_cast << " rays cast; wrote "
              << (in.write_frames ? n_steps : 0) << " frame(s), 5 statistics maps and "
              << "statistics/shadow_depth.json\n";
    if (in.direct_solar) {
        std::cout << "[shadow-depth] consistency gate over " << gate_n_evaluated
                  << " face-epochs: " << gate_v_direct_pos_depth_nonzero
                  << " / " << gate_v_depth_pos_direct_pos << " / "
                  << gate_v_zero_cospos_direct_zero
                  << " violations (direct>0=>depth==0 / depth>0=>direct==0 / "
                     "depth==0 & cos>0 => direct>0); "
                  << gate_n_zero_cos_nonpositive
                  << " face-epochs are lit at the surface with cos_inc <= 0, where the "
                     "two fields legitimately differ\n";
    } else {
        std::cout << "[shadow-depth] consistency gate DID NOT RUN (no direct_solar "
                     "supplied); the manifest records that, and it is not a pass\n";
    }
    std::cout << "[shadow-depth] monotonicity audit: " << mono_failures << " / "
              << mono_checked << " sampled rungs above the first clear rung were umbral "
                 "again (overhangs)\n";
    return true;
}

}  // namespace shadowdepth
}  // namespace ptm
