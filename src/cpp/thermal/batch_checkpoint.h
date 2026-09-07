// batch_checkpoint.h -- BATCH-COMPLETION PARTIALS ("PTMB"), plan phase 1D'.
//
// ---------------------------------------------------------------------------
// WHY THIS IS NOT "CHECKPOINT EVERY N OUTPUT STEPS"
// ---------------------------------------------------------------------------
// The thermal solvers are BATCH-OUTER / TIME-INNER: each face batch runs ALL
// timesteps, then the next batch starts. There is therefore no wall-clock
// moment at which all faces share the same output step, and a whole-mesh dump
// taken mid-run banks un-solved faces. tools/config_parser.h says so at the key
// `thermal.checkpoint_every_output_steps`, ptm_thermal_sim warns loudly if it is
// set, and docs/PORTABILITY-AUDIT.md section 8 calls the partial dump "a resume
// hazard". Per-output-step checkpointing belongs to the CPU-first, time-outer
// solver on the portability roadmap and is deliberately NOT implemented here.
//
// What a batch-outer solver CAN offer is the inverse: at a batch boundary a
// well-defined set of faces is FINISHED FOR THE WHOLE WINDOW. Bank that. On a
// restart the finished batches are skipped and only the unfinished ones are
// integrated.
//
// ---------------------------------------------------------------------------
// A PARTIAL IS NOT A STATE, AND CANNOT BE MISTAKEN FOR ONE
// ---------------------------------------------------------------------------
// The plan (section 1.3.3) specified partials as PTMT files carrying `flags`
// bit3. They are a SEPARATE FORMAT here, and the reason is a correctness one,
// not a taste one:
//
//   * A PTMT body is DEFINED as T[n_layers * n_faces] for the whole mesh. A
//     partial covering 8 faces of 2,397,890 could only be written by lying
//     about `n_faces` (losing the global face range, so nothing could place the
//     columns back) or by laying down a full-width body that is mostly zeros --
//     which is EXACTLY the "banks un-solved faces at 0 K" footgun the audit
//     names and the PTM-78 zero-column runaway already cost this project once.
//   * A partial must carry far more than a column. A completed batch also owns
//     its faces' whole SURFACE TIME SERIES (every statistic, every per-step
//     product and every location export is a reduction of it) and its slice of
//     the depth-map accumulator. PTMT has nowhere to put either.
//
// So a partial announces itself with a DIFFERENT MAGIC and therefore fails a
// state reader at byte 0 -- a stronger guarantee than a flag bit, because it
// holds even for a reader that never learned what bit3 means.
// `state_io::flag::kBatchCompletionPartial` stays reserved and stays refused;
// nothing here writes a PTMT.
//
// ---------------------------------------------------------------------------
// ⭐ THE PARTIAL IS KEYED BY THE BYTES OF ITS BATCH'S INPUTS
// ---------------------------------------------------------------------------
// Identity by configuration alone would not be enough. `config_identity_hash`
// explicitly does NOT cover per-face maps, view factors or the radiation
// coupling (state_identity.h says so), so two runs that differ in any of those
// hash EQUAL -- and a partial from one would silently load into the other.
//
// Every partial therefore records `input_digest`, a CRC32 over the exact bytes
// the batch integrated: its absorbed-flux rows, its seeded initial column and
// its per-face interior heat flux. A resume recomputes that digest from its own
// live inputs and REFUSES any partial that disagrees. If the forcing moved --
// different albedo map, different view factors, radiation on instead of off,
// a different epoch -- the digest moves with it and the partial is rejected.
// This is the one guard that does not depend on remembering to hash a new
// input: the inputs ARE the hash.
//
// ---------------------------------------------------------------------------
// FORMAT (PTMB v1) -- field by field, little-endian, no struct padding
// ---------------------------------------------------------------------------
//   uint32   magic         = 0x424D5450  ('PTMB' read MSB-first)
//   uint32   version       = 1
//   uint32   header_bytes  offset of the body = size of everything above it
//   uint32   flags         which body sections are present (see `part::`)
//
//   uint64   run_identity  fold of every scalar below; a cheap first reject
//   uint32   n_records     how many batches this file carries
//   uint32   n_faces_total
//   uint32   n_layers
//   uint32   n_time_steps  length of one banked surface row
//   uint32   n_depths      depth-map entries, 0 when the product is off
//   uint32   batch_size    faces per batch, so face ranges are derivable
//
//   float64  z[n_layers]           the depth grid
//   float64  et_seconds            epoch of the window's first output step
//   float64  output_cadence_s
//   float64  refgrid_H
//   int32    grid_m, grid_n, grid_b, reserved0
//   float64  mesh_geomhash
//   uint64   config_identity_hash  thermal/state_identity.h
//   char[32] body_name             NUL-padded
//   char[48] engine_git_sha        NUL-padded
//
//   -- record table, n_records entries, ascending batch_idx --
//   uint32   batch_idx
//   uint32   batch_n               faces actually in it (the last one is short)
//   uint32   input_digest          CRC32 of THIS batch's inputs
//   uint32   reserved1
//
//   -- body, records in table order; each section present iff its flag is set --
//   bit0 kSurfaceSeries : float64 [batch_n * n_time_steps]  ([f_local*T + t])
//   bit1 kFinalColumn   : float32 [batch_n * n_layers]      ([f_local*L + l])
//   bit2 kDepthAccum    : float64 sum, vmax, vmin, each [n_depths * batch_n]
//                         (depth-major within the record), then
//                         uint32 count[batch_n]
//
//   uint32   crc32       over EVERY byte of the file before it
//
// The CRC covers the whole file for the same reason the PTMT one does: a
// OneDrive lazy-fetch placeholder has the right LENGTH and the right first
// block, and only a checksum over the body catches it.
//
// ---------------------------------------------------------------------------
// SIZING -- the plan's estimate was two orders of magnitude low, and here is why
// ---------------------------------------------------------------------------
// Plan section 1.3.3 sized a partial at "0.138 s / 210 MiB", which is the cost
// of a PTMT COLUMN dump. A partial must also carry the surface time series,
// and that is the dominant term by ~125x:
//
//   per face   = n_time_steps * 8 B  (series)
//              + n_layers * 4 B      (column)
//              + n_depths * 28 B     (sum/max/min as f64 + count amortized)
//
//   Lunar polar production shape, F = 2,397,890, L = 23, T = 1440, D = 0:
//     series  11.52 kB/face -> 27.6 GB      column 92 B/face -> 0.22 GB
//
// 27.6 GB is the honest number for a full production window, and it is the
// same order as the `surface_temps` array the run already holds in RAM. It is
// NOT free, it is disclosed at startup by estimateBytes(), and the caller is
// expected to refuse above a ceiling rather than fill a disk quietly.
//
// ---------------------------------------------------------------------------
// WHAT THIS DOES **NOT** BOUND
// ---------------------------------------------------------------------------
// A run that dies inside a batch loses that batch's whole time integration --
// that is the granularity, by construction. For finer blast-radius bounding the
// answer remains the runner's process-level chunk split
// (scripts/run_chunked_simulation.py), which is already built, already gated
// and already mathematically equivalent to a single shot. A resilience feature
// that quietly bounds less than a user assumes is worse than none, so this is
// stated here, in the config-key documentation and in the startup banner.

#ifndef PTM_THERMAL_BATCH_CHECKPOINT_H
#define PTM_THERMAL_BATCH_CHECKPOINT_H

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "thermal_state_io.h"

namespace thermal {
namespace batchckpt {

constexpr std::uint32_t kMagic   = 0x424D5450u;  // 'PTMB' printed MSB-first
constexpr std::uint32_t kVersion = 1u;

constexpr std::size_t kBodyNameBytes = state_io::kBodyNameBytes;  // 32
constexpr std::size_t kGitShaBytes   = state_io::kGitShaBytes;    // 48

// Which per-face products a partial carries. A resume REFUSES a partial whose
// section set does not match what the live run needs -- a missing section is a
// product that would silently come back zero.
namespace part {
constexpr std::uint32_t kSurfaceSeries = 1u << 0;
constexpr std::uint32_t kFinalColumn   = 1u << 1;
constexpr std::uint32_t kDepthAccum    = 1u << 2;
}  // namespace part

// ---------------------------------------------------------------------------
// The identity every partial in one run shares. Two partials whose Identity
// differs in ANY field belong to different computations and must never be
// merged; the fold in `hash()` is the cheap first reject and the field-by-field
// comparison in `describeMismatch()` is what a human is shown.
// ---------------------------------------------------------------------------
struct Identity {
    std::vector<double> z;
    double et_seconds       = std::numeric_limits<double>::quiet_NaN();
    double output_cadence_s = std::numeric_limits<double>::quiet_NaN();
    double refgrid_H        = std::numeric_limits<double>::quiet_NaN();
    int    grid_m = -1, grid_n = -1, grid_b = -1;
    double mesh_geomhash = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t config_identity_hash = 0;
    std::string body_name;
    std::string engine_git_sha;

    std::uint32_t n_faces_total = 0;
    std::uint32_t n_layers      = 0;
    std::uint32_t n_time_steps  = 0;
    std::uint32_t n_depths      = 0;
    std::uint32_t batch_size    = 0;
    std::uint32_t sections      = 0;   // the `part::` bit set

    // ⭐ THE DEPTH PRODUCT'S OWN IDENTITY. n_depths alone is NOT enough, and
    // nothing else covers this: `config_identity_hash` hashes the resolved
    // Planet and the conduction grid (state_identity.h says so explicitly), and
    // `input_digest` folds the batch's FORCING, which does not move when the
    // requested depths do. Without these two fields a run killed halfway,
    // re-pointed at a different depth list of the SAME LENGTH and resumed,
    // would write the old depths' accumulator for the restored batches and the
    // new depths' for the rest -- one product, two meanings, complete coverage,
    // and nothing downstream able to tell. Found by code-review, not by a gate.
    std::uint64_t depth_plan_hash = 0;   // 0 when no depth product is requested
    std::int32_t  depth_win_start = -1;  // the accumulation window, in output
    std::int32_t  depth_win_end   = -1;  // steps -- it moves the product too

    // FNV-1a over the canonical text of every field above. NaN is folded by its
    // printed form, so "not recorded" hashes consistently instead of comparing
    // unequal to itself the way a raw double would.
    std::uint64_t hash() const {
        std::ostringstream os;
        os.precision(17);
        os << "z=";
        for (double v : z) os << v << ',';
        os << ";et=" << et_seconds
           << ";cad=" << output_cadence_s
           << ";H=" << refgrid_H
           << ";m=" << grid_m << ";n=" << grid_n << ";b=" << grid_b
           << ";geom=" << mesh_geomhash
           << ";cfg=" << config_identity_hash
           << ";body=" << body_name
           << ";sha=" << engine_git_sha
           << ";F=" << n_faces_total
           << ";L=" << n_layers
           << ";T=" << n_time_steps
           << ";D=" << n_depths
           << ";BS=" << batch_size
           << ";sec=" << sections
           << ";dplan=" << depth_plan_hash
           << ";dwin=" << depth_win_start << ',' << depth_win_end;
        const std::string s = os.str();
        std::uint64_t h = 1469598103934665603ull;
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
        return h;
    }

    // Empty when the two agree. Otherwise one line per differing field, so a
    // refusal names WHICH clause failed rather than reporting "hash mismatch".
    std::string describeMismatch(const Identity& other) const {
        std::ostringstream os;
        auto cmpD = [&](const char* name, double a, double b) {
            const bool both_nan = (a != a) && (b != b);
            if (!both_nan && !(a == b))
                os << "    " << name << ": file " << b << ", run " << a << "\n";
        };
        auto cmpU = [&](const char* name, unsigned long long a,
                        unsigned long long b) {
            if (a != b)
                os << "    " << name << ": file " << b << ", run " << a << "\n";
        };
        auto cmpS = [&](const char* name, const std::string& a,
                        const std::string& b) {
            if (a != b)
                os << "    " << name << ": file '" << b << "', run '" << a
                   << "'\n";
        };
        os.precision(17);
        if (z.size() != other.z.size()) {
            os << "    z[]: file has " << other.z.size() << " nodes, run has "
               << z.size() << "\n";
        } else {
            for (std::size_t i = 0; i < z.size(); ++i) {
                if (z[i] != other.z[i]) {
                    os << "    z[" << i << "]: file " << other.z[i] << ", run "
                       << z[i] << "  (the depth grid moved -- restoring onto it "
                       << "would reinterpret every column)\n";
                    break;
                }
            }
        }
        cmpD("et_seconds", et_seconds, other.et_seconds);
        cmpD("output_cadence_s", output_cadence_s, other.output_cadence_s);
        cmpD("refgrid_H", refgrid_H, other.refgrid_H);
        cmpU("grid_m", static_cast<unsigned long long>(grid_m),
             static_cast<unsigned long long>(other.grid_m));
        cmpU("grid_n", static_cast<unsigned long long>(grid_n),
             static_cast<unsigned long long>(other.grid_n));
        cmpU("grid_b", static_cast<unsigned long long>(grid_b),
             static_cast<unsigned long long>(other.grid_b));
        cmpD("mesh_geomhash", mesh_geomhash, other.mesh_geomhash);
        cmpU("config_identity_hash", config_identity_hash,
             other.config_identity_hash);
        cmpS("body_name", body_name, other.body_name);
        cmpS("engine_git_sha", engine_git_sha, other.engine_git_sha);
        cmpU("n_faces_total", n_faces_total, other.n_faces_total);
        cmpU("n_layers", n_layers, other.n_layers);
        cmpU("n_time_steps", n_time_steps, other.n_time_steps);
        cmpU("n_depths", n_depths, other.n_depths);
        cmpU("batch_size", batch_size, other.batch_size);
        cmpU("sections", sections, other.sections);
        if (depth_plan_hash != other.depth_plan_hash)
            os << "    depth_plan_hash: file " << other.depth_plan_hash
               << ", run " << depth_plan_hash
               << "  (the requested depths, or the grid nodes they resolved "
                  "onto, are not the same product)\n";
        cmpU("depth_win_start", static_cast<unsigned long long>(depth_win_start),
             static_cast<unsigned long long>(other.depth_win_start));
        cmpU("depth_win_end", static_cast<unsigned long long>(depth_win_end),
             static_cast<unsigned long long>(other.depth_win_end));
        return os.str();
    }
};

// One completed batch's payload, in memory.
struct Record {
    std::uint32_t batch_idx    = 0;
    std::uint32_t batch_n      = 0;
    std::uint32_t input_digest = 0;

    std::vector<double>        surface;      // batch_n * n_time_steps
    std::vector<float>         column;       // batch_n * n_layers
    std::vector<double>        depth_sum;    // n_depths * batch_n
    std::vector<double>        depth_max;
    std::vector<double>        depth_min;
    std::vector<std::uint32_t> depth_count;  // batch_n
};

// ---------------------------------------------------------------------------
// The input digest. CRC32 over the exact bytes a batch integrates, in a fixed
// order. Callers feed it the same way on the write and on the resume; any
// divergence in either operand is a refusal, which is the whole point.
//
// `flux` is read as flux[t * flux_stride + f_local], matching both the array
// mode (stride = n_faces_total, base offset already applied) and the streamed
// mode (stride = batch_size) the SIMD-batch solver uses.
// ---------------------------------------------------------------------------
// `seed_column` is float64 and laid out [layer * seed_stride + f_local] --
// the SIMD-batch solver's own `batch.T`, read directly rather than through a
// float32 copy, so the digest sees the bits the solver will integrate.
//
// ⛔ Only the first `batch_n` lanes of each row are folded. The lanes past
// batch_n are SIMD padding holding whatever the previous batch left in them; a
// digest that included them would differ between two identical runs and make
// every resume fail at random.
inline std::uint32_t inputDigest(const double* flux,
                                 std::size_t flux_stride,
                                 std::size_t n_time_steps,
                                 std::size_t batch_n,
                                 const double* seed_column,
                                 std::size_t seed_stride,
                                 std::size_t n_layers,
                                 const double* face_qb) {   // may be null
    std::uint32_t crc = state_io::crc32Init();
    if (flux) {
        for (std::size_t t = 0; t < n_time_steps; ++t)
            crc = state_io::crc32Update(crc, flux + t * flux_stride,
                                        batch_n * sizeof(double));
    }
    if (seed_column) {
        for (std::size_t l = 0; l < n_layers; ++l)
            crc = state_io::crc32Update(crc, seed_column + l * seed_stride,
                                        batch_n * sizeof(double));
    }
    if (face_qb)
        crc = state_io::crc32Update(crc, face_qb, batch_n * sizeof(double));
    return state_io::crc32Final(crc);
}

// ---------------------------------------------------------------------------
// Bytes one partial set will occupy, for the startup disclosure and the
// ceiling refusal. Exact, not an estimate: it is the same arithmetic write()
// lays down.
// ---------------------------------------------------------------------------
inline std::uint64_t bytesPerFace(const Identity& id) {
    std::uint64_t b = 0;
    if (id.sections & part::kSurfaceSeries)
        b += static_cast<std::uint64_t>(id.n_time_steps) * 8u;
    if (id.sections & part::kFinalColumn)
        b += static_cast<std::uint64_t>(id.n_layers) * 4u;
    if (id.sections & part::kDepthAccum)
        b += static_cast<std::uint64_t>(id.n_depths) * 24u + 4u;
    return b;
}

inline std::uint64_t estimateBytes(const Identity& id) {
    return bytesPerFace(id) * static_cast<std::uint64_t>(id.n_faces_total);
}

inline std::string humanBytes(std::uint64_t b) {
    const char* u[] = {"B", "kiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(b);
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", v, u[i]);
    return buf;
}

// ---------------------------------------------------------------------------
// SERIALIZE / WRITE
// ---------------------------------------------------------------------------
namespace detail {

inline void putIdentity(std::vector<unsigned char>& b, const Identity& id,
                        std::uint32_t n_records, std::uint32_t header_bytes) {
    using namespace state_io::detail;
    putU32(b, kMagic);
    putU32(b, kVersion);
    putU32(b, header_bytes);
    putU32(b, id.sections);
    putU64(b, id.hash());
    putU32(b, n_records);
    putU32(b, id.n_faces_total);
    putU32(b, id.n_layers);
    putU32(b, id.n_time_steps);
    putU32(b, id.n_depths);
    putU32(b, id.batch_size);
    for (double v : id.z) putF64(b, v);
    putF64(b, id.et_seconds);
    putF64(b, id.output_cadence_s);
    putF64(b, id.refgrid_H);
    putI32(b, id.grid_m);
    putI32(b, id.grid_n);
    putI32(b, id.grid_b);
    putI32(b, 0);
    putF64(b, id.mesh_geomhash);
    putU64(b, id.config_identity_hash);
    putU64(b, id.depth_plan_hash);
    putI32(b, id.depth_win_start);
    putI32(b, id.depth_win_end);
    putStr(b, id.body_name, kBodyNameBytes);
    putStr(b, id.engine_git_sha, kGitShaBytes);
}

// Bytes from `magic` through `engine_git_sha`, i.e. everything before the
// record table. Computed, never measured after the fact, so a serializer bug
// shows up as a refusal rather than as a silently shifted body.
inline std::uint32_t headerBytes(std::size_t n_layers) {
    return static_cast<std::uint32_t>(
        4 * 4                       // magic, version, header_bytes, flags
        + 8                         // run_identity
        + 4 * 6                     // n_records..batch_size
        + n_layers * 8              // z[]
        + 8 * 3                     // et, cadence, refgrid_H
        + 4 * 4                     // grid_m/n/b/reserved
        + 8                         // mesh_geomhash
        + 8                         // config_identity_hash
        + 8                         // depth_plan_hash
        + 4 + 4                     // depth_win_start, depth_win_end
        + kBodyNameBytes + kGitShaBytes);
}

}  // namespace detail

// Serialize a whole partial file into memory (the CRC is appended last).
// Returns false with `err` set if a record's payload does not match the shapes
// the identity promises -- a mis-sized buffer must never reach the disk.
inline bool serialize(const Identity& id, const std::vector<Record>& recs,
                      std::vector<unsigned char>& out, std::string& err) {
    err.clear();
    out.clear();
    if (id.z.size() != id.n_layers) {
        err = "identity carries " + std::to_string(id.z.size()) +
              " depth nodes but claims n_layers=" + std::to_string(id.n_layers);
        return false;
    }
    for (const Record& r : recs) {
        const std::size_t bn = r.batch_n;
        if (bn == 0 || bn > id.batch_size) {
            err = "record for batch " + std::to_string(r.batch_idx) +
                  " claims batch_n=" + std::to_string(bn) +
                  ", which is not in [1," + std::to_string(id.batch_size) + "]";
            return false;
        }
        auto need = [&](const char* what, std::size_t have, std::size_t want) {
            if (have == want) return true;
            err = std::string("record for batch ") +
                  std::to_string(r.batch_idx) + " section " + what + " holds " +
                  std::to_string(have) + " values, shape needs " +
                  std::to_string(want);
            return false;
        };
        // A section that IS selected must be exactly the right shape, and a
        // section that is NOT selected must be EMPTY. The second half matters
        // as much as the first: a record carrying depth data while `sections`
        // says there is none would have that data silently dropped at the
        // write and come back as zeros at the resume, with the file's own
        // length and checksum both agreeing.
        if (id.sections & part::kSurfaceSeries) {
            if (!need("surface", r.surface.size(), bn * id.n_time_steps))
                return false;
        } else if (!need("surface (not selected, must be empty)",
                         r.surface.size(), 0)) {
            return false;
        }
        if (id.sections & part::kFinalColumn) {
            if (!need("column", r.column.size(), bn * id.n_layers))
                return false;
        } else if (!need("column (not selected, must be empty)",
                         r.column.size(), 0)) {
            return false;
        }
        if (id.sections & part::kDepthAccum) {
            if (!need("depth_sum", r.depth_sum.size(), bn * id.n_depths) ||
                !need("depth_max", r.depth_max.size(), bn * id.n_depths) ||
                !need("depth_min", r.depth_min.size(), bn * id.n_depths) ||
                !need("depth_count", r.depth_count.size(), bn))
                return false;
        } else if (!need("depth_sum (not selected, must be empty)",
                         r.depth_sum.size(), 0) ||
                   !need("depth_max (not selected, must be empty)",
                         r.depth_max.size(), 0) ||
                   !need("depth_min (not selected, must be empty)",
                         r.depth_min.size(), 0) ||
                   !need("depth_count (not selected, must be empty)",
                         r.depth_count.size(), 0)) {
            return false;
        }
    }

    const std::uint32_t hb = detail::headerBytes(id.n_layers);
    detail::putIdentity(out, id, static_cast<std::uint32_t>(recs.size()), hb);
    if (out.size() != hb) {
        err = "internal: header serialized to " + std::to_string(out.size()) +
              " bytes, headerBytes() says " + std::to_string(hb);
        return false;
    }
    for (const Record& r : recs) {
        state_io::detail::putU32(out, r.batch_idx);
        state_io::detail::putU32(out, r.batch_n);
        state_io::detail::putU32(out, r.input_digest);
        state_io::detail::putU32(out, 0u);
    }
    auto blob = [&](const void* p, std::size_t n) {
        const unsigned char* c = static_cast<const unsigned char*>(p);
        out.insert(out.end(), c, c + n);
    };
    for (const Record& r : recs) {
        if (id.sections & part::kSurfaceSeries)
            blob(r.surface.data(), r.surface.size() * sizeof(double));
        if (id.sections & part::kFinalColumn)
            blob(r.column.data(), r.column.size() * sizeof(float));
        if (id.sections & part::kDepthAccum) {
            blob(r.depth_sum.data(), r.depth_sum.size() * sizeof(double));
            blob(r.depth_max.data(), r.depth_max.size() * sizeof(double));
            blob(r.depth_min.data(), r.depth_min.size() * sizeof(double));
            blob(r.depth_count.data(),
                 r.depth_count.size() * sizeof(std::uint32_t));
        }
    }
    const std::uint32_t crc =
        state_io::crc32Final(state_io::crc32Update(state_io::crc32Init(),
                                                   out.data(), out.size()));
    state_io::detail::putU32(out, crc);
    return true;
}

// Atomic write: temp + fsync + rename, the same discipline state_io::write got
// in phase 1A. A killed writer therefore leaves either the previous file or
// nothing at that path -- never a truncated partial that would restore a
// half-integrated batch.
inline bool write(const std::string& path, const Identity& id,
                  const std::vector<Record>& recs, std::string& err) {
    std::vector<unsigned char> buf;
    if (!serialize(id, recs, buf, err)) return false;

    std::string tmp;
    int fd = -1;
    const long pid = static_cast<long>(::getpid());
    for (int attempt = 0; attempt < 1024; ++attempt) {
        tmp = path + ".tmp-" + std::to_string(pid);
        if (attempt > 0) tmp += "-" + std::to_string(attempt);
        fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd >= 0) break;
        if (errno != EEXIST) {
            err = "cannot create the temporary file '" + tmp + "' for '" +
                  path + "': " + std::strerror(errno);
            return false;
        }
    }
    if (fd < 0) {
        err = "cannot create a temporary file beside '" + path + "'";
        return false;
    }
    auto fail = [&](const std::string& what) -> bool {
        err = what + " (writing '" + path + "' via '" + tmp + "')";
        if (fd >= 0) ::close(fd);
        ::unlink(tmp.c_str());
        return false;
    };
    std::string w_err;
    if (!state_io::detail::writeAll(fd, buf.data(), buf.size(), w_err))
        return fail(w_err);
    while (::fsync(fd) != 0) {
        if (errno == EINTR) continue;
        return fail(std::string("fsync failed: ") + std::strerror(errno));
    }
    if (::close(fd) != 0) {
        const std::string why = std::strerror(errno);
        fd = -1;
        return fail("close failed: " + why);
    }
    fd = -1;
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        const std::string why = std::strerror(errno);
        ::unlink(tmp.c_str());
        err = "cannot rename '" + tmp + "' onto '" + path + "': " + why;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// READ
// ---------------------------------------------------------------------------
// Every failure is a REFUSAL with a complete message. There is no "load what we
// can" mode: a partial that cannot be fully verified is a partial that would
// restore un-solved faces, and the whole feature exists to prevent that.
// ---------------------------------------------------------------------------
inline bool read(const std::string& path, Identity& id,
                 std::vector<Record>& recs, std::string& err) {
    err.clear();
    recs.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "cannot open '" + path + "'"; return false; }
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    if (buf.size() < 4 + detail::headerBytes(0)) {
        err = "'" + path + "' is " + std::to_string(buf.size()) +
              " bytes, too short to be a PTMB partial";
        return false;
    }
    state_io::detail::Cursor c{buf.data(), buf.size(), 0, true};
    std::uint32_t magic = 0, version = 0, header_bytes = 0, sections = 0;
    std::uint64_t run_identity = 0;
    std::uint32_t n_records = 0;
    c.take(&magic, 4);
    if (!c.ok || magic != kMagic) {
        char got[32];
        std::snprintf(got, sizeof(got), "0x%08X", magic);
        err = "'" + path + "' is not a PTMB batch partial (magic " + got +
              ", expected 0x424D5450). A thermal STATE checkpoint is PTMT and "
              "is a different format on purpose.";
        return false;
    }
    c.take(&version, 4);
    c.take(&header_bytes, 4);
    c.take(&sections, 4);
    c.take(&run_identity, 8);
    c.take(&n_records, 4);
    if (!c.ok) { err = "'" + path + "': truncated header"; return false; }
    if (version != kVersion) {
        err = "'" + path + "' is PTMB version " + std::to_string(version) +
              "; this engine reads version " + std::to_string(kVersion) +
              " only. Refusing rather than guessing at a layout.";
        return false;
    }
    id = Identity{};
    id.sections = sections;
    c.take(&id.n_faces_total, 4);
    c.take(&id.n_layers, 4);
    c.take(&id.n_time_steps, 4);
    c.take(&id.n_depths, 4);
    c.take(&id.batch_size, 4);
    if (!c.ok) { err = "'" + path + "': truncated shape block"; return false; }
    if (header_bytes != detail::headerBytes(id.n_layers)) {
        err = "'" + path + "': header_bytes=" + std::to_string(header_bytes) +
              " disagrees with the " + std::to_string(id.n_layers) +
              "-layer layout (" +
              std::to_string(detail::headerBytes(id.n_layers)) + ")";
        return false;
    }
    id.z.resize(id.n_layers);
    for (std::uint32_t i = 0; i < id.n_layers; ++i) c.take(&id.z[i], 8);
    c.take(&id.et_seconds, 8);
    c.take(&id.output_cadence_s, 8);
    c.take(&id.refgrid_H, 8);
    std::int32_t reserved0 = 0;
    c.take(&id.grid_m, 4);
    c.take(&id.grid_n, 4);
    c.take(&id.grid_b, 4);
    c.take(&reserved0, 4);
    c.take(&id.mesh_geomhash, 8);
    c.take(&id.config_identity_hash, 8);
    c.take(&id.depth_plan_hash, 8);
    c.take(&id.depth_win_start, 4);
    c.take(&id.depth_win_end, 4);
    char body[kBodyNameBytes + 1] = {0};
    char sha[kGitShaBytes + 1]    = {0};
    c.take(body, kBodyNameBytes);
    c.take(sha, kGitShaBytes);
    if (!c.ok) { err = "'" + path + "': truncated identity block"; return false; }
    id.body_name      = body;
    id.engine_git_sha = sha;

    if (id.hash() != run_identity) {
        err = "'" + path + "': the recorded run-identity fold does not match "
              "the identity fields in the same file. The file is internally "
              "inconsistent -- refusing.";
        return false;
    }

    // ⛔ BOUND BEFORE ALLOCATE. `n_records` is read off the disk and is not
    // covered by the identity fold; the whole-file CRC that would catch a
    // corrupt value is not verified until after the body is parsed. Without
    // this check a single flipped byte makes read() default-construct up to
    // 4.29e9 Records -- hundreds of gigabytes -- so the documented refusal
    // ("REFUSING to resume ... CRC32") becomes an allocation storm instead.
    // Measured, not reasoned about: code-review reproduced the hang.
    const std::uint64_t need_table =
        static_cast<std::uint64_t>(header_bytes) +
        static_cast<std::uint64_t>(n_records) * 16ull + 4ull;
    if (need_table > buf.size()) {
        err = "'" + path + "': the record table claims " +
              std::to_string(n_records) + " batches, which needs " +
              std::to_string(need_table) + " bytes before any payload, but the "
              "file is only " + std::to_string(buf.size()) +
              ". Refusing rather than allocating for a corrupt count.";
        return false;
    }
    recs.resize(n_records);
    for (std::uint32_t i = 0; i < n_records; ++i) {
        std::uint32_t reserved1 = 0;
        c.take(&recs[i].batch_idx, 4);
        c.take(&recs[i].batch_n, 4);
        c.take(&recs[i].input_digest, 4);
        c.take(&reserved1, 4);
    }
    if (!c.ok) { err = "'" + path + "': truncated record table"; return false; }

    for (std::uint32_t i = 0; i < n_records; ++i) {
        Record& r = recs[i];
        if (r.batch_n == 0 || r.batch_n > id.batch_size) {
            err = "'" + path + "': record " + std::to_string(i) +
                  " claims batch_n=" + std::to_string(r.batch_n) +
                  ", outside [1," + std::to_string(id.batch_size) + "]";
            return false;
        }
        if (id.sections & part::kSurfaceSeries) {
            r.surface.resize(static_cast<std::size_t>(r.batch_n) *
                             id.n_time_steps);
            c.take(r.surface.data(), r.surface.size() * sizeof(double));
        }
        if (id.sections & part::kFinalColumn) {
            r.column.resize(static_cast<std::size_t>(r.batch_n) * id.n_layers);
            c.take(r.column.data(), r.column.size() * sizeof(float));
        }
        if (id.sections & part::kDepthAccum) {
            const std::size_t nd =
                static_cast<std::size_t>(r.batch_n) * id.n_depths;
            r.depth_sum.resize(nd);
            r.depth_max.resize(nd);
            r.depth_min.resize(nd);
            r.depth_count.resize(r.batch_n);
            c.take(r.depth_sum.data(), nd * sizeof(double));
            c.take(r.depth_max.data(), nd * sizeof(double));
            c.take(r.depth_min.data(), nd * sizeof(double));
            c.take(r.depth_count.data(), r.batch_n * sizeof(std::uint32_t));
        }
    }
    if (!c.ok) { err = "'" + path + "': truncated body"; return false; }

    std::uint32_t crc_stored = 0;
    if (c.off + 4 != buf.size()) {
        err = "'" + path + "': " + std::to_string(buf.size()) +
              " bytes on disk, the layout accounts for " +
              std::to_string(c.off + 4) +
              ". A partial of the wrong LENGTH is refused, not truncated to fit.";
        return false;
    }
    c.take(&crc_stored, 4);
    const std::uint32_t crc_calc = state_io::crc32Final(
        state_io::crc32Update(state_io::crc32Init(), buf.data(),
                              buf.size() - 4));
    if (crc_calc != crc_stored) {
        char a[32], b[32];
        std::snprintf(a, sizeof(a), "0x%08X", crc_calc);
        std::snprintf(b, sizeof(b), "0x%08X", crc_stored);
        err = "'" + path + "': CRC32 " + std::string(a) + " does not match the "
              "recorded " + b + ". The file is corrupt, truncated, or a "
              "cloud-storage placeholder.";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// COVERAGE -- the completeness manifest, judged from the files, never claimed
// ---------------------------------------------------------------------------
// The plan requires that a restore refuse unless every face is covered exactly
// once, checked against n_faces rather than assumed from a file count. That is
// what Coverage does. It is also the thing a lying manifest cannot fake,
// because it is computed from the records that actually validated.
// ---------------------------------------------------------------------------
struct Coverage {
    std::map<std::uint32_t, std::uint32_t> batch_to_digest;  // idx -> digest
    std::map<std::uint32_t, std::string>   batch_to_file;
    std::uint64_t faces_covered = 0;
    std::vector<std::string> duplicates;   // batches claimed by two files

    bool has(std::uint32_t batch_idx) const {
        return batch_to_digest.find(batch_idx) != batch_to_digest.end();
    }
    // Add one file's records. A batch claimed twice with DIFFERENT payload
    // digests is a hard conflict; claimed twice with the same digest is a
    // duplicate write (a crash tail) and is reported but tolerated.
    void add(const std::string& file, const Identity& id,
             const std::vector<Record>& recs) {
        for (const Record& r : recs) {
            auto it = batch_to_digest.find(r.batch_idx);
            if (it != batch_to_digest.end()) {
                std::ostringstream os;
                os << "batch " << r.batch_idx << " appears in '"
                   << batch_to_file[r.batch_idx] << "' and in '" << file << "'"
                   << (it->second == r.input_digest
                           ? " with the same input digest"
                           : " WITH DIFFERENT INPUT DIGESTS");
                duplicates.push_back(os.str());
                continue;
            }
            batch_to_digest[r.batch_idx] = r.input_digest;
            batch_to_file[r.batch_idx]   = file;
            faces_covered += r.batch_n;
        }
        (void)id;
    }
    bool complete(std::uint64_t n_faces_total) const {
        return faces_covered == n_faces_total;
    }
};

// ---------------------------------------------------------------------------
// SESSION -- the run-time half: scan, validate, skip, stage, flush, manifest.
//
// Engine-free on purpose. It knows about files, records and coverage and
// nothing about meshes, solvers or OpenCL, so the whole restart protocol is
// gateable without a run.
//
// ⭐ THE MANIFEST IS AN INDEX, NOT THE AUTHORITY. Completeness is computed from
// the partials that actually validated (Coverage), never from what the manifest
// claims. A manifest that names a file which is missing, corrupt, or from
// another run is itself the failure and REFUSES the resume; a manifest that is
// merely BEHIND the scan is the normal crash tail (the last flush landed, the
// manifest rewrite did not) and is reported, not fatal. Making the JSON
// authoritative would put the one thing a crash can truncate in charge of
// deciding which faces are solved.
// ---------------------------------------------------------------------------
class Session {
  public:
    bool enabled() const { return enabled_; }
    const std::string& dir() const { return dir_; }
    std::uint64_t restoredFaces() const { return cov_.faces_covered; }
    std::size_t restoredBatches() const { return cov_.batch_to_digest.size(); }
    std::size_t filesWritten() const { return files_.size(); }

    // Open the partial directory for run `id`. Scans, validates every file it
    // finds, cross-checks the manifest, and leaves the session ready to answer
    // banked()/takeRecord(). Returns false with `err` set on any REFUSAL.
    // `flush_mib` of 0 DERIVES the flush size from the projected set (see
    // below); any positive value is used as given.
    bool open(const std::string& dir, const Identity& id, double max_gib,
              double flush_mib, std::ostream& log, std::string& err) {
        err.clear();
        dir_ = dir;
        id_  = id;
        enabled_ = true;

        const std::uint64_t projected = estimateBytes(id);
        // ⛔ A FIXED flush threshold makes the feature silently useless on any
        // run smaller than it: nothing reaches disk until the end, so a crash
        // loses everything while the log says checkpointing is on. Measured on
        // the shipped 12,800-face crater arm -- 23 MiB total against a 64 MiB
        // threshold, exactly ONE flush, at the end. So the default is DERIVED:
        // sixteen flushes, i.e. a crash costs at most ~1/16 of the window's
        // completed work, clamped so a tiny run does not thrash and a
        // production run does not hold gigabytes of staged records in RAM.
        if (flush_mib > 0.0) {
            flush_bytes_ = static_cast<std::uint64_t>(flush_mib * 1024.0 *
                                                      1024.0);
            if (flush_bytes_ == 0) flush_bytes_ = 1;
        } else {
            const std::uint64_t derived = projected / 16u;
            const std::uint64_t lo = 1ull * 1024 * 1024;
            const std::uint64_t hi = 64ull * 1024 * 1024;
            flush_bytes_ = derived < lo ? lo : (derived > hi ? hi : derived);
        }
        const std::uint64_t faces_per_flush =
            bytesPerFace(id) ? (flush_bytes_ / bytesPerFace(id)) : 0;

        log << "[batch-ckpt] ENABLED. Partials: " << dir_ << "\n"
            << "[batch-ckpt] projected partial-set size: "
            << humanBytes(projected) << " (" << bytesPerFace(id)
            << " B/face x " << id.n_faces_total << " faces)\n"
            << "[batch-ckpt] a RESUME holds the restored payload in memory "
               "until the solver consumes it, so that same\n"
               "             figure is also the peak extra RSS of a resume -- "
               "on top of surface_temps, not instead of it.\n"
            << "[batch-ckpt] flush every " << humanBytes(flush_bytes_)
            << " (~" << faces_per_flush << " faces), so an abrupt kill costs "
               "at most that much\n"
            << "             completed work plus whatever was in flight.\n"
            << "[batch-ckpt] granularity is ONE FACE BATCH (" << id.batch_size
            << " faces). A run that dies INSIDE a batch loses that batch's\n"
            << "             whole time integration -- for finer bounding use "
               "the runner's chunk split.\n";
        if (max_gib > 0.0) {
            const double gib = static_cast<double>(projected) /
                               (1024.0 * 1024.0 * 1024.0);
            if (gib > max_gib) {
                std::ostringstream os;
                os.precision(3);
                os << "thermal.batch_checkpoint would write " << std::fixed
                   << gib << " GiB of partials (" << humanBytes(projected)
                   << "), over the thermal.batch_checkpoint_max_gib ceiling of "
                   << max_gib << " GiB. That figure bounds BOTH the disk the "
                      "partials occupy AND the peak memory a resume holds "
                      "while restoring them. Refusing rather than filling "
                      "either quietly. Raise the ceiling deliberately, or "
                      "leave blast-radius bounding to the runner's chunk "
                      "split.";
                err = os.str();
                return false;
            }
        }

        if (!ensureDir(dir_, err)) return false;

        std::vector<std::string> found;
        if (!listPartials(dir_, found, err)) return false;
        std::sort(found.begin(), found.end());

        std::set<std::string> validated;
        for (const std::string& name : found) {
            const std::string path = dir_ + "/" + name;
            Identity fid;
            std::vector<Record> recs;
            std::string rerr;
            if (!read(path, fid, recs, rerr)) {
                err = "REFUSING to resume: " + rerr +
                      "\n  A partial that cannot be fully verified would "
                      "restore un-solved or wrong faces into a finished "
                      "product, which nothing downstream can detect. Delete "
                      "the partial directory to start the window over.";
                return false;
            }
            const std::string mm = id_.describeMismatch(fid);
            if (!mm.empty()) {
                err = "REFUSING to resume: '" + path +
                      "' was written by a DIFFERENT computation:\n" + mm +
                      "  A partial is only reusable by a run with the same "
                      "mesh, grid, epoch, window and physics.";
                return false;
            }
            gen_ = std::max(gen_, generationOf(name));
            cov_.add(name, fid, recs);
            for (Record& r : recs)
                loaded_[r.batch_idx] = std::move(r);
            validated.insert(name);
        }
        ++gen_;

        // Manifest cross-check. Every file the manifest names must be one that
        // validated; the reverse is allowed and reported.
        std::vector<std::string> claimed;
        if (readManifestFiles(manifestPath(), claimed)) {
            for (const std::string& c : claimed) {
                if (validated.find(c) == validated.end()) {
                    err = "REFUSING to resume: the completeness manifest '" +
                          manifestPath() + "' names '" + c +
                          "', which is not present or did not validate. The "
                          "manifest is an INDEX; when it disagrees with the "
                          "partials on disk, the partials are the truth and "
                          "the disagreement is the failure.";
                    return false;
                }
            }
            if (claimed.size() < validated.size()) {
                log << "[batch-ckpt] the manifest lists " << claimed.size()
                    << " partial(s) but " << validated.size()
                    << " validated on disk -- the extra file(s) are a crash "
                       "tail (a flush landed, the manifest rewrite did not). "
                       "Using them.\n";
            }
        } else if (!found.empty()) {
            log << "[batch-ckpt] no readable manifest beside " << found.size()
                << " validated partial(s); rebuilding it from the files.\n";
        }

        for (const std::string& d : cov_.duplicates)
            log << "[batch-ckpt] duplicate: " << d << "\n";
        for (const std::string& d : cov_.duplicates) {
            if (d.find("DIFFERENT INPUT DIGESTS") != std::string::npos) {
                err = "REFUSING to resume: " + d +
                      ". Two different integrations of one face batch cannot "
                      "both be right, and nothing here can say which is.";
                return false;
            }
        }

        if (!found.empty()) {
            log << "[batch-ckpt] RESUMING: " << cov_.batch_to_digest.size()
                << " batch(es) / " << cov_.faces_covered << " face(s) of "
                << id.n_faces_total << " restored from "
                << validated.size() << " partial(s); the rest will be "
                   "integrated.\n";
        } else {
            // Said out loud, because the alternative is the silent-substitution
            // class: a MISTYPED batch_checkpoint_dir would otherwise create an
            // empty directory, integrate the whole mesh again, and report
            // nothing to distinguish that from a resume that found its work.
            log << "[batch-ckpt] COLD START: no partials found at " << dir_
                << " -- integrating every batch. If you expected to resume, "
                   "this is the\n"
                   "             wrong directory.\n";
        }
        for (const std::string& n : validated) files_.push_back(n);
        return true;
    }

    // Was this batch banked by an earlier process? `digest_out` receives the
    // input digest that was recorded with it, which the caller MUST compare
    // against the digest of its own live inputs before trusting the payload.
    // ⛔ UNDER THE LOCK. These are called by every solver thread from inside
    // `#pragma omp for`, while any thread that crosses the flush threshold is
    // INSERTING into the same maps from flush(). A std::map find() racing an
    // insert() is undefined behavior, and its benign-looking outcome -- a false
    // "not banked" -- would re-integrate and re-bank a batch that was already
    // done. Found by code-review; the window widens exactly as the flush size
    // is lowered, which is the configuration the crash gate uses.
    bool banked(std::uint32_t batch_idx, std::uint32_t* digest_out) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = cov_.batch_to_digest.find(batch_idx);
        if (it == cov_.batch_to_digest.end()) return false;
        if (digest_out) *digest_out = it->second;
        return true;
    }

    std::string fileFor(std::uint32_t batch_idx) const {
        std::lock_guard<std::mutex> lk(mu_);
        static const std::string none;
        auto it = cov_.batch_to_file.find(batch_idx);
        return (it == cov_.batch_to_file.end()) ? none : it->second;
    }

    // Move a restored record out. Thread-safe; the payload is released from the
    // session as it is consumed so a resume does not hold two copies.
    bool takeRecord(std::uint32_t batch_idx, Record& out) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = loaded_.find(batch_idx);
        if (it == loaded_.end()) return false;
        out = std::move(it->second);
        loaded_.erase(it);
        return true;
    }

    // Bank a completed batch. Thread-safe. Flushes to disk when the staged set
    // crosses the flush threshold; the serialization and the write happen
    // OUTSIDE the lock so the solver threads are not serialized behind I/O.
    bool stage(Record&& rec, std::string& err) {
        std::vector<Record> to_write;
        {
            std::lock_guard<std::mutex> lk(mu_);
            staged_bytes_ += recordBytes(rec);
            staged_.push_back(std::move(rec));
            if (staged_bytes_ < flush_bytes_) return true;
            to_write.swap(staged_);
            staged_bytes_ = 0;
        }
        return flush(to_write, err);
    }

    // Flush whatever is staged, rewrite the manifest, and verify that every
    // face is covered exactly once -- against n_faces, never against a file
    // count. This is the plan's completeness rule and it is checked here even
    // on an uninterrupted run, because a coverage hole is exactly what a silent
    // partial-restore defect would look like.
    bool finish(std::ostream& log, std::string& err) {
        std::vector<Record> to_write;
        {
            std::lock_guard<std::mutex> lk(mu_);
            to_write.swap(staged_);
            staged_bytes_ = 0;
        }
        if (!to_write.empty() && !flush(to_write, err)) return false;
        if (!cov_.complete(id_.n_faces_total)) {
            std::ostringstream os;
            os << "batch-checkpoint coverage is INCOMPLETE at the end of the "
                  "solve: " << cov_.faces_covered << " of "
               << id_.n_faces_total << " faces are accounted for across "
               << cov_.batch_to_digest.size() << " batch(es). Every face must "
                  "be covered exactly once or the partial set cannot be "
                  "trusted to reconstruct the run.";
            err = os.str();
            return false;
        }
        log << "[batch-ckpt] complete: " << cov_.batch_to_digest.size()
            << " batch(es), " << cov_.faces_covered << "/"
            << id_.n_faces_total << " faces, " << files_.size()
            << " partial file(s) in " << dir_ << "\n";
        return true;
    }

    // A RESTORED batch needs no bookkeeping here: open() already folded it into
    // the coverage from its file, which is why finish() can check the whole
    // mesh with one rule regardless of how many processes it took.

    void setFlushBytes(std::uint64_t b) { flush_bytes_ = b; }

  private:
    static std::uint64_t recordBytes(const Record& r) {
        return r.surface.size() * 8 + r.column.size() * 4 +
               (r.depth_sum.size() + r.depth_max.size() + r.depth_min.size()) *
                   8 +
               r.depth_count.size() * 4 + 16;
    }

    std::string manifestPath() const { return dir_ + "/manifest.json"; }

    bool flush(std::vector<Record>& recs, std::string& err) {
        std::sort(recs.begin(), recs.end(),
                  [](const Record& a, const Record& b) {
                      return a.batch_idx < b.batch_idx;
                  });
        char name[64];
        const unsigned seq = seq_.fetch_add(1u);
        std::snprintf(name, sizeof(name), "part_g%04u_s%06u.ptmb", gen_, seq);
        const std::string path = dir_ + "/" + name;
        if (!write(path, id_, recs, err)) return false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            cov_.add(name, id_, recs);
            files_.push_back(name);
        }
        std::string merr;
        if (!writeManifest(merr)) {
            // The manifest is an index; failing to refresh it must not lose a
            // partial that IS on disk. Say so rather than swallowing it.
            err = "the partial '" + path +
                  "' was written but the manifest could not be refreshed: " +
                  merr;
            return false;
        }
        return true;
    }

    bool writeManifest(std::string& err) {
        std::ostringstream os;
        std::vector<std::string> snapshot;
        std::uint64_t faces = 0;
        std::size_t batches = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            snapshot = files_;
            faces = cov_.faces_covered;
            batches = cov_.batch_to_digest.size();
        }
        os << "{\n  \"schema\": \"ptm3d-batch-partial-manifest/1\",\n";
        os << "  \"note\": \"An INDEX, not the authority. Completeness is "
              "recomputed from the partials themselves on every resume; a "
              "manifest that names a file which does not validate REFUSES the "
              "resume, and a manifest that is merely behind the scan is a "
              "crash tail.\",\n";
        os << "  \"n_faces_total\": " << id_.n_faces_total << ",\n";
        os << "  \"batch_size\": " << id_.batch_size << ",\n";
        os << "  \"n_layers\": " << id_.n_layers << ",\n";
        os << "  \"n_time_steps\": " << id_.n_time_steps << ",\n";
        os << "  \"n_depths\": " << id_.n_depths << ",\n";
        os << "  \"sections\": " << id_.sections << ",\n";
        os << "  \"run_identity\": " << id_.hash() << ",\n";
        os << "  \"batches_banked\": " << batches << ",\n";
        os << "  \"faces_banked\": " << faces << ",\n";
        os << "  \"complete\": "
           << ((faces == id_.n_faces_total) ? "true" : "false") << ",\n";
        os << "  \"files\": [";
        for (std::size_t i = 0; i < snapshot.size(); ++i)
            os << (i ? ", " : "") << "\"" << snapshot[i] << "\"";
        os << "]\n}\n";
        const std::string body = os.str();

        const std::string path = manifestPath();
        // ⛔ O_EXCL AND A PER-CALL NAME. flush() deliberately runs OUTSIDE the
        // mutex so solver threads are not serialized behind I/O, so two
        // manifest refreshes can be in flight at once. With a per-PROCESS temp
        // name and O_TRUNC the second writer truncated the first's bytes and
        // the rename shipped a torn JSON -- whose "files" array the next resume
        // would read as a partial that "did not validate", refusing a perfectly
        // good partial set. The PTMB writer got this right; this one had not.
        std::string tmp;
        int fd = -1;
        const unsigned tag = manifest_seq_.fetch_add(1u);
        for (int attempt = 0; attempt < 1024; ++attempt) {
            tmp = path + ".tmp-" +
                  std::to_string(static_cast<long>(::getpid())) + "-" +
                  std::to_string(tag);
            if (attempt > 0) tmp += "-" + std::to_string(attempt);
            fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
            if (fd >= 0) break;
            if (errno != EEXIST) break;
        }
        if (fd < 0) {
            err = std::string("cannot create '") + tmp + "': " +
                  std::strerror(errno);
            return false;
        }
        std::string werr;
        if (!state_io::detail::writeAll(fd, body.data(), body.size(), werr)) {
            ::close(fd);
            ::unlink(tmp.c_str());
            err = werr;
            return false;
        }
        while (::fsync(fd) != 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            ::unlink(tmp.c_str());
            err = std::string("fsync failed: ") + std::strerror(errno);
            return false;
        }
        ::close(fd);
        if (::rename(tmp.c_str(), path.c_str()) != 0) {
            err = std::string("cannot rename onto '") + path + "': " +
                  std::strerror(errno);
            ::unlink(tmp.c_str());
            return false;
        }
        return true;
    }

    // Deliberately a hand-rolled scan of the "files" array: the manifest exists
    // to be cross-checked, and pulling a JSON dependency into the thermal
    // library to read eleven scalars would be a worse trade.
    static bool readManifestFiles(const std::string& path,
                                  std::vector<std::string>& out) {
        out.clear();
        std::ifstream in(path);
        if (!in) return false;
        std::string all((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
        const std::size_t k = all.find("\"files\"");
        if (k == std::string::npos) return false;
        const std::size_t lb = all.find('[', k);
        const std::size_t rb = all.find(']', lb == std::string::npos ? k : lb);
        if (lb == std::string::npos || rb == std::string::npos) return false;
        std::size_t i = lb + 1;
        while (i < rb) {
            const std::size_t q1 = all.find('"', i);
            if (q1 == std::string::npos || q1 > rb) break;
            const std::size_t q2 = all.find('"', q1 + 1);
            if (q2 == std::string::npos || q2 > rb) break;
            out.push_back(all.substr(q1 + 1, q2 - q1 - 1));
            i = q2 + 1;
        }
        return true;
    }

    static unsigned generationOf(const std::string& name) {
        // part_gNNNN_sMMMMMM.ptmb
        const std::size_t k = name.find("_g");
        if (k == std::string::npos) return 0u;
        return static_cast<unsigned>(std::strtoul(name.c_str() + k + 2,
                                                  nullptr, 10));
    }

    static bool ensureDir(const std::string& d, std::string& err) {
        // One level at a time, so a missing parent is created too. mkdir(2)
        // returning EEXIST is success here.
        std::string acc;
        for (std::size_t i = 0; i <= d.size(); ++i) {
            if (i == d.size() || d[i] == '/') {
                if (!acc.empty() && ::mkdir(acc.c_str(), 0777) != 0 &&
                    errno != EEXIST) {
                    err = "cannot create the partial directory '" + acc +
                          "': " + std::strerror(errno);
                    return false;
                }
            }
            if (i < d.size()) acc += d[i];
        }
        return true;
    }

    static bool listPartials(const std::string& d,
                             std::vector<std::string>& out, std::string& err) {
        out.clear();
        DIR* dp = ::opendir(d.c_str());
        if (!dp) {
            err = "cannot read the partial directory '" + d +
                  "': " + std::strerror(errno);
            return false;
        }
        while (struct dirent* e = ::readdir(dp)) {
            const std::string n = e->d_name;
            if (n.size() > 5 && n.compare(n.size() - 5, 5, ".ptmb") == 0)
                out.push_back(n);
        }
        ::closedir(dp);
        return true;
    }

    bool enabled_ = false;
    std::string dir_;
    Identity id_;
    Coverage cov_;
    std::map<std::uint32_t, Record> loaded_;
    std::vector<std::string> files_;
    std::vector<Record> staged_;
    std::uint64_t staged_bytes_ = 0;
    std::uint64_t flush_bytes_ = 64ull * 1024 * 1024;
    unsigned gen_ = 0;
    std::atomic<unsigned> seq_{0};
    std::atomic<unsigned> manifest_seq_{0};
    mutable std::mutex mu_;
};

}  // namespace batchckpt
}  // namespace thermal

#endif  // PTM_THERMAL_BATCH_CHECKPOINT_H
