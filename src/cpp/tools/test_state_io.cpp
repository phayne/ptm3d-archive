// test_state_io -- gate for the thermal-state checkpoint ("PTMT") format and
// for Model::restoreState / Model::captureState (BUGS.md PTM-76).
//
// What this gate is for. thermal.initial_state_file / thermal.final_state_file
// were read and written by the GPU persistent solver only; on any CPU path both
// keys were dropped with no file and no error. The CPU drivers now carry the
// same format, which means the format itself, the round trip, and the refusal
// behavior all need a test that does not require a GPU.
//
// It checks:
//   G1  round trip -- write from a float64 CPU column, read it back, and the
//       restored column is BIT-IDENTICAL to what was written (and the file
//       bytes are reproducible).
//   G2  the float32 boundary is exactly one cast at the write, stated as a
//       measured bound rather than hidden in a tolerance.
//   G4  every failure mode refuses with a message: missing file, bad magic,
//       truncated body, wrong face count, wrong layer count, wrong buffer size
//       on write, and a restoreState onto a mismatched grid.
//   G5  the write is ATOMIC (BUGS.md PTM-STATE-ATOMIC): a writer killed
//       mid-stream leaves the PREVIOUS checkpoint intact and loadable, and a
//       write that fails leaves the destination untouched and no temp behind.
//   G6  probe() answers "what is this file" without being told the shape, and
//       refuses the same malformed inputs read() does.
//   G7  the PTMT v2 IDENTITY TRAILER round-trips every field, its CRC32 covers
//       the whole file (so a single flipped BODY byte is caught), a v1 file
//       still loads and reports itself as v1, and a malformed trailer refuses
//       rather than being read as sentinels.
//   G8  checkRestore() -- the refusal semantics. Wrong mesh, wrong physics,
//       wrong body, wrong grid, wrong depth grid, wrong epoch and wrong season
//       each refuse INDIVIDUALLY and name themselves; a one-cadence-stale
//       hand-off warns instead of refusing; a tiling hand-off passes clean; the
//       named per-field opt-out downgrades exactly the field it names and
//       cannot reach the partial-state refusal at all.

#include "../thermal/thermal_state_io.h"
#include "../thermal/thermal_model.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <csignal>
#include <dirent.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::cout << "  FAIL  " << what << std::endl;
    } else {
        std::cout << "  ok    " << what << std::endl;
    }
}

std::string tmpPath(const char* stem) {
    // A test artifact, not a result: /tmp is the right place for it and it is
    // removed on the way out.
    return std::string("/tmp/ptm_test_state_io_") + stem + ".ptmt";
}

std::vector<char> readAllBytes(const std::string& p) {
    std::ifstream is(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(is)),
                             std::istreambuf_iterator<char>());
}

// Names in `dir` that start with `stem` -- used to count the temp files an
// atomic write creates and, when a writer is killed, leaves behind. Sorted so
// the report is reproducible.
std::vector<std::string> namesStartingWith(const std::string& dir,
                                           const std::string& stem) {
    std::vector<std::string> hits;
    DIR* d = ::opendir(dir.c_str());
    if (!d) return hits;
    while (struct dirent* e = ::readdir(d)) {
        const std::string n(e->d_name);
        if (n.size() > stem.size() && n.compare(0, stem.size(), stem) == 0)
            hits.push_back(dir + "/" + n);
    }
    ::closedir(d);
    std::sort(hits.begin(), hits.end());
    return hits;
}

}  // namespace

int main() {
    std::cout << "=== test_state_io: thermal-state checkpoint format (PTM-76) ==="
              << std::endl;

    const std::size_t NF = 37;    // deliberately not a multiple of the SIMD width
    const std::size_t NL = 30;

    // A field with structure in BOTH indices, so a layer/face transposition
    // cannot pass: T = 20 + f*0.5 + l*3.25, plus a subnormal-adjacent value and
    // a value that is not exactly representable in float32.
    std::vector<double> src64(NF * NL);
    for (std::size_t l = 0; l < NL; ++l)
        for (std::size_t f = 0; f < NF; ++f)
            src64[l * NF + f] = 20.0 + 0.5 * f + 3.25 * l + 1.0 / 3.0;

    std::vector<float> src32(NF * NL);
    for (std::size_t i = 0; i < src32.size(); ++i)
        src32[i] = static_cast<float>(src64[i]);

    // ---- G1: round trip is bit-identical --------------------------------
    {
        const std::string p = tmpPath("roundtrip");
        std::string err;
        check(thermal::state_io::write(p, NF, NL, src32, nullptr, err),
              "write() succeeds (" + err + ")");

        std::vector<float> back;
        check(thermal::state_io::read(p, NF, NL, back, err),
              "read() succeeds (" + err + ")");
        check(back.size() == src32.size(), "read() returns n_faces*n_layers values");

        bool bit_identical = (back.size() == src32.size());
        for (std::size_t i = 0; i < back.size() && bit_identical; ++i)
            bit_identical = (std::memcmp(&back[i], &src32[i], sizeof(float)) == 0);
        check(bit_identical, "G1: restored column is BIT-IDENTICAL to what was written");

        // The file itself is reproducible: writing the same buffer twice gives
        // the same bytes. (A timestamp or padding sneaking in would show here.)
        const std::string p2 = tmpPath("roundtrip2");
        thermal::state_io::write(p2, NF, NL, src32, nullptr, err);
        check(readAllBytes(p) == readAllBytes(p2),
              "G1: two writes of the same buffer produce identical bytes");

        // Header is exactly 12 bytes and the body is exactly float32.
        check(readAllBytes(p).size() == 12 + NF * NL * sizeof(float),
              "G1: file size == 12-byte header + n_faces*n_layers float32");

        std::remove(p.c_str());
        std::remove(p2.c_str());
    }

    // ---- G2: the float32 boundary, measured -----------------------------
    {
        const std::string p = tmpPath("f32");
        std::string err;
        thermal::state_io::write(p, NF, NL, src32, nullptr, err);
        std::vector<float> back;
        thermal::state_io::read(p, NF, NL, back, err);

        double max_abs = 0.0, max_rel_ulp = 0.0;
        for (std::size_t i = 0; i < back.size(); ++i) {
            const double d = std::fabs(static_cast<double>(back[i]) - src64[i]);
            if (d > max_abs) max_abs = d;
            const double ulp = std::fabs(src64[i]) *
                               std::numeric_limits<float>::epsilon();
            if (ulp > 0.0 && d / ulp > max_rel_ulp) max_rel_ulp = d / ulp;
        }
        std::cout << "        float64 source vs restored: max |dT| = " << max_abs
                  << " K, max " << max_rel_ulp << " float32 ulp" << std::endl;
        // One rounding, at the write. Not a tolerance on the solver -- a bound
        // on the format.
        check(max_rel_ulp <= 0.5 + 1e-12,
              "G2: the ONLY loss is a single float32 rounding at the write "
              "(<= 0.5 ulp)");

        // And the restored float32 is exact when widened back to float64.
        bool exact_widen = true;
        for (std::size_t i = 0; i < back.size() && exact_widen; ++i)
            exact_widen = (static_cast<double>(back[i]) ==
                           static_cast<double>(src32[i]));
        check(exact_widen, "G2: float32 -> float64 widening on restore is exact");

        std::remove(p.c_str());
    }

    // ---- G4: every failure refuses, with a reason -----------------------
    {
        std::string err;
        std::vector<float> back;

        check(!thermal::state_io::read("/tmp/ptm_no_such_state_file.ptmt",
                                       NF, NL, back, err) && !err.empty(),
              "G4: missing file refuses -- \"" + err + "\"");

        // Bad magic.
        const std::string pbad = tmpPath("badmagic");
        {
            std::ofstream os(pbad, std::ios::binary);
            const std::uint32_t junk = 0xDEADBEEFu;
            const std::uint32_t nf = static_cast<std::uint32_t>(NF);
            const std::uint32_t nl = static_cast<std::uint32_t>(NL);
            os.write(reinterpret_cast<const char*>(&junk), 4);
            os.write(reinterpret_cast<const char*>(&nf), 4);
            os.write(reinterpret_cast<const char*>(&nl), 4);
            std::vector<float> body(NF * NL, 100.0f);
            os.write(reinterpret_cast<const char*>(body.data()),
                     static_cast<std::streamsize>(body.size() * sizeof(float)));
        }
        check(!thermal::state_io::read(pbad, NF, NL, back, err) && !err.empty(),
              "G4: bad magic refuses -- \"" + err + "\"");
        std::remove(pbad.c_str());

        // Wrong face count and wrong layer count.
        const std::string pgood = tmpPath("shape");
        thermal::state_io::write(pgood, NF, NL, src32, nullptr, err);
        check(!thermal::state_io::read(pgood, NF + 1, NL, back, err) && !err.empty(),
              "G4: wrong face count refuses -- \"" + err + "\"");
        check(!thermal::state_io::read(pgood, NF, NL + 1, back, err) && !err.empty(),
              "G4: wrong layer count refuses -- \"" + err + "\"");

        // Truncated body.
        const std::string ptrunc = tmpPath("trunc");
        {
            std::vector<char> bytes = readAllBytes(pgood);
            bytes.resize(bytes.size() - 4 * sizeof(float));
            std::ofstream os(ptrunc, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        check(!thermal::state_io::read(ptrunc, NF, NL, back, err) && !err.empty(),
              "G4: truncated body refuses -- \"" + err + "\"");
        std::remove(ptrunc.c_str());

        // A header shorter than 12 bytes.
        const std::string pshort = tmpPath("shorthdr");
        { std::ofstream os(pshort, std::ios::binary); os.write("PTM", 3); }
        check(!thermal::state_io::read(pshort, NF, NL, back, err) && !err.empty(),
              "G4: short header refuses -- \"" + err + "\"");
        std::remove(pshort.c_str());

        // Writer refuses a buffer that does not match the header it would claim.
        std::vector<float> wrong(NF * NL - 1, 0.0f);
        check(!thermal::state_io::write(tmpPath("badbuf"), NF, NL, wrong, nullptr, err) &&
                  !err.empty(),
              "G4: write() refuses a mis-sized buffer -- \"" + err + "\"");

        std::remove(pgood.c_str());
    }

    // ---- G5: the write is ATOMIC (PTM-STATE-ATOMIC) ----------------------
    //
    // The falsifier. Before the temp-then-rename, write() opened the
    // destination with ios::trunc, so the PREVIOUS checkpoint was destroyed the
    // instant a new write began. Every arm below fails on that writer.
    {
        // Big enough that an interrupted write is unambiguously partial, small
        // enough to be instant: 4096 faces x 32 layers = 524,300 bytes.
        const std::size_t AF = 4096, AL = 32;
        const std::string p = tmpPath("atomic");
        const std::string dir = "/tmp";
        const std::string stem = "ptm_test_state_io_atomic.ptmt.tmp-";

        std::vector<float> A(AF * AL), B(AF * AL);
        for (std::size_t i = 0; i < A.size(); ++i) {
            A[i] = 111.0f + static_cast<float>(i % 977);   // the banked state
            B[i] = -1.0f  - static_cast<float>(i % 977);   // the replacement
        }

        std::string err;
        check(thermal::state_io::write(p, AF, AL, A, nullptr, err),
              "G5: banked a prior checkpoint to interrupt over (" + err + ")");
        const std::vector<char> bytes_A = readAllBytes(p);

        // A completed write leaves NO temp behind.
        check(namesStartingWith(dir, stem).empty(),
              "G5: a COMPLETED write leaves no temporary file behind");

        // --- G5a: kill the writer mid-stream, deterministically -----------
        // RLIMIT_FSIZE makes the kill deterministic: the kernel raises SIGXFSZ
        // the moment the file would exceed the limit, which is past the 12-byte
        // header and far short of the 524,300-byte body. No sleep, no race.
        const rlim_t kLimit = 65536;
        int wstatus = 0;
        const pid_t kid = ::fork();
        if (kid == 0) {
            // CHILD. No iostreams, no exit() -- a forked child must not flush
            // the parent's buffers.
            ::signal(SIGXFSZ, SIG_DFL);
            struct rlimit rl;
            rl.rlim_cur = kLimit;
            rl.rlim_max = kLimit;
            if (::setrlimit(RLIMIT_FSIZE, &rl) != 0) _exit(97);
            std::string cerr_unused;
            const bool ok = thermal::state_io::write(p, AF, AL, B, nullptr, cerr_unused);
            // Reaching here at all means the limit did not bite. Report which
            // way, so the parent can fail LOUDLY rather than pass vacuously.
            _exit(ok ? 98 : 99);
        }
        check(kid > 0, "G5a: forked a writer to interrupt");
        if (kid > 0) {
            while (::waitpid(kid, &wstatus, 0) < 0 && errno == EINTR) {}

            // ANTI-VACUITY. If the child was not killed by SIGXFSZ then no
            // write was interrupted and everything below proves nothing.
            const bool killed = WIFSIGNALED(wstatus) &&
                                WTERMSIG(wstatus) == SIGXFSZ;
            if (!killed) {
                std::cout << "        child status: "
                          << (WIFEXITED(wstatus)
                                  ? ("exited " + std::to_string(WEXITSTATUS(wstatus)))
                                  : ("signal " + std::to_string(WTERMSIG(wstatus))))
                          << " (98 = the write COMPLETED, 99 = it refused, "
                             "97 = setrlimit failed)" << std::endl;
            }
            check(killed,
                  "G5a: the writer really was killed MID-WRITE by SIGXFSZ "
                  "(without this the arm below is vacuous)");

            // THE ASSERTION. The previous checkpoint is still there, byte for
            // byte, and still loads.
            check(readAllBytes(p) == bytes_A,
                  "G5a: the PRIOR checkpoint survives the killed write "
                  "BYTE-IDENTICALLY");
            std::vector<float> back;
            const bool loaded =
                thermal::state_io::read(p, AF, AL, back, err);
            check(loaded, "G5a: the prior checkpoint still LOADS (" + err + ")");
            bool same = loaded && back.size() == A.size();
            for (std::size_t i = 0; i < A.size() && same; ++i)
                same = (std::memcmp(&back[i], &A[i], sizeof(float)) == 0);
            check(same, "G5a: and it restores the ORIGINAL values, not the "
                        "half-written replacement");

            // A killed writer cannot clean up after itself. That is a real,
            // deliberate consequence, so it is asserted rather than hidden --
            // and its presence is second proof the kill landed inside the
            // open..rename window.
            const std::vector<std::string> orphans = namesStartingWith(dir, stem);
            check(orphans.size() == 1,
                  "G5a: exactly one '" + stem + "<pid>' orphan remains (a "
                  "killed writer cannot unlink its own temp) -- found " +
                  std::to_string(orphans.size()));
            for (const auto& o : orphans) std::remove(o.c_str());
        }

        // --- G5b: a write that FAILS leaves no trace ----------------------
        // Same forced failure, in-process: ignore SIGXFSZ so ::write returns
        // EFBIG instead of killing us. write() must refuse, leave the
        // destination untouched, and remove its own temp.
        {
            struct rlimit saved;
            const bool have_saved = (::getrlimit(RLIMIT_FSIZE, &saved) == 0);
            void (*saved_h)(int) = ::signal(SIGXFSZ, SIG_IGN);
            struct rlimit rl;
            rl.rlim_cur = kLimit;
            rl.rlim_max = have_saved ? saved.rlim_max : kLimit;
            const bool set_ok = (::setrlimit(RLIMIT_FSIZE, &rl) == 0);
            check(set_ok, "G5b: armed a forced write failure (RLIMIT_FSIZE)");

            std::string ferr;
            const bool wrote = set_ok
                ? thermal::state_io::write(p, AF, AL, B, nullptr, ferr) : true;

            if (have_saved) ::setrlimit(RLIMIT_FSIZE, &saved);
            ::signal(SIGXFSZ, saved_h);

            check(!wrote && !ferr.empty(),
                  "G5b: a write that cannot complete REFUSES -- \"" + ferr + "\"");
            check(readAllBytes(p) == bytes_A,
                  "G5b: the destination is untouched after the failed write");
            check(namesStartingWith(dir, stem).empty(),
                  "G5b: the failed write removed its own temporary file");
        }

        // --- G5c: an unwritable destination refuses and leaves nothing -----
        {
            std::string derr;
            check(!thermal::state_io::write(
                      "/tmp/ptm_no_such_dir_for_state_io/x.ptmt", AF, AL, A, nullptr, derr) &&
                      !derr.empty(),
                  "G5c: an unreachable destination refuses -- \"" + derr + "\"");
        }

        std::remove(p.c_str());
    }

    // ---- G6: probe() -- "what is this file", shape not known in advance --
    {
        const std::string p = tmpPath("probe");
        std::string err;
        thermal::state_io::write(p, NF, NL, src32, nullptr, err);

        thermal::state_io::FileInfo fi;
        check(thermal::state_io::probe(p, fi, err),
              "G6: probe() reads a valid checkpoint (" + err + ")");
        check(fi.magic == thermal::state_io::kMagic, "G6: probe() reports the magic");
        check(fi.n_faces == NF && fi.n_layers == NL,
              "G6: probe() recovers the shape WITHOUT being told it");
        check(fi.bytes_expected == 12 + NF * NL * sizeof(float),
              "G6: probe() computes the v1 length rule");
        check(fi.bytes_actual == fi.bytes_expected && fi.bytes_extra == 0,
              "G6: an exact-length v1 file reports zero trailing bytes");

        // A LONGER file is reported, not swallowed -- this is what a future v2
        // trailer will look like, and silence about it would be the bug.
        const std::string plong = tmpPath("probelong");
        {
            std::vector<char> bytes = readAllBytes(p);
            bytes.resize(bytes.size() + 64, '\x7f');
            std::ofstream os(plong, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        check(thermal::state_io::probe(plong, fi, err) && fi.bytes_extra == 64,
              "G6: a file LONGER than the v1 length reports its 64 trailing bytes");
        std::remove(plong.c_str());

        // Refusals mirror read().
        const std::string ptrunc = tmpPath("probetrunc");
        {
            std::vector<char> bytes = readAllBytes(p);
            bytes.resize(bytes.size() - 8);
            std::ofstream os(ptrunc, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        check(!thermal::state_io::probe(ptrunc, fi, err) && !err.empty(),
              "G6: probe() refuses a truncated file -- \"" + err + "\"");
        std::remove(ptrunc.c_str());

        const std::string pzero = tmpPath("probezero");
        {
            std::ofstream os(pzero, std::ios::binary);
            const std::uint32_t m = thermal::state_io::kMagic, z = 0, l = 30;
            os.write(reinterpret_cast<const char*>(&m), 4);
            os.write(reinterpret_cast<const char*>(&z), 4);
            os.write(reinterpret_cast<const char*>(&l), 4);
        }
        check(!thermal::state_io::probe(pzero, fi, err) && !err.empty(),
              "G6: probe() refuses a degenerate shape -- \"" + err + "\"");
        std::remove(pzero.c_str());

        check(!thermal::state_io::probe("/tmp/ptm_no_such_state_file.ptmt", fi, err) &&
                  !err.empty(),
              "G6: probe() refuses a missing file -- \"" + err + "\"");

        std::remove(p.c_str());
    }

    // ======================================================================
    // G7 -- the PTMT v2 identity trailer
    // ======================================================================
    namespace sio = thermal::state_io;
    // One more than this build understands, for the forward-compatibility arm.
    const std::uint32_t kFutureVersion = sio::kTrailerVersion + 1u;
    // The reference trailer every G7/G8 arm starts from. Values are chosen to
    // be distinguishable from each other and from any plausible default, so a
    // field read into the WRONG slot cannot pass.
    auto refTrailer = [&]() {
        sio::Trailer t;
        t.z.resize(NL);
        for (std::size_t l = 0; l < NL; ++l) t.z[l] = 0.001 * (l + 1) * (l + 1);
        t.et_seconds       = 1.1046816e9;      // ~2035-01-01, a real ET
        t.output_cadence_s = 1800.0;
        t.refgrid_H        = 0.06;
        t.grid_m = 10; t.grid_n = 15; t.grid_b = 15;
        t.mesh_geomhash    = 123456.789012345;
        t.config_identity_hash = 0xFEEDFACECAFEBEEDull;
        t.body_name        = "Moon";
        t.engine_git_sha   = "0123456789abcdef0123456789abcdef01234567";
        t.flags            = 0;
        return t;
    };

    {
        const std::string p = tmpPath("v2roundtrip");
        std::string err;
        const sio::Trailer w = refTrailer();
        check(sio::write(p, NF, NL, src32, &w, err),
              "G7: write() with a v2 trailer succeeds (" + err + ")");

        // The v1 PREFIX must be untouched: a v2 file's first 12 + F*L*4 bytes
        // are exactly the v1 file's. If this ever moves, every banked state and
        // every ptm_eq_certify reading breaks at once.
        const std::string p1 = tmpPath("v2prefix_v1");
        sio::write(p1, NF, NL, src32, nullptr, err);
        {
            const std::vector<char> b2 = readAllBytes(p);
            const std::vector<char> b1 = readAllBytes(p1);
            check(b2.size() > b1.size() &&
                      std::equal(b1.begin(), b1.end(), b2.begin()),
                  "G7: the v2 file's first " + std::to_string(b1.size()) +
                      " bytes are BYTE-IDENTICAL to the v1 file's");
            check(b2.size() - b1.size() == w.sizeBytes(),
                  "G7: the trailer is exactly sizeBytes() = " +
                      std::to_string(w.sizeBytes()) + " bytes long");
        }

        // The v1 reader is untouched by the trailer.
        std::vector<float> back;
        check(sio::read(p, NF, NL, back, err) && back == src32,
              "G7: read() recovers the body from a v2 file, ignoring the trailer");

        sio::Trailer r;
        std::string terr;
        check(sio::readTrailer(p, r, terr) == sio::TrailerStatus::Ok,
              "G7: readTrailer() accepts the file and verifies its CRC32 (" +
                  terr + ")");
        check(r.z == w.z, "G7: z[] round-trips exactly");
        check(r.et_seconds == w.et_seconds, "G7: et_seconds round-trips exactly");
        check(r.output_cadence_s == w.output_cadence_s,
              "G7: output_cadence_s round-trips");
        check(r.refgrid_H == w.refgrid_H, "G7: refgrid_H round-trips");
        check(r.grid_m == w.grid_m && r.grid_n == w.grid_n &&
                  r.grid_b == w.grid_b,
              "G7: grid m/n/b round-trip");
        check(r.mesh_geomhash == w.mesh_geomhash, "G7: mesh_geomhash round-trips");
        check(r.config_identity_hash == w.config_identity_hash,
              "G7: config_identity_hash round-trips");
        check(r.body_name == w.body_name, "G7: body_name round-trips");
        check(r.engine_git_sha == w.engine_git_sha,
              "G7: engine_git_sha round-trips (40-char SHA fits the field)");
        check(!std::isfinite(r.w0_mean) && !std::isfinite(r.w0_p05) &&
                  !std::isfinite(r.w0_p95),
              "G7: w0_* read back as NaN -- 'not recorded', not zero");

        // The whole point of a checksum over the BODY: a length-correct file
        // whose temperatures were replaced is exactly the cloud-placeholder
        // failure mode, and length + magic both pass on it.
        const std::string pbody = tmpPath("v2bodyflip");
        {
            std::vector<char> bytes = readAllBytes(p);
            bytes[12 + 4 * (NF * NL / 2)] ^= 0x01;   // one bit, deep in the body
            std::ofstream os(pbody, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        {
            sio::FileInfo fi;
            std::string perr;
            check(sio::probe(pbody, fi, perr),
                  "G7: probe() still ACCEPTS the body-flipped file -- length and "
                  "magic cannot see it");
            sio::Trailer bad_t;
            std::string berr;
            check(sio::readTrailer(pbody, bad_t, berr) == sio::TrailerStatus::Malformed &&
                      berr.find("CRC32") != std::string::npos,
                  "G7: ONE flipped body bit is caught by the CRC32 -- \"" +
                      berr.substr(0, 80) + "...\"");
        }
        std::remove(pbody.c_str());

        // Zero-filled body at the correct length -- the OneDrive lazy-fetch
        // signature the trailer's checksum exists for.
        const std::string pzf = tmpPath("v2zerofill");
        {
            std::vector<char> bytes = readAllBytes(p);
            std::fill(bytes.begin() + 12,
                      bytes.begin() + 12 + static_cast<long>(NF * NL * 4), '\0');
            std::ofstream os(pzf, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        {
            sio::Trailer bad_t;
            std::string berr;
            check(sio::readTrailer(pzf, bad_t, berr) == sio::TrailerStatus::Malformed,
                  "G7: a length-correct ZERO-FILLED body is refused by the CRC32");
        }
        std::remove(pzf.c_str());

        // A flipped byte in the trailer itself.
        const std::string ptf = tmpPath("v2trailerflip");
        {
            std::vector<char> bytes = readAllBytes(p);
            bytes[bytes.size() - 20] ^= 0x02;
            std::ofstream os(ptf, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        {
            sio::Trailer bad_t;
            std::string berr;
            check(sio::readTrailer(ptf, bad_t, berr) == sio::TrailerStatus::Malformed,
                  "G7: a flipped bit INSIDE the trailer is caught too");
        }
        std::remove(ptf.c_str());

        // Extra bytes that are not a trailer.
        const std::string pjunk = tmpPath("v2junk");
        {
            std::vector<char> bytes = readAllBytes(p1);
            bytes.resize(bytes.size() + sio::kTrailerFixedBytes + NL * 8, '\x5a');
            std::ofstream os(pjunk, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        {
            sio::Trailer bad_t;
            std::string berr;
            check(sio::readTrailer(pjunk, bad_t, berr) == sio::TrailerStatus::Malformed &&
                      berr.find("PTMX") != std::string::npos,
                  "G7: trailing junk with no 'PTMX' magic refuses, naming the "
                  "magic it wanted");
        }
        std::remove(pjunk.c_str());

        // A truncated trailer: shorter than the fixed part.
        const std::string pshortt = tmpPath("v2shorttrailer");
        {
            std::vector<char> bytes = readAllBytes(p);
            bytes.resize(bytes.size() - 16);
            std::ofstream os(pshortt, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        {
            sio::Trailer bad_t;
            std::string berr;
            check(sio::readTrailer(pshortt, bad_t, berr) == sio::TrailerStatus::Malformed,
                  "G7: a truncated trailer refuses rather than reading short "
                  "fields as sentinels");
        }
        std::remove(pshortt.c_str());

        // A v1 file reports Absent, NOT Malformed -- every banked state on disk
        // depends on this distinction.
        {
            sio::Trailer none;
            std::string nerr;
            check(sio::readTrailer(p1, none, nerr) == sio::TrailerStatus::Absent,
                  "G7: a v1 file reports Absent (it LOADS; it is not corrupt)");
        }

        // write() refuses a trailer that disagrees with its own header.
        {
            sio::Trailer wrong = refTrailer();
            wrong.z.pop_back();
            std::string werr;
            check(!sio::write(tmpPath("v2badtrailer"), NF, NL, src32, &wrong, werr) &&
                      !werr.empty(),
                  "G7: write() REFUSES a trailer whose z[] length disagrees with "
                  "n_layers -- \"" + werr.substr(0, 70) + "...\"");
        }

        // --- forward compatibility -----------------------------------------
        // A LATER writer appends fields between the v2 prefix and the trailing
        // crc32. This build must read the prefix it knows, say the file is
        // newer, and still verify the checksum -- not fail with a CRC error,
        // which is what taking the checksum from the end of the KNOWN fields
        // instead of the end of the TRAILER would do.
        {
            const std::string pv3 = tmpPath("v2future");
            std::vector<char> bytes = readAllBytes(p);
            const std::size_t t0 = bytes.size() - w.sizeBytes();
            // version 2 -> 3
            const std::uint32_t v3 = kFutureVersion;
            std::memcpy(bytes.data() + t0 + 4, &v3, 4);
            // Insert 16 unknown bytes just before the trailing crc32.
            bytes.insert(bytes.end() - 4, 16, '\x11');
            const std::uint32_t new_bytes =
                static_cast<std::uint32_t>(bytes.size() - t0);
            std::memcpy(bytes.data() + t0 + 8, &new_bytes, 4);
            std::uint32_t c = sio::crc32Init();
            c = sio::crc32Update(c, bytes.data(), bytes.size() - 4);
            const std::uint32_t cf = sio::crc32Final(c);
            std::memcpy(bytes.data() + bytes.size() - 4, &cf, 4);
            std::ofstream os(pv3, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            os.close();

            sio::Trailer fut;
            std::string ferr2;
            const auto st = sio::readTrailer(pv3, fut, ferr2);
            check(st == sio::TrailerStatus::Ok,
                  "G7: a NEWER trailer version with appended fields still "
                  "verifies (" + ferr2.substr(0, 60) + ")");
            check(ferr2.find("NEWER") != std::string::npos,
                  "G7: and it SAYS the file is newer rather than reading it as "
                  "if it were understood");
            check(fut.et_seconds == w.et_seconds && fut.z == w.z &&
                      fut.config_identity_hash == w.config_identity_hash,
                  "G7: the version-2 prefix of a newer file reads correctly");
            std::remove(pv3.c_str());
        }

        // A newer version that keeps the SAME byte layout (a redefined field,
        // not an appended one) must still be disclosed. Nesting the version
        // notice inside the size check would read this file silently.
        {
            const std::string pv3s = tmpPath("v2futuresamesize");
            std::vector<char> bytes = readAllBytes(p);
            const std::size_t t0 = bytes.size() - w.sizeBytes();
            const std::uint32_t v3 = kFutureVersion;
            std::memcpy(bytes.data() + t0 + 4, &v3, 4);
            std::uint32_t c = sio::crc32Init();
            c = sio::crc32Update(c, bytes.data(), bytes.size() - 4);
            const std::uint32_t cf = sio::crc32Final(c);
            std::memcpy(bytes.data() + bytes.size() - 4, &cf, 4);
            std::ofstream os(pv3s, std::ios::binary);
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            os.close();

            sio::Trailer fut;
            std::string ferr3;
            check(sio::readTrailer(pv3s, fut, ferr3) == sio::TrailerStatus::Ok &&
                      ferr3.find("NEWER") != std::string::npos,
                  "G7: a newer version at the SAME size is still disclosed, not "
                  "read as if it were understood");
            std::remove(pv3s.c_str());
        }

        // The checksum must not depend on how the bytes were chunked.
        {
            const std::vector<char> bytes = readAllBytes(p);
            std::uint32_t c1 = sio::crc32Init();
            c1 = sio::crc32Update(c1, bytes.data(), bytes.size() - 4);
            std::uint32_t c2 = sio::crc32Init();
            for (std::size_t i = 0; i + 4 < bytes.size(); i += 7)
                c2 = sio::crc32Update(
                    c2, bytes.data() + i,
                    std::min<std::size_t>(7, bytes.size() - 4 - i));
            check(sio::crc32Final(c1) == sio::crc32Final(c2) &&
                      sio::crc32Final(c1) == r.crc32_stored,
                  "G7: CRC32 is chunk-invariant and reproduces the stored value");
        }

        std::remove(p.c_str());
        std::remove(p1.c_str());
    }

    // ======================================================================
    // G8 -- restore admissibility. THE refusal gate.
    // ======================================================================
    {
        const sio::Trailer file = refTrailer();
        // The live run, matching the file exactly. et_first is the epoch the
        // NEXT chunk starts at, which under checkpoint_lookahead_steps: 1 is
        // exactly where the previous chunk's checkpoint was taken.
        auto matchingRun = [&]() {
            sio::Identity r;
            r.z = file.z;
            r.et_first         = file.et_seconds;
            r.output_cadence_s = file.output_cadence_s;
            r.refgrid_H        = file.refgrid_H;
            r.grid_m = file.grid_m; r.grid_n = file.grid_n; r.grid_b = file.grid_b;
            r.mesh_geomhash    = file.mesh_geomhash;
            r.config_identity_hash = file.config_identity_hash;
            r.body_name        = file.body_name;
            return r;
        };
        const std::vector<std::string> no_allow;
        sio::RestoreVerdict v;

        // --- the tiling hand-off: silent pass ------------------------------
        sio::checkRestore(&file, sio::TrailerStatus::Ok, matchingRun(), no_allow, v);
        check(v.ok() && v.warnings.empty() && v.unverified.empty(),
              "G8: a TILING chunk hand-off passes clean -- no refusal, no "
              "warning, nothing unverified");

        // --- wrong mesh ----------------------------------------------------
        {
            sio::Identity r = matchingRun();
            r.mesh_geomhash *= 1.0000001;      // a different mesh, same face count
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("mesh") != std::string::npos,
                  "G8: WRONG MESH refuses, and only that clause fires");
        }

        // --- wrong physics -------------------------------------------------
        {
            sio::Identity r = matchingRun();
            r.config_identity_hash ^= 1ull;
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("physics") != std::string::npos,
                  "G8: WRONG CONFIG (physics hash) refuses");
        }

        // --- wrong body ----------------------------------------------------
        {
            sio::Identity r = matchingRun();
            r.body_name = "Europa";
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("Europa") != std::string::npos,
                  "G8: WRONG BODY refuses, naming both bodies");
        }

        // --- wrong depth grid, SAME layer count: PTM-STATE-GRID-BLIND ------
        {
            sio::Identity r = matchingRun();
            r.z[NL / 2] *= 1.001;              // same count, different depths
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("depth grid") != std::string::npos &&
                      v.refusals[0].find("node " + std::to_string(NL / 2)) !=
                          std::string::npos,
                  "G8: a DIFFERENT depth grid with the SAME layer count refuses, "
                  "naming the first node that differs (PTM-STATE-GRID-BLIND)");
        }

        // --- wrong grid_b (the D14 clause's own knob) ----------------------
        // ⛔ The message must name the YAML KEY `thermal.thermal_b`, NOT the
        // C++ field `thermal.grid_b`. The clause NAME stays `grid_b` -- that is
        // what `state_restore_allow_mismatch` accepts and what
        // docs/configuration.md lists -- but a refusal telling a user to change
        // `thermal.grid_b` sends them after a key the parser never reads
        // (PTM-STATEIO-REFUSAL-NAMES-A-NONEXISTENT-KEY). Both directions are
        // asserted: the old spelling is not a substring of the new one, so the
        // negative half is what fails if it ever comes back.
        {
            sio::Identity r = matchingRun();
            r.grid_b = 32;
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            bool named_b = false, named_field = false;
            for (const auto& s : v.refusals) {
                if (s.find("thermal.thermal_b") != std::string::npos) named_b = true;
                if (s.find("thermal.grid_b") != std::string::npos) named_field = true;
            }
            check(!v.ok() && named_b,
                  "G8: a different bottom-depth setting refuses BY NAME, naming "
                  "the YAML key thermal.thermal_b, not just a depth-grid "
                  "difference");
            check(!named_field,
                  "G8: ...and does NOT name thermal.grid_b, which is the C++ "
                  "field and not a config key");
            // ...and the CLAUSE name is unchanged, so an existing
            // `state_restore_allow_mismatch: [grid_b]` still downgrades it.
            // Without this arm, renaming the clause along with the message
            // would silently break every config carrying that opt-out.
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, {"grid_b"}, v);
            bool warned_b = false;
            for (const auto& s : v.warnings)
                if (s.find("thermal.thermal_b") != std::string::npos) warned_b = true;
            check(v.ok() && warned_b,
                  "G8: state_restore_allow_mismatch [grid_b] still downgrades "
                  "this clause -- the CLAUSE name did not change with the "
                  "message");
        }

        // --- wrong epoch, ONE CADENCE: warn, do not refuse -----------------
        // This is what a chunk chain looks like WITHOUT
        // checkpoint_lookahead_steps: 1 -- scripts/run_yearlong_wedge_safe.py
        // does exactly this. Refusing it would break a working production
        // script over a known, documented one-step seam; staying silent is what
        // let the seam exist. So: warn, with the number.
        {
            sio::Identity r = matchingRun();
            r.et_first = file.et_seconds + file.output_cadence_s;
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(v.ok() && v.warnings.size() == 1 &&
                      v.warnings[0].find("one output cadence") != std::string::npos,
                  "G8: a ONE-CADENCE-stale hand-off WARNS and continues");
        }

        // --- wrong epoch, one cadence + a hair: refuse ---------------------
        {
            sio::Identity r = matchingRun();
            r.et_first = file.et_seconds + file.output_cadence_s * 1.5;
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("epoch") != std::string::npos,
                  "G8: just PAST one cadence refuses -- the boundary is a "
                  "boundary, not a slope");
        }

        // --- the checkpoint is AHEAD of the run: an OVERLAP, not staleness --
        // The gap is signed, and the sign changes what happened. Branching on
        // |gap| would tell this operator to set checkpoint_lookahead_steps,
        // which makes the overlap one step worse.
        {
            sio::Identity r = matchingRun();
            r.et_first = file.et_seconds - file.output_cadence_s;
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("OVERLAP") != std::string::npos &&
                      v.refusals[0].find("look-ahead") != std::string::npos,
                  "G8: a checkpoint AHEAD of the run refuses as an OVERLAP, and "
                  "says it is not a look-ahead problem");
        }

        // --- wrong season: the D13 case ------------------------------------
        // Consecutive lunar polar windows sit ~165 days apart. Measured cost of
        // taking it: temperature_min 5.75 K rms / 49.9 K max, and the
        // ice-stability depth product moved 19.3 percentage points.
        {
            sio::Identity r = matchingRun();
            r.et_first = file.et_seconds + 165.0 * 86400.0;
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("165.0") != std::string::npos,
                  "G8: a CROSS-SEASON warm start refuses, printing the gap in "
                  "days (D13)");
        }

        // --- a partial state can never load as a full one ------------------
        {
            sio::Trailer partial = file;
            partial.flags |= sio::flag::kBatchCompletionPartial;
            sio::checkRestore(&partial, sio::TrailerStatus::Ok, matchingRun(),
                              no_allow, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("NOT overridable") != std::string::npos,
                  "G8: a BATCH-COMPLETION PARTIAL refuses");
            // And the opt-out cannot reach it. Every field key at once.
            const std::vector<std::string> everything = {
                "depth_grid", "mesh", "config", "body", "grid_m", "grid_n",
                "grid_b", "refgrid_H", "epoch", "partial", "force", "all"};
            sio::checkRestore(&partial, sio::TrailerStatus::Ok, matchingRun(),
                              everything, v);
            check(!v.ok(),
                  "G8: state_restore_allow_mismatch CANNOT downgrade the "
                  "partial-state refusal, whatever it names");
        }

        // --- the named opt-out downgrades exactly one field ----------------
        {
            sio::Identity r = matchingRun();
            r.et_first = file.et_seconds + 165.0 * 86400.0;
            r.mesh_geomhash *= 1.0000001;
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, {"epoch"}, v);
            check(!v.ok() && v.refusals.size() == 1 &&
                      v.refusals[0].find("mesh") != std::string::npos &&
                      v.warnings.size() == 1 &&
                      v.warnings[0].find("epoch") != std::string::npos,
                  "G8: allow_mismatch [epoch] downgrades ONLY the epoch clause; "
                  "the mesh clause still refuses");
        }

        // --- a v1 file: loads, and every gap is named ----------------------
        {
            sio::checkRestore(nullptr, sio::TrailerStatus::Absent, matchingRun(),
                              no_allow, v);
            check(v.ok() && v.refusals.empty() && v.unverified.size() >= 5,
                  "G8: a v1 file LOADS and reports " +
                      std::to_string(v.unverified.size()) +
                      " unverified quantities by name");
            bool names_grid = false, names_epoch = false;
            for (const auto& u : v.unverified) {
                if (u.find("depth grid z[]") != std::string::npos &&
                    u.find("accepted in silence") != std::string::npos) names_grid = true;
                if (u.rfind("epoch", 0) == 0 &&
                    u.find("cannot be") != std::string::npos) names_epoch = true;
            }
            check(names_grid && names_epoch,
                  "G8: the v1 report names the unchecked depth grid and the "
                  "unrecorded epoch explicitly");
        }

        // --- coupling provenance -------------------------------------------
        {
            sio::Identity r = matchingRun();
            r.coupled_run = true;
            sio::checkRestore(&file, sio::TrailerStatus::Ok, r, no_allow, v);
            check(v.ok() && v.warnings.size() == 1 &&
                      v.warnings[0].find("PRE-coupling") != std::string::npos,
                  "G8: a PRE-coupling checkpoint into a coupled run warns, "
                  "saying so");

            sio::Trailer post = file;
            post.flags |= sio::flag::kPostCouplingComplete;
            sio::checkRestore(&post, sio::TrailerStatus::Ok, r, no_allow, v);
            check(v.ok() && v.warnings.size() == 1 &&
                      v.warnings[0].find("re-thermal'd selection") != std::string::npos,
                  "G8: post-coupling WITHOUT all_faces_patched says so");

            post.flags |= sio::flag::kAllFacesPatched;
            sio::checkRestore(&post, sio::TrailerStatus::Ok, r, no_allow, v);
            check(v.ok() && v.warnings.empty(),
                  "G8: post-coupling with every face patched is silent");
        }

        // --- the F4 guard ---------------------------------------------------
        {
            sio::Trailer sel = file;
            sel.flags |= sio::flag::kCouplingSelectDtPositive;
            sio::checkRestore(&sel, sio::TrailerStatus::Ok, matchingRun(),
                              no_allow, v);
            bool named = false;
            for (const auto& w : v.warnings)
                if (w.find("coupling_select_dt") != std::string::npos) named = true;
            check(named, "G8: coupling_select_dt > 0 in the writing run is "
                         "reported to the restore (Fermi F4)");
        }
    }

    // ---- Model::restoreState / captureState ------------------------------
    {
        thermal::Planet moon("Moon");
        thermal::Config cfg;
        std::vector<double> times;
        const double day = moon.day;
        for (int i = 0; i < 8; ++i) times.push_back(i * day / 8.0);
        Eigen::MatrixXd flux(2, static_cast<int>(times.size()));
        for (int i = 0; i < static_cast<int>(times.size()); ++i) {
            flux(0, i) = times[i];
            flux(1, i) = 100.0;
        }
        thermal::Model model(moon, 0.0, times, flux, cfg, false);
        const int nz = model.N_z;
        check(nz > 1, "reference Model has a usable depth grid (N_z=" +
                          std::to_string(nz) + ")");

        std::vector<float> col(nz);
        for (int i = 0; i < nz; ++i) col[i] = 111.5f + 0.25f * i;

        check(model.restoreState(col.data(), nz),
              "restoreState accepts a matching column");
        check(model.warm_started,
              "restoreState marks the model warm-started (pre-conditioner is "
              "then skipped in run())");

        bool exact = true;
        for (int i = 0; i < nz && exact; ++i)
            exact = (model.profile.T()[i] == static_cast<double>(col[i]));
        check(exact, "restoreState installs the column exactly (no rounding)");

        // Properties were recomputed against the restored temperatures, not
        // left over from the seed.
        bool props_ok = true;
        for (int i = 0; i < nz && props_ok; ++i)
            props_ok = std::isfinite(model.profile.k()[i]) &&
                       std::isfinite(model.profile.cp()[i]) &&
                       std::isfinite(model.profile.kappa()[i]) &&
                       model.profile.cp()[i] > 0.0;
        check(props_ok, "restoreState recomputes k, cp and kappa consistently");

        std::vector<float> outcol(nz, 0.0f);
        check(model.captureState(outcol.data(), nz),
              "captureState accepts a matching column");
        bool rt = true;
        for (int i = 0; i < nz && rt; ++i) rt = (outcol[i] == col[i]);
        check(rt, "G1: restore -> capture with no integration is bit-identical");

        // G4: refusal, not a silent partial restore.
        check(!model.restoreState(col.data(), nz - 1),
              "G4: restoreState REFUSES a column with the wrong layer count");
        check(!model.restoreState(nullptr, nz),
              "G4: restoreState REFUSES a null column");
        check(!model.captureState(outcol.data(), nz + 1),
              "G4: captureState REFUSES a mismatched destination");
    }

    std::cout << "=== " << (g_checks - g_failed) << "/" << g_checks
              << " checks passed ===" << std::endl;
    return (g_failed == 0) ? 0 : 1;
}
