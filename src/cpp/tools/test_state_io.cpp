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
        check(thermal::state_io::write(p, NF, NL, src32, err),
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
        thermal::state_io::write(p2, NF, NL, src32, err);
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
        thermal::state_io::write(p, NF, NL, src32, err);
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
        thermal::state_io::write(pgood, NF, NL, src32, err);
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
        check(!thermal::state_io::write(tmpPath("badbuf"), NF, NL, wrong, err) &&
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
        check(thermal::state_io::write(p, AF, AL, A, err),
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
            const bool ok = thermal::state_io::write(p, AF, AL, B, cerr_unused);
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
                ? thermal::state_io::write(p, AF, AL, B, ferr) : true;

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
                      "/tmp/ptm_no_such_dir_for_state_io/x.ptmt", AF, AL, A, derr) &&
                      !derr.empty(),
                  "G5c: an unreachable destination refuses -- \"" + derr + "\"");
        }

        std::remove(p.c_str());
    }

    // ---- G6: probe() -- "what is this file", shape not known in advance --
    {
        const std::string p = tmpPath("probe");
        std::string err;
        thermal::state_io::write(p, NF, NL, src32, err);

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
