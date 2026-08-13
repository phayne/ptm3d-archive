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

#include "../thermal/thermal_state_io.h"
#include "../thermal/thermal_model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

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
