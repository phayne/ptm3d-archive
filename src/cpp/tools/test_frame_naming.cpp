// test_frame_naming -- per-step frame file names (BUGS.md PTM-664).
//
// Frames were zero-padded to FOUR digits, a MINIMUM width, so frame 10000 was
// "10000" and an alphabetical listing of a >10,000-frame run came out
// 0000, ..., 1000, 10000, 10001, ..., 1001. The writer now pads every frame of
// a run to the same width, max(6, digits(n_frames - 1)); readers resolve a
// frame by index under either generation.
//
// Arms:
//   1. width rule: minimum 6; grows exactly when n_frames - 1 needs more.
//   2. within one run, lexicographic order == numeric order, including a run
//      that crosses 10^6 frames.
//   3. resolveFrameFile finds new names, legacy 4-digit names, legacy 5-digit
//      rollover names (frame 10000 of an old run), and returns "" for a gap.
//   4. removeOtherWidthFrames clears a previous run's other-width frames and
//      nothing else.
//
// Standalone, like every other test here: no ctest, no gtest.
//   ./build/bin/test_frame_naming

#include "../fileio/frame_naming.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::cout << (ok ? "  ok    " : "  FAIL  ") << what << "\n";
    if (!ok) ++g_failed;
}

void touch(const std::string& p) { std::ofstream(p) << "x"; }

}  // namespace

int main() {
    std::cout << "=== test_frame_naming ===\n";

    // 1. Width rule.
    check(ptm::frameIndexWidth(0) == 6, "width(0 frames) = 6");
    check(ptm::frameIndexWidth(1) == 6, "width(1 frame) = 6");
    check(ptm::frameIndexWidth(10000) == 6, "width(10,000 frames) = 6");
    check(ptm::frameIndexWidth(161694) == 6, "width(161,694 frames: Justitia Case 4) = 6");
    check(ptm::frameIndexWidth(1000000) == 6, "width(1,000,000 frames; last index 999999) = 6");
    check(ptm::frameIndexWidth(1000001) == 7, "width(1,000,001 frames; last index 1000000) = 7");
    check(ptm::frameIndexString(7, 192) == "000007", "frame 7 of 192 -> 000007");
    check(ptm::frameIndexString(10000, 20000) == "010000", "frame 10000 of 20000 -> 010000");
    check(ptm::frameIndexString(3, 2000000) == "0000003", "frame 3 of 2,000,000 -> 0000003");

    // 2. Lexicographic == numeric within one run, for runs straddling the
    //    old 10,000 rollover and the new 10^6 one.
    for (long long n : {12345LL, 1000002LL}) {
        std::vector<std::string> names;
        for (long long k : {0LL, 1LL, 999LL, 1000LL, 1001LL, 9999LL, 10000LL, 10001LL,
                            n - 2, n - 1}) {
            names.push_back("temperature_" + ptm::frameIndexString(k, n) + ".bin");
        }
        std::vector<std::string> sorted = names;
        std::sort(sorted.begin(), sorted.end());
        check(sorted == names, "lexicographic order == numeric order for a " +
                                   std::to_string(n) + "-frame run");
        // The LEGACY rule fails this for the same run -- the property the
        // change exists for, pinned so the test cannot pass vacuously.
        std::vector<std::string> legacy;
        for (long long k : {0LL, 9999LL, 10000LL, 1001LL}) {
            char b[32]; std::snprintf(b, sizeof b, "%04lld", k);
            legacy.push_back(std::string("temperature_") + b + ".bin");
        }
        std::vector<std::string> ls = legacy;
        std::sort(ls.begin(), ls.end());
        check(ls != legacy, "legacy 4-digit names do NOT sort numerically (control)");
    }

    // 3. Reader resolution over both generations.
    char tmpl[] = "/tmp/ptm_frame_naming_XXXXXX";   // scratch only; removed below
    const char* d = mkdtemp(tmpl);
    if (!d) { std::cerr << "mkdtemp failed\n"; return 2; }
    const std::string dir(d);
    touch(dir + "/temperature_000012.bin");        // new
    touch(dir + "/temperature_0013.bin");          // legacy
    touch(dir + "/temperature_10000.bin");         // legacy rollover (width 5)
    touch(dir + "/temperature_1234567.bin");       // new, 7 digits
    check(ptm::resolveFrameFile(dir, "temperature", 12) == dir + "/temperature_000012.bin",
          "resolves a new 6-digit frame");
    check(ptm::resolveFrameFile(dir, "temperature", 13) == dir + "/temperature_0013.bin",
          "resolves a legacy 4-digit frame");
    check(ptm::resolveFrameFile(dir, "temperature", 10000) == dir + "/temperature_10000.bin",
          "resolves a legacy run's frame 10000 (written 5 digits wide)");
    check(ptm::resolveFrameFile(dir, "temperature", 1234567) == dir + "/temperature_1234567.bin",
          "resolves a 7-digit frame");
    check(ptm::resolveFrameFile(dir, "temperature", 14).empty(),
          "a missing frame resolves to \"\" (no silent substitute)");
    check(ptm::resolveFrameFile(dir, "direct_solar", 12).empty(),
          "a different field prefix does not match");
    // 4. Stale other-width frames from a previous run are removed; this run's
    //    width, other fields and non-frame companions are untouched.
    touch(dir + "/temperature_000013.bin");
    touch(dir + "/temperature_z0p05m_max.bin");
    touch(dir + "/direct_solar_0013.bin");
    const long long removed = ptm::removeOtherWidthFrames(dir, "temperature", 6);
    auto exists = [](const std::string& p) { std::ifstream f(p); return f.good(); };
    check(removed == 3, "removeOtherWidthFrames removed the 3 other-width temperature frames "
                        "(0013, 10000, 1234567), got " + std::to_string(removed));
    check(exists(dir + "/temperature_000012.bin") && exists(dir + "/temperature_000013.bin"),
          "this run's 6-digit frames are kept");
    check(exists(dir + "/temperature_z0p05m_max.bin"), "a non-frame companion is kept");
    check(exists(dir + "/direct_solar_0013.bin"), "another field's frames are kept");
    check(ptm::removeOtherWidthFrames(dir + "/nope", "temperature", 6) == 0,
          "a missing directory removes nothing");

    std::string cmd = "rm -rf '" + dir + "'";
    if (std::system(cmd.c_str()) != 0) std::cerr << "warning: could not remove " << dir << "\n";

    std::cout << (g_failed ? "FAIL" : "PASS") << ": " << (g_checks - g_failed) << "/"
              << g_checks << " checks\n";
    return g_failed ? 1 : 0;
}
