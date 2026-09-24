// test_epoch_parse -- generic_body epoch strings (BUGS.md PTM-663).
//
// Pins the rule orbital/epoch_parse.h documents: every accepted form is read on
// the TDB scale, the J2000 spelling every shipped config carries resolves to
// JD 2451545.0 EXACTLY (which is what keeps those runs byte-identical), and
// anything else -- another time scale, a malformed date, a bare year -- is
// refused rather than guessed.
//
// Standalone, like every other test here: no ctest, no gtest.
//   ./build/bin/test_epoch_parse

#include "../orbital/epoch_parse.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {
int g_checks = 0, g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::cout << (ok ? "  ok    " : "  FAIL  ") << what << "\n";
    if (!ok) ++g_failed;
}

void accepts(const std::string& s, double want, double tol = 0.0) {
    double jd = -1.0; std::string err;
    const bool ok = orbital::parseEpochJdTdb(s, jd, err);
    check(ok && std::fabs(jd - want) <= tol,
          "'" + s + "' -> JD " + std::to_string(want) + (ok ? "" : "  [refused: " + err + "]"));
}

void refuses(const std::string& s, const std::string& must_mention) {
    double jd = -1.0; std::string err;
    const bool ok = orbital::parseEpochJdTdb(s, jd, err);
    check(!ok && err.find(must_mention) != std::string::npos,
          "'" + s + "' refused (" + (ok ? "ACCEPTED as " + std::to_string(jd) : err) + ")");
}
}  // namespace

int main() {
    std::cout << "=== test_epoch_parse ===\n";
    // Exactness of the shipped J2000 spelling is the byte-identity argument.
    {
        double jd = 0; std::string err;
        orbital::parseEpochJdTdb("2000-01-01T12:00:00", jd, err);
        check(jd == 2451545.0, "'2000-01-01T12:00:00' == 2451545.0 EXACTLY (bitwise)");
    }
    accepts("2000-01-01 12:00:00", 2451545.0);
    accepts("2000-01-01T12:00:00 TDB", 2451545.0);
    accepts("  2000-01-01T12:00:00tdb  ", 2451545.0);
    accepts("2000-01-01", 2451544.5);
    accepts("2000-01-01T00:00", 2451544.5);
    accepts("1984-07-01T06:00:00", 2445882.75);
    accepts("2034-10-30T00:00:00.5", 2464265.5 + 0.5 / 86400.0, 1e-9);
    accepts("2024-02-29T18:00:00", 2460370.25);       // leap day
    accepts("JD 2445918.0", 2445918.0);                // DAMIT JD0 of Justitia
    accepts("jd2451545", 2451545.0);
    accepts("2451545.25", 2451545.25);
    refuses("2000-01-01T12:00:00 UTC", "time scale other than TDB");
    refuses("2000-01-01T12:00:00Z", "time scale other than TDB");
    refuses("2000-01-01T12:00:00 TT", "time scale other than TDB");
    refuses("2026", "outside the Julian-date range");
    refuses("20260924", "outside the Julian-date range");      // compact YYYYMMDD
    refuses("JD 20260924", "outside the Julian-date range");
    refuses("JD", "must be followed by a number");
    refuses("2023-02-29", "not a valid calendar date");
    refuses("2023-13-01", "not a valid calendar date");
    refuses("2023-1-5", "is not YYYY-MM-DD");
    refuses("2023-01-05T7:00", "HH:MM");
    refuses("2023-01-05T24:00:00", "out of range");
    refuses("2023-01-05T12:00:xx", "seconds");
    refuses("", "empty");
    std::cout << (g_failed ? "FAIL" : "PASS") << ": " << (g_checks - g_failed) << "/"
              << g_checks << " checks\n";
    return g_failed ? 1 : 0;
}
