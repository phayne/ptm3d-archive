#ifndef ORBITAL_EPOCH_PARSE_H
#define ORBITAL_EPOCH_PARSE_H

// Epoch strings for `generic_body.orbit.epoch` and `generic_body.spin.epoch`.
// Both keys were parsed and then dropped, so every generic
// body ran with its Kepler mean anomaly and its prime meridian both referred
// to J2000 whatever the config said.
//
// ACCEPTED FORMS, all read on the TDB time scale (the scale SPICE ET and every
// JPL osculating-element epoch use):
//   "2000-01-01T12:00:00"         calendar date and time; ":SS", ":SS.fff" and
//   "2000-01-01 12:00:00"         the whole time-of-day are optional, and a
//   "2000-01-01"                  space may stand in for the 'T'
//   "2000-01-01T12:00:00 TDB"     an explicit TDB suffix (same meaning)
//   "JD 2451545.0", "JD2451545",  a Julian date (TDB), which must lie in
//   "2451545.0"                   [1e6, 1e7); "2026" or "20260924" are refused
// Every other form is REFUSED, not guessed. In particular a "UTC" / "Z" suffix
// is refused: UTC and TDB differ by 64.184 s at J2000 (69.184 s in 2026),
// which for a 5-hour spin is 1.3 deg of rotational phase -- enough that
// "close enough" is exactly the silent substitution this rule forbids.
//
// Why TDB for a bare calendar string: every shipped config and the HORIZONS
// body-config writer spell J2000 as '2000-01-01T12:00:00', and J2000 is
// 2000-01-01 12:00:00 TDB (JD 2451545.0 exactly). Reading that string as UTC
// would move every existing Keplerian run by 64 s. Read as TDB, it resolves
// to exactly JD 2451545.0 and the run is byte-identical to before.
//
// The Gregorian-calendar conversion is Fliegel & Van Flandern (1968), exact in
// integer arithmetic for any date after -4713-11-24.

#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <string>

namespace orbital {

constexpr double kJ2000JulianDate = 2451545.0;

namespace epoch_detail {

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

inline bool allDigits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

// Strict decimal: digits, at most one '.', nothing else.
inline bool parseDecimal(const std::string& s, double& out) {
    if (s.empty()) return false;
    int dots = 0, digits = 0;
    for (char c : s) {
        if (c == '.') ++dots;
        else if (std::isdigit(static_cast<unsigned char>(c))) ++digits;
        else return false;
    }
    if (dots > 1 || digits == 0) return false;
    out = std::strtod(s.c_str(), nullptr);
    return true;
}

// Julian Day Number of the Gregorian date at 12:00 (Fliegel & Van Flandern).
inline long long julianDayNumber(long long y, long long m, long long d) {
    const long long a = (m - 14) / 12;   // C++ truncates toward zero, as F&VF assume
    return (1461 * (y + 4800 + a)) / 4
         + (367 * (m - 2 - 12 * a)) / 12
         - (3 * ((y + 4900 + a) / 100)) / 4
         + d - 32075;
}

}  // namespace epoch_detail

// Parse `text` into a Julian date on the TDB scale. Returns false with `err`
// naming what was wrong; never substitutes a default.
inline bool parseEpochJdTdb(const std::string& text, double& jd, std::string& err) {
    using namespace epoch_detail;
    std::string s = trim(text);
    if (s.empty()) { err = "empty epoch string"; return false; }

    // Optional explicit scale suffix. TDB is accepted; anything else refused.
    auto endsWithWord = [&](const char* w) {
        const std::string word(w);
        if (s.size() < word.size()) return false;
        std::string tail = s.substr(s.size() - word.size());
        for (auto& c : tail) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return tail == word;
    };
    if (endsWithWord("TDB")) {
        s = trim(s.substr(0, s.size() - 3));
    } else if (endsWithWord("UTC") || endsWithWord("TT") || endsWithWord("TAI") ||
               endsWithWord("Z") || endsWithWord("UT")) {
        err = "epoch '" + text + "' names a time scale other than TDB; this key is read on "
              "the TDB scale (JPL osculating-element epochs are TDB). Give the TDB "
              "calendar time, a 'TDB' suffix, or a Julian date 'JD <number>'";
        return false;
    }

    // Julian date forms.
    {
        std::string t = s;
        bool had_jd = false;
        if (t.size() >= 2 && (t[0] == 'J' || t[0] == 'j') && (t[1] == 'D' || t[1] == 'd')) {
            t = trim(t.substr(2));
            had_jd = true;
        }
        double v = 0.0;
        if (parseDecimal(t, v)) {
            // A number is a JD only inside [1e6, 1e7) -- years -1975 to +22666.
            // Outside it, it is something else written as a number: "2026" is
            // a year, "20260924" a compact YYYYMMDD. Refusing both beats
            // reading them as JD 2026 or a date 50,000 years from now.
            if (!(v >= 1.0e6 && v < 1.0e7)) {
                err = "epoch '" + text + "' is a number outside the Julian-date range "
                      "[1e6, 1e7) (years -1975 to 22666); write 'JD <number>' with a "
                      "real Julian date, or a calendar date YYYY-MM-DDTHH:MM:SS";
                return false;
            }
            jd = v;
            return true;
        }
        if (had_jd) {
            err = "epoch '" + text + "': 'JD' must be followed by a number";
            return false;
        }
    }

    // Calendar form YYYY-MM-DD[(T| )HH:MM[:SS[.fff]]].
    std::string date = s, tod;
    const size_t sep = s.find_first_of("T ");
    if (sep != std::string::npos) { date = s.substr(0, sep); tod = trim(s.substr(sep + 1)); }
    if (date.size() != 10 || date[4] != '-' || date[7] != '-' ||
        !allDigits(date.substr(0, 4)) || !allDigits(date.substr(5, 2)) ||
        !allDigits(date.substr(8, 2))) {
        err = "epoch '" + text + "' is not YYYY-MM-DD[THH:MM[:SS[.fff]]] or 'JD <number>'";
        return false;
    }
    const long long Y = std::atoll(date.substr(0, 4).c_str());
    const long long M = std::atoll(date.substr(5, 2).c_str());
    const long long D = std::atoll(date.substr(8, 2).c_str());
    static const int kDays[12] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = (Y % 4 == 0 && Y % 100 != 0) || (Y % 400 == 0);
    if (M < 1 || M > 12 || D < 1 || D > kDays[M - 1] || (M == 2 && D == 29 && !leap)) {
        err = "epoch '" + text + "' is not a valid calendar date";
        return false;
    }
    double hh = 0.0, mm = 0.0, ss = 0.0;
    if (!tod.empty()) {
        const size_t c1 = tod.find(':');
        if (c1 != 2 || !allDigits(tod.substr(0, 2))) {
            err = "epoch '" + text + "': time of day must be HH:MM[:SS[.fff]]";
            return false;
        }
        const size_t c2 = tod.find(':', c1 + 1);
        const std::string mm_s = tod.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1);
        if (mm_s.size() != 2 || !allDigits(mm_s)) {
            err = "epoch '" + text + "': time of day must be HH:MM[:SS[.fff]]";
            return false;
        }
        hh = std::atof(tod.substr(0, 2).c_str());
        mm = std::atof(mm_s.c_str());
        if (c2 != std::string::npos && !parseDecimal(tod.substr(c2 + 1), ss)) {
            err = "epoch '" + text + "': seconds must be SS or SS.fff";
            return false;
        }
        if (hh > 23.0 || mm > 59.0 || ss >= 61.0) {
            err = "epoch '" + text + "': time of day out of range";
            return false;
        }
    }
    const double jdn = static_cast<double>(julianDayNumber(Y, M, D));
    // JDN is the Julian date at 12:00; 2000-01-01T12:00:00 -> 2451545.0 exactly.
    jd = jdn + ((hh - 12.0) * 3600.0 + mm * 60.0 + ss) / 86400.0;
    return true;
}

}  // namespace orbital

#endif  // ORBITAL_EPOCH_PARSE_H
