// test_config_shape -- the YAML SHAPE guards in ConfigParser::loadFromYAML.
//
// WHY THIS EXISTS
// ---------------
// yaml-cpp iterates a SCALAR node as an EMPTY range. Every key the parser reads
// with `for (const auto& item : node)` therefore has the same latent failure: a
// value written without brackets clears the destination container and then
// never fills it, and the run proceeds on an empty list.
//
// Measured on the shipped depth-map binary (2026-09-06), before the guard:
//
//   output:                     the run COMPLETED, exit 0, wrote no subsurface
//     map_depths_m: 0.05        maps, and said NOTHING -- a forgotten pair of
//                               brackets is a silent surface-only run
//                               (PTM-DEPTHMAP-SCALAR-SILENT-OFF)
//
//   output:                     the run STOPPED, exit 1, reporting
//     map_depth_stats: avg      "output.map_depth_stats is empty" -- for a file
//                               that plainly named a statistic. Right verdict,
//                               wrong diagnosis, and the wrong fix to try
//                               (PTM-DEPTHMAP-STATS-SCALAR-MISDIAGNOSED)
//
// The arms below pin BOTH directions: the malformed forms are refused and the
// message names the list form, and every legal spelling -- a list, an empty
// list, a null, an absent key -- still parses to exactly the values it did
// before. Without that second half the guard could refuse everything and still
// look like a fix.
//
// Standalone, like every other test here: no ctest, no gtest.
//   ./build/bin/test_config_shape

#include "config_parser.h"

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
    if (ok) {
        std::cout << "  ok    " << what << "\n";
    } else {
        ++g_failed;
        std::cout << "  FAIL  " << what << "\n";
    }
}

// Scratch YAML files, removed when the test ends. One throwaway directory, so
// nothing is left behind and no two arms can collide on a name.
std::string g_dir;
std::vector<std::string> g_temps;

std::string scratchDir() {
    if (!g_dir.empty()) return g_dir;
    char tmpl[] = "/tmp/ptm_config_shape_XXXXXX";
    const char* made = ::mkdtemp(tmpl);
    if (!made) {
        std::cerr << "could not create a scratch directory\n";
        std::exit(2);
    }
    g_dir = made;
    return g_dir;
}

std::string writeConfig(const std::string& body) {
    const std::string path = scratchDir() + "/case" +
                             std::to_string(g_temps.size()) + ".yaml";
    std::ofstream out(path);
    out << body;
    out.close();
    g_temps.push_back(path);
    return path;
}

void cleanup() {
    for (const auto& p : g_temps) ::remove(p.c_str());
    if (!g_dir.empty()) ::rmdir(g_dir.c_str());
}

// `(threw, message)` from one load. The parser wraps every internal throw as
// "Error loading config: <msg>", so the caller matches on the inner text.
struct Load {
    bool threw = false;
    std::string message;
    ptm::SimulationConfig config;
};

Load load(const std::string& body) {
    Load result;
    const std::string path = writeConfig(body);
    try {
        ptm::ConfigParser parser;
        result.config = parser.loadFromYAML(path);
    } catch (const std::exception& e) {
        result.threw = true;
        result.message = e.what();
    }
    return result;
}

bool says(const Load& load, const std::string& needle) {
    return load.message.find(needle) != std::string::npos;
}

// One load, then one arm per thing the message has to say. Loading the same
// config once per needle would parse it three times and print the refusal three
// times, which buries the one line a reader wants.
void refuses(const std::string& body,
             std::initializer_list<const char*> needles,
             const std::string& what) {
    const Load r = load(body);
    if (!r.threw) {
        ++g_checks;
        ++g_failed;
        std::cout << "  FAIL  " << what << " -- the config was ACCEPTED\n";
        return;
    }
    check(true, what);
    std::cout << "        -> " << r.message << "\n";
    for (const char* needle : needles) {
        check(says(r, needle),
              what + " -- the message names '" + needle + "'");
    }
}

}  // namespace

int main() {
    std::cout << "test_config_shape -- YAML list keys that are not lists\n";

    // ------------------------------------------------------------------
    // 1. output.map_depths_m
    // ------------------------------------------------------------------
    std::cout << "\n[1] output.map_depths_m\n";
    refuses("output:\n  map_depths_m: 0.05\n",
            {"must be a YAML LIST",              // the shape expected
             "map_depths_m: [0.05, 0.25]",       // the edit that fixes it
             "(0.05)"},                          // the value it was given
            "a bare scalar depth is refused");
    refuses("output:\n  map_depths_m: \"0.05, 0.25\"\n", {"must be a YAML LIST"},
            "a QUOTED list is one scalar, and is refused too");
    refuses("output:\n  map_depths_m:\n    a: 1\n", {"a mapping"},
            "a mapping is refused and named as one");
    {
        // The quoted-back value is file content on its way to a terminal.
        // A scalar carrying an escape sequence must not reach it intact.
        const Load r = load("output:\n  map_depths_m: \"\\e[31mred\"\n");
        check(r.threw && r.message.find('\x1b') == std::string::npos &&
                  says(r, "?[31mred"),
              "a control character in the offending value is neutralized, not "
              "written to the console");
        const Load r2 = load(
            "output:\n  map_depths_m: "
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n");
        check(r2.threw && says(r2, "aaa...") &&
                  r2.message.find(std::string(61, 'a')) == std::string::npos,
              "an over-long value is truncated rather than pasted in whole");
    }

    {
        const Load r = load("output:\n  map_depths_m: [0.05, 0.25]\n");
        check(!r.threw && r.config.output.map_depths_m.size() == 2 &&
                  r.config.output.map_depths_m[0] == 0.05 &&
                  r.config.output.map_depths_m[1] == 0.25,
              "a real list still parses to its two depths");
    }
    {
        const Load r = load("output:\n  map_depths_m:\n    - 0.0\n    - 0.5\n");
        check(!r.threw && r.config.output.map_depths_m.size() == 2 &&
                  r.config.output.map_depths_m[0] == 0.0 &&
                  r.config.output.map_depths_m[1] == 0.5,
              "the block form still parses");
    }
    {
        const Load r = load("output:\n  map_depths_m: []\n");
        check(!r.threw && r.config.output.map_depths_m.empty(),
              "an EMPTY list is accepted -- it is how the product is off");
    }
    {
        // `map_depths_m:` with nothing after it. yaml-cpp's operator bool() is
        // IsDefined(), so this arrives at the guard truthy while being neither
        // a sequence nor a scalar. It is the natural spelling of "none" and
        // must keep meaning the same as [], or a legal file starts failing.
        const Load r = load("output:\n  map_depths_m:\n");
        check(!r.threw && r.config.output.map_depths_m.empty(),
              "a NULL value means off, exactly as [] does");
    }
    {
        const Load r = load("output:\n  directory: out\n");
        check(!r.threw && r.config.output.map_depths_m.empty(),
              "an ABSENT key is off, and is not a refusal");
    }

    // ------------------------------------------------------------------
    // 2. output.map_depth_stats
    // ------------------------------------------------------------------
    std::cout << "\n[2] output.map_depth_stats\n";
    refuses("output:\n  map_depth_stats: avg\n",
            {"must be a YAML LIST",
             "map_depth_stats: [avg, max, min]",
             "(avg)"},
            "a bare scalar statistic is refused");
    {
        // The misdiagnosis this replaced. "output.map_depth_stats is empty" was
        // the WRONG sentence for a file that named a statistic, and it sent the
        // reader to look for a missing value rather than a missing pair of
        // brackets. The replacement may MENTION that old report -- it explains
        // where the run would otherwise have stopped -- but it must not be the
        // diagnosis, and the value the file gave has to appear.
        const Load r = load("output:\n  map_depth_stats: avg\n");
        check(r.threw && !says(r, "output.map_depth_stats is empty"),
              "the refusal no longer DIAGNOSES a named statistic as an empty list");
    }
    {
        const Load r = load("output:\n  map_depth_stats: [avg, min]\n");
        check(!r.threw && r.config.output.map_depth_stats.size() == 2 &&
                  r.config.output.map_depth_stats[0] == "avg" &&
                  r.config.output.map_depth_stats[1] == "min",
              "a real list still parses to its two statistics");
    }
    {
        // A NULL statistic list stays legal AT THE PARSER and is caught later,
        // by depth_map_plan's empty-statistic clause, where "is empty" is the
        // true sentence. Moving it here would refuse a file the engine has
        // always been able to read and diagnose correctly.
        const Load r = load("output:\n  map_depth_stats:\n");
        check(!r.threw && r.config.output.map_depth_stats.empty(),
              "a NULL statistic list parses to empty and is left to the plan");
    }
    {
        const Load r = load("output:\n  directory: out\n");
        check(!r.threw && r.config.output.map_depth_stats.size() == 3,
              "an ABSENT statistic key keeps the engine's default three");
    }

    // ------------------------------------------------------------------
    // 3. do no harm: the guard fires on these two keys and nothing else
    // ------------------------------------------------------------------
    std::cout << "\n[3] the guard is scoped -- and here is what it does NOT cover\n";
    {
        // ⛔ THIS ARM RECORDS A KNOWN HOLE, it does not bless one. Four other
        // keys are read the same way and are NOT guarded:
        //   thermal.gpu_device_ids, illumination.gpu_device_ids,
        //   radiation.gpu_device_ids, generic_body.shape.semi_axes_km
        // (flux_cubes.locations is a sequence of maps and is already gated on
        // IsSequence()). A scalar in any of them keeps the default in silence:
        // `gpu_device_ids: 0`, meant as "pin device 0", leaves the list empty,
        // which means "use every device" -- the opposite of the request, with
        // no message. That is PTM-CONFIG-SEQUENCE-SCALAR-SILENT, filed rather
        // than fixed here because widening the guard changes the behavior of
        // configs that run today and is outside this change's approved scope.
        //
        // Kept as an arm because it is also the control that proves the arms
        // above measure THE GUARD and not some blanket new strictness. When the
        // hole is closed, this arm flips to `refuses(...)` and this comment goes.
        const Load r = load("illumination:\n  gpu_device_ids: 0\n");
        check(!r.threw,
              "an unguarded sequence key elsewhere is unchanged -- the guard is "
              "the two depth-map keys and nothing else (see the comment: this "
              "records a hole, it does not approve of it)");
    }
    {
        const Load r = load("output:\n  map_depths_m: [0.05]\n"
                            "  map_depth_stats: [avg]\n"
                            "  map_depth_snap_to_node: false\n");
        check(!r.threw && r.config.output.map_depths_m.size() == 1 &&
                  r.config.output.map_depth_stats.size() == 1 &&
                  !r.config.output.map_depth_snap_to_node,
              "a complete, well-formed depth-map block parses unchanged");
    }

    cleanup();
    std::cout << "\n" << (g_checks - g_failed) << "/" << g_checks
              << " checks passed\n";
    if (g_failed) {
        std::cout << "test_config_shape: FAIL\n";
        return 1;
    }
    std::cout << "test_config_shape: PASS\n";
    return 0;
}
