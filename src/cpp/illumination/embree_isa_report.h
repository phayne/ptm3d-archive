#pragma once
// ---------------------------------------------------------------------------
// Report, once per process, which Embree SIMD kernels we actually got.
//
// Embree compiles one BVH traversal kernel per SIMD ISA, fixed at ITS OWN build
// time by EMBREE_MAX_ISA, and dispatches at run time to the widest one that was
// compiled in AND that the CPU supports. A redistributable build has to run on
// every CPU it might land on, so Homebrew's embree 4.3.3 bottle is built
// EMBREE_MAX_ISA=SSE2. On this project's Mac Pro -- a Cascade Lake Xeon W with
// AVX-512 -- that means PTM3D has been running SSE2 BVH4 traversal all along,
// and nothing in Embree's default output or in PTM3D said so.
//
// Ray traversal is roughly 83 % of the illumination phase. Measured 2026-08-15
// against the IDENTICAL Embree 4.3.3 source rebuilt with EMBREE_MAX_ISA=AVX512:
// -7.0 % on the illumination phase of the calib120 arm end to end, over 18
// paired interleaved repeats, 18/18 favoring the multi-ISA build, with the
// ray-free thermal phase measured alongside as a null control. (A traversal
// micro-benchmark showed rtcOccluded1 1.22-1.45x; that ratio is NOT an
// end-to-end ratio and should not be quoted as one.)
//
// So: say it, every run, unprompted. A silent few percent -- attached to a
// library swap that also moves results -- is exactly the kind of thing that
// hides for a year. This is a DIAGNOSTIC ONLY: it casts no ray, changes no
// result, and changes no default.
//
// Build-side companion: CMakeLists.txt prints the library path and its
// EMBREE_MAX_ISA at configure time, and honors -DEMBREE_ROOT=<prefix>.
// See INSTALL.md, "Embree ISA (performance)".
// ---------------------------------------------------------------------------

#include <embree4/rtcore.h>

#include <atomic>
#include <cstdlib>
#include <iostream>

namespace ptm {

// Call after creating any RTCDevice. Safe to call from every Embree consumer
// (illumination, view factors, camera, viewshed): it prints on the first call
// only. `inline` and not `static` on purpose -- a function-local static in an
// inline function is one object across the whole program, so "once" means once
// per process rather than once per translation unit.
inline void reportEmbreeIsaOnce(RTCDevice device) {
    // Atomic exchange, not a plain bool: illumination, view factors and the
    // camera each initialize their own device, and a multi-site driver can do
    // so concurrently. A plain `static bool` read-then-write is a data race
    // that can print the banner twice or interleave two copies of it, which
    // would make the comment above ("once per process") a promise the code
    // does not keep. Zero cost on the non-first call.
    static std::atomic<bool> done{false};
    if (!device || done.exchange(true)) return;

    const bool ray4  = rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED)  != 0;
    const bool ray8  = rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED)  != 0;
    const bool ray16 = rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED) != 0;

    // The widest native packet width is the ISA tier, in the only terms Embree
    // exposes programmatically: 16-wide requires AVX-512 kernels, 8-wide
    // requires AVX/AVX2, 4-wide-only means SSE2/SSE4.2. These properties report
    // what this RUN resolved to -- the intersection of what the library was
    // compiled with and what this CPU has -- which is the number that matters,
    // not either input alone.
    //
    // ONE CAVEAT, and it is the reason for the `packets_off` branch below:
    // Embree gates these properties on EMBREE_RAY_PACKETS as well as on the
    // ISA. A library built AVX512 but RAY_PACKETS=OFF reports 4=8=16=no while
    // happily running AVX-512 single-ray traversal -- and PTM3D's hot path IS
    // single-ray (rtcOccluded1). Inferring "SSE2" there would be a false alarm
    // from the one line whose job is to be trustworthy, so that case reports
    // that it cannot tell instead of guessing. All-widths-absent is the
    // signature: no shipped ISA configuration produces it, since even an SSE2
    // build has native 4-wide.
    const bool packets_off = !ray4 && !ray8 && !ray16;
    const char* tier = packets_off ? "cannot determine (built without ray packets)"
                     : ray16       ? "AVX-512 (16-wide)"
                     : ray8        ? "AVX/AVX2 (8-wide)"
                                   : "SSE (4-wide)";

    // Ask the CPU at RUN time, not the __AVX__ compile macros: those follow
    // PTM3D's own -march flags, so a build without -march=native would claim
    // "no AVX" on an AVX machine and the warning below would silently never
    // fire. The question is what this CPU has, not what we were compiled for.
    bool cpu_has_avx = false, cpu_has_avx512 = false;
    // __builtin_cpu_supports is a GCC/Clang builtin; MSVC (the Phase 2 Windows
    // target) has no such thing and would fail to compile here.
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    cpu_has_avx    = __builtin_cpu_supports("avx");
    cpu_has_avx512 = __builtin_cpu_supports("avx512f");
#endif

    std::cerr << "[Embree] " << RTC_VERSION_STRING
              << "  traversal kernels: " << tier
              << "  (native packets 4=" << (ray4 ? "yes" : "no")
              << " 8=" << (ray8 ? "yes" : "no")
              << " 16=" << (ray16 ? "yes" : "no") << ")"
              << std::endl;

    if (packets_off) {
        std::cerr << "[Embree] NOTE: this Embree was built EMBREE_RAY_PACKETS=OFF, so its"
                  << " SIMD ISA\n         cannot be read from device properties. Check the"
                  << " configure-time\n         \"Embree: ... max ISA ...\" line, or"
                  << " provenance.txt, instead." << std::endl;
    } else if (cpu_has_avx && !ray8) {
        std::cerr
            << "[Embree] WARNING: this CPU has AVX but this Embree has no AVX kernels.\n"
            << "         Ray traversal is running SSE2 BVH4 and is ~83 % of the\n"
            << "         illumination phase. A multi-ISA build of the SAME Embree\n"
            << "         version measured -7.0 % on the illumination phase of this\n"
            << "         project's calib120 arm (18 paired repeats, 18/18 same sign,\n"
            << "         2026-08-15). This is the Homebrew embree default.\n"
            << "         Speed only -- but NOT result-neutral: on a lunar polar DEM\n"
            << "         patch the ISA swap moved 75 of 260 output files, worst\n"
            << "         face-frame |dT| 0.0091 K, because grazing occlusion verdicts\n"
            << "         differ between ISA kernels. See INSTALL.md,\n"
            << "         \"Embree ISA (performance)\"." << std::endl;
    } else if (cpu_has_avx512 && !ray16) {
        std::cerr << "[Embree] NOTE: this CPU has AVX-512 but this Embree has no"
                  << " AVX-512 kernels; running " << tier << "." << std::endl;
    }

    // Embree's own verbose block is the definitive statement of which targets
    // were compiled in and which the CPU supports. It is many lines, so it is
    // opt-in. A throwaway device keeps the engine's real device untouched --
    // it is a different instance, but an identically-configured one, so its
    // target report is the same.
    //
    // The announcement goes to std::cout, not std::cerr, because Embree writes
    // its verbose block to std::cout: split across two streams, `2>log` would
    // file the announcement and the report it announces in different places.
    // cerr is flushed first so the banner above still precedes both.
    const char* v = std::getenv("PTM_EMBREE_VERBOSE");
    if (v && *v && *v != '0') {
        std::cerr.flush();
        std::cout << "[Embree] PTM_EMBREE_VERBOSE=1: Embree's own device report follows."
                  << std::endl;
        RTCDevice probe = rtcNewDevice("verbose=1");
        if (probe) rtcReleaseDevice(probe);
    }
}

}  // namespace ptm
