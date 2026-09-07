// test_batch_checkpoint -- the format, the identity guard and the refusal
// surface of thermal/batch_checkpoint.h, with no mesh, no solver and no GPU.
//
// WHY EVERY REFUSAL GETS ITS OWN ARM. A partial that loads when it should not
// restores un-solved or wrong-run faces into a finished product, and nothing
// downstream can tell. That is the PTM-78 zero-column runaway's failure mode,
// and the only defense is that each refusal has actually been FIRED here.
//
// WHY THE ROUND TRIP IS BYTE-EXACT AND NOT "CLOSE". The surface series is
// float64 on the wire because the solver's own array is float64: a partial that
// quantized it would make a resumed run differ from an uninterrupted one in the
// last bits of every statistic, and the acceptance test for this whole feature
// is BYTE identity. So the round trip is compared with memcmp, not a tolerance.
//
// Build: part of the standalone C++ test set (no ctest, no pytest).

#include "../thermal/batch_checkpoint.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

using namespace thermal::batchckpt;

namespace {

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::cout << "  FAIL  " << what << "\n";
    } else {
        std::cout << "  ok    " << what << "\n";
    }
}

// A refusal arm: the read must fail AND the message must name the reason, so a
// refusal that fires for the wrong cause cannot pass as the right one.
void checkRefusedNaming(const std::string& path, const char* needle,
                        const std::string& what) {
    Identity id;
    std::vector<Record> recs;
    std::string err;
    const bool ok = read(path, id, recs, err);
    ++g_checks;
    if (ok) {
        ++g_failed;
        std::cout << "  FAIL  " << what << " -- it LOADED\n";
        return;
    }
    if (err.find(needle) == std::string::npos) {
        ++g_failed;
        std::cout << "  FAIL  " << what << " -- refused, but the message does "
                  << "not name '" << needle << "': " << err << "\n";
        return;
    }
    std::cout << "  ok    " << what << "\n";
}

std::string g_dir;

std::string p(const std::string& name) { return g_dir + "/" + name; }

Identity makeIdentity(std::uint32_t F, std::uint32_t L, std::uint32_t T,
                      std::uint32_t D, std::uint32_t BS) {
    Identity id;
    id.z.resize(L);
    for (std::uint32_t i = 0; i < L; ++i)
        id.z[i] = 0.001 * (i + 1) * (i + 1);
    id.et_seconds        = 1.234567890123e9;
    id.output_cadence_s  = 1800.0;
    id.refgrid_H         = 0.07;
    id.grid_m = 10; id.grid_n = 3; id.grid_b = 15;
    id.mesh_geomhash     = 3.14159265358979e12;
    id.config_identity_hash = 0xDEADBEEFCAFEF00Dull;
    id.body_name         = "Moon";
    id.engine_git_sha    = "0123456789abcdef0123456789abcdef01234567";
    id.n_faces_total = F;
    id.n_layers      = L;
    id.n_time_steps  = T;
    id.n_depths      = D;
    id.batch_size    = BS;
    id.sections = part::kSurfaceSeries | part::kFinalColumn | part::kDepthAccum;
    id.depth_plan_hash = 0xFEEDFACE12345678ull;
    id.depth_win_start = 0;
    id.depth_win_end   = static_cast<std::int32_t>(T);
    return id;
}

Record makeRecord(const Identity& id, std::uint32_t batch_idx,
                  std::uint32_t batch_n, unsigned seed) {
    Record r;
    r.batch_idx = batch_idx;
    r.batch_n   = batch_n;
    r.input_digest = 0xA5A50000u + batch_idx;
    // Only the SELECTED sections are populated. A record that carries a
    // deselected section is itself a refusal (see serialize()), so the fixture
    // must respect the same rule the engine does.
    if (id.sections & part::kSurfaceSeries) {
        r.surface.resize(static_cast<std::size_t>(batch_n) * id.n_time_steps);
        for (std::size_t i = 0; i < r.surface.size(); ++i)
            r.surface[i] = 100.0 + 1e-9 * static_cast<double>(i + seed);
    }
    if (id.sections & part::kFinalColumn) {
        r.column.resize(static_cast<std::size_t>(batch_n) * id.n_layers);
        for (std::size_t i = 0; i < r.column.size(); ++i)
            r.column[i] = 200.0f + 0.25f * static_cast<float>(i);
    }
    if (id.sections & part::kDepthAccum) {
        const std::size_t nd = static_cast<std::size_t>(batch_n) * id.n_depths;
        r.depth_sum.assign(nd, 0.0);
        r.depth_max.assign(nd, 0.0);
        r.depth_min.assign(nd, 0.0);
        for (std::size_t i = 0; i < nd; ++i) {
            r.depth_sum[i] = 1234.5 + i;
            r.depth_max[i] = 300.0 - 0.5 * i;
            r.depth_min[i] = 40.0 + 0.25 * i;
        }
        r.depth_count.assign(batch_n, id.n_time_steps);
    }
    return r;
}

bool sameRecord(const Record& a, const Record& b) {
    if (a.batch_idx != b.batch_idx || a.batch_n != b.batch_n ||
        a.input_digest != b.input_digest) return false;
    auto sameBytes = [](const void* x, const void* y, std::size_t n) {
        return n == 0 || std::memcmp(x, y, n) == 0;
    };
    if (a.surface.size() != b.surface.size() ||
        !sameBytes(a.surface.data(), b.surface.data(),
                   a.surface.size() * sizeof(double))) return false;
    if (a.column.size() != b.column.size() ||
        !sameBytes(a.column.data(), b.column.data(),
                   a.column.size() * sizeof(float))) return false;
    if (a.depth_sum.size() != b.depth_sum.size() ||
        !sameBytes(a.depth_sum.data(), b.depth_sum.data(),
                   a.depth_sum.size() * sizeof(double))) return false;
    if (a.depth_max.size() != b.depth_max.size() ||
        !sameBytes(a.depth_max.data(), b.depth_max.data(),
                   a.depth_max.size() * sizeof(double))) return false;
    if (a.depth_min.size() != b.depth_min.size() ||
        !sameBytes(a.depth_min.data(), b.depth_min.data(),
                   a.depth_min.size() * sizeof(double))) return false;
    if (a.depth_count.size() != b.depth_count.size() ||
        !sameBytes(a.depth_count.data(), b.depth_count.data(),
                   a.depth_count.size() * sizeof(std::uint32_t))) return false;
    return true;
}

// Flip one byte at `off` in a copy of `src` written to `dst`.
void corruptByte(const std::string& src, const std::string& dst,
                 std::size_t off) {
    std::ifstream in(src, std::ios::binary);
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
    if (off < b.size()) b[off] ^= 0xFFu;
    std::ofstream out(dst, std::ios::binary);
    out.write(reinterpret_cast<const char*>(b.data()),
              static_cast<std::streamsize>(b.size()));
}

void truncateCopy(const std::string& src, const std::string& dst,
                  std::size_t keep) {
    std::ifstream in(src, std::ios::binary);
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
    if (keep > b.size()) keep = b.size();
    std::ofstream out(dst, std::ios::binary);
    out.write(reinterpret_cast<const char*>(b.data()),
              static_cast<std::streamsize>(keep));
}

// ===========================================================================
void testRoundTrip() {
    std::cout << "\n[round trip] a partial reads back byte-exact\n";
    const Identity id = makeIdentity(/*F=*/40, /*L=*/5, /*T=*/7, /*D=*/3,
                                     /*BS=*/8);
    std::vector<Record> recs;
    recs.push_back(makeRecord(id, 0, 8, 11));
    recs.push_back(makeRecord(id, 3, 8, 22));
    recs.push_back(makeRecord(id, 4, 8, 33));
    std::string err;
    check(write(p("rt.ptmb"), id, recs, err), "write succeeds: " + err);

    Identity id2;
    std::vector<Record> recs2;
    check(read(p("rt.ptmb"), id2, recs2, err), "read succeeds: " + err);
    check(id2.hash() == id.hash(), "identity fold round-trips");
    check(id.describeMismatch(id2).empty(),
          "identity compares field-by-field equal");
    check(recs2.size() == recs.size(), "record count round-trips");
    bool all = recs2.size() == recs.size();
    for (std::size_t i = 0; all && i < recs.size(); ++i)
        all = sameRecord(recs[i], recs2[i]);
    check(all, "every record round-trips BYTE-exact (memcmp, not a tolerance)");

    // The last batch of a mesh is short. It must survive as a short record --
    // padding it to batch_size would bank faces that do not exist.
    Identity idS = makeIdentity(/*F=*/20, 5, 7, 3, 8);
    std::vector<Record> shortRecs;
    shortRecs.push_back(makeRecord(idS, 2, 4, 7));   // 20 = 8 + 8 + 4
    check(write(p("short.ptmb"), idS, shortRecs, err),
          "a SHORT final batch writes: " + err);
    Identity idS2; std::vector<Record> shortRecs2;
    check(read(p("short.ptmb"), idS2, shortRecs2, err),
          "a SHORT final batch reads: " + err);
    check(!shortRecs2.empty() && shortRecs2[0].batch_n == 4,
          "the short batch keeps batch_n=4, not padded to batch_size");
    check(!shortRecs2.empty() && sameRecord(shortRecs[0], shortRecs2[0]),
          "the short batch's payload round-trips byte-exact");
}

// ===========================================================================
void testRefusals() {
    std::cout << "\n[refusals] every one is fired, and names its cause\n";
    const Identity id = makeIdentity(40, 5, 7, 3, 8);
    std::vector<Record> recs;
    recs.push_back(makeRecord(id, 1, 8, 5));
    std::string err;
    if (!write(p("good.ptmb"), id, recs, err)) {
        std::cout << "  FATAL: could not lay down the good file: " << err << "\n";
        ++g_failed;
        return;
    }

    // 1. Not a PTMB at all.
    {
        std::ofstream o(p("notptmb.bin"), std::ios::binary);
        const std::uint32_t ptmt = 0x50544D54u;   // a real PTMT magic
        o.write(reinterpret_cast<const char*>(&ptmt), 4);
        std::vector<char> pad(4096, 0);
        o.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    }
    checkRefusedNaming(p("notptmb.bin"), "not a PTMB",
                       "a PTMT state offered as a partial is refused at the MAGIC");

    // 2. A wrong version is refused rather than parsed hopefully.
    corruptByte(p("good.ptmb"), p("badver.ptmb"), 4);
    checkRefusedNaming(p("badver.ptmb"), "PTMB version",
                       "an unknown PTMB version is refused, not guessed at");

    // 3. Corrupt the BODY. Length and magic both still pass; only the CRC
    //    catches it. This is the OneDrive placeholder class.
    {
        struct stat st{};
        ::stat(p("good.ptmb").c_str(), &st);
        corruptByte(p("good.ptmb"), p("badbody.ptmb"),
                    static_cast<std::size_t>(st.st_size) - 64);
    }
    checkRefusedNaming(p("badbody.ptmb"), "CRC32",
                       "a corrupt body is caught by the CRC (length and magic "
                       "both pass)");

    // 4. Truncated: the layout accounts for more bytes than exist.
    {
        struct stat st{};
        ::stat(p("good.ptmb").c_str(), &st);
        truncateCopy(p("good.ptmb"), p("trunc.ptmb"),
                     static_cast<std::size_t>(st.st_size) - 32);
    }
    checkRefusedNaming(p("trunc.ptmb"), "truncated",
                       "a truncated partial is refused");

    // 5. A file with EXTRA trailing bytes -- the length rule is exact both
    //    ways, because trailing slack is how a half-overwritten file looks.
    {
        std::ifstream in(p("good.ptmb"), std::ios::binary);
        std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
        b.insert(b.end(), 16, 0);
        std::ofstream out(p("long.ptmb"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(b.data()),
                  static_cast<std::streamsize>(b.size()));
    }
    checkRefusedNaming(p("long.ptmb"), "bytes on disk",
                       "a partial with trailing slack is refused, not read to "
                       "the layout's length");

    // 5b. A corrupt record COUNT must be refused BEFORE it is allocated for.
    //     One flipped byte here made read() default-construct 4.29e9 Records
    //     (~690 GB) and hang, in place of the documented CRC32 refusal.
    {
        std::ifstream in(p("good.ptmb"), std::ios::binary);
        std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
        // n_records sits at offset 24 (after magic/version/header_bytes/flags
        // and the 8-byte run identity).
        b[24] = 0xFF; b[25] = 0xFF; b[26] = 0xFF; b[27] = 0xFF;
        std::ofstream out(p("bigcount.ptmb"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(b.data()),
                  static_cast<std::streamsize>(b.size()));
    }
    checkRefusedNaming(p("bigcount.ptmb"), "record table claims",
                       "a corrupt record COUNT is refused before allocating "
                       "for it");

    // 6. Doctored identity: flip a byte inside z[] so the file's own
    //    run-identity fold no longer matches its fields. THIS is the arm that
    //    catches a hand-edited partial claiming to belong to another run.
    corruptByte(p("good.ptmb"), p("badident.ptmb"),
                detail::headerBytes(5) - 40);
    checkRefusedNaming(p("badident.ptmb"), "internally inconsistent",
                       "a doctored identity field is caught by the file's own "
                       "identity fold");

    // 7. A mis-sized payload never reaches the disk.
    {
        Identity bad = makeIdentity(40, 5, 7, 3, 8);
        std::vector<Record> r;
        r.push_back(makeRecord(bad, 0, 8, 1));
        r[0].surface.resize(r[0].surface.size() - 1);
        std::string e;
        const bool ok = write(p("never.ptmb"), bad, r, e);
        check(!ok && e.find("surface") != std::string::npos,
              "a mis-sized surface buffer is refused BEFORE any file is "
              "created: " + e);
        check(::access(p("never.ptmb").c_str(), F_OK) != 0,
              "and no file was left behind at that path");
    }

    // 8. batch_n larger than batch_size is a lie about the mesh partition.
    {
        Identity bad = makeIdentity(40, 5, 7, 3, 8);
        std::vector<Record> r;
        r.push_back(makeRecord(bad, 0, 8, 1));
        r[0].batch_n = 9;
        std::string e;
        check(!write(p("never2.ptmb"), bad, r, e) &&
                  e.find("batch_n") != std::string::npos,
              "batch_n > batch_size is refused: " + e);
    }

    // 8b. A record carrying a section the identity did not select. The write
    //     would drop it silently and the resume would read zeros, with the
    //     file's own length and checksum both agreeing -- so it is refused.
    {
        Identity bad = makeIdentity(40, 5, 7, 3, 8);
        bad.sections = part::kSurfaceSeries;      // no column, no depth
        std::vector<Record> r;
        r.push_back(makeRecord(bad, 0, 8, 1));
        r[0].column.assign(40, 1.0f);             // populated anyway
        std::string e;
        check(!write(p("never4.ptmb"), bad, r, e) &&
                  e.find("not selected") != std::string::npos,
              "a payload for a DESELECTED section is refused, not dropped: " + e);
    }

    // 9. An identity whose z[] disagrees with its own n_layers.
    {
        Identity bad = makeIdentity(40, 5, 7, 3, 8);
        bad.z.resize(4);
        std::vector<Record> r;
        std::string e;
        check(!write(p("never3.ptmb"), bad, r, e) &&
                  e.find("depth nodes") != std::string::npos,
              "an identity carrying the wrong number of depth nodes is "
              "refused: " + e);
    }
}

// ===========================================================================
void testIdentityGuard() {
    std::cout << "\n[identity] a partial from another run does not load into "
                 "this one\n";
    const Identity a = makeIdentity(40, 5, 7, 3, 8);

    // Each clause on its own, so a refusal cannot pass for the wrong reason.
    struct Case { const char* name; const char* field; Identity id; };
    std::vector<Case> cases;
    { Identity b = a; b.z[2] += 1e-12;
      cases.push_back({"a moved depth grid", "z[2]", b}); }
    { Identity b = a; b.et_seconds += 1.0;
      cases.push_back({"a different epoch", "et_seconds", b}); }
    { Identity b = a; b.output_cadence_s *= 2.0;
      cases.push_back({"a different output cadence", "output_cadence_s", b}); }
    { Identity b = a; b.mesh_geomhash += 1.0;
      cases.push_back({"a different mesh", "mesh_geomhash", b}); }
    { Identity b = a; b.config_identity_hash ^= 1ull;
      cases.push_back({"different physics", "config_identity_hash", b}); }
    { Identity b = a; b.body_name = "Europa";
      cases.push_back({"a different body", "body_name", b}); }
    { Identity b = a; b.n_time_steps += 1;
      cases.push_back({"a different window length", "n_time_steps", b}); }
    { Identity b = a; b.grid_b = 32;
      cases.push_back({"a different grid_b", "grid_b", b}); }
    { Identity b = a; b.refgrid_H *= 1.5;
      cases.push_back({"a different reference H", "refgrid_H", b}); }
    { Identity b = a; b.batch_size = 4;
      cases.push_back({"a different batch size", "batch_size", b}); }
    { Identity b = a; b.sections = part::kSurfaceSeries;
      cases.push_back({"a different section set", "sections", b}); }
    // ⭐ The depth product's own identity. n_depths alone is not enough: two
    // requests of the SAME LENGTH resolving onto different nodes are different
    // products, and neither config_identity_hash nor the input digest moves
    // with them. Without these clauses a resume could merge two of them.
    { Identity b = a; b.depth_plan_hash = 0x1234ull;
      cases.push_back({"a different depth list of the same length",
                       "depth_plan_hash", b}); }
    { Identity b = a; b.depth_win_start = 5;
      cases.push_back({"a different depth accumulation window",
                       "depth_win_start", b}); }
    { Identity b = a; b.depth_win_end = 999;
      cases.push_back({"a different depth window end", "depth_win_end", b}); }

    for (const Case& c : cases) {
        const std::string m = a.describeMismatch(c.id);
        check(!m.empty() && m.find(c.field) != std::string::npos,
              std::string(c.name) + " is named by describeMismatch (" +
                  c.field + ")");
        check(a.hash() != c.id.hash(),
              std::string(c.name) + " changes the identity fold");
    }
    check(a.describeMismatch(a).empty(),
          "an identity compares equal to itself with no mismatch lines");
}

// ===========================================================================
void testInputDigest() {
    std::cout << "\n[input digest] the partial is keyed by the BYTES its batch "
                 "integrated\n";
    const std::size_t T = 6, BN = 4, BSZ = 8, L = 3;
    std::vector<double> flux(T * BSZ, 0.0);
    for (std::size_t i = 0; i < flux.size(); ++i) flux[i] = 0.5 * i;
    std::vector<double> seed(L * BSZ, 0.0);
    for (std::size_t i = 0; i < seed.size(); ++i) seed[i] = 100.0 + i;
    std::vector<double> qb(BN, 0.018);

    const std::uint32_t d0 = inputDigest(flux.data(), BSZ, T, BN, seed.data(),
                                         BSZ, L, qb.data());
    check(d0 == inputDigest(flux.data(), BSZ, T, BN, seed.data(), BSZ, L,
                            qb.data()),
          "the digest is deterministic");

    // The discriminating cases: each input moved on its own.
    {
        std::vector<double> f2 = flux;
        f2[3 * BSZ + 1] += 1e-12;         // one timestep, one face
        check(inputDigest(f2.data(), BSZ, T, BN, seed.data(), BSZ, L, qb.data())
                  != d0,
              "a 1e-12 change in ONE flux sample changes the digest -- this is "
              "what catches a different albedo map / view factors / radiation "
              "state, none of which config_identity_hash covers");
    }
    {
        std::vector<double> s2 = seed;
        s2[L * BSZ - BSZ] = std::nextafter(s2[L * BSZ - BSZ], 1e9);
        check(inputDigest(flux.data(), BSZ, T, BN, s2.data(), BSZ, L, qb.data())
                  != d0,
              "a one-ulp change in the seeded initial column changes the digest");
    }
    // The seed's own padding lanes are excluded for the same reason as the
    // flux's: they are whatever the previous batch left behind.
    {
        std::vector<double> s2 = seed;
        for (std::size_t l = 0; l < L; ++l) s2[l * BSZ + BN] = -9999.0;
        check(inputDigest(flux.data(), BSZ, T, BN, s2.data(), BSZ, L, qb.data())
                  == d0,
              "the seed column's padding lanes past batch_n are NOT digested");
    }
    {
        std::vector<double> q2 = qb;
        q2[BN - 1] += 1e-9;
        check(inputDigest(flux.data(), BSZ, T, BN, seed.data(), BSZ, L,
                          q2.data()) != d0,
              "a change in the per-face interior heat flux changes the digest");
    }
    // Faces BEYOND batch_n are padding in the SIMD lanes and must NOT be
    // digested: they hold whatever the previous batch left, so including them
    // would make a valid resume fail at random.
    {
        std::vector<double> f2 = flux;
        for (std::size_t t = 0; t < T; ++t)
            f2[t * BSZ + BN] = 12345.0;      // lane BN, past the live faces
        check(inputDigest(f2.data(), BSZ, T, BN, seed.data(), BSZ, L, qb.data())
                  == d0,
              "the padding lanes past batch_n are NOT digested");
    }
    // No Qb map at all is a different run from a Qb map of zeros, and the
    // digest must not silently equate them.
    check(inputDigest(flux.data(), BSZ, T, BN, seed.data(), BSZ, L, nullptr)
              != d0,
          "omitting the Qb map is not the same digest as supplying one");
}

// ===========================================================================
void testCoverage() {
    std::cout << "\n[coverage] completeness is COMPUTED from the files, never "
                 "claimed\n";
    Identity id = makeIdentity(/*F=*/20, 5, 7, 0, /*BS=*/8);
    id.sections = part::kSurfaceSeries;

    Coverage cov;
    std::vector<Record> f1, f2;
    f1.push_back(makeRecord(id, 0, 8, 1));
    f2.push_back(makeRecord(id, 1, 8, 2));
    cov.add("a.ptmb", id, f1);
    check(!cov.complete(20), "16 of 20 faces is NOT complete");
    cov.add("b.ptmb", id, f2);
    check(!cov.complete(20), "still not complete without the short tail");

    std::vector<Record> f3;
    f3.push_back(makeRecord(id, 2, 4, 3));
    cov.add("c.ptmb", id, f3);
    check(cov.complete(20),
          "the short tail batch closes coverage at exactly n_faces");
    check(cov.faces_covered == 20, "faces_covered counts faces, not files");
    check(cov.has(0) && cov.has(1) && cov.has(2), "every batch is indexed");
    check(!cov.has(3), "a batch that was never banked is not claimed");
    check(cov.duplicates.empty(), "no spurious duplicate reports");

    // A crash tail: the same batch banked twice with the SAME digest is
    // tolerated and reported. With DIFFERENT digests it is a hard conflict --
    // two different integrations of the same faces.
    Coverage cov2;
    cov2.add("a.ptmb", id, f1);
    cov2.add("a-again.ptmb", id, f1);
    check(cov2.duplicates.size() == 1 &&
              cov2.duplicates[0].find("same input digest") != std::string::npos,
          "a re-banked identical batch is reported as a duplicate, not "
          "double-counted");
    check(cov2.faces_covered == 8,
          "and it does not inflate the face count (would fake completeness)");

    Coverage cov3;
    std::vector<Record> f1b = f1;
    f1b[0].input_digest ^= 0xFFu;
    cov3.add("a.ptmb", id, f1);
    cov3.add("a-conflict.ptmb", id, f1b);
    check(cov3.duplicates.size() == 1 &&
              cov3.duplicates[0].find("DIFFERENT INPUT DIGESTS") !=
                  std::string::npos,
          "two DIFFERENT integrations of one batch are flagged loudly");

    // The completeness test is against n_faces, not against a file count --
    // 3 files is the same number of files whether they cover the mesh or not.
    Coverage cov4;
    cov4.add("a.ptmb", id, f1);
    cov4.add("b.ptmb", id, f2);
    cov4.add("b-again.ptmb", id, f2);
    check(!cov4.complete(20),
          "three files that cover 16 faces do NOT pass as complete");
}

// ===========================================================================
void testSizing() {
    std::cout << "\n[sizing] the disclosure is arithmetic, not a guess\n";
    // The lunar polar production shape the plan sizes against.
    Identity id = makeIdentity(/*F=*/2397890, /*L=*/23, /*T=*/1440, /*D=*/0,
                               /*BS=*/8);
    id.sections = part::kSurfaceSeries | part::kFinalColumn;
    check(bytesPerFace(id) == 1440ull * 8 + 23ull * 4,
          "bytes per face = series + column");
    const std::uint64_t total = estimateBytes(id);
    // 11612 B/face * 2,397,890 faces
    check(total == 11612ull * 2397890ull, "the total is exact, not rounded");
    check(total > 25ull * 1024 * 1024 * 1024,
          "a production window's partial set is > 25 GiB -- the plan's "
          "'210 MiB' sized the COLUMN only and is ~125x low");

    Identity col = id;
    col.sections = part::kFinalColumn;
    check(estimateBytes(col) == 92ull * 2397890ull,
          "the column-only figure reproduces the plan's 210 MiB number");
    check(humanBytes(estimateBytes(col)).find("MiB") != std::string::npos,
          "humanBytes reports the column set in MiB");
    check(humanBytes(total).find("GiB") != std::string::npos,
          "and the full set in GiB");
}

// ===========================================================================
// The restart protocol. This is the arm set that stands behind the crash-resume
// claim: a session opened over a half-written partial set must restore exactly
// the banked batches, refuse anything it cannot verify, and end with every face
// covered exactly once.
// ===========================================================================
void testSession() {
    std::cout << "\n[session] the restart protocol: scan, verify, skip, "
                 "complete\n";
    Identity id = makeIdentity(/*F=*/20, /*L=*/5, /*T=*/7, /*D=*/0, /*BS=*/8);
    id.sections = part::kSurfaceSeries | part::kFinalColumn;
    const std::string dir = g_dir + "/sess";
    std::ostringstream log;
    std::string err;

    // --- pass 1: a run that dies after banking 2 of 3 batches --------------
    {
        Session s;
        check(s.open(dir, id, /*max_gib=*/64.0, /*flush_mib=*/0.0, log, err),
              "a fresh session opens on an empty directory: " + err);
        check(s.restoredBatches() == 0, "nothing is restored on a fresh start");
        s.setFlushBytes(1);   // flush every stage, i.e. the finest crash grain
        check(s.stage(Record(makeRecord(id, 0, 8, 1)), err),
              "batch 0 banks: " + err);
        check(s.stage(Record(makeRecord(id, 1, 8, 2)), err),
              "batch 1 banks: " + err);
        // batch 2 never banked -- the process "dies" here.
        check(!s.finish(log, err) && err.find("INCOMPLETE") != std::string::npos,
              "finishing with a coverage hole REFUSES and says so: " + err);
    }

    // --- pass 2: the resume ------------------------------------------------
    {
        Session s;
        check(s.open(dir, id, 64.0, /*flush_mib=*/0.0, log, err),
              "the resume session opens over the surviving partials: " + err);
        check(s.restoredBatches() == 2 && s.restoredFaces() == 16,
              "it restores exactly the 2 banked batches / 16 faces");
        std::uint32_t d = 0;
        check(s.banked(0, &d) && d == 0xA5A50000u + 0,
              "batch 0 is banked and reports the input digest it was written "
              "with");
        check(s.banked(1, &d) && d == 0xA5A50000u + 1, "batch 1 likewise");
        check(!s.banked(2, &d), "batch 2 is NOT claimed -- it never landed");

        Record got;
        check(s.takeRecord(0, got), "the payload for batch 0 comes back");
        check(sameRecord(makeRecord(id, 0, 8, 1), got),
              "and it is BYTE-identical to what was banked");
        check(!s.takeRecord(0, got),
              "a consumed payload is released, not handed out twice");

        // Integrate the rest and close the mesh.
        s.setFlushBytes(1);
        check(s.stage(Record(makeRecord(id, 2, 4, 3)), err),
              "the missing short tail batch banks: " + err);
        check(s.finish(log, err),
              "coverage is now complete across TWO processes: " + err);
    }

    // --- the manifest that lies -------------------------------------------
    {
        // Name a partial that is not there. This is the "doctored manifest"
        // case: the index claims a batch is banked that nothing can verify.
        std::ifstream in(dir + "/manifest.json");
        std::string body((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        const std::size_t lb = body.find('[', body.find("\"files\""));
        body.insert(lb + 1, "\"part_g9999_s999999.ptmb\", ");
        std::ofstream out(dir + "/manifest.json");
        out << body;
        out.close();

        Session s;
        const bool ok = s.open(dir, id, 64.0, /*flush_mib=*/0.0, log, err);
        check(!ok && err.find("manifest") != std::string::npos &&
                  err.find("part_g9999_s999999.ptmb") != std::string::npos,
              "a manifest naming a partial that does not validate REFUSES the "
              "resume, and names the file: " + err);
    }

    // --- a manifest that is merely BEHIND is a crash tail, not a failure ---
    {
        std::ofstream out(dir + "/manifest.json");
        out << "{\n  \"schema\": \"ptm3d-batch-partial-manifest/1\",\n"
            << "  \"files\": []\n}\n";
        out.close();
        std::ostringstream lg;
        Session s;
        check(s.open(dir, id, 64.0, /*flush_mib=*/0.0, lg, err),
              "an EMPTY manifest beside valid partials still resumes: " + err);
        check(s.restoredBatches() == 3,
              "and the partials on disk are the authority (3 batches)");
        check(lg.str().find("crash tail") != std::string::npos,
              "the disagreement is REPORTED, not swallowed");
    }

    // --- a partial from a different run ------------------------------------
    {
        Identity other = id;
        other.et_seconds += 86400.0;    // a different epoch, same everything else
        std::vector<Record> r;
        r.push_back(makeRecord(other, 5, 8, 9));
        std::string e;
        check(write(dir + "/part_g0500_s000000.ptmb", other, r, e),
              "lay down a partial from another run: " + e);
        Session s;
        const bool ok = s.open(dir, id, 64.0, /*flush_mib=*/0.0, log, err);
        check(!ok && err.find("DIFFERENT computation") != std::string::npos &&
                  err.find("et_seconds") != std::string::npos,
              "a partial from another run REFUSES the resume and names the "
              "failing field: " + err);
        ::unlink((dir + "/part_g0500_s000000.ptmb").c_str());
    }

    // --- a corrupt partial -------------------------------------------------
    {
        // Restore a sane manifest first, so the refusal below is attributable
        // to the corruption and not to the index.
        Session fix;
        std::ostringstream lg;
        check(fix.open(dir, id, 64.0, /*flush_mib=*/0.0, lg, err),
              "the directory reopens cleanly before the corruption arm: " + err);

        std::vector<std::string> names;
        DIR* dp = ::opendir(dir.c_str());
        while (struct dirent* e = ::readdir(dp)) {
            const std::string n = e->d_name;
            if (n.size() > 5 && n.compare(n.size() - 5, 5, ".ptmb") == 0)
                names.push_back(n);
        }
        ::closedir(dp);
        std::sort(names.begin(), names.end());
        check(!names.empty(), "there is a partial to corrupt");
        if (!names.empty()) {
            const std::string victim = dir + "/" + names[0];
            struct stat st{};
            ::stat(victim.c_str(), &st);
            corruptByte(victim, victim + ".bak", 0);   // read into a copy
            corruptByte(victim, victim,
                        static_cast<std::size_t>(st.st_size) - 40);
            ::unlink((victim + ".bak").c_str());
            Session s;
            const bool ok = s.open(dir, id, 64.0, /*flush_mib=*/0.0, log, err);
            check(!ok && err.find("CRC32") != std::string::npos &&
                      err.find("REFUSING to resume") != std::string::npos,
                  "a corrupt partial REFUSES the resume rather than loading "
                  "what it can: " + err);
        }
    }

    // --- the size ceiling --------------------------------------------------
    {
        Identity big = makeIdentity(/*F=*/2397890, 23, 1440, 0, 8);
        big.sections = part::kSurfaceSeries | part::kFinalColumn;
        Session s;
        std::ostringstream lg;
        const bool ok = s.open(g_dir + "/big", big, /*max_gib=*/1.0, /*flush_mib=*/0.0, lg, err);
        check(!ok && err.find("ceiling") != std::string::npos,
              "a projected partial set over the ceiling REFUSES before the "
              "run, naming the size: " + err);
        check(err.find("GiB") != std::string::npos,
              "and the refusal quotes the projected size in GiB");
        Session s2;
        check(s2.open(g_dir + "/big2", big, /*max_gib=*/0.0, /*flush_mib=*/0.0, lg, err),
              "a ceiling of 0 disables the check (opt-out is explicit): " + err);
        check(lg.str().find("granularity is ONE FACE BATCH") !=
                  std::string::npos,
              "the banner states the granularity, so nobody assumes per-step "
              "bounding");
    }
}

}  // namespace

int main() {
    char tmpl[] = "/tmp/ptm_batchckpt_XXXXXX";
    const char* d = ::mkdtemp(tmpl);
    if (!d) {
        std::cerr << "FATAL: cannot create a scratch directory\n";
        return 2;
    }
    g_dir = d;
    std::cout << "test_batch_checkpoint -- PTMB partial format, identity guard, "
                 "refusals\n";
    std::cout << "scratch: " << g_dir << "\n";

    testRoundTrip();
    testRefusals();
    testIdentityGuard();
    testInputDigest();
    testCoverage();
    testSizing();
    testSession();

    std::cout << "\n" << (g_checks - g_failed) << "/" << g_checks
              << " checks passed\n";
    if (g_failed) {
        std::cout << "FAILED (" << g_failed << ")\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
