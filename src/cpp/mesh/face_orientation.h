#pragma once
// PTM-733 (2026-10-15): ONE definition of "which side of each face is the surface", shared by
// the engine and the mc_ref_pp view-factor referee, so the two can never disagree again.
//
// Background (PTM-732): the Moon3D 509k crater mesh is consistently wound INTO the Moon. The
// engine orients faces outward for illumination, but mc_ref_pp traced along the raw winding, so
// the paper's v4 view-factor matrix described the crater's UNDERSIDE. Nothing caught it: the
// CSR's mesh-identity hash is built from face areas, which do not depend on the winding.
//
// The two rules are the engine's own (illumination_model.cpp):
//   radial_per_face   (default)  flip a face iff dot(center, n_winding) < 0
//                                (illumination_model.cpp radial-flip block; also MeshUtils
//                                ensureOutwardNormals / auto_orient_normals).
//   winding_majority  (env PTM_ILLUM_TRUE_NORMALS=1)  keep the winding, with ONE global sign
//                                per edge-connected component chosen by a radial-outward
//                                majority vote; refused (falls back to radial_per_face) when
//                                more than 1% of faces' worth of manifold half-edges are
//                                inconsistent (illumination_model.cpp PTM-Class1 block). The
//                                union-find, the half-edge consistency count and the vote are
//                                reproduced here verbatim.
// The result is a per-face sign relative to the RAW winding (+1 keep, -1 reverse) and an FNV-1a
// digest of it, which mc_ref_pp records in its STATS, the assembler copies into the CSR
// provenance, and the engine compares against its own before it accepts a CSR.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ptm { namespace orient {

struct FaceOrientation {
    std::string rule;             // rule actually applied: "radial_per_face" | "winding_majority"
    std::string rule_requested;   // what the environment asked for
    std::vector<int8_t> sign;     // +1 keep raw winding, -1 reverse it
    size_t n_faces = 0;
    size_t n_reversed = 0;        // faces with sign -1
    size_t inconsistent = 0;      // same-direction manifold half-edges (winding_majority only)
    bool fell_back = false;       // winding_majority requested but refused (> 1% inconsistent)
    uint64_t digest = 0;          // FNV-1a 64 over n_faces bytes (1 = keep, 0 = reverse)
};

// Same parsing as illumination_model.cpp (PTM_ILLUM_TRUE_NORMALS: 1/t/T/y/Y = on).
inline bool trueNormalsRequestedFromEnv() {
    if (const char* tn = std::getenv("PTM_ILLUM_TRUE_NORMALS"))
        return tn[0]=='1'||tn[0]=='t'||tn[0]=='T'||tn[0]=='y'||tn[0]=='Y';
    return false;
}

inline uint64_t signDigest(const std::vector<int8_t>& s) {
    uint64_t h = 1469598103934665603ULL;
    for (int8_t v : s) { h ^= static_cast<uint64_t>(v > 0 ? 1 : 0); h *= 1099511628211ULL; }
    return h;
}

inline std::string hex16(uint64_t v) {
    char b[17]; std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
    return std::string(b);
}

// V: vertex positions in the body-centered frame the engine uses (the radial test needs the
// body center at the origin). F: triangles (any container of 3 ints).
template <class VArr, class FArr>
FaceOrientation computeFaceOrientation(const VArr& V, const FArr& F, bool want_winding_majority) {
    FaceOrientation r;
    const size_t nfc = F.size();
    r.n_faces = nfc;
    r.rule_requested = want_winding_majority ? "winding_majority" : "radial_per_face";
    r.sign.assign(nfc, 1);
    std::vector<std::array<double,3>> nwv(nfc), ctr(nfc);
    for (size_t i = 0; i < nfc; ++i) {
        const auto& a = V[F[i][0]]; const auto& b = V[F[i][1]]; const auto& c = V[F[i][2]];
        ctr[i] = {{ (a[0]+b[0]+c[0])/3.0, (a[1]+b[1]+c[1])/3.0, (a[2]+b[2]+c[2])/3.0 }};
        const double e1[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
        const double e2[3] = { c[0]-a[0], c[1]-a[1], c[2]-a[2] };
        std::array<double,3> n = {{ e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2],
                                    e1[0]*e2[1]-e1[1]*e2[0] }};
        const double L = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if (L > 0) { n[0]/=L; n[1]/=L; n[2]/=L; }
        nwv[i] = n;
    }
    bool use_majority = want_winding_majority;
    if (use_majority) {
        std::vector<int> parent(nfc);
        for (size_t i = 0; i < nfc; ++i) parent[i] = static_cast<int>(i);
        auto uf_find = [&parent](int x){ while (parent[x]!=x){ parent[x]=parent[parent[x]]; x=parent[x]; } return x; };
        auto uf_union = [&](int a,int b){ int ra=uf_find(a), rb=uf_find(b); if (ra!=rb) parent[ra]=rb; };
        std::unordered_map<uint64_t,std::pair<int,int>> emap;
        emap.reserve(nfc*3);
        size_t inconsistent = 0;
        for (size_t i = 0; i < nfc; ++i) {
            const int vv[3] = { static_cast<int>(F[i][0]), static_cast<int>(F[i][1]), static_cast<int>(F[i][2]) };
            for (int e = 0; e < 3; ++e) {
                int s = vv[e], d = vv[(e+1)%3];
                int dir = (s < d) ? 1 : -1;
                uint32_t lo = static_cast<uint32_t>(s < d ? s : d);
                uint32_t hi = static_cast<uint32_t>(s < d ? d : s);
                uint64_t key = (static_cast<uint64_t>(lo) << 32) | hi;
                auto it = emap.find(key);
                if (it == emap.end()) { emap.emplace(key, std::make_pair(static_cast<int>(i), dir)); }
                else {
                    uf_union(static_cast<int>(i), it->second.first);
                    if (it->second.second == dir) ++inconsistent;
                    it->second.first = static_cast<int>(i);
                }
            }
        }
        r.inconsistent = inconsistent;
        if (inconsistent > nfc / 100) {
            r.fell_back = true;            // the engine refuses TRUE normals here -> radial flip
            use_majority = false;
        } else {
            std::unordered_map<int,long> vote;
            for (size_t i = 0; i < nfc; ++i) {
                const auto& c = ctr[i];
                double dr = c[0]*nwv[i][0] + c[1]*nwv[i][1] + c[2]*nwv[i][2];
                vote[uf_find(static_cast<int>(i))] += (dr >= 0.0) ? 1 : -1;
            }
            for (size_t i = 0; i < nfc; ++i)
                r.sign[i] = (vote[uf_find(static_cast<int>(i))] >= 0) ? 1 : -1;
        }
    }
    if (!use_majority) {
        for (size_t i = 0; i < nfc; ++i) {
            const auto& c = ctr[i];
            r.sign[i] = (c[0]*nwv[i][0] + c[1]*nwv[i][1] + c[2]*nwv[i][2] < 0.0) ? -1 : 1;
        }
    }
    r.rule = use_majority ? "winding_majority" : "radial_per_face";
    for (int8_t s : r.sign) if (s < 0) ++r.n_reversed;
    r.digest = signDigest(r.sign);
    return r;
}

}}  // namespace ptm::orient
