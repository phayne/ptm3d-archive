// Shadow-ray rise-epsilon unit test: the OUT-OF-PLANE RISE discriminator vs the 1 cm ray-origin offset
// on a synthetic steep sunward wall inside a rim shadow. Reproduces the exact algorithm in
// IlluminationModel::traceOccludedRiseEps (iterative rtcIntersect1 from the TRUE centroid, tnear~0,
// NO metric offset, NO ring skip; a hit is a real occluder iff its rise dz = (P_hit - c_src)*n_src
// >= h_eps = rise_keps * local_edge, else it is self-relief and the march continues past it) on a
// hand-built Embree4 scene, and shows:
//   (A) the production offset (fc + 2cm*n + 1cm*s, tnear 1cm) HOPS the real 2.47 mm rim -> LEAK;
//   (B) the 0.12 mm coplanar self-relief neighbor is SKIPPED (dz < h_eps) -> no acne;
//   (C) the 2.47 mm real rim occluder is DETECTED (dz >= h_eps) -> shadowed;
//   (D) a near-coplanar 0.10 mm real occluder (dz < h_eps) is SKIPPED -> lit. This DOCUMENTS the
//       ~10% near-coplanar residual: a single-hit dz test cannot separate a 0.1 mm-rise genuine
//       occluder from self-relief. It is a known limitation of a single-hit rise test, NOT a bug
//       (the per-face horizon/skyline integral is the more robust cure for that tail).
// Geometry mirrors the diagnosed leak population: a near-vertical wall facing the ~6.8deg sun near-
// normally (cos_inc up to 1), a real occluder rising ~2.47 mm out of the wall's own tangent plane
// (median of the 107 float64-confirmed spurious leaks) vs the ~0.12 mm coplanar self-relief -- the
// ~20x rise separation the offset (distance) and self-skip (topology) both miss.
//
// Standalone build/run (no CMake target; Embree4 at /usr/local per CMakeLists):
//   DEVELOPER_DIR=/Library/Developer/CommandLineTools clang++ -std=c++17 -O2 -I/usr/local/include \
//       src/cpp/tests/test_riseeps_occlusion.cpp /usr/local/lib/libembree4.dylib \
//       -o /tmp/test_riseeps_occlusion && /tmp/test_riseeps_occlusion
#include <embree4/rtcore.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

using V3 = std::array<float,3>;

// Sun at azimuth +x, elevation el (deg); up = +z. A near-vertical wall with normal +x sees this
// grazing sun near-normally (cos_inc = n.sun ~ cos(el) ~ 1), exactly like the leak population.
static std::array<double,3> sunDir(double el_deg) {
    double e = el_deg * M_PI / 180.0;
    return {{ std::cos(e), 0.0, std::sin(e) }};
}

// Build a scene: source wall (prim 0, x=0 plane, normal +x), plus any subset of parallel occluding
// walls at chosen x-offsets. A wall parallel to the source at x=X rises exactly X out of the source
// tangent plane, so the ray's hit there has dz ~ X (dz = t*(sun.n) and sun.n ~ 1). primIDs in build
// order: source=0, then each requested wall as two triangles.
// OPTION 3: per-prim double vertices, captured at scene build so the rise-eps2 OR layer can
// recompute the hit-triangle max rise + a float64 barycentric-interior test (mirrors production
// face_verts_local_). g_tri[prim] = {A,B,C} for the prim built in order.
static std::vector<std::array<std::array<double,3>,3>> g_tri;

static RTCScene buildScene(RTCDevice dev, const std::vector<double>& wall_x) {
    RTCScene scene = rtcNewScene(dev);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    RTCGeometry geom = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);

    std::vector<V3> verts;
    std::vector<std::array<unsigned,3>> tris;
    g_tri.clear();
    auto addTri = [&](V3 a, V3 b, V3 c) {
        unsigned i0 = (unsigned)verts.size(); verts.push_back(a);
        unsigned i1 = (unsigned)verts.size(); verts.push_back(b);
        unsigned i2 = (unsigned)verts.size(); verts.push_back(c);
        tris.push_back({i0,i1,i2});
        g_tri.push_back({{ {{a[0],a[1],a[2]}}, {{b[0],b[1],b[2]}}, {{c[0],c[1],c[2]}} }});
    };
    // prim 0: source near-vertical wall at x=0, centroid (0,0,0), normal +x (edges ~2 mm).
    addTri({0,-0.001f,-0.001f}, {0,0.001f,-0.001f}, {0,0,0.002f});
    // each requested occluding wall: a parallel vertical wall at x=X, spanning +/-5 mm in y and z so
    // the y=0 center ray (which hits x=X at z ~ X*tan(el) < 0.3 mm) lands on it.
    for (double x : wall_x) {
        float xf = (float)x;
        addTri({xf,-0.005f,-0.005f}, {xf,0.005f,-0.005f}, {xf,0.005f,0.005f});
        addTri({xf,-0.005f,-0.005f}, {xf,0.005f,0.005f}, {xf,-0.005f,0.005f});
    }
    V3* vb = (V3*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                          sizeof(V3), verts.size());
    for (size_t i=0;i<verts.size();++i) vb[i]=verts[i];
    unsigned* ib = (unsigned*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                                      sizeof(unsigned)*3, tris.size());
    for (size_t i=0;i<tris.size();++i){ ib[i*3]=tris[i][0]; ib[i*3+1]=tris[i][1]; ib[i*3+2]=tris[i][2]; }
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);
    return scene;
}

// OPTION 3 geometry: source wall (prim 0) + ONE TILTED occluder triangle (prim 1) that the center
// ray CROSSES near its low shared edge (hit-point rise ~0.2 mm < h_eps) while a far vertex rises to
// ~3 mm (>> h_eps). This is the near-coplanar residual the single hit-POINT dz test misses but the
// hit-TRIANGLE rise catches -- the exact case OPTION 3 targets.
static RTCScene buildSceneTilted(RTCDevice dev) {
    RTCScene scene = rtcNewScene(dev);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    RTCGeometry geom = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
    std::vector<V3> verts; std::vector<std::array<unsigned,3>> tris;
    g_tri.clear();
    auto addTri = [&](V3 a, V3 b, V3 c) {
        unsigned i0=(unsigned)verts.size(); verts.push_back(a);
        unsigned i1=(unsigned)verts.size(); verts.push_back(b);
        unsigned i2=(unsigned)verts.size(); verts.push_back(c);
        tris.push_back({i0,i1,i2});
        g_tri.push_back({{ {{a[0],a[1],a[2]}}, {{b[0],b[1],b[2]}}, {{c[0],c[1],c[2]}} }});
    };
    addTri({0,-0.001f,-0.001f}, {0,0.001f,-0.001f}, {0,0,0.002f});            // prim 0: source wall
    // prim 1: low edge AB at x=0.2 mm (z=0), far vertex C at x=3 mm (z=+10 mm). Center ray from the
    // origin (dir ~ +x, +z 6.8deg) crosses near AB at x~0.21 mm; C rises to 3 mm.
    addTri({0.0002f,-0.005f,0.0f}, {0.0002f,0.005f,0.0f}, {0.003f,0.0f,0.010f});
    V3* vb=(V3*)rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(V3),verts.size());
    for(size_t i=0;i<verts.size();++i) vb[i]=verts[i];
    unsigned* ib=(unsigned*)rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(unsigned)*3,tris.size());
    for(size_t i=0;i<tris.size();++i){ ib[i*3]=tris[i][0]; ib[i*3+1]=tris[i][1]; ib[i*3+2]=tris[i][2]; }
    rtcCommitGeometry(geom); rtcAttachGeometry(scene,geom); rtcReleaseGeometry(geom); rtcCommitScene(scene);
    return scene;
}

struct Hit { bool hit; unsigned prim; float t; };
static Hit nearestHit(RTCScene s, std::array<double,3> o, std::array<double,3> d, float tnear) {
    double dl = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    RTCRayHit rh{};
    rh.ray.org_x=(float)o[0]; rh.ray.org_y=(float)o[1]; rh.ray.org_z=(float)o[2];
    rh.ray.dir_x=(float)(d[0]/dl); rh.ray.dir_y=(float)(d[1]/dl); rh.ray.dir_z=(float)(d[2]/dl);
    rh.ray.tnear=tnear; rh.ray.tfar=1e30f; rh.ray.mask=0xFFFFFFFFu;
    rh.hit.geomID=RTC_INVALID_GEOMETRY_ID; rh.hit.primID=RTC_INVALID_GEOMETRY_ID;
    RTCIntersectArguments a; rtcInitIntersectArguments(&a); a.flags=RTC_RAY_QUERY_FLAG_INCOHERENT;
    rtcIntersect1(s, &rh, &a);
    return { rh.hit.geomID!=RTC_INVALID_GEOMETRY_ID, rh.hit.primID, rh.ray.tfar };
}

// Production offset path: origin = fc + 2cm*n + 1cm*s, tnear = 1cm floor. Occluded iff any hit.
static bool occludedBuggy(RTCScene s, std::array<double,3> fc, std::array<double,3> n,
                          std::array<double,3> sd) {
    double dl = std::sqrt(sd[0]*sd[0]+sd[1]*sd[1]+sd[2]*sd[2]);
    std::array<double,3> shat = {sd[0]/dl, sd[1]/dl, sd[2]/dl};
    const double eps_n = 0.02, eps_d = 0.01;   // 2x/1x of the 1cm floor
    std::array<double,3> o = { fc[0]+n[0]*eps_n+shat[0]*eps_d,
                               fc[1]+n[1]*eps_n+shat[1]*eps_d,
                               fc[2]+n[2]*eps_n+shat[2]*eps_d };
    return nearestHit(s, o, sd, 0.01f).hit;
}

// Rise-epsilon (the traceOccludedRiseEps algorithm): march from the centroid (tnear~0), rejecting a
// hit as self-relief iff prim==src OR its rise dz = t*(dir_hat . n_src) < h_eps; the first hit with
// dz >= h_eps is a real occluder. Reports the occluder's dz and the largest self-relief dz skipped.
static bool occludedRiseEps(RTCScene s, std::array<double,3> fc, std::array<double,3> sd,
                            std::array<double,3> ns, unsigned src, double h_eps,
                            unsigned& occ_prim, double& occ_dz, double& max_skip_dz) {
    double dl = std::sqrt(sd[0]*sd[0]+sd[1]*sd[1]+sd[2]*sd[2]);
    std::array<double,3> dh = {sd[0]/dl, sd[1]/dl, sd[2]/dl};
    double dir_dot_n = dh[0]*ns[0]+dh[1]*ns[1]+dh[2]*ns[2];
    float tnear = 0.0f; occ_prim = RTC_INVALID_GEOMETRY_ID; occ_dz = 0.0; max_skip_dz = 0.0;
    for (int it=0; it<64; ++it) {
        Hit h = nearestHit(s, fc, sd, tnear);
        if (!h.hit) return false;                       // clear to the sun
        double dz = (double)h.t * dir_dot_n;
        if (h.prim==src || dz < h_eps) {                // self / self-relief -> skip, march past it
            if (h.prim != src) max_skip_dz = std::max(max_skip_dz, dz);
            float tn = h.t + std::max(h.t*1e-4f, 1e-6f);
            if (!(tn > tnear)) return false;
            tnear = tn; continue;
        }
        occ_prim = h.prim; occ_dz = dz; return true;    // real occluder (rises >= h_eps out of plane)
    }
    return false;
}

// Rise-epsilon OPTION 3 (use_rise_eps2_): identical march, but a would-be-skip hit (prim!=src,
// dz<h_eps) STILL blocks iff it is a float64-INTERIOR crossing (bary_min > bary_eps) of a triangle
// whose max-vertex rise >= h_eps. Mirrors the production traceOccludedRiseEps OR layer exactly.
static bool occludedRiseEps2(RTCScene s, std::array<double,3> fc, std::array<double,3> sd,
                             std::array<double,3> ns, unsigned src, double h_eps, double bary_eps) {
    double dl = std::sqrt(sd[0]*sd[0]+sd[1]*sd[1]+sd[2]*sd[2]);
    std::array<double,3> dh = {sd[0]/dl, sd[1]/dl, sd[2]/dl};
    double dir_dot_n = dh[0]*ns[0]+dh[1]*ns[1]+dh[2]*ns[2];
    float tnear = 0.0f;
    for (int it=0; it<64; ++it) {
        Hit h = nearestHit(s, fc, sd, tnear);
        if (!h.hit) return false;
        double dz = (double)h.t * dir_dot_n;
        bool self_or_low = (h.prim==src) || (dz < h_eps);
        if (self_or_low && h.prim!=src && h.prim < g_tri.size()) {
            const auto& tv = g_tri[h.prim];
            double tri_max = -1e300;
            for (int k=0;k<3;++k){
                double rk=(tv[k][0]-fc[0])*ns[0]+(tv[k][1]-fc[1])*ns[1]+(tv[k][2]-fc[2])*ns[2];
                if(rk>tri_max) tri_max=rk;
            }
            if (tri_max >= h_eps) {
                const auto&A=tv[0]; const auto&B=tv[1]; const auto&C=tv[2];
                double e1x=B[0]-A[0],e1y=B[1]-A[1],e1z=B[2]-A[2];
                double e2x=C[0]-A[0],e2y=C[1]-A[1],e2z=C[2]-A[2];
                double px=dh[1]*e2z-dh[2]*e2y, py=dh[2]*e2x-dh[0]*e2z, pz=dh[0]*e2y-dh[1]*e2x;
                double det=e1x*px+e1y*py+e1z*pz;
                if (std::fabs(det)>1e-30){
                    double invd=1.0/det;
                    double sx=fc[0]-A[0],sy=fc[1]-A[1],sz=fc[2]-A[2];
                    double u=(sx*px+sy*py+sz*pz)*invd;
                    double qx=sy*e1z-sz*e1y, qy=sz*e1x-sx*e1z, qz=sx*e1y-sy*e1x;
                    double vv=(dh[0]*qx+dh[1]*qy+dh[2]*qz)*invd;
                    double bmin=std::min(u,std::min(vv,1.0-u-vv));
                    if (bmin > bary_eps) self_or_low = false;   // real interior occluder
                }
            }
        }
        if (self_or_low) {
            float tn=h.t+std::max(h.t*1e-4f,1e-6f); if(!(tn>tnear)) return false; tnear=tn; continue;
        }
        return true;
    }
    return false;
}

// Raw first non-source hit and its dz -- used to MEASURE each wall's out-of-plane rise directly.
static bool firstNonSrcDz(RTCScene s, std::array<double,3> fc, std::array<double,3> sd,
                          std::array<double,3> ns, unsigned src, double& dz_out) {
    double dl = std::sqrt(sd[0]*sd[0]+sd[1]*sd[1]+sd[2]*sd[2]);
    std::array<double,3> dh = {sd[0]/dl, sd[1]/dl, sd[2]/dl};
    double dir_dot_n = dh[0]*ns[0]+dh[1]*ns[1]+dh[2]*ns[2];
    float tnear=0.0f;
    for (int it=0; it<64; ++it) {
        Hit h = nearestHit(s, fc, sd, tnear);
        if (!h.hit) return false;
        if (h.prim==src) { float tn=h.t+std::max(h.t*1e-4f,1e-6f); if(!(tn>tnear)) return false; tnear=tn; continue; }
        dz_out = (double)h.t * dir_dot_n; return true;
    }
    return false;
}

int main() {
    RTCDevice dev = rtcNewDevice("");

    const std::array<double,3> fc = {0,0,0};              // TRUE centroid (rise-eps starts here)
    const std::array<double,3> n  = {1,0,0};              // steep wall normal, sunward, ~90deg slope
    const auto sd = sunDir(6.8);                          // grazing max-elevation sun (near-normal on the wall)
    const unsigned src = 0;                               // source wall primID
    // h_eps = rise_keps * local_edge. Production defaults: k=0.2, mesh median edge ~2.5 mm -> 0.5 mm.
    const double local_edge = 0.0025;                     // 2.5 mm facet
    const double rise_keps  = 0.2;
    const double h_eps = rise_keps * local_edge;          // 0.5 mm

    // Rise magnitudes matching the diag: self-relief 0.12 mm, real rim 2.47 mm, near-coplanar 0.10 mm.
    const double X_SELF = 0.00012, X_RIM = 0.00247, X_NEAR = 0.00010;

    RTCScene sRim   = buildScene(dev, {X_SELF, X_RIM});   // self-relief + real rim
    RTCScene sSelf  = buildScene(dev, {X_SELF});          // self-relief only (acne check)
    RTCScene sNear  = buildScene(dev, {X_NEAR});          // near-coplanar real occluder (residual)
    RTCScene sOpen  = buildScene(dev, {});                // bare wall
    RTCScene sRimO  = buildScene(dev, {X_RIM});           // real rim only (dz measurement)

    int pass=0, fail=0;
    auto check=[&](const char* name, bool got, bool want){
        if(got==want){++pass; printf("  PASS %-56s (%s)\n",name,got?"SHADOWED":"lit");}
        else {++fail; printf("  FAIL %-56s got %s want %s\n",name,got?"SHADOWED":"lit",want?"SHADOWED":"lit");}
    };
    auto checkd=[&](const char* name, double got, double lo, double hi){
        if(got>=lo && got<=hi){++pass; printf("  PASS %-56s (%.4f mm in [%.3f,%.3f])\n",name,got*1000,lo*1000,hi*1000);}
        else {++fail; printf("  FAIL %-56s got %.4f mm want [%.3f,%.3f]\n",name,got*1000,lo*1000,hi*1000);}
    };

    printf("Rise-epsilon occlusion test (steep sunward wall, 6.8deg sun; h_eps = %.1f x %.1f mm = %.2f mm)\n",
           rise_keps, local_edge*1000, h_eps*1000);

    // -- MEASURE the separating quantity: self-relief rise vs real occluder rise --
    double dz_self=0, dz_rim=0;
    firstNonSrcDz(sSelf, fc, sd, n, src, dz_self);
    firstNonSrcDz(sRimO, fc, sd, n, src, dz_rim);
    printf("-- out-of-plane rise dz (the physical discriminator; self-relief << h_eps << real rim) --\n");
    checkd("self-relief neighbor rise dz (expect ~0.12 mm, < h_eps)", dz_self, 0.00008, 0.00016);
    checkd("real rim occluder rise dz (expect ~2.47 mm, >= h_eps)",   dz_rim,  0.00230, 0.00260);

    // -- rim occluder PRESENT: face is geometrically shadowed; correct = SHADOWED --
    printf("-- rim occluder PRESENT (correct = SHADOWED) --\n");
    // (A) production offset LEAKS: lit when the 2.47 mm rim must shadow it (offset hops the rim).
    bool leak = occludedBuggy(sRim, fc, n, sd);
    check("BUGGY offset (2cm n + 1cm s, tnear 1cm) -> LEAK (lit)", leak, false);
    printf("     [buggy offset lifts the origin ~29 mm off the wall plane -> above the 2.47 mm rim: %s]\n",
           leak?"SHADOWED":"lit (the diagnosed leak)");
    // (B)+(C) rise-eps skips the 0.12 mm self-relief and DETECTS the 2.47 mm rim -> SHADOWED.
    unsigned op; double odz, msdz;
    bool ss = occludedRiseEps(sRim, fc, sd, n, src, h_eps, op, odz, msdz);
    check("RISE-EPS skips 0.12mm self-relief, detects 2.47mm rim", ss, true);
    printf("     [rise-eps skipped self-relief up to %.3f mm (< %.2f mm), then blocked on rim primID=%u at dz=%.3f mm]\n",
           msdz*1000, h_eps*1000, op, odz*1000);

    // -- rim occluder ABSENT: open sunward wall; correct = lit; acne check --
    printf("-- self-relief ONLY, no rim (correct = lit; acne check) --\n");
    // (B) rise-eps stays LIT on the near-coplanar self-relief neighbor -> no acne.
    bool acne = occludedRiseEps(sSelf, fc, sd, n, src, h_eps, op, odz, msdz);
    check("RISE-EPS open wall + 0.12mm self-relief stays lit (no acne)", acne, false);
    // Bare wall stays lit too (baseline).
    check("RISE-EPS bare wall lit", occludedRiseEps(sOpen, fc, sd, n, src, h_eps, op, odz, msdz), false);
    // Production offset also lit on the bare wall (consistent baseline).
    check("BUGGY offset bare wall lit", occludedBuggy(sOpen, fc, n, sd), false);

    // -- DOCUMENTED RESIDUAL: a genuine 0.10 mm-rise occluder (< h_eps) is skipped -> lit --
    printf("-- near-coplanar 0.10 mm REAL occluder (dz < h_eps): DOCUMENTED ~10%% residual --\n");
    bool nearres = occludedRiseEps(sNear, fc, sd, n, src, h_eps, op, odz, msdz);
    check("RISE-EPS near-coplanar 0.10mm occluder SKIPPED (residual, lit)", nearres, false);
    printf("     [KNOWN LIMITATION, NOT a bug: a single-hit dz test cannot separate a %.2f mm-rise\n"
           "      genuine occluder from self-relief; ~10%% of real occluders rise < 0.15 mm. The\n"
           "      per-face horizon/skyline integral is the more robust cure for this near-coplanar tail.]\n",
           X_NEAR*1000);

    // -- OPTION 3 (rise-eps2): the TILTED near-coplanar occluder the single hit-POINT dz misses --
    printf("-- OPTION 3 (rise-eps2): tilted occluder crossed near its low edge (dz~0.2mm < h_eps) but\n"
           "   rising to ~3mm at a far vertex; single-hit dz MISSES it, the hit-TRIANGLE rise CATCHES it --\n");
    const double bary_eps = 0.0;   // production default: strictly-inside interior test
    { // rebuild so g_tri matches the scene being queried (g_tri is scene-scoped)
        RTCScene sT = buildSceneTilted(dev);
        double dz_t=0; firstNonSrcDz(sT, fc, sd, n, src, dz_t);
        checkd("tilted occluder hit-POINT rise dz (expect ~0.2 mm, < h_eps)", dz_t, 0.00010, 0.00045);
        check("BASE rise-eps SKIPS the tilted occluder (dz<h_eps) -> LEAK (lit)",
              occludedRiseEps(sT, fc, sd, n, src, h_eps, op, odz, msdz), false);
        check("rise-eps2 CATCHES it via interior hit-TRIANGLE rise -> SHADOWED",
              occludedRiseEps2(sT, fc, sd, n, src, h_eps, bary_eps), true);
        rtcReleaseScene(sT);
    }
    { RTCScene sSb = buildScene(dev, {X_SELF});                     // self-relief only: tri_max<h_eps
      check("rise-eps2 open wall + 0.12mm self-relief stays lit (NO acne)",
            occludedRiseEps2(sSb, fc, sd, n, src, h_eps, bary_eps), false); rtcReleaseScene(sSb); }
    { RTCScene sOb = buildScene(dev, {});
      check("rise-eps2 bare wall lit (NO acne)",
            occludedRiseEps2(sOb, fc, sd, n, src, h_eps, bary_eps), false); rtcReleaseScene(sOb); }
    { RTCScene sRb = buildScene(dev, {X_SELF, X_RIM});             // superset: still catches the 2.47mm rim
      check("rise-eps2 still detects the 2.47mm rim (superset of base) -> SHADOWED",
            occludedRiseEps2(sRb, fc, sd, n, src, h_eps, bary_eps), true); rtcReleaseScene(sRb); }

    rtcReleaseScene(sRim); rtcReleaseScene(sSelf); rtcReleaseScene(sNear);
    rtcReleaseScene(sOpen); rtcReleaseScene(sRimO); rtcReleaseDevice(dev);
    printf("\n%d passed, %d failed\n", pass, fail);
    printf("Summary: the offset (distance) hops the 2.47 mm rim (LEAK); rise-epsilon separates the\n"
           "0.12 mm self-relief (skipped, no acne) from the 2.47 mm real rim (detected, shadowed) by\n"
           "out-of-plane RISE. The near-coplanar 0.10 mm occluder is the documented ~10%% residual.\n");
    return fail==0 ? 0 : 1;
}
