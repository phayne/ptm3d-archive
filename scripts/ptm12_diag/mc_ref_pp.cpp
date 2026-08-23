// PTM-12 CORRECTED occlusion-aware MC VF referee (Fermi's corrections folded in).
//   - 0.1 mm ray offset (default; param), forced-outward normals
//   - grazing guard: rise-march skips self-relief hits (out-of-plane rise < k*edge), re-casts
//   - deterministic Halton QMC (bit-reproducible, no std::rand)
// Usage: mc_ref <mesh.ply> <tmax.bin> <facelist.txt> <nrays> [offset_m=1e-4] [keps=0.2]
//              [min_hits_or_legacy_frac] [outbin]
//   env PTM_PP_RING_PLY=<ring.ply>  attaches a second mesh as an OCCLUDER ONLY: rays that hit
//     it are treated as blocked (not sky) but are NEVER recorded as partners -- so no partner
//     index >= core-face-count can ever enter the output .bin. Distinguished by Embree geomID,
//     NOT primID, so a ring face whose primID coincides with the launch face index cannot be
//     mistaken for a self-hit.
// Out per face: SumF F_sky F_warm384 Qir  (occlusion-exact, grazing-robust)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <embree4/rtcore.h>
#include <unordered_map>
#include <algorithm>  // std::max/min({...}) initializer-list overloads: NOT transitive on Linux libstdc++
using V3=std::array<double,3>;
static double dot(const V3&a,const V3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V3 cross(const V3&a,const V3&b){return{{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
static V3 nrm(const V3&a){double L=std::sqrt(dot(a,a));return L>0?V3{{a[0]/L,a[1]/L,a[2]/L}}:a;}
static double halton(int i,int b){double f=1,r=0; while(i>0){f/=b; r+=f*(i%b); i/=b;} return r;}

// ---- fail-loud, header-aware binary-little-endian PLY loader ----------------------------------
// Proves the byte layout it deduces from the header by reconstructing the EXACT file size
// (header + nv*vstride + nf*face_stride) and aborting on any mismatch. Never silently misreads:
// an unknown property type, a missing x/y/z, a non-triangulated face, an out-of-range index, or
// a non-finite coordinate all abort loudly. Handles the three production layouts (verified by
// file-size identity): moon3d 48 B vertex (6xf64) / 21 B face (uchar+3xi32+2xf32); AGT 60 B
// vertex (3xf64 xyz + 3xf32 normal + 3xf64 latlon-elev) / 13 B face (uchar+3xi32); Haworth
// 24 B vertex (6xf32) / 13 B face.
struct Mesh { std::vector<V3> V; std::vector<std::array<int,3>> F; };
static int ply_typesize(const std::string& ty,const char* tag){
    if(ty=="float"||ty=="float32"||ty=="int"||ty=="int32"||ty=="uint"||ty=="uint32")return 4;
    if(ty=="double"||ty=="float64")return 8;
    if(ty=="short"||ty=="ushort"||ty=="int16"||ty=="uint16")return 2;
    if(ty=="char"||ty=="uchar"||ty=="int8"||ty=="uint8")return 1;
    fprintf(stderr,"FATAL[%s]: unknown PLY property type '%s' -- cannot prove byte layout; refusing to guess.\n",tag,ty.c_str());
    std::exit(3);
}
static unsigned long ply_rd_le(const unsigned char* b,int n){ // little-endian unsigned reconstruct
    unsigned long v=0; for(int k=n-1;k>=0;--k) v=(v<<8)|b[k]; return v; }
static Mesh loadPLY(const char* path,const char* tag){
    std::ifstream f(path,std::ios::binary);
    if(!f){ fprintf(stderr,"FATAL[%s]: cannot open PLY '%s'\n",tag,path); std::exit(3); }
    std::string line,cur_elem; size_t nv=0,nf=0;
    struct Prop{std::string name,type;};
    std::vector<Prop> vprops;                 // vertex scalar props, in order
    std::vector<std::string> face_lead,face_trail; bool have_list=false; std::string fcnt_t,fidx_t;
    bool fmt_seen=false,is_le=false;
    while(std::getline(f,line)){ if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream is(line); std::string t; is>>t;
        if(t=="format"){ std::string fm; is>>fm; fmt_seen=true; is_le=(fm=="binary_little_endian");
            if(fm=="ascii"){ fprintf(stderr,"FATAL[%s]: ascii PLY not supported (need binary_little_endian)\n",tag); std::exit(3);} }
        else if(t=="element"){ std::string w; size_t n; is>>w>>n; cur_elem=w; if(w=="vertex")nv=n; else if(w=="face")nf=n; }
        else if(t=="property"){ std::string a; is>>a;
            if(a=="list"){ std::string ct,it,nm; is>>ct>>it>>nm; if(cur_elem=="face"){fcnt_t=ct;fidx_t=it;have_list=true;} }
            else if(cur_elem=="vertex"){ std::string nm; is>>nm; vprops.push_back({nm,a}); }
            else if(cur_elem=="face"){ std::string nm; is>>nm; (have_list?face_trail:face_lead).push_back(a); } }
        else if(t=="end_header") break; }
    if(!fmt_seen||!is_le){ fprintf(stderr,"FATAL[%s]: missing/unsupported PLY format line (need binary_little_endian)\n",tag); std::exit(3); }
    if(nv==0||nf==0){ fprintf(stderr,"FATAL[%s]: nv=%zu nf=%zu -- both must be > 0\n",tag,nv,nf); std::exit(3); }
    if(!have_list){ fprintf(stderr,"FATAL[%s]: face element has no 'property list ... vertex_indices'\n",tag); std::exit(3); }
    long long header_bytes=(long long)f.tellg();
    int vstride=0,x_off=-1,y_off=-1,z_off=-1; bool x_dbl=false,y_dbl=false,z_dbl=false;
    for(auto&pr:vprops){ bool dbl=(pr.type=="double"||pr.type=="float64");
        if(pr.name=="x"){x_off=vstride;x_dbl=dbl;} else if(pr.name=="y"){y_off=vstride;y_dbl=dbl;}
        else if(pr.name=="z"){z_off=vstride;z_dbl=dbl;}
        vstride+=ply_typesize(pr.type,tag); }
    if(x_off<0||y_off<0||z_off<0){ fprintf(stderr,"FATAL[%s]: vertex x/y/z not all present in header\n",tag); std::exit(3); }
    int fcnt_sz=ply_typesize(fcnt_t,tag),fidx_sz=ply_typesize(fidx_t,tag);
    int lead=0; for(auto&t:face_lead) lead+=ply_typesize(t,tag);
    int trail=0; for(auto&t:face_trail) trail+=ply_typesize(t,tag);
    long long face_stride=(long long)lead+fcnt_sz+3LL*fidx_sz+trail; // assumes triangles; proven below
    long long predicted=header_bytes+(long long)nv*vstride+(long long)nf*face_stride;
    f.seekg(0,std::ios::end); long long actual=(long long)f.tellg(); f.seekg(header_bytes);
    fprintf(stderr,"# PLY[%s] %s\n#   nv=%zu vstride=%d (x@%d%s y@%d%s z@%d%s)  nf=%zu faceStride=%lld (lead=%d cnt=%d idx=%d*3 trail=%d)  header=%lld\n",
            tag,path,nv,vstride,x_off,x_dbl?"d":"f",y_off,y_dbl?"d":"f",z_off,z_dbl?"d":"f",nf,face_stride,lead,fcnt_sz,fidx_sz,trail,header_bytes);
    if(predicted!=actual){
        fprintf(stderr,"FATAL[%s]: byte layout NOT proven -- predicted %lld != actual filesize %lld (delta=%lld).\n"
                       "         Deduced layout does not reproduce the file exactly; refusing to misread.\n",
                tag,predicted,actual,actual-predicted); std::exit(3); }
    Mesh M; M.V.resize(nv); std::vector<char> vb(vstride);
    for(size_t i=0;i<nv;++i){ f.read(vb.data(),vstride);
        double x=x_dbl? *(double*)(vb.data()+x_off):(double)*(float*)(vb.data()+x_off);
        double y=y_dbl? *(double*)(vb.data()+y_off):(double)*(float*)(vb.data()+y_off);
        double z=z_dbl? *(double*)(vb.data()+z_off):(double)*(float*)(vb.data()+z_off);
        M.V[i]={{x,y,z}}; }
    M.F.resize(nf); std::vector<char> lb(lead>0?lead:1),tb(trail>0?trail:1);
    unsigned char cbuf[8],ibuf[8];
    for(size_t i=0;i<nf;++i){ if(lead>0) f.read(lb.data(),lead);
        f.read((char*)cbuf,fcnt_sz); unsigned long cnt=ply_rd_le(cbuf,fcnt_sz);
        if(cnt!=3){ fprintf(stderr,"FATAL[%s]: face %zu has %lu vertices -- tool requires a triangulated mesh\n",tag,i,cnt); std::exit(3); }
        int id[3]; for(int k=0;k<3;++k){ f.read((char*)ibuf,fidx_sz); id[k]=(int)ply_rd_le(ibuf,fidx_sz); }
        if(trail>0) f.read(tb.data(),trail);
        if(id[0]<0||id[1]<0||id[2]<0||(size_t)id[0]>=nv||(size_t)id[1]>=nv||(size_t)id[2]>=nv){
            fprintf(stderr,"FATAL[%s]: face %zu vertex index out of range (%d,%d,%d) nv=%zu\n",tag,i,id[0],id[1],id[2],nv); std::exit(3); }
        M.F[i]={{id[0],id[1],id[2]}}; }
    // xyz sanity spot-check: non-finite is fatal; not-near-lunar is a loud WARN (local-frame
    // meshes are legitimate, e.g. the Haworth validation ring), so it does not abort.
    { double rmin=1e300,rmax=0; size_t step=std::max<size_t>(1,nv/1000),ns=0;
      for(size_t i=0;i<nv;i+=step){ double r=std::sqrt(dot(M.V[i],M.V[i]));
          if(!std::isfinite(r)){ fprintf(stderr,"FATAL[%s]: non-finite vertex coordinate at vertex %zu\n",tag,i); std::exit(3);}
          rmin=std::min(rmin,r); rmax=std::max(rmax,r); ++ns; }
      fprintf(stderr,"# PLY[%s] radius spot-check (n=%zu): min=%.1f max=%.1f m\n",tag,ns,rmin,rmax);
      if(rmax<1.0e6||rmin>2.5e6) fprintf(stderr,"# WARN[%s]: vertex radii not near lunar (~1.737e6 m) -- non-lunar body or local frame?\n",tag); }
    return M;
}

int main(int argc,char**argv){
    if(argc<5){fprintf(stderr,"usage: mc_ref mesh.ply tmax.bin facelist.txt nrays [offset_m] [keps] [min_hits_or_legacy_frac] [outbin]\n"
                              "  env PTM_PP_RING_PLY=<ring.ply>  ring occluder (blocked, never a recorded partner)\n");return 2;}
    const char* ply=argv[1]; const char* tf=argv[2]; const char* fl=argv[3]; int NR=atoi(argv[4]);
    double OFF=(argc>5)?atof(argv[5]):1e-4; double KEPS=(argc>6)?atof(argv[6]):0.2;
    double THR_ARG=(argc>7)?atof(argv[7]):1e-5; const char* OUTB=(argc>8)?argv[8]:"pp_out.bin";
    // 2026-07-21 fix (Sagan-approved, Fig-8/Haworth VF design): the keep-rule was a FIXED
    // FRACTION of rays (hits/tot>=THR), so raising NR with THR held fixed made the absolute
    // detection floor STRICTER, not looser -- measured on haworth_radcore_r32c.ply: 200k vs
    // 2M rays made the far-field-below-threshold fraction WORSE (99.4%->99.9%), and >99% of a
    // floor face's aggregate far-crater-wall flux was silently dropped either way. Fix: keep by
    // an ABSOLUTE minimum hit count, independent of NR. Backward-compatible dual interpretation
    // (no new CLI flag): THR_ARG>=1 is read directly as an integer min-hit-count (NEW calls,
    // e.g. "1"); THR_ARG<1 is treated as the OLD fractional convention and converted ONCE to
    // its equivalent absolute count at THIS run's NR -- so an old-style invocation at the
    // HISTORICAL NR=200000, THR=1e-5 reproduces MIN_HITS=2 exactly (byte-identical moon3d
    // regression), it just no longer silently re-scales if NR changes.
    long MIN_HITS = (THR_ARG>=1.0) ? (long)llround(THR_ARG)
                                    : std::max(1L, (long)llround(THR_ARG*NR));
    fprintf(stderr,"# keep-rule: MIN_HITS=%ld (from THR_ARG=%.6g, NR=%d)\n",MIN_HITS,THR_ARG,NR);

    // ---- load core mesh (fail-loud, header-aware) ----
    Mesh core=loadPLY(ply,"core");
    size_t nv=core.V.size(), nf=core.F.size();
    std::vector<V3>& Vv=core.V; std::vector<std::array<int,3>>& Fc=core.F;

    // ---- optional ring occluder (env PTM_PP_RING_PLY) ----
    const char* ringpath=getenv("PTM_PP_RING_PLY");
    bool hasRing = ringpath && ringpath[0];
    Mesh ring; if(hasRing){ ring=loadPLY(ringpath,"ring");
        fprintf(stderr,"# RING occluder attached: %s (%zu verts, %zu faces) -- hits BLOCK the ray but are NEVER recorded as partners\n",
                ringpath,ring.V.size(),ring.F.size()); }

    std::vector<V3> ctr(nf),nn(nf); std::vector<double> edge(nf);
    for(size_t i=0;i<nf;++i){ const V3&a=Vv[Fc[i][0]],&b=Vv[Fc[i][1]],&c=Vv[Fc[i][2]];
        ctr[i]={{(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3,(a[2]+b[2]+c[2])/3}};
        V3 e1={{b[0]-a[0],b[1]-a[1],b[2]-a[2]}},e2={{c[0]-a[0],c[1]-a[1],c[2]-a[2]}},e3={{c[0]-b[0],c[1]-b[1],c[2]-b[2]}};
        V3 n=nrm(cross(e1,e2)); nn[i]=n; /* RAW winding = builder normal, NO radial flip */
        double l1=std::sqrt(dot(e1,e1)),l2=std::sqrt(dot(e2,e2)),l3=std::sqrt(dot(e3,e3));
        double mx=std::max({l1,l2,l3}),mn=std::min({l1,l2,l3}); edge[i]=l1+l2+l3-mx-mn; }  // median edge
    V3 bbmin=Vv[0],bbmax=Vv[0]; for(auto&v:Vv)for(int k=0;k<3;++k){if(v[k]<bbmin[k])bbmin[k]=v[k];if(v[k]>bbmax[k])bbmax[k]=v[k];}
    V3 sh={{0.5*(bbmin[0]+bbmax[0]),0.5*(bbmin[1]+bbmax[1]),0.5*(bbmin[2]+bbmax[2])}};
    RTCDevice dev=rtcNewDevice(nullptr); RTCScene sc=rtcNewScene(dev); rtcSetSceneFlags(sc,RTC_SCENE_FLAG_ROBUST);
    RTCGeometry g=rtcNewGeometry(dev,RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vb=(float*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,12,nv);
    for(size_t i=0;i<nv;++i){vb[3*i]=float(Vv[i][0]-sh[0]);vb[3*i+1]=float(Vv[i][1]-sh[1]);vb[3*i+2]=float(Vv[i][2]-sh[2]);}
    unsigned* ib=(unsigned*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,12,nf);
    for(size_t i=0;i<nf;++i){ib[3*i]=Fc[i][0];ib[3*i+1]=Fc[i][1];ib[3*i+2]=Fc[i][2];}
    rtcCommitGeometry(g); unsigned core_geom=rtcAttachGeometry(sc,g); rtcReleaseGeometry(g);
    unsigned ring_geom=RTC_INVALID_GEOMETRY_ID;
    if(hasRing){ size_t rnv=ring.V.size(),rnf=ring.F.size();
        RTCGeometry rg=rtcNewGeometry(dev,RTC_GEOMETRY_TYPE_TRIANGLE);
        float* rvb=(float*)rtcSetNewGeometryBuffer(rg,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,12,rnv);
        for(size_t i=0;i<rnv;++i){rvb[3*i]=float(ring.V[i][0]-sh[0]);rvb[3*i+1]=float(ring.V[i][1]-sh[1]);rvb[3*i+2]=float(ring.V[i][2]-sh[2]);}
        unsigned* rib=(unsigned*)rtcSetNewGeometryBuffer(rg,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,12,rnf);
        for(size_t i=0;i<rnf;++i){rib[3*i]=ring.F[i][0];rib[3*i+1]=ring.F[i][1];rib[3*i+2]=ring.F[i][2];}
        rtcCommitGeometry(rg); ring_geom=rtcAttachGeometry(sc,rg); rtcReleaseGeometry(rg);
        fprintf(stderr,"# geomID: core=%u ring=%u\n",core_geom,ring_geom); }
    rtcCommitScene(sc);
    std::vector<float> tmax(nf,0.f);
    { std::ifstream t(tf,std::ios::binary);
      if(!t){ fprintf(stderr,"FATAL: cannot open tmax.bin '%s'\n",tf); std::exit(3); }
      t.seekg(0,std::ios::end); long long tsz=(long long)t.tellg(); t.seekg(0);
      long long need=4+(long long)nf*4;
      if(tsz<need) fprintf(stderr,"# WARN: tmax.bin '%s' is %lld B < expected %lld B (4 + nf*4); missing tail read as 0 (Qir/F_warm undercounted)\n",tf,tsz,need);
      int h=0; t.read((char*)&h,4);
      if(h!=(int)nf && h!=0) fprintf(stderr,"# WARN: tmax.bin header count %d != mesh face count %zu\n",h,nf);
      t.read((char*)tmax.data(),nf*4); }
    const double SIG=5.670374419e-8, EPS=0.95;
    std::vector<long> want; { std::ifstream L(fl); long x; while(L>>x)want.push_back(x); }
    // FC-08 follow-up (2026-08-19, Paul's ruling): BACK-FACE CULL, default ON.
    // A ray that strikes the BACK of a core face has reached the underside of an opaque,
    // one-sided surface. That face cannot radiate toward the source and cannot receive from
    // it, so it is NOT a radiative partner -- but the surface is still opaque, so the ray is
    // BLOCKED, exactly as a ring hit is: not sky, not a partner. Before this, such a hit was
    // recorded as a full partner; after the divisor fix it would have been kept at half
    // weight, which is not better than dropping a physically impossible exchange.
    // The test is n_j . d >= 0, with n_j the RAW winding normal (this tool never radially
    // flips) and d the ray direction. On a closed mesh it can never fire.
    // WINDING WARNING: on a mesh with inconsistent winding a genuine partner reads as
    // back-facing and its flux is lost. The cull count is therefore ALWAYS reported, and a
    // large fraction means "check the winding", not "the cull is working".
    // Set PTM_PP_NO_BACKFACE_CULL=1 to restore the pre-ruling behavior (A/B measurement only).
    bool BACKCULL=true;
    if(const char* e=std::getenv("PTM_PP_NO_BACKFACE_CULL")) if(e[0]=='1'){
        BACKCULL=false;
        fprintf(stderr,"# *** WARNING: PTM_PP_NO_BACKFACE_CULL=1 -- back-face hits are recorded as "
                       "radiative partners. This is the pre-2026-08-19 behavior and is PHYSICALLY "
                       "WRONG on open meshes. A/B measurement only. ***\n"); }
    fprintf(stderr,"# CORRECTED referee: offset=%.6g m  keps=%.3f  nrays=%d%s  backface_cull=%s\n",
            OFF,KEPS,NR,hasRing?"  (+ring occluder)":"",BACKCULL?"ON":"OFF");
    printf("# face SumF F_sky F_warm384 Qir\n");
    RTCIntersectArguments ia; rtcInitIntersectArguments(&ia);
    long ring_total=0, ray_total=0, back_total=0, backseen_total=0;
    for(long pf:want){ if(pf<0||pf>=(long)nf)continue; const V3&c=ctr[pf]; V3 ax=nn[pf]; double heps=KEPS*edge[pf];
        V3 t=(std::fabs(ax[2])<0.9)?V3{{0,0,1}}:V3{{1,0,0}}; V3 bx=nrm(cross(ax,t)),by=cross(ax,bx);
        long terr=0,warm=0,tot=0,ringhit=0,backhit=0,backseen=0; double Qir=0; std::unordered_map<int,long> hits;
        for(int s=0;s<NR;++s){
            double u1=halton(s+1,2),u2=halton(s+1,3), rr=std::sqrt(u1),ph=2*M_PI*u2;  // deterministic QMC
            V3 d=nrm({{ rr*std::cos(ph)*bx[0]+rr*std::sin(ph)*by[0]+std::sqrt(1-u1)*ax[0],
                        rr*std::cos(ph)*bx[1]+rr*std::sin(ph)*by[1]+std::sqrt(1-u1)*ax[1],
                        rr*std::cos(ph)*bx[2]+rr*std::sin(ph)*by[2]+std::sqrt(1-u1)*ax[2] }});
            tot++;
            // rise-march: skip self-relief hits (rise<heps), take first real-terrain hit.
            // hitface: >=0 core partner primID ; -2 ring occlusion (blocked, no partner) ; -1 sky.
            double tstart=OFF; int steps=0; long hitface=-1;
            while(steps++<32){
                RTCRayHit rh; std::memset(&rh,0,sizeof(rh));
                rh.ray.org_x=float(c[0]-sh[0]+tstart*d[0]);rh.ray.org_y=float(c[1]-sh[1]+tstart*d[1]);rh.ray.org_z=float(c[2]-sh[2]+tstart*d[2]);
                rh.ray.dir_x=float(d[0]);rh.ray.dir_y=float(d[1]);rh.ray.dir_z=float(d[2]);
                rh.ray.tnear=0.f;rh.ray.tfar=1e12f;rh.ray.mask=0xFFFFFFFF;rh.hit.geomID=RTC_INVALID_GEOMETRY_ID;
                rtcIntersect1(sc,&rh,&ia);
                if(rh.hit.geomID==RTC_INVALID_GEOMETRY_ID){hitface=-1;break;}  // sky
                bool isCore=(rh.hit.geomID==core_geom);
                unsigned p=rh.hit.primID; double thit=tstart+rh.ray.tfar;
                V3 P={{c[0]+thit*d[0], c[1]+thit*d[1], c[2]+thit*d[2]}};
                double rise=dot(V3{{P[0]-c[0],P[1]-c[1],P[2]-c[2]}}, ax);  // out-of-plane rise from face pf
                // A ring hit is a different geometry, so it can never be the launch face's own
                // self-relief (the p==pf skip is gated on isCore). Above the relief floor it
                // blocks the ray as a genuine occluder without becoming a partner.
                if(rise>=heps && !(isCore && (long)p==pf)){
                    // Back-face cull: we struck the underside of an opaque one-sided facet.
                    // It blocks the ray but can exchange no radiation -> -3, like a ring hit.
                    // The geometric condition is counted whether or not the cull ACTS, so the
                    // OFF arm reports the same population it is choosing to keep.
                    if(isCore && dot(nn[p],d)>=0.0){
                        backseen++;
                        if(BACKCULL){ hitface=-3; break; }
                    }
                    hitface = isCore ? (long)p : -2; break;
                }
                tstart=thit+1e-5;  // self-relief or self: advance past and continue
            }
            if(hitface>=0){ terr++; double Tj=tmax[hitface]; if(Tj>384.f)warm++; Qir+=SIG*Tj*Tj*Tj*Tj; hits[hitface]++; }
            else if(hitface==-2){ ringhit++; }  // occluded by ring: not sky, not a partner
            else if(hitface==-3){ backhit++; }  // opaque back side: not sky, not a partner
        }
        // F_sky excludes ring occlusion and back-face blocking (a blocked ray is not sky).
        // Byte-identical to 1-SumF when neither fires (ringhit==0 && backhit==0).
        double SumF=double(terr)/tot, Fsky=double(tot-terr-ringhit-backhit)/tot, Fw=double(warm)/tot, Qir_abs=EPS*EPS*(Qir/tot);
        printf("%7ld %.4f %.4f %.4f %.1f\n",pf,SumF,Fsky,Fw,Qir_abs);
        ring_total+=ringhit; ray_total+=tot; back_total+=backhit; backseen_total+=backseen;
        { static FILE* fb=fopen(OUTB,"wb"); int nnz=0; for(auto&kv:hits){ if(kv.second>=MIN_HITS) nnz++; }
          int src=(int)pf; fwrite(&src,4,1,fb); fwrite(&nnz,4,1,fb);
          for(auto&kv:hits){ double F=(double)kv.second/tot; if(kv.second>=MIN_HITS){ int j=kv.first; float ff=(float)F; fwrite(&j,4,1,fb); fwrite(&ff,4,1,fb);} }
          fflush(fb); }
    }
    if(hasRing) fprintf(stderr,"# RING summary: %ld ring-occlusion hits over %ld rays (%.4f) -- none recorded as partners\n",
                        ring_total,ray_total,ray_total?double(ring_total)/ray_total:0.0);
    // ALWAYS reported, including the zero: a closed mesh must show 0, and a large fraction on
    // an open mesh means "check the mesh winding" before it means "the cull is working".
    fprintf(stderr,"# BACKFACE summary: cull=%s  %ld back-side hits DETECTED over %ld rays (%.6f); "
                   "%ld BLOCKED (not sky, not a partner), %ld recorded as partners anyway%s\n",
            BACKCULL?"ON":"OFF",backseen_total,ray_total,
            ray_total?double(backseen_total)/ray_total:0.0,back_total,backseen_total-back_total,
            BACKCULL?"":"  <-- PHYSICALLY WRONG, cull disabled by env");
    rtcReleaseScene(sc);rtcReleaseDevice(dev); return 0;
}
