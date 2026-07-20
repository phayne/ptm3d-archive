// PTM-12 CORRECTED occlusion-aware MC VF referee (Fermi's corrections folded in).
//   - 0.1 mm ray offset (default; param), forced-outward normals
//   - grazing guard: rise-march skips self-relief hits (out-of-plane rise < k*edge), re-casts
//   - deterministic Halton QMC (bit-reproducible, no std::rand)
// Usage: mc_ref <mesh.ply> <tmax.bin> <facelist.txt> <nrays> [offset_m=1e-4] [keps=0.2]
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
#include <cstdio>
using V3=std::array<double,3>;
static double dot(const V3&a,const V3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V3 cross(const V3&a,const V3&b){return{{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
static V3 nrm(const V3&a){double L=std::sqrt(dot(a,a));return L>0?V3{{a[0]/L,a[1]/L,a[2]/L}}:a;}
static double halton(int i,int b){double f=1,r=0; while(i>0){f/=b; r+=f*(i%b); i/=b;} return r;}
int main(int argc,char**argv){
    if(argc<5){fprintf(stderr,"usage: mc_ref mesh.ply tmax.bin facelist.txt nrays [offset_m] [keps]\n");return 2;}
    const char* ply=argv[1]; const char* tf=argv[2]; const char* fl=argv[3]; int NR=atoi(argv[4]);
    double OFF=(argc>5)?atof(argv[5]):1e-4; double KEPS=(argc>6)?atof(argv[6]):0.2; double THR=(argc>7)?atof(argv[7]):1e-5; const char* OUTB=(argc>8)?argv[8]:"pp_out.bin";
    std::ifstream f(ply,std::ios::binary); std::string line; size_t nv=0,nf=0;
    while(std::getline(f,line)){ if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream is(line); std::string t; is>>t;
        if(t=="element"){std::string w;size_t n;is>>w>>n; if(w=="vertex")nv=n; if(w=="face")nf=n;}
        else if(t=="end_header")break; }
    std::vector<V3> Vv(nv); std::vector<double> bf(6);
    for(size_t i=0;i<nv;++i){ f.read((char*)bf.data(),48); Vv[i]={{bf[0],bf[1],bf[2]}}; }
    std::vector<std::array<int,3>> Fc(nf);
    for(size_t i=0;i<nf;++i){ unsigned char c; f.read((char*)&c,1); int id[3]; f.read((char*)id,12); float ll[2]; f.read((char*)ll,8); Fc[i]={{id[0],id[1],id[2]}}; }
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
    rtcCommitGeometry(g);rtcAttachGeometry(sc,g);rtcReleaseGeometry(g);rtcCommitScene(sc);
    std::vector<float> tmax(nf,0.f); { std::ifstream t(tf,std::ios::binary); int h; t.read((char*)&h,4); t.read((char*)tmax.data(),nf*4); }
    const double SIG=5.670374419e-8, EPS=0.95;
    std::vector<long> want; { std::ifstream L(fl); long x; while(L>>x)want.push_back(x); }
    fprintf(stderr,"# CORRECTED referee: offset=%.6g m  keps=%.3f  nrays=%d\n",OFF,KEPS,NR);
    printf("# face SumF F_sky F_warm384 Qir\n");
    RTCIntersectArguments ia; rtcInitIntersectArguments(&ia);
    for(long pf:want){ if(pf<0||pf>=(long)nf)continue; const V3&c=ctr[pf]; V3 ax=nn[pf]; double heps=KEPS*edge[pf];
        V3 t=(std::fabs(ax[2])<0.9)?V3{{0,0,1}}:V3{{1,0,0}}; V3 bx=nrm(cross(ax,t)),by=cross(ax,bx);
        long terr=0,warm=0,tot=0; double Qir=0; std::unordered_map<int,long> hits;
        for(int s=0;s<NR;++s){
            double u1=halton(s+1,2),u2=halton(s+1,3), rr=std::sqrt(u1),ph=2*M_PI*u2;  // deterministic QMC
            V3 d=nrm({{ rr*std::cos(ph)*bx[0]+rr*std::sin(ph)*by[0]+std::sqrt(1-u1)*ax[0],
                        rr*std::cos(ph)*bx[1]+rr*std::sin(ph)*by[1]+std::sqrt(1-u1)*ax[1],
                        rr*std::cos(ph)*bx[2]+rr*std::sin(ph)*by[2]+std::sqrt(1-u1)*ax[2] }});
            tot++;
            // rise-march: skip self-relief hits (rise<heps), take first real-terrain hit
            double tstart=OFF; int steps=0; long hitface=-1;
            while(steps++<32){
                RTCRayHit rh; std::memset(&rh,0,sizeof(rh));
                rh.ray.org_x=float(c[0]-sh[0]+tstart*d[0]);rh.ray.org_y=float(c[1]-sh[1]+tstart*d[1]);rh.ray.org_z=float(c[2]-sh[2]+tstart*d[2]);
                rh.ray.dir_x=float(d[0]);rh.ray.dir_y=float(d[1]);rh.ray.dir_z=float(d[2]);
                rh.ray.tnear=0.f;rh.ray.tfar=1e12f;rh.ray.mask=0xFFFFFFFF;rh.hit.geomID=RTC_INVALID_GEOMETRY_ID;
                rtcIntersect1(sc,&rh,&ia);
                if(rh.hit.geomID==RTC_INVALID_GEOMETRY_ID){hitface=-1;break;}  // sky
                unsigned p=rh.hit.primID; double thit=tstart+rh.ray.tfar;
                V3 P={{c[0]+thit*d[0], c[1]+thit*d[1], c[2]+thit*d[2]}};
                double rise=dot(V3{{P[0]-c[0],P[1]-c[1],P[2]-c[2]}}, ax);  // out-of-plane rise from face pf
                if((long)p!=pf && rise>=heps){ hitface=(long)p; break; }    // real terrain/occluder
                tstart=thit+1e-5;  // self-relief or self: advance past and continue
            }
            if(hitface>=0){ terr++; double Tj=tmax[hitface]; if(Tj>384.f)warm++; Qir+=SIG*Tj*Tj*Tj*Tj; hits[hitface]++; }
        }
        double SumF=double(terr)/tot, Fsky=1.0-SumF, Fw=double(warm)/tot, Qir_abs=EPS*EPS*(Qir/tot);
        printf("%7ld %.4f %.4f %.4f %.1f\n",pf,SumF,Fsky,Fw,Qir_abs);
        { static FILE* fb=fopen(OUTB,"wb"); int nnz=0; for(auto&kv:hits){ if((double)kv.second/tot>=THR) nnz++; }
          int src=(int)pf; fwrite(&src,4,1,fb); fwrite(&nnz,4,1,fb);
          for(auto&kv:hits){ double F=(double)kv.second/tot; if(F>=THR){ int j=kv.first; float ff=(float)F; fwrite(&j,4,1,fb); fwrite(&ff,4,1,fb);} }
          fflush(fb); }
    }
    rtcReleaseScene(sc);rtcReleaseDevice(dev); return 0;
}
