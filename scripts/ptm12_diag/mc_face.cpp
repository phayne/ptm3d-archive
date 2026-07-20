// PTM-12 per-face occlusion-aware MC view factor (ground truth), outward normals forced.
// Usage: mc_face <mesh.ply> <coupled_tmax.bin> <facelist.txt> <n_rays>
// For each listed face: true SumF (terrain hit-fraction, exact occlusion), F_sky,
// F_warm(>384K), and MC incident IR = eps*(1/N)Sum eps*sigma*Tmax_hit^4  (cosine-weighted).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <embree4/rtcore.h>
using V3=std::array<double,3>;
static double dot(const V3&a,const V3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V3 cross(const V3&a,const V3&b){return{{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
static V3 nrm(const V3&a){double L=std::sqrt(dot(a,a));return L>0?V3{{a[0]/L,a[1]/L,a[2]/L}}:a;}
int main(int argc,char**argv){
    if(argc<5){fprintf(stderr,"usage: mc_face mesh.ply tmax.bin facelist.txt nrays\n");return 2;}
    const char* ply=argv[1]; const char* tf=argv[2]; const char* fl=argv[3]; int NR=atoi(argv[4]);
    // read PLY (double x,y,z,nx,ny,nz ; face = uchar count + 3 int + 2 float)
    std::ifstream f(ply,std::ios::binary); std::string line; size_t nv=0,nf=0;
    while(std::getline(f,line)){ if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream is(line); std::string t; is>>t;
        if(t=="element"){std::string w;size_t n;is>>w>>n; if(w=="vertex")nv=n; if(w=="face")nf=n;}
        else if(t=="end_header")break; }
    std::vector<V3> Vv(nv); std::vector<double> buf(6);
    for(size_t i=0;i<nv;++i){ f.read((char*)buf.data(),6*8); Vv[i]={{buf[0],buf[1],buf[2]}}; }
    std::vector<std::array<int,3>> Fc(nf);
    for(size_t i=0;i<nf;++i){ unsigned char c; f.read((char*)&c,1); int idx[3]; f.read((char*)idx,12); float ll[2]; f.read((char*)ll,8); Fc[i]={{idx[0],idx[1],idx[2]}}; }
    // centroid, area-normal, up, outward normal
    std::vector<V3> ctr(nf),nn(nf);
    V3 mean={{0,0,0}};
    for(size_t i=0;i<nf;++i){ const V3&a=Vv[Fc[i][0]],&b=Vv[Fc[i][1]],&c=Vv[Fc[i][2]];
        ctr[i]={{(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3,(a[2]+b[2]+c[2])/3}};
        for(int k=0;k<3;++k)mean[k]+=ctr[i][k]; }
    for(int k=0;k<3;++k)mean[k]/=nf;
    for(size_t i=0;i<nf;++i){ const V3&a=Vv[Fc[i][0]],&b=Vv[Fc[i][1]],&c=Vv[Fc[i][2]];
        V3 e1={{b[0]-a[0],b[1]-a[1],b[2]-a[2]}},e2={{c[0]-a[0],c[1]-a[1],c[2]-a[2]}};
        V3 n=nrm(cross(e1,e2)); V3 up=nrm(ctr[i]); // radial up (selenocentric)
        if(dot(n,up)<0){n[0]=-n[0];n[1]=-n[1];n[2]=-n[2];} nn[i]=n; }
    // recenter for float32 Embree
    V3 bbmin=Vv[0],bbmax=Vv[0]; for(auto&v:Vv)for(int k=0;k<3;++k){if(v[k]<bbmin[k])bbmin[k]=v[k];if(v[k]>bbmax[k])bbmax[k]=v[k];}
    V3 sh={{0.5*(bbmin[0]+bbmax[0]),0.5*(bbmin[1]+bbmax[1]),0.5*(bbmin[2]+bbmax[2])}};
    RTCDevice dev=rtcNewDevice(nullptr); RTCScene sc=rtcNewScene(dev);
    rtcSetSceneFlags(sc,RTC_SCENE_FLAG_ROBUST);
    RTCGeometry g=rtcNewGeometry(dev,RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vb=(float*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,3*sizeof(float),nv);
    for(size_t i=0;i<nv;++i){vb[3*i]=float(Vv[i][0]-sh[0]);vb[3*i+1]=float(Vv[i][1]-sh[1]);vb[3*i+2]=float(Vv[i][2]-sh[2]);}
    unsigned* ib=(unsigned*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,3*sizeof(unsigned),nf);
    for(size_t i=0;i<nf;++i){ib[3*i]=Fc[i][0];ib[3*i+1]=Fc[i][1];ib[3*i+2]=Fc[i][2];}
    rtcCommitGeometry(g);rtcAttachGeometry(sc,g);rtcReleaseGeometry(g);rtcCommitScene(sc);
    // tmax
    std::vector<float> tmax(nf,0.f); { std::ifstream t(tf,std::ios::binary); int hdr; t.read((char*)&hdr,4); t.read((char*)tmax.data(),nf*sizeof(float)); }
    const double SIG=5.670374419e-8, EPS=0.95;
    // face list
    std::vector<long> want; { std::ifstream L(fl); long x; while(L>>x)want.push_back(x); }
    printf("# face  SumF_MC  F_sky_MC  F_warm384  Qir_MC  Qir_MC_selfT(usesCoupledT)  domEmit  domT  domF\n");
    RTCIntersectArguments ia; rtcInitIntersectArguments(&ia);
    for(long pf:want){ if(pf<0||pf>=(long)nf)continue; const V3&c=ctr[pf]; V3 ax=nn[pf];
        V3 t=(std::fabs(ax[2])<0.9)?V3{{0,0,1}}:V3{{1,0,0}}; V3 bx=nrm(cross(ax,t)),by=cross(ax,bx);
        std::mt19937 rng(12345+pf); std::uniform_real_distribution<double> U(0,1);
        long terr=0,warm=0; double Qir=0; long tot=0;
        std::vector<double> emitF(0);
        // track dominant emitter by accumulated F (hits) weighted
        long domE=-1; double domW=-1; std::vector<std::pair<long,long>> hitcount; // simple: track max sigmaT4 contributor
        double bestContribT=0; long bestContribFace=-1; double bestContribF=0;
        // accumulate per-hit; to find dominant, keep a small running max of per-face contribution
        // (approx: single best face by sigma T^4 * (1/N))
        for(int s=0;s<NR;++s){ double u1=U(rng),u2=U(rng),rr=std::sqrt(u1),ph=2*M_PI*u2;
            V3 d=nrm({{ rr*std::cos(ph)*bx[0]+rr*std::sin(ph)*by[0]+std::sqrt(1-u1)*ax[0],
                        rr*std::cos(ph)*bx[1]+rr*std::sin(ph)*by[1]+std::sqrt(1-u1)*ax[1],
                        rr*std::cos(ph)*bx[2]+rr*std::sin(ph)*by[2]+std::sqrt(1-u1)*ax[2] }});
            tot++;
            RTCRayHit rh; std::memset(&rh,0,sizeof(rh));
            rh.ray.org_x=float(c[0]-sh[0]+0.002*d[0]);rh.ray.org_y=float(c[1]-sh[1]+0.002*d[1]);rh.ray.org_z=float(c[2]-sh[2]+0.002*d[2]);
            rh.ray.dir_x=float(d[0]);rh.ray.dir_y=float(d[1]);rh.ray.dir_z=float(d[2]);
            rh.ray.tnear=0.f;rh.ray.tfar=1e12f;rh.ray.mask=0xFFFFFFFF;rh.hit.geomID=RTC_INVALID_GEOMETRY_ID;
            rtcIntersect1(sc,&rh,&ia);
            if(rh.hit.geomID!=RTC_INVALID_GEOMETRY_ID){ unsigned p=rh.hit.primID; if((long)p==pf)continue; terr++;
                double Tj=(p<nf)?tmax[p]:0.0; if(Tj>384.f)warm++;
                Qir += SIG*Tj*Tj*Tj*Tj;
                if(SIG*Tj*Tj*Tj*Tj>bestContribT){bestContribT=SIG*Tj*Tj*Tj*Tj;bestContribFace=p;bestContribF=Tj;}
            }
        }
        double SumF=double(terr)/tot, Fsky=1.0-SumF, Fwarm=double(warm)/tot;
        double Qir_abs=EPS*EPS*(Qir/tot); // eps absorb * eps emit * mean sigmaT4
        printf("%7ld  %.4f    %.4f    %.4f   %.1f   (coupledT)  %7ld  %.1f  -\n",
               pf,SumF,Fsky,Fwarm,Qir_abs,bestContribFace,bestContribF);
    }
    rtcReleaseScene(sc);rtcReleaseDevice(dev);
    return 0;
}
