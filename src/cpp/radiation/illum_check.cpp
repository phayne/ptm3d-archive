/**
 * @file illum_check.cpp
 * @brief Independent grazing-Sun shadow-ray check: for a given Sun direction, trace
 *        shadow rays from every Sun-facing face with a CONFIGURABLE origin lift and
 *        report the lit fraction. Sweeping the lift reveals whether the production
 *        illumination's shadowing of the inner crater ring at grazing Sun is REAL
 *        (lit fraction stable vs lift) or a fine-mesh self-shadow ARTIFACT (lit
 *        fraction jumps up as the ray origin clears sub-grid roughness).
 *
 * Usage: illum_check <mesh.ply> <sx> <sy> <sz>  [lift_m=2]  [graze_cos=0.07] [occluder.ply]
 *        sx,sy,sz = Sun position (any units; only the direction is used).
 *        occluder.ply = optional extra occlusion geometry (e.g. the 200km ring) added
 *        to the Embree scene so the shadow test matches the production illumination.
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <embree4/rtcore.h>

using V3 = std::array<double,3>;
static double dot(const V3&a,const V3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V3 nrm(const V3&a){double L=std::sqrt(dot(a,a));return L>0?V3{{a[0]/L,a[1]/L,a[2]/L}}:a;}
static int psz(const std::string&t){ if(t=="double"||t=="float64")return 8; if(t=="char"||t=="uchar"||t=="int8"||t=="uint8")return 1; if(t=="short"||t=="ushort")return 2; return 4; }
static bool readPLY(const std::string&p,std::vector<V3>&V,std::vector<std::array<int,3>>&F,std::vector<V3>&N){
    std::ifstream f(p,std::ios::binary); if(!f)return false; std::string ln; size_t nv=0,nf=0; bool iv=false;
    std::vector<std::pair<std::string,std::string>> vp;
    while(std::getline(f,ln)){ if(!ln.empty()&&ln.back()=='\r')ln.pop_back(); std::istringstream is(ln); std::string t; is>>t;
        if(t=="element"){std::string w;size_t n;is>>w>>n; if(w=="vertex"){nv=n;iv=true;}else iv=false; if(w=="face")nf=n;}
        else if(t=="property"){if(iv){std::string a,b;is>>a>>b;vp.push_back({a,b});}} else if(t=="end_header")break; }
    if(!nv||!nf)return false; int o[3]={-1,-1,-1},no[3]={-1,-1,-1}; std::string ty[3],nty[3]; int st=0;
    for(auto&q:vp){ if(q.second=="x"){o[0]=st;ty[0]=q.first;}else if(q.second=="y"){o[1]=st;ty[1]=q.first;}
        else if(q.second=="z"){o[2]=st;ty[2]=q.first;}else if(q.second=="nx"){no[0]=st;nty[0]=q.first;}
        else if(q.second=="ny"){no[1]=st;nty[1]=q.first;}else if(q.second=="nz"){no[2]=st;nty[2]=q.first;} st+=psz(q.first); }
    bool hv=no[0]>=0; V.resize(nv);N.assign(nv,{{0,0,0}}); std::vector<char> r(st);
    auto rd=[&](int off,const std::string&t){ if(t=="double"||t=="float64"){double d;std::memcpy(&d,r.data()+off,8);return d;} float fl;std::memcpy(&fl,r.data()+off,4);return (double)fl; };
    for(size_t i=0;i<nv;++i){f.read(r.data(),st); for(int k=0;k<3;++k)V[i][k]=rd(o[k],ty[k]); if(hv)for(int k=0;k<3;++k)N[i][k]=rd(no[k],nty[k]); }
    F.resize(nf); for(size_t i=0;i<nf;++i){unsigned char c;f.read((char*)&c,1);std::vector<int> id(c);f.read((char*)id.data(),c*4);F[i]={id[0],id[1],id[2]};}
    return true;
}

int main(int argc,char**argv){
    if(argc<5){std::cerr<<"usage: illum_check <mesh.ply> <sx> <sy> <sz> [lift_m=2] [graze_cos=0.07]\n";return 2;}
    V3 sun=nrm({{std::atof(argv[2]),std::atof(argv[3]),std::atof(argv[4])}});
    double lift=argc>5?std::atof(argv[5]):2.0;
    double gc =argc>6?std::atof(argv[6]):0.07;
    std::vector<V3> V,N; std::vector<std::array<int,3>> F;
    if(!readPLY(argv[1],V,F,N)){std::cerr<<"load fail\n";return 1;}
    size_t nf=F.size(); std::vector<V3> ctr(nf),fn(nf);
    for(size_t i=0;i<nf;++i){const V3&a=V[F[i][0]],&b=V[F[i][1]],&c=V[F[i][2]];
        ctr[i]={{(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3,(a[2]+b[2]+c[2])/3}};
        fn[i]=nrm({{N[F[i][0]][0]+N[F[i][1]][0]+N[F[i][2]][0],N[F[i][0]][1]+N[F[i][1]][1]+N[F[i][2]][1],N[F[i][0]][2]+N[F[i][1]][2]+N[F[i][2]][2]}}); }
    // Optional occluder mesh (e.g. the 200km ring) — adds occlusion geometry so
    // the shadow test matches the production illumination scene (core + ring).
    std::vector<V3> Vo,No; std::vector<std::array<int,3>> Fo;
    bool have_occ = (argc>7);
    if(have_occ){ if(!readPLY(argv[7],Vo,Fo,No)){std::cerr<<"occluder load fail\n";return 1;} }

    // recenter to COMBINED bbox center (mirror illumination_model.cpp float32 fix
    // over the full core+occluder scene)
    V3 bbmin=V[0],bbmax=V[0];
    auto upd=[&](const std::vector<V3>&P){ for(auto&v:P)for(int k=0;k<3;++k){bbmin[k]=std::min(bbmin[k],v[k]);bbmax[k]=std::max(bbmax[k],v[k]);} };
    upd(V); if(have_occ) upd(Vo);
    V3 sh={{0.5*(bbmin[0]+bbmax[0]),0.5*(bbmin[1]+bbmax[1]),0.5*(bbmin[2]+bbmax[2])}};
    RTCDevice dev=rtcNewDevice(nullptr); RTCScene sc=rtcNewScene(dev);
    auto addgeo=[&](const std::vector<V3>&P,const std::vector<std::array<int,3>>&FF){
        RTCGeometry g=rtcNewGeometry(dev,RTC_GEOMETRY_TYPE_TRIANGLE);
        float* vb=(float*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,12,P.size());
        for(size_t i=0;i<P.size();++i){vb[3*i]=float(P[i][0]-sh[0]);vb[3*i+1]=float(P[i][1]-sh[1]);vb[3*i+2]=float(P[i][2]-sh[2]);}
        unsigned* ib=(unsigned*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,12,FF.size());
        for(size_t i=0;i<FF.size();++i){ib[3*i]=FF[i][0];ib[3*i+1]=FF[i][1];ib[3*i+2]=FF[i][2];}
        rtcCommitGeometry(g);rtcAttachGeometry(sc,g);rtcReleaseGeometry(g);
    };
    addgeo(V,F); if(have_occ) addgeo(Vo,Fo);
    rtcCommitScene(sc);
    if(have_occ) std::cerr<<"  (occluder "<<argv[7]<<": "<<Fo.size()<<" faces added)\n";

    long facing=0, lit=0;
    std::vector<char> litflag(nf,0);
    RTCOccludedArguments oa; rtcInitOccludedArguments(&oa);
    #pragma omp parallel for reduction(+:facing,lit) schedule(dynamic,1024)
    for(size_t i=0;i<nf;++i){
        double c=dot(fn[i],sun); if(c<=gc) continue; facing++;
        V3 o={{ctr[i][0]-sh[0]+lift*fn[i][0], ctr[i][1]-sh[1]+lift*fn[i][1], ctr[i][2]-sh[2]+lift*fn[i][2]}};
        RTCRay ray; ray.org_x=float(o[0]);ray.org_y=float(o[1]);ray.org_z=float(o[2]);
        ray.dir_x=float(sun[0]);ray.dir_y=float(sun[1]);ray.dir_z=float(sun[2]);
        ray.tnear=0.f; ray.tfar=1e12f; ray.mask=0xFFFFFFFF;ray.flags=0;ray.time=0.f;
        RTCOccludedArguments la; rtcInitOccludedArguments(&la);
        rtcOccluded1(sc,&ray,&la); bool vis=(ray.tfar>=0.0f);
        if(vis){lit++; litflag[i]=1;}
    }
    std::cout<<"lift="<<lift<<"m graze_cos="<<gc<<" : Sun-facing="<<facing<<"  LIT="<<lit
             <<"  ("<<(100.0*lit/std::max(1L,facing))<<"% of facing)\n";
    if(const char* op=std::getenv("ILLUM_OUT")){ std::ofstream o(op,std::ios::binary); o.write(litflag.data(),nf); }
    rtcReleaseScene(sc);rtcReleaseDevice(dev); return 0;
}
