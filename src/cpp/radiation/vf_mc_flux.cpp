/**
 * @file vf_mc_flux.cpp
 * @brief Geometric GROUND-TRUTH indirect flux on the crater floor by cosine-weighted
 *        hemisphere Monte-Carlo (Embree), weighting each ray's hit by eps*sigma*T_hit^4.
 *
 *        This is the DEFINITION of the absorbed indirect IR a floor face should receive:
 *          H_i = eps_i * (1/N) * sum_rays [ eps_j * sigma * T_hit^4 ]   (cosine-sampled => F_ij)
 *        independent of the H-matrix view-factor estimator. Comparing this to the model's
 *        delivered indirect (the QIDT dump, ~0.49 W/m2 median at 100k) decides whether the
 *        floor-center coupling deficit is a VF under-count (H_mc >> delivered) or physical
 *        (H_mc ~ delivered).
 *
 * Hemisphere is taken around the MACROSCOPIC UP (the flat-floor normal), the physically
 * correct receiving hemisphere; the per-facet normal is reported by vf_hemisphere_mc.
 *
 * Usage: vf_mc_flux <mesh.ply> <tmax.bin> [n_rays=8000] [eps=0.95] [out_perface.bin]
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <algorithm>
#include <embree4/rtcore.h>

using V3 = std::array<double, 3>;
static double dot(const V3&a,const V3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V3 cross(const V3&a,const V3&b){return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
static V3 norm3(const V3&a){double L=std::sqrt(dot(a,a));return L>0?V3{{a[0]/L,a[1]/L,a[2]/L}}:a;}

static int plySize(const std::string&t){
    if(t=="float"||t=="float32"||t=="int"||t=="int32"||t=="uint"||t=="uint32")return 4;
    if(t=="double"||t=="float64")return 8; if(t=="char"||t=="uchar"||t=="int8"||t=="uint8")return 1;
    if(t=="short"||t=="ushort"||t=="int16"||t=="uint16")return 2; return 4; }
static bool readPLY(const std::string&path,std::vector<V3>&verts,std::vector<std::array<int,3>>&faces,std::vector<V3>&vn){
    std::ifstream f(path,std::ios::binary); if(!f)return false;
    std::string line; size_t nv=0,nf=0; bool inV=false; std::vector<std::pair<std::string,std::string>> vp;
    while(std::getline(f,line)){ if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream is(line); std::string t; is>>t;
        if(t=="element"){std::string w;size_t n;is>>w>>n; if(w=="vertex"){nv=n;inV=true;}else{inV=false;} if(w=="face")nf=n;}
        else if(t=="property"){if(inV){std::string ty,nm;is>>ty>>nm;vp.push_back({ty,nm});}}
        else if(t=="end_header")break; }
    if(nv==0||nf==0)return false;
    int off[3]={-1,-1,-1},nof[3]={-1,-1,-1}; std::string ty[3],nty[3]; int stride=0;
    for(auto&p:vp){ if(p.second=="x"){off[0]=stride;ty[0]=p.first;} else if(p.second=="y"){off[1]=stride;ty[1]=p.first;}
        else if(p.second=="z"){off[2]=stride;ty[2]=p.first;} else if(p.second=="nx"){nof[0]=stride;nty[0]=p.first;}
        else if(p.second=="ny"){nof[1]=stride;nty[1]=p.first;} else if(p.second=="nz"){nof[2]=stride;nty[2]=p.first;}
        stride+=plySize(p.first); }
    bool hv=(nof[0]>=0&&nof[1]>=0&&nof[2]>=0);
    verts.resize(nv); vn.assign(nv,{{0,0,0}}); std::vector<char> row(stride);
    auto rd=[&](int o,const std::string&t)->double{ if(t=="double"||t=="float64"){double d;std::memcpy(&d,row.data()+o,8);return d;} float fl;std::memcpy(&fl,row.data()+o,4);return fl; };
    for(size_t i=0;i<nv;++i){ f.read(row.data(),stride); for(int k=0;k<3;++k)verts[i][k]=rd(off[k],ty[k]); if(hv)for(int k=0;k<3;++k)vn[i][k]=rd(nof[k],nty[k]); }
    faces.resize(nf);
    for(size_t i=0;i<nf;++i){ unsigned char c; f.read((char*)&c,1); std::vector<int> idx(c); f.read((char*)idx.data(),c*sizeof(int)); faces[i]={idx[0],idx[1],idx[2]}; }
    return true;
}

int main(int argc,char**argv){
    if(argc<3){ std::cerr<<"usage: vf_mc_flux <mesh.ply> <tmax.bin> [n_rays=8000] [eps=0.95] [out.bin]\n"; return 2; }
    int NR = (argc>3)?std::atoi(argv[3]):8000;
    double EPS = (argc>4)?std::atof(argv[4]):0.95;
    std::string outp = (argc>5)?argv[5]:"";
    const double SIG = 5.670374e-8;
    std::vector<V3> V,VN; std::vector<std::array<int,3>> Fc;
    if(!readPLY(argv[1],V,Fc,VN)){ std::cerr<<"load failed\n"; return 1; }
    size_t nf=Fc.size();
    std::vector<V3> ctr(nf),nrm(nf);
    for(size_t i=0;i<nf;++i){ const V3&a=V[Fc[i][0]],&b=V[Fc[i][1]],&c=V[Fc[i][2]];
        ctr[i]={{(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3,(a[2]+b[2]+c[2])/3}};
        V3 vnn={{VN[Fc[i][0]][0]+VN[Fc[i][1]][0]+VN[Fc[i][2]][0],VN[Fc[i][0]][1]+VN[Fc[i][1]][1]+VN[Fc[i][2]][1],VN[Fc[i][0]][2]+VN[Fc[i][1]][2]+VN[Fc[i][2]][2]}};
        nrm[i]=norm3(vnn); }
    V3 bbmin=V[0],bbmax=V[0]; for(auto&v:V)for(int k=0;k<3;++k){bbmin[k]=std::min(bbmin[k],v[k]);bbmax[k]=std::max(bbmax[k],v[k]);}
    V3 sh={{0.5*(bbmin[0]+bbmax[0]),0.5*(bbmin[1]+bbmax[1]),0.5*(bbmin[2]+bbmax[2])}};
    RTCDevice dev=rtcNewDevice(nullptr); RTCScene sc=rtcNewScene(dev);
    RTCGeometry g=rtcNewGeometry(dev,RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vb=(float*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,3*sizeof(float),V.size());
    for(size_t i=0;i<V.size();++i){vb[3*i]=float(V[i][0]-sh[0]);vb[3*i+1]=float(V[i][1]-sh[1]);vb[3*i+2]=float(V[i][2]-sh[2]);}
    unsigned* ib=(unsigned*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,3*sizeof(unsigned),nf);
    for(size_t i=0;i<nf;++i){ib[3*i]=Fc[i][0];ib[3*i+1]=Fc[i][1];ib[3*i+2]=Fc[i][2];}
    rtcCommitGeometry(g);rtcAttachGeometry(sc,g);rtcReleaseGeometry(g);rtcCommitScene(sc);

    std::vector<float> tmax(nf,0.f); { std::ifstream tf(argv[2],std::ios::binary); tf.read((char*)tmax.data(),nf*sizeof(float)); }

    V3 up=norm3(sh);
    V3 ref=(std::fabs(up[2])<0.9)?V3{{0,0,1}}:V3{{1,0,0}};
    V3 ex=norm3(cross(up,ref)), en=cross(up,ex);
    std::vector<double> hr(nf),uu(nf); double cx=0,cy=0;
    { std::vector<double> xs(nf),ys(nf); for(size_t i=0;i<nf;++i){xs[i]=dot(ctr[i],ex);ys[i]=dot(ctr[i],en);uu[i]=dot(ctr[i],up);}
      std::vector<double> a=xs,b=ys; std::nth_element(a.begin(),a.begin()+nf/2,a.end()); cx=a[nf/2];
      std::nth_element(b.begin(),b.begin()+nf/2,b.end()); cy=b[nf/2];
      for(size_t i=0;i<nf;++i)hr[i]=std::hypot(xs[i]-cx,ys[i]-cy); }
    std::vector<size_t> floorf; { std::vector<double> el; for(size_t i=0;i<nf;++i)if(hr[i]<2500.0)el.push_back(uu[i]);
      std::nth_element(el.begin(),el.begin()+el.size()/4,el.end()); double thr=el[el.size()/4];
      for(size_t i=0;i<nf;++i)if(hr[i]<2500.0&&uu[i]<=thr)floorf.push_back(i); }
    std::cout<<"floor faces="<<floorf.size()<<"  rays/face="<<NR<<"  eps="<<EPS<<"\n";

    // Per-floor-face geometric indirect flux around the MACROSCOPIC UP hemisphere.
    std::vector<float> perface(nf,0.f);
    std::vector<double> Hvals(floorf.size(),0.0);
    #pragma omp parallel for schedule(dynamic)
    for(size_t fi=0; fi<floorf.size(); ++fi){
        size_t i=floorf[fi]; const V3&c=ctr[i];
        const V3 axis=up;
        V3 t=(std::fabs(axis[2])<0.9)?V3{{0,0,1}}:V3{{1,0,0}};
        V3 bx=norm3(cross(axis,t)), by=cross(axis,bx);
        std::mt19937 rng(1234567u + (unsigned)fi*2654435761u);
        std::uniform_real_distribution<double> U(0,1);
        RTCIntersectArguments ia; rtcInitIntersectArguments(&ia);
        double acc=0.0;
        for(int s=0;s<NR;++s){
            double u1=U(rng),u2=U(rng), r=std::sqrt(u1), ph=2*M_PI*u2;
            V3 d=norm3({{ r*std::cos(ph)*bx[0]+r*std::sin(ph)*by[0]+std::sqrt(1-u1)*axis[0],
                          r*std::cos(ph)*bx[1]+r*std::sin(ph)*by[1]+std::sqrt(1-u1)*axis[1],
                          r*std::cos(ph)*bx[2]+r*std::sin(ph)*by[2]+std::sqrt(1-u1)*axis[2] }});
            RTCRayHit rh; std::memset(&rh,0,sizeof(rh));
            rh.ray.org_x=float(c[0]-sh[0]+1.0*d[0]); rh.ray.org_y=float(c[1]-sh[1]+1.0*d[1]); rh.ray.org_z=float(c[2]-sh[2]+1.0*d[2]);
            rh.ray.dir_x=float(d[0]); rh.ray.dir_y=float(d[1]); rh.ray.dir_z=float(d[2]);
            rh.ray.tnear=0.f; rh.ray.tfar=1e12f; rh.ray.mask=0xFFFFFFFF; rh.hit.geomID=RTC_INVALID_GEOMETRY_ID;
            rtcIntersect1(sc,&rh,&ia);
            if(rh.hit.geomID!=RTC_INVALID_GEOMETRY_ID){ unsigned p=rh.hit.primID;
                if(p<nf){
                    // Only count the EMITTING (front) side: the ray must arrive on the
                    // hit face's outward-normal side (ray_dir . n_j < 0). A back-hit means
                    // face j's hot front faces away from the floor -> no emission toward it.
                    double cj = d[0]*nrm[p][0]+d[1]*nrm[p][1]+d[2]*nrm[p][2];
                    if(cj < 0.0){ double T=tmax[p]; acc += EPS*SIG*T*T*T*T; }
                } }
        }
        double H = EPS * acc / NR;   // absorbed indirect = eps_i * mean(eps_j sigma T^4)
        Hvals[fi]=H; perface[i]=float(H);
    }
    std::sort(Hvals.begin(),Hvals.end());
    auto pct=[&](double p){ return Hvals[std::min(Hvals.size()-1,(size_t)(p*Hvals.size())) ]; };
    double mean=0; for(double h:Hvals)mean+=h; mean/=Hvals.size();
    std::cout<<"--- GEOMETRIC ground-truth floor indirect (W/m2), hemisphere around UP, weighted by eps*sigma*Tmax^4 ---\n";
    std::cout<<"  median="<<pct(0.5)<<"  mean="<<mean<<"  p90="<<pct(0.9)<<"  max="<<Hvals.back()<<"  min="<<Hvals.front()<<"\n";
    double Tmed=std::pow((pct(0.5)+0.018)/(EPS*SIG),0.25);
    std::cout<<"  EB floor T from median geometric indirect = "<<Tmed<<" K\n";
    if(!outp.empty()){ std::ofstream o(outp,std::ios::binary); o.write((char*)perface.data(),nf*sizeof(float)); std::cout<<"  wrote per-face -> "<<outp<<"\n"; }
    rtcReleaseScene(sc); rtcReleaseDevice(dev);
    return 0;
}
