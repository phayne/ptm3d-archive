/**
 * @file vf_hemisphere_mc.cpp
 * @brief Ground-truth view factor by cosine-weighted hemisphere Monte-Carlo ray
 *        casting (Embree), the DEFINITION of a view factor with exact occlusion.
 *        For floor faces, classify each ray's hit as warm (Tmax>thr) / cold terrain
 *        / sky(miss). Cosine weighting => hit-fraction = VF to that set.
 *
 * Decides whether the production H-matrix floor->warm-rim coupling (~0 at 400k) is
 * CORRECT (the floor genuinely can't see warm terrain per-facet) or an M1/visibility
 * under-count. Reports VF around (a) the facet normal and (b) the macroscopic up.
 *
 * Usage: vf_hemisphere_mc <mesh.ply> <tmax.bin> [n_rays=4000] [warm_K=250]
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
    if(argc<3){ std::cerr<<"usage: vf_hemisphere_mc <mesh.ply> <tmax.bin> [n_rays] [warm_K]\n"; return 2; }
    int NR = (argc>3)?std::atoi(argv[3]):4000;
    double warmK = (argc>4)?std::atof(argv[4]):250.0;
    std::vector<V3> V,VN; std::vector<std::array<int,3>> Fc;
    if(!readPLY(argv[1],V,Fc,VN)){ std::cerr<<"load failed\n"; return 1; }
    size_t nf=Fc.size();
    std::vector<V3> ctr(nf),nrm(nf);
    for(size_t i=0;i<nf;++i){ const V3&a=V[Fc[i][0]],&b=V[Fc[i][1]],&c=V[Fc[i][2]];
        ctr[i]={{(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3,(a[2]+b[2]+c[2])/3}};
        V3 vnn={{VN[Fc[i][0]][0]+VN[Fc[i][1]][0]+VN[Fc[i][2]][0],VN[Fc[i][0]][1]+VN[Fc[i][1]][1]+VN[Fc[i][2]][1],VN[Fc[i][0]][2]+VN[Fc[i][1]][2]+VN[Fc[i][2]][2]}};
        nrm[i]=norm3(vnn); }
    // recenter (match production) + Embree scene
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

    // local frame: up = radial = normalize(sh); floor = within 2500 m of center, lowest 25% u
    V3 up=norm3(sh);
    const double LIFT = std::getenv("PTM_VF_MC_LIFT") ? std::atof(std::getenv("PTM_VF_MC_LIFT")) : 0.0;
    std::cout<<"  ray lift along up = "<<LIFT<<" m\n";
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
    size_t step=std::max<size_t>(1,floorf.size()/200);

    auto sample=[&](const V3& axis, double& Fterr,double& Fwarm,double& Ffar){
        V3 t=(std::fabs(axis[2])<0.9)?V3{{0,0,1}}:V3{{1,0,0}};
        V3 bx=norm3(cross(axis,t)), by=cross(axis,bx);
        std::mt19937 rng(12345); std::uniform_real_distribution<double> U(0,1);
        long terr=0,warm=0,far=0,tot=0;
        RTCIntersectArguments ia; rtcInitIntersectArguments(&ia);
        for(size_t fi=0; fi<floorf.size(); fi+=step){
            size_t i=floorf[fi]; const V3&c=ctr[i];
            for(int s=0;s<NR;++s){
                double u1=U(rng),u2=U(rng), r=std::sqrt(u1), ph=2*M_PI*u2;
                V3 d={{ r*std::cos(ph)*bx[0]+r*std::sin(ph)*by[0]+std::sqrt(1-u1)*axis[0],
                        r*std::cos(ph)*bx[1]+r*std::sin(ph)*by[1]+std::sqrt(1-u1)*axis[1],
                        r*std::cos(ph)*bx[2]+r*std::sin(ph)*by[2]+std::sqrt(1-u1)*axis[2] }};
                d=norm3(d); tot++;
                RTCRayHit rh; std::memset(&rh,0,sizeof(rh));
                rh.ray.org_x=float(c[0]-sh[0]+1.0*d[0]+LIFT*up[0]); rh.ray.org_y=float(c[1]-sh[1]+1.0*d[1]+LIFT*up[1]); rh.ray.org_z=float(c[2]-sh[2]+1.0*d[2]+LIFT*up[2]);
                rh.ray.dir_x=float(d[0]); rh.ray.dir_y=float(d[1]); rh.ray.dir_z=float(d[2]);
                rh.ray.tnear=0.f; rh.ray.tfar=1e12f; rh.ray.mask=0xFFFFFFFF; rh.hit.geomID=RTC_INVALID_GEOMETRY_ID;
                rtcIntersect1(sc,&rh,&ia);
                if(rh.hit.geomID!=RTC_INVALID_GEOMETRY_ID){ terr++; unsigned p=rh.hit.primID;
                    if(p<nf && tmax[p]>warmK){ warm++; if(hr[p]>6000.0) far++; } }
            }
        }
        Fterr=double(terr)/tot; Fwarm=double(warm)/tot; Ffar=double(far)/tot;
    };
    double a1,a2,a3,b1,b2,b3;
    sample(up, a1,a2,a3);                 // cosine-weighted around macroscopic UP
    double Ft,Fw,Ff; long np=0; // facet: per-face axis -> do a combined pass
    { std::mt19937 rng(777); std::uniform_real_distribution<double> U(0,1);
      long terr=0,warm=0,far=0,tot=0; RTCIntersectArguments ia; rtcInitIntersectArguments(&ia);
      for(size_t fi=0;fi<floorf.size();fi+=step){ size_t i=floorf[fi]; const V3&c=ctr[i]; const V3 axis=nrm[i];
        V3 t=(std::fabs(axis[2])<0.9)?V3{{0,0,1}}:V3{{1,0,0}}; V3 bx=norm3(cross(axis,t)),by=cross(axis,bx); np++;
        for(int s=0;s<NR;++s){ double u1=U(rng),u2=U(rng),r=std::sqrt(u1),ph=2*M_PI*u2;
          V3 d=norm3({{ r*std::cos(ph)*bx[0]+r*std::sin(ph)*by[0]+std::sqrt(1-u1)*axis[0],
                        r*std::cos(ph)*bx[1]+r*std::sin(ph)*by[1]+std::sqrt(1-u1)*axis[1],
                        r*std::cos(ph)*bx[2]+r*std::sin(ph)*by[2]+std::sqrt(1-u1)*axis[2] }}); tot++;
          RTCRayHit rh; std::memset(&rh,0,sizeof(rh));
          rh.ray.org_x=float(c[0]-sh[0]+1.0*d[0]+LIFT*up[0]);rh.ray.org_y=float(c[1]-sh[1]+1.0*d[1]+LIFT*up[1]);rh.ray.org_z=float(c[2]-sh[2]+1.0*d[2]+LIFT*up[2]);
          rh.ray.dir_x=float(d[0]);rh.ray.dir_y=float(d[1]);rh.ray.dir_z=float(d[2]);
          rh.ray.tnear=0.f;rh.ray.tfar=1e12f;rh.ray.mask=0xFFFFFFFF;rh.hit.geomID=RTC_INVALID_GEOMETRY_ID;
          rtcIntersect1(sc,&rh,&ia);
          if(rh.hit.geomID!=RTC_INVALID_GEOMETRY_ID){terr++;unsigned p=rh.hit.primID; if(p<nf&&tmax[p]>warmK){warm++; if(hr[p]>6000.0)far++;}}
        } }
      Ft=double(terr)/tot;Fw=double(warm)/tot;Ff=double(far)/tot; }
    std::cout<<"  floor faces sampled="<<np<<"  rays/face="<<NR<<"  warm thr="<<warmK<<"K\n";
    std::cout<<"  [around FACET normal] F_terrain="<<Ft<<"  F_warm="<<Fw<<"  F_warm_far(>6km)="<<Ff<<"\n";
    std::cout<<"  [around UP (macro)  ] F_terrain="<<a1<<"  F_warm="<<a2<<"  F_warm_far(>6km)="<<a3<<"\n";
    rtcReleaseScene(sc); rtcReleaseDevice(dev);
    return 0;
}
