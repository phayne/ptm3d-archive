/**
 * @file vf_examine_pairs.cpp
 * @brief Careful, concrete re-examination of whether deep-floor faces geometrically
 *        see the warm rim, decoupling the three effects that could zero F_warm:
 *        (1) facet-normal backface cull (cos_i<0), (2) target backface (cos_j<0),
 *        (3) occlusion. Uses a GENEROUS ray offset (lift along macroscopic up) to
 *        avoid grazing self-occlusion artifacts, and reports counts under both the
 *        per-facet normal and the macroscopic up normal.
 *
 * Usage: vf_examine_pairs <mesh.ply> <tmax.bin> [warm_K=250] [n_floor=20] [lift_m=15]
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
static V3 sub(const V3&a,const V3&b){return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};}
static V3 nrm3(const V3&a){double L=std::sqrt(dot(a,a));return L>0?V3{{a[0]/L,a[1]/L,a[2]/L}}:a;}

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
    if(argc<3){std::cerr<<"usage: vf_examine_pairs <mesh.ply> <tmax.bin> [warmK] [nfloor] [lift_m]\n";return 2;}
    double warmK=argc>3?std::atof(argv[3]):250.0; int nfloor=argc>4?std::atoi(argv[4]):20; double lift=argc>5?std::atof(argv[5]):15.0;
    std::vector<V3> V,N; std::vector<std::array<int,3>> F;
    if(!readPLY(argv[1],V,F,N)){std::cerr<<"load fail\n";return 1;}
    size_t nf=F.size(); std::vector<V3> ctr(nf),nrm(nf);
    for(size_t i=0;i<nf;++i){const V3&a=V[F[i][0]],&b=V[F[i][1]],&c=V[F[i][2]];
        ctr[i]={{(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3,(a[2]+b[2]+c[2])/3}};
        nrm[i]=nrm3({{N[F[i][0]][0]+N[F[i][1]][0]+N[F[i][2]][0],N[F[i][0]][1]+N[F[i][1]][1]+N[F[i][2]][1],N[F[i][0]][2]+N[F[i][1]][2]+N[F[i][2]][2]}}); }
    V3 bbmin=V[0],bbmax=V[0]; for(auto&v:V)for(int k=0;k<3;++k){bbmin[k]=std::min(bbmin[k],v[k]);bbmax[k]=std::max(bbmax[k],v[k]);}
    V3 sh={{0.5*(bbmin[0]+bbmax[0]),0.5*(bbmin[1]+bbmax[1]),0.5*(bbmin[2]+bbmax[2])}};
    RTCDevice dev=rtcNewDevice(nullptr); RTCScene sc=rtcNewScene(dev);
    RTCGeometry g=rtcNewGeometry(dev,RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vb=(float*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,12,V.size());
    for(size_t i=0;i<V.size();++i){vb[3*i]=float(V[i][0]-sh[0]);vb[3*i+1]=float(V[i][1]-sh[1]);vb[3*i+2]=float(V[i][2]-sh[2]);}
    unsigned* ib=(unsigned*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,12,nf);
    for(size_t i=0;i<nf;++i){ib[3*i]=F[i][0];ib[3*i+1]=F[i][1];ib[3*i+2]=F[i][2];}
    rtcCommitGeometry(g);rtcAttachGeometry(sc,g);rtcReleaseGeometry(g);rtcCommitScene(sc);
    std::vector<float> tmax(nf,0.f);{std::ifstream tf(argv[2],std::ios::binary);tf.read((char*)tmax.data(),nf*sizeof(float));}

    V3 up=nrm3(sh); V3 rf=(std::fabs(up[2])<0.9)?V3{{0,0,1}}:V3{{1,0,0}};
    V3 ex=nrm3({{up[1]*rf[2]-up[2]*rf[1],up[2]*rf[0]-up[0]*rf[2],up[0]*rf[1]-up[1]*rf[0]}});
    V3 en={{up[1]*ex[2]-up[2]*ex[1],up[2]*ex[0]-up[0]*ex[2],up[0]*ex[1]-up[1]*ex[0]}};
    std::vector<double> hr(nf),uu(nf); for(size_t i=0;i<nf;++i){double e=dot(ctr[i],ex),n=dot(ctr[i],en);hr[i]=std::hypot(e,n);uu[i]=dot(ctr[i],up);}
    // crater center via median, recompute hr about it
    { std::vector<double> xs(nf),ys(nf); for(size_t i=0;i<nf;++i){xs[i]=dot(ctr[i],ex);ys[i]=dot(ctr[i],en);}
      auto md=[&](std::vector<double>v){std::nth_element(v.begin(),v.begin()+v.size()/2,v.end());return v[v.size()/2];};
      double cx=md(xs),cy=md(ys); for(size_t i=0;i<nf;++i)hr[i]=std::hypot(xs[i]-cx,ys[i]-cy); }
    std::vector<size_t> fl; {std::vector<double> el;for(size_t i=0;i<nf;++i)if(hr[i]<2500.0)el.push_back(uu[i]);
        std::nth_element(el.begin(),el.begin()+el.size()/4,el.end());double thr=el[el.size()/4];
        for(size_t i=0;i<nf;++i)if(hr[i]<2500.0&&uu[i]<=thr)fl.push_back(i);}
    size_t step=std::max<size_t>(1,fl.size()/nfloor);

    long warm_total=0, vis_line=0, facet_front=0, up_front=0, facet_front_vis=0, up_front_vis=0, far_vis=0;
    long cj_front=0, phys_couple=0;   // warm face faces the floor (cos_j>0); physically-coupling pairs
    RTCOccludedArguments oa; rtcInitOccludedArguments(&oa);
    int probed=0;
    for(size_t k=0;k<fl.size();k+=step){ size_t i=fl[k]; probed++;
        const V3&ci=ctr[i];
        for(size_t j=0;j<nf;++j){ if(tmax[j]<=warmK) continue; warm_total++;
            V3 rv=sub(ctr[j],ci); double r=std::sqrt(dot(rv,rv)); if(r<1)continue; V3 d={{rv[0]/r,rv[1]/r,rv[2]/r}};
            double ci_f=dot(nrm[i],d), ci_u=dot(up,d), cj=-dot(nrm[j],d);
            if(ci_f>0.0) facet_front++; if(ci_u>0.0) up_front++; if(cj>0.0) cj_front++;
            // generous-offset straight-line occlusion: lift BOTH ends along up by `lift`
            V3 o={{ci[0]-sh[0]+lift*up[0], ci[1]-sh[1]+lift*up[1], ci[2]-sh[2]+lift*up[2]}};
            V3 t={{ctr[j][0]-sh[0]+lift*up[0], ctr[j][1]-sh[1]+lift*up[1], ctr[j][2]-sh[2]+lift*up[2]}};
            V3 dv=sub(t,o); double dl=std::sqrt(dot(dv,dv)); V3 dh={{dv[0]/dl,dv[1]/dl,dv[2]/dl}};
            RTCRay ray; ray.org_x=float(o[0]);ray.org_y=float(o[1]);ray.org_z=float(o[2]);
            ray.dir_x=float(dh[0]);ray.dir_y=float(dh[1]);ray.dir_z=float(dh[2]);
            ray.tnear=float(0.001*dl); ray.tfar=float(dl*0.999); ray.mask=0xFFFFFFFF;ray.flags=0;ray.time=0.f;
            rtcOccluded1(sc,&ray,&oa); bool vis=(ray.tfar>=0.0f);
            if(vis){ vis_line++; if(hr[j]>6000.0)far_vis++; if(ci_f>0.0)facet_front_vis++; if(ci_u>0.0)up_front_vis++;
                     if(ci_u>0.0 && cj>0.0) phys_couple++; }
        }
    }
    std::cout<<"probed "<<probed<<" deep-floor faces; warm faces (T>"<<warmK<<")/probe="<<(warm_total/probed)<<"  lift="<<lift<<"m\n";
    auto pc=[&](long x){return 100.0*x/std::max(1L,warm_total);};
    std::cout<<"  cos_i>0 facet-normal : "<<pc(facet_front)<<"%   up-normal: "<<pc(up_front)<<"%   (of all floor-warm pairs)\n";
    std::cout<<"  straight line UNOCCLUDED (lifted "<<lift<<"m): "<<pc(vis_line)<<"%   of those, far(>6km): "<<pc(far_vis)<<"%\n";
    std::cout<<"  UNOCCLUDED & facet-front: "<<pc(facet_front_vis)<<"%    UNOCCLUDED & up-front: "<<pc(up_front_vis)<<"%\n";
    std::cout<<"  warm face faces floor (cos_j>0): "<<pc(cj_front)<<"%   PHYSICALLY-COUPLING (vis & up-front & cos_j>0): "<<pc(phys_couple)<<"%\n";
    rtcReleaseScene(sc);rtcReleaseDevice(dev); return 0;
}
