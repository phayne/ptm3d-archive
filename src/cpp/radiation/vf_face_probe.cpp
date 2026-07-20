/**
 * @file vf_face_probe.cpp
 * @brief Ground-truth probe of a single face's view-factor row, replicating the
 *        production VF visibility (recentered Embree core scene + the exact
 *        geometric culling of computeViewFactorRow) but per-pair rtcOccluded1,
 *        with RTC_SCENE_FLAG_ROBUST toggleable via env PTM_VF_ROBUST.
 *
 * Decides whether the fine-mesh floor->far-rim coupling collapse is a
 * non-robust-BVH false-occlusion artifact (recovered by robust mode) vs true
 * occlusion. Reports the probe face's F-total / F-far(>6km) / F-near(<2km) /
 * F-warm(T>250) and the far-band passed-cull / occluded / visible counts.
 *
 * Usage: vf_face_probe <mesh.ply binary_little_endian> [tmax.bin] [face_idx|-1]
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <embree4/rtcore.h>

using V3 = std::array<double, 3>;
static V3 sub(const V3&a,const V3&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static double dot(const V3&a,const V3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V3 cross(const V3&a,const V3&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}

// Minimal binary_little_endian PLY reader handling mixed property types
// (float/double/int...). Extracts vertex x,y,z (whatever their declared type).
static int plySize(const std::string& t){
    if(t=="float"||t=="float32"||t=="int"||t=="int32"||t=="uint"||t=="uint32") return 4;
    if(t=="double"||t=="float64") return 8;
    if(t=="char"||t=="uchar"||t=="int8"||t=="uint8") return 1;
    if(t=="short"||t=="ushort"||t=="int16"||t=="uint16") return 2;
    return 4;
}
static bool readPLY(const std::string& path, std::vector<V3>& verts,
                    std::vector<std::array<int,3>>& faces, std::vector<V3>& vnorms) {
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;
    std::string line; size_t nv=0,nf=0; bool inV=false;
    std::vector<std::pair<std::string,std::string>> vprops; // (type,name)
    while(std::getline(f,line)){
        if(!line.empty()&&line.back()=='\r') line.pop_back();
        std::istringstream is(line); std::string tok; is>>tok;
        if(tok=="element"){ std::string what; size_t n; is>>what>>n;
            if(what=="vertex"){nv=n;inV=true;} else {inV=false;} if(what=="face")nf=n; }
        else if(tok=="property"){ if(inV){ std::string t,nm; is>>t>>nm; vprops.push_back({t,nm}); } }
        else if(tok=="end_header") break;
    }
    if(nv==0||nf==0){ std::cerr<<"bad PLY counts\n"; return false; }
    int off[3]={-1,-1,-1}, noff[3]={-1,-1,-1}; std::string ty[3], nty[3]; int stride=0;
    for(auto&p:vprops){
        if(p.second=="x"){off[0]=stride;ty[0]=p.first;}
        else if(p.second=="y"){off[1]=stride;ty[1]=p.first;}
        else if(p.second=="z"){off[2]=stride;ty[2]=p.first;}
        else if(p.second=="nx"){noff[0]=stride;nty[0]=p.first;}
        else if(p.second=="ny"){noff[1]=stride;nty[1]=p.first;}
        else if(p.second=="nz"){noff[2]=stride;nty[2]=p.first;}
        stride+=plySize(p.first); }
    if(off[0]<0||off[1]<0||off[2]<0){ std::cerr<<"no x/y/z\n"; return false; }
    bool has_vn = (noff[0]>=0&&noff[1]>=0&&noff[2]>=0);
    verts.resize(nv); vnorms.assign(nv, {{0,0,0}});
    std::vector<char> row(stride);
    auto rd=[&](int o,const std::string&t)->double{ if(t=="double"||t=="float64"){double d;std::memcpy(&d,row.data()+o,8);return d;} float fl;std::memcpy(&fl,row.data()+o,4);return fl; };
    for(size_t i=0;i<nv;++i){ f.read(row.data(), stride);
        for(int k=0;k<3;++k) verts[i][k]=rd(off[k],ty[k]);
        if(has_vn) for(int k=0;k<3;++k) vnorms[i][k]=rd(noff[k],nty[k]); }
    faces.resize(nf);
    for(size_t i=0;i<nf;++i){ unsigned char cnt; f.read((char*)&cnt,1);
        std::vector<int> idx(cnt); f.read((char*)idx.data(), cnt*sizeof(int));
        faces[i]={idx[0],idx[1],idx[2]}; }
    return (bool)f || f.eof();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: vf_face_probe <mesh.ply> [tmax.bin] [face_idx]\n"; return 2; }
    std::string ply = argv[1];
    std::string tmaxf = (argc > 2) ? argv[2] : "";
    long want = (argc > 3) ? std::atol(argv[3]) : -1;

    std::vector<V3> V, VN; std::vector<std::array<int,3>> Fc;
    if(!readPLY(ply,V,Fc,VN)){ std::cerr<<"mesh load failed\n"; return 1; }
    size_t nf=Fc.size();
    bool has_vn = false; for(auto&n:VN){ if(dot(n,n)>1e-12){has_vn=true;break;} }
    std::cout << "loaded " << V.size() << " verts, " << nf << " faces  (PLY normals: "
              << (has_vn?"yes":"no") << ")\n";

    std::vector<V3> ctr(nf), nrm(nf); std::vector<double> area(nf);
    for (size_t i=0;i<nf;++i){
        const V3&v0=V[Fc[i][0]],&v1=V[Fc[i][1]],&v2=V[Fc[i][2]];
        ctr[i]={(v0[0]+v1[0]+v2[0])/3,(v0[1]+v1[1]+v2[1])/3,(v0[2]+v1[2]+v2[2])/3};
        V3 e1=sub(v1,v0),e2=sub(v2,v0),n=cross(e1,e2);
        double L=std::sqrt(dot(n,n)); area[i]=0.5*L; if(L>0){n[0]/=L;n[1]/=L;n[2]/=L;}
        if(has_vn){ // use PLY vertex normals (consistent outward orientation), keep cross for area
            V3 a=VN[Fc[i][0]],b=VN[Fc[i][1]],c=VN[Fc[i][2]];
            V3 vn={a[0]+b[0]+c[0],a[1]+b[1]+c[1],a[2]+b[2]+c[2]};
            double Ln=std::sqrt(dot(vn,vn));
            if(Ln>0){ n={vn[0]/Ln,vn[1]/Ln,vn[2]/Ln}; }
        }
        nrm[i]=n;
    }
    V3 bbmin=V[0],bbmax=V[0];
    for(const auto&v:V) for(int k=0;k<3;++k){bbmin[k]=std::min(bbmin[k],v[k]);bbmax[k]=std::max(bbmax[k],v[k]);}
    V3 shift={0.5*(bbmin[0]+bbmax[0]),0.5*(bbmin[1]+bbmax[1]),0.5*(bbmin[2]+bbmax[2])};

    bool robust = std::getenv("PTM_VF_ROBUST") && std::atoi(std::getenv("PTM_VF_ROBUST"))!=0;
    RTCDevice dev=rtcNewDevice(nullptr); RTCScene scene=rtcNewScene(dev);
    if(robust) rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    RTCGeometry g=rtcNewGeometry(dev,RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vb=(float*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,3*sizeof(float),V.size());
    for(size_t i=0;i<V.size();++i){vb[3*i]=float(V[i][0]-shift[0]);vb[3*i+1]=float(V[i][1]-shift[1]);vb[3*i+2]=float(V[i][2]-shift[2]);}
    unsigned* ib=(unsigned*)rtcSetNewGeometryBuffer(g,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,3*sizeof(unsigned),nf);
    for(size_t i=0;i<nf;++i){ib[3*i]=Fc[i][0];ib[3*i+1]=Fc[i][1];ib[3*i+2]=Fc[i][2];}
    rtcCommitGeometry(g);rtcAttachGeometry(scene,g);rtcReleaseGeometry(g);rtcCommitScene(scene);
    std::cout<<"scene built; ROBUST="<<(robust?"ON":"OFF")<<"\n";

    std::vector<float> tmax;
    if(!tmaxf.empty()){std::ifstream tf(tmaxf,std::ios::binary);tmax.assign(nf,0.f);tf.read((char*)tmax.data(),nf*sizeof(float));}

    V3 mean={0,0,0}; for(auto&c:ctr){mean[0]+=c[0];mean[1]+=c[1];mean[2]+=c[2];}
    for(int k=0;k<3;++k)mean[k]/=nf;
    int up=(std::abs(mean[0])>std::abs(mean[1])&&std::abs(mean[0])>std::abs(mean[2]))?0:(std::abs(mean[1])>std::abs(mean[2])?1:2);
    int h0=(up==0)?1:0, h1=(up==2)?1:2;
    std::vector<double> hr(nf);
    { std::vector<double> xc(nf),yc(nf); for(size_t i=0;i<nf;++i){xc[i]=ctr[i][h0];yc[i]=ctr[i][h1];}
      std::vector<double> xs=xc,ys=yc; std::nth_element(xs.begin(),xs.begin()+nf/2,xs.end()); double cx=xs[nf/2];
      std::nth_element(ys.begin(),ys.begin()+nf/2,ys.end()); double cy=ys[nf/2];
      for(size_t i=0;i<nf;++i)hr[i]=std::hypot(xc[i]-cx,yc[i]-cy); }

    // Build probe set: want>=0 -> that face; -1 -> deepest within 1km;
    // -2 -> sample up to 150 floor faces (hr<2500, lowest 25% elevation).
    std::vector<long> probes;
    if(want>=0) probes.push_back(want);
    else if(want==-1){ double best=1e30; long pf=-1; for(size_t i=0;i<nf;++i) if(hr[i]<1000.0&&ctr[i][up]<best){best=ctr[i][up];pf=(long)i;} probes.push_back(pf); }
    else { // -2 aggregate
        std::vector<long> fl; std::vector<double> el;
        for(size_t i=0;i<nf;++i) if(hr[i]<2500.0){fl.push_back((long)i);el.push_back(ctr[i][up]);}
        std::vector<double> es=el; std::nth_element(es.begin(),es.begin()+es.size()/4,es.end());
        double thr=es[es.size()/4];
        std::vector<long> floor; for(size_t k=0;k<fl.size();++k) if(el[k]<=thr) floor.push_back(fl[k]);
        size_t step=std::max<size_t>(1,floor.size()/150);
        for(size_t k=0;k<floor.size();k+=step) probes.push_back(floor[k]);
        std::cout<<"aggregate over "<<probes.size()<<" floor faces (of "<<floor.size()<<")\n";
    }

    double nlift = std::getenv("PTM_VF_NLIFT") ? std::atof(std::getenv("PTM_VF_NLIFT")) : 0.0;
    int msample = std::getenv("PTM_VF_MSAMPLE") ? std::atoi(std::getenv("PTM_VF_MSAMPLE")) : 1; // 1 or 4
    double ignore_near = std::getenv("PTM_VF_IGNORE_NEAR") ? std::atof(std::getenv("PTM_VF_IGNORE_NEAR")) : 0.0;
    std::cout<<"  normal lift = "<<nlift<<" m  msample="<<msample<<"  ignore_near(t/r<)="<<ignore_near<<"\n";
    RTCOccludedArguments oargs; rtcInitOccludedArguments(&oargs);
    RTCIntersectArguments iargs; rtcInitIntersectArguments(&iargs);
    double sFtot=0,sFfar=0,sFnear=0,sFwarm=0; long sCull=0,sOcc=0,sVis=0; int npr=0;
    long hit_near=0, hit_far=0; double sum_thr=0; long nhit=0;  // occluded-far hit location
    for(long pf: probes){ if(pf<0) continue; npr++;
        const V3& ci=ctr[pf]; const V3& ni=nrm[pf];
        double Ftot=0,Ffar=0,Fnear=0,Fwarm=0; long cull_far=0,occ_far=0,vis_far=0;
        for(size_t j=0;j<nf;++j){
            if((long)j==pf) continue;
            V3 rv=sub(ctr[j],ci); double r2=dot(rv,rv); if(r2<1e-20)continue;
            double r=std::sqrt(r2); V3 rh={rv[0]/r,rv[1]/r,rv[2]/r};
            double di=dot(ni,rh),dj=dot(nrm[j],rh);
            bool farband=hr[j]>6000.0;
            if(di<=0.0||dj>=0.0)continue;
            double ci_=di,cj_=-dj; if(ci_<0.01||cj_<0.01)continue;
            double estF=(area[j]/M_PI)*(ci_*cj_)/r2; if(estF<1e-9)continue;
            if(farband)cull_far++;
            // Fractional visibility: trace from face i's centroid (lifted along ni)
            // to msample points on the TARGET face j (centroid + vertices, lifted
            // along nrm[j]). V = visible fraction = area-integrated visibility,
            // which captures the sliver of a grazing target visible above an
            // intervening wall (the M1 centroid line alone misses it).
            std::vector<V3> tgtpts; tgtpts.push_back(ctr[j]);
            if(msample>1){ for(int k=0;k<3;++k) tgtpts.push_back(V[Fc[j][k]]); }
            V3 P0base={ci[0]-shift[0]+nlift*ni[0], ci[1]-shift[1]+nlift*ni[1], ci[2]-shift[2]+nlift*ni[2]};
            int nvis=0, ntr=0;
            for(const V3& tp : tgtpts){
                V3 P0=P0base;
                V3 P1={tp[0]-shift[0]+nlift*nrm[j][0], tp[1]-shift[1]+nlift*nrm[j][1], tp[2]-shift[2]+nlift*nrm[j][2]};
                V3 dv=sub(P1,P0); double dlen=std::sqrt(dot(dv,dv)); if(dlen<1e-6){nvis++;ntr++;continue;}
                V3 dh={dv[0]/dlen,dv[1]/dlen,dv[2]/dlen};
                double off=std::max(dlen*0.001,0.5);off=std::min(off,dlen*0.1);
                double em=std::max(dlen*0.001,0.5);em=std::min(em,dlen*0.1);
                double tfar=dlen-off-em; ntr++;
                if(tfar<0.1){nvis++;continue;}
                RTCRayHit rh1; std::memset(&rh1,0,sizeof(rh1));
                rh1.ray.org_x=float(P0[0]+off*dh[0]);rh1.ray.org_y=float(P0[1]+off*dh[1]);rh1.ray.org_z=float(P0[2]+off*dh[2]);
                rh1.ray.dir_x=float(dh[0]);rh1.ray.dir_y=float(dh[1]);rh1.ray.dir_z=float(dh[2]);
                rh1.ray.tnear=0.f;rh1.ray.tfar=float(tfar);rh1.ray.mask=0xFFFFFFFF;rh1.ray.flags=0;rh1.ray.time=0.f;
                rh1.hit.geomID=RTC_INVALID_GEOMETRY_ID; rtcIntersect1(scene,&rh1,&iargs);
                bool hit = (rh1.hit.geomID!=RTC_INVALID_GEOMETRY_ID);
                double thr = hit ? (rh1.ray.tfar/dlen) : 1.0;
                // Occluding only if the hit is beyond the near self-occlusion zone.
                bool occ = hit && (thr >= ignore_near);
                if(!occ) nvis++;
                if(farband && hit){ sum_thr+=thr; nhit++; if(thr<0.5)hit_near++; else hit_far++; }
            }
            double Vfrac = ntr? double(nvis)/ntr : 1.0;
            if(farband){ if(Vfrac<0.5)occ_far++; else vis_far++; }
            double Fcj = estF*Vfrac;
            if(Fcj>1e-9){ Ftot+=Fcj; if(hr[j]>6000.0)Ffar+=Fcj; if(hr[j]<2000.0)Fnear+=Fcj;
                          if(!tmax.empty()&&tmax[j]>250.f)Fwarm+=Fcj; }
        }
        sFtot+=Ftot;sFfar+=Ffar;sFnear+=Fnear;sFwarm+=Fwarm;sCull+=cull_far;sOcc+=occ_far;sVis+=vis_far;
    }
    std::cout<<"  MEAN over "<<npr<<" probe(s): F_total="<<sFtot/npr<<"  F_far(>6km)="<<sFfar/npr
             <<"  F_near(<2km)="<<sFnear/npr<<"  F_warm(T>250)="<<sFwarm/npr<<"\n";
    std::cout<<"  far band totals: passed_cull="<<sCull<<"  occluded="<<sOcc<<"  visible="<<sVis
             <<"  (occluded "<<(sCull?100.0*sOcc/sCull:0)<<"%)\n";
    std::cout<<"  occluded-far hit location: hit_near(t<0.5r)="<<hit_near<<"  hit_far(t>=0.5r)="<<hit_far
             <<"  mean t/r="<<(nhit?sum_thr/nhit:0)<<"  (near=>self/origin artifact, far=>real far-wall)\n";
    rtcReleaseScene(scene); rtcReleaseDevice(dev);
    return 0;
}
