"""COPY for the PTM-727 decisive control (Ada 2026-10-14) of Fermi's replay_f2.py, UNCHANGED physics.
Three additions only: (0) the AS-RUN post-equilibration columns (Fermi C1: the A' arm) and (1) the spun-up column state at the L1-start phase (Ts right after the
spin loop, BEFORE the output lunations) is saved IN FULL (all NZ nodes, replay grid z) to
spun_L1start_<tag>.npz for injection into a PTM state file; (2) outputs go to this directory,
never to Fermi's. Running it with Fermi's arguments (0.93 60) must reproduce replay_f093.npz.

Fermi 2026-10-13 (PTM-727 adjudication). Independent 1-D column replay, numpy, CPU only.
Drives each face of forcing_subset.npz with its RECORDED absorbed forcing (PTM dt65_coupled_3lun frames).
  AS-RUN : seed isothermal-ish at T_w = T_rad(<absorbed_at>+Qb) [no indirect; eq_seed.cpp:421 fallback,
           ptm_thermal_sim.cpp:10331-10358], + 2 lunations at eq cadence with constant q_eq added, then 3 output lunations.
  SPUNUP : same output forcing, after NSPIN periodic repeats of the L1 forcing.
Difference AS-RUN - SPUNUP at FIXED forcing = spin-up memory (F1 + F2); no seasonal confound.
Properties: Hayne et al. 2017 (config values); backward-Euler, massless surface node (PTM BC form), Qb flux bottom.
NOT PTM's solver (own implementation, own grid spacing): a consistency instrument, validated against PTM frames below.
usage: replay_f2.py <qeq_factor> <nspin> <tag>
"""
import numpy as np, sys, time, json
P='/Users/paha3326/projects/moon3d_R1/runs/stage3_f2/fermi/'          # inputs (read-only)
P2='/Users/paha3326/projects/moon3d_R1/runs/ptm727_control/'          # outputs
fq=float(sys.argv[1]); NSPIN=int(sys.argv[2]); TAG=sys.argv[3]
eps,sig,Qb=0.95,5.670374419e-8,0.022
ks,kd,H,rhos,rhod,chi=7.4e-4,3.4e-3,0.06,1100.,1800.,2.7
cpc=np.array([-3.6125,2.7431,2.3616e-3,-1.2340e-5,8.9093e-9])
S=1361.0; A0=0.12; w=0.496358; gam=np.sqrt(1-w)
# grid: 23 nodes, z_max 0.47063, dz0 0.00330791 (PTM pre-flight log), geometric fill
NZ,ZMAX,DZ0=23,0.47063,0.00330791
from scipy.optimize import brentq
r=brentq(lambda r: DZ0*(r**(NZ-1)-1)/(r-1)-ZMAX,1.0001,2.0)
dz=DZ0*r**np.arange(NZ-1); z=np.r_[0,np.cumsum(dz)]
rho=(rhod-(rhod-rhos)*np.exp(-z/H))[:,None]; kc=(kd-(kd-ks)*np.exp(-z/H))[:,None]
kci=0.5*(kc[1:]+kc[:-1]); hcv=np.r_[0,0.5*(dz[:-1]+dz[1:]),0.5*dz[-1]][:,None]
def cp(T): return cpc[0]+T*(cpc[1]+T*(cpc[2]+T*(cpc[3]+T*cpc[4])))
D=np.load(P+'forcing_subset.npz'); idx=D['idx']; N=idx.size
ds=D['direct_solar'].astype(np.float64)
mu=np.clip(ds/S,1e-6,1.0); Ah=1-gam*(1+2*mu)/(1+2*gam*mu)
Qout=(1-Ah)*ds+(1-A0)*D['primary_reflected']+eps*D['primary_ir']+D['indirect_solar']+D['indirect_ir']
Qabs_at=(1-Ah)*ds+(1-A0)*D['primary_reflected']+eps*D['primary_ir']    # seed forcing (no indirect)
qind=(D['indirect_solar']+D['indirect_ir']).astype(np.float64).mean(0)
Tptm=D['temperature'].astype(np.float64)
del ds,mu,Ah
def step(T,Q,dt):
    """one backward-Euler step; T (NZ,N); Q (N,) absorbed surface flux; props lagged."""
    k=kc*(1+chi*(T/350.)**3); ki=0.5*(k[1:]+k[:-1]); C=rho*cp(T)*hcv
    g=ki/dz[:,None]                        # (NZ-1,N)
    a=np.zeros_like(T); b=np.zeros_like(T); c=np.zeros_like(T); d=np.zeros_like(T)
    T0=T[0]; h=4*eps*sig*T0**3
    b[0]=h+g[0]; c[0]=-g[0]; d[0]=Q+3*eps*sig*T0**4
    a[1:-1]=-g[:-1]; c[1:-1]=-g[1:]; b[1:-1]=C[1:-1]/dt+g[:-1]+g[1:]; d[1:-1]=C[1:-1]/dt*T[1:-1]
    a[-1]=-g[-1]; b[-1]=C[-1]/dt+g[-1]; d[-1]=C[-1]/dt*T[-1]+Qb
    for i in range(1,NZ):
        m=a[i]/b[i-1]; b[i]-=m*c[i-1]; d[i]-=m*d[i-1]
    X=np.empty_like(T); X[-1]=d[-1]/b[-1]
    for i in range(NZ-2,-1,-1): X[i]=(d[i]-c[i]*X[i+1])/b[i]
    return X
def run(T,Qser,dt,nsub,rec=False):
    if rec: out=np.empty((Qser.shape[0],N))
    E0=None
    for t in range(Qser.shape[0]):
        for s in range(nsub): T=step(T,Qser[t],dt/nsub)
        if rec: out[t]=T[0]
    return (T,out) if rec else T
def stats(Ts):
    o={}
    for k in range(3):
        S_=Ts[360*k:360*(k+1)]
        o[k+1]=dict(Tmax=S_.max(0),Tmean=S_.mean(0),Tmin=S_.min(0),T4=(S_**4).mean(0),Tlast=S_[-1],Tfirst=S_[0])
    return o
OUT_DT=7087.2; EQ_DT=21262.0; NSUB=6
t0=time.time()
# ---- AS-RUN
Tw=((Qabs_at.mean(0)+Qb)/(eps*sig))**0.25
T=Tw[None,:]+Qb*np.r_[0,np.cumsum(dz/kci[:,0])][:,None]   # conductive steady profile anchored at T_w
qeq=fq*qind
eqF=np.vstack([Qabs_at[0:360:3]]*2)+qeq[None,:]                    # 240 eq samples (L1 pattern subsampled, x2)
T=run(T,eqF,EQ_DT,6)
print('eq done',round(time.time()-t0),'s',flush=True)
Tdeep_asrun0=T.copy()
np.savez(P2+f'asrun_posteq_{TAG}.npz',idx=idx,z=z,T=T.astype(np.float64))   # C1: A' columns
print('wrote full as-run post-eq columns',flush=True)
T,Ta=run(T,Qout,OUT_DT,NSUB,rec=True)
print('as-run done',round(time.time()-t0),'s',flush=True)
# ---- SPUN-UP: periodic L1 forcing NSPIN times from the as-run end state, then the 3 output lunations
Ts=T.copy(); conv=[]
for n in range(NSPIN):
    Tprev=Ts.copy(); Ts=run(Ts,Qout[0:360:3],3*OUT_DT,3)   # spin-up at eq cadence (120/lunation)
    conv.append(float(np.percentile(np.abs(Ts[-1]-Tprev[-1]),99)))
    if n%10==9: print('spin',n+1,conv[-1],round(time.time()-t0),'s',flush=True)
print('spin conv p99 |dT_bottom| per cycle (last 5):',conv[-5:],round(time.time()-t0),'s',flush=True)
np.savez(P2+f'spun_L1start_{TAG}.npz',idx=idx,z=z,T=Ts.astype(np.float64),conv=np.array(conv))
print('wrote full spun-up L1-start columns',flush=True)
Ts,Tb=run(Ts,Qout,OUT_DT,NSUB,rec=True)
print('spun-up done',round(time.time()-t0),'s',flush=True)
A=stats(Ta); B=stats(Tb); C=stats(Tptm)
np.savez(P2+f'replay_{TAG}.npz',idx=idx,**{f'{nm}_L{k}_{f}':v[k][f] for nm,v in (('asrun',A),('spun',B),('ptm',C)) for k in (1,2,3) for f in v[k]},
         Tdeep0=Tdeep_asrun0[[0,8,14,18,22]],Tdeep_spun=Ts[[0,8,14,18,22]],z=z,qeq=qeq,qind=qind,conv=np.array(conv))
print('wrote',TAG)
