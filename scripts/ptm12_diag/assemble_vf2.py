#!/usr/bin/env python3
"""Assemble v2: MIN-HIT filter + SYMMETRIC ΣF<=1 cap (reciprocity-preserving congruence)
+ GEOMETRY-HASH header. Verify ΣF -> anchor.
Hardcoded to the moon3d 509,337-face mesh and its anchor files; assemble_vf_haworth.py
is the general version and the one to prefer.
The MIN-HIT filter was originally introduced to suppress single-ray survivors that the
symmetrization stamped onto both rows. That was a mitigation, not a repair: the actual
defect was the symmetrization divisor, fixed below (FC-08).
Usage: assemble_vf2.py <MIN_HITS=5>   (hits = F*NRAYS, NRAYS=40000)
CSR: int32 n, float64 geomhash, int64 indptr[n+1], int64 nnz, int32 cols[nnz], float32 data[nnz]."""
import numpy as np, glob, time, sys, os
DIAG="/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/scripts/ptm12_diag"
N=509337; NRAYS=int(sys.argv[2]) if len(sys.argv)>2 else 40000
MIN_HITS=float(sys.argv[1]) if len(sys.argv)>1 else 5.0
FTHR=MIN_HITS/NRAYS
G=np.load(DIAG+"/geom.npz"); A=G['area'].astype(np.float64)
GEOMHASH=float(A.sum()) + float((A*np.arange(1,N+1)).sum())*1e-9   # mesh-identity checksum
t0=time.time()
Is=[]; Js=[]; Vs=[]
for p in sorted(glob.glob("/tmp/pp_sweep/pp_*.bin")):
    raw=np.frombuffer(open(p,'rb').read(),dtype=np.uint8); b=memoryview(raw); q=0; L=len(raw)
    while q+8<=L:
        src=int(np.frombuffer(b[q:q+4],'<i4')[0]); nnz=int(np.frombuffer(b[q+4:q+8],'<i4')[0]); q+=8
        if nnz>0:
            arr=np.frombuffer(b[q:q+nnz*8],dtype=np.dtype([('j','<i4'),('f','<f4')]))
            keep=arr['f']>=FTHR                       # MIN-HIT: drop <MIN_HITS-ray survivors
            if keep.any():
                Is.append(np.full(int(keep.sum()),src,dtype=np.int32)); Js.append(arr['j'][keep].copy()); Vs.append(arr['f'][keep].copy())
        q+=nnz*8
I=np.concatenate(Is); J=np.concatenate(Js); V=np.concatenate(Vs); del Is,Js,Vs
print("MIN_HITS=%g (F>=%.3g) -> directed nnz %d (%.1fM)  %.0fs"%(MIN_HITS,FTHR,len(I),len(I)/1e6,time.time()-t0)); sys.stdout.flush()
S=A[I]*V.astype(np.float64)
lo=np.minimum(I,J).astype(np.int64); hi=np.maximum(I,J).astype(np.int64); del I,J,V
key=lo*np.int64(N)+hi
o=np.argsort(key,); key=key[o]; S=S[o]; lo=lo[o]; hi=hi[o]; del o
bnd=np.concatenate(([0],np.nonzero(np.diff(key))[0]+1,[len(key)]))
gs=bnd[:-1]; ge=bnd[1:]; csum=np.concatenate(([0.0],np.cumsum(S)))
# FC-08 (2026-08-19): divide by the 2 ray directions TRIED, not by the number that
# happened to hit. A direction with no hit is the measured zero it is (the sweep visits
# every face in both roles); dividing by the hit count conditions on detection and
# inflates far-field entries by up to 2x. See assemble_vf_haworth.py for the full note
# and test_vf_assembler_gate.py for the closed-form gate.
Ssym=(csum[ge]-csum[gs])/2.0                            # mean over the two directions tried
a=lo[gs]; bb=hi[gs]; del S,key,lo,hi
rows=np.concatenate([a,bb]); cols=np.concatenate([bb,a]); Sd=np.concatenate([Ssym,Ssym])
Aend=np.concatenate([A[a],A[bb]])                        # A_i for each directed entry -> F=S/A_i
del a,bb,Ssym
# symmetric ΣF<=1 cap: iterate S_ij *= sqrt(c_i c_j), c_i=1/max(1,ΣF_i)  (reciprocity-preserving)
F=Sd/Aend
for it in range(40):
    rs=np.zeros(N); np.add.at(rs,rows,F)
    if rs.max()<=1.0+1e-9: break
    c=np.ones(N); over=rs>1.0; c[over]=1.0/rs[over]
    F=F*np.sqrt(c[rows]*c[cols])
rs=np.zeros(N); np.add.at(rs,rows,F)
print("symmetric-cap sweeps=%d -> ΣF max %.4f (#>1=%d) mean %.4f  %.0fs"%(it+1,rs.max(),(rs>1.0+1e-6).sum(),rs.mean(),time.time()-t0))
ref=np.fromfile(DIAG+"/moon3d_rowfix_refSigF.bin",'<f8')
print("ANCHOR: assembled ΣF mean %.4f vs anchor %.4f ; frac(assembled>anchor) %.3f ; |Δ| mean %.4f"%(
    rs.mean(),ref.mean(),(rs>ref).mean(),np.abs(rs-ref).mean()))
def rd(p):
    with open(p,'rb') as f: raw=f.read()
    n=np.frombuffer(raw[:4],'<i4')[0]; return np.frombuffer(raw[4:4+4*n],'<f4').copy()
dsol=rd("/tmp/ptm12_rfd/direct_solar_max.bin"); tavg=rd("/tmp/ptm12_rfd/temperature_avg.bin")
floor=(dsol<1)&(tavg<100)
print("  FLOOR ΣF: assembled %.4f vs anchor %.4f (target ~0.131)"%(rs[floor].mean(),ref[floor].mean()))
# build CSR
o2=np.argsort(rows,); rows=rows[o2]; cols=cols[o2]; F=F[o2]; del o2
indptr=np.zeros(N+1,dtype=np.int64); np.add.at(indptr,rows+1,1); indptr=np.cumsum(indptr)
out="/tmp/pp_sweep/corrected_vf_v2.csr"
with open(out,"wb") as f:
    np.array([N],dtype=np.int32).tofile(f)
    np.array([GEOMHASH],dtype=np.float64).tofile(f)
    indptr.astype(np.int64).tofile(f)
    np.array([len(F)],dtype=np.int64).tofile(f)
    cols.astype(np.int32).tofile(f)
    F.astype(np.float32).tofile(f)
print("wrote %s  nnz=%d  %.2f GB  geomhash=%.6f  %.0fs"%(out,len(F),os.path.getsize(out)/1e9,GEOMHASH,time.time()-t0))
np.save(DIAG+"/assembled_rowsum_v2.npy",rs)
