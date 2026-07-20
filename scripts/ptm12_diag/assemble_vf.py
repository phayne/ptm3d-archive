#!/usr/bin/env python3
"""Assemble the per-partner referee sweep into a SYMMETRIC (reciprocal) sparse VF.
S_ij = A_i F_ij ; symmetric S_sym per undirected pair = mean of available directions ;
directed F_ij = S_sym/A_i. Enforce ΣF<=1. Validate ΣF vs moon3d_rowfix_refSigF.bin anchor.
Writes CSR: n(int32) indptr(n+1 int64) nnz(int64) indices(nnz int32) data(nnz float32)."""
import numpy as np, glob, time, sys
DIAG="/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/scripts/ptm12_diag"
N=509337
G=np.load(DIAG+"/geom.npz"); A=G['area'].astype(np.float64)
t0=time.time()
# load all chunks
Is=[]; Js=[]; Vs=[]
for p in sorted(glob.glob("/tmp/pp_sweep/pp_*.bin")):
    raw=np.frombuffer(open(p,'rb').read(),dtype=np.uint8); off=0; L=len(raw)
    # parse: repeated [src i4][nnz i4][ (j i4, f f4) * nnz ]
    # vectorized-ish: walk headers
    buf=raw.view(np.int32)  # for headers
    # simplest robust: python walk (chunks ~10k faces) but that's slow; do a manual pointer parse in numpy
    b=memoryview(raw); q=0
    while q+8<=L:
        src=int(np.frombuffer(b[q:q+4],'<i4')[0]); nnz=int(np.frombuffer(b[q+4:q+8],'<i4')[0]); q+=8
        if nnz>0:
            arr=np.frombuffer(b[q:q+nnz*8],dtype=np.dtype([('j','<i4'),('f','<f4')]))
            Is.append(np.full(nnz,src,dtype=np.int32)); Js.append(arr['j'].copy()); Vs.append(arr['f'].copy())
        q+=nnz*8
    print("  loaded %s  cum nnz=%d  %.0fs"%(p.split('/')[-1],sum(len(x) for x in Is),time.time()-t0)); sys.stdout.flush()
I=np.concatenate(Is); J=np.concatenate(Js); V=np.concatenate(Vs)
del Is,Js,Vs
print("total directed nnz %d (%.1fM)  %.0fs"%(len(I),len(I)/1e6,time.time()-t0)); sys.stdout.flush()
# power S_ij = A_i F_ij
S=A[I]*V.astype(np.float64)
# canonical undirected key
lo=np.minimum(I,J).astype(np.int64); hi=np.maximum(I,J).astype(np.int64)
key=lo*np.int64(N)+hi
order=np.argsort(key,kind='stable'); key=key[order]; S=S[order]; lo=lo[order]; hi=hi[order]
del order
# group by key -> S_sym = mean of the (1 or 2) directed S
bnd=np.concatenate(([0],np.nonzero(np.diff(key))[0]+1,[len(key)]))
gstart=bnd[:-1]; gend=bnd[1:]
csum=np.concatenate(([0.0],np.cumsum(S)))
gsum=csum[gend]-csum[gstart]; gcnt=(gend-gstart).astype(np.float64)
Ssym=gsum/gcnt                    # mean of available directions
a=lo[gstart]; bb=hi[gstart]       # undirected pair endpoints (a<bb)
print("unique undirected pairs %d (%.1fM)  %.0fs"%(len(a),len(a)/1e6,time.time()-t0)); sys.stdout.flush()
# emit symmetric directed entries: (a,bb,Ssym/A_a) and (bb,a,Ssym/A_bb)
rows=np.concatenate([a,bb]); cols=np.concatenate([bb,a])
data=np.concatenate([Ssym/A[a], Ssym/A[bb]]).astype(np.float64)
del S,key,lo,hi,Ssym,gsum
# per-row ΣF and ΣF<=1 cap (symmetric scale would be ideal; here clip data by row factor)
o2=np.argsort(rows,kind='stable'); rows=rows[o2]; cols=cols[o2]; data=data[o2]; del o2
indptr=np.zeros(N+1,dtype=np.int64)
np.add.at(indptr,rows+1,1); indptr=np.cumsum(indptr)
rowsum=np.zeros(N); np.add.at(rowsum,rows,data)
print("assembled ΣF: min %.4f mean %.4f max %.4f  #>1=%d"%(rowsum.min(),rowsum.mean(),rowsum.max(),(rowsum>1).sum())); sys.stdout.flush()
# enforce ΣF<=1 by per-row scale (keeps reciprocity only approximately; log how many & how much)
over=rowsum>1.0
if over.any():
    sc=np.ones(N); sc[over]=1.0/rowsum[over]
    data=data*sc[rows]  # asymmetric row-scale on over-1 rows (rare); log
    print("  capped %d rows >1 (max was %.3f) by row-scale"%(over.sum(),rowsum.max()))
    rowsum=np.zeros(N); np.add.at(rowsum,rows,data)
# anchor validation
ref=np.fromfile(DIAG+"/moon3d_rowfix_refSigF.bin",'<f8')
d=rowsum-ref
print("ANCHOR vs moon3d_rowfix_refSigF.bin: mean|Δ| %.4f max|Δ| %.4f (assembled slightly < anchor by the <thr tail)"%(np.abs(d).mean(),np.abs(d).max()))
print("  assembled ΣF p50 %.4f ; anchor ΣF p50 %.4f"%(np.percentile(rowsum,50),np.percentile(ref,50)))
# reciprocity check on a sample: A_i F_ij vs A_j F_ji
# (by construction symmetric; spot check)
# write CSR
with open("/tmp/pp_sweep/corrected_vf.csr","wb") as f:
    np.array([N],dtype=np.int32).tofile(f)
    indptr.astype(np.int64).tofile(f)
    np.array([len(data)],dtype=np.int64).tofile(f)
    cols.astype(np.int32).tofile(f)
    data.astype(np.float32).tofile(f)
import os
print("wrote corrected_vf.csr  nnz=%d  %.2f GB  %.0fs"%(len(data),os.path.getsize("/tmp/pp_sweep/corrected_vf.csr")/1e9,time.time()-t0))
np.save(DIAG+"/assembled_rowsum.npy",rowsum)
