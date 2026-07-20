#!/usr/bin/env python3
"""EMA Chimaera thermal-inertia fitting tool (DEV PHASE 2, tool #2).

Recovers thermal inertia (TI) -- with visible albedo as a free-but-constrained
confounder -- from a NOISY IR diurnal temperature curve T(local-time), by fitting
against the 15-point Chimaera TI x albedo sweep grid (interpolated). STANDALONE:
imports ema_campaign_driver and ema_camera_noise READ-ONLY (their __main__ guards
mean no campaign/CLI code runs on import), so it does not race Rocky's driver.

OBSERVABLE (forward model AND observation, extracted the SAME way so definitions
cancel): surface T vs local solar time. Per facet, local time is set by
(facet_longitude - subsolar_longitude); subsolar longitude comes from the body-
frame Sun vector in body_positions.csv. Default epoch='ca' = the single timestep
nearest closest approach (the resolved flyby disk, spatial->local-time snapshot);
epoch='all' averages over the rotation window (denser, less flyby-realistic).

FIT: chi^2(TI, p) = sum_bins [(T_obs - M(TI,p))/sigma_T]^2 + [(p - p_vis)/sigma_p]^2
  - M(TI,p): bilinear interpolation of the grid curves in (log10 TI, p).
  - sigma_T per bin from the IR NEDT model (ema_camera_noise.ir_sigma_T), so cold,
    noisy bins are correctly downweighted.
  - p (geometric albedo) is FREE but penalized by a Gaussian prior (p_vis, sigma_p)
    derived from the CNAC visible image (disk-integrated reflectance).
  - Best fit = argmin on a fine (log TI, p) mesh; 1-sigma from the Delta chi^2 = 1
    profile of chi^2(TI) = min_p chi^2(TI, p).

CAVEAT (flagged in dev-1, carried here): the sweep varies SURFACE TI via ks while
holding kd=0.5 (deep TI ~ sqrt(kd*rhos*cp0) ~ 775 SI). For grid TI >~ few hundred,
the diurnal skin depth reaches the ks->kd transition (H=0.05 m), so the deep layer
caps the EFFECTIVE diurnal TI near ~775-1000. Interpret recovered TI at the high
end as an effective surface TI, not an unbounded bulk value.

Subcommands:
  selftest  -- build a synthetic analytic grid, inject noise, fit, verify recovery
               (unit-tests the interpolation + chi^2 + uncertainty machinery).
  extract   -- extract T(local-time) from one real thermal output dir (smoke test).
  fit       -- run the real campaign fit once Rocky's 15-point sweep + cameras exist.
"""
import os, sys, re, glob, json, argparse, importlib.util
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PTM = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm"
TROOT = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/thermal"

def _load(mod, path):
    s = importlib.util.spec_from_file_location(mod, path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m

# EMA-01 (RESOLVED 2026-07-09): `alb_point` and `disk_p_from_w` were Hapke
# disk-integrated photometry helpers that lived ONLY in the deleted scratch module
# /tmp/ema_sweep_driver.py, which left the `fit` path dead on import. Fermi's formula
# survived in a comment at ema_campaign_driver.py:45-48 and is now transcribed
# verbatim into the tracked, pure module scripts/ema_hapke.py (with its provenance
# and opposition-convention caveats). We import the recovered helpers from there.
#
# The tracked scripts/ema_campaign_driver.py carries an `if __name__ == "__main__":`
# guard, so it imports inertly and still supplies `thermal_done` + `HAPKE_DEFAULTS`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ema_campaign_driver as _driver
import ema_hapke


class _Drv:
    """Tracked driver symbols (thermal_done, HAPKE_DEFAULTS) plus the recovered
    Hapke photometry helpers from ema_hapke (EMA-01, resolved 2026-07-09)."""
    thermal_done   = staticmethod(_driver.thermal_done)
    HAPKE_DEFAULTS = _driver.HAPKE_DEFAULTS
    alb_point      = staticmethod(ema_hapke.alb_point)
    disk_p_from_w  = staticmethod(ema_hapke.disk_p_from_w)


_drv = _Drv()
_noise = _load("ema_noise", PTM+"/scripts/ema_camera_noise.py")


def _body_num_from_sweep_root(sweep_root):
    """Derive the body number ('623', '23871', ...) from a sweep-root path such as
    '.../thermal/623chimaera/sweep' -> '623', by matching the '<num><name>'
    directory component. Returns None if no such component is present. Mirrors the
    dynamic pattern at ema_ousha_extra_offgrid.py:148 and fixes EMA-02: the `fit`
    path must NOT silently reuse Chimaera's albedo for every body."""
    for part in reversed(os.path.normpath(sweep_root).split(os.sep)):
        m = re.match(r"(\d+)[a-z]", part)
        if m:
            return m.group(1)
    return None

# ================= real thermal-output extraction =================
def read_temp_bin(path):
    """PTM3D per-timestep bin: int32 count + float32*count."""
    raw = open(path, 'rb').read()
    n = int(np.frombuffer(raw[:4], np.int32)[0])
    return np.frombuffer(raw[4:4+4*n], np.float32).astype(float)

def parse_ply_facets(mesh_ply):
    """Return per-face (latitude, longitude) [deg] from the driver's exported
    binary PLY, in the same index order as the temperature array. The PLY stores
    'latitude'/'longitude' float properties on each face."""
    with open(mesh_ply, 'rb') as f:
        raw = f.read()
    hdr_end = raw.find(b"end_header\n") + len(b"end_header\n")
    header = raw[:hdr_end].decode('latin-1').splitlines()
    nv = nf = 0
    in_vertex = in_face = False; vprops = []
    for ln in header:
        t = ln.split()
        if not t: continue
        if t[0] == 'element' and t[1] == 'vertex': nv = int(t[2]); in_vertex, in_face = True, False
        elif t[0] == 'element' and t[1] == 'face':  nf = int(t[2]); in_vertex, in_face = False, True
        elif t[0] == 'property' and in_vertex: vprops.append(t[-1])
    # vertex block: nv rows of len(vprops) doubles (x,y,z,nx,ny,nz)
    off = hdr_end + nv * len(vprops) * 8
    lat = np.empty(nf); lon = np.empty(nf)
    p = off
    for i in range(nf):
        k = raw[p]; p += 1                     # uchar list length (=3)
        p += 4 * k                             # int32 vertex indices
        lat[i], lon[i] = np.frombuffer(raw[p:p+8], '<f4'); p += 8
    return lat, lon

def read_sun_vectors(csv):
    """Return subsolar longitude[deg] per timestep from body_positions.csv."""
    import csv as _csv
    lon = []
    with open(csv) as f:
        r = _csv.DictReader(f)
        for row in r:
            sx, sy = float(row['sun_x']), float(row['sun_y'])
            lon.append(np.degrees(np.arctan2(sy, sx)) % 360.0)
    return np.array(lon)

def ca_timestep(thermal_dir):
    """Index of the timestep with the most illuminated facets (~closest approach /
    peak dayside), from statistics or the direct_solar per-step series."""
    ds = sorted(glob.glob(thermal_dir+"/direct_solar/*.bin"))
    if not ds:
        return len(glob.glob(thermal_dir+"/temperature/*.bin")) // 2
    best, bi = -1, 0
    for i, fp in enumerate(ds):
        lit = int((read_temp_bin(fp) > 0).sum())
        if lit > best: best, bi = lit, i
    return bi

def extract_local_time_curve(thermal_dir, lat_band=25.0, nbins=24, epoch='ca'):
    """T vs local solar time (hours) diurnal curve. Returns dict with lst (bin
    centers), T (mean per bin), Tstd, count."""
    lat, lon = parse_ply_facets(thermal_dir+"/mesh.ply")
    sublon = read_sun_vectors(thermal_dir+"/body_positions.csv")
    tfiles = sorted(glob.glob(thermal_dir+"/temperature/*.bin"))
    band = np.abs(lat) <= lat_band
    edges = np.linspace(0, 24, nbins+1); cen = 0.5*(edges[:-1]+edges[1:])
    acc = np.zeros(nbins); acc2 = np.zeros(nbins); cnt = np.zeros(nbins)
    steps = [ca_timestep(thermal_dir)] if epoch == 'ca' else range(len(tfiles))
    for t in steps:
        T = read_temp_bin(tfiles[t])
        lst = ((lon - sublon[t])/15.0 + 12.0) % 24.0     # noon at subsolar meridian
        # IR sees emitted thermal from BOTH hemispheres, so keep day AND night side
        # facets in the band (do not mask on illumination); T>0 drops any missing.
        sel = band & (T > 0)
        idx = np.clip((lst[sel]/24.0*nbins).astype(int), 0, nbins-1)
        Tv = T[sel]
        np.add.at(acc, idx, Tv); np.add.at(acc2, idx, Tv*Tv); np.add.at(cnt, idx, 1.0)
    good = cnt > 0
    Tm = np.full(nbins, np.nan); Ts = np.full(nbins, np.nan)
    Tm[good] = acc[good]/cnt[good]
    Ts[good] = np.sqrt(np.maximum(0.0, acc2[good]/cnt[good] - Tm[good]**2))
    return dict(lst=cen, T=Tm, Tstd=Ts, count=cnt)

# ================= grid model =================
class ModelGrid:
    """TI x albedo grid of T(local-time) curves; bilinear interp in (log10 TI, p)."""
    def __init__(self, tis, albs, cube, lst):
        self.tis = np.asarray(tis, float); self.logti = np.log10(self.tis)
        self.albs = np.asarray(albs, float); self.cube = np.asarray(cube, float)  # [nti,nalb,nbins]
        self.lst = np.asarray(lst, float)
    def predict(self, ti, alb):
        lt = np.clip(np.log10(ti), self.logti[0], self.logti[-1])
        pa = np.clip(alb, self.albs[0], self.albs[-1])
        i = np.searchsorted(self.logti, lt) - 1; i = np.clip(i, 0, len(self.logti)-2)
        j = np.searchsorted(self.albs, pa) - 1; j = np.clip(j, 0, len(self.albs)-2)
        u = (lt - self.logti[i])/(self.logti[i+1]-self.logti[i])
        v = (pa - self.albs[j])/(self.albs[j+1]-self.albs[j])
        c = self.cube
        return ((1-u)*(1-v)*c[i,j] + u*(1-v)*c[i+1,j]
                + (1-u)*v*c[i,j+1] + u*v*c[i+1,j+1])

def load_sweep_grid(sweep_root, lat_band=25.0, nbins=24, epoch='ca'):
    """Build a ModelGrid from a real sweep dir (TROOT/<body>/sweep/ti*_alb*/)."""
    pts = []
    for d in sorted(glob.glob(sweep_root+"/ti*_alb*")):
        sp = os.path.join(d, "sweep_point.json")
        if not (os.path.exists(sp) and _drv.thermal_done(d)): continue
        meta = json.load(open(sp))
        cur = extract_local_time_curve(d, lat_band, nbins, epoch)
        pts.append((float(meta['ti']), float(meta['geom_albedo']), cur['T']))
    if not pts:
        raise SystemExit(f"no complete sweep points under {sweep_root} (Rocky's sweep not done yet?)")
    tis = sorted(set(round(p[0], 6) for p in pts)); albs = sorted(set(round(p[1], 6) for p in pts))
    if len(tis) < 2 or len(albs) < 2:
        raise SystemExit(f"incomplete grid under {sweep_root}: found {len(pts)} point(s) spanning "
                         f"{len(tis)} TI x {len(albs)} albedo; need >=2 x >=2 for interpolation "
                         f"(Rocky's 15-point sweep not done yet).")
    lst = extract_local_time_curve(glob.glob(sweep_root+"/ti*_alb*")[0], lat_band, nbins, epoch)['lst']
    cube = np.full((len(tis), len(albs), nbins), np.nan)
    for ti, alb, T in pts:
        cube[tis.index(round(ti,6)), albs.index(round(alb,6))] = T
    miss = [(tis[i], albs[j]) for i in range(len(tis)) for j in range(len(albs))
            if not np.isfinite(cube[i, j]).all()]
    if miss:
        raise SystemExit(f"grid under {sweep_root} has {len(miss)} missing/incomplete cell(s) "
                         f"(e.g. TI={miss[0][0]:g}, p={miss[0][1]:.4f}); wait for the full sweep.")
    # Guard: the TI axis MUST carry signal. If curves are (near-)identical across TI
    # at fixed albedo, the solver did not apply thermal inertia and TI is
    # unrecoverable -- refuse rather than return a meaningless unconstrained fit.
    # (Found 2026-07-03 on binary 35b052e: ks AND constant_thermal_inertia both had
    #  ZERO effect on EMA custom-mesh GPU thermal output -- a solver-level bug.)
    ti_spread = float(np.nanmax([np.nanmax(np.nanmax(cube[:, j], 0) - np.nanmin(cube[:, j], 0))
                                 for j in range(len(albs))]))
    if ti_spread < 1.0:
        raise SystemExit(
            f"DEGENERATE TI axis under {sweep_root}: T(local-time) varies by only "
            f"{ti_spread:.3f} K across TI={[f'{t:g}' for t in tis]} at fixed albedo. The "
            f"thermal solver is NOT applying thermal inertia -- TI is unrecoverable. "
            f"This is a SOLVER bug (verified: ks and constant_thermal_inertia both have "
            f"no effect on 35b052e EMA runs), not a driver/tool issue. Escalate before fitting.")
    return ModelGrid(tis, albs, cube, lst)

# ================= fit =================
def fit_ti(obs_T, sigma_T, grid, p_vis, sigma_p, nti=161, nalb=81):
    """Return best-fit dict minimizing chi^2 over a fine (log TI, p) mesh."""
    valid = np.isfinite(obs_T) & np.isfinite(sigma_T) & (sigma_T > 0)
    tis = np.logspace(np.log10(grid.tis[0]), np.log10(grid.tis[-1]), nti)
    albs = np.linspace(grid.albs[0], grid.albs[-1], nalb)
    C = np.empty((nti, nalb))
    for a, ti in enumerate(tis):
        for b, p in enumerate(albs):
            m = grid.predict(ti, p)
            r = (obs_T[valid] - m[valid])/sigma_T[valid]
            C[a, b] = np.sum(r*r) + ((p - p_vis)/sigma_p)**2
    ia, ib = np.unravel_index(np.argmin(C), C.shape)
    prof = C.min(axis=1)                                   # profile out albedo
    dchi = prof - prof.min()
    below = tis[dchi <= 1.0]
    ti_lo, ti_hi = (below.min(), below.max()) if below.size else (tis[ia], tis[ia])
    # albedo 1-sigma at best-fit TI
    ca = C[ia] - C[ia].min(); ab = albs[ca <= 1.0]
    alb_sig = 0.5*(ab.max()-ab.min()) if ab.size > 1 else (albs[1]-albs[0])
    return dict(ti=float(tis[ia]), ti_lo=float(ti_lo), ti_hi=float(ti_hi),
                alb=float(albs[ib]), alb_sig=float(alb_sig),
                chi2=float(C[ia, ib]), ndof=int(valid.sum()),
                tis=tis, prof=prof, valid=valid)

# ================= synthetic (selftest) =================
def _toy_curve(ti, alb, lst):
    """Analytic diurnal T(lst) with distinguishable TI (amplitude+lag) and albedo
    (mean level) signatures -- for unit-testing the fitter, not physics."""
    Tbar = 200.0*((1.0-1.3*alb)/(1.0-1.3*0.037))**0.25
    Amp  = 62.0/(1.0+(ti/100.0)**0.6)
    lag  = 0.6 + 1.6*(ti/100.0)**0.4
    skew = 8.0/(1.0+ti/100.0)
    th = 2*np.pi*(lst - 12.0 - lag)/24.0
    return Tbar + Amp*np.cos(th) + skew*np.cos(2*th)

def _toy_grid(tis, albs, nbins=24):
    lst = 0.5*(np.linspace(0,24,nbins+1)[:-1]+np.linspace(0,24,nbins+1)[1:])
    cube = np.array([[_toy_curve(ti, p, lst) for p in albs] for ti in tis])
    return ModelGrid(tis, albs, cube, lst)

def selftest(seed=1, outdir="/tmp/ema_ti_fit_selftest"):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    tis = [10.0, 32.0, 100.0, 316.0, 1000.0]
    # EMA-02 note: selftest is a SYNTHETIC machinery test (analytic toy grid, no real
    # sweep dir). There is no sweep_root / body number in scope here, so w is a fixed
    # Chimaera anchor (w_nom=0.143) BY CONSTRUCTION -- not the silent-default bug. The
    # per-body fix lives in the `fit` path, which derives the body from --sweep-root.
    albs = sorted(_drv.alb_point(0.143, s)[0] for s in (0.5, 1.0, 2.0))
    p_nom = _drv.disk_p_from_w(0.143)
    grid = _toy_grid(tis, albs)
    truths = [32.0, 316.0]
    results = []
    for ti_true in truths:
        clean = grid.predict(ti_true, p_nom)
        sigT = np.array([float(_noise.ir_sigma_T(T)) for T in clean])
        sigT = np.clip(sigT, 0.3, 40.0)                    # NEDT-model floor/ceiling
        obs = clean + rng.normal(0, 1, clean.size)*sigT
        # CNAC visible pins albedo tightly (disk-integrated) -> small sigma_p
        sigma_p = 0.03*p_nom
        p_vis = p_nom + rng.normal(0, sigma_p)
        r = fit_ti(obs, sigT, grid, p_vis, sigma_p)
        r['ti_true'] = ti_true; r['obs'] = obs; r['clean'] = clean; r['sigT'] = sigT
        # albedo pinning: fit with a WEAK vis prior for comparison
        r_weak = fit_ti(obs, sigT, grid, p_vis, 0.5*p_nom)
        r['alb_sig_weakvis'] = r_weak['alb_sig']
        results.append(r)
        print(f"  truth TI={ti_true:6.1f}  ->  TI_fit={r['ti']:6.1f} "
              f"[{r['ti_lo']:.0f},{r['ti_hi']:.0f}]  alb_fit={r['alb']:.4f}"
              f"+-{r['alb_sig']:.4f} (vis-constrained; weak-prior alb sigma "
              f"{r['alb_sig_weakvis']:.4f})  chi2/ndof={r['chi2']:.1f}/{r['ndof']}")
    _plot_curves(grid, results, p_nom, outdir+"/selftest_curves.png")
    _plot_recovery(results, outdir+"/selftest_recovery.png")
    print("  plots ->", outdir+"/selftest_curves.png ,", outdir+"/selftest_recovery.png")
    ok = all(abs(np.log10(r['ti'])-np.log10(r['ti_true'])) < np.log10(2.0) for r in results)
    print("  SELFTEST", "PASS" if ok else "FAIL", "(both truths recovered within a factor of 2)")
    return ok

def _plot_curves(grid, results, p_nom, path):
    fig, axs = plt.subplots(1, len(results), figsize=(6*len(results), 4.5))
    axs = np.atleast_1d(axs)
    for ax, r in zip(axs, results):
        ax.plot(grid.lst, r['clean'], 'k-', lw=2, label=f"truth TI={r['ti_true']:.0f}")
        ax.errorbar(grid.lst, r['obs'], yerr=r['sigT'], fmt='o', ms=3, color='tab:red',
                    alpha=0.6, label="noisy obs (NEDT)")
        ax.plot(grid.lst, grid.predict(r['ti'], r['alb']), '--', color='tab:blue', lw=2,
                label=f"best fit TI={r['ti']:.0f}")
        ax.set_xlabel("local solar time (h)"); ax.set_ylabel("T (K)")
        ax.set_title(f"Chimaera TI fit (synthetic), truth TI={r['ti_true']:.0f}"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()

def _plot_recovery(results, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    tt = [r['ti_true'] for r in results]; th = [r['ti'] for r in results]
    lo = [r['ti']-r['ti_lo'] for r in results]; hi = [r['ti_hi']-r['ti'] for r in results]
    ax.plot([8, 1200], [8, 1200], 'k:', lw=1, label="1:1")
    ax.errorbar(tt, th, yerr=[lo, hi], fmt='o', ms=8, color='tab:blue', capsize=4, label="recovered")
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(20, 500); ax.set_ylim(20, 500)
    ax.set_xlabel("true TI (SI)"); ax.set_ylabel("recovered TI (SI)")
    ax.set_title("TI recovery (Delta chi^2=1 errors)"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()

# ================= CLI =================
def main(argv=None):
    ap = argparse.ArgumentParser(description="EMA Chimaera thermal-inertia fit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest"); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", default="/tmp/ema_ti_fit_selftest")
    e = sub.add_parser("extract"); e.add_argument("--thermal-dir", required=True)
    e.add_argument("--lat-band", type=float, default=25.0); e.add_argument("--nbins", type=int, default=24)
    e.add_argument("--epoch", choices=["ca", "all"], default="ca"); e.add_argument("--out", default=None)
    f = sub.add_parser("fit")
    f.add_argument("--sweep-root", default=TROOT+"/623chimaera/sweep")
    f.add_argument("--truth-ti", type=float, nargs="+", default=[32.0, 316.0])
    f.add_argument("--vis-sigma-frac", type=float, default=0.03)
    f.add_argument("--lat-band", type=float, default=25.0); f.add_argument("--nbins", type=int, default=24)
    f.add_argument("--epoch", choices=["ca", "all"], default="ca")
    f.add_argument("--seed", type=int, default=1); f.add_argument("--out", default="/tmp/ema_ti_fit")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        ok = selftest(a.seed, a.out); sys.exit(0 if ok else 1)

    if a.cmd == "extract":
        cur = extract_local_time_curve(a.thermal_dir, a.lat_band, a.nbins, a.epoch)
        good = cur['count'] > 0
        print(f"local-time curve ({a.epoch}) from {a.thermal_dir}")
        for i in range(len(cur['lst'])):
            if good[i]:
                print(f"  LST {cur['lst'][i]:5.1f} h : T={cur['T'][i]:6.1f} K "
                      f"(std {cur['Tstd'][i]:5.1f}, n={int(cur['count'][i])})")
        if a.out:
            np.savetxt(a.out, np.column_stack([cur['lst'], cur['T'], cur['Tstd'], cur['count']]),
                       header="lst_h T_K Tstd_K count", fmt="%.4f")
            print("wrote", a.out)
        return

    if a.cmd == "fit":
        os.makedirs(a.out, exist_ok=True)
        grid = load_sweep_grid(a.sweep_root, a.lat_band, a.nbins, a.epoch)
        rng = np.random.default_rng(a.seed)
        # EMA-02: derive the body from --sweep-root; never default to Chimaera's albedo.
        num = _body_num_from_sweep_root(a.sweep_root)
        if num is None or num not in _drv.HAPKE_DEFAULTS:
            raise SystemExit(
                f"ema_ti_fit fit: cannot resolve a body number from --sweep-root "
                f"{a.sweep_root!r} (expected a '<num><name>' component such as "
                f"'623chimaera' with an entry in HAPKE_DEFAULTS). Refusing to guess "
                f"the albedo -- see BUGS.md EMA-02.")
        p_nom = _drv.disk_p_from_w(_drv.HAPKE_DEFAULTS[num]['w'])
        results = []
        for ti_true in a.truth_ti:
            clean = grid.predict(ti_true, p_nom)
            sigT = np.clip([float(_noise.ir_sigma_T(T)) for T in clean], 0.3, 40.0)
            obs = clean + rng.normal(0, 1, clean.size)*sigT
            sigma_p = a.vis_sigma_frac*p_nom
            p_vis = p_nom + rng.normal(0, sigma_p)
            r = fit_ti(obs, sigT, grid, p_vis, sigma_p)
            r.update(ti_true=ti_true, obs=obs, clean=clean, sigT=sigT)
            results.append(r)
            print(f"truth TI={ti_true:.0f} -> TI_fit={r['ti']:.0f} "
                  f"[{r['ti_lo']:.0f},{r['ti_hi']:.0f}] alb={r['alb']:.4f}+-{r['alb_sig']:.4f} "
                  f"chi2/ndof={r['chi2']:.1f}/{r['ndof']}")
        _plot_curves(grid, results, p_nom, a.out+"/fit_curves.png")
        _plot_recovery(results, a.out+"/fit_recovery.png")
        json.dump([{k: r[k] for k in ('ti_true','ti','ti_lo','ti_hi','alb','alb_sig','chi2','ndof')}
                   for r in results], open(a.out+"/fit_results.json", 'w'), indent=2)
        print("wrote", a.out+"/fit_results.json,", a.out+"/fit_curves.png,", a.out+"/fit_recovery.png")

if __name__ == "__main__":
    main()
