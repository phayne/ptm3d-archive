#!/usr/bin/env python3
"""EMA / MBR Explorer 6-target flyby campaign driver.
Per target: 40-orbit thermal sim -> ECAM-IR3S render (3 snapshots + dense flyby
sequence) -> thermal + reflected snapshot PNGs and flyby MP4s. Serial (one GPU),
resumable. Run detached; writes progress to campaign_manifest.json + campaign.log.

Products per target (in CROOT/<num><name>/flyby/):
  <name>_flyby_snapshots.png            thermal-IR bracket stills (brightness T)
  <name>_flyby.mp4                      thermal-IR flyby movie
  <name>_flyby_reflected_snapshots.png  reflected/visible bracket stills (RADIANCE HDU)
  <name>_flyby_reflected.mp4            reflected/visible flyby movie

Reflectance is Hapke: per-body single-scattering albedo w is calibrated to the
body's geometric albedo (opposition-excluded disk integral), so the RADIANCE
(reflected) HDU is now physically normalized — no longer the ~8x-dark Lommel-
Seeliger level nor the old fudge. Structural Hapke params (B0/h/g/theta_bar) are
Fermi's provisional class-based defaults (see HAPKE_DEFAULTS), overridable per
body via the body_config 'surface.hapke' block.
"""
import os, sys, re, json, glob, time, math, subprocess
import numpy as np, yaml
import spiceypy as sp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio

PTM   = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm"
TBIN  = PTM+"/build/bin/ptm_thermal_sim"
CBIN  = PTM+"/build/bin/ptm_camera_sim"
CSP   = "/Users/paha3326/software/cspice/data"
SCSPK = "/Users/paha3326/Research/asteroid/EMA/trajectory/ema_khamis_reftraj_280318_v2.bsp"
BODIES= PTM+"/data/horizons_cache/bodies"
TROOT = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/thermal"
CROOT = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/camera"
CAMYAML = PTM+"/examples/cameras/ema_ecam_ir3s_rockox.yaml"
MANIFEST = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/campaign_manifest.json"
STATUSDIR = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/status"
LOGDIR = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/logs"
LOG = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/campaign.log"
LAM=10.5e-6; EPS=0.95; HP=6.62607015e-34; CC=2.99792458e8; KB=1.380649e-23
# Rockox first (thermal already done -> validates render/MP4 fast), then chronological
TARGETS=[("13294","Rockox"),("10253","Westerwald"),("623","Chimaera"),
         ("88055","Ghaf"),("23871","Ousha"),("59980","Moza")]

# Per-body Hapke parameters (Fermi). w is the single-scattering albedo calibrated
# to reproduce each body's geometric albedo via the opposition-EXCLUDED disk
# integral p = w/8 + 0.49*r0 + 0.196*r0^2, r0 = (1-gamma)/(1+gamma),
# gamma = sqrt(1-w). B0/h/g/theta_bar are Fermi's class-based PROVISIONAL
# defaults (measured values pending a literature lookup) — override per body
# here or via the body_config 'surface.hapke' block (which wins if present).
# Keyed by (num, name) -> {w, B0, h, g, theta_bar}.
HAPKE_DEFAULTS={
    "10253":{"w":0.797,"B0":0.8,"h":0.06,"g":-0.27,"theta_bar":20.0},  # Westerwald p=0.313
    "13294":{"w":0.465,"B0":1.0,"h":0.05,"g":-0.30,"theta_bar":20.0},  # Rockox     p=0.139
    "59980":{"w":0.416,"B0":1.0,"h":0.05,"g":-0.30,"theta_bar":20.0},  # Moza       p=0.121
    "88055":{"w":0.204,"B0":1.5,"h":0.05,"g":-0.35,"theta_bar":20.0},  # Ghaf       p=0.054
    "23871":{"w":0.193,"B0":1.5,"h":0.05,"g":-0.35,"theta_bar":20.0},  # Ousha      p=0.051
    "623"  :{"w":0.143,"B0":1.5,"h":0.05,"g":-0.35,"theta_bar":20.0},  # Chimaera   p=0.037
}

def hapke_params(num,gb):
    """Resolve Hapke params for a body: body_config 'surface.hapke' overrides the
    class-based defaults; falls back to a mid dark-asteroid default if unknown."""
    hp=dict(HAPKE_DEFAULTS.get(num,{"w":0.20,"B0":1.5,"h":0.05,"g":-0.35,"theta_bar":20.0}))
    cfg_hp=(gb.get('surface',{}) or {}).get('hapke',{}) or {}
    for k in ("w","B0","h","g","theta_bar"):
        if k in cfg_hp: hp[k]=float(cfg_hp[k])
    return hp

# ---- albedo x thermal-inertia sweep (DEV PHASE 1) ----
# TI is set PHYSICALLY via surface conductivity ks: TI = sqrt(ks*rhos*cp0), holding
# rhos=1500 and cp0=800 (the nominal driver values). The solver consumes
# generic_body.surface.ks (config_parser.cpp:272); thermal_model.cpp:720 seeds the
# surface layer kc = ks, so the SURFACE thermal inertia is exactly sqrt(ks*rhos*cp0).
# (thermophysical.ks in the YAML is a dead/ignored key -- the parser only reads
#  ks_override there -- but it is kept in lockstep for a human-readable config.)
# Deep conductivity kd is held at the nominal 0.5 (the sweep varies SURFACE TI; the
# ks->kd depth transition over H=0.05 m is unchanged, matching the nominal model).
# Albedo: 3 points {0.5x,1x,2x} the body's nominal geometric albedo p, each mapped
# back to a Hapke single-scattering w (inverting the opposition-excluded disk
# integral), clipped physical. The thermal absorbed flux consumes
# thermophysical.albedo_hapke_w (config_parser.cpp:313; incidence_angle_albedo=true
# -> directional-hemispheric A_h(i)); generic_body.surface.albedo carries p.
TI_GRID=[10.0,32.0,100.0,316.0,1000.0]   # SI (J m^-2 K^-1 s^-1/2), 5 log-spaced
ALB_SCALES=[0.5,1.0,2.0]                  # multiples of the nominal geometric albedo p
TI_RHOS=1500.0; TI_CP0=800.0              # held constant (nominal values)
CNACYAML=PTM+"/examples/cameras/ema_cnac.yaml"  # CNAC / ECAM-C50 visible camera

def ti_to_ks(ti,rhos=TI_RHOS,cp0=TI_CP0):
    """Surface conductivity ks giving thermal inertia ti = sqrt(ks*rhos*cp0)."""
    return (ti*ti)/(rhos*cp0)

def disk_p_from_w(w):
    """Geometric albedo p from Hapke single-scattering albedo w via the opposition-
    EXCLUDED disk integral the driver uses to calibrate w to a body's known p:
    p = w/8 + 0.49*r0 + 0.196*r0^2, r0=(1-g)/(1+g), g=sqrt(1-w)."""
    g=math.sqrt(max(0.0,1.0-w)); r0=(1.0-g)/(1.0+g)
    return w/8.0 + 0.49*r0 + 0.196*r0*r0

def w_from_p(p):
    """Invert disk_p_from_w for w in (0,1) by bisection (p is monotonic in w)."""
    lo,hi=1e-6,1.0-1e-9
    p=min(max(p,disk_p_from_w(lo)),disk_p_from_w(hi))  # clip to achievable range
    for _ in range(80):
        mid=0.5*(lo+hi)
        if disk_p_from_w(mid)<p: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)

def alb_point(w_nom,scale):
    """One albedo sweep point: scale the nominal geometric albedo p_nom by `scale`,
    map back to w, clip physical (w in (0,1)). Returns (p_used, w_used) where p_used
    is recomputed from the clipped w so the logged (p,w) pair is self-consistent."""
    p=max(1e-4,scale*disk_p_from_w(w_nom))
    w=min(max(w_from_p(p),1e-6),1.0-1e-9)
    return disk_p_from_w(w),w

def build_thermal_config(*, body_name, bcfg, bdir, naif, ra, dec, radius, period,
                         use_mesh, shape_mesh_file, shape_mesh_scale, gb_orbit,
                         start_time, duration_days, outdir, ks, geom_albedo, hapke_w):
    """Single source of truth for the thermal-sim YAML, shared by the nominal flyby
    path and the albedo x TI sweep. A sweep point differs from nominal ONLY in
    (ks, geom_albedo, hapke_w, outdir). Load-bearing physics keys the solver reads:
    generic_body.surface.ks (surface conductivity -> TI) and thermophysical.
    albedo_hapke_w (absorbed-flux albedo). radiation OFF; SPICE ephemeris; 292 steps,
    40 equilibration orbits (matches the nominal EMA config exactly at ks=0.01)."""
    return {'simulation':{'start_time':start_time,'duration_days':duration_days,
                'timesteps':292,'body':body_name},
            'mesh':{'resolution':6,'import_file':(shape_mesh_file if use_mesh else ''),
                    'scale_factor':(shape_mesh_scale if use_mesh else 1.0),'export_basic':False},
            'thermal':{'run_thermal_mesh':True,'threads':0,'equilibration_orbits':40,'use_gpu':True,
                'compute_statistics':True,'integration_days':-1.0,'integration_steps':-1,
                'integration_offset_days':0.0,'integration_offset_steps':0,'geothermal_heat_flow':-1.0,
                'enable_shadow_flux':False,'crater_dd_ratio':8.0,'thermal_inertia_map':''},
            'thermophysical':{'use_default_albedo':False,'albedo':geom_albedo,'use_default_emissivity':False,
                'emissivity':0.95,'use_default_ks':False,'ks':ks,'use_default_kd':False,'kd':0.5,
                'use_default_H':False,'H':0.05,'use_default_rhos':False,'rhos':1500.0,'use_default_rhod':False,
                'rhod':2000.0,'use_default_cp0':False,'cp0':800.0,'use_default_Qb':False,'Qb':0.0,
                'use_default_chi':False,'chi':0.0,'incidence_angle_albedo':True,
                'albedo_hapke_w':hapke_w},
            'radiation':{'enabled':False},
            'illumination':{'include_primary_reflection':False,'use_batch_processing':True},
            'output':{'directory':outdir,'format':'binary','vtk_mode':'none','series_output':'per_timestep'},
            'advanced':{'verbose':False,'default_visible_albedo':-1.0},
            'generic_body':{'name':body_name,'config_file':bcfg,'ephemeris_source':'spice',
                'spice_metakernel':bdir+'/metakernel.tm','naif_id':naif,
                'shape':{'type':('custom_mesh' if use_mesh else 'spherical'),'radius_km':radius,
                         'mesh_file':(shape_mesh_file if use_mesh else ''),
                         'mesh_scale':(shape_mesh_scale if use_mesh else 1.0),
                         'semi_axes_km':[0.001,0.001,0.001]},
                'orbit':gb_orbit,
                'spin':{'period_hours':period,'pole_ra_deg':ra,'pole_dec_deg':dec,'prime_meridian_deg':0.0},
                'surface':{'albedo':geom_albedo,'emissivity':0.95,'ks':ks,'kd':0.5,'rhos':1500.0,'rhod':2000.0,
                           'H':0.05,'cp0':800.0,'Qb':0.0,'chi':0.0}}}

def log(m):
    s=f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s,flush=True)
    open(LOG,"a").write(s+"\n")

def invplanck(L):
    a=2*HP*CC**2/LAM**5
    return HP*CC/(LAM*KB*np.log(a/(L*1e6)+1.0))

def spk_body_id(path):
    h=sp.dafopr(path); nd,ni,_,_,_,_=sp.dafrfr(h); sp.dafbfs(h); ids=set()
    while sp.daffna():
        s=sp.dafgs(); dc,ic=sp.dafus(s,nd,ni); ids.add(int(ic[0]))
    sp.dafcls(h); return [b for b in sorted(ids) if b>2000000]

def ecl_pole_to_radec(lon,lat):  # ecliptic lon/lat (deg) -> equatorial RA/Dec
    e=math.radians(23.43928); lo=math.radians(lon); la=math.radians(lat)
    x=math.cos(la)*math.cos(lo); y=math.cos(la)*math.sin(lo); z=math.sin(la)
    yr=y*math.cos(e)-z*math.sin(e); zr=y*math.sin(e)+z*math.cos(e)
    return math.degrees(math.atan2(yr,x))%360, math.degrees(math.asin(zr))

def find_ca(AST):
    et0=sp.str2et('2028-03-19'); et1=sp.str2et('2035-07-11'); dmin=1e30; etm=et0
    for et in np.arange(et0,et1,86400.0):
        try: d=np.linalg.norm(sp.spkpos("-60000",et,'J2000','NONE',AST)[0])
        except: continue
        if d<dmin: dmin=d; etm=et
    for et in np.arange(etm-2*86400,etm+2*86400,10.0):
        try: d=np.linalg.norm(sp.spkpos("-60000",et,'J2000','NONE',AST)[0])
        except: continue
        if d<dmin: dmin=d; etm=et
    scv=sp.spkpos("-60000",etm,'J2000','NONE',AST)[0]
    vrel=np.linalg.norm(sp.spkezr("-60000",etm,'J2000','NONE',AST)[0][3:6])
    return etm,dmin,vrel

def write_spin_pck(path,naif,frame,ra,dec,period_h,radius):
    pm=360.0*24.0/period_h
    open(path,"w").write(f"""KPL/PCK
\\begindata
  FRAME_{frame}        = {naif}
  FRAME_{naif}_NAME    = '{frame}'
  FRAME_{naif}_CLASS   = 2
  FRAME_{naif}_CLASS_ID= {naif}
  FRAME_{naif}_CENTER  = {naif}
  OBJECT_{naif}_FRAME  = '{frame}'
  BODY{naif}_POLE_RA   = ( {ra:.4f}   0.0   0.0 )
  BODY{naif}_POLE_DEC  = ( {dec:.4f}  0.0   0.0 )
  BODY{naif}_PM        = ( 0.0   {pm:.6f}   0.0 )
  BODY{naif}_RADII     = ( {radius} {radius} {radius} )
\\begintext
""")

def cont(path,n=70):
    ch=[path[i:i+n] for i in range(0,len(path),n)]
    return "\n    ".join("'"+c+("+'" if i<len(ch)-1 else "'") for i,c in enumerate(ch))

def write_cam_metakernel(path,bdir,spin_name):
    open(path,"w").write(f"""KPL/MK
\\begindata
  PATH_VALUES = (
    '{CSP}'
    {cont(bdir)}
  )
  PATH_SYMBOLS = ( 'KERNELS', 'BODY' )
  KERNELS_TO_LOAD = (
    '$KERNELS/lsk/naif0012.tls'
    '$KERNELS/pck/pck00011.tpc'
    '$KERNELS/spk/de440s.bsp'
    '$BODY/ephemeris.spk'
    '$BODY/{spin_name}'
  )
\\begintext
""")

def run(cmd,logf):
    with open(logf,"w") as f:
        return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode

def thermal_done(outdir):
    return (len(glob.glob(outdir+"/temperature/*.bin"))>=200 and
            os.path.exists(outdir+"/statistics/temperature_avg.bin"))

def render(ptm_output, times, outdir, naif, frame, mk, hp, camyaml=CAMYAML, render_mode="both"):
    # Reflectance is now Hapke: single-scattering albedo hp['w'] is calibrated so
    # the disk-integrated geometric albedo matches the body's known value; B0/h/g/
    # theta_bar carry the phase dependence away from opposition. The C++ seeds the
    # uniform vertex albedo from hapke_w so the calibrated model is rendered as-is
    # (no separate geometric-albedo scalar rescale). Up-vector uses the boresight-
    # projected north pole (keeps north in the image plane).
    #   camyaml/render_mode default to the ECAM-IR3S "both" pass; the CNAC visible
    # pass reuses this same function with camyaml=CNACYAML, render_mode="reflected"
    # (single-band grayscale -> the RADIANCE HDU), consuming the SAME thermal output.
    os.makedirs(outdir,exist_ok=True)
    cmd=[CBIN,"--ptm-output",ptm_output,
         "--camera-yaml",camyaml,"--meta-kernel",mk,"--trajectory",SCSPK,
         "--traj-format","spk","--traj-spacecraft","-60000","--target-body",naif,
         "--target-frame",frame,"--render-mode",render_mode,"--thermal-wavelength","10.5",
         "--emissivity","0.95","--photometric","hapke",
         "--hapke",str(hp['w']),str(hp['B0']),str(hp['h']),str(hp['g']),str(hp['theta_bar']),
         "--up-boresight-pole","--supersample","2","--fits","--pgm","--output-dir",outdir]
    for t in times: cmd+=["--time",t]
    return run(cmd, outdir+"/render.log")

def fits_tb(fp):
    from astropy.io import fits
    th=np.asarray(fits.open(fp)['THERMAL'].data,float)
    tb=np.zeros(th.shape); on=th>0; tb[on]=invplanck(th[on]/EPS); return tb,on

def fits_reflected(fp):
    # Reflected/visible-light radiance lives in the RADIANCE HDU (W/m^2/sr).
    # Hapke-normalized (per-body w calibrated to geometric albedo). Rendered linearly.
    from astropy.io import fits
    r=np.asarray(fits.open(fp)['RADIANCE'].data,float)
    on=r>0
    return r,on

def snapshot_png(camdir,out_png,title):
    from astropy.io import fits
    tbs=[fits_tb(camdir+f"/frame_000{i}.fits")[0] for i in range(3)]
    vmax=max(tb[tb>0].max() for tb in tbs)+3
    fig,axs=plt.subplots(1,3,figsize=(15,5.2),facecolor='black')
    labs=["CA − 2 min","CA (closest approach)","CA + 2 min"]
    for k,(tb,lab) in enumerate(zip(tbs,labs)):
        ys,xs=np.where(tb>0)
        if len(xs):
            x0,x1=max(0,xs.min()-14),xs.max()+14; y0,y1=max(0,ys.min()-14),ys.max()+14; sub=tb[y0:y1,x0:x1]
        else: sub=tb
        im=axs[k].imshow(sub,origin='lower',cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
        axs[k].set_title(lab,color='white'); axs[k].set_xticks([]); axs[k].set_yticks([])
        cb=plt.colorbar(im,ax=axs[k],fraction=0.046,pad=0.04); cb.set_label('Brightness T (K) @ 10.5 µm',color='white')
        cb.ax.yaxis.set_tick_params(color='white'); plt.setp(plt.getp(cb.ax,'yticklabels'),color='white')
    fig.suptitle(title,color='white',fontsize=12); plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_png,dpi=130,facecolor='black'); plt.close()

def snapshot_png_reflected(camdir,out_png,title):
    # Reflected/visible-light bracket stills from the RADIANCE HDU (Hapke-
    # normalized). Independent of the thermal stills; does not read/overwrite them.
    rads=[fits_reflected(camdir+f"/frame_000{i}.fits")[0] for i in range(3)]
    pos=np.concatenate([r[r>0] for r in rads]) if any((r>0).any() for r in rads) else np.array([0.0])
    vmax=float(np.percentile(pos,99.5)) if pos.size and pos.max()>0 else 1.0
    if vmax<=0: vmax=1.0
    fig,axs=plt.subplots(1,3,figsize=(15,5.2),facecolor='black')
    labs=["CA − 2 min","CA (closest approach)","CA + 2 min"]
    for k,(r,lab) in enumerate(zip(rads,labs)):
        ys,xs=np.where(r>0)
        if len(xs):
            x0,x1=max(0,xs.min()-14),xs.max()+14; y0,y1=max(0,ys.min()-14),ys.max()+14; sub=r[y0:y1,x0:x1]
        else: sub=r
        im=axs[k].imshow(sub,origin='lower',cmap='gray',vmin=0,vmax=vmax,interpolation='nearest')
        axs[k].set_title(lab,color='white'); axs[k].set_xticks([]); axs[k].set_yticks([])
        cb=plt.colorbar(im,ax=axs[k],fraction=0.046,pad=0.04); cb.set_label('Reflected radiance (W m⁻² sr⁻¹)',color='white')
        cb.ax.yaxis.set_tick_params(color='white'); plt.setp(plt.getp(cb.ax,'yticklabels'),color='white')
    fig.suptitle(title,color='white',fontsize=12); plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_png,dpi=130,facecolor='black'); plt.close()

def make_mp4(animdir,out_mp4,title,fps=12):
    from astropy.io import fits
    fitsf=sorted(glob.glob(animdir+"/frame_*.fits"))
    tbs=[]; rngs=[]
    for fp in fitsf:
        d=fits.open(fp); th=np.asarray(d['THERMAL'].data,float); tb=np.zeros(th.shape); on=th>0
        tb[on]=invplanck(th[on]/EPS); tbs.append(tb)
        dk=np.asarray(d['DIST_KM'].data,float); rngs.append(float(dk[on].min()) if on.any() else float('nan'))
    vmax=max((tb[tb>0].max() if (tb>0).any() else 0) for tb in tbs)+3
    pngdir=animdir+"/png"; os.makedirs(pngdir,exist_ok=True); pngs=[]
    for i,(tb,rng) in enumerate(zip(tbs,rngs)):
        fig,ax=plt.subplots(figsize=(6.4,4.8),facecolor='black')
        ax.imshow(tb,origin='lower',cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03,0.95,title,color='white',fontsize=8,transform=ax.transAxes,va='top')
        ax.text(0.03,0.05,f"range {rng:,.0f} km" if rng==rng else "",color='white',fontsize=8,transform=ax.transAxes)
        p=pngdir+f"/f{i:04d}.png"; plt.savefig(p,dpi=100,facecolor='black'); plt.close(); pngs.append(p)
    with imageio.get_writer(out_mp4,fps=fps,codec='libx264',quality=8,macro_block_size=16) as w:
        for p in pngs: w.append_data(imageio.imread(p))

def make_mp4_reflected(animdir,out_mp4,title,fps=12):
    # Reflected/visible-light flyby movie from the RADIANCE HDU. Writes its own
    # PNG cache (png_reflected/) so the thermal MP4's frames are untouched.
    from astropy.io import fits
    fitsf=sorted(glob.glob(animdir+"/frame_*.fits"))
    rads=[]; rngs=[]
    for fp in fitsf:
        d=fits.open(fp); r=np.asarray(d['RADIANCE'].data,float); rads.append(r)
        on=r>0; dk=np.asarray(d['DIST_KM'].data,float)
        rngs.append(float(dk[on].min()) if on.any() else float('nan'))
    pos=np.concatenate([r[r>0] for r in rads]) if any((r>0).any() for r in rads) else np.array([0.0])
    vmax=float(np.percentile(pos,99.5)) if pos.size and pos.max()>0 else 1.0
    if vmax<=0: vmax=1.0
    pngdir=animdir+"/png_reflected"; os.makedirs(pngdir,exist_ok=True); pngs=[]
    for i,(r,rng) in enumerate(zip(rads,rngs)):
        fig,ax=plt.subplots(figsize=(6.4,4.8),facecolor='black')
        ax.imshow(r,origin='lower',cmap='gray',vmin=0,vmax=vmax,interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03,0.95,title,color='white',fontsize=8,transform=ax.transAxes,va='top')
        ax.text(0.03,0.05,f"range {rng:,.0f} km" if rng==rng else "",color='white',fontsize=8,transform=ax.transAxes)
        p=pngdir+f"/f{i:04d}.png"; plt.savefig(p,dpi=100,facecolor='black'); plt.close(); pngs.append(p)
    with imageio.get_writer(out_mp4,fps=fps,codec='libx264',quality=8,macro_block_size=16) as w:
        for p in pngs: w.append_data(imageio.imread(p))

# ---- per-target resolution (shared by nominal flyby + sweep) ----
def resolve_target(num,name):
    """Resolve per-target SPICE + shape + pole + closest-approach state used by BOTH
    the nominal flyby path and the sweep. Loads kernels (leaves them furnished);
    returns None if the body has no horizons cache dir. Mirrors the nominal setup so
    the two paths never diverge in geometry."""
    sp.kclear()
    cand=[d for d in os.listdir(BODIES) if d.startswith(num+"_")]
    if not cand: log(f"{name}: NO CACHE DIR -> skip"); return None
    bdir=os.path.join(BODIES,cand[0]); spkf=bdir+"/ephemeris.spk"; bcfg=bdir+"/body_config.yaml"
    cfg=yaml.safe_load(open(bcfg)); gb=cfg.get('generic_body',cfg)
    for k in [CSP+"/lsk/naif0012.tls",CSP+"/spk/de440s.bsp",spkf,SCSPK]: sp.furnsh(k)
    naif=str(spk_body_id(spkf)[0])
    radius=float(gb['shape']['radius_km']); period=float(gb['spin']['period_hours'])
    albedo=float(gb['surface']['albedo'])
    # Shape selection: honor whatever the body_config declares. A body may ship a
    # real shape model (type: custom_mesh + mesh_file); fall back to the spherical
    # approximation (with a warning emitted by the caller) when none is present.
    shape_type=str(gb['shape'].get('type','spherical')).strip().lower()
    shape_mesh_file=str(gb['shape'].get('mesh_file','') or '')
    shape_mesh_scale=float(gb['shape'].get('mesh_scale',1.0))
    use_mesh=bool(shape_type=='custom_mesh' and shape_mesh_file and os.path.exists(shape_mesh_file))
    ra=gb['spin'].get('pole_ra_deg'); dec=gb['spin'].get('pole_dec_deg'); pole_assumed=False
    if ra is None or dec is None:  # no measured pole -> orbital pole (obliquity 0)
        o=gb['orbit']; i=math.radians(o['inclination_deg']); Om=math.radians(o['lon_ascending_node_deg'])
        nx=math.sin(i)*math.sin(Om); ny=-math.sin(i)*math.cos(Om); nz=math.cos(i)
        lon=math.degrees(math.atan2(ny,nx))%360; lat=math.degrees(math.asin(nz))
        ra,dec=ecl_pole_to_radec(lon,lat); pole_assumed=True
    ra=float(ra); dec=float(dec)
    etm,dmin,vrel=find_ca(naif); ca_utc=sp.et2utc(etm,'ISOC',0)
    hp=hapke_params(num,gb)
    return dict(bdir=bdir,bcfg=bcfg,gb=gb,naif=naif,radius=radius,period=period,albedo=albedo,
        use_mesh=use_mesh,shape_type=shape_type,shape_mesh_file=shape_mesh_file,
        shape_mesh_scale=shape_mesh_scale,ra=ra,dec=dec,pole_assumed=pole_assumed,
        etm=etm,dmin=dmin,vrel=vrel,ca_utc=ca_utc,hp=hp)

# ---- albedo x thermal-inertia sweep runner (DEV PHASE 1) ----
def run_sweep(targets,ti_sel,alb_sel):
    """Run the TI x albedo grid per target (kept GENERAL; the campaign runs it for
    Chimaera only). Idempotent: a point whose output dir is already complete
    (thermal_done) is skipped. Each point writes its own config.yaml + sweep_point.json
    (auditable k_s<->TI + (p,w)); a per-target sweep_manifest.json records the whole
    grid. Non-colliding: TROOT/<num><name>/sweep/ti<TI>_alb<p>/, leaving flyby/
    untouched. Binary=35b052e, radiation OFF, SPICE ephemeris (all as nominal)."""
    log("SWEEP MODE targets=%s TI=%s alb_scales=%s"%([t[1] for t in targets],
        [f"{t:g}" for t in ti_sel],alb_sel))
    log("k_s<->TI map (rhos=%.0f cp0=%.0f): "%(TI_RHOS,TI_CP0)+
        ", ".join(f"TI={t:g}->ks={ti_to_ks(t):.6g}" for t in TI_GRID))
    for num,name in targets:
        try:
            R=resolve_target(num,name)
            if R is None: continue
            log(f"{name}: NAIF {R['naif']} CA {R['ca_utc']} range {R['dmin']:.0f}km "
                f"vrel {R['vrel']:.1f} period {R['period']:.3f}h "
                f"{'mesh' if R['use_mesh'] else 'sphere'} pole({R['ra']:.1f},{R['dec']:.1f})"
                f"{' ASSUMED' if R['pole_assumed'] else ''}")
            win_h=1.5*R['period']; start_et=R['etm']-0.75*R['period']*3600.0
            start_time=sp.et2utc(start_et,'ISOC',0).replace('Z',''); duration_days=win_h/24.0
            sweep_root=f"{TROOT}/{num}{name.lower()}/sweep"; os.makedirs(sweep_root,exist_ok=True)
            grid=[]
            for ti in ti_sel:
                ks=ti_to_ks(ti)
                for scale in alb_sel:
                    p,w=alb_point(R['hp']['w'],scale)
                    outdir=f"{sweep_root}/ti{ti:g}_alb{p:.4f}"
                    grid.append(dict(ti=ti,ks=ks,alb_scale=scale,geom_albedo=round(p,6),
                                     hapke_w=round(w,6),outdir=outdir))
                    if thermal_done(outdir):
                        log(f"{name}: [TI={ti:g} ks={ks:.5g} p={p:.4f} w={w:.4f}] complete -> skip")
                        continue
                    os.makedirs(outdir,exist_ok=True)
                    tconf=build_thermal_config(body_name=R['gb'].get('name',name),bcfg=R['bcfg'],
                        bdir=R['bdir'],naif=R['naif'],ra=R['ra'],dec=R['dec'],radius=R['radius'],
                        period=R['period'],use_mesh=R['use_mesh'],shape_mesh_file=R['shape_mesh_file'],
                        shape_mesh_scale=R['shape_mesh_scale'],gb_orbit=R['gb']['orbit'],
                        start_time=start_time,duration_days=duration_days,outdir=outdir,
                        ks=ks,geom_albedo=p,hapke_w=w)
                    cfgp=outdir+"/config.yaml"; yaml.safe_dump(tconf,open(cfgp,'w'),sort_keys=False)
                    json.dump(dict(target=f"{num}{name}",naif=R['naif'],ti=ti,ks=ks,alb_scale=scale,
                        geom_albedo=p,hapke_w=w,rhos=TI_RHOS,cp0=TI_CP0,binary="35b052e",config=cfgp),
                        open(outdir+"/sweep_point.json",'w'),indent=2)
                    log(f"{name}: run TI={ti:g} ks={ks:.5g} p={p:.4f} w={w:.4f} -> {outdir}")
                    rc=run([TBIN,"--config",cfgp],f"{LOGDIR}/ema_{num}_sweep_ti{ti:g}_alb{p:.4f}.log")
                    ok=thermal_done(outdir)
                    log(f"{name}: TI={ti:g} p={p:.4f} rc={rc} done={ok}")
            json.dump(dict(target=f"{num}{name}",naif=R['naif'],rhos=TI_RHOS,cp0=TI_CP0,
                nominal_w=R['hp']['w'],nominal_p=disk_p_from_w(R['hp']['w']),
                ti_grid=TI_GRID,alb_scales=ALB_SCALES,points=grid),
                open(sweep_root+"/sweep_manifest.json",'w'),indent=2)
            log(f"{name}: SWEEP grid -> {sweep_root}/sweep_manifest.json ({len(grid)} points)")
        except Exception as e:
            import traceback; log(f"{name}: SWEEP ERROR {e}\n{traceback.format_exc()}")
    log("SWEEP DRIVER FINISHED")

def parse_args(argv):
    """Nominal (default): bare positional target names (back-compatible). Sweep:
    --sweep [--target NAME]... [--ti T]... [--alb-scale S]... (bare names also select
    targets in sweep mode). Returns (sweep, names, ti_sel, alb_sel)."""
    sweep=False; tgt=[]; ti_sel=[]; alb_sel=[]; i=0
    while i<len(argv):
        a=argv[i]
        if a=='--sweep': sweep=True
        elif a=='--target' and i+1<len(argv): tgt.append(argv[i+1]); i+=1
        elif a=='--ti' and i+1<len(argv): ti_sel.append(float(argv[i+1])); i+=1
        elif a=='--alb-scale' and i+1<len(argv): alb_sel.append(float(argv[i+1])); i+=1
        elif not a.startswith('--'): tgt.append(a)
        i+=1
    return sweep,tgt,ti_sel,alb_sel

# ---- main ----
def main(argv):
    os.makedirs(STATUSDIR,exist_ok=True)
    os.makedirs(LOGDIR,exist_ok=True)
    sweep,names,ti_sel,alb_sel=parse_args(argv)
    tgts=[t for t in TARGETS if (not names or t[1] in names)]
    if sweep:
        run_sweep(tgts, ti_sel or TI_GRID, alb_sel or ALB_SCALES)
        return
    man=json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    for num,name in tgts:
        try:
            R=resolve_target(num,name)
            if R is None: continue
            bdir=R['bdir']; bcfg=R['bcfg']; gb=R['gb']; naif=R['naif']
            radius=R['radius']; period=R['period']; albedo=R['albedo']
            use_mesh=R['use_mesh']; shape_mesh_file=R['shape_mesh_file']
            shape_mesh_scale=R['shape_mesh_scale']; ra=R['ra']; dec=R['dec']
            pole_assumed=R['pole_assumed']; etm=R['etm']; dmin=R['dmin']; vrel=R['vrel']
            ca_utc=R['ca_utc']; hp=R['hp']
            if R['shape_type']=='custom_mesh' and not use_mesh:
                log(f"{name}: WARNING custom_mesh declared but mesh_file "
                    f"'{shape_mesh_file}' missing -> using spherical approximation")
            log(f"{name}: NAIF {naif} CA {ca_utc} range {dmin:.0f}km vrel {vrel:.1f} "
                f"period {period:.3f}h {'mesh' if use_mesh else 'sphere'} "
                f"pole({ra:.1f},{dec:.1f}){' ASSUMED' if pole_assumed else ''}")
            THERMDIR=f"{TROOT}/{num}{name.lower()}/flyby"; os.makedirs(THERMDIR,exist_ok=True)
            # thermal config: 1.5-rotation window centered on CA, 40-orbit eq
            win_h=1.5*period; start_et=etm-0.75*period*3600.0
            tconf=build_thermal_config(body_name=gb.get('name',name),bcfg=bcfg,bdir=bdir,naif=naif,
                ra=ra,dec=dec,radius=radius,period=period,use_mesh=use_mesh,
                shape_mesh_file=shape_mesh_file,shape_mesh_scale=shape_mesh_scale,gb_orbit=gb['orbit'],
                start_time=sp.et2utc(start_et,'ISOC',0).replace('Z',''),duration_days=win_h/24.0,
                outdir=THERMDIR,ks=0.01,geom_albedo=albedo,hapke_w=hp['w'])
            tcfg_path=PTM+f"/examples/ema_{num}{name.lower()}_flyby.yaml"
            yaml.safe_dump(tconf,open(tcfg_path,'w'),sort_keys=False)

            if thermal_done(THERMDIR):
                log(f"{name}: thermal already complete -> skip run")
            else:
                log(f"{name}: thermal run starting ...")
                rc=run([TBIN,"--config",tcfg_path], f"{LOGDIR}/ema_{num}_thermal.log")
                ok=os.path.exists(THERMDIR+"/statistics/temperature_avg.bin")
                log(f"{name}: thermal rc={rc} done={ok}")
                if not ok: man[name]={'status':'THERMAL_FAILED','ca':ca_utc}; json.dump(man[name],open(STATUSDIR+f"/{name}.json",'w'),indent=2); continue

            # spin PCK + camera metakernel
            frame=f"IAU_{name.upper()}_{num}"; spin_name=f"{name.lower()}_spin.tpc"
            write_spin_pck(bdir+"/"+spin_name,naif,frame,ra,dec,period,radius)
            cam_mk=bdir+"/metakernel_camera.tm"; write_cam_metakernel(cam_mk,bdir,spin_name)

            camdir=f"{CROOT}/{num}{name.lower()}/flyby"; animdir=camdir+"/anim"
            os.makedirs(camdir,exist_ok=True)
            # hp was resolved above (before the thermal config) so thermal + camera share w.
            log(f"{name}: Hapke w={hp['w']} B0={hp['B0']} h={hp['h']} g={hp['g']} "
                f"theta_bar={hp['theta_bar']} (geom-albedo target {albedo})")
            # 3 snapshots at CA-2/CA/CA+2 min
            snaps=[sp.et2utc(etm+dt,'ISOC',0) for dt in (-120.0,0.0,120.0)]
            render(THERMDIR,snaps,camdir,naif,frame,cam_mk,hp)
            title=f"({num}) {name} - EMA ECAM-IR3S VNFOV thermal-IR | CA {ca_utc} {dmin:.0f} km"
            rtitle=f"({num}) {name} - EMA ECAM-IR3S VNFOV reflected | CA {ca_utc} {dmin:.0f} km"
            snapshot_png(camdir,camdir+f"/{name.lower()}_flyby_snapshots.png",title)
            snapshot_png_reflected(camdir,camdir+f"/{name.lower()}_flyby_reflected_snapshots.png",rtitle)
            # dense flyby sequence (consistent ~6x zoom), 49 frames
            half=min(900.0, 5.9*dmin/vrel); times=[sp.et2utc(etm+t,'ISOC',0) for t in np.linspace(-half,half,49)]
            render(THERMDIR,times,animdir,naif,frame,cam_mk,hp)
            make_mp4(animdir,camdir+f"/{name.lower()}_flyby.mp4",title)
            make_mp4_reflected(animdir,camdir+f"/{name.lower()}_flyby_reflected.mp4",rtitle)

            # --- CNAC visible pass (single-band grayscale reflectance) ---
            # Consumes the SAME thermal per-timestep output (THERMDIR) as the IR pass;
            # render_mode=reflected -> the RADIANCE HDU carries the grayscale image.
            # Output parallels the IR camera set in flyby/visible/ (non-colliding).
            visdir=camdir+"/visible"; visanim=visdir+"/anim"; os.makedirs(visdir,exist_ok=True)
            vtitle=f"({num}) {name} - EMA CNAC (ECAM-C50) visible/grayscale | CA {ca_utc} {dmin:.0f} km"
            log(f"{name}: CNAC visible pass -> {visdir}")
            render(THERMDIR,snaps,visdir,naif,frame,cam_mk,hp,camyaml=CNACYAML,render_mode="reflected")
            snapshot_png_reflected(visdir,visdir+f"/{name.lower()}_cnac_visible_snapshots.png",vtitle)
            render(THERMDIR,times,visanim,naif,frame,cam_mk,hp,camyaml=CNACYAML,render_mode="reflected")
            make_mp4_reflected(visanim,visdir+f"/{name.lower()}_cnac_visible.mp4",vtitle)

            man[name]={'status':'DONE','ca':ca_utc,'range_km':round(dmin,1),'phase_naif':naif,
                       'pole_assumed':pole_assumed,'snapshots':camdir+f"/{name.lower()}_flyby_snapshots.png",
                       'mp4':camdir+f"/{name.lower()}_flyby.mp4",
                       'reflected_snapshots':camdir+f"/{name.lower()}_flyby_reflected_snapshots.png",
                       'reflected_mp4':camdir+f"/{name.lower()}_flyby_reflected.mp4",
                       'cnac_visible_snapshots':visdir+f"/{name.lower()}_cnac_visible_snapshots.png",
                       'cnac_visible_mp4':visdir+f"/{name.lower()}_cnac_visible.mp4"}
            json.dump(man[name],open(STATUSDIR+f"/{name}.json",'w'),indent=2)
            log(f"{name}: DONE -> thermal + IR(thermal+reflected) + CNAC visible snapshots + MP4s")
        except Exception as e:
            import traceback; log(f"{name}: ERROR {e}\n{traceback.format_exc()}")
            man[name]={'status':'ERROR','error':str(e)}; json.dump(man[name],open(STATUSDIR+f"/{name}.json",'w'),indent=2)
    log("CAMPAIGN DRIVER FINISHED")

if __name__=='__main__':
    main(sys.argv[1:])
