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

def render(ptm_output, times, outdir, naif, frame, mk, hp):
    # Reflectance is now Hapke: single-scattering albedo hp['w'] is calibrated so
    # the disk-integrated geometric albedo matches the body's known value; B0/h/g/
    # theta_bar carry the phase dependence away from opposition. The C++ seeds the
    # uniform vertex albedo from hapke_w so the calibrated model is rendered as-is
    # (no separate geometric-albedo scalar rescale). Up-vector uses the boresight-
    # projected north pole (keeps north in the image plane).
    os.makedirs(outdir,exist_ok=True)
    cmd=[CBIN,"--ptm-output",ptm_output,
         "--camera-yaml",CAMYAML,"--meta-kernel",mk,"--trajectory",SCSPK,
         "--traj-format","spk","--traj-spacecraft","-60000","--target-body",naif,
         "--target-frame",frame,"--render-mode","both","--thermal-wavelength","10.5",
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

# ---- main ----
if __name__ == "__main__":
    man=json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    os.makedirs(STATUSDIR,exist_ok=True)
    os.makedirs(LOGDIR,exist_ok=True)
    if len(sys.argv)>1:
        sel=set(sys.argv[1:]); TARGETS=[t for t in TARGETS if t[1] in sel]
    for num,name in TARGETS:
        try:
            sp.kclear()
            cand=[d for d in os.listdir(BODIES) if d.startswith(num+"_")]
            if not cand: log(f"{name}: NO CACHE DIR -> skip"); continue
            bdir=os.path.join(BODIES,cand[0]); spkf=bdir+"/ephemeris.spk"; bcfg=bdir+"/body_config.yaml"
            cfg=yaml.safe_load(open(bcfg)); gb=cfg.get('generic_body',cfg)
            for k in [CSP+"/lsk/naif0012.tls",CSP+"/spk/de440s.bsp",spkf,SCSPK]: sp.furnsh(k)
            naif=str(spk_body_id(spkf)[0])
            radius=float(gb['shape']['radius_km']); period=float(gb['spin']['period_hours'])
            albedo=float(gb['surface']['albedo'])
            # Shape selection: honor whatever the body_config declares. A body may
            # ship a real shape model (type: custom_mesh + mesh_file); the previous
            # driver hardcoded type: spherical and blanked mesh.import_file, silently
            # discarding it. Carry the declared type + mesh_file through to BOTH
            # generic_body.shape.mesh_file and top-level mesh.import_file. When no
            # shape file is present, fall back to the spherical approximation with a
            # clear warning.
            shape_type=str(gb['shape'].get('type','spherical')).strip().lower()
            shape_mesh_file=str(gb['shape'].get('mesh_file','') or '')
            shape_mesh_scale=float(gb['shape'].get('mesh_scale',1.0))
            if shape_type=='custom_mesh' and shape_mesh_file and os.path.exists(shape_mesh_file):
                use_mesh=True
                log(f"{name}: shape model -> custom_mesh {shape_mesh_file} (scale {shape_mesh_scale})")
            else:
                use_mesh=False
                if shape_type=='custom_mesh':
                    log(f"{name}: WARNING custom_mesh declared but mesh_file "
                        f"'{shape_mesh_file}' missing -> using spherical approximation")
                elif shape_type not in ('spherical',''):
                    log(f"{name}: WARNING unsupported shape type '{shape_type}' -> "
                        f"using spherical approximation")
                else:
                    log(f"{name}: no shape model -> using spherical approximation "
                        f"(radius {radius} km)")
            ra=gb['spin'].get('pole_ra_deg'); dec=gb['spin'].get('pole_dec_deg')
            pole_assumed=False
            if ra is None or dec is None:  # no measured pole -> orbital pole (obliquity 0)
                o=gb['orbit']; i=math.radians(o['inclination_deg']); Om=math.radians(o['lon_ascending_node_deg'])
                nx=math.sin(i)*math.sin(Om); ny=-math.sin(i)*math.cos(Om); nz=math.cos(i)
                lon=math.degrees(math.atan2(ny,nx))%360; lat=math.degrees(math.asin(nz))
                ra,dec=ecl_pole_to_radec(lon,lat); pole_assumed=True
            ra=float(ra); dec=float(dec)
            etm,dmin,vrel=find_ca(naif)
            ca_utc=sp.et2utc(etm,'ISOC',0)
            log(f"{name}: NAIF {naif} CA {ca_utc} range {dmin:.0f}km vrel {vrel:.1f} period {period:.3f}h pole({ra:.1f},{dec:.1f}){' ASSUMED' if pole_assumed else ''}")

            # Resolve Hapke params (incl. single-scattering albedo w) up front so the
            # thermal directional-hemispheric albedo path and the camera reflectance use
            # the SAME per-body w. body_config 'surface.hapke' overrides win here too.
            hp=hapke_params(num,gb)
            THERMDIR=f"{TROOT}/{num}{name.lower()}/flyby"; os.makedirs(THERMDIR,exist_ok=True)
            # thermal config: 1.5-rotation window centered on CA, 40-orbit eq
            win_h=1.5*period; start_et=etm-0.75*period*3600.0
            tconf={'simulation':{'start_time':sp.et2utc(start_et,'ISOC',0).replace('Z',''),
                    'duration_days':win_h/24.0,'timesteps':292,'body':gb.get('name',name)},
                'mesh':{'resolution':6,'import_file':(shape_mesh_file if use_mesh else ''),
                        'scale_factor':(shape_mesh_scale if use_mesh else 1.0),'export_basic':False},
                'thermal':{'run_thermal_mesh':True,'threads':0,'equilibration_orbits':40,'use_gpu':True,
                    'compute_statistics':True,'integration_days':-1.0,'integration_steps':-1,
                    'integration_offset_days':0.0,'integration_offset_steps':0,'geothermal_heat_flow':-1.0,
                    'enable_shadow_flux':False,'crater_dd_ratio':8.0,'thermal_inertia_map':''},
                'thermophysical':{'use_default_albedo':False,'albedo':albedo,'use_default_emissivity':False,
                    'emissivity':0.95,'use_default_ks':False,'ks':0.01,'use_default_kd':False,'kd':0.5,
                    'use_default_H':False,'H':0.05,'use_default_rhos':False,'rhos':1500.0,'use_default_rhod':False,
                    'rhod':2000.0,'use_default_cp0':False,'cp0':800.0,'use_default_Qb':False,'Qb':0.0,
                    'use_default_chi':False,'chi':0.0,'incidence_angle_albedo':True,
                    'albedo_hapke_w':hp['w']},
                'radiation':{'enabled':False},
                'illumination':{'include_primary_reflection':False,'use_batch_processing':True},
                'output':{'directory':THERMDIR,'format':'binary','vtk_mode':'none','series_output':'per_timestep'},
                'advanced':{'verbose':False,'default_visible_albedo':-1.0},
                'generic_body':{'name':gb.get('name',name),'config_file':bcfg,'ephemeris_source':'spice',
                    'spice_metakernel':bdir+'/metakernel.tm','naif_id':naif,
                    'shape':{'type':('custom_mesh' if use_mesh else 'spherical'),'radius_km':radius,
                             'mesh_file':(shape_mesh_file if use_mesh else ''),
                             'mesh_scale':(shape_mesh_scale if use_mesh else 1.0),
                             'semi_axes_km':[0.001,0.001,0.001]},
                    'orbit':gb['orbit'],
                    'spin':{'period_hours':period,'pole_ra_deg':ra,'pole_dec_deg':dec,'prime_meridian_deg':0.0},
                    'surface':{'albedo':albedo,'emissivity':0.95,'ks':0.01,'kd':0.5,'rhos':1500.0,'rhod':2000.0,
                               'H':0.05,'cp0':800.0,'Qb':0.0,'chi':0.0}}}
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
            man[name]={'status':'DONE','ca':ca_utc,'range_km':round(dmin,1),'phase_naif':naif,
                       'pole_assumed':pole_assumed,'snapshots':camdir+f"/{name.lower()}_flyby_snapshots.png",
                       'mp4':camdir+f"/{name.lower()}_flyby.mp4",
                       'reflected_snapshots':camdir+f"/{name.lower()}_flyby_reflected_snapshots.png",
                       'reflected_mp4':camdir+f"/{name.lower()}_flyby_reflected.mp4"}
            json.dump(man[name],open(STATUSDIR+f"/{name}.json",'w'),indent=2)
            log(f"{name}: DONE -> thermal + reflected snapshots + MP4s")
        except Exception as e:
            import traceback; log(f"{name}: ERROR {e}\n{traceback.format_exc()}")
            man[name]={'status':'ERROR','error':str(e)}; json.dump(man[name],open(STATUSDIR+f"/{name}.json",'w'),indent=2)
    log("CAMPAIGN DRIVER FINISHED")
