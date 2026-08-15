#!/usr/bin/env python3
"""Autonomous Diviner-validation driver (CORE + OCCLUDER-RING architecture).

Per site: high-res 20 m core (R=5 km, thermal+radiation+output) + coarse LOLA
occluder annulus (5->30 km, shadow-ray BVH only via mesh.occluder_file) so
distant crater rims shadow the PSR floor. Year-long radiation-coupled run
(single-GPU, watchdog-safe batches, retry), then compare max bolometric T to
Diviner. Combined summary at the end. Meshes are pre-built.
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

# Raise the H-matrix ACA block cap so floor<->wall view-factor blocks on the
# ~100k radiation core compress instead of zeroing (full wall->floor coupling).
RUN_ENV = {**os.environ, "PTM_HMATRIX_ACA_CAP": "60000"}

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")
PTM = REPO / "build/bin/ptm_thermal_sim"
PY = "/Users/paha3326/anaconda3/envs/europa_viz/bin/python"
MAT = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/Collaborators/LandisM/moon/Tbols_SP_new_full_run_hires_bad_orbit_removed_channel_all.mat"
MESHDIR = REPO / "data/dem/validation/meshes"
OUTROOT = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/output")
FIGDIR = REPO / "docs/figures/diviner_validation"

SITES = [("haworth", -87.45, -2.1), ("faustini", -87.3, 84.0),
         ("shackleton", -89.66, 129.2), ("sunlit", -88.181, 6.041)]

CONFIG = """simulation:
  start_time: '2026-01-01T00:00:00'
  duration_days: 365.25
  timesteps: 365
  body: Moon
mesh:
  resolution: 9
  import_file: {core}
  occluder_file: {ring}
  scale_factor: 1.0
  use_topographic: false
thermal:
  run_thermal_mesh: true
  threads: 0
  use_gpu: true
  gpu_devices: 1
  gpu_batch_size: 20000
  gpu_inter_batch_pause_ms: 1000
  equilibration_orbits: 2.0
  equilibration_steps_per_day: -1.0
  equilibration_samples_per_diurnal_cycle: 120.0
  compute_statistics: true
  use_default_property_map: true
  property_maps_directory: {repo}/data/property_maps
  thermal_inertia_map: ''
  h_parameter_map: ''
  enable_shadow_flux: false
  crater_dd_ratio: 8.0
  integration_days: -1.0
  integration_steps: -1
  integration_offset_days: 0.0
  integration_offset_steps: 0
  geothermal_heat_flow: -1.0
thermophysical:
  use_default_albedo: true
  albedo: 0.12
  use_default_emissivity: true
  emissivity: 0.95
  incidence_angle_albedo: true
  use_default_ks: true
  ks: 0.00074
  use_default_kd: true
  kd: 0.0034
  use_default_H: true
  H: 0.06
  use_default_rhos: true
  rhos: 1100.0
  use_default_rhod: true
  rhod: 1800.0
  use_default_cp0: true
  cp0: 800.0
  use_default_Qb: true
  Qb: 0.018
  use_default_chi: true
  chi: 2.7
  constant_thermal_inertia: -1.0
radiation:
  enabled: true
  min_shadow_fraction: 0.8
  verbose: false
  use_gpu: true
  view_factor_file: {vf}
  # 2026-08-15: was 5, pinned by habit. The outer Picard rate is a property of
  # the view-factor spectrum, r <= eps*rho(F)/(1-(1-eps)*rho(F)); on an enclosed
  # lunar mesh (rho(F) = 0.7766) the sweep-to-sweep increment first falls below
  # 1 K at sweep 16, and a 5-sweep cap left the worst face +26.2 K COLD. 16 is
  # now the engine default too, so this line only restates it -- the engine's
  # own '[rho-advisory]' warns per site if a mesh needs more.
  max_coupling_iterations: 16
  coupling_temp_threshold: 2.0
  coupling_flux_threshold: 0.1
  coupling_face_fraction: 0.01
  use_hmatrix: true
  use_aca: true
  aca_tolerance: 0.001
  aca_max_rank: 50
  hmatrix_threads: 0
  adaptive_neighbors: true
  captured_energy_target: 0.95
  max_neighbors: 500
  min_neighbors: 20
illumination:
  include_primary_reflection: true
  use_batch_processing: true
  use_albedo_map: false
  albedo_map: ''
  positions_file: ''
output:
  directory: {out}
  format: binary
  vtk_mode: thermal
  include_primary_ir: false
  skip_time_series: false
advanced:
  verbose: true
  default_visible_albedo: -1.0
"""


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_sim(name):
    core = MESHDIR / f"{name}_radcore_r25.ply"   # crater+walls in radiation VF
    ring = MESHDIR / f"{name}_ring_25_40.ply"    # distant-terrain shadows only
    if not core.exists() or not ring.exists():
        log(f"{name}: MISSING core/ring mesh"); return None
    out = OUTROOT / f"divval_{name}_cr"
    vf = MESHDIR / f"{name}_radcore_r25.vf"
    cfg = REPO / f"examples/divval_{name}_cr.yaml"
    cfg.write_text(CONFIG.format(core=core, ring=ring, vf=vf, out=out, repo=REPO))
    stats = out / "statistics" / "temperature_max.bin"
    for attempt in range(1, 4):
        log(f"{name}: sim attempt {attempt}")
        with open(f"/tmp/divval_{name}_cr.log", "w") as rl:
            try:
                subprocess.run([str(PTM), "--config", str(cfg)], stdout=rl,
                               stderr=subprocess.STDOUT, timeout=5 * 3600, cwd=str(REPO),
                               env=RUN_ENV)
            except subprocess.TimeoutExpired:
                log(f"{name}: TIMEOUT attempt {attempt}")
        if stats.exists() and stats.stat().st_size > 1000:
            log(f"{name}: sim OK"); return out
        log(f"{name}: no stats (GPU hang?), retry"); time.sleep(60)
    log(f"{name}: SIM FAILED"); return None


def compare(name, lat, lon, out):
    r = subprocess.run([PY, str(REPO / "src/python/diviner_compare.py"), "--diviner", MAT,
                        "--run-dir", str(out), "--center-lat", str(lat), "--center-lon", str(lon),
                        "--window-km", "15", "--core-radius-km", "15", "--cell-m", "250",
                        "--out", str(FIGDIR / f"{name}_cr"),
                        "--label", f"{name}-CR"], capture_output=True, text=True, timeout=1800)
    txt = r.stdout + r.stderr
    log(f"{name}: compare ->\n" + "\n".join(l for l in txt.splitlines() if "bias" in l or "PSR" in l or "sunlit" in l))
    m = {}
    for line in txt.splitlines():
        mm = re.search(r"N=(\d+)\s+bias=([+-][\d.]+).*RMSE=([\d.]+)", line)
        if mm: m["N"], m["bias"], m["rmse"] = int(mm[1]), float(mm[2]), float(mm[3])
        for reg in ("PSR", "sunlit"):
            mm = re.search(rf"{reg}.*n=(\d+) bias=([+-][\d.]+) RMSE=([\d.]+)", line)
            if mm: m[f"{reg}_n"], m[f"{reg}_bias"], m[f"{reg}_rmse"] = int(mm[1]), float(mm[2]), float(mm[3])
    return m


def main():
    results = {}
    for name, lat, lon in SITES:
        log(f"===== SITE {name} (core+ring) ({lat},{lon}) =====")
        out = run_sim(name)
        if not out:
            results[name] = {"error": "sim"}
        else:
            try:
                results[name] = compare(name, lat, lon, out); results[name].update(lat=lat, lon=lon)
            except Exception as e:
                results[name] = {"error": f"compare: {e}"}
        vf = MESHDIR / f"{name}_core_r5.vf"
        if vf.exists(): vf.unlink(); log(f"{name}: deleted VF")
        (FIGDIR / "summary_cr.json").write_text(json.dumps(results, indent=2))
        log(f"{name}: DONE -> {results[name]}")
    log("===== SUMMARY (core+ring) =====")
    for k, v in results.items(): log(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
