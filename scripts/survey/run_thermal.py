#!/usr/bin/env python3
"""
Run thermal + radiation exchange simulations for selected lunations.
"""
import subprocess
import shutil
import sys
from pathlib import Path
from string import Template

from survey_config import SurveyConfig, load_config
from utils import save_state, load_state


THERMAL_TEMPLATE = """simulation:
  start_time: '${START_TIME}'
  duration_days: ${LUNATION_DAYS}
  timesteps: ${TIMESTEPS}
  body: ${BODY}
mesh:
  resolution: 5
  import_file: ${MESH_FILE}
  scale_factor: 1.0
thermal:
  run_thermal_mesh: true
  threads: 0
  equilibration_orbits: ${EQUIL_ORBITS}
  use_gpu: ${USE_GPU}
  compute_statistics: true
thermophysical:
  use_default_albedo: true
  albedo: ${ALBEDO}
  use_default_emissivity: true
  emissivity: ${EMISSIVITY}
  incidence_angle_albedo: ${INC_ALBEDO}
  use_default_ks: true
  ks: ${KS}
  use_default_kd: true
  kd: ${KD}
  use_default_H: true
  H: ${H}
  use_default_rhos: true
  rhos: ${RHOS}
  use_default_rhod: true
  rhod: ${RHOD}
  use_default_cp0: true
  cp0: ${CP0}
  use_default_Qb: true
  Qb: ${QB}
  use_default_chi: true
  chi: ${CHI}
radiation:
  enabled: ${RAD_ENABLED}
  use_gpu: false
  view_factor_file: '${VF_FILE}'
  use_hmatrix: true
  use_aca: true
  max_coupling_iterations: ${RAD_ITER}
  adaptive_neighbors: true
  captured_energy_target: 0.95
illumination:
  include_primary_reflection: true
  use_batch_processing: true
  use_albedo_map: false
output:
  directory: '${OUTPUT_DIR}'
  format: binary
  locations_file: '${LOCATIONS_FILE}'
  series_output: 'location_timeseries.csv'
"""


def find_ptm_binary():
    candidates = [
        Path(__file__).parent.parent.parent / "build" / "bin" / "ptm_thermal_sim",
        Path("build/bin/ptm_thermal_sim"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def run_thermal(cfg: SurveyConfig) -> bool:
    """Run thermal simulations for configured lunations."""
    if not cfg.thermal.enabled:
        print("Thermal model disabled in config.")
        return True

    ptm_bin = find_ptm_binary()
    if not ptm_bin:
        print("ERROR: ptm_thermal_sim not found.")
        return False

    if not cfg.locations_csv.exists():
        print(f"ERROR: Locations file not found: {cfg.locations_csv}")
        print("Run site selection first.")
        return False

    state = load_state(cfg.state_file)
    done = state.get('thermal', {}).get('lunations_done', [])
    tp = cfg.thermophysical

    print(f"=== Thermal Simulations ({len(cfg.thermal.lunations)} lunations) ===")

    for lun_idx in cfg.thermal.lunations:
        lun_dir = cfg.thermal_dir / f"lunation_{lun_idx:02d}"
        ts_file = lun_dir / "location_timeseries.csv"

        if lun_idx in done and ts_file.exists():
            print(f"  L{lun_idx:02d}: skipping (already done)")
            continue

        lun_dir.mkdir(parents=True, exist_ok=True)

        config_text = Template(THERMAL_TEMPLATE).substitute(
            START_TIME=cfg.time.lunation_start(lun_idx),
            LUNATION_DAYS=cfg.time.lunation_days,
            TIMESTEPS=cfg.time.timesteps_per_lunation,
            BODY=cfg.body,
            MESH_FILE=str(cfg.mesh_path),
            EQUIL_ORBITS=cfg.thermal.equilibration_orbits,
            USE_GPU=str(cfg.thermal.use_gpu).lower(),
            ALBEDO=tp.albedo, EMISSIVITY=tp.emissivity,
            INC_ALBEDO=str(tp.incidence_angle_albedo).lower(),
            KS=tp.ks, KD=tp.kd, H=tp.H,
            RHOS=tp.rhos, RHOD=tp.rhod, CP0=tp.cp0, QB=tp.Qb, CHI=tp.chi,
            RAD_ENABLED=str(cfg.thermal.radiation_exchange).lower(),
            VF_FILE=str(cfg.vf_path),
            RAD_ITER=cfg.thermal.max_coupling_iterations,
            OUTPUT_DIR=str(lun_dir),
            LOCATIONS_FILE=str(cfg.locations_csv),
        )

        config_file = cfg.thermal_dir / f"lunation_{lun_idx:02d}.yaml"
        config_file.write_text(config_text)

        print(f"  L{lun_idx:02d}: running ({cfg.time.lunation_start(lun_idx)})...")
        log_file = cfg.thermal_dir / f"lunation_{lun_idx:02d}_stdout.log"
        with open(log_file, 'w') as log:
            result = subprocess.run(
                [ptm_bin, "--config", str(config_file)],
                stdout=log, stderr=subprocess.STDOUT, timeout=36000
            )

        if result.returncode != 0 or not ts_file.exists():
            print(f"    FAILED (see {log_file})")
            continue

        # Clean per-timestep data
        if cfg.output.clean_per_timestep_data:
            for subdir in ['direct_solar', 'illumination', 'primary_reflected', 'primary_ir', 'temperature']:
                p = lun_dir / subdir
                if p.is_dir(): shutil.rmtree(p)
        print(f"    OK -> {ts_file}")

        done.append(lun_idx)
        state['thermal'] = {'status': 'partial', 'lunations_done': sorted(done)}
        save_state(cfg.state_file, state)

    state['thermal'] = {'status': 'complete', 'lunations_done': sorted(done)}
    save_state(cfg.state_file, state)
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_thermal.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    run_thermal(cfg)
