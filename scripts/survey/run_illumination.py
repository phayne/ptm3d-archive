#!/usr/bin/env python3
"""
Run illumination survey for all lunations and compute duration maps.
"""
import subprocess
import shutil
import sys
from pathlib import Path
from string import Template

import numpy as np

from survey_config import SurveyConfig, load_config
from utils import read_stats_binary, write_stats_binary, save_state, load_state


SURVEY_TEMPLATE = """simulation:
  start_time: '${START_TIME}'
  duration_days: ${LUNATION_DAYS}
  timesteps: ${TIMESTEPS}
  body: ${BODY}
mesh:
  resolution: 5
  import_file: ${MESH_FILE}
  scale_factor: 1.0
thermal:
  run_thermal_mesh: false
  compute_statistics: true
radiation:
  enabled: false
illumination:
  include_primary_reflection: ${INCLUDE_REFLECTION}
  use_batch_processing: true
  use_albedo_map: ${USE_ALBEDO_MAP}
output:
  directory: '${OUTPUT_DIR}'
  format: ${FORMAT}
advanced:
  verbose: false
"""


def find_ptm_binary():
    """Find ptm_thermal_sim binary."""
    candidates = [
        Path(__file__).parent.parent.parent / "build" / "bin" / "ptm_thermal_sim",
        Path("build/bin/ptm_thermal_sim"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def run_lunation(cfg, lunation_idx, ptm_bin, output_format='none'):
    """Run one lunation illumination simulation."""
    survey_dir = cfg.survey_dir
    lun_dir = survey_dir / f"lunation_{lunation_idx:02d}"
    lun_dir.mkdir(parents=True, exist_ok=True)

    config_text = Template(SURVEY_TEMPLATE).substitute(
        START_TIME=cfg.time.lunation_start(lunation_idx),
        LUNATION_DAYS=cfg.time.lunation_days,
        TIMESTEPS=cfg.time.timesteps_per_lunation,
        BODY=cfg.body,
        MESH_FILE=str(cfg.mesh_path),
        INCLUDE_REFLECTION=str(cfg.illumination.include_primary_reflection).lower(),
        USE_ALBEDO_MAP=str(cfg.illumination.use_albedo_map).lower(),
        OUTPUT_DIR=str(lun_dir),
        FORMAT=output_format,
    )

    config_file = survey_dir / f"lunation_{lunation_idx:02d}.yaml"
    config_file.write_text(config_text)

    result = subprocess.run(
        [ptm_bin, "--config", str(config_file)],
        capture_output=True, text=True, timeout=3600
    )
    return result.returncode == 0


def run_illumination_survey(cfg: SurveyConfig) -> bool:
    """Run all lunation illumination simulations."""
    ptm_bin = find_ptm_binary()
    if not ptm_bin:
        print("ERROR: ptm_thermal_sim not found. Build first.")
        return False

    state = load_state(cfg.state_file)
    done_lunations = state.get('illumination', {}).get('lunations_done', [])
    n_lun = cfg.illumination.num_lunations

    print(f"=== Illumination Survey ({n_lun} lunations) ===")

    for i in range(n_lun):
        stats_file = cfg.survey_dir / f"lunation_{i:02d}" / "statistics" / "direct_solar_avg.bin"
        if i in done_lunations and stats_file.exists():
            print(f"  L{i:02d}: skipping (already done)")
            continue

        fmt = 'binary' if cfg.illumination.compute_duration_maps else 'none'
        print(f"  L{i:02d}: running ({cfg.time.lunation_start(i)})...", end=' ', flush=True)
        ok = run_lunation(cfg, i, ptm_bin, output_format=fmt)
        if not ok:
            print("FAILED")
            continue
        print("OK")

        if i not in done_lunations:
            done_lunations.append(i)

        state['illumination'] = {'status': 'partial', 'lunations_done': sorted(done_lunations)}
        save_state(cfg.state_file, state)

    state['illumination']['status'] = 'complete'
    save_state(cfg.state_file, state)
    return True


def compute_duration_maps(cfg: SurveyConfig) -> bool:
    """Compute illumination fraction and max continuous duration per face."""
    print("=== Computing Duration Maps ===")
    n_lun = cfg.illumination.num_lunations
    nt = cfg.time.timesteps_per_lunation
    dt_hours = cfg.time.hours_per_timestep
    total_timesteps = n_lun * nt

    # Detect face count
    sample = cfg.survey_dir / "lunation_00" / "statistics" / "direct_solar_avg.bin"
    if not sample.exists():
        print("ERROR: No survey data found. Run illumination first.")
        return False
    nf = len(read_stats_binary(str(sample)))
    print(f"  Faces: {nf:,}, timesteps/lunation: {nt}, total: {total_timesteps}")

    total_illuminated = np.zeros(nf, dtype=np.int64)
    max_continuous = np.zeros(nf, dtype=np.int64)
    current_run = np.zeros(nf, dtype=np.int64)

    for i in range(n_lun):
        ds_dir = cfg.survey_dir / f"lunation_{i:02d}" / "direct_solar"
        if not ds_dir.exists():
            print(f"  L{i:02d}: no per-timestep data, skipping")
            current_run[:] = 0
            continue

        files = sorted(ds_dir.glob("direct_solar_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
        if len(files) < nt:
            print(f"  L{i:02d}: incomplete ({len(files)}/{nt} timesteps)")
            current_run[:] = 0
            continue

        # Read all timesteps into boolean array
        lit = np.zeros((nf, len(files)), dtype=bool)
        for t, fpath in enumerate(files):
            data = read_stats_binary(str(fpath))
            if len(data) == nf:
                lit[:, t] = data > 0

        total_illuminated += lit.sum(axis=1)

        # Find max continuous run with cross-lunation tracking
        for fi in range(nf):
            face_seq = lit[fi]
            if current_run[fi] > 0 and face_seq[0]:
                run_ext = 0
                while run_ext < len(files) and face_seq[run_ext]:
                    run_ext += 1
                max_continuous[fi] = max(max_continuous[fi], current_run[fi] + run_ext)
                if run_ext < len(files):
                    from _longest_run import longest_consecutive_true
                    internal = longest_consecutive_true(face_seq[run_ext:])
                    max_continuous[fi] = max(max_continuous[fi], internal)
            else:
                internal = _longest_run_np(face_seq)
                max_continuous[fi] = max(max_continuous[fi], internal)

            # Track trailing run for next lunation
            if face_seq[-1]:
                trail = 0
                for k in range(len(files)-1, -1, -1):
                    if face_seq[k]: trail += 1
                    else: break
                current_run[fi] = trail
            else:
                current_run[fi] = 0

        # Clean per-timestep data if configured
        if cfg.output.clean_per_timestep_data:
            for subdir in ['direct_solar', 'illumination', 'primary_reflected', 'primary_ir', 'mesh.ply']:
                p = cfg.survey_dir / f"lunation_{i:02d}" / subdir
                if p.is_dir(): shutil.rmtree(p)
                elif p.is_file(): p.unlink()

        print(f"  L{i:02d}: processed ({lit.sum()/(nf*len(files))*100:.1f}% illuminated)")

    # Write output
    illum_frac = total_illuminated.astype(np.float32) / total_timesteps
    max_cont_hrs = (max_continuous * dt_hours).astype(np.float32)

    frac_path = cfg.survey_dir / "illum_total_fraction.bin"
    hours_path = cfg.survey_dir / "illum_max_continuous_hours.bin"
    write_stats_binary(str(frac_path), illum_frac)
    write_stats_binary(str(hours_path), max_cont_hrs)

    print(f"  Illum fraction: {illum_frac.min():.4f}–{illum_frac.max():.4f}")
    print(f"  Max continuous: {max_cont_hrs.min():.0f}–{max_cont_hrs.max():.0f} hr")
    print(f"  PSR faces: {np.sum(illum_frac == 0):,}")

    state = load_state(cfg.state_file)
    state['duration_maps'] = {'status': 'complete'}
    save_state(cfg.state_file, state)
    return True


def _longest_run_np(arr):
    """Longest consecutive True run in a boolean array."""
    if not np.any(arr): return 0
    changes = np.diff(arr.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if arr[0]: starts = np.concatenate([[0], starts])
    if arr[-1]: ends = np.concatenate([ends, [len(arr)]])
    if len(starts) == 0: return 0
    return int(np.max(ends - starts))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_illumination.py <config.yaml> [--duration-only]")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    if '--duration-only' in sys.argv:
        compute_duration_maps(cfg)
    else:
        run_illumination_survey(cfg)
        if cfg.illumination.compute_duration_maps:
            compute_duration_maps(cfg)
