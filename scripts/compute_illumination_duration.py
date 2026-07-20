#!/usr/bin/env python3
"""
Compute per-face illumination duration maps from the 25-lunation survey.

For each lunation:
  1. Run PTM3D with format: binary to get per-timestep direct_solar files
  2. Read timestep files and build per-face illuminated/dark boolean sequence
  3. Accumulate total illuminated timesteps and longest continuous run
  4. Delete per-timestep files to conserve disk space

Outputs:
  output/survey/illum_total_fraction.bin  — float32 per face: fraction of total time illuminated
  output/survey/illum_max_continuous_hours.bin — float32 per face: longest continuous illumination (hours)

Usage:
  python3 scripts/compute_illumination_duration.py [--num-lunations 25] [--skip-sim]
"""

import argparse
import struct
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np


LUNATION_DAYS = 29.530588
NUM_TIMESTEPS = 360  # per lunation
DT_HOURS = LUNATION_DAYS * 24.0 / NUM_TIMESTEPS  # hours per timestep (~1.97 hr)

BASE = Path(__file__).parent.parent
BUILD = BASE / "build"
BIN = BUILD / "bin" / "ptm_thermal_sim"
SURVEY_DIR = BASE / "output" / "survey"
TEMPLATE = BASE / "configs" / "amundsen_survey_template.yaml"


def read_bin(path):
    """Read uint32 count + float32 array."""
    with open(path, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        return np.frombuffer(f.read(count * 4), dtype=np.float32)


def write_bin(path, data):
    """Write uint32 count + float32 array."""
    arr = data.astype(np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(arr)))
        f.write(arr.tobytes())


def run_lunation(lunation_idx, output_dir):
    """Run one lunation with binary output, return True on success."""
    start = datetime(2026, 4, 1) + timedelta(days=lunation_idx * LUNATION_DAYS)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%S')

    # Generate config with format: binary
    config_path = output_dir / f"lunation_{lunation_idx:02d}.yaml"
    template_text = TEMPLATE.read_text()
    # Override format to binary for per-timestep output
    config_text = template_text.replace("{START_TIME}", start_str)
    config_text = config_text.replace("{OUTPUT_DIR}", str(output_dir / f"lunation_{lunation_idx:02d}"))
    config_text = config_text.replace("format: none", "format: binary")
    config_path.write_text(config_text)

    (output_dir / f"lunation_{lunation_idx:02d}").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [str(BIN), "--config", str(config_path)],
        capture_output=True, text=True, timeout=3600
    )
    return result.returncode == 0


def process_lunation_timesteps(lunation_dir, nf):
    """Read all direct_solar timestep files and return boolean array (nf, nt)."""
    ds_dir = lunation_dir / "direct_solar"
    if not ds_dir.exists():
        return None

    files = sorted(ds_dir.glob("direct_solar_*.bin"))
    nt = len(files)
    if nt == 0:
        return None

    lit = np.zeros((nf, nt), dtype=bool)
    for t, fpath in enumerate(files):
        data = read_bin(str(fpath))
        if len(data) == nf:
            lit[:, t] = data > 0

    return lit


def longest_consecutive_true(arr):
    """For a 1D boolean array, return length of longest consecutive True run."""
    if not np.any(arr):
        return 0
    changes = np.diff(arr.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    # Handle edge cases
    if arr[0]:
        starts = np.concatenate([[0], starts])
    if arr[-1]:
        ends = np.concatenate([ends, [len(arr)]])
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-lunations', type=int, default=25)
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, only process existing timestep files')
    args = parser.parse_args()

    # Determine face count from existing statistics
    sample = SURVEY_DIR / "lunation_00" / "statistics" / "direct_solar_avg.bin"
    if sample.exists():
        nf = len(read_bin(str(sample)))
    else:
        nf = 4001388  # fallback
    print(f"Face count: {nf:,}")

    total_timesteps = args.num_lunations * NUM_TIMESTEPS
    total_illuminated = np.zeros(nf, dtype=np.int64)
    max_continuous = np.zeros(nf, dtype=np.int64)

    # Track state at lunation boundaries for cross-lunation continuity
    prev_tail_lit = np.zeros(nf, dtype=bool)  # last timestep of previous lunation
    current_run = np.zeros(nf, dtype=np.int64)  # ongoing run length from prev lunation

    for i in range(args.num_lunations):
        lun_dir = SURVEY_DIR / f"lunation_{i:02d}"
        print(f"--- Lunation {i:02d} ---")

        # Run simulation if needed
        if not args.skip_sim:
            ds_dir = lun_dir / "direct_solar"
            if not ds_dir.exists() or len(list(ds_dir.glob("*.bin"))) < NUM_TIMESTEPS:
                print(f"  Running PTM3D...")
                ok = run_lunation(i, SURVEY_DIR)
                if not ok:
                    print(f"  FAILED — skipping")
                    prev_tail_lit[:] = False
                    current_run[:] = 0
                    continue
            else:
                print(f"  Per-timestep data already exists")

        # Process timestep files
        lit = process_lunation_timesteps(lun_dir, nf)
        if lit is None:
            print(f"  No timestep data — skipping")
            prev_tail_lit[:] = False
            current_run[:] = 0
            continue

        nt = lit.shape[1]
        print(f"  Read {nt} timesteps, {np.sum(lit)/(nf*nt)*100:.1f}% face-timesteps illuminated")

        # Count illuminated timesteps
        total_illuminated += lit.sum(axis=1)

        # Find longest continuous illumination run, accounting for cross-lunation continuity
        for f_start in range(0, nf, 100000):
            f_end = min(f_start + 100000, nf)
            chunk = lit[f_start:f_end]

            for j in range(f_end - f_start):
                fi = f_start + j
                face_seq = chunk[j]

                # Prepend the ongoing run from previous lunation
                if current_run[fi] > 0 and face_seq[0]:
                    # Continuation of a run across lunation boundary
                    # Find how far the run extends into this lunation
                    run_ext = 0
                    while run_ext < nt and face_seq[run_ext]:
                        run_ext += 1
                    total_run = current_run[fi] + run_ext
                    max_continuous[fi] = max(max_continuous[fi], total_run)

                    # Also check internal runs
                    if run_ext < nt:
                        internal_max = longest_consecutive_true(face_seq[run_ext:])
                        max_continuous[fi] = max(max_continuous[fi], internal_max)

                    # Update current_run for next lunation
                    if face_seq[-1]:
                        # Find trailing run length
                        trail = 0
                        for k in range(nt - 1, -1, -1):
                            if face_seq[k]:
                                trail += 1
                            else:
                                break
                        current_run[fi] = trail if trail < nt else total_run
                    else:
                        current_run[fi] = 0
                else:
                    # No cross-lunation continuation
                    internal_max = longest_consecutive_true(face_seq)
                    max_continuous[fi] = max(max_continuous[fi], internal_max)

                    if face_seq[-1]:
                        trail = 0
                        for k in range(nt - 1, -1, -1):
                            if face_seq[k]:
                                trail += 1
                            else:
                                break
                        current_run[fi] = trail
                    else:
                        current_run[fi] = 0

                prev_tail_lit[fi] = face_seq[-1]

        # Delete per-timestep files to save disk space
        for subdir in ['direct_solar', 'illumination', 'primary_reflected', 'primary_ir',
                        'temperature', 'mesh.ply']:
            p = lun_dir / subdir
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
        print(f"  Cleaned per-timestep data")

    # Compute output arrays
    illum_fraction = total_illuminated.astype(np.float32) / total_timesteps
    max_continuous_hours = (max_continuous * DT_HOURS).astype(np.float32)

    # Write outputs
    frac_path = SURVEY_DIR / "illum_total_fraction.bin"
    hours_path = SURVEY_DIR / "illum_max_continuous_hours.bin"
    write_bin(str(frac_path), illum_fraction)
    write_bin(str(hours_path), max_continuous_hours)

    print(f"\n=== RESULTS ===")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Hours per timestep: {DT_HOURS:.2f}")
    print(f"Illumination fraction: min={illum_fraction.min():.4f}, max={illum_fraction.max():.4f}, "
          f"mean={illum_fraction.mean():.4f}")
    print(f"Max continuous hours: min={max_continuous_hours.min():.1f}, "
          f"max={max_continuous_hours.max():.1f}, mean={max_continuous_hours.mean():.1f}")
    print(f"PSR faces (frac=0): {np.sum(illum_fraction == 0):,}")
    print(f"\nOutputs written:")
    print(f"  {frac_path}")
    print(f"  {hours_path}")


if __name__ == '__main__':
    main()
