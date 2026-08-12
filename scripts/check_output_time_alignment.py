#!/usr/bin/env python3
"""Assert that PTM3D per-step frames are filed under the right time index.

THE CONVENTION, which every solver path must honor:

    temperature/temperature_NNNN.bin holds the surface state AT the epoch on
    line NNNN of time_steps.txt -- the same epoch the illumination frames
    direct_solar_NNNN.bin / illumination_NNNN.bin are sampled at.

A path that stores the state at the END of interval NNNN under index NNNN
produces a file whose temperatures run one output step ahead of their own
epochs and one output step ahead of their own illumination. That defect is
invisible to `statistics/*.bin` (integrated over the window, it moved the
per-face mean by 0.037 K max on the 80 N crater) and invisible to any
comparison of a path against itself, so it needs an anchor outside the series.

Two anchors work, and this script checks either one. Both pin index 0, where
ZERO time has elapsed, so neither depends on the flux quadrature inside an
interval:

  --checkpoint <file.ptmt>   A warm-started run (thermal.initial_state_file)
                             must reproduce the checkpoint's surface layer
                             exactly at index 0. GPU path only -- the T(z)
                             checkpoint is read and written only there.

  --seed-csv <file.csv>      A run with thermal.equilibration_orbits = 0 starts
                             at the shared analytic spin-up seed, which every
                             path records under PTM_DUMP_SEED=<file.csv>. Works
                             on all three solver paths.

Exit status is 1 on failure, so this can gate a build.

Example
-------
  PTM_DUMP_SEED=/tmp/seed.csv build/bin/ptm_thermal_sim \\
      --config my_noeq.yaml --output /tmp/run --no-simd-batch
  python3 scripts/check_output_time_alignment.py /tmp/run --seed-csv /tmp/seed.csv
"""
import argparse
import csv
import glob
import os
import struct
import sys

PTMT_MAGIC = 0x50544D54


def read_frame(path):
    """A PTM3D frame is an int32 face count then that many FLOAT32 -- not f64."""
    with open(path, 'rb') as fh:
        raw = fh.read()
    if len(raw) < 4:
        raise SystemExit('%s: truncated' % path)
    n = struct.unpack('<i', raw[:4])[0]
    if 4 + 4 * n != len(raw):
        raise SystemExit('%s: header says %d faces but the file is %d bytes'
                         % (path, n, len(raw)))
    return list(struct.unpack('<%df' % n, raw[4:]))


def first_temperature_frame(rundir):
    files = sorted(glob.glob(os.path.join(rundir, 'temperature', 'temperature_*.bin')))
    if not files:
        raise SystemExit('no temperature/temperature_*.bin under ' + rundir)
    return read_frame(files[0]), files[0]


def checkpoint_surface(path):
    """PTMT: magic, n_faces, n_layers (uint32) then layer-major float32."""
    with open(path, 'rb') as fh:
        head = fh.read(12)
        if len(head) < 12:
            raise SystemExit('%s: truncated checkpoint' % path)
        magic, n_faces, n_layers = struct.unpack('<III', head)
        if magic != PTMT_MAGIC:
            raise SystemExit('%s: bad PTMT magic 0x%08x' % (path, magic))
        body = fh.read(n_faces * 4)          # layer 0 == the surface
        if len(body) < n_faces * 4:
            raise SystemExit('%s: short read on the surface layer' % path)
    return list(struct.unpack('<%df' % n_faces, body))


def seed_surface(path):
    with open(path) as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith('#')))
    if not rows:
        raise SystemExit('%s: no seed rows -- was PTM_DUMP_SEED set?' % path)
    out = [0.0] * (max(int(r['face']) for r in rows) + 1)
    for r in rows:
        out[int(r['face'])] = float(r['T_surface_K'])
    return out


def compare(frame, reference, tol, what, frame_path, ref_path):
    if len(frame) != len(reference):
        raise SystemExit('face-count mismatch: frame %d, %s %d'
                         % (len(frame), what, len(reference)))
    worst, worst_f, n_over = -1.0, 0, 0
    for f, (a, b) in enumerate(zip(frame, reference)):
        d = abs(a - b)
        if d > tol:
            n_over += 1
        if d > worst:
            worst, worst_f = d, f
    print('  frame     %s' % frame_path)
    print('  %-9s %s' % (what + ':', ref_path))
    print('  |T[f,0] - %s[f]|  max = %.6f K at face %d   faces over %.2g K: %d/%d'
          % (what, worst, worst_f, tol, n_over, len(frame)))
    if n_over:
        print('  FAIL: output index 0 is not the state at times[0] -- the frame has'
              ' already been integrated past its own epoch.')
        return 1
    print('  PASS: output index 0 is the state at times[0].')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rundir')
    ap.add_argument('--checkpoint', help='PTMT T(z) file the run warm-started from')
    ap.add_argument('--seed-csv', help='PTM_DUMP_SEED csv from the same run')
    ap.add_argument('--tol', type=float, default=1e-3,
                    help='kelvin; default 1e-3 covers the float32 frame quantization')
    args = ap.parse_args()

    if not (args.checkpoint or args.seed_csv):
        raise SystemExit('give --checkpoint or --seed-csv: index 0 needs an anchor '
                         'outside the series, or there is nothing to compare against')

    frame, frame_path = first_temperature_frame(args.rundir)
    rc = 0
    if args.checkpoint:
        print('=== output time alignment: warm-start checkpoint anchor ===')
        rc |= compare(frame, checkpoint_surface(args.checkpoint), args.tol,
                      'checkpoint', frame_path, args.checkpoint)
    if args.seed_csv:
        print('=== output time alignment: spin-up seed anchor ===')
        rc |= compare(frame, seed_surface(args.seed_csv), args.tol,
                      'seed', frame_path, args.seed_csv)
    return rc


if __name__ == '__main__':
    sys.exit(main())
