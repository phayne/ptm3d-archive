#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Staleness guard: mc_ref_pp is an untracked compiled artifact with no build
# step keeping it in sync with mc_ref_pp.cpp. A copy built before a merged
# source fix (e.g. the back-face cull, 44210d8/ce3ca8c) silently produces
# defective view factors, and this script used to run ./mc_ref_pp by its bare
# fixed path with no check at all. ensure_mc_ref_pp.py ALWAYS compiles from
# current source (cached by source hash, so an unchanged tree costs a hash
# check, not a recompile) and refuses -- nonzero exit, message naming the
# stale path and the rebuild command -- if a copy at the legacy fixed path
# disagrees with a fresh compile. Do not replace this with a bare ./mc_ref_pp.
BIN="$(python3 "$SCRIPT_DIR/ensure_mc_ref_pp.py")" || {
  echo "FATAL: mc_ref_pp staleness guard failed (see message above); sweep aborted." >&2
  exit 1
}

MESH=/Users/paha3326/dev/heat3d/moon3d_c_cartesian_NEW_500k_clean_native.ply
TMAX=/tmp/ptm12_rfd/temperature_max.bin
for c in $(seq -w 0 47); do
  "$BIN" "$MESH" "$TMAX" sweep_chunks/chunk_${c}.txt 40000 1e-4 0.2 1e-5 /tmp/pp_sweep/pp_${c}.bin > /tmp/pp_sweep/log_${c}.txt 2>&1 &
done
wait
echo "sweep done $(date); total bytes: $(du -sh /tmp/pp_sweep/pp_*.bin | tail -1)" > /tmp/pp_sweep/DONE.txt
du -ch /tmp/pp_sweep/pp_*.bin | tail -1 >> /tmp/pp_sweep/DONE.txt
