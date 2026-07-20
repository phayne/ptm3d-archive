#!/bin/bash
cd /Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/scripts/ptm12_diag
MESH=/Users/paha3326/dev/heat3d/moon3d_c_cartesian_NEW_500k_clean_native.ply
TMAX=/tmp/ptm12_rfd/temperature_max.bin
for c in $(seq -w 0 47); do
  ./mc_ref_pp "$MESH" "$TMAX" sweep_chunks/chunk_${c}.txt 40000 1e-4 0.2 1e-5 /tmp/pp_sweep/pp_${c}.bin > /tmp/pp_sweep/log_${c}.txt 2>&1 &
done
wait
echo "sweep done $(date); total bytes: $(du -sh /tmp/pp_sweep/pp_*.bin | tail -1)" > /tmp/pp_sweep/DONE.txt
du -ch /tmp/pp_sweep/pp_*.bin | tail -1 >> /tmp/pp_sweep/DONE.txt
