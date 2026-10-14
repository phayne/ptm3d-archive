#!/bin/bash
# PTM-727 fix: post-batch build, gates and the DECISIVE CONTROL (Ada, 2026-10-14; Paul-approved).
# Do NOT launch directly: wait_and_run.sh holds the gate and execs this body when it opens.
# Nothing runs until ALL hold: the R1 benchmark batch is terminal (BATCH_DONE / ABORT / GUARD_STOP),
# FERMI_APPROVED exists in this directory (Fermi's review of the seed formulation), and no
# ptm_thermal_sim is running. STATUS: ./CONTROL_STATUS.txt  (terminal: VERDICT_<...> | ABORT_<...>)
#
#   1 BUILD   fresh configure, GPU, AVX-512 Embree -> ptm_bench/build_727 (the batch binary in
#             ptm_bench/build is NOT touched: it is the benchmark's provenance)
#   2 GATES   G1 test_bowl_crater 16/16 (GPU)
#             G2 Moon GPU coupled short window, switch OFF: new binary == batch binary (byte)
#             G3 Moon CPU SIMD coupled short window, switch OFF: new == batch binary
#             G4 generic body (Justitia res5) and G5 Europa (res5), CPU: new == batch binary
#             G6 switch ON acts on ONE-LUNATION windows: armed + whole-cycle (R2) + fold (R1) lines,
#                anchor census leaves fallback-radiation-coupled, outputs DIFFER from OFF (GPU, CPU)
#             G8 R2: series on the 30-step sub-cycle window REFUSES
#             G7 PTM-728 measurement (non-blocking): CPU SIMD vs GPU, same config, OFF and series
#   3 REPLAY  replay_spun_state.py 0.93 60 (CPU, ~70 min) -- must reproduce Fermi's replay_f093
#   4 DONOR   one output step ending at the control start, look-ahead 1 -> donor_state.bin
#   5 RUNS    ctrlA (raw donor, DIAGNOSTIC only) and fixON on the GPU while the replay runs on the CPU
#   6 INJECT  asrun_state.bin = donor + replay AS-RUN post-eq columns (A'), spun_state.bin = donor + spun-up columns (B); run A' then B
#   7 VERDICT verdict.py (Fermi's bar on B - A', own AND fixed set must agree; fix acceptance |R - B| <= 0.5 K)
set -u
C=/Users/paha3326/projects/moon3d_R1/runs/ptm727_control
WT=/Users/paha3326/projects/moon3d_R1/ptm_bench
NEWB=$WT/build_727
OLDB=$WT/build
PY=/Users/paha3326/anaconda3/envs/europa_viz/bin/python3
G=/Library/Developer/CommandLineTools/usr/bin/git
BATCH_ST=/Users/paha3326/projects/moon3d_R1/bench/runs/BATCH_STATUS.txt
ST=$C/CONTROL_STATUS.txt
VF_V4=/Users/paha3326/projects/moon3d/vf/resweep_v4_2026-08-24/moon3d509k/moon3d_c_cartesian_NEW_500k_clean_v4.csr
ENVS=(PTM_ILLUM_TRUE_NORMALS=1 PTM_ILLUM_RISE_EPS=1 PTM_ILLUM_RISE_KEPS=0.2 PTM_ILLUM_RISE_HABS=0
      PTM_ILLUM_SEAL_HOLES=1 PTM_RAD_RERUN_MAX_GPUS=1 PTM_HMATRIX_VF_REPLACE=$VF_V4)
status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" >> "$ST"; }
die() { status "ABORT_$1" "${2:-}"; exit 1; }
status CHAIN_STARTED "pid=$$"

# ---- 0. gate: done by wait_and_run.sh, which execs THIS file only after the gate opens ----
[ "${PTM727_GATE_OPEN:-0}" = 1 ] || { echo "run via wait_and_run.sh (gate not checked)" >&2; exit 2; }

# ---- 1. build ---------------------------------------------------------------------------
SHA=$($G -C $WT rev-parse --short HEAD); DIRTY=$($G -C $WT status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$DIRTY" = 0 ] || die build_dirty_tree "$DIRTY modified tracked files at $SHA; commit first"
WT=$WT /Users/paha3326/projects/moon3d_R1/bench/build_logs/build_gpu.sh $NEWB > $C/build.out 2>&1 || die build "see $NEWB/build.log"
grep -q '"AVX512"' $NEWB/include/ptm_build_info.h || die build_isa "not AVX512"
status BUILD_OK "sha=$SHA exe_md5=$(md5 -q $NEWB/bin/ptm_thermal_sim) libthermal=$(md5 -q $NEWB/src/cpp/thermal/libthermal.dylib)"

# ---- 2. gates ---------------------------------------------------------------------------
mkdir -p $C/gates/bowl && ( cd $C/gates/bowl && $NEWB/bin/test_bowl_crater > bowl.log 2>&1 )
grep -q "PASS: 16 of 16" $C/gates/bowl/bowl.log || die G1_bowl "$(grep -E 'PASS:|FAIL' $C/gates/bowl/bowl.log | tail -1)"
status G1_OK "test_bowl_crater 16/16"
# G2..G6 configs and comparisons live in gates.py (one place, byte comparisons excluding run metadata)
$PY $C/gates.py "$OLDB" "$NEWB" > $C/gates/gates.log 2>&1 || die G2_G6 "$(tail -3 $C/gates/gates.log | tr '\n' ' ')"
status GATES_OK "$(grep '^GATE' $C/gates/gates.log | tr '\n' ' ')"

# ---- 3. replay (CPU) in the background while the GPU runs the donor ---------------------
( cd $C && $PY $C/replay_spun_state.py 0.93 60 f093 > $C/replay.log 2>&1 ) &
RPID=$!

# ---- 4. donor ---------------------------------------------------------------------------
run() {  # name
  ( cd $WT && env "${ENVS[@]}" /usr/bin/time -l $NEWB/bin/ptm_thermal_sim --config $C/configs/$1.yaml > $C/$1.log 2> $C/$1.time.txt )
}
run donor_dt65 || die donor "rc=$?"
[ -f $C/donor_state.bin ] || die donor_state "no donor_state.bin"
status DONOR_OK "$(ls -la $C/donor_state.bin | awk '{print $5}') bytes"

# ---- 5. control runs that do not need the replay (GPU), replay still running on CPU -----
check_run() {  # name
  [ -f $C/$1/statistics/temperature_avg.bin ] || die stats_$1 "no statistics"
  case $1 in ctrl*) grep -q "identity trailer verified" $C/$1.log || die warmstart_$1 "state not restored with a verified trailer";; esac
  case $1 in fixON*) grep -q "seed_indirect: series armed" $C/$1.log || die fix_not_armed "fixON log lacks the armed line";; esac
  status RUN_OK "$1"
}
for r in ctrlA_dt65_donor fixON_dt65; do run $r || die run_$r "rc=$?"; check_run $r; done

# ---- 6. replay -> inject -> control B -------------------------------------------------------
wait $RPID || die replay "see replay.log"
$PY - <<EOF || die replay_mismatch "my replay does not reproduce Fermi's replay_f093"
import numpy as np
a=np.load('$C/replay_f093.npz'); b=np.load('/Users/paha3326/projects/moon3d_R1/runs/stage3_f2/fermi/replay_f093.npz')
bad=[k for k in b.files if not np.allclose(a[k],b[k],rtol=1e-9,atol=1e-9)]
print('replay reproduction: differing keys', bad); raise SystemExit(1 if bad else 0)
EOF
status REPLAY_OK "reproduces replay_f093.npz"
$PY $C/inject_state.py $C/donor_state.bin $C/asrun_posteq_f093.npz $C/asrun_state.bin > $C/inject_asrun.log 2>&1 || die inject_asrun "see inject_asrun.log"
$PY $C/inject_state.py $C/donor_state.bin $C/spun_L1start_f093.npz $C/spun_state.bin > $C/inject.log 2>&1 || die inject "see inject.log"
status INJECT_OK "A': $(grep 'patched' $C/inject_asrun.log) | B: $(grep 'patched' $C/inject.log)"

# A' (Fermi C1: the unbiased reference) and B differ ONLY in the injected columns
for r in ctrlAp_dt65_asrun ctrlB_dt65_spun; do run $r || die run_$r "rc=$?"; check_run $r; done

# ---- 7. verdict -----------------------------------------------------------------------------
$PY $C/verdict.py > $C/verdict.log 2>&1 || die verdict "see verdict.log"
status "VERDICT_$($PY -c "import json;print(json.load(open('$C/verdict.json'))['verdict'])")" "$(grep -E '^B - A|^fix:' $C/verdict.log | tr '\n' ' ')"
