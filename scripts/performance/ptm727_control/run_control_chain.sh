#!/bin/bash
# PTM-727 fix: post-batch build, gates and the DECISIVE CONTROL (Ada, 2026-10-14; Paul-approved).
# Do NOT launch directly: wait_and_run.sh holds the gate and execs this body when it opens.
# Code + configs: this directory (tracked in ptm_bench, so the build SHA covers them).
# Outputs:        ~/projects/moon3d_R1/runs/ptm727_control
#
# STATUS: CONTROL_STATUS.txt. The LAST line is terminal and is exactly one of
#   VERDICT_CONFIRMED | VERDICT_REFUTED | VERDICT_INCONCLUSIVE   (numbers: verdict.json)
#   ABORT_<reason>                                                (gate, build, run or signal)
# Every step is deterministic: nothing is retried; the first failure ends the chain.
#
#   1 BUILD   HEAD (must be clean; FERMI_APPROVED's sha if pinned) -> ptm_bench/build_727: fresh
#             configure, GPU (OpenCL), Embree 4.3.3 AVX-512, conda-free env (PTM-718). Checks: ISA
#             AVX512 and no conda path in any load command / rpath of the exe or its dylibs.
#   2 GATES   G1 test_bowl_crater 16/16 (GPU)
#             gates.py, FAIL-FAST in this order (the ABORT reason names the gate):
#             G2 Moon GPU coupled short window, switch OFF: new binary == batch binary (byte)
#             G3 Moon CPU SIMD coupled short window, switch OFF: new == batch binary
#             G4 generic body (Justitia res5) and G5 Europa (res5), CPU: new == batch binary
#             G6 switch ON acts on ONE-LUNATION windows (GPU series, CPU series, GPU mean):
#                armed + whole-cycle (R2) + fold (R1) lines, anchor census leaves
#                fallback-radiation-coupled, outputs DIFFER from OFF
#             G8 R2: series on the 30-step sub-cycle window REFUSES
#             G7 PTM-728 measurement (non-blocking): CPU SIMD vs GPU, same config, OFF and series
#             -> FIX_BINARY.txt (path, SHA, exe + every dylib md5) is written ONLY after all pass
#   3 REPLAY  replay_spun_state.py 0.93 60 f093 (CPU, ~69 min measured) in the background
#   4 DONOR   one output step ending at the control start, look-ahead 1 -> donor_state.bin
#   5 RUNS    ctrlA (raw donor, DIAGNOSTIC only) and fixON on the GPU while the replay runs
#   6 INJECT  asrun_state.bin = donor + replay AS-RUN post-eq columns (A', Fermi C1),
#             spun_state.bin = donor + spun-up columns (B); run A' then B
#   7 VERDICT verdict.py: Fermi's bar on B - A', own AND fixed set must agree (C2);
#             fix acceptance |R - B| <= 0.5 K on the fixed set (C3)
{
set -u
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
C=/Users/paha3326/projects/moon3d_R1/runs/ptm727_control
WT=/Users/paha3326/projects/moon3d_R1/ptm_bench
NEWB=$WT/build_727
OLDB=$WT/build
PY=/Users/paha3326/anaconda3/envs/europa_viz/bin/python3
G=/Library/Developer/CommandLineTools/usr/bin/git
ST=$C/CONTROL_STATUS.txt
FIXB=$C/FIX_BINARY.txt
ASRUN_LOG=/Users/paha3326/projects/moon3d_R1/runs/stage3/dt65_coupled.log
VF_V4=/Users/paha3326/projects/moon3d/vf/resweep_v4_2026-08-24/moon3d509k/moon3d_c_cartesian_NEW_500k_clean_v4.csr
ENVS=(PTM_ILLUM_TRUE_NORMALS=1 PTM_ILLUM_RISE_EPS=1 PTM_ILLUM_RISE_KEPS=0.2 PTM_ILLUM_RISE_HABS=0
      PTM_ILLUM_SEAL_HOLES=1 PTM_RAD_RERUN_MAX_GPUS=1 PTM_HMATRIX_VF_REPLACE=$VF_V4)
# ---- 0. gate: done by wait_and_run.sh, which execs THIS file only after the gate opens ----
[ "${PTM727_GATE_OPEN:-0}" = 1 ] || { echo "run via wait_and_run.sh (gate not checked)" >&2; exit 2; }
TERMINAL=0; RPID=
status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" >> "$ST"; }
die() { TERMINAL=1; status "ABORT_$1" "${2:-}"; exit 1; }
on_exit() {
  [ -n "$RPID" ] && kill "$RPID" 2>/dev/null
  [ $TERMINAL = 1 ] || status ABORT_unexpected_exit "chain pid $$ exited without a terminal state (rc=$1)"
}
trap 'on_exit $?' EXIT
for sig in TERM HUP INT QUIT; do trap "die signal_$sig 'chain pid $$ received SIG$sig'" $sig; done
status CHAIN_STARTED "pid=$$ script=$S/run_control_chain.sh"

# ---- 1. build ---------------------------------------------------------------------------
SHA=$($G -C $WT rev-parse HEAD); SHORT=${SHA:0:9}; BRANCH=$($G -C $WT branch --show-current)
DIRTY=$($G -C $WT status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$DIRTY" = 0 ] || die build_dirty_tree "$DIRTY modified tracked files at $SHORT; commit first"
ASHA=$(head -1 $C/FERMI_APPROVED 2>/dev/null | grep -oE '\b[0-9a-f]{7,40}\b' | head -1)
if [ -n "$ASHA" ]; then
  ASHA=$($G -C $WT rev-parse --verify --quiet "$ASHA^{commit}") || die approval_sha_unknown "FERMI_APPROVED sha not in the worktree"
  [ "$ASHA" = "$SHA" ] || die approval_sha_mismatch "approved ${ASHA:0:9}, HEAD $SHORT"
fi
status BUILD_START "sha=$SHORT branch=$BRANCH -> $NEWB"
T0=$(date +%s)
WT=$WT /Users/paha3326/projects/moon3d_R1/bench/build_logs/build_gpu.sh $NEWB > $C/build.out 2>&1 || die build "see $NEWB/build.log"
grep -q '"AVX512"' $NEWB/include/ptm_build_info.h || die build_isa "Embree max ISA is not AVX512"
LIBS=$(find $NEWB/src -name '*.dylib' -not -path '*/CMakeFiles/*' | sort)
for f in $NEWB/bin/ptm_thermal_sim $LIBS; do
  otool -l "$f" | grep -qiE 'anaconda|/conda' && die build_conda "conda path in load commands of $f (PTM-718)"
done
# Every @rpath library must resolve, in LC_RPATH order (what dyld does), and every PTM library
# (lib*.dylib built here) must resolve INSIDE build_727 -- never the batch's build/ and never a
# copy earlier on the rpath (e.g. /usr/local/lib). Otherwise the md5s below would describe a
# library the engine does not load.
RESOLVED=
RPATHS=$(otool -l $NEWB/bin/ptm_thermal_sim | awk '/LC_RPATH/{f=1} f&&/ path /{print $2; f=0}')
for lib in $(otool -L $NEWB/bin/ptm_thermal_sim | awk '/@rpath\//{sub("@rpath/","",$1); print $1}'); do
  hit=; for d in $RPATHS; do [ -f "$d/$lib" ] && { hit="$d/$lib"; break; }; done
  [ -n "$hit" ] || die build_link "@rpath/$lib resolves nowhere"
  RESOLVED="$RESOLVED $hit"
  if [ -n "$(find $NEWB/src -name "$lib" -not -path '*/CMakeFiles/*' | head -1)" ]; then
    case $hit in "$NEWB"/*) ;; *) die build_link "@rpath/$lib resolves to $hit, not build_727";; esac
  fi
done
EXE_MD5=$(md5 -q $NEWB/bin/ptm_thermal_sim); THERM_MD5=$(md5 -q $NEWB/src/cpp/thermal/libthermal.dylib)
status BUILD_OK "sha=$SHORT exe_md5=$EXE_MD5 libthermal=$THERM_MD5 wall_s=$(( $(date +%s) - T0 ))"

# ---- 2. gates ---------------------------------------------------------------------------
T0=$(date +%s)
mkdir -p $C/gates/bowl && ( cd $C/gates/bowl && $NEWB/bin/test_bowl_crater > bowl.log 2>&1 )
grep -q "PASS: 16 of 16" $C/gates/bowl/bowl.log || die G1_bowl "$(grep -E 'PASS:|FAIL' $C/gates/bowl/bowl.log | tail -1)"
status G1_OK "test_bowl_crater 16/16 wall_s=$(( $(date +%s) - T0 ))"
T0=$(date +%s)
if ! $PY $S/gates.py "$OLDB" "$NEWB" > $C/gates/gates.log 2>&1; then
  g=$(grep -oE '^GATE_FAILED [A-Za-z0-9_]+' $C/gates/gates.log | tail -1 | cut -d' ' -f2)
  die "${g:-gates_py}" "$(grep -E '^GATE ' $C/gates/gates.log | tail -1 | cut -c1-300)"
fi
status GATES_OK "$(grep '^GATE ' $C/gates/gates.log | cut -d: -f1 | tr '\n' ' ') wall_s=$(( $(date +%s) - T0 )); $(grep '^MEASURE' $C/gates/gates.log | tr '\n' ' ')"
status SERIES_OVERHEAD_G6 "$(grep '^TIMING_RATIO' $C/gates/gates.log | tr '\n' ' ')"

# Provenance for the re-run chain: written only now (the binary passed every gate), atomically.
{
  echo "# PTM-727 fix binary, written by run_control_chain.sh after G1-G8 passed."
  echo "# The VERDICT is NOT in this file: read the last line of CONTROL_STATUS.txt / verdict.json."
  echo "status=GATES_PASSED"
  echo "written_utc=$(date -u +%FT%TZ)"
  echo "sha=$SHA"
  echo "sha_short=$SHORT"
  echo "branch=$BRANCH"
  echo "fermi_approved_sha=${ASHA:-UNPINNED}"
  echo "worktree=$WT"
  echo "build_dir=$NEWB"
  echo "recipe=/Users/paha3326/projects/moon3d_R1/bench/build_logs/build_gpu.sh (fresh configure, GPU OpenCL, Embree 4.3.3 AVX-512, conda-free env, PTM-718 checked: no conda path in any load command)"
  echo "exe=$NEWB/bin/ptm_thermal_sim"
  echo "exe_md5=$EXE_MD5"
  echo "libthermal=$NEWB/src/cpp/thermal/libthermal.dylib"
  echo "libthermal_md5=$THERM_MD5"
  for f in $RESOLVED; do echo "loaded_dylib_md5 $(md5 -q $f) $f"; done
  echo "test_bowl_crater_md5=$(md5 -q $NEWB/bin/test_bowl_crater)"
} > $FIXB.tmp && mv -f $FIXB.tmp $FIXB || die provenance "could not write $FIXB"
status FIX_BINARY_WRITTEN "$FIXB exe_md5=$EXE_MD5 libthermal=$THERM_MD5"

# ---- 3. replay (CPU) in the background while the GPU runs the donor ---------------------
( cd $C && exec $PY $S/replay_spun_state.py 0.93 60 f093 > $C/replay.log 2>&1 ) &
RPID=$!
status REPLAY_STARTED "pid=$RPID"

# ---- 4. donor ---------------------------------------------------------------------------
run() {  # name
  local t0=$(date +%s)
  ( cd $WT && env "${ENVS[@]}" /usr/bin/time -l $NEWB/bin/ptm_thermal_sim --config $S/configs/$1.yaml > $C/$1.log 2> $C/$1.time.txt ) || return $?
  status RUN_WALL "$1 wall_s=$(( $(date +%s) - t0 )) engine_s=$(grep -oE 'Simulation completed in [0-9.]+' $C/$1.log | awk '{print $4}')"
}
run donor_dt65 || die donor "rc=$? see donor_dt65.log"
[ -f $C/donor_state.bin ] || die donor_state "no donor_state.bin"
status DONOR_OK "$(ls -la $C/donor_state.bin | awk '{print $5}') bytes"

# ---- 5. control runs that do not need the replay (GPU), replay still running on CPU -----
check_run() {  # name
  [ -f $C/$1/statistics/temperature_avg.bin ] || die stats_$1 "no statistics"
  case $1 in ctrl*) grep -q "identity trailer verified" $C/$1.log || die warmstart_$1 "state not restored with a verified trailer";; esac
  case $1 in fixON*)
    grep -q "seed_indirect: series armed" $C/$1.log || die fix_not_armed "fixON log lacks the armed line"
    grep -q "whole cycle(s)" $C/$1.log && grep -q "(R1)" $C/$1.log || die fix_no_fold "fixON log lacks the R2 whole-cycle / R1 fold line";;
  esac
  status RUN_OK "$1"
}
for r in ctrlA_dt65_donor fixON_dt65; do run $r || die run_$r "rc=$? see $r.log"; check_run $r; done
ASRUN_S=$(grep -oE 'Simulation completed in [0-9.]+' $ASRUN_LOG | awk '{print $4}')
FIXON_S=$(grep -oE 'Simulation completed in [0-9.]+' $C/fixON_dt65.log | awk '{print $4}')
status FIXON_TIMING "fixON engine_s=$FIXON_S vs as-run dt65_coupled engine_s=$ASRUN_S ratio=$(awk -v a=$FIXON_S -v b=$ASRUN_S 'BEGIN{printf "%.3f", a/b}') (fixON shared the CPU with the replay; as-run did not)"

# ---- 6. replay -> inject -> controls A' and B ---------------------------------------------
wait $RPID; rc=$?; RPID=
[ $rc = 0 ] || die replay "rc=$rc see replay.log"
$PY - <<EOF || die replay_mismatch "my replay does not reproduce Fermi's replay_f093"
import numpy as np
a=np.load('$C/replay_f093.npz'); b=np.load('/Users/paha3326/projects/moon3d_R1/runs/stage3_f2/fermi/replay_f093.npz')
missing=[k for k in b.files if k not in a.files]
bad=[k for k in b.files if k in a.files and not np.allclose(a[k],b[k],rtol=1e-9,atol=1e-9)]
print('replay reproduction: missing keys', missing, 'differing keys', bad); raise SystemExit(1 if (bad or missing) else 0)
EOF
status REPLAY_OK "reproduces replay_f093.npz"
$PY $S/inject_state.py $C/donor_state.bin $C/asrun_posteq_f093.npz $C/asrun_state.bin > $C/inject_asrun.log 2>&1 || die inject_asrun "see inject_asrun.log"
$PY $S/inject_state.py $C/donor_state.bin $C/spun_L1start_f093.npz $C/spun_state.bin > $C/inject.log 2>&1 || die inject "see inject.log"
status INJECT_OK "A': $(grep 'patched' $C/inject_asrun.log) | B: $(grep 'patched' $C/inject.log)"

# A' (Fermi C1: the unbiased reference) and B differ ONLY in the injected columns
for r in ctrlAp_dt65_asrun ctrlB_dt65_spun; do run $r || die run_$r "rc=$? see $r.log"; check_run $r; done

# ---- 7. verdict -----------------------------------------------------------------------------
$PY $S/verdict.py > $C/verdict.log 2>&1 || die verdict "$(tail -1 $C/verdict.log | cut -c1-300)"
V=$($PY -c "import json;d=json.load(open('$C/verdict.json'));print(d['verdict'])") || die verdict_json "unreadable verdict.json"
case $V in CONFIRMED|REFUTED|INCONCLUSIVE) ;; *) die verdict_value "verdict.json verdict='$V'";; esac
LINE=$($PY -c "import json;d=json.load(open('$C/verdict.json'));print('B-Ap_own_K=%+.3f B-Ap_fixed_K=%+.3f fix_R-B_fixed_K=%+.3f fix_accepted=%s rerun_with_fix_ok=%s sha=$SHORT' % (d['B_minus_Ap_own_K'],d['B_minus_Ap_fixed_K'],d['fix_R_minus_B_fixed_K'],str(d['fix_accepted']).lower(),str(d['rerun_with_fix_ok']).lower()))")
TERMINAL=1
status "VERDICT_$V" "$LINE"
exit 0
}
