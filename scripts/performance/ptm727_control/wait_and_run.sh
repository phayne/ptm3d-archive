#!/bin/bash
# PTM-727 control LAUNCHER (Ada, 2026-10-14). Holds ONLY the gate, then execs the tracked chain
# body (run_control_chain.sh, same directory). Launch DETACHED, never from a shell that can reap it:
#   S=~/projects/moon3d_R1/ptm_bench/scripts/performance/ptm727_control
#   C=~/projects/moon3d_R1/runs/ptm727_control
#   nohup caffeinate -i bash $S/wait_and_run.sh >> $C/chain_nohup.log 2>&1 < /dev/null & disown
#
# The WHOLE script is one { ... } compound command, so bash parses it completely before running
# it: editing this file on disk while it waits cannot corrupt the running launcher.
#
# Gate (all must hold on two consecutive 60 s polls):
#   - the R1 benchmark batch is terminal: BATCH_STATUS.txt has BATCH_DONE / ABORT_* / GUARD_STOP_disk
#   - FERMI_APPROVED exists in the run directory. Its first line should read "APPROVED <sha>"
#     (the first hex token of 7-40 chars is taken as the sha).
#     If <sha> is not the worktree HEAD, the approval is STALE (it reviewed other code): logged
#     once per version of the file as APPROVAL_STALE, and the gate stays CLOSED until Fermi
#     re-pins to HEAD. It never runs unreviewed code and it never silently waits.
#     No sha at all: APPROVAL_UNPINNED is logged and HEAD at gate-open is what runs.
#   - no ptm_thermal_sim and no run_bench.py process is running
# Liveness: CONTROL_HEARTBEAT.txt is rewritten every poll; a stale heartbeat means a dead launcher.
# A catchable signal writes LAUNCHER_KILLED_<sig> to CONTROL_STATUS.txt (SIGKILL cannot be caught:
# the heartbeat is the backstop).
{
set -u
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
C=/Users/paha3326/projects/moon3d_R1/runs/ptm727_control
WT=/Users/paha3326/projects/moon3d_R1/ptm_bench
G=/Library/Developer/CommandLineTools/usr/bin/git
BATCH_ST=/Users/paha3326/projects/moon3d_R1/bench/runs/BATCH_STATUS.txt
ST=$C/CONTROL_STATUS.txt
HB=$C/CONTROL_HEARTBEAT.txt
mkdir -p $C
status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" >> "$ST"; }
for sig in TERM HUP INT QUIT USR1 USR2; do
  trap "status LAUNCHER_KILLED_$sig 'pid=$$ (signal $sig while waiting at the gate)'; rm -f $HB; exit 143" $sig
done
status LAUNCHER_STARTED "pid=$$ script=$S/wait_and_run.sh head=$($G -C $WT rev-parse --short HEAD)"

approval_sha() {  # prints the full SHA named on FERMI_APPROVED's first line, or nothing
  local tok
  tok=$(head -1 "$C/FERMI_APPROVED" 2>/dev/null | grep -oE '\b[0-9a-f]{7,40}\b' | head -1)
  [ -n "$tok" ] && $G -C $WT rev-parse --verify --quiet "$tok^{commit}" 2>/dev/null
}

n=0; quiet=0; seen_approval=
while true; do
  bt=0; grep -Eq $'\t(BATCH_DONE|ABORT_[A-Za-z_]*|GUARD_STOP_disk)\t' "$BATCH_ST" 2>/dev/null && bt=1
  fa=0
  if [ -f $C/FERMI_APPROVED ]; then
    fa=1
    asha=$(approval_sha); head=$($G -C $WT rev-parse HEAD)
    ver=$(md5 -q $C/FERMI_APPROVED)
    if [ -n "$asha" ] && [ "$asha" != "$head" ]; then
      fa=stale
      [ "$seen_approval" = "$ver" ] || status APPROVAL_STALE "FERMI_APPROVED names ${asha:0:9}, worktree HEAD is ${head:0:9}: gate CLOSED until Fermi re-pins (APPROVED ${head:0:9})"
    elif [ -z "$asha" ]; then
      [ "$seen_approval" = "$ver" ] || status APPROVAL_UNPINNED "FERMI_APPROVED names no commit; HEAD ${head:0:9} at gate-open is what will run"
    elif [ "$seen_approval" != "$ver" ]; then
      status APPROVAL_PINNED "FERMI_APPROVED names ${asha:0:9} == HEAD"
    fi
    seen_approval=$ver
  fi
  sims=$(pgrep -x ptm_thermal_sim | tr '\n' ' '); bench=$(pgrep -f run_bench.py | tr '\n' ' ')
  if [ $bt = 1 ] && [ $fa = 1 ] && [ -z "$sims" ] && [ -z "$bench" ]; then quiet=$((quiet+1)); else quiet=0; fi
  printf '%s\tpid=%s\tbatch_terminal=%s\tfermi_approved=%s\tptm_thermal_sim=[%s]\trun_bench=[%s]\n' \
    "$(date -u +%FT%TZ)" $$ $bt $fa "$sims" "$bench" > $HB.tmp && mv -f $HB.tmp $HB
  [ $quiet -ge 2 ] && break
  [ $((n % 15)) = 0 ] && status WAITING "batch_terminal=$bt fermi_approved=$fa ptm_thermal_sim=[$sims]"
  n=$((n+1)); sleep 60 & wait $!
done
status GATE_OPEN "batch: $(grep -E $'\t(BATCH_DONE|ABORT_|GUARD_STOP)' $BATCH_ST | tail -1 | cut -f2) fermi: $(head -1 $C/FERMI_APPROVED | cut -c1-120)"
rm -f $HB
for sig in TERM HUP INT QUIT USR1 USR2; do trap - $sig; done
export PTM727_GATE_OPEN=1
exec bash $S/run_control_chain.sh
}
