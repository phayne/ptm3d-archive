#!/bin/bash
# PTM-727 control LAUNCHER (Ada, 2026-10-14). Holds ONLY the gate, so the chain body
# (run_control_chain.sh) and its Python helpers can still be edited while this waits:
# bash reads a running script by byte offset, and editing the script it is executing
# corrupts it. The body is read fresh from disk when the gate opens.
#   nohup caffeinate -i bash wait_and_run.sh > chain_nohup.log 2>&1 &
set -u
C=/Users/paha3326/projects/moon3d_R1/runs/ptm727_control
BATCH_ST=/Users/paha3326/projects/moon3d_R1/bench/runs/BATCH_STATUS.txt
ST=$C/CONTROL_STATUS.txt
status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" >> "$ST"; }
status LAUNCHER_STARTED "pid=$$"
# ---- 0. gate --------------------------------------------------------------------------
n=0; quiet=0
while true; do
  bt=0; grep -Eq $'\t(BATCH_DONE|ABORT_[A-Za-z_]*|GUARD_STOP_disk)\t' "$BATCH_ST" 2>/dev/null && bt=1
  fa=0; [ -f $C/FERMI_APPROVED ] && fa=1
  if [ $bt = 1 ] && [ $fa = 1 ] && [ -z "$(pgrep -x ptm_thermal_sim)" ] && [ -z "$(pgrep -f run_bench.py)" ]; then
    quiet=$((quiet+1)); else quiet=0; fi
  [ $quiet -ge 2 ] && break
  [ $((n % 30)) = 0 ] && status WAITING "batch_terminal=$bt fermi_approved=$fa ptm_thermal_sim=[$(pgrep -x ptm_thermal_sim | tr '\n' ' ')]"
  n=$((n+1)); sleep 60
done
status GATE_OPEN "batch: $(grep -E $'\t(BATCH_DONE|ABORT_|GUARD_STOP)' $BATCH_ST | tail -1 | cut -f2) fermi: $(head -1 $C/FERMI_APPROVED)"

export PTM727_GATE_OPEN=1
exec bash $C/run_control_chain.sh
