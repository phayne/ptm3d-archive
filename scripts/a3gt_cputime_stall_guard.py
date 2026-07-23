#!/usr/bin/env python3
"""CPU-time stall guard for the AGT unattended campaign (kill-on-soft-hang).

The June campaign stalled ~38 h with the orchestrator and its GPU sim child
BOTH alive but not computing (a soft GPU wedge). `a3gt_rad_corering_watchdog.py`
keys its stall detection on WINDOW-COMPLETION progress and leaves that
"alive-but-frozen" case (WEDGE_SUSPECT) notify-only -- it never killed anything,
so the resume-by-default machinery never got a chance to fire. This guard adds
the missing action: it watches a process's *CPU time* -- the one signal that
keeps rising while the process actually computes and FREEZES the instant it
soft-hangs -- and KILLS the process when CPU time stops advancing for longer
than a threshold. The orchestrator's resume-by-default then re-runs the window.

The stall detector itself is the already-unit-tested `StallWatchdog` (R3c) from
`ptm_run_guard.py`; this script only adds process sampling + the kill action, so
there is one stall definition across the runner, orchestrator, and this guard.

Usage:
  # monitor the live GPU sim; kill it if its CPU time freezes for 30 min:
  a3gt_cputime_stall_guard.py --pattern ptm_thermal_sim --stall-min 30 --poll-sec 60
  # observe only, never kill (safe first pass):
  a3gt_cputime_stall_guard.py --pattern ptm_thermal_sim --stall-min 30 --dry-run
  # live-fire self-test (SIGSTOP a burner, prove detect+kill; prove a busy proc survives):
  a3gt_cputime_stall_guard.py --drill

Every sample, decision, and kill is logged to stdout (line-buffered) -- a silent
guard is a useless guard.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptm_run_guard import StallWatchdog  # noqa: E402  (reuse the R3c stall detector)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[cputime-guard {_ts()}] {msg}", flush=True)


def parse_cputime(s: str) -> float:
    """Parse a `ps -o cputime` string to seconds. Handles the macOS/BSD and
    Linux forms: 'MM:SS.ss', 'HH:MM:SS', 'DD-HH:MM:SS'. Raises on anything it
    cannot parse -- never silently returns 0 (a bogus 0 would look like a stall
    and trigger a spurious kill)."""
    s = s.strip()
    if not s:
        raise ValueError("empty cputime string")
    days = 0.0
    if "-" in s:
        d, s = s.split("-", 1)
        days = float(d)
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return days * 86400.0 + h * 3600.0 + m * 60.0 + sec


def pids_for(pattern: str) -> list[int]:
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True,
                           text=True, timeout=10)
        if r.returncode != 0:
            return []
        return [int(x) for x in r.stdout.split()]
    except Exception:
        return []


def cputime_seconds(pids: list[int]) -> float | None:
    """Total CPU seconds across `pids`. None if none are alive (caller decides
    whether a vanished process is a stall or a normal exit)."""
    if not pids:
        return None
    total = 0.0
    alive = 0
    for pid in pids:
        try:
            r = subprocess.run(["ps", "-o", "cputime=", "-p", str(pid)],
                               capture_output=True, text=True, timeout=10)
            out = r.stdout.strip()
            if out:
                total += parse_cputime(out)
                alive += 1
        except Exception:
            continue
    return total if alive else None


def kill_pids(pids: list[int], dry_run: bool) -> None:
    for pid in pids:
        try:
            if dry_run:
                log(f"DRY-RUN would SIGKILL pid {pid}")
                continue
            # SIGCONT first so a SIGSTOPped/soft-hung process is resumed enough
            # to receive and act on the kill promptly, then SIGKILL.
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            os.kill(pid, signal.SIGKILL)
            log(f"SIGKILL sent to pid {pid}")
        except ProcessLookupError:
            log(f"pid {pid} already gone")
        except Exception as e:  # pragma: no cover
            log(f"kill pid {pid} FAILED: {e}")


def monitor(pattern: str, stall_limit_s: float, poll_s: float,
            dry_run: bool, max_ticks: int | None = None) -> int:
    """Watch `pattern`'s aggregate CPU time; kill on freeze > stall_limit_s.
    Returns 0 if it killed a stalled process, 1 if the process exited on its
    own, 2 if max_ticks elapsed with the process healthy (used by the drill)."""
    wd = StallWatchdog(stall_limit_s)
    log(f"monitoring pattern={pattern!r}  stall_limit={stall_limit_s:.0f}s  "
        f"poll={poll_s:.0f}s  dry_run={dry_run}")
    ticks = 0
    while True:
        pids = pids_for(pattern)
        cpu = cputime_seconds(pids)
        if cpu is None:
            log(f"no process matches {pattern!r} -- exited (nothing to kill)")
            return 1
        advanced = wd.update(cpu)
        quiet = wd.seconds_since_progress()
        log(f"pids={pids} cputime={cpu:.2f}s "
            f"{'advanced' if advanced else 'FROZEN'} quiet={quiet:.0f}s")
        if wd.is_stalled():
            log(f"STALL: CPU time frozen {quiet:.0f}s >= {stall_limit_s:.0f}s "
                f"-- KILLING {pids} (soft hang; orchestrator resume re-runs it)")
            kill_pids(pids, dry_run)
            return 0
        ticks += 1
        if max_ticks is not None and ticks >= max_ticks:
            return 2
        time.sleep(poll_s)


def drill() -> int:
    """Live-fire self-test. Spawns a CPU burner and proves:
      (A) while it computes (CPU time advancing), the guard does NOT kill it;
      (B) once SIGSTOPped (CPU time frozen), the guard detects the freeze and
          kills it within ~stall_limit.
    Exits 0 only if BOTH hold."""
    stall_limit = 2.0
    poll = 0.4
    burner = subprocess.Popen(
        [sys.executable, "-c", "x=0\nwhile True: x+=1"],
    )
    pid = burner.pid

    # We monitor THIS pid directly (not a name pattern) to keep the drill
    # hermetic -- no risk of matching an unrelated process.
    def pid_cpu() -> float | None:
        return cputime_seconds([pid])

    try:
        log(f"DRILL: burner pid={pid} launched (busy loop)")
        # ---- Phase A: computing -> must NOT be killed ----
        wd = StallWatchdog(stall_limit)
        healthy_ticks = 0
        for _ in range(8):  # ~3.2 s of computing
            cpu = pid_cpu()
            if cpu is None:
                log("DRILL FAIL: burner died during compute phase")
                return 1
            adv = wd.update(cpu)
            log(f"  [A compute] cputime={cpu:.2f}s {'advanced' if adv else 'FROZEN'} "
                f"quiet={wd.seconds_since_progress():.1f}s stalled={wd.is_stalled()}")
            if wd.is_stalled():
                log("DRILL FAIL: guard flagged a busy (computing) process as stalled")
                return 1
            healthy_ticks += 1
            time.sleep(poll)
        log(f"DRILL Phase A PASS: {healthy_ticks} ticks, busy process never flagged stalled")

        # ---- Phase B: SIGSTOP -> CPU freezes -> must be detected + killed ----
        os.kill(pid, signal.SIGSTOP)
        log(f"DRILL: SIGSTOP sent to pid {pid} (CPU time now frozen)")
        wd = StallWatchdog(stall_limit)
        killed = False
        t_start = time.time()
        for _ in range(20):  # up to ~8 s, well past stall_limit
            cpu = pid_cpu()
            if cpu is None:
                log("DRILL: process vanished unexpectedly during freeze phase")
                break
            adv = wd.update(cpu)
            log(f"  [B frozen] cputime={cpu:.2f}s {'advanced' if adv else 'FROZEN'} "
                f"quiet={wd.seconds_since_progress():.1f}s stalled={wd.is_stalled()}")
            if wd.is_stalled():
                dt = time.time() - t_start
                log(f"DRILL: stall detected after {dt:.1f}s -> issuing kill")
                kill_pids([pid], dry_run=False)
                killed = True
                break
            time.sleep(poll)
        if not killed:
            log("DRILL FAIL: frozen process was NOT detected as stalled")
            return 1
        # confirm it is actually dead
        time.sleep(0.5)
        try:
            burner.wait(timeout=5)
        except Exception:
            pass
        gone = cputime_seconds([pid]) is None
        log(f"DRILL Phase B {'PASS' if gone else 'FAIL'}: frozen process "
            f"{'killed and reaped' if gone else 'STILL ALIVE after kill'}")
        return 0 if gone else 1
    finally:
        # never leak the burner
        try:
            os.kill(pid, signal.SIGCONT)
        except Exception:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            burner.wait(timeout=3)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drill", action="store_true",
                    help="run the SIGSTOP live-fire self-test and exit")
    ap.add_argument("--pattern", help="pgrep -f pattern of the process to guard")
    ap.add_argument("--stall-min", type=float, default=30.0,
                    help="kill if CPU time freezes for this many minutes (default 30)")
    ap.add_argument("--poll-sec", type=float, default=60.0,
                    help="sample interval in seconds (default 60)")
    ap.add_argument("--dry-run", action="store_true",
                    help="log the kill decision but do not actually kill")
    a = ap.parse_args()

    if a.drill:
        rc = drill()
        print(f"DRILL RESULT: {'PASS' if rc == 0 else 'FAIL'} (rc={rc})", flush=True)
        return rc
    if not a.pattern:
        ap.error("--pattern is required unless --drill is given")
    if a.stall_min <= 0:
        ap.error("--stall-min must be > 0")
    return monitor(a.pattern, a.stall_min * 60.0, a.poll_sec, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
