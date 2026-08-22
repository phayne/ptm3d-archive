#!/usr/bin/env python3
"""Measure the PEAK resident set size of a PTM3D run (or any child process).

Why this exists
---------------
Memory claims about the engine ("this refactor drops N GB of host transient")
are only believable if somebody measured the peak, and a peak is easy to MISS:
a transient that lives for four seconds is invisible to a sampler polling every
five. This tool polls fast (50 ms by default), records every sample, and can
stop the child as soon as the phase under test has provably happened -- so a
load-phase measurement does not have to sit through the whole simulation.

It samples the child AND every descendant, because ptm_thermal_sim's chunked
runner and merge steps are separate processes.

Usage
-----
    ptm_peak_rss.py --json out.json -- ptm_thermal_sim config.yaml

    # stop 10 s after the run's stdout first matches a regex
    ptm_peak_rss.py --json out.json \
        --stop-on 'ROUTE = (GPU|CPU)' --stop-delay 10 \
        -- ptm_thermal_sim config.yaml

Exit status is the child's, except when --stop-on fired (then 0 with
"stopped_on_marker": true in the JSON, so a caller cannot mistake a deliberate
early stop for a clean completion).
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time


def descendants(root_pid):
    """Every pid in root_pid's process tree, root included.

    One `ps` call, then a walk -- cheaper than the recursive `pgrep -P` chain,
    which at a 50 ms cadence costs more than the thing being measured.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,rss="],
                             capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    children = {}
    rss = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, kb = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb
    tree, stack = {}, [root_pid]
    while stack:
        p = stack.pop()
        if p in tree:
            continue
        if p in rss:
            tree[p] = rss[p]
        stack.extend(children.get(p, []))
    return tree


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write the sample summary here")
    ap.add_argument("--log", help="tee the child's stdout+stderr here")
    ap.add_argument("--interval", type=float, default=0.05,
                    help="sampling interval in seconds (default 0.05)")
    ap.add_argument("--stop-on", default=None,
                    help="regex; once the child's output matches, stop it after --stop-delay")
    ap.add_argument("--stop-delay", type=float, default=10.0,
                    help="seconds to keep sampling after --stop-on matches (default 10)")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="hard wall-clock cap in seconds (0 = none)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- followed by the command to run")
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given (put it after `--`)")

    marker = re.compile(args.stop_on) if args.stop_on else None
    logf = open(args.log, "w") if args.log else None

    t0 = time.time()
    # Own a process GROUP, so the stop paths below can kill the whole tree. The
    # thing being measured is usually a runner script whose ptm_thermal_sim
    # GRANDCHILD holds the GPU and the tens of GB; killing only the direct child
    # would orphan it with no handle left to reap it.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)

    # The output reader has to be non-blocking relative to the sampler, or a
    # quiet child would freeze sampling for minutes. A thread is the simplest
    # thing that cannot deadlock on a full pipe buffer either.
    import threading
    state = {"marker_at": None}

    def pump():
        for line in proc.stdout:
            if logf:
                logf.write(line)
                logf.flush()
            if marker and state["marker_at"] is None and marker.search(line):
                state["marker_at"] = time.time()

    th = threading.Thread(target=pump, daemon=True)
    th.start()

    samples = []          # (t_since_start, total_rss_bytes, n_procs)
    peak_kb, peak_t, peak_n = 0, 0.0, 0
    stopped_on_marker = False

    while True:
        tree = descendants(proc.pid)
        total_kb = sum(tree.values())
        now = time.time() - t0
        if tree:
            samples.append((round(now, 3), total_kb * 1024, len(tree)))
            if total_kb > peak_kb:
                peak_kb, peak_t, peak_n = total_kb, now, len(tree)

        if proc.poll() is not None:
            break
        if (state["marker_at"] is not None
                and time.time() - state["marker_at"] >= args.stop_delay):
            stopped_on_marker = True
            break
        if args.timeout and (time.time() - t0) > args.timeout:
            break
        time.sleep(args.interval)

    rc = proc.poll()
    if rc is None:
        # SIGKILL the whole process GROUP, not just the child: SIGTERM is no good
        # because ptm_thermal_sim installs handlers that can take a while to
        # unwind a multi-GB run, and pid-only would leave the grandchildren
        # running. We already took the sample we came for.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        proc.wait()
        rc = proc.returncode

    # Join the pump BEFORE closing the log: the child may still have been
    # writing when we broke out of the sampling loop, and closing under it
    # either raises out of the daemon thread or truncates the tail of the very
    # log the measurement is justified from.
    th.join(timeout=10)
    if logf:
        logf.close()

    result = {
        "cmd": cmd,
        "peak_rss_bytes": peak_kb * 1024,
        "peak_rss_gib": round(peak_kb * 1024 / (1024 ** 3), 3),
        "peak_at_s": round(peak_t, 3),
        "peak_n_procs": peak_n,
        "n_samples": len(samples),
        "interval_s": args.interval,
        "wall_s": round(time.time() - t0, 3),
        "stopped_on_marker": stopped_on_marker,
        "returncode": rc,
        "samples": samples,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    sys.stderr.write(
        "peak RSS %.3f GiB at t=%.1f s over %d samples (%.0f ms cadence)%s\n"
        % (result["peak_rss_gib"], peak_t, len(samples), args.interval * 1e3,
           "  [stopped on marker]" if stopped_on_marker else ""))
    return 0 if stopped_on_marker else (rc or 0)


if __name__ == "__main__":
    sys.exit(main())
