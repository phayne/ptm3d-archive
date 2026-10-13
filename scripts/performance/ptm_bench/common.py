"""Shared plumbing for the Moon3D R1 benchmark harness (changes C2/C3).

Provenance, the machine-quiet gate, the resumable run ledger, and one
subprocess runner that stamps every output line with harness wall clock and
captures peak RSS through /usr/bin/time -l. Kept deliberately small: this is
what R1 needs, not a general framework.

Requires numpy and PyYAML (e.g. ~/anaconda3/envs/europa_viz/bin/python3).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GIT = "/Library/Developer/CommandLineTools/usr/bin/git"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_state(repo=REPO):
    def g(*args):
        r = subprocess.run([GIT, "-C", str(repo), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"sha": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len((g("status", "--porcelain", "--untracked-files=no") or "").splitlines())}


def linked_libraries(binary):
    """Every dylib the binary loads that is NOT a system library, with md5. The
    engine links libthermal / libillumination_model / ... dynamically, so the exe
    md5 alone does not identify the engine (a thermal-only change moves only the
    dylib). Embree is recorded the same way."""
    out = []
    if sys.platform != "darwin":
        return out
    r = subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True)
    rpaths = []
    rl = subprocess.run(["otool", "-l", str(binary)], capture_output=True, text=True).stdout
    for m in re.finditer(r"cmd LC_RPATH\s+cmdsize \d+\s+path (\S+)", rl):
        rpaths.append(m.group(1).replace("@loader_path", str(Path(binary).parent))
                      .replace("@executable_path", str(Path(binary).parent)))
    for ln in r.stdout.splitlines()[1:]:
        p = ln.strip().split(" (")[0]
        if p.startswith("/usr/lib") or p.startswith("/System"):
            continue
        cands = [p]
        if p.startswith("@rpath/"):
            cands = [str(Path(rp) / p[len("@rpath/"):]) for rp in rpaths]
        real = next((c for c in cands if os.path.exists(c)), None)
        out.append({"name": p, "path": real, "md5": md5(real) if real else None})
    return out


def binary_provenance(binary):
    b = Path(binary).resolve()
    return {"path": str(b), "md5": md5(b), "libs": linked_libraries(b)}


def machine_state():
    """What else is running. The benchmark is worthless on a shared machine,
    so the caller refuses to time anything unless this is quiet (see
    require_quiet)."""
    load = os.getloadavg()
    # CPU in use by OTHER processes right now: the sum of ps %cpu over every process except
    # this harness. A finished row's processes are gone from ps, so (unlike the 1-min load
    # average, which stays near 48 for minutes after a 48-way row) this measures only
    # what someone else is running.
    me = os.getpid()
    ps = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,%cpu=,comm="], capture_output=True, text=True).stdout
    foreign, top = 0.0, []
    for ln in ps.splitlines():
        f = ln.split(None, 3)
        if len(f) < 4 or int(f[0]) == me or int(f[1]) == me:
            continue
        c = float(f[2])
        foreign += c
        if c >= 50:
            top.append((c, f[3][-60:]))
    r = subprocess.run(["pgrep", "-x", "ptm_thermal_sim"], capture_output=True, text=True)
    pids = [int(x) for x in r.stdout.split()]
    r2 = subprocess.run(["pgrep", "-f", "mc_ref_pp"], capture_output=True, text=True)
    return {"loadavg": load, "ncpu": os.cpu_count(), "ptm_thermal_sim_pids": pids,
            "foreign_cpu_pct": foreign, "foreign_top": sorted(top, reverse=True)[:5],
            "mc_ref_pp_pids": [int(x) for x in r2.stdout.split()],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def require_quiet(max_cpu_pct=200.0, allow_shared=False, wait_min=20.0, poll_s=15.0):
    """Wait until the machine is quiet, then return its state. Quiet = no
    ptm_thermal_sim / mc_ref_pp process (the harness's own have exited by the
    time this is called) and other processes using <= max_cpu_pct CPU in total
    (ps %cpu; 100 = one core). Not the load average: the harness's own previous
    48-way row keeps that near 48 for minutes and would idle the batch for ~9 h
    over 180 rows. A busy machine is WAITED on (up to wait_min). Still busy after that -> FATAL, unless --allow-shared, which
    marks every record shared_machine=true."""
    deadline = time.time() + 60.0 * wait_min
    while True:
        st = machine_state()
        busy = bool(st["ptm_thermal_sim_pids"] or st["mc_ref_pp_pids"]
                    or st["foreign_cpu_pct"] > max_cpu_pct)
        if not busy or allow_shared or time.time() > deadline:
            break
        time.sleep(poll_s)
    if busy and not allow_shared:
        raise SystemExit(
            "FATAL: machine is not quiet after %.0f min (ptm_thermal_sim pids %s, mc_ref_pp "
            "pids %s, other processes %.0f %% CPU > %.0f %%; top %s). Timings on a shared machine are worthless; the "
            "R1 batch runs only after Stage 1 and Stage 3. --allow-shared overrides for SMOKE "
            "tests only, and every record it writes is marked shared_machine=true."
            % (wait_min, st["ptm_thermal_sim_pids"], st["mc_ref_pp_pids"], st["foreign_cpu_pct"],
               max_cpu_pct, st["foreign_top"]))
    st["shared_machine"] = bool(busy)
    return st


class Ledger:
    """Append-only JSONL. A run is done iff a 'done' record with its id exists;
    resume skips done runs and re-runs anything else (a crashed run leaves
    only a 'start' record)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self):
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass   # a torn final line from a killed harness is not fatal
        return out

    def done(self, run_id):
        return any(r.get("id") == run_id and r.get("event") == "done" for r in self.records())

    def write(self, **rec):
        rec.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


TIME_RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size")


def run_stamped(cmd, log_path, env=None, cwd=None, timeout=None):
    """Run cmd under /usr/bin/time -l; every stdout/stderr line is written to
    log_path prefixed with harness wall seconds since launch. Returns
    (rc, wall_s, peak_rss_bytes). The prefix never enters the engine's own
    output files, only this log."""
    full = ["/usr/bin/time", "-l", *map(str, cmd)]
    t0 = time.time()
    rss = None
    with open(log_path, "w", errors="replace") as lf:
        lf.write("# cmd: %s\n" % " ".join(map(str, full)))
        p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=env, cwd=cwd, text=True, errors="replace", bufsize=1)
        try:
            for ln in p.stdout:
                lf.write("%10.3f %s" % (time.time() - t0, ln))
                m = TIME_RSS_RE.match(ln)
                if m:
                    rss = int(m.group(1))   # bytes on macOS
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -9
    return rc, time.time() - t0, rss


def which_python_has(*mods):
    import importlib.util as u
    missing = [m for m in mods if u.find_spec(m) is None]
    if missing:
        raise SystemExit(f"FATAL: this script needs {missing}; run it with a Python that has "
                         "them (e.g. ~/anaconda3/envs/europa_viz/bin/python3).")

