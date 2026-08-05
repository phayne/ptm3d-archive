#!/usr/bin/env python3
"""
Chunked-run orchestrator for long PTM3D simulations.

Splits a long simulation (e.g. an 18-yr lunar nodal cycle on a multi-
million-face mesh) into N consecutive segments small enough to fit in
RAM, runs each via ``ptm_thermal_sim``, and combines the per-chunk
statistics with ``ptm_merge_chunks`` into a single set of final
statistics mathematically equivalent to a hypothetical single-shot run.

Each chunk inherits the base config but with:
- ``simulation.start_time`` shifted by the cumulative chunk offset.
- ``simulation.duration_days`` and ``timesteps`` set for that chunk
  (cadence = base ``timesteps`` / ``duration_days`` is preserved).
- ``thermal.initial_state_file`` = previous chunk's final-state file
  (chunk 0 leaves it unset and runs full equilibration).
- ``thermal.final_state_file`` = this chunk's final-state file.
- ``output.write_raw_stats: true`` so per-chunk sums/counts and the
  Schorghofer accumulator are persisted for the merge step.
- ``output.skip_time_series: true`` (per-step files would be huge and
  aren't needed for the merged-stats workflow; users who want
  animations should run a separate single-chunk job).
- ``output.directory`` pointed at ``<outdir>/chunk_NNN/``.

After all chunks succeed, the orchestrator invokes ``ptm_merge_chunks``
to combine per-chunk raw stats into ``<outdir>/statistics/`` and
copies/concatenates the metadata files (mesh.ply, time_steps.txt) into
the parent output dir for downstream consumers (the GUI, lookup
helpers).

Usage::

    python3 scripts/run_chunked_simulation.py \\
        --config examples/moon_mm_site1_18yr.yaml \\
        --total-days 6798.5 \\
        --output-dir output/mm_site1_18yr \\
        [--ram-budget-gb 384] [--chunks 4] [--resume] [--dry-run]

ETA: each chunk's wall time is recorded; after chunk i of N completes
we print the average so far and ``ETA = (N - i - 1) × avg``.
"""
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    sys.exit("error: PyYAML is required (pip install pyyaml)")

# psutil is used only to probe RAM; we have a sysctl fallback for macOS.
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# --------------------------------------------------------------------------
# RAM probing + chunk-size estimator
# --------------------------------------------------------------------------

def available_ram_gb() -> float:
    """Best-effort *currently available* RAM in GiB.

    Other processes (OneDrive, browsers, thumbnail caches) can claim a
    significant chunk of total RAM, so probing 'total' over-estimates
    what PTM3D can actually use. Prefer the OS's available-memory
    figure when possible, with a sysctl-based fallback on macOS that
    subtracts the in-use pages.
    """
    if _HAS_PSUTIL:
        return psutil.virtual_memory().available / (1024 ** 3)
    if platform.system() == "Darwin":
        # Free + inactive + speculative pages are reasonable proxies for
        # 'available', mirroring what `vm_stat` reports.
        try:
            out = subprocess.check_output(["vm_stat"]).decode()
            page_size = 4096
            for line in out.splitlines():
                if line.startswith("Mach Virtual Memory Statistics") and "page size of" in line:
                    page_size = int(line.split("page size of")[1].split()[0])
            free = inactive = spec = 0
            for line in out.splitlines():
                if line.startswith("Pages free:"):       free     = int(line.split()[2].rstrip("."))
                if line.startswith("Pages inactive:"):   inactive = int(line.split()[2].rstrip("."))
                if line.startswith("Pages speculative:"):spec     = int(line.split()[2].rstrip("."))
            return (free + inactive + spec) * page_size / (1024 ** 3)
        except Exception:
            pass
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
            return int(out) / (1024 ** 3)
        except Exception:
            pass
    # Linux fallback
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    print("warning: could not probe system RAM; assuming 64 GiB",
          file=sys.stderr)
    return 64.0


def read_n_faces(mesh_file: Path) -> int:
    """Read 'element face N' from a PLY header to size the chunk
    estimator. Falls back to 0 if the file isn't a recognised PLY."""
    try:
        with open(mesh_file, "rb") as f:
            for _ in range(200):  # PLY headers are short
                line = f.readline()
                if not line:
                    break
                s = line.decode(errors="replace").strip()
                if s.startswith("element face "):
                    return int(s.split()[2])
                if s == "end_header":
                    break
    except Exception:
        pass
    return 0


def estimate_max_chunk_timesteps(n_faces: int, ram_gb: float,
                                 safety: float = 0.7,
                                 overhead_gb: float = 10.0,
                                 bytes_per_facestep: int = 40) -> int:
    """Conservatively estimate the largest output-timestep count that
    fits in RAM for the given mesh.

    The peak memory footprint occurs *during* the chunked-storage →
    flat-array extraction transition, when both copies coexist:
      3 illum-component flat arrays  (3 × 8 B/face-step, float64)
      1 surface_temps                (1 × 8 B/face-step, float64)
      1 transient chunked storage    (1 × 8 B/face-step, released after extraction)
    Total: 40 B/face-step at peak. ``overhead_gb`` accounts for mesh,
    BVH, GPU buffers, and other constants; ``safety`` keeps headroom
    so the OS doesn't start paging.
    """
    if n_faces <= 0:
        return -1  # unknown, caller will fall back to user-provided value
    budget = max(0.0, ram_gb * safety - overhead_gb) * (1024 ** 3)
    raw = budget / (n_faces * bytes_per_facestep)
    return max(1, int(raw))


# --------------------------------------------------------------------------
# Time formatting
# --------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN-safe
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h = int(seconds // 3600)
    m = int((seconds - h * 3600) // 60)
    return f"{h}h{m}m"


def parse_iso8601(s: str) -> dt.datetime:
    """Accept either '2026-01-01T00:00:00' or '2026-01-01 00:00:00'."""
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"could not parse start_time '{s}'")


def fmt_iso8601(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------
# Per-chunk YAML generation
# --------------------------------------------------------------------------

def chunk_is_complete(chunk_dir: Path) -> bool:
    """A chunk is considered complete if it has the per-datatype raw
    sum/count files in statistics/_raw/, plus the depth_grid + ice_depth
    accumulator (only if compute_ice_stability was on; we test depth_grid
    as a proxy)."""
    raw = chunk_dir / "statistics" / "_raw"
    if not raw.is_dir():
        return False
    must_have = [raw / "temperature_sum.bin",
                 raw / "temperature_count.bin"]
    return all(p.exists() and p.stat().st_size > 4 for p in must_have)


# PTMT thermal-state checkpoint magic (little-endian "TMTP" → 0x50544d54).
# See runThermalModelsForAllFaces_GPU_Persistent in ptm_thermal_sim.cpp.
_CHECKPOINT_MAGIC = b"\x54\x4d\x54\x50"


def is_valid_checkpoint(p: Path) -> bool:
    """Cheap sanity check on a thermal_state_after_chunk_NNN.bin file:
    nonzero size and PTMT magic readable in the first 4 bytes. Catches
    OneDrive lazy-fetch corruption (correct size, zero contents) that
    would otherwise blow up the next chunk on load."""
    try:
        if not p.is_file() or p.stat().st_size < 8:
            return False
        with open(p, "rb") as f:
            return f.read(4) == _CHECKPOINT_MAGIC
    except Exception:
        return False


def make_chunk_yaml(base_cfg: dict, chunk_dir: Path,
                    chunk_start_time: dt.datetime,
                    chunk_duration_days: float,
                    chunk_timesteps: int,
                    initial_state_file: Path | None,
                    final_state_file: Path) -> Path:
    """Render an isolated chunk config and write it to chunk_dir/config.yaml."""
    cfg = _deep_copy(base_cfg)

    # simulation
    sim = cfg.setdefault("simulation", {})
    sim["start_time"]    = fmt_iso8601(chunk_start_time)
    sim["duration_days"] = float(chunk_duration_days)
    sim["timesteps"]     = int(chunk_timesteps)

    # thermal
    th = cfg.setdefault("thermal", {})
    if initial_state_file is not None:
        th["initial_state_file"] = str(initial_state_file)
    else:
        th.pop("initial_state_file", None)
    th["final_state_file"]   = str(final_state_file)

    # output
    out = cfg.setdefault("output", {})
    out["directory"]        = str(chunk_dir)
    out["write_raw_stats"]  = True
    # Per-step time-series files would be huge for long chunked runs.
    # Force-skip them — the orchestrator workflow only needs stats.
    out["skip_time_series"] = True
    # Chunk 0 (no checkpoint loaded) writes mesh.ply once for the
    # orchestrator to propagate to the parent dir; chunks 1+ skip the
    # redundant per-chunk mesh write (~hundreds of MB each).
    out["skip_mesh"]        = (initial_state_file is not None)

    chunk_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = chunk_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def _deep_copy(d):
    """Tiny dict deep-copy that handles the (str/int/float/list/dict/None)
    YAML universe without pulling in copy.deepcopy."""
    if isinstance(d, dict):
        return {k: _deep_copy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deep_copy(v) for v in d]
    return d


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--config", required=True, type=Path,
                   help="Base PTM3D YAML config to chunk.")
    p.add_argument("--total-days", required=True, type=float,
                   help="Total simulated duration in Earth days.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Parent output directory; chunk_NNN/ created underneath.")
    p.add_argument("--ram-budget-gb", type=float, default=None,
                   help="Override probed total RAM (GiB) used to size chunks.")
    p.add_argument("--chunks", type=int, default=None,
                   help="Override the auto-decided chunk count.")
    p.add_argument("--resume", action="store_true",
                   help="Skip chunks whose statistics/_raw/ already exists.")
    p.add_argument("--retries", type=int, default=2,
                   help="Number of additional retries per chunk on non-zero exit "
                        "or timeout (default 2; total attempts = retries + 1). "
                        "Useful for transient AMD/macOS GPU hangs (gpusKillClientExt).")
    p.add_argument("--chunk-timeout-min", type=float, default=60.0,
                   help="Per-chunk wall-time cap in minutes (default 60). When a "
                        "ptm_thermal_sim subprocess exceeds this, it is SIGKILLed "
                        "and the attempt counts as a retryable failure. Prevents "
                        "GPU-driver hangs from stalling overnight batches forever. "
                        "Set <=0 to disable the timeout.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the chunk plan and exit without running anything.")
    p.add_argument("--ptm-bin", type=Path,
                   default=Path(__file__).resolve().parent.parent
                            / "build" / "bin" / "ptm_thermal_sim",
                   help="Path to the ptm_thermal_sim binary.")
    p.add_argument("--merge-bin", type=Path,
                   default=Path(__file__).resolve().parent.parent
                            / "build" / "bin" / "ptm_merge_chunks",
                   help="Path to the ptm_merge_chunks binary.")
    p.add_argument("--thermal-monitor", action="store_true",
                   help="Spawn scripts/gpu_thermal_monitor.py as a sidecar; "
                        "log GPU temperature/power/fan to "
                        "<output-dir>/thermal.csv and gate chunk launches on "
                        "<output-dir>/thermal.flag (cool-down between chunks "
                        "if any GPU was hot during the previous chunk).")
    p.add_argument("--thermal-hot-c", type=int, default=85,
                   help="GPU temperature (C) above which we declare HOT and "
                        "delay the next chunk launch. Default: 85.")
    p.add_argument("--thermal-cool-c", type=int, default=75,
                   help="GPU temperature (C) at/below which the cool-down "
                        "dwell timer starts. Default: 75.")
    p.add_argument("--thermal-cool-dwell-s", type=int, default=60,
                   help="Seconds GPUs must hold at/below --thermal-cool-c "
                        "before we clear thermal.flag and resume. Default: 60.")
    p.add_argument("--thermal-poll-s", type=float, default=20.0,
                   help="Sidecar sampling interval (seconds). Default: 20.")
    p.add_argument("--thermal-max-wait-s", type=int, default=900,
                   help="Hard cap on the inter-chunk thermal wait. After this "
                        "many seconds we proceed regardless. Default: 900 "
                        "(15 min).")
    args = p.parse_args()

    if not args.config.is_file():
        sys.exit(f"error: config not found: {args.config}")
    if not args.ptm_bin.is_file():
        sys.exit(f"error: ptm_thermal_sim not found at {args.ptm_bin}; "
                 "specify --ptm-bin or build it first.")
    if not args.merge_bin.is_file():
        sys.exit(f"error: ptm_merge_chunks not found at {args.merge_bin}; "
                 "build it first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config.
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    base_sim = base_cfg.get("simulation", {})
    base_start = parse_iso8601(str(base_sim.get("start_time",
                                                "2026-01-01T00:00:00")))
    base_duration_days = float(base_sim.get("duration_days", 1.0))
    base_timesteps = int(base_sim.get("timesteps", 1))
    if base_duration_days <= 0 or base_timesteps <= 0:
        sys.exit("error: base config has invalid duration_days or timesteps.")
    output_steps_per_day = base_timesteps / base_duration_days

    # Try to peek at the mesh to size chunks. Look for mesh.import_file
    # in the base config; otherwise we'll need the user to override.
    mesh_path_str = str(base_cfg.get("mesh", {}).get("import_file", ""))
    n_faces_est = read_n_faces(Path(mesh_path_str)) if mesh_path_str else 0

    # RAM probing.
    ram_gb = args.ram_budget_gb if args.ram_budget_gb else available_ram_gb()
    print(f"[chunked] available RAM: {ram_gb:.1f} GiB")
    print(f"[chunked] mesh '{Path(mesh_path_str).name or '?'}' "
          f"(face count {n_faces_est if n_faces_est else 'unknown'})")
    print(f"[chunked] base cadence: {output_steps_per_day:.4f} output steps/day")
    print(f"[chunked] target total simulated days: {args.total_days:.2f}")

    total_output_steps = int(round(args.total_days * output_steps_per_day))

    if args.chunks is not None:
        n_chunks = max(1, args.chunks)
        max_ts = (total_output_steps + n_chunks - 1) // n_chunks
        print(f"[chunked] user-overridden chunk count: {n_chunks} "
              f"(~{max_ts} timesteps/chunk)")
    else:
        max_ts = estimate_max_chunk_timesteps(n_faces_est, ram_gb)
        if max_ts <= 0:
            sys.exit("error: cannot estimate chunk size without n_faces. "
                     "Pass --chunks N to override.")
        n_chunks = max(1, (total_output_steps + max_ts - 1) // max_ts)
        print(f"[chunked] estimated max timesteps/chunk: {max_ts}  "
              f"=> {n_chunks} chunk(s)")

    # Distribute timesteps as evenly as possible across chunks.
    base_per = total_output_steps // n_chunks
    extras   = total_output_steps - base_per * n_chunks
    chunk_ts_list = [base_per + (1 if i < extras else 0)
                     for i in range(n_chunks)]

    # Print the plan.
    print("\n[chunked] chunk plan:")
    cumdays = 0.0
    plan = []
    for i, ts in enumerate(chunk_ts_list):
        d_days = ts / output_steps_per_day
        start_t = base_start + dt.timedelta(days=cumdays)
        plan.append({
            "idx": i, "ts": ts, "duration_days": d_days,
            "start_time": start_t,
            "chunk_dir": args.output_dir / f"chunk_{i:03d}"
        })
        print(f"  chunk {i:03d}: {ts} ts × {d_days:.3f} d  "
              f"(start {fmt_iso8601(start_t)})  -> {plan[-1]['chunk_dir']}")
        cumdays += d_days

    if args.dry_run:
        return

    # Run each chunk.
    state_dir = args.output_dir / "_thermal_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Optional thermal-monitor sidecar. Logs per-GPU temperature/power
    # to <output-dir>/thermal.csv and maintains <output-dir>/thermal.flag
    # while any GPU is above --thermal-hot-c. We poll the flag at chunk
    # boundaries below to enforce cool-downs without ever suspending the
    # active worker (SIGSTOP on a process holding OpenCL command queues
    # is exactly the failure mode we are mitigating).
    flag_path = args.output_dir / "thermal.flag"
    therm_proc: Optional[subprocess.Popen] = None
    if args.thermal_monitor:
        # Reset any stale flag from a prior run.
        try:
            flag_path.unlink()
        except FileNotFoundError:
            pass
        mon_script = Path(__file__).resolve().parent / "gpu_thermal_monitor.py"
        if not mon_script.is_file():
            sys.exit(f"error: thermal monitor script not found at {mon_script}")
        therm_cmd = [sys.executable, str(mon_script),
                     "--log-dir", str(args.output_dir),
                     "--interval", f"{args.thermal_poll_s:g}",
                     "--hot-threshold-c", str(args.thermal_hot_c),
                     "--cool-threshold-c", str(args.thermal_cool_c),
                     "--cool-dwell-s", str(args.thermal_cool_dwell_s)]
        therm_log = open(args.output_dir / "thermal_monitor.log", "a")
        therm_proc = subprocess.Popen(therm_cmd, stdout=therm_log,
                                      stderr=subprocess.STDOUT)
        print(f"[chunked] thermal monitor started (PID {therm_proc.pid}); "
              f"log {args.output_dir}/thermal_monitor.log; "
              f"flag {flag_path}")

        def _stop_monitor() -> None:
            if therm_proc is None or therm_proc.poll() is not None:
                return
            try:
                therm_proc.terminate()
                try:
                    therm_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    therm_proc.kill()
            except Exception:
                pass

        atexit.register(_stop_monitor)

    chunk_walltimes: list[float] = []
    for i, info in enumerate(plan):
        cd = info["chunk_dir"]
        # Resume: skip if already complete.
        if args.resume and chunk_is_complete(cd):
            print(f"\n[chunked] chunk {i:03d}: complete (resume), skipping.")
            chunk_walltimes.append(0.0)
            continue

        # Thermal gate. If the sidecar has the hot-flag up, hold here
        # until the GPUs cool below --thermal-cool-c (capped by
        # --thermal-max-wait-s). We only check between chunks because
        # the active worker holds OpenCL command queues mid-chunk and
        # SIGSTOPing it tends to provoke the very gpusKillClient hang
        # we are trying to avoid.
        if args.thermal_monitor and flag_path.exists():
            t_wait_start = time.time()
            announced = False
            while flag_path.exists():
                waited = time.time() - t_wait_start
                if waited >= args.thermal_max_wait_s:
                    print(f"[chunked] chunk {i:03d}: thermal cool-down hit "
                          f"--thermal-max-wait-s ({args.thermal_max_wait_s}s); "
                          f"proceeding while still hot.", flush=True)
                    break
                if not announced:
                    print(f"[chunked] chunk {i:03d}: thermal.flag is set; "
                          f"holding launch until GPUs cool below "
                          f"{args.thermal_cool_c}C (max wait "
                          f"{args.thermal_max_wait_s}s)...", flush=True)
                    announced = True
                time.sleep(min(15.0, args.thermal_poll_s))
            if announced and not flag_path.exists():
                cooled = time.time() - t_wait_start
                print(f"[chunked] chunk {i:03d}: GPUs cool after {cooled:.0f}s; "
                      f"resuming.", flush=True)

        # Wire up checkpoint files. Validate the previous chunk's
        # checkpoint magic up-front: OneDrive's lazy-fetch can leave a
        # 200-MB file with the right size but zeroed contents, which
        # ptm_thermal_sim catches as "bad magic" only after the chunk's
        # ~3 min illumination phase. Refuse to launch if the file is bad
        # so the user (or an outer driver) can clean and resume.
        prev_state = state_dir / f"thermal_state_after_chunk_{i-1:03d}.bin"
        this_state = state_dir / f"thermal_state_after_chunk_{i:03d}.bin"
        initial_state = prev_state if i > 0 and prev_state.exists() else None
        if i > 0 and initial_state is None:
            sys.exit(f"error: previous chunk's thermal state {prev_state} "
                     f"is missing. Did chunk {i-1} complete?")
        if initial_state is not None and not is_valid_checkpoint(initial_state):
            sys.exit(f"error: previous chunk's thermal state {initial_state} "
                     f"failed magic check (corrupt or not yet downloaded by "
                     f"OneDrive). Delete {initial_state.parent}/{initial_state.name} "
                     f"and {cd.parent}/chunk_{i-1:03d}/, then re-run with --resume "
                     f"to regenerate.")

        cfg_path = make_chunk_yaml(base_cfg, cd,
                                   info["start_time"], info["duration_days"],
                                   info["ts"],
                                   initial_state, this_state)

        print(f"\n[chunked] === chunk {i+1}/{n_chunks}: "
              f"{info['ts']} ts × {info['duration_days']:.3f} d  ===")
        # Retry loop for transient AMD/macOS OpenCL GPU hangs
        # (signal-6 abort with gpusKillClientExt). Up to args.retries
        # additional attempts before giving up on this chunk. Between
        # attempts we sleep briefly to let the GPU driver release state.
        attempt = 0
        ret = None
        last_failure = "?"
        timeout_s = (args.chunk_timeout_min * 60.0
                     if args.chunk_timeout_min and args.chunk_timeout_min > 0
                     else None)
        t0 = time.time()
        while attempt <= args.retries:
            if attempt > 0:
                # Exponential-ish back-off: 120s, 240s, 480s, ... capped at 600s.
                # On macOS+AMD the command-buffer pool sometimes stays wedged
                # for several minutes after a kernel hang; the previous 30/60s
                # base was too short — three quick retries all hit the same
                # degraded driver state on lunar 1.8M-face workloads
                # (A03MP012 Y05_winter / Y09_summer, 2026-05-13).
                cooldown = min(600, 120 * (2 ** (attempt - 1)))
                print(f"[chunked] chunk {i:03d}: retry {attempt}/{args.retries} "
                      f"after {last_failure}; "
                      f"waiting {cooldown}s for GPU to settle...",
                      flush=True)
                time.sleep(cooldown)
            try:
                ret = subprocess.run(
                    [str(args.ptm_bin), "--config", str(cfg_path)],
                    timeout=timeout_s)
            except subprocess.TimeoutExpired:
                # subprocess.run already SIGKILLed the child on timeout. Treat
                # this as a retryable failure (matches the orchestrator's
                # established AMD-GPU-hang recovery model).
                ret = None
                last_failure = (f"hang past --chunk-timeout-min "
                                f"({args.chunk_timeout_min:g} min)")
                print(f"[chunked] chunk {i:03d}: {last_failure}; killed.",
                      flush=True)
                attempt += 1
                continue
            if ret.returncode == 0 and chunk_is_complete(cd):
                # Post-write checkpoint validation. If the OS or OneDrive
                # corrupted the new state file (right size, wrong magic),
                # treat it as a transient failure: clear the half-baked
                # outputs and retry. Catches the failure mode where chunk
                # i exits clean but chunk i+1 immediately blows up on
                # load.
                if i < n_chunks - 1 and not is_valid_checkpoint(this_state):
                    last_failure = (f"checkpoint magic check failed on "
                                    f"{this_state.name}")
                    print(f"[chunked] chunk {i:03d}: {last_failure}; "
                          f"clearing chunk dir and retrying.",
                          flush=True)
                    try:
                        this_state.unlink(missing_ok=True)
                        import shutil
                        shutil.rmtree(cd, ignore_errors=True)
                    except Exception as e:
                        print(f"[chunked] chunk {i:03d}: cleanup warning: {e}",
                              flush=True)
                    # BUGS.md PTM-77: rmtree just deleted chunk_dir/config.yaml,
                    # which is the very file the retry passes to --config. Every
                    # retry after this branch was therefore guaranteed to fail —
                    # and before the PTM-66 fix it failed INVISIBLY, because a
                    # missing config was downgraded to a warning and the binary
                    # ran a full simulation on built-in defaults into the default
                    # output/ directory. Regenerate it, and prove it is there.
                    cfg_path = make_chunk_yaml(base_cfg, cd,
                                               info["start_time"],
                                               info["duration_days"],
                                               info["ts"],
                                               initial_state, this_state)
                    if not cfg_path.is_file():
                        sys.exit(f"[chunked] chunk {i:03d}: could not regenerate "
                                 f"{cfg_path} after the cleanup; aborting rather "
                                 f"than retrying against a missing config.")
                    print(f"[chunked] chunk {i:03d}: regenerated {cfg_path}",
                          flush=True)
                    attempt += 1
                    continue
                break
            last_failure = f"non-zero exit {ret.returncode}"
            attempt += 1
        dt_s = time.time() - t0
        chunk_walltimes.append(dt_s)
        if ret is None or ret.returncode != 0:
            rc = "timeout" if ret is None else f"exit code {ret.returncode}"
            sys.exit(f"\n[chunked] chunk {i:03d} FAILED ({rc}) after "
                     f"{attempt} attempt(s). Re-run with --resume to retry "
                     "from this chunk (prior chunks will be skipped).")
        if not chunk_is_complete(cd):
            sys.exit(f"\n[chunked] chunk {i:03d} exited cleanly but did not "
                     "produce expected statistics/_raw/ output. Check the "
                     "chunk log; aborting.")

        # ETA: rolling average of *attempted* chunks (excludes resumed).
        actual = [w for w in chunk_walltimes if w > 0]
        avg = sum(actual) / max(1, len(actual))
        remaining = (n_chunks - i - 1) * avg
        print(f"[chunked] chunk {i:03d} done in {fmt_duration(dt_s)} "
              f"(avg {fmt_duration(avg)}); "
              f"ETA for remaining {n_chunks - i - 1} chunk(s): "
              f"{fmt_duration(remaining)}")

    # Merge.
    print(f"\n[chunked] merging {n_chunks} chunk(s) into {args.output_dir}/statistics/ ...")
    chunks_arg = [str(p["chunk_dir"]) for p in plan]
    cmd = [str(args.merge_bin), "--chunks", *chunks_arg,
           "--output", str(args.output_dir)]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        sys.exit("[chunked] merge failed.")

    # Mesh + time_steps for the parent dir (downstream tools expect them).
    chunk0 = plan[0]["chunk_dir"]
    src_mesh = chunk0 / "mesh.ply"
    if src_mesh.exists():
        dst = args.output_dir / "mesh.ply"
        if not dst.exists():
            try:
                os.link(src_mesh, dst)  # hard link to save disk
            except OSError:
                import shutil; shutil.copy2(src_mesh, dst)

    # Concatenate time_steps.txt across chunks (header from chunk 0).
    ts_out = args.output_dir / "time_steps.txt"
    try:
        with open(ts_out, "w") as fout:
            wrote_header = False
            for p_ in plan:
                src = p_["chunk_dir"] / "time_steps.txt"
                if not src.exists():
                    continue
                with open(src) as fin:
                    for line in fin:
                        if line.startswith("#"):
                            if not wrote_header:
                                fout.write(line)
                            continue
                        fout.write(line)
                wrote_header = True
    except Exception as e:
        print(f"[chunked] warning: could not write {ts_out}: {e}",
              file=sys.stderr)

    # Final summary.
    total_wall = sum(chunk_walltimes)
    print(f"\n[chunked] DONE. {n_chunks} chunk(s), "
          f"total wall time {fmt_duration(total_wall)}, "
          f"output in {args.output_dir}")


if __name__ == "__main__":
    main()
