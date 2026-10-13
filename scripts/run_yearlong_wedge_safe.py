#!/usr/bin/env python3
"""
Wedge-safe two-stage driver for the Moon3D native year-long coupled run.

ROOT CAUSE (verified in source, not assumed — see SESSION-NOTES-2026-07-20-
moon3d-yearlong-wedge-redesign.md for the full citation trail):

  ThermalOpenCLSolver::solvePersistentMultipleCycles (thermal_opencl.cpp)
  submits `n_equil_cycles + 1` back-to-back clEnqueueNDRangeKernel calls with
  ZERO host synchronization between them -- a single clFinish() only at the
  very end. Each "cycle" runs the FULL requested timestep count continuously.
  For the equilibration phase this means ALL of
  `equilibration_orbits * equilibration_samples_per_diurnal_cycle` timesteps
  execute inside ONE unbroken GPU dispatch. The year-long config's 13 orbits
  x 120 samples = 1560 equilibration timesteps, at n_layers=50 (thermal_b=32),
  is a ~6.5x-timestep x ~1.7x-layer (~11x combined) heavier single dispatch
  than the proven-safe lunation baseline (240 steps, n_layers=30, 3.9s/cycle,
  run dozens of times without incident) -- and it hit the documented macOS+
  AMD gpusKillClientExt command-buffer watchdog (memory gpu_hang_amd_macos.md)
  partway through Phase 1, before any output was ever written.

KEY LIMITATION DISCOVERED (this blocks the obvious fix of just chunking the
output window with the existing scripts/run_chunked_simulation.py):
  ptm_thermal_sim.cpp:9490 and :10383 both FORCE
  equilibration_orbits/equilibration_steps to ZERO whenever
  `thermal.initial_state_file` is set ("the loaded T(z) is already
  equilibrated by the previous chunk"). There is NO mechanism to load a
  partially-equilibrated checkpoint and continue equilibrating -- a single
  invocation either does ALL of the requested equilibration in one
  continuous dispatch, or none at all. run_chunked_simulation.py's chunk 0
  ALWAYS gets the full base-config equilibration_orbits (its own i==0 guard
  unconditionally sets initial_state_file=None) -- so naively chunking the
  output window would still force the SAME 1560-step dispatch into chunk 0
  and reproduce the exact wedge. This script does NOT modify
  run_chunked_simulation.py (a shared, validated tool); it implements a
  parallel, purpose-built two-stage chain using the identical, already-
  proven mechanisms (TMTP-magic checkpoint file, retry/cooldown pattern,
  ptm_merge_chunks) it imports from that module.

CONTINUITY GUARANTEE: the state handoff between stages is the SAME
mechanism the equilibration paper documents as already load-bearing and
validated within a single process: "T(z,t) at the end of the equilibration
phase is the natural starting point for the radiation-coupling Picard
iteration... the same post-equilibration profile is reused as the initial
condition for every radiation-coupling iteration" (docs/equilibration-
paper.md Sec.7, "Coupling to inter-face radiation"). That in-memory reuse
and this cross-process reuse via thermal_state_after_*.bin (TMTP magic,
validated on load, ptm_thermal_sim.cpp:3316-3356) serialize/deserialize the
IDENTICAL T(z) array -- there is no approximation or interpolation in the
handoff, just a disk round-trip of the same floats. This is the same
checkpoint format scripts/run_chunked_simulation.py already relies on for
production A3GT chunked+radiation-coupled runs.

TWO STAGES (chunk numbering continues run_chunked_simulation.py's
`chunk_NNN` convention for downstream-tool compatibility):
  Stage A (chunk_000): FULL 13-orbit equilibration (1560 steps, unavoidably
    one continuous dispatch) + a SMALL real output slice (default 20
    timesteps, ~4.9 days) that also counts toward the final merged stats.
    gpu_batch_size is EXPLICITLY REDUCED (this script's one wedge-safety
    lever, per the documented mitigation in memory/gpu_hang_amd_macos.md:
    "gpu_batch_size_override ... lets the user shrink batches (watchdog
    mitigation on macOS+AMD)"). Run `--mode calibrate` first to measure the
    actual ms/cycle at the candidate batch size on a CHEAP 2-orbit probe
    before committing to the full 13-orbit stage.
  Stage B (chunk_001): the REMAINING output timesteps (default 1464),
    loaded from chunk_000's checkpoint (equilibration auto-forced to 0 by
    the code above), STANDARD gpu_batch_size (this phase carries no
    equilibration burden -- estimated ~18s persistent-kernel dispatch,
    comfortably inside the proven-safe envelope; see the design note).

Both stages write raw per-chunk stats (write_raw_stats, skip_time_series)
exactly as run_chunked_simulation.py does, so the existing ptm_merge_chunks
binary combines them into a single, standard statistics/ directory
mathematically equivalent to an unchunked run.

EPOCH GRID / HAND-OFF (2026-10-12). The output grid is half-open: chunk k
exports epochs t0 + (S_k + i)*dt for i in [0, N_k), where S_k is the integer
count of output steps before it and dt = duration_days*86400/timesteps. Two
things make the hand-off land EXACTLY on the successor's first epoch, the
same two run_chunked_simulation.py uses:
  * thermal.checkpoint_lookahead_steps = 1 on every chunk: the engine
    integrates one grid point past the export window (not exported, not
    reduced) and writes final_state_file THERE. Without it the state was the
    last EXPORTED frame, one cadence stale.
  * simulation.start_time is left as the base config has it, and the chunk
    start is carried as simulation.start_time_offset_seconds = S_k*dt in ET
    seconds. Shifting the UTC string instead is off the ET grid by the
    TDB-UTC periodic term (measured 0.57 ms at the Jan-21 seam of the Stage-1
    Haworth run, and a whole second across a leap second).
Before this fix the engine refused the hand-off (the 1-cadence-stale gap plus
the 0.57 ms UTC error exceeded its one-cadence warn band). Older engines took
the stale state silently, so every earlier run of this driver carries a
1-cadence seam at each chunk boundary (BUGS.md PTM-719).

PICARD CAP (2026-08-15). Every radiation knob is inherited from the base config
byte for byte, so a k_max chosen once by habit propagates silently to every
later generation of that config. The moon3d year-long lineage carried
`radiation.max_coupling_iterations: 5` from 2026-07 through the V2 launch --
half the then-default 10, and a third of what that mesh needs (measured
truncation on the 509k flagship at 5 sweeps: +26.2 K on the worst face,
one-signed cold; the increment first falls below 1 K at sweep 16). The engine
default is now 16 and this driver PRINTS the effective cap on every coupled
launch and WARNS when the base config pins it lower. It does not override --
see check_coupling_iterations().

⚠️ The base configs that lineage inherits from
(examples/moon3d_native_yearlong/moon3d_native_yearlong_{coupled,radoff}.yaml,
themselves derived from examples/moon3d_500k_clean.yaml) are UNTRACKED: they
exist only in the ~/Projects/ptm working tree, in one checkout, and carry
machine-absolute mesh/VF/output paths. A campaign's inputs living outside
version control is how the moon3d_lat_final dataset was lost in 2026-08; treat
the launch warning above as the only guard until they are brought in.

Usage:
    # 1. Calibrate (cheap, ~a few minutes): measures ms/cycle at n_layers=50
    #    for a 2-orbit probe at the candidate batch size, extrapolates to
    #    the real 13-orbit dispatch, and recommends proceed/shrink-further.
    python3 scripts/run_yearlong_wedge_safe.py --mode calibrate \\
        --config examples/moon3d_native_yearlong/moon3d_native_yearlong_coupled.yaml \\
        --candidate-batch-size 8000 --output-dir <scratch>/calibrate

    # 2. Run (only after calibrate recommends PROCEED, and only after the
    #    GPU health-check gate passes):
    python3 scripts/run_yearlong_wedge_safe.py --mode run \\
        --config examples/moon3d_native_yearlong/moon3d_native_yearlong_coupled.yaml \\
        --output-dir <output-parent>/coupled \\
        --stage-a-batch-size <calibrated value> --stage-a-output-steps 20 \\
        --stage-b-batch-size 200000 \\
        --ptm-bin <bin> --merge-bin <bin>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("error: PyYAML is required (pip install pyyaml)")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_chunked_simulation import (  # noqa: E402
    fmt_duration, parse_iso8601, fmt_iso8601, is_valid_checkpoint,
    _deep_copy, chunk_is_complete, CHECKPOINT_LOOKAHEAD_STEPS,
)
# Phase R (R3/R5): shared runtime-robustness gates. Kept OPT-IN here so this
# script's default behavior is byte-for-byte the same as before -- the gate
# only engages when --require-env / --strict-preflight are passed on the
# NEXT launch (never mid-flight).
import ptm_run_guard as guard  # noqa: E402


def _parse_require_env(spec: str):
    """Parse a --require-env spec 'NAME,NAME2=VAL,...' into the manifest that
    ptm_run_guard.check_env_completeness accepts (dict name->expected|None)."""
    manifest = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            manifest[k.strip()] = v.strip()
        else:
            manifest[tok] = None
    return manifest

CYCLE_RE = re.compile(
    r"Persistent kernel:\s*(\d+)\s*cycles\s*x\s*(\d+)\s*timesteps.*?in\s*([\d.]+)\s*ms"
)
NLAYERS_RE = re.compile(r"n_layers=(\d+)")


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------

class Ledger:
    """Chunk-progress ledger -- JSON, appended-to after every stage/attempt,
    so a watcher (ptm-monitor or a human) can see exactly where a long
    unattended run stands without re-parsing raw sim logs."""

    def __init__(self, path: Path):
        self.path = path
        self.entries = []
        if path.exists():
            try:
                self.entries = json.loads(path.read_text())
            except Exception:
                self.entries = []

    def record(self, **kw):
        kw["ts"] = fmt_iso8601(dt.datetime.now())
        self.entries.append(kw)
        self.path.write_text(json.dumps(self.entries, indent=2))
        print(f"[ledger] {kw}", flush=True)


# --------------------------------------------------------------------------
# Per-attempt log scraping (mirrors the addon-driver gate checks used
# elsewhere in this campaign, applied here per-stage rather than shell grep)
# --------------------------------------------------------------------------

def scrape_cycle_stats(log_path: Path):
    """Return list of (total_cycles, n_timesteps, ms) tuples found in the
    log, plus n_layers if reported. Used both for calibration and for
    post-hoc confirmation that a stage's dispatch stayed inside its
    intended shape."""
    text = log_path.read_text(errors="replace")
    cycles = [(int(a), int(b), float(c)) for a, b, c in CYCLE_RE.findall(text)]
    nlayers_m = NLAYERS_RE.search(text)
    nlayers = int(nlayers_m.group(1)) if nlayers_m else None
    hung = "GPU hang occurred" in text or "gpusKillClientExt" in text
    return cycles, nlayers, hung


def gate_check_common(log_path: Path, expect_hole_seal_count: int | None = 30) -> list[str]:
    """Same illumination/VF banners the other addon drivers gate on.
    Returns a list of problems (empty = clean).

    `expect_hole_seal_count`: PTM-18 interior-hole-seal count is MESH-SPECIFIC
    (2026-07-21 finding: moon3d seals 30, haworth_radcore_r32c legitimately
    seals 140 -- verified via a dedicated hole-census probe before launch, not
    guessed). The original hardcoded "sealed 30" gate false-failed a healthy
    Haworth chunk_000 that had already completed correctly (rc=0), which would
    have burned a full ~18-min retry loop for nothing. Pass the mesh's OWN
    verified count here; default 30 preserves prior behavior for moon3d
    callers that don't pass this. None = skip the hole-seal check entirely
    (accept ENABLED with any positive count) if the count hasn't been
    independently verified for a new mesh.
    """
    text = log_path.read_text(errors="replace")
    problems = []
    if re.search(r"does not match mesh face count|Empty view factor matrix", text):
        problems.append("VF mismatch/empty")
    if re.search(r"Using default icosphere|Failed to load mesh from file", text):
        problems.append("mesh load failed -> icosphere fallback")
    if "TRUE (winding-consistent) normals ENABLED" not in text:
        problems.append("TRUE_NORMALS did not engage")
    if "rise-epsilon self-relief discriminator ENABLED (k=0.20" not in text:
        problems.append("rise-eps k=0.20 did not engage")
    if "ABSOLUTE self-relief floor OVERRIDE h_eps = 0.00 mm" not in text:
        problems.append("RISE_HABS h_eps=0 did not engage")
    if expect_hole_seal_count is not None:
        if f"PTM-18 interior hole-seal ENABLED: sealed {expect_hole_seal_count} interior hole" not in text:
            problems.append(f"PTM-18 seal count != {expect_hole_seal_count}")
    elif not re.search(r"PTM-18 interior hole-seal ENABLED: sealed \d+ interior hole", text):
        problems.append("PTM-18 hole-seal banner missing entirely")
    return problems


def gate_check_radiation(log_path: Path) -> list[str]:
    text = log_path.read_text(errors="replace")
    problems = []
    if not re.search(r"Loaded view factors from:|Loaded H-matrix view factors:", text):
        problems.append("VF not loaded from file")
    if re.search(r"Building H-matrix view factor for|Computing view factors for", text):
        problems.append("VF BUILD started (should be LOADED)")
    if "REPLACED H-matrix with symmetric per-partner referee VF" not in text:
        problems.append("VF_REPLACE did not engage")
    if "mesh-hash OK" not in text:
        problems.append("corrected-VF mesh-hash NOT OK")
    # 2026-07-21 fix: the old regex hardcoded "max ΣF=1(.0+)?" -- true for moon3d
    # (whose tight concave micro-crater geometry saturates at least one row against
    # the ΣF<=1 cap) but NOT a general invariant. Haworth's r32c core is much more
    # open (mostly sky-facing) and legitimately never approaches the cap (observed
    # max ΣF=0.2067, rows>1=0) -- the OLD regex false-failed that healthy build.
    # The actual invariants are: rows>1=0 (no cap violation) AND max ΣF<=1
    # (numerically, not exactly "==1"). Parse and check both directly.
    m = re.search(r"REPLACED H-matrix .*max ΣF=([\d.]+),\s*rows>1=(\d+)", text)
    if not m:
        problems.append("corrected-VF invariants (max SigmaF<=1, rows>1=0) not confirmed")
    else:
        max_sigf, n_over1 = float(m.group(1)), int(m.group(2))
        if max_sigf > 1.0 + 1e-6:
            problems.append(f"corrected-VF max SigmaF={max_sigf} > 1")
        if n_over1 != 0:
            problems.append(f"corrected-VF rows>1={n_over1} (expected 0)")
    return problems


def gate_check_radoff(log_path: Path) -> list[str]:
    """RADOFF gate: a radiation-disabled run NEVER loads/builds/replaces a view
    factor matrix, so the coupled VF/two-band checks are STRUCTURALLY
    inapplicable (they aborted a healthy radoff chunk on 2026-07-21). Instead we
    assert radiation is GENUINELY off -- none of the coupling machinery engaged
    -- and rely on gate_check_common (already run per-attempt) for the
    illumination banners. Anything radiation-y appearing here means the run was
    NOT actually radoff (a misconfigured config), which we DO want to catch."""
    text = log_path.read_text(errors="replace")
    problems = []
    if re.search(r"Loaded view factors from:|Loaded H-matrix view factors:", text):
        problems.append("VF LOADED in a radoff run (radiation should be disabled)")
    if re.search(r"Building H-matrix view factor for|Computing view factors for", text):
        problems.append("VF BUILD started in a radoff run (radiation should be disabled)")
    if "REPLACED H-matrix with symmetric per-partner referee VF" in text:
        problems.append("corrected-VF replacement engaged in a radoff run")
    if re.search(r"two-band radiosity|indirect flux .*iteration", text):
        problems.append("two-band/indirect radiosity engaged in a radoff run")
    return problems


def resolve_gate_mode(gates: str, base_cfg: dict) -> str:
    """Return 'coupled' or 'radoff'. In 'auto', read whether inter-face
    radiation is enabled from the config (radiation.enabled, or the legacy
    radiation.enable_radiation_exchange). Defaults to 'coupled' (strict) when
    the key is absent so an unknown config is never under-gated."""
    if gates in ("coupled", "radoff"):
        return gates
    rad = base_cfg.get("radiation", {}) or {}
    enabled = rad.get("enabled", rad.get("enable_radiation_exchange", True))
    return "coupled" if bool(enabled) else "radoff"


def gate_check_by_mode(mode: str, log_path: Path) -> list[str]:
    return gate_check_radoff(log_path) if mode == "radoff" \
        else gate_check_radiation(log_path)


def _engine_default_coupling_iterations() -> int | None:
    """The engine's own default for radiation.max_coupling_iterations.

    Read from src/python/ptm/engine_defaults.py, which is GENERATED from
    src/cpp/tools/config_parser.{h,cpp} and re-derived by
    src/python/tests/test_config_parity.py on every run -- so this number
    cannot drift away from the compiled default the way a literal here would.
    Returns None if the package is not importable; the caller then says so
    rather than silently skipping the check.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "python"))
        from ptm.engine_defaults import ENGINE_DEFAULTS  # noqa: PLC0415
        v = ENGINE_DEFAULTS.get("radiation.max_coupling_iterations")
        return int(v) if v is not None else None
    except Exception:
        return None


def check_coupling_iterations(base_cfg: dict, gate_mode: str) -> tuple[int | None, list[str]]:
    """Report the base config's Picard cap and WARN when it is below the engine
    default.

    A campaign inherits every radiation knob from its base config byte for byte,
    so a k_max chosen once by habit propagates to every later generation of that
    config unnoticed. The moon3d year-long lineage carried
    `max_coupling_iterations: 5` from 2026-07 through the V2 launch -- HALF the
    then-default 10, and a third of what that mesh needs: measured truncation on
    the 509k flagship at 5 sweeps is +26.2 K on the worst face, one-signed cold.

    This warns and returns; it does not override. The engine prints the
    spectrum-derived sweep count for the actual view-factor operator, which is
    the authoritative check -- this one fires at launch, before 3 hours of GPU
    time, on the value the config chose.
    """
    if gate_mode == "radoff":
        return None, []
    rad = base_cfg.get("radiation", {}) or {}
    k = rad.get("max_coupling_iterations")
    engine_default = _engine_default_coupling_iterations()
    notes: list[str] = []
    if engine_default is None:
        notes.append("could not import ptm.engine_defaults; the Picard cap was NOT checked "
                     "against the engine default")
        return (int(k) if k is not None else None), notes
    if k is None:
        notes.append(f"radiation.max_coupling_iterations is unset -- the run inherits the "
                     f"engine default {engine_default}")
        return None, notes
    k = int(k)
    if k < engine_default:
        notes.append(
            f"radiation.max_coupling_iterations = {k} is BELOW the engine default "
            f"{engine_default}. The coupling loop truncates on the most enclosed faces and "
            f"the error is one-signed (a capped run is COLD there: measured +26.2 K on the "
            f"worst face of the 509k lunar flagship at 5 sweeps). Drop the key to inherit "
            f"{engine_default}, or set it deliberately from the engine's own "
            f"'[rho-advisory]' line. NOT overridden -- this run uses {k}.")
    return k, notes


# --------------------------------------------------------------------------
# Chunk YAML generation (mirrors run_chunked_simulation.make_chunk_yaml,
# plus the per-stage gpu_batch_size override that script doesn't support)
# --------------------------------------------------------------------------

def make_stage_yaml(base_cfg: dict, chunk_dir: Path, start_time: dt.datetime,
                    duration_days: float, timesteps: int,
                    equilibration_orbits, gpu_batch_size: int,
                    initial_state_file, final_state_file: Path,
                    skip_time_series: bool = True,
                    start_time_offset_s: float | None = None,
                    checkpoint_lookahead_steps: int = 0) -> Path:
    """`start_time_offset_s` not None: leave simulation.start_time as the base
    config has it and place the chunk with simulation.start_time_offset_seconds
    (ET seconds, absolute -- it REPLACES any base offset; the caller folds the
    base offset in). None: legacy UTC-string placement via `start_time`, still
    used by the calibrate probe, which has no successor.

    `checkpoint_lookahead_steps`: written to thermal.checkpoint_lookahead_steps
    (config_parser.cpp:702) so final_state_file is the state at the NEXT
    chunk's first epoch. 0 leaves the key out entirely."""
    cfg = _deep_copy(base_cfg)
    sim = cfg.setdefault("simulation", {})
    if start_time_offset_s is None:
        sim["start_time"] = fmt_iso8601(start_time)
    else:
        sim["start_time_offset_seconds"] = float(start_time_offset_s)
    sim["duration_days"] = float(duration_days)
    sim["timesteps"] = int(timesteps)

    th = cfg.setdefault("thermal", {})
    if equilibration_orbits is not None:
        th["equilibration_orbits"] = float(equilibration_orbits)
    th["gpu_batch_size"] = int(gpu_batch_size)
    if initial_state_file is not None:
        th["initial_state_file"] = str(initial_state_file)
    else:
        th.pop("initial_state_file", None)
    th["final_state_file"] = str(final_state_file)
    if checkpoint_lookahead_steps:
        th["checkpoint_lookahead_steps"] = int(checkpoint_lookahead_steps)
    else:
        th.pop("checkpoint_lookahead_steps", None)

    out = cfg.setdefault("output", {})
    out["directory"] = str(chunk_dir)
    out["write_raw_stats"] = True
    # 2026-07-21 (Luna): was hardcoded True unconditionally -- silently
    # discarded a base config's own `output.skip_time_series: false` with NO
    # warning, which would have defeated a rigorous (per-step-frame) Diviner
    # comparison run without any indication the frames were never written.
    # Default True preserves every existing caller's behavior exactly
    # (moon3d yearlong, Haworth production both rely on the old hardcoded
    # True); pass skip_time_series=False (via the new --keep-time-series CLI
    # flag) to retain per-step frames for BOTH stages.
    out["skip_time_series"] = bool(skip_time_series)
    out["skip_mesh"] = (initial_state_file is not None)

    chunk_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = chunk_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def resume_epoch_problems(chunk_dir: Path, offset_s: float, lookahead: int) -> list[str]:
    """Why a COMPLETE chunk on disk must not be reused by this plan.

    A chunk written by the pre-2026-10-12 driver has no look-ahead and a
    UTC-shifted start: its checkpoint is one cadence stale. Reusing it would
    hand that stale state to the successor, which the engine only WARNS about
    when the gap is exactly one cadence -- i.e. the seam would come back with
    nothing but a log line. So the resume path compares the chunk's own
    config.yaml against this plan and refuses on any difference."""
    cfg_path = chunk_dir / "config.yaml"
    if not cfg_path.is_file():
        return [f"{cfg_path} missing -- cannot verify how this chunk was placed"]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    sim = cfg.get("simulation", {}) or {}
    th = cfg.get("thermal", {}) or {}
    problems = []
    have_la = int(th.get("checkpoint_lookahead_steps", 0) or 0)
    if have_la != lookahead:
        problems.append(f"thermal.checkpoint_lookahead_steps={have_la}, plan wants "
                        f"{lookahead} (a chunk without it checkpoints one cadence stale)")
    have_off = sim.get("start_time_offset_seconds")
    if have_off is None or abs(float(have_off) - offset_s) > 1e-6:
        problems.append(f"simulation.start_time_offset_seconds={have_off}, plan wants "
                        f"{offset_s:.6f} (chunk was placed on a different epoch grid)")
    return problems


def gate_check_handoff(log_path: Path) -> list[str]:
    """A warm-started chunk must have restored its predecessor's state with the
    epoch clause passing OUTRIGHT. The engine refuses a gap of more than one
    cadence, but only WARNS on exactly one cadence (thermal_state_io.h, the
    'epoch clause'); that warning is the stale seam this driver used to
    produce, so here it is a failure."""
    text = log_path.read_text(errors="replace")
    problems = []
    if "identity trailer verified" not in text:
        problems.append("checkpoint restore did not report 'identity trailer verified'")
    if re.search(r"WARNING: epoch:", text):
        problems.append("restored checkpoint is off this chunk's first epoch "
                        "(engine 'WARNING: epoch:' -- stale hand-off)")
    return problems


# --------------------------------------------------------------------------
# Retry-wrapped invocation (same backoff schedule as run_chunked_simulation.py)
# --------------------------------------------------------------------------

def run_with_retry(ptm_bin: Path, cfg_path: Path, log_path: Path,
                    retries: int, timeout_min: float, ledger: Ledger, label: str,
                    expect_hole_seal_count: int | None = 30):
    """NOTE (2026-07-20, post-mortem on the batch=200000 stage-B wedge): each
    attempt's log is now written to a PER-ATTEMPT file (log_path with
    '_attemptN' inserted before the suffix) and log_path itself is left
    pointing at the LATEST attempt (copied, not just referenced) -- the
    original version overwrote log_path on every retry, which destroyed
    attempt 0's diagnostic content before it could be inspected (exactly
    what happened live: attempt 0's crash detail was unrecoverable, and we
    only had attempt 1's because it happened to still be running when
    checked). This is the fix for that gap, not just this run's fallback."""
    attempt = 0
    timeout_s = timeout_min * 60.0 if timeout_min and timeout_min > 0 else None
    t0 = time.time()
    while attempt <= retries:
        if attempt > 0:
            cooldown = min(600, 120 * (2 ** (attempt - 1)))
            print(f"[{label}] retry {attempt}/{retries}; waiting {cooldown}s for GPU to settle...",
                  flush=True)
            ledger.record(stage=label, event="retry_wait", attempt=attempt, cooldown_s=cooldown)
            time.sleep(cooldown)
        ledger.record(stage=label, event="attempt_start", attempt=attempt)
        attempt_log = log_path.with_name(f"{log_path.stem}_attempt{attempt}{log_path.suffix}")
        try:
            with open(attempt_log, "w") as lf:
                ret = subprocess.run([str(ptm_bin), "--config", str(cfg_path)],
                                     stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # A timeout is NOT a GPU fault: the job was still computing when the
            # wall-clock budget ran out. Retrying the identical job cannot
            # succeed (2026-10-13: three 90-min timeouts of a healthy ~1.9 h
            # Moon3D annual chunk_000 burned ~4.5 h). Fail fast so the caller
            # re-plans the budget from measured rates; retries stay reserved
            # for real failures (rc != 0 / gate problems).
            ledger.record(stage=label, event="attempt_timeout", attempt=attempt,
                          log=str(attempt_log), timeout_min=timeout_min,
                          retry="not retried: timeout is not a GPU fault")
            print(f"[{label}] TIMEOUT after {timeout_min} min -- NOT retrying "
                  f"(raise --chunk-timeout-min from a measured estimate).", flush=True)
            return False
        import shutil
        shutil.copy2(attempt_log, log_path)
        problems = gate_check_common(attempt_log, expect_hole_seal_count=expect_hole_seal_count)
        if ret.returncode == 0 and not problems:
            ledger.record(stage=label, event="attempt_ok", attempt=attempt,
                          wall_s=round(time.time() - t0, 1), log=str(attempt_log))
            return True
        ledger.record(stage=label, event="attempt_failed", attempt=attempt,
                      rc=ret.returncode, problems=problems, log=str(attempt_log))
        attempt += 1
    return False


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def _export_radiance_prune_eps(base_cfg: dict) -> None:
    """Propagate radiation.radiance_prune_eps from the config into the environment
    as PTM_RADIANCE_PRUNE_EPS so any VF-ASSEMBLY step this runner invokes
    (scripts/ptm12_diag/assemble_vf_haworth.py reads that env var as its default)
    sees it. The sim itself does not prune -- it only logs the value.

    Precedence: an already-set env var WINS over the config (require-env
    compatible); we fill it from the config ONLY when unset. To keep the default
    case byte-for-byte unchanged we export NOTHING when the config value is 0.0
    (off). Every override/mismatch is reported so the choice is never silent.
    """
    rad = base_cfg.get("radiation", {}) or {}
    try:
        cfg_eps = float(rad.get("radiance_prune_eps", 0.0) or 0.0)
    except (TypeError, ValueError):
        print("[run] WARNING: radiation.radiance_prune_eps is not a number; "
              "ignoring (no PTM_RADIANCE_PRUNE_EPS exported).", flush=True)
        return
    env_eps = os.environ.get("PTM_RADIANCE_PRUNE_EPS")
    if env_eps not in (None, ""):
        try:
            env_val = float(env_eps)
        except ValueError:
            env_val = None
        if cfg_eps > 0.0 and (env_val is None or env_val != cfg_eps):
            print(f"[run] NOTE: PTM_RADIANCE_PRUNE_EPS={env_eps} is already set and "
                  f"OVERRIDES radiation.radiance_prune_eps={cfg_eps} from the config "
                  f"(env wins at VF-assembly time).", flush=True)
        return
    if cfg_eps > 0.0:
        os.environ["PTM_RADIANCE_PRUNE_EPS"] = repr(cfg_eps)
        print(f"[run] exporting PTM_RADIANCE_PRUNE_EPS={cfg_eps} from "
              f"radiation.radiance_prune_eps for any VF-assembly step "
              f"(sim logs only; assembly prunes).", flush=True)


def mode_calibrate(args):
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    _export_radiance_prune_eps(base_cfg)
    sim = base_cfg.get("simulation", {})
    start = parse_iso8601(str(sim.get("start_time", "2030-09-01T00:00:00")))
    output_steps_per_day = float(sim.get("timesteps", 1)) / float(sim.get("duration_days", 1))

    probe_orbits = args.probe_orbits
    probe_out_steps = args.probe_output_steps
    probe_days = probe_out_steps / output_steps_per_day

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = make_stage_yaml(
        base_cfg, args.output_dir / "probe", start, probe_days, probe_out_steps,
        equilibration_orbits=probe_orbits, gpu_batch_size=args.candidate_batch_size,
        initial_state_file=None, final_state_file=args.output_dir / "probe_state.bin")
    log_path = args.output_dir / "probe.log"
    print(f"[calibrate] running {probe_orbits}-orbit probe at gpu_batch_size="
          f"{args.candidate_batch_size} ...", flush=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        ret = subprocess.run([str(args.ptm_bin), "--config", str(cfg_path)],
                             stdout=lf, stderr=subprocess.STDOUT,
                             timeout=args.probe_timeout_min * 60.0)
    wall = time.time() - t0
    cycles, nlayers, hung = scrape_cycle_stats(log_path)
    print(f"[calibrate] probe rc={ret.returncode} wall={wall:.1f}s n_layers={nlayers} "
          f"hung={hung}")
    if not cycles:
        sys.exit(f"[calibrate] no 'Persistent kernel:' lines found in {log_path} -- "
                  "probe did not reach the GPU thermal phase; inspect the log.")
    for tc, nt, ms in cycles:
        print(f"  reported: {tc} cycles x {nt} timesteps in {ms:.0f} ms "
              f"({ms/max(1,tc):.1f} ms/cycle)")

    # Use the EQUILIBRATION-phase report (the one whose n_timesteps matches
    # the probe's equilibration step count) to get ms/cycle at n_layers.
    probe_eq_steps = int(round(probe_orbits * 120))  # matches
    # equilibration_samples_per_diurnal_cycle=120 in this project's configs
    eq_reports = [c for c in cycles if c[1] == probe_eq_steps]
    if not eq_reports:
        print("[calibrate] WARNING: could not isolate the equilibration-phase "
              "report by timestep count; using the FIRST reported cycle block.")
        eq_reports = cycles[:1]
    tc, nt, ms = eq_reports[0]
    ms_per_step_per_batch = ms / tc / nt  # ms per (cycle-equivalent, timestep), ONE batch

    # --- FIX (2026-07-20, post-mortem on the live stage-B wedge) ---
    # The original version stopped here and reported `ms_per_step_per_batch
    # * real_total_cycles * real_eq_steps` as "the projected dispatch
    # duration" -- that is the kernel time for ONE batch's dispatch, silently
    # assuming a single batch covers the whole mesh. It does not: at
    # candidate_batch_size=8000 the 509,337-face mesh needs
    # ceil(n_faces/batch) batches, and empirically (stage A's real,
    # completed run) each batch ALSO carries substantial per-batch overhead
    # that has nothing to do with kernel time (uploading/packing per-batch
    # buffers, thread-pool bookkeeping, etc.) -- measured directly from
    # chunk_000.log: 64 batches, kernel time summed across all batches
    # =687.7 s, but TOTAL GPU-thermal-phase wall time implied by the
    # attempt's 3833.7 s minus its own ~167 s illumination and ~small
    # radiation-coupling-for-20-steps tail was closer to ~3600 s -- i.e.
    # ~46.5 s/batch of non-kernel overhead. This calibration mode's own
    # PROCEED verdict on THIS run (8.9 s projected) was checked against a
    # probe whose OWN measured wall time was 616 s for the same 64 batches
    # -- a ~69x discrepancy that should have been the tell. Fixed here: the
    # projection now (a) multiplies by n_batches, and (b) adds a per-batch
    # overhead term measured from the PROBE's own wall time (not the
    # broken single-batch kernel estimate), rather than pretending overhead
    # is zero.
    n_faces_total = 509337  # this mesh; the probe estimator doesn't need the
                             # exact mesh path parsed, this campaign is fixed
    n_batches = -(-n_faces_total // args.candidate_batch_size)  # ceil div
    probe_kernel_ms_all_batches = sum(m for tc_, nt_, m in eq_reports)
    probe_kernel_s = probe_kernel_ms_all_batches / 1000.0
    # Attribute everything in probe_wall_s that ISN'T eq-phase kernel time
    # to "other" (illumination + radiation-coupling-for-the-tiny-output-
    # slice + per-batch overhead, all lumped) -- we cannot cleanly separate
    # per-batch overhead from illumination/coupling in a probe this small,
    # so this is a CONSERVATIVE (upper-bound-ish for overhead, since some of
    # "other" is fixed illum/coupling cost that does NOT scale with
    # timesteps) per-batch overhead estimate. State this explicitly.
    other_s = max(0.0, wall - probe_kernel_s)
    overhead_per_batch_s = other_s / n_batches

    real_eq_steps = args.real_orbits * 120
    real_total_cycles = 2  # matches the observed n_equil_cycles=1 pattern
    kernel_s_per_batch_real = (ms_per_step_per_batch * real_total_cycles * real_eq_steps) / 1000.0
    projected_s = n_batches * (kernel_s_per_batch_real + overhead_per_batch_s)

    print(f"\n[calibrate] measured {ms_per_step_per_batch*1000:.4f} us/(cycle-timestep), "
          f"ONE batch, at batch_size={args.candidate_batch_size}, n_layers={nlayers}")
    print(f"[calibrate] n_batches for the full {n_faces_total}-face mesh: {n_batches}")
    print(f"[calibrate] probe: wall={wall:.1f}s, summed eq-kernel time across all "
          f"batches={probe_kernel_s:.1f}s, other (illum+coupling+per-batch "
          f"overhead)={other_s:.1f}s -> ~{overhead_per_batch_s:.1f} s/batch overhead "
          f"(conservative lump; some of 'other' is fixed, not per-batch)")
    print(f"[calibrate] PROJECTED stage duration ({args.real_orbits} orbits = "
          f"{real_eq_steps} steps, {real_total_cycles} total cycles, {n_batches} "
          f"batches): ~{projected_s:.1f} s "
          f"(kernel {n_batches*kernel_s_per_batch_real:.1f}s + overhead "
          f"{n_batches*overhead_per_batch_s:.1f}s)")
    verdict = "PROCEED" if projected_s <= args.safe_ceiling_s else "SHRINK batch size further"
    print(f"[calibrate] NOTE: --safe-ceiling-s is a PER-BATCH dispatch-duration "
          f"ceiling (memory gpu_hang_amd_macos.md), not a whole-stage wall-time "
          f"budget; compare it against (kernel_s_per_batch_real + "
          f"overhead_per_batch_s)={kernel_s_per_batch_real+overhead_per_batch_s:.1f}s, "
          f"not against projected_s.")
    per_batch_verdict_s = kernel_s_per_batch_real + overhead_per_batch_s
    verdict = "PROCEED" if per_batch_verdict_s <= args.safe_ceiling_s else "SHRINK batch size further"
    print(f"[calibrate] safe ceiling = {args.safe_ceiling_s:.0f} s per batch -> VERDICT: {verdict} "
          f"(per-batch projected {per_batch_verdict_s:.1f}s)")
    report = dict(candidate_batch_size=args.candidate_batch_size, n_layers=nlayers,
                  probe_wall_s=wall, n_batches=n_batches,
                  probe_kernel_s_all_batches=probe_kernel_s, overhead_per_batch_s=overhead_per_batch_s,
                  projected_whole_stage_s=projected_s, per_batch_projected_s=per_batch_verdict_s,
                  safe_ceiling_s=args.safe_ceiling_s, verdict=verdict, hung=hung)
    (args.output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output_dir / 'calibration_report.json'}")
    return 0 if verdict == "PROCEED" else 1


def mode_run(args):
    # Phase R (R3): pre-launch ENV-COMPLETENESS gate. Runs BEFORE any launch so
    # a bare relaunch that dropped the physics env fails at t=0, not 4.3 h in.
    manifest = _parse_require_env(getattr(args, "require_env", ""))
    if manifest:
        print(f"[run] ENV-COMPLETENESS gate: requiring {sorted(manifest)}", flush=True)
        env_problems = guard.check_env_completeness(manifest)
        if env_problems:
            sys.exit("[run] ENV-COMPLETENESS gate FAILED before launch "
                     "(Phase R, R3) -- refusing to start a long run with an "
                     "incomplete physics env:\n  - " + "\n  - ".join(env_problems))
        print("[run] ENV-COMPLETENESS gate PASSED.", flush=True)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    _export_radiance_prune_eps(base_cfg)
    sim = base_cfg.get("simulation", {})
    start = parse_iso8601(str(sim.get("start_time", "2030-09-01T00:00:00")))
    total_output_steps = int(sim.get("timesteps"))
    output_steps_per_day = float(total_output_steps) / float(sim.get("duration_days"))
    eq_orbits = base_cfg.get("thermal", {}).get("equilibration_orbits")

    # Post-chunk radiation gate mode. 'auto' reads radiation.enabled from the
    # config: a radoff run loads no VF, so the coupled VF/two-band gates are
    # structurally inapplicable and must be skipped (they aborted a healthy
    # radoff chunk on 2026-07-21).
    gate_mode = resolve_gate_mode(getattr(args, "gates", "auto"), base_cfg)
    print(f"[run] radiation gate mode: {gate_mode} "
          f"(--gates {getattr(args, 'gates', 'auto')})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(args.output_dir / "chunk_ledger.json")

    # Picard cap disclosure. Printed for every coupled launch, warned about when
    # the config pins it below the engine default, and recorded either way so a
    # finished campaign carries the number it actually ran with.
    k_cap, k_notes = check_coupling_iterations(base_cfg, gate_mode)
    if gate_mode != "radoff":
        print(f"[run] radiation.max_coupling_iterations: "
              f"{k_cap if k_cap is not None else 'unset (engine default)'}", flush=True)
        for note in k_notes:
            print(f"[run] WARNING: {note}", flush=True)
    ledger.record(event="coupling_iterations", gate_mode=gate_mode,
                  max_coupling_iterations=k_cap, warnings=k_notes)

    chunk_a = args.output_dir / "chunk_000"
    chunk_b = args.output_dir / "chunk_001"
    state_dir = args.output_dir / "_thermal_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_a = state_dir / "thermal_state_after_chunk_000.bin"
    state_b = state_dir / "thermal_state_after_chunk_001.bin"

    a_out_steps = args.stage_a_output_steps
    a_days = a_out_steps / output_steps_per_day
    b_out_steps_total = total_output_steps - a_out_steps

    # Half-open ET grid (see "EPOCH GRID / HAND-OFF" in the module docstring).
    # Offsets are INTEGER step counts times one cadence, converted once, so no
    # chunk start inherits a running floating-point sum. A base config's own
    # start_time_offset_seconds is folded in, not overwritten.
    cadence_s = float(sim.get("duration_days")) * 86400.0 / total_output_steps
    base_offset_s = float(sim.get("start_time_offset_seconds", 0.0) or 0.0)
    lookahead = CHECKPOINT_LOOKAHEAD_STEPS

    def offset_for(steps_before: int) -> float:
        return base_offset_s + steps_before * cadence_s

    # Stage B is split into N sequential sub-chunks (chunk_001..chunk_00N),
    # each chained via the SAME checkpoint mechanism as stage A -> stage B,
    # each using stage_b_batch_size (default now 8000 -- the PROVEN-safe
    # value from stage A's own successful run, not the 200000 that wedged
    # the FIRST batch of a single-chunk stage B twice in a row, deterministically,
    # at ~213-214s both times -- see the 2026-07-20 post-mortem in the
    # session note). N>1 also bounds the blast radius of any future wedge to
    # 1/N of stage B's work instead of losing the whole stage.
    nb = max(1, args.stage_b_nchunks)
    base_steps = b_out_steps_total // nb
    extra = b_out_steps_total - base_steps * nb
    steps_list = [base_steps + (1 if i < extra else 0) for i in range(nb)]

    a_offset_s = offset_for(0)
    print(f"[run] plan: chunk_000 = {eq_orbits}-orbit equilibration + "
          f"{a_out_steps} output ts ({a_days:.3f} d, steps [0, {a_out_steps}), "
          f"start +{a_offset_s:.6f} s ET) @ batch={args.stage_a_batch_size}")
    cum_steps = a_out_steps
    b_plan = []
    for i, steps in enumerate(steps_list):
        days = steps / output_steps_per_day
        off = offset_for(cum_steps)
        b_plan.append((i + 1, steps, days, off))
        # The UTC string is DISPLAY ONLY; the engine is placed by the ET offset.
        st = start + dt.timedelta(seconds=off)
        print(f"[run] plan: chunk_{i+1:03d} = {steps} output ts ({days:.3f} d, steps "
              f"[{cum_steps}, {cum_steps + steps}), start +{off:.6f} s ET "
              f"~ {fmt_iso8601(st)}) @ batch={args.stage_b_batch_size}")
        cum_steps += steps
    print(f"[run] plan: merged grid = {cum_steps} half-open output epochs at "
          f"{cadence_s:.6f} s cadence; every chunk integrates "
          f"{lookahead} look-ahead step(s) past its window (not exported, not "
          f"reduced) and checkpoints there, on its successor's first epoch.")
    if cum_steps != total_output_steps:
        sys.exit(f"[run] internal error: plan covers {cum_steps} output steps, "
                 f"config asks for {total_output_steps}")
    ledger.record(event="plan", chunk_000_out_steps=a_out_steps,
                  stage_b_nchunks=nb, stage_b_out_steps_total=b_out_steps_total,
                  eq_orbits=eq_orbits, stage_a_batch=args.stage_a_batch_size,
                  stage_b_batch=args.stage_b_batch_size,
                  cadence_s=cadence_s, checkpoint_lookahead_steps=lookahead,
                  start_offsets_s=[a_offset_s] + [p[3] for p in b_plan])

    # Phase R (R3d): launch-time consistency check vs the host machine profile.
    # The solver's own self-chunking (R2) enforces the timestep ceiling, so here
    # we only flag batch sizes above the host's proven-safe recommendation
    # (the 200000-batch stage-B wedge class). Advisory by default; --strict-
    # preflight makes it fatal.
    prof = guard.load_machine_profile()
    if prof is None:
        print("[run] no GPU machine profile found (run `ptm_thermal_sim "
              "--calibrate-gpu`); preflight uses conservative defaults.", flush=True)
    for label, batch in (("stage-a", args.stage_a_batch_size),
                         ("stage-b", args.stage_b_batch_size)):
        problems = guard.preflight_consistency(
            timeout_s=args.chunk_timeout_min * 60.0,
            est_runtime_s=0.0,  # per-chunk runtime not modeled here; batch is the lever
            batch_size=batch, max_timesteps_per_dispatch=0, profile=prof)
        for pr in problems:
            msg = f"[run] PREFLIGHT ({label}): {pr}"
            if args.strict_preflight:
                ledger.record(event="ABORT", reason=msg)
                sys.exit(msg)
            print(msg + "  (advisory; pass --strict-preflight to make fatal)", flush=True)

    if args.dry_run:
        return 0

    # --- Resume consistency (before ANY launch) ---
    # A complete chunk on disk is reused only if it was placed on THIS plan's
    # epoch grid with the look-ahead; otherwise its checkpoint is stale.
    resume_bad = []
    for cdir, off in [(chunk_a, a_offset_s)] + [
            (args.output_dir / f"chunk_{idx:03d}", off) for idx, _s, _d, off in b_plan]:
        if chunk_is_complete(cdir):
            for pr in resume_epoch_problems(cdir, off, lookahead):
                resume_bad.append(f"{cdir.name}: {pr}")
    if resume_bad:
        ledger.record(event="ABORT", reason="stale chunks on disk", problems=resume_bad)
        sys.exit("[run] REFUSING to resume: complete chunk(s) on disk were not "
                 "written on this plan's epoch grid with the checkpoint "
                 "look-ahead, so their hand-off states are stale:\n  - "
                 + "\n  - ".join(resume_bad)
                 + "\nMove those chunk_NNN/ dirs and _thermal_state/ aside and relaunch.")

    # --- Stage A ---
    if not (chunk_is_complete(chunk_a) and is_valid_checkpoint(state_a)):
        if chunk_a.exists() and not any(chunk_a.iterdir()):
            pass  # empty stub from a prior crashed attempt -- fine to reuse
        cfg_a = make_stage_yaml(base_cfg, chunk_a, start, a_days, a_out_steps,
                                equilibration_orbits=eq_orbits,
                                gpu_batch_size=args.stage_a_batch_size,
                                initial_state_file=None, final_state_file=state_a,
                                skip_time_series=not args.keep_time_series,
                                start_time_offset_s=a_offset_s,
                                checkpoint_lookahead_steps=lookahead)
        ok = run_with_retry(args.ptm_bin, cfg_a, args.output_dir / "chunk_000.log",
                            args.retries, args.chunk_timeout_min, ledger, "chunk_000",
                            expect_hole_seal_count=args.expect_hole_seal_count)
        if not ok or not is_valid_checkpoint(state_a) or not chunk_is_complete(chunk_a):
            ledger.record(event="ABORT", reason="chunk_000 failed or produced no valid checkpoint")
            sys.exit("[run] chunk_000 (equilibration stage) FAILED after retries -- see log + ledger.")
        rad_problems = gate_check_by_mode(gate_mode, args.output_dir / "chunk_000.log")
        if rad_problems:
            ledger.record(event="ABORT", reason=f"chunk_000 radiation gates failed: {rad_problems}")
            sys.exit(f"[run] chunk_000 radiation gate FAILED: {rad_problems}")
        ledger.record(event="chunk_000_done", checkpoint=str(state_a))
    else:
        print("[run] chunk_000 already complete (resume, REUSING the already-valid "
              f"checkpoint {state_a}) -- skipping.")

    # --- Stage B, N sub-chunks ---
    prev_state = state_a
    all_chunk_dirs = [chunk_a]
    for idx, steps, days, off in b_plan:
        cdir = args.output_dir / f"chunk_{idx:03d}"
        this_state = state_dir / f"thermal_state_after_chunk_{idx:03d}.bin"
        all_chunk_dirs.append(cdir)
        label = f"chunk_{idx:03d}"
        if chunk_is_complete(cdir):
            if not is_valid_checkpoint(this_state):
                ledger.record(event="ABORT", reason=f"{label} complete but checkpoint invalid")
                sys.exit(f"[run] {label} is complete but {this_state} is not a valid "
                         "checkpoint; its successor cannot be chained. Move the chunk "
                         "aside and relaunch.")
            print(f"[run] {label} already complete (resume, placed on this plan's "
                  "grid with the look-ahead) -- skipping.")
            prev_state = this_state
            continue
        cfg_b = make_stage_yaml(base_cfg, cdir, start, days, steps,
                                equilibration_orbits=None,  # inherited but will be
                                                            # force-zeroed by the code
                                gpu_batch_size=args.stage_b_batch_size,
                                initial_state_file=prev_state, final_state_file=this_state,
                                skip_time_series=not args.keep_time_series,
                                start_time_offset_s=off,
                                checkpoint_lookahead_steps=lookahead)
        ok = run_with_retry(args.ptm_bin, cfg_b, args.output_dir / f"{label}.log",
                            args.retries, args.chunk_timeout_min, ledger, label,
                            expect_hole_seal_count=args.expect_hole_seal_count)
        if not ok or not chunk_is_complete(cdir):
            ledger.record(event="ABORT", reason=f"{label} failed")
            sys.exit(f"[run] {label} (output stage) FAILED after retries -- see log + ledger.")
        rad_problems = gate_check_by_mode(gate_mode, args.output_dir / f"{label}.log")
        rad_problems += gate_check_handoff(args.output_dir / f"{label}.log")
        if rad_problems:
            ledger.record(event="ABORT", reason=f"{label} radiation gates failed: {rad_problems}")
            sys.exit(f"[run] {label} radiation gate FAILED: {rad_problems}")
        ledger.record(event=f"{label}_done", checkpoint=str(this_state))
        prev_state = this_state

    # --- Merge ---
    print(f"[run] merging {len(all_chunk_dirs)} chunk(s) into {args.output_dir}/statistics/ ...")
    ret = subprocess.run([str(args.merge_bin), "--chunks",
                         *[str(c) for c in all_chunk_dirs],
                         "--output", str(args.output_dir)])
    if ret.returncode != 0:
        ledger.record(event="ABORT", reason="merge failed")
        sys.exit("[run] merge FAILED.")
    ledger.record(event="merge_done")

    # mesh.ply + time_steps.txt for downstream consumers (mirrors run_chunked_simulation.py)
    src_mesh = chunk_a / "mesh.ply"
    dst_mesh = args.output_dir / "mesh.ply"
    if src_mesh.exists() and not dst_mesh.exists():
        import shutil
        try:
            import os
            os.link(src_mesh, dst_mesh)
        except OSError:
            shutil.copy2(src_mesh, dst_mesh)

    print(f"\n[run] DONE. output in {args.output_dir}")
    ledger.record(event="ALL_DONE")
    return 0


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--mode", required=True, choices=["calibrate", "run"])
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--ptm-bin", type=Path, required=True)
    p.add_argument("--merge-bin", type=Path)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--chunk-timeout-min", type=float, default=90.0)
    p.add_argument("--dry-run", action="store_true")
    # Phase R (R3): pre-launch ENV-COMPLETENESS gate. Comma-separated list of
    # required physics env vars; a token 'NAME=VAL' pins the exact value. Empty
    # (default) = gate OFF = current behavior. RECOMMENDED for this campaign:
    #   --require-env PTM_ILLUM_TRUE_NORMALS,PTM_ILLUM_RISE_KEPS=0.2,\
    #                 PTM_ILLUM_RISE_HABS=0,PTM_ILLUM_SEAL_HOLES=30
    # This is the pre-launch form of the log-banner gate in gate_check_common;
    # it stops the 2026-07-20 dropped-physics-env relaunch BEFORE it burns hours.
    p.add_argument("--require-env", type=str, default="",
                   help="comma-separated required physics env (NAME or NAME=VAL); "
                        "refuses to launch if any are missing/mismatched")
    p.add_argument("--strict-preflight", action="store_true",
                   help="ABORT (not warn) on a launch-time consistency "
                        "contradiction vs the machine profile (Phase R, R3d)")
    p.add_argument("--expect-hole-seal-count", type=int, default=30,
                   help="PTM-18 interior-hole-seal count the gate expects in the log banner "
                        "(mesh-specific -- moon3d seals 30; verify a new mesh's own count with "
                        "a dedicated probe run before trusting a different value here, "
                        "2026-07-21 finding). Pass -1 to skip this specific check entirely.")
    p.add_argument("--keep-time-series", action="store_true",
                   help="Write output.skip_time_series=false into BOTH stage configs "
                        "(default: false here, i.e. skip_time_series=true, matching the "
                        "prior hardcoded behavior byte-for-byte). Retains per-step frames "
                        "(temperature/, illumination/, etc.) needed for diviner_compare.py "
                        "--max-mode rigorous. WARNING: substantially increases disk usage "
                        "and is NOT stitched across chunks by ptm_merge_chunks (which only "
                        "merges statistics/_raw accumulators) -- point downstream tools at "
                        "the individual chunk dirs, or stitch frame files manually.")
    p.add_argument("--gates", choices=["coupled", "radoff", "auto"], default="auto",
                   help="post-chunk radiation gate set. 'coupled' = require the "
                        "VF/two-band banners (default for radiation-enabled runs); "
                        "'radoff' = a radiation-disabled run loads no VF, so assert "
                        "radiation is genuinely OFF and SKIP the VF gates; 'auto' "
                        "(default) reads radiation.enabled from the config.")
    # calibrate-mode args
    p.add_argument("--candidate-batch-size", type=int, default=8000)
    p.add_argument("--probe-orbits", type=float, default=2.0)
    p.add_argument("--probe-output-steps", type=int, default=10)
    p.add_argument("--probe-timeout-min", type=float, default=15.0)
    p.add_argument("--real-orbits", type=float, default=13.0)
    p.add_argument("--safe-ceiling-s", type=float, default=25.0,
                   help="Per-BATCH stage-duration verdict threshold for this "
                        "calibrate mode. NOTE: this is NOT the single-kernel "
                        "watchdog budget — that lives in C++ "
                        "(gpu/machine_profile.h kSafeDispatchCeilingSeconds, "
                        "2 s) and is enforced by the solver's own R2 "
                        "self-chunking, which splits a batch into many "
                        "dispatches. The old help here claimed '~15-30s "
                        "launches survive repeatedly'; that reading of memory "
                        "gpu_hang_amd_macos.md is WRONG and cost a run on "
                        "2026-08-16 (a 5.0 s launch was killed with "
                        "gpusKillClientExt).")
    # run-mode args
    p.add_argument("--stage-a-output-steps", type=int, default=20)
    p.add_argument("--stage-a-batch-size", type=int, default=8000)
    # 2026-07-20 post-mortem: 200000 (the old default, extrapolated from a
    # 360-step/n_layers=30 baseline never validated at n_layers=50) wedged
    # the FIRST batch of a single-chunk stage B TWICE, deterministically, at
    # ~213-214s both times, with zero "Persistent kernel:" completion lines
    # -- illumination finished at 157s and the process never got past batch
    # 1/3 before the watchdog killed it. 8000 is now the default because it
    # is the ONE value proven safe on this exact config (stage A succeeded
    # with an even heavier 1560-step x 2-cycle dispatch shape at that batch
    # size). Do not raise this without a fresh calibration probe.
    p.add_argument("--stage-b-batch-size", type=int, default=8000)
    p.add_argument("--stage-b-nchunks", type=int, default=1,
                   help="Split stage B's output steps into this many "
                        "sequential sub-chunks chained via checkpoints. "
                        "COST WARNING (measured from stage A, 2026-07-20): "
                        "at batch_size=8000 the mesh needs 64 batches, and "
                        "each batch carries ~43.6s of PER-BATCH overhead "
                        "unrelated to actual kernel time (64 x 43.6s = "
                        "~46.5 min PER STAGE-B INVOCATION, i.e. per chunk). "
                        "This overhead is invariant to how many timesteps "
                        "the invocation covers, so N sub-chunks pay it N "
                        "times: N=1 costs ~46.5 min total overhead, N=4 "
                        "costs ~186 min (~2.3h MORE than N=1). Default is "
                        "1 (minimize wall-clock; batch=8000 is independently "
                        "proven reliable from stage A's success). Raise to "
                        "2-4 only if bounding blast-radius against an "
                        "UNRELATED failure mode is worth ~45-140 extra "
                        "minutes -- it is not needed to fix the wedge itself.")
    args = p.parse_args()
    if args.expect_hole_seal_count is not None and args.expect_hole_seal_count < 0:
        args.expect_hole_seal_count = None  # -1 sentinel -> skip this specific check

    if not args.ptm_bin.is_file():
        sys.exit(f"error: ptm_thermal_sim not found at {args.ptm_bin}")
    if args.mode == "run" and (not args.merge_bin or not args.merge_bin.is_file()):
        sys.exit(f"error: ptm_merge_chunks not found at {args.merge_bin}")

    if args.mode == "calibrate":
        return mode_calibrate(args)
    return mode_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
