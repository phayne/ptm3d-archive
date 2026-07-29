#!/usr/bin/env python3
"""
Phase R runtime-robustness helpers for the PTM3D wedge-safe runners.

This module packages the runner-side half of the Phase R crash-prevention
package (plan §Phase R, items R3 + R5) as small, dependency-free, unit-tested
functions so every long-run driver (run_yearlong_wedge_safe.py,
run_chunked_simulation.py, the campaign orchestrator) enforces the SAME gates:

  R3a  check_env_completeness  -- the pre-launch ENV-COMPLETENESS gate. The
       07-20 incident burned 4.3 h because a bare relaunch dropped the physics
       env (PTM_ILLUM_TRUE_NORMALS / RISE_* / SEAL_HOLES ...); the ledger's log
       banners caught it only POST-hoc. This moves that check to BEFORE launch:
       refuse to start a multi-hour run whose required physics env is missing.

  R3b  derive_timeout_s        -- wall-clock ceilings are ONLY k x the planner's
       own runtime estimate, never hand-set below it (the 90-min-timeout-vs-3-h-
       chunk contradiction the plan calls out).

  R3c  StallWatchdog           -- the PRIMARY kill signal is a progress stall
       (no step / cputime advance for N minutes), so a healthy-but-slow run is
       timer-unkillable.

  R3d  preflight_consistency   -- estimate runtime / dispatch shape / batch vs
       the machine profile and ABORT LOUDLY at t=0 on a contradiction.

  R5   RetryLadder / deescalate_batch_size -- on an abort-class failure
       (rc=134 SIGABRT and the SIG* family) retries must STRICTLY reduce the
       batch/dispatch parameters; identical-parameter retries are forbidden.

Design rule (Phase R family): "compute what you're about to do, check it's
sane, refuse loudly if not." Nothing here silently substitutes a default; every
gate returns explicit problems or raises.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# R3a: pre-launch env-completeness gate
# ---------------------------------------------------------------------------

def check_env_completeness(
    required: Union[Sequence[str], Mapping[str, Optional[str]]],
    env: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Return a list of problems ([] == complete) for the required physics env.

    `required` is either:
      * a sequence of env-var NAMES     -> each must be set and non-empty, or
      * a mapping name -> expected value:
          - expected is None  -> name must be set and non-empty (any value)
          - expected is a str -> name must be set AND equal that exact string
                                 (this is how you pin PTM_ILLUM_RISE_KEPS=0.2
                                  etc. so a dropped-or-wrong value fails LOUD).
    """
    if env is None:
        env = os.environ
    problems: List[str] = []
    if isinstance(required, Mapping):
        items = required.items()
    else:
        items = [(name, None) for name in required]
    for name, expected in items:
        val = env.get(name)
        if val is None or val == "":
            problems.append(f"required env {name} is missing/empty")
        elif expected is not None and str(val) != str(expected):
            problems.append(
                f"env {name}={val!r} does not match required {expected!r}")
    return problems


def assert_env_complete(
    required: Union[Sequence[str], Mapping[str, Optional[str]]],
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Raise RuntimeError listing every missing/mismatched var. Call this at the
    top of a runner BEFORE any multi-hour launch."""
    problems = check_env_completeness(required, env)
    if problems:
        raise RuntimeError(
            "ENV-COMPLETENESS gate FAILED before launch (Phase R, R3) -- "
            "refusing to start a long run with an incomplete physics env:\n  - "
            + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# R3b: derived timeouts (single source of truth = the planner estimate)
# ---------------------------------------------------------------------------

def derive_timeout_s(
    planner_estimate_s: float,
    k: float = 3.0,
    floor_s: float = 600.0,
    hand_set_s: Optional[float] = None,
) -> float:
    """timeout = max(k * planner_estimate_s, floor_s).

    If `hand_set_s` is provided it is a REQUEST, not an override: a hand-set
    value BELOW the planner estimate is the 90-min-vs-3-h contradiction and
    raises. A hand-set value above the derived timeout is honored (more slack
    is always allowed)."""
    if planner_estimate_s <= 0:
        raise ValueError("planner_estimate_s must be > 0 (it is the single "
                         "source of truth for the timeout)")
    derived = max(k * planner_estimate_s, floor_s)
    if hand_set_s is not None:
        if hand_set_s < planner_estimate_s:
            raise ValueError(
                f"hand-set timeout {hand_set_s:.0f}s is BELOW the planner "
                f"runtime estimate {planner_estimate_s:.0f}s -- this is the "
                f"self-inflicted-timeout class; refusing. Raise the timeout or "
                f"fix the estimate.")
        return max(derived, hand_set_s)
    return derived


# ---------------------------------------------------------------------------
# R3c: stall-based watchdog (primary kill signal)
# ---------------------------------------------------------------------------

class StallWatchdog:
    """Track a monotonic progress signal; report a stall when it has not
    advanced for `stall_limit_s`. Feed it completed-step count, growing
    cputime, or output-file byte size -- anything that only increases while the
    run makes progress. A healthy-but-slow run keeps advancing the signal and
    is therefore timer-unkillable."""

    def __init__(self, stall_limit_s: float, now: Optional[float] = None):
        if stall_limit_s <= 0:
            raise ValueError("stall_limit_s must be > 0")
        self.stall_limit_s = float(stall_limit_s)
        self._last_progress = None  # type: Optional[float]
        self._last_advance_t = now if now is not None else time.time()

    def update(self, progress_value: float, now: Optional[float] = None) -> bool:
        """Record the latest progress value. Returns True if progress advanced
        (or this is the first sample), False if it stagnated."""
        t = now if now is not None else time.time()
        if self._last_progress is None or progress_value > self._last_progress:
            self._last_progress = progress_value
            self._last_advance_t = t
            return True
        return False

    def seconds_since_progress(self, now: Optional[float] = None) -> float:
        t = now if now is not None else time.time()
        return t - self._last_advance_t

    def is_stalled(self, now: Optional[float] = None) -> bool:
        return self.seconds_since_progress(now) >= self.stall_limit_s


# ---------------------------------------------------------------------------
# R3d: launch-time consistency check (abort at t=0 on contradictions)
# ---------------------------------------------------------------------------

def load_machine_profile(path: Optional[str] = None) -> Optional[dict]:
    """Read the per-host GPU profile written by `ptm_thermal_sim
    --calibrate-gpu` (or $PTM_MACHINE_PROFILE / ~/.ptm3d/machine_profile.json).
    Returns None if absent -- callers then fall back to conservative defaults
    (and should WARN, per the Phase R design rule)."""
    if path is None:
        path = os.environ.get("PTM_MACHINE_PROFILE")
        if not path:
            home = os.environ.get("HOME", "")
            path = str(Path(home) / ".ptm3d" / "machine_profile.json") if home else ""
    if not path or not Path(path).is_file():
        return None
    try:
        prof = json.loads(Path(path).read_text())
    except Exception:
        return None
    if prof.get("max_timesteps_per_dispatch", 0) <= 0:
        return None
    return prof


# Conservative defaults consumed when no profile exists (mirror machine_profile.h).
CONSERVATIVE_MAX_TIMESTEPS_PER_DISPATCH = 300
CONSERVATIVE_RECOMMENDED_BATCH = 8000


def preflight_consistency(
    *,
    timeout_s: float,
    est_runtime_s: float,
    batch_size: int,
    max_timesteps_per_dispatch: int,
    profile: Optional[dict] = None,
) -> List[str]:
    """Return a list of contradictions ([] == consistent) that should ABORT the
    launch at t=0. Catches the failure classes in the plan's crash taxonomy:

      * timeout below the estimated runtime (the 90-min-vs-3-h self-kill),
      * a per-dispatch timestep count above the machine's watchdog ceiling
        (would wedge the GPU deterministically),
      * a batch size above the machine's proven-safe recommendation.
    """
    problems: List[str] = []
    max_ts_ceiling = (profile or {}).get(
        "max_timesteps_per_dispatch", CONSERVATIVE_MAX_TIMESTEPS_PER_DISPATCH)
    rec_batch = (profile or {}).get(
        "recommended_batch_size", CONSERVATIVE_RECOMMENDED_BATCH)

    if timeout_s < est_runtime_s:
        problems.append(
            f"timeout {timeout_s:.0f}s is below the estimated runtime "
            f"{est_runtime_s:.0f}s -- the run would be killed while healthy")
    if max_timesteps_per_dispatch > max_ts_ceiling:
        problems.append(
            f"requested {max_timesteps_per_dispatch} timesteps/dispatch exceeds "
            f"the host ceiling {max_ts_ceiling} (would risk a gpusKillClientExt "
            f"wedge); reduce it or the solver's self-chunking will override it")
    if batch_size > rec_batch:
        problems.append(
            f"gpu_batch_size {batch_size} exceeds the host's proven-safe "
            f"recommendation {rec_batch}; the 200000-batch stage-B wedge on "
            f"2026-07-20 is exactly this")
    # F6 (caveat): the profiled ceiling/overhead were measured AT batch_size_probe;
    # a different batch changes per-batch kernel cost, so the profile may not apply.
    probe_batch = (profile or {}).get("batch_size_probe", 0)
    if probe_batch and batch_size != probe_batch:
        problems.append(
            f"gpu_batch_size {batch_size} != the profile's batch_size_probe "
            f"{probe_batch}; the calibrated ceiling/overhead were measured at the "
            f"probe batch and may not transfer. Re-run --calibrate-gpu with "
            f"--batch-size {batch_size} for an exact profile.")
    return problems


def assert_launch_consistent(**kw) -> None:
    """Raise on any launch-time contradiction. See preflight_consistency."""
    problems = preflight_consistency(**kw)
    if problems:
        raise RuntimeError(
            "LAUNCH-CONSISTENCY gate FAILED at t=0 (Phase R, R3):\n  - "
            + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# R5: de-escalating retry ladder
# ---------------------------------------------------------------------------

# Return codes that indicate an abort-class crash (SIGABRT/SIGSEGV/SIGBUS and
# the 128+signal shell convention). rc=134 (128+6, SIGABRT) is the macOS+AMD
# gpusKillClientExt class the retry ladder de-escalates against.
ABORT_CLASS_RCS = frozenset({134, 139, 138, 137, -6, -11, -10, -9})


def is_abort_class(rc: Optional[int]) -> bool:
    return rc is not None and rc in ABORT_CLASS_RCS


def deescalate_batch_size(batch_size: int, factor: float = 0.5,
                          floor: int = 500) -> int:
    """Reduce the batch for the next attempt. Never below `floor`, and always
    STRICTLY smaller than the input (so an identical-parameter retry is
    impossible) until the floor is reached."""
    if batch_size <= floor:
        return floor
    nxt = int(batch_size * factor)
    if nxt >= batch_size:          # guard against factor >= 1
        nxt = batch_size - 1
    return max(floor, nxt)


class RetryLadder:
    """Enforce the Phase R retry discipline: on an abort-class rc the next
    attempt MUST use strictly de-escalated parameters; an identical-parameter
    retry is forbidden (raises). Non-abort-class failures may retry once at the
    same parameters (transient I/O etc.), but the ladder still tracks the
    parameter history to forbid infinite identical loops.

    Usage:
        ladder = RetryLadder(initial_batch=8000, retries=3)
        while ladder.should_try():
            batch = ladder.current_batch
            rc = run(batch=batch)
            if rc == 0:
                break
            ladder.record_failure(rc)   # de-escalates batch on abort-class rc
    """

    def __init__(self, initial_batch: int, retries: int = 2,
                 factor: float = 0.5, floor: int = 500):
        self.current_batch = int(initial_batch)
        self.retries = int(retries)
        self.factor = float(factor)
        self.floor = int(floor)
        self.attempt = 0
        self._history = [int(initial_batch)]  # batch used per attempt index

    def should_try(self) -> bool:
        return self.attempt <= self.retries

    def record_failure(self, rc: Optional[int]) -> None:
        """Advance to the next attempt, de-escalating on abort-class rc. Raises
        if the next attempt would repeat the previous parameters after an
        abort-class failure (identical-parameter retry -- forbidden)."""
        self.attempt += 1
        prev = self.current_batch
        if is_abort_class(rc):
            self.current_batch = deescalate_batch_size(
                prev, self.factor, self.floor)
            if self.current_batch >= prev and prev > self.floor:
                raise RuntimeError(
                    "retry ladder failed to de-escalate an abort-class failure "
                    f"(batch stayed {prev}); identical-parameter retry forbidden")
            if self.current_batch == prev and prev == self.floor:
                raise RuntimeError(
                    f"retry ladder hit the batch floor ({self.floor}) and the "
                    "run still aborts; giving up rather than looping identically")
        self._history.append(self.current_batch)

    @property
    def history(self) -> List[int]:
        return list(self._history)
