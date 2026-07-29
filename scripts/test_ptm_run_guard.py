#!/usr/bin/env python3
"""Unit tests for scripts/ptm_run_guard.py (Phase R, R3 + R5).

Pure-python, no GPU, no sim binary -- runnable anytime:
    python3 scripts/test_ptm_run_guard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptm_run_guard as g  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    if cond:
        print(f"ok:   {msg}")
    else:
        print(f"FAIL: {msg}")
        _fail += 1


def expect_raises(fn, msg):
    try:
        fn()
    except Exception:
        print(f"ok:   {msg} (raised)")
        return
    global _fail
    print(f"FAIL: {msg} (did NOT raise)")
    _fail += 1


# --- R3a env-completeness ---------------------------------------------------
env_ok = {
    "PTM_ILLUM_TRUE_NORMALS": "1",
    "PTM_ILLUM_RISE_KEPS": "0.2",
    "PTM_ILLUM_SEAL_HOLES": "30",
}
check(g.check_env_completeness(["PTM_ILLUM_TRUE_NORMALS"], env_ok) == [],
      "present var passes list gate")
check(len(g.check_env_completeness(["PTM_MISSING"], env_ok)) == 1,
      "missing var fails list gate")
check(g.check_env_completeness({"PTM_ILLUM_RISE_KEPS": "0.2"}, env_ok) == [],
      "matching value passes dict gate")
check(len(g.check_env_completeness({"PTM_ILLUM_RISE_KEPS": "0.5"}, env_ok)) == 1,
      "wrong value fails dict gate (the dropped/changed-env class)")
check(len(g.check_env_completeness({"PTM_ILLUM_TRUE_NORMALS": None,
                                    "PTM_ABSENT": None}, env_ok)) == 1,
      "dict gate with None expects presence only")
expect_raises(lambda: g.assert_env_complete(["PTM_ABSENT"], env_ok),
              "assert_env_complete raises on incomplete env")

# The exact 07-20 incident: a bare relaunch with an EMPTY env must be caught
# pre-launch, not post-hoc.
incident_required = {
    "PTM_ILLUM_TRUE_NORMALS": "1",
    "PTM_ILLUM_RISE_KEPS": "0.2",
    "PTM_ILLUM_RISE_HABS": "0",
    "PTM_ILLUM_SEAL_HOLES": "30",
}
check(len(g.check_env_completeness(incident_required, {})) == 4,
      "empty env (the 4.3h-burn relaunch) flags ALL required physics vars")

# --- R3b derived timeout ----------------------------------------------------
check(g.derive_timeout_s(3600, k=3.0, floor_s=600) == 10800,
      "timeout = k*estimate above floor")
check(g.derive_timeout_s(60, k=3.0, floor_s=600) == 600,
      "timeout clamps up to floor")
expect_raises(lambda: g.derive_timeout_s(10800, hand_set_s=5400),
              "hand-set timeout below estimate raises (90min-vs-3h class)")
check(g.derive_timeout_s(3600, k=2.0, hand_set_s=99999) == 99999,
      "hand-set timeout above derived is honored")

# --- R3c stall watchdog -----------------------------------------------------
w = g.StallWatchdog(stall_limit_s=600, now=0.0)
check(w.update(10, now=10) is True, "first/advancing progress not a stall")
check(w.update(20, now=100) is True, "further progress advances")
check(w.update(20, now=300) is False, "no progress => update returns False")
check(w.is_stalled(now=200) is False, "not stalled before limit")
check(w.is_stalled(now=750) is True, "stalled after limit with no advance")
w.update(30, now=800)  # progress resumes
check(w.is_stalled(now=900) is False, "resumed progress clears the stall")

# --- R3d launch consistency -------------------------------------------------
prof = {"max_timesteps_per_dispatch": 400, "recommended_batch_size": 8000}
check(g.preflight_consistency(timeout_s=10800, est_runtime_s=3600,
                              batch_size=8000, max_timesteps_per_dispatch=300,
                              profile=prof) == [],
      "consistent launch passes")
check(len(g.preflight_consistency(timeout_s=5400, est_runtime_s=10800,
                                  batch_size=8000, max_timesteps_per_dispatch=300,
                                  profile=prof)) == 1,
      "timeout below runtime flagged")
check(len(g.preflight_consistency(timeout_s=10800, est_runtime_s=3600,
                                  batch_size=200000, max_timesteps_per_dispatch=1560,
                                  profile=prof)) == 2,
      "over-batch AND over-ceiling both flagged (the 200000/1560 wedge)")
expect_raises(lambda: g.assert_launch_consistent(
                  timeout_s=60, est_runtime_s=3600, batch_size=8000,
                  max_timesteps_per_dispatch=300, profile=prof),
              "assert_launch_consistent raises on contradiction")

# no profile -> conservative defaults are used
check(len(g.preflight_consistency(timeout_s=10800, est_runtime_s=3600,
                                  batch_size=9000, max_timesteps_per_dispatch=301,
                                  profile=None)) == 2,
      "no profile uses conservative 300/8000 ceilings")

# F6: batch != profile's batch_size_probe -> caveat
prof6 = {"max_timesteps_per_dispatch": 400, "recommended_batch_size": 16000,
         "batch_size_probe": 8000}
check(len(g.preflight_consistency(timeout_s=10800, est_runtime_s=3600,
                                  batch_size=12000, max_timesteps_per_dispatch=300,
                                  profile=prof6)) == 1,
      "F6: batch != batch_size_probe flags a caveat")
check(g.preflight_consistency(timeout_s=10800, est_runtime_s=3600,
                              batch_size=8000, max_timesteps_per_dispatch=300,
                              profile=prof6) == [],
      "F6: batch == batch_size_probe is clean")

# --- R5 retry ladder --------------------------------------------------------
check(g.is_abort_class(134) and g.is_abort_class(139),
      "134/139 recognized as abort-class")
check(not g.is_abort_class(0) and not g.is_abort_class(1),
      "0/1 are not abort-class")
check(g.deescalate_batch_size(8000) == 4000, "batch halves")
check(g.deescalate_batch_size(600, floor=500) == 500, "batch clamps to floor")
check(g.deescalate_batch_size(500, floor=500) == 500, "batch stays at floor")

ladder = g.RetryLadder(initial_batch=8000, retries=3)
seq = [ladder.current_batch]
ladder.record_failure(134); seq.append(ladder.current_batch)
ladder.record_failure(134); seq.append(ladder.current_batch)
check(seq == [8000, 4000, 2000], f"ladder de-escalates on abort-class: {seq}")
check(all(seq[i] > seq[i+1] for i in range(len(seq)-1)),
      "ladder is strictly decreasing (no identical-parameter retry)")

# non-abort rc does not de-escalate (transient) but still advances the attempt
ladder2 = g.RetryLadder(initial_batch=8000, retries=3)
ladder2.record_failure(1)
check(ladder2.current_batch == 8000 and ladder2.attempt == 1,
      "non-abort rc keeps batch, advances attempt")

# floor + still aborting -> give up rather than loop identically
ladder3 = g.RetryLadder(initial_batch=500, retries=5, floor=500)
expect_raises(lambda: ladder3.record_failure(134),
              "abort at the batch floor raises instead of looping identically")

if _fail == 0:
    print("\nALL RUN-GUARD TESTS PASSED")
    sys.exit(0)
print(f"\n{_fail} RUN-GUARD TEST(S) FAILED")
sys.exit(1)
