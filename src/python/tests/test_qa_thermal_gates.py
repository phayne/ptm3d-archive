#!/usr/bin/env python3
"""`ptm_qa.py thermal`: the never-lit gate must fail the right runs.

The gate used to be one line -- ``std(Tmin) <= 3 K`` over every face with
``direct_solar_max == 0`` -- calibrated on 623 Chimaera, a run with no
radiation coupling at all. It then FAILED the 2026-09-02 lunar polar flux-cube
product, which 57 independent checks passed: 10,707 never-lit faces at
std 8.17 K, of which the 9 receiving zero indirect flux sit at std 0.0020 K and
the other 10,698 track the indirect flux their neighbors send them. The
scatter was physics.

Raising the threshold would have been a clamp, and a clamp in the direction
that hides the worse failure: a run whose view-factor matrix contributes
nothing puts every never-lit face at exactly the geothermal floor, std 0.00 K,
and the old check PASSES it.

Both directions are measured here -- on the archived fixtures where they
exist, and on synthesized runs where a defect has to be manufactured.

Run directly::

    python3 src/python/tests/test_qa_thermal_gates.py
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_fixtures                                     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
QA = REPO_ROOT / "scripts" / "ptm_qa.py"

SIGMA = 5.670374419e-8
EPS = 0.95
#: The Moon's documented geothermal floor, (Q_b/(eps*sigma))^(1/4).
T_GEO = 24.0428

#: Real runs, used where they exist. Each arm SKIPS rather than passing when
#: its fixture is absent -- a gate that cannot look is not a gate that agrees.
EMA = Path(os.environ.get("PTM_EMA_ROOT", "/nonexistent/ema"))
CHIMAERA_BROKEN = EMA / "thermal/623chimaera/flyby_preTerminatorFix_20260704_142231"
CHIMAERA_GOOD = EMA / "thermal/623chimaera/flyby"
#: A finished, radiation-coupled lunar polar run. Registered locally rather
#: than named here -- see src/python/tests/local_fixtures.py.
LUNAR_RUN_FIXTURE = "lunar_radiation_run"

_SCRATCH = None


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def scratch():
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = tempfile.mkdtemp(prefix="ptm_qa_gate_")
    return Path(_SCRATCH)


def t_eq(flux, floor=T_GEO, eps=EPS):
    """The closed form the gate uses: eps*sigma*T^4 = F_absorbed + Q_b."""
    return (floor ** 4 + flux / (eps * SIGMA)) ** 0.25


def write_bin(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", len(values)))
        fh.write(struct.pack("<%df" % len(values), *values))


def make_run(name, tmin, direct_max, f_lo=None, f_hi=None, radiation=None):
    """A run directory with just the files the thermal gate reads."""
    run = scratch() / name
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    n = len(tmin)
    (run / "mesh.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n"
        f"element face {n}\nproperty list uchar int vertex_indices\n"
        "end_header\n")
    write_bin(run / "statistics" / "temperature_min.bin", tmin)
    write_bin(run / "statistics" / "direct_solar_max.bin", direct_max)
    if f_lo is not None:
        # The two bands sum to the absorbed indirect flux; splitting the whole
        # of it into indirect_ir and leaving indirect_solar at zero is a legal
        # decomposition and keeps the arithmetic here readable.
        zeros = [0.0] * n
        write_bin(run / "statistics" / "indirect_ir_min.bin", f_lo)
        write_bin(run / "statistics" / "indirect_ir_max.bin", f_hi)
        write_bin(run / "statistics" / "indirect_solar_min.bin", zeros)
        write_bin(run / "statistics" / "indirect_solar_max.bin", zeros)
    if radiation is not None:
        (run / "config_used.yaml").write_text(
            "simulation:\n  name: synthetic\n"
            f"radiation:\n  enabled: {'true' if radiation else 'false'}\n"
            "  use_gpu: false\nthermal:\n  use_gpu: false\n")
    return run


def run_qa(run_dir, *extra):
    out = scratch() / (Path(run_dir).name + "_qa.json")
    p = subprocess.run([sys.executable, str(QA), "thermal", str(run_dir),
                        "--out", str(out)] + list(extra),
                       capture_output=True, text=True, timeout=900)
    report = json.loads(out.read_text()) if out.exists() else {}
    return p.returncode, report, p.stdout + p.stderr


def check(report, needle):
    """The one check whose name contains `needle`."""
    hits = [c for c in report.get("checks", []) if needle in c["check"]]
    assert len(hits) == 1, (
        f"expected exactly one check matching {needle!r}, found "
        f"{[c['check'] for c in report.get('checks', [])]}")
    return hits[0]


# ---------------------------------------------------------------- synthetic
def _coherent_shadow_field(n_never=40, n_lit=60):
    """A run with no indirect flux and coherent shadow faces. PASSES."""
    tmin = [T_GEO + 0.1 * (i % 3) for i in range(n_never)] + [200.0] * n_lit
    direct = [0.0] * n_never + [800.0] * n_lit
    return tmin, direct


def test_a_run_with_no_indirect_flux_still_uses_the_tight_threshold():
    """DO-NO-HARM. Nothing changes for a run the old gate judged correctly."""
    tmin, direct = _coherent_shadow_field()
    rc, rep, _ = run_qa(make_run("nobands_good", tmin, direct))
    c = check(rep, "never-lit face coherence")
    assert c["status"] == "PASS", c
    assert rc == 0, f"a coherent radiation-off run must pass (exit {rc})"


def test_scrambled_shadow_faces_with_no_indirect_flux_still_fail():
    """THE ORIGINAL DETECTOR. The 5bfbdbe signature, synthesized.

    A scrambled initial condition freezes never-lit faces at unrelated
    temperatures. With no indirect flux there is nothing that could explain
    the spread, and it must still be a FAILURE.
    """
    _, direct = _coherent_shadow_field()
    n_never = sum(1 for d in direct if d <= 0)
    scrambled = [30.0 + 18.0 * ((i * 7) % 5 - 2) for i in range(n_never)]
    tmin = scrambled + [200.0] * (len(direct) - n_never)
    rc, rep, _ = run_qa(make_run("nobands_scrambled", tmin, direct))
    c = check(rep, "never-lit face coherence")
    assert c["status"] == "FAIL", c
    assert rc == 1, f"a scrambled shadow field must fail (exit {rc})"


def _forced_field(n_unforced=12, n_forced=80, n_lit=40, temper=lambda t, i: t):
    """A radiation-coupled run: unforced faces at the floor, forced faces at
    equilibrium with a flux that varies from face to face.

    `temper` is where a defect is injected: it receives the physically correct
    temperature and the face index, and returns what the run actually recorded.
    """
    tmin, direct, f_lo, f_hi = [], [], [], []
    for _ in range(n_unforced):
        tmin.append(T_GEO)
        direct.append(0.0)
        f_lo.append(0.0)
        f_hi.append(0.0)
    for i in range(n_forced):
        lo = 0.02 + 0.35 * i          # W/m^2, absorbed
        hi = lo + 2.0 + 0.05 * i
        mid = 0.5 * (lo + hi)
        tmin.append(temper(t_eq(mid), i))
        direct.append(0.0)
        f_lo.append(lo)
        f_hi.append(hi)
    for _ in range(n_lit):
        tmin.append(260.0)
        direct.append(900.0)
        f_lo.append(4.0)
        f_hi.append(9.0)
    return tmin, direct, f_lo, f_hi


def test_a_radiation_coupled_run_whose_faces_track_their_flux_passes():
    """THE REAL RUN'S SHAPE, synthesized: wide scatter, entirely explained."""
    tmin, direct, lo, hi = _forced_field()
    run = make_run("bands_good", tmin, direct, lo, hi, radiation=True)
    rc, rep, _ = run_qa(run)
    below = check(rep, "not colder than their own equilibrium")
    inside = check(rep, "inside the equilibrium bracket")
    unforced = check(rep, "unforced never-lit face coherence")
    assert unforced["status"] == "PASS", unforced
    assert below["status"] == "PASS", below
    assert inside["status"] == "PASS", inside
    assert rc == 0, f"a coherent coupled run must pass (exit {rc})"
    # And the arm has to be about something: the OLD statistic on this run is
    # far outside the old threshold, so a gate that still applied it would fail.
    import statistics as st
    never = [t for t, d in zip(tmin, direct) if d <= 0]
    assert st.pstdev(never) > 3.0, (
        f"this synthetic run's never-lit std is {st.pstdev(never):.2f} K, "
        f"inside the old 3 K threshold -- it does not exercise the never-lit coherence check")


def test_faces_colder_than_their_own_forcing_fail():
    """The energy-forbidden direction, and the one that is a FAILURE."""
    tmin, direct, lo, hi = _forced_field(
        temper=lambda t, i: t - 25.0 if i % 2 == 0 else t)
    tmin = [max(t, 3.0) for t in tmin]
    rc, rep, _ = run_qa(make_run("bands_cold", tmin, direct, lo, hi,
                                 radiation=True))
    below = check(rep, "not colder than their own equilibrium")
    assert below["status"] == "FAIL", below
    assert rc == 1, f"faces below their own equilibrium must fail (exit {rc})"


def test_flux_that_never_reached_the_temperatures_fails_where_std_alone_passed():
    """⭐ THE INVERSION. The failure the old check waved through.

    Every never-lit face sits at exactly the geothermal floor while the run's
    own indirect-flux statistics say those faces are being warmed by their
    neighbors -- the signature of a coupling that was computed and never
    reached the boundary condition, and of a view-factor matrix that
    contributes nothing. `std(Tmin)` over the never-lit faces is 0.00 K, which
    the old threshold PASSES.
    """
    tmin, direct, lo, hi = _forced_field(temper=lambda t, i: T_GEO)
    rc, rep, _ = run_qa(make_run("bands_flat", tmin, direct, lo, hi,
                                 radiation=True))
    import statistics as st
    never = [t for t, d in zip(tmin, direct) if d <= 0]
    assert st.pstdev(never) == 0.0, (
        f"this arm's premise is that the old statistic is 0.00 K; it is "
        f"{st.pstdev(never):.4f} K")
    below = check(rep, "not colder than their own equilibrium")
    assert below["status"] == "FAIL", below
    assert rc == 1, (
        "a run whose never-lit faces ignore their own indirect flux must "
        "fail, and this is exactly the case a std-only check passes")


def test_coupling_without_band_statistics_is_a_warning_not_a_verdict():
    """"I cannot tell" said out loud, rather than a coin toss either way.

    Gray-radiosity and multi-GPU-pool runs couple indirect flux and write no
    band statistics (ptm_thermal_sim.cpp:21754). Failing them is the never-lit coherence defect;
    passing them silently is the same defect with the sign flipped.
    """
    _, direct = _coherent_shadow_field()
    n_never = sum(1 for d in direct if d <= 0)
    tmin = [40.0 + 6.0 * (i % 4) for i in range(n_never)] \
        + [200.0] * (len(direct) - n_never)
    rc, rep, _ = run_qa(make_run("bands_missing", tmin, direct, radiation=True))
    c = check(rep, "never-lit face coherence")
    assert c["status"] == "WARN", c
    assert "no indirect_solar/indirect_ir statistics" in c["note"], c["note"]
    assert rc == 0, (
        f"an unattributable run must not be reported as a failure (exit {rc})")


def test_the_bracket_verdict_does_not_turn_on_the_assumed_emissivity():
    """If the answer moved with `--emissivity`, it would be a tuning knob."""
    tmin, direct, lo, hi = _forced_field()
    run = make_run("bands_eps", tmin, direct, lo, hi, radiation=True)
    for eps in ("0.85", "0.90", "0.95", "1.00"):
        rc, rep, _ = run_qa(run, "--emissivity", eps)
        c = check(rep, "not colder than their own equilibrium")
        assert c["status"] == "PASS" and rc == 0, (
            f"eps={eps} changed the verdict: {c}")


# ----------------------------------------------------------- real fixtures
def _require(path, what):
    if not (Path(path) / "statistics" / "temperature_min.bin").exists():
        raise Skip(f"{what} is not on this machine ({path})")


def test_the_archived_broken_chimaera_field_still_fails():
    """REAL DATA, the direction the threshold was calibrated on: 18.00 K."""
    _require(CHIMAERA_BROKEN, "the archived broken 623 Chimaera field")
    rc, rep, _ = run_qa(CHIMAERA_BROKEN)
    c = check(rep, "never-lit face coherence")
    assert c["status"] == "FAIL" and rc == 1, c
    assert c["measured"].startswith("18."), (
        f"the archived broken field measures {c['measured']}, not the 18.00 K "
        f"this threshold was derived from")


def test_the_archived_good_chimaera_field_still_passes():
    """REAL DATA, do-no-harm: 0.96 K on the fixed binary."""
    _require(CHIMAERA_GOOD, "the archived good 623 Chimaera field")
    rc, rep, _ = run_qa(CHIMAERA_GOOD)
    c = check(rep, "never-lit face coherence")
    assert c["status"] == "PASS" and rc == 0, c


def test_the_real_flux_cube_run_passes():
    """⭐ REAL DATA, THE RUN THAT PROVOKED THIS. 57/57 other checks agreed."""
    run = local_fixtures.fixture_path(LUNAR_RUN_FIXTURE, Skip)
    _require(run, "the lunar polar flux-cube run")
    rc, rep, _ = run_qa(run)
    unforced = check(rep, "unforced never-lit face coherence")
    below = check(rep, "not colder than their own equilibrium")
    assert unforced["status"] == "PASS", unforced
    assert below["status"] == "PASS", below
    assert below["measured"].startswith("0 of "), (
        f"the run must have NO face below its own equilibrium: {below}")
    assert rc == 0, f"the run must pass (exit {rc})"


TESTS = [
    test_a_run_with_no_indirect_flux_still_uses_the_tight_threshold,
    test_scrambled_shadow_faces_with_no_indirect_flux_still_fail,
    test_a_radiation_coupled_run_whose_faces_track_their_flux_passes,
    test_faces_colder_than_their_own_forcing_fail,
    test_flux_that_never_reached_the_temperatures_fails_where_std_alone_passed,
    test_coupling_without_band_statistics_is_a_warning_not_a_verdict,
    test_the_bracket_verdict_does_not_turn_on_the_assumed_emissivity,
    test_the_archived_broken_chimaera_field_still_fails,
    test_the_archived_good_chimaera_field_still_passes,
    test_the_real_flux_cube_run_passes,
]


def main():
    failures = skipped = 0
    for test in TESTS:
        try:
            test()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    print(f"\n[test_qa_thermal_gates] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if _SCRATCH:
        shutil.rmtree(_SCRATCH, ignore_errors=True)
    if skipped == total:
        print("[test_qa_thermal_gates] SKIPPED ENTIRELY -- proved NOTHING here.")
    if failures:
        print("[test_qa_thermal_gates] FAIL")
        return 1
    print("[test_qa_thermal_gates] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
