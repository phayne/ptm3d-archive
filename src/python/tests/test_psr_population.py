"""Unresolved facets must leave BOTH the numerator and the denominator.

Paul's ruling, 2026-09-15, verbatim:

    "If a facet has no value, then it should leave the denominator and the
     numerator"

Every arm below is built to FAIL on the pre-ruling code, not to skip: a skip
would be green-by-nothing, which is how this defect survived a sweep once
already. The arms use numpy and the standard library only, so nothing here can
be silenced by a missing plotting, Qt or mesh dependency -- the single
end-to-end arm that genuinely needs pyvista says so explicitly.

The fixture injects non-finite values at KNOWN indices and is deliberately
built so the unguarded answer differs materially from the guarded one: -inf
sails through `< threshold` into the numerator, NaN and +inf pad the
denominator, and the unresolved facets carry most of the area.
"""
from __future__ import annotations

import ast
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SRC_PYTHON = REPO / "src" / "python"

# ---------------------------------------------------------------------------
# The fixture: 100 facets, 20 of them unresolved at known indices.
# ---------------------------------------------------------------------------
N_FACES = 100
IDX_NAN = list(range(0, 10))       # 10 facets: NaN
IDX_POSINF = list(range(10, 15))   # 5 facets: +inf
IDX_NEGINF = list(range(15, 20))   # 5 facets: -inf
IDX_UNRESOLVED = IDX_NAN + IDX_POSINF + IDX_NEGINF
N_RESOLVED = N_FACES - len(IDX_UNRESOLVED)          # 80
PSR_THRESH = 1.0

# Of the 80 resolved facets, exactly 40 sit below the PSR threshold.
IDX_DARK = list(range(20, 60))
TRUE_COUNT_FRACTION = len(IDX_DARK) / N_RESOLVED    # 0.50 -- the honest answer
# The unguarded answer: NaN < 1.0 and +inf < 1.0 are both False, but
# -inf < 1.0 is True, so 5 corrupt facets join the numerator and all 20 pad
# the denominator: (40 + 5) / 100 = 0.45.
UNGUARDED_COUNT_FRACTION = (len(IDX_DARK) + len(IDX_NEGINF)) / N_FACES


def make_field():
    """direct-solar-max-like field with the injected non-finite facets."""
    field = np.full(N_FACES, 500.0, dtype=np.float64)
    field[IDX_DARK] = 0.0
    field[IDX_NAN] = np.nan
    field[IDX_POSINF] = np.inf
    field[IDX_NEGINF] = -np.inf
    return field


def make_areas():
    """Areas with teeth: the unresolved facets carry most of the total."""
    area = np.full(N_FACES, 1.0, dtype=np.float64)
    area[IDX_UNRESOLVED] = 100.0
    return area


# ---------------------------------------------------------------------------
# Arm 1 -- the shared helper. Absent on the pre-ruling tree, so this FAILS
# there rather than skipping.
# ---------------------------------------------------------------------------
def _load_ptm_population():
    path = SRC_PYTHON / "ptm_population.py"
    assert path.is_file(), (
        f"{path} does not exist: this tree has no shared definition of the "
        "resolved-facet population, so unresolved facets cannot be leaving "
        "both the numerator and the denominator.")
    if str(SRC_PYTHON) not in sys.path:
        sys.path.insert(0, str(SRC_PYTHON))
    import ptm_population

    return ptm_population


def test_resolved_mask_is_a_finiteness_test_not_an_isnan_test():
    pop = _load_ptm_population()
    field = make_field()
    resolved = pop.resolved_mask(field)
    assert int(resolved.sum()) == N_RESOLVED
    for i in IDX_UNRESOLVED:
        assert not resolved[i], f"facet {i} has no value but stayed in the population"
    # An isnan-only test would have kept the 10 infinite facets.
    assert int((~np.isnan(field)).sum()) == N_FACES - len(IDX_NAN)


def test_count_fraction_drops_unresolved_from_numerator_and_denominator():
    pop = _load_ptm_population()
    field = make_field()
    resolved = pop.resolved_mask(field)
    frac = pop.count_fraction(field < PSR_THRESH, resolved)
    assert frac == pytest.approx(TRUE_COUNT_FRACTION), (
        "the count fraction was not taken over the resolved population")
    # and it is materially different from what the unguarded expression gives
    assert abs(frac - UNGUARDED_COUNT_FRACTION) > 0.04
    assert float(np.mean(field < PSR_THRESH)) == pytest.approx(
        UNGUARDED_COUNT_FRACTION), "fixture lost its teeth"


def test_area_fraction_removes_unresolved_area_from_the_total():
    pop = _load_ptm_population()
    field, area = make_field(), make_areas()
    resolved = pop.resolved_mask(field, area)
    frac = pop.area_fraction(field < PSR_THRESH, resolved, area)
    # 40 qualifying facets of unit area, out of 80 resolved facets of unit area
    assert frac == pytest.approx(40.0 / 80.0)
    # The unguarded area fraction divides by the FULL area, 80 + 20*100 = 2080,
    # and lets the five -inf facets (100 m^2 each) into the numerator.
    unguarded = float(area[field < PSR_THRESH].sum() / area.sum())
    assert unguarded == pytest.approx((40.0 + 5 * 100.0) / 2080.0)
    assert abs(frac - unguarded) > 0.2, "fixture lost its teeth"


def test_empty_population_is_undefined_not_zero():
    pop = _load_ptm_population()
    field = np.full(8, np.nan)
    resolved = pop.resolved_mask(field)
    with pytest.raises(ValueError):
        pop.count_fraction(field < PSR_THRESH, resolved)
    with pytest.raises(ValueError):
        pop.area_fraction(field < PSR_THRESH, resolved, np.ones(8))


def test_order_statistics_also_leave_the_unresolved_facets():
    """Median/mean/max next to these fractions must use the same population.

    Unguarded they are worse than wrong: np.median goes NaN, max goes
    Infinity, and json.dump writes both as bare literals that are not valid
    JSON -- while every NaN comparison downstream is False, so a tolerance
    gate built on them reports "within tolerance" and exits 0.
    """
    pop = _load_ptm_population()
    field = make_field()
    resolved = pop.resolved_mask(field)
    vals = pop.resolved_values(field, resolved)
    assert vals.size == N_RESOLVED
    assert np.isfinite(vals).all()
    assert np.isfinite(np.median(vals)) and np.isfinite(vals.max())
    # unguarded: NaN dominates both the median and the maximum
    assert np.isnan(np.median(field)) and np.isnan(field.max())
    # and with no NaN at all, a single +inf turns the maximum into Infinity,
    # which json.dump writes as a bare `Infinity` literal (not valid JSON)
    inf_only = np.where(np.isnan(field), 500.0, field)
    assert np.isinf(inf_only.max())
    assert np.isfinite(pop.resolved_values(inf_only,
                                           pop.resolved_mask(inf_only)).max())
    with pytest.raises(ValueError):
        pop.resolved_values(np.full(4, np.nan), pop.resolved_mask(np.full(4, np.nan)))


def test_a_non_boolean_mask_is_not_silently_bitwise_anded():
    pop = _load_ptm_population()
    field = make_field()
    resolved = pop.resolved_mask(field)
    as_int = (field < PSR_THRESH).astype(np.int8)
    assert pop.count_fraction(as_int, resolved) == pytest.approx(TRUE_COUNT_FRACTION)
    as_float = (field < PSR_THRESH).astype(np.float64)
    assert pop.count_fraction(as_float, resolved) == pytest.approx(TRUE_COUNT_FRACTION)


def test_warn_if_unresolved_cannot_be_silent():
    pop = _load_ptm_population()
    import io

    buf = io.StringIO()
    resolved = pop.resolved_mask(make_field())
    n = pop.warn_if_unresolved("fixture", resolved, stream=buf)
    assert n == len(IDX_UNRESOLVED)
    assert "WARNING" in buf.getvalue()
    assert str(len(IDX_UNRESOLVED)) in buf.getvalue()


# ---------------------------------------------------------------------------
# Arm 2 -- the GUI classifier, exercised WITHOUT importing PyQt6.
# The two helpers are pure numpy; they are lifted out of the source text by
# AST so this arm runs anywhere numpy does and fails loudly when they are
# missing (which is the case on the pre-ruling tree).
# ---------------------------------------------------------------------------
def _exec_functions(path: Path, names):
    assert path.is_file(), f"{path} is missing"
    tree = ast.parse(path.read_text())
    wanted = {n: None for n in names}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            wanted[node.name] = node
    missing = [n for n, v in wanted.items() if v is None]
    assert not missing, (
        f"{path.name} defines no {missing}: this tree still classifies facets "
        "with a bare threshold comparison, so a facet with no value is "
        "silently reported as qualifying or not-qualifying.")
    # _WARNED is the module-level dedupe registry the lifted helpers use to
    # avoid a warning storm; seed a fresh one per call so each arm sees the
    # warning path exactly once.
    ns = {"np": np, "sys": sys, "_WARNED": set()}
    mod = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), ns)
    return ns


def test_gui_thermal_analysis_classifies_over_the_resolved_population():
    ns = _exec_functions(
        REPO / "gui" / "module_adapters" / "thermal_analysis.py",
        ["resolved_population", "classify_at_or_below"])
    field = make_field()
    mask, n_qualifying, n_resolved, n_total = ns["classify_at_or_below"](
        field, PSR_THRESH)
    assert n_total == N_FACES
    assert n_resolved == N_RESOLVED
    assert n_qualifying == len(IDX_DARK)
    for i in IDX_NEGINF:
        assert not mask[i], f"facet {i} is -inf and was reported as a PSR"
    assert n_qualifying / n_resolved == pytest.approx(TRUE_COUNT_FRACTION)


# The polar cold-trap fraction helper is the same defect on a different body.
# Its real inputs are not reachable from this machine, so it gets a behavioral
# arm here instead of a before/after run: the helper is lifted out of the
# source text and made to answer the fixture.
FRACTION_HELPERS = [
    ("scripts/plot_polar_cold_traps.py", "_frac_below"),
]


@pytest.mark.parametrize("rel,name", FRACTION_HELPERS,
                         ids=[r.split("/")[-1] for r, _ in FRACTION_HELPERS])
def test_cold_trap_fraction_helpers_use_the_resolved_population(rel, name):
    names = [name] if rel != "scripts/plot_polar_cold_traps.py" \
        else ["_warn_unresolved", name]
    ns = _exec_functions(REPO / rel, names)
    field = make_field()
    frac = ns[name](field, PSR_THRESH)
    assert frac == pytest.approx(TRUE_COUNT_FRACTION), (
        f"{rel}:{name} did not take the fraction over the resolved facets")
    assert abs(frac - UNGUARDED_COUNT_FRACTION) > 0.04
    # an all-unresolved population is not zero
    assert np.isnan(ns[name](np.full(8, np.nan), PSR_THRESH))


# ---------------------------------------------------------------------------
# Arm 2b -- NOTHING this work does may reach a figure as text.
#
# Paul, 2026-09-15: "Do not restyle figures or present disclosures/caveats on
# figures -- these can be recorded in notes outside of the paper that I can
# incorporate as needed." Standing crew convention
# (crew-figure-style-no-onfigure-notes), not a one-off. The disclosure lives in
# printed output and in the JSON/CSV/text sidecars instead.
# ---------------------------------------------------------------------------
FIGURE_TEXT_FORBIDDEN = [
    ("src/python/moon3d_tmax_by_lat.py", "resolved faces)"),
    ("src/python/moon3d_tmax_by_lat.py", "lab +="),
    ("scripts/plot_polar_cold_traps.py", "No value (excluded)"),
]

# on-figure text call sites, and how many each file is allowed to have. Taken
# from the parent commit: this work may not add one.
FIGURE_TEXT_BUDGET = [
    ("scripts/plot_polar_cold_traps.py", "legend_entries.append", 2),
    ("src/python/moon3d_tmax_by_lat.py", "ax.text2D", 1),
    ("src/python/moon3d_tilt_effect.py", "ax.set_title", 1),
    ("src/python/moon3d_psr_coldtrap_by_lat.py", "set_title", 2),
]


@pytest.mark.parametrize("rel,expr", FIGURE_TEXT_FORBIDDEN,
                         ids=[r.split("/")[-1] + ":" + e[:14]
                              for r, e in FIGURE_TEXT_FORBIDDEN])
def test_no_disclosure_is_drawn_on_a_figure(rel, expr):
    path = REPO / rel
    assert path.is_file(), f"{path} is missing"
    assert expr not in path.read_text(), (
        f"{rel} puts {expr!r} on a figure. Population disclosures belong in "
        "printed output, the sidecars and the session notes -- never on a "
        "figure (Paul, 2026-09-15).")


@pytest.mark.parametrize("rel,call,budget", FIGURE_TEXT_BUDGET,
                         ids=[f"{r.split('/')[-1]}:{c}" for r, c, _ in FIGURE_TEXT_BUDGET])
def test_no_figure_gained_a_text_call(rel, call, budget):
    path = REPO / rel
    assert path.is_file(), f"{path} is missing"
    found = path.read_text().count(call)
    assert found == budget, (
        f"{rel} now has {found} {call!r} call(s), the parent commit had "
        f"{budget}. No figure may gain text from this work.")


# ---------------------------------------------------------------------------
# Arm 3 -- static gate. The pre-ruling expressions must be gone from every
# file the ruling touches. Unskippable: source text only.
# ---------------------------------------------------------------------------
FORBIDDEN = [
    ("src/python/moon3d_deltas.py", "psr = dsm < PSR_THRESH"),
    ("src/python/moon3d_deltas.py", "float(area[psr].sum() / area.sum())"),
    ("src/python/moon3d_deltas.py", "float(psr.mean())"),
    ("src/python/moon3d_deltas.py", "int((tmc > 400.0).sum())"),
    ("src/python/moon3d_psr_coldtrap_by_lat.py", "float(area[mask].sum() / area.sum())"),
    ("src/python/moon3d_tilt_effect.py", "psr = dsm < PSR_THRESH"),
    ("src/python/moon3d_tilt_effect.py", "area[tmax < COLD_THRESH].sum() / area.sum()"),
    ("src/python/moon3d_tmax_by_lat.py", "float(np.mean(tmax < a.cold_thresh))"),
    ("src/python/moon3d_tmax_by_lat.py", "int((tmax > a.mask_tmax).sum())"),
    ("scripts/yearlong_analysis/make_yearlong_figures.py",
     "np.array([area[tmax < th].sum() / total for th in thresholds])"),
    ("scripts/yearlong_analysis/make_yearlong_figures.py",
     "float(area[tmax < th].sum() / area_tot)"),
    ("scripts/yearlong_analysis/make_yearlong_figures.py", "float(np.mean(tmax < th))"),
    ("scripts/yearlong_analysis/make_yearlong_figures.py", "int((tmax > 400).sum())"),
    ("gui/module_adapters/thermal_analysis.py", "psr_mask = data <= threshold"),
    ("gui/module_adapters/thermal_analysis.py", "ct_mask = data <= threshold"),
    ("scripts/plot_polar_cold_traps.py", "np.sum(data < T_stab) / len(data)"),
    ("scripts/plot_polar_cold_traps.py", "np.sum(T_max < T_stab) / n_faces"),
    # round 2: the order statistics and the figure paint that sit next to
    # these fractions must use the same population, or one bad facet turns a
    # caption number into nan, a JSON maximum into Infinity, or paints a facet
    # the console said it did not mask
    ("src/python/moon3d_deltas.py", "med_tmax = float(np.median(tmc))"),
    ("src/python/moon3d_deltas.py", "flag_t = abs(dtmax) > a.tol * 100"),
    ("src/python/moon3d_tilt_effect.py", "median_tmax=float(np.median(tmax))"),
    ("src/python/moon3d_tmax_by_lat.py", "tmed = float(np.median(tmax))"),
    ("src/python/moon3d_tmax_by_lat.py", 'fc[p["tmax"] > a.mask_tmax]'),
    ("scripts/yearlong_analysis/make_yearlong_figures.py",
     "median_tmax_K=float(np.median(tmax)), mean_tmax_K=float(tmax.mean())"),
    ("scripts/yearlong_analysis/make_yearlong_figures.py",
     "max_tmax_K=float(tmax.max()), min_tmax_K=float(tmax.min())"),
]


@pytest.mark.parametrize("rel,expr", FORBIDDEN,
                         ids=[f"{i}:{r.split('/')[-1]}" for i, (r, _) in enumerate(FORBIDDEN)])
def test_no_unguarded_threshold_fraction_remains(rel, expr):
    path = REPO / rel
    assert path.is_file(), f"{path} is missing"
    text = path.read_text()
    assert expr not in text, (
        f"{rel} still contains the unguarded expression {expr!r}: a facet with "
        "no value is being counted as qualifying or not-qualifying instead of "
        "leaving the population.")


# ---------------------------------------------------------------------------
# Arm 4 -- end to end through the real driver, on a synthetic run tree.
# Needs pyvista (the driver reads the mesh through it); that is a genuine
# environment requirement, not a way to make the gate pass by doing nothing --
# arms 1-3 above already fail on the pre-ruling tree without it.
# ---------------------------------------------------------------------------
def _write_binary_ply(path: Path, verts: np.ndarray, faces: np.ndarray):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    body = [header, verts.astype("<f8").tobytes()]
    for tri in faces:
        body.append(struct.pack("<B3i", 3, int(tri[0]), int(tri[1]), int(tri[2])))
    path.write_bytes(b"".join(body))


def _write_stat(path: Path, values: np.ndarray):
    """statistics/<name>.bin: a leading count word, then float32 values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.empty(len(values) + 1, dtype=np.float32)
    out[0] = np.float32(len(values))
    out[1:] = values.astype(np.float32)
    out.tofile(path)


def _build_run(root: Path, dsm: np.ndarray, tmax: np.ndarray, scale: np.ndarray):
    """One synthetic run directory: mesh.ply + the statistics this driver reads.

    Facet i is an isolated right triangle with legs (scale[i], 1), so its area
    is scale[i]/2 -- that is how the area weights get their teeth.
    """
    verts, faces = [], []
    for i in range(len(dsm)):
        b = len(verts)
        verts += [(0.0, 0.0, float(i)), (float(scale[i]), 0.0, float(i)),
                  (0.0, 1.0, float(i))]
        faces.append((b, b + 1, b + 2))
    root.mkdir(parents=True, exist_ok=True)
    _write_binary_ply(root / "mesh.ply", np.array(verts), np.array(faces))
    _write_stat(root / "statistics" / "direct_solar_max.bin", dsm)
    _write_stat(root / "statistics" / "temperature_max.bin", tmax)


def test_psr_coldtrap_by_lat_reports_the_resolved_population(tmp_path):
    pytest.importorskip(
        "pyvista",
        reason="moon3d_psr_coldtrap_by_lat.py reads the mesh through pyvista; "
               "the unskippable arms above already fail on the pre-ruling tree")
    dsm = make_field()
    # Cold traps: the same 40 facets, so the cold-trap fraction has the same
    # honest answer and the same unguarded one.
    tmax = np.full(N_FACES, 300.0)
    tmax[IDX_DARK] = 50.0
    tmax[IDX_NAN] = np.nan
    tmax[IDX_POSINF] = np.inf
    tmax[IDX_NEGINF] = -np.inf
    scale = make_areas() * 2.0          # area = scale/2 = make_areas()

    coupled, radoff = tmp_path / "coupled", tmp_path / "radoff"
    _build_run(coupled, dsm, tmax, scale)
    _build_run(radoff, dsm, tmax, scale)

    out_json = tmp_path / "fractions.json"
    driver = SRC_PYTHON / "moon3d_psr_coldtrap_by_lat.py"
    proc = subprocess.run(
        [sys.executable, str(driver), "--run", "-85", str(coupled), str(radoff),
         "--json", str(out_json), "--no-figure"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    d = json.loads(out_json.read_text())
    assert d["psr_frac"][0] == pytest.approx(40.0 / 80.0), (
        "PSR area fraction was not taken over the resolved facets: got "
        f"{d['psr_frac'][0]}, the unguarded value is "
        f"{(40.0 + 5 * 100.0) / 2080.0}")
    assert d["coldtrap_coupled"][0] == pytest.approx(40.0 / 80.0)
    assert d["n_resolved_psr"][0] == N_RESOLVED, (
        "the driver did not disclose the population its fraction was over")
    assert d["n_faces_coupled"][0] == N_FACES
    assert "WARNING" in proc.stderr and "no value" in proc.stderr, (
        "the driver dropped facets from the population without warning")
