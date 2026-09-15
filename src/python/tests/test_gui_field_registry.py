#!/usr/bin/env python3
"""Every product the engine writes must be offerable in the 3D viewer's field menu.

⛔ THE DEFECT THIS EXISTS TO STOP, STATED ONCE: THERE ARE TWO FIELD REGISTRIES.

``ptm.results.STATISTICS_FIELDS`` is the shared one -- the exporter, the COG
writer and ``detect_available_statistics`` all read it, and the shadow-depth
pass duly added its five products there. ``VisualizationPanel.FIELD_INFO`` is a
SECOND, hand-maintained dict inside ``gui/gui_panels/visualization_panel.py``,
and ``_populate_field_combo`` iterates THAT. Nobody updated it. The result was a
finished 2.4-million-facet MP026 run whose shadow-depth fields exported
correctly, read correctly from Python, and did not appear in the viewer's
drop-down at all -- with no warning anywhere, because a menu that is missing an
item looks exactly like a run that did not produce one.

The same gap had already swallowed ``primary_reflected_min`` and
``primary_ir_min`` -- two statistics every run has written for far longer than
shadow depth, absent from the menu for exactly the same reason and never
noticed. One missing entry is an oversight; three is a class.

Section 1 is the gate: it reads the keys out of BOTH registries and asserts the
GUI's covers the shared one. It is deliberately the cheapest thing here --
``ast`` over the panel source, so it needs neither PyQt6 nor a display and
CANNOT be skipped on a machine that lacks them. A skipped gate reads as a pass,
and this project has lost a hygiene check that way before.

Sections 2 and 3 cover the other half of the same report: a viewer that lists
``shadow_depth`` and then renders it as one flat color has not fixed anything.
Shadow depth encodes ``0.0`` = the Sun is visible standing on the facet and
``+inf`` = still umbral at the search ceiling, in one float32 array, and inside
a permanently shadowed region ``+inf`` is the MAJORITY value by design. Those
are opposite meanings; ``(min, max)`` over that array is ``(0, inf)``.

Section 4 drives the real widget: a real ``QApplication``, a real
``VisualizationPanel``, a real run directory, and the assertion that the
drop-down CONTAINS the entries -- with ``isVisibleTo`` and a viewport rectangle
computed from Qt, per the discipline in ``test_gui_flux_cubes.py`` section 11.
Section 5 repeats the menu arm on the REAL ``PTM3DMainWindow`` toolbar, because
the panel's own combo is a sub-widget and a harness that only ever drives
sub-widgets is how three rounds of a flux-cube defect got past green suites.

⚠️ WHAT THIS GATE DOES **NOT** COVER, stated so nobody reads it as broader than
it is. It binds ``FIELD_INFO`` to ``ptm.results.STATISTICS_FIELDS`` and to
nothing else. Three products live in ``ptm.results.Results.EXTRA_FIELD_INFO``
instead -- ``indirect_solar``, ``indirect_ir`` and ``effective_albedo`` -- by a
deliberate decision recorded there, and a two-band radiosity run writes all
three to disk while the viewer offers none of them. That gap is real and this
gate is blind to it by construction; closing it is a decision about what the
menu should contain, not a test change.

Run directly::

    python3 src/python/tests/test_gui_field_registry.py

Set ``PTM_TEST_RUN_DIR`` to a finished run directory to add section 5, which
repeats section 4 against real engine output instead of the synthetic fixture.
"""

from __future__ import annotations

import ast
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

# ⛔ BEFORE ANY GUI IMPORT. `gui_panels.visualization_panel` reads this at
# module import time; setting it later has no effect. Without it, building the
# PyVista canvas SEGFAULTS under the offscreen platform plugin.
os.environ.setdefault("PTM_GUI_DISABLE_3D", "1")

import numpy as np                                             # noqa: E402

PANEL_SOURCE = REPO_ROOT / "gui" / "gui_panels" / "visualization_panel.py"

#: A run directory the caller wants section 5 run against. Optional by design:
#: the real MP026 output this bug was found on is 100 GB of machine-local
#: files, and a test that needs it is a test nobody else can run.
REAL_RUN = os.environ.get("PTM_TEST_RUN_DIR", "")


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


# --- 1. The gate: the GUI's catalog covers the shared registry --------------
#
# By `ast`, not by import: this must run with no PyQt6, no display and no
# PyVista, because that is the configuration in which it is most likely to be
# the only thing standing between a new engine product and a silently missing
# menu item.

def _panel_class_body():
    tree = ast.parse(PANEL_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VisualizationPanel":
            return node
    raise AssertionError(
        f"{PANEL_SOURCE} no longer defines class VisualizationPanel; this "
        f"gate reads its FIELD_INFO literal and cannot find it")


def _class_attr(name):
    for stmt in _panel_class_body().body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return stmt.value
    raise AssertionError(
        f"VisualizationPanel.{name} not found in {PANEL_SOURCE}")


def gui_field_info_keys():
    """The keys of the `VisualizationPanel.FIELD_INFO` literal, from source."""
    node = _class_attr("FIELD_INFO")
    assert isinstance(node, ast.Dict), (
        "VisualizationPanel.FIELD_INFO is no longer a dict literal; this gate "
        "reads it with ast and would silently pass on anything else")
    keys = []
    for key in node.keys:
        assert isinstance(key, ast.Constant) and isinstance(key.value, str), (
            f"a non-literal key appeared in FIELD_INFO ({ast.dump(key)[:80]}); "
            f"the menu is built from this dict and a computed key cannot be "
            f"checked against the shared registry")
        keys.append(key.value)
    assert len(keys) == len(set(keys)), (
        f"FIELD_INFO has duplicate keys: "
        f"{sorted(k for k in keys if keys.count(k) > 1)}")
    return keys


def gui_scalar_stats_keys():
    """The `VisualizationPanel.SCALAR_STATS_FIELDS` literal, from source."""
    node = _class_attr("SCALAR_STATS_FIELDS")
    values = []
    for literal in ast.walk(node):
        if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
            values.append(literal.value)
    assert values, "SCALAR_STATS_FIELDS parsed as empty; the ast shape changed"
    return set(values)


def shared_registry_keys():
    """Every field key `ptm.results` knows, expanded to what a menu must offer.

    A scalar product is one key (`ice_depth`). Everything else is the per-step
    series plus its three statistics -- which is exactly what the viewer's two
    scan branches look for on disk.
    """
    from ptm.results import STATISTICS_FIELDS, STAT_TYPES
    expected = {}
    for base, info in STATISTICS_FIELDS.items():
        if info.get("scalar"):
            expected[base] = info
            continue
        expected[base] = info
        for stat in STAT_TYPES:
            expected[f"{base}_{stat}"] = info
    return expected


def test_every_shared_registry_field_is_viewable_in_the_gui():
    """⭐ THE GATE. Every key the engine's own catalog knows has a menu entry.

    Fails on the code as it stood before the shadow-depth entries were added,
    naming all ten keys that were missing -- the six shadow-depth ones and the
    two `_min` statistics that had been missing far longer.
    """
    have = set(gui_field_info_keys())
    want = shared_registry_keys()
    missing = sorted(set(want) - have)
    assert not missing, (
        f"{len(missing)} field key(s) that `ptm.results.STATISTICS_FIELDS` "
        f"knows have no `VisualizationPanel.FIELD_INFO` entry, so the 3D "
        f"viewer's field drop-down cannot offer them no matter what the run "
        f"wrote: {missing}. Add them to FIELD_INFO in {PANEL_SOURCE.name} "
        f"with a label and a colormap.")


def test_the_gate_can_fail():
    """The gate is not vacuous: remove an entry and it reports that entry.

    ⛔ Without this arm, a gate that computed an empty `want` would pass
    forever. That is how a green suite hid this defect in the first place.
    """
    have = set(gui_field_info_keys())
    want = shared_registry_keys()
    assert want, "the shared registry parsed as empty; the gate would be vacuous"
    victim = "shadow_depth_avg"
    assert victim in want, f"{victim} is no longer in the shared registry"
    # Stated as a DIFFERENCE, so this arm means the same thing on a tree where
    # the gate above is already failing -- which is the tree it matters on.
    before = want.keys() - have
    after = want.keys() - (have - {victim})
    assert after - before == {victim}, (
        f"removing {victim} from the GUI catalog should add exactly "
        f"{{{victim!r}}} to what the gate reports; it added "
        f"{sorted(after - before)}")


def test_the_scalar_statistics_lists_agree():
    """`is_stats_field`'s scalar set is the registry's `scalar: True` set.

    These are the products with no _avg/_max/_min suffix, so the suffix rule is
    blind to them: `shadow_lit_frac` would have been listed in the menu and
    then looked for under a per-timestep directory no run writes.
    """
    from ptm.results import STATISTICS_FIELDS
    want = {k for k, v in STATISTICS_FIELDS.items() if v.get("scalar")}
    have = gui_scalar_stats_keys()
    assert have == want, (
        f"VisualizationPanel.SCALAR_STATS_FIELDS is {sorted(have)} but "
        f"`ptm.results.STATISTICS_FIELDS` marks {sorted(want)} as scalar. A "
        f"scalar product missing here is routed to "
        f"<run>/<field>/<field>_0000.bin, which does not exist.")


def test_is_stats_field_classifies_every_registry_key():
    """The live rule, not the literal: statistics classify as statistics.

    Imports the panel, so it skips without PyQt6 -- which is why the gate above
    is `ast`-based and this is the supplement, not the gate.
    """
    try:
        from gui_panels.visualization_panel import VisualizationPanel
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    from ptm.results import STATISTICS_FIELDS, STAT_TYPES
    for base, info in STATISTICS_FIELDS.items():
        if info.get("scalar"):
            assert VisualizationPanel.is_stats_field(base), (
                f"{base} is a scalar statistic but is_stats_field says it is a "
                f"per-timestep series; it would be read from {base}/{base}_0000.bin")
            continue
        assert not VisualizationPanel.is_stats_field(base), (
            f"{base} is a per-timestep series but is_stats_field claims it is "
            f"a statistic; it would be read from statistics/{base}.bin")
        for stat in STAT_TYPES:
            assert VisualizationPanel.is_stats_field(f"{base}_{stat}"), (
                f"{base}_{stat} is a statistic and is_stats_field says otherwise")


# --- 2. The sentinel arithmetic --------------------------------------------
#
# Pure functions, no Qt, no PyVista, no render window: the decision that keeps
# "clear at the surface" and "deeper than we searched" distinguishable is made
# here, so it is checkable on a machine that cannot draw anything.

def _sentinels():
    try:
        from gui_panels.visualization_panel import (
            SHADOW_DEPTH_FIELDS, finite_data_range, sentinel_legend,
            shadow_depth_ceiling_m, shadow_depth_clim)
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    return (SHADOW_DEPTH_FIELDS, finite_data_range, sentinel_legend,
            shadow_depth_ceiling_m, shadow_depth_clim)


def test_a_plain_min_max_over_a_shadow_depth_array_is_the_defect():
    """State the failure this guards against, in numbers, before fixing it.

    ⛔ If this arm ever stops holding, the sentinel encoding changed and every
    other arm in this section is measuring the wrong thing.
    """
    _, finite_data_range, _, _, _ = _sentinels()
    data = np.array([0.0, 0.0, 1.5, 3.25, np.inf, np.inf, np.nan],
                    dtype=np.float32)
    assert np.isnan(np.min(data)) and np.isnan(np.max(data)), (
        "np.min/np.max over an array holding NaN should propagate it; they "
        f"returned {np.min(data)}, {np.max(data)}")
    assert np.nanmin(data) == 0.0 and np.isposinf(np.nanmax(data)), (
        "even ignoring NaN, the range of a shadow-depth array is (0, +inf) -- "
        "that is the flat-color defect, in one line; got "
        f"({np.nanmin(data)}, {np.nanmax(data)})")
    assert finite_data_range(data) == (0.0, 3.25), (
        f"finite_data_range must ignore both sentinels, it returned "
        f"{finite_data_range(data)}")


def test_finite_data_range_refuses_to_invent_a_range():
    """All-sentinel input yields None, not (0, 0).

    A permanently shadowed patch really can be +inf everywhere. Returning
    (0, 0) there would claim every facet is lit at the surface.
    """
    _, finite_data_range, _, _, _ = _sentinels()
    assert finite_data_range(np.array([np.inf, np.inf, np.nan], dtype=np.float32)) is None
    assert finite_data_range(np.array([], dtype=np.float32)) is None
    assert finite_data_range(None) is None
    assert finite_data_range(np.array([0.0, 0.0], dtype=np.float32)) == (0.0, 0.0), (
        "an all-zero array is a real measurement -- everything lit at the "
        "surface -- and must NOT be confused with having no data")


def test_the_color_scale_bottom_is_pinned_at_exactly_zero():
    """0 m is a meaning, not a small number.

    The avg/max/min files exclude zero epochs, so their own minimum is the
    bisection's finest step (~0.0098 m in the MP026 run). Autoscaling to that
    paints a one-centimeter shadow the same color as no shadow at all.
    """
    _, _, _, _, shadow_depth_clim = _sentinels()
    assert shadow_depth_clim((0.0098, 4.9902), 5.0) == (0.0, 5.0)
    assert shadow_depth_clim((0.0098, 4.9902), None) == (0.0, 4.9902)
    assert shadow_depth_clim(None, 5.0) == (0.0, 5.0)
    assert shadow_depth_clim(None, None) is None
    # A sidecar that disagrees with its own data must not push a real value
    # off-scale and relabel it "deeper than the ceiling".
    assert shadow_depth_clim((0.0, 7.5), 5.0) == (0.0, 7.5)
    assert shadow_depth_clim((0.0, 0.0), 0.0) is None


def test_the_search_ceiling_comes_from_the_engine_or_not_at_all():
    """`h_max_m` is read from the sidecar; a missing one yields None."""
    _, _, _, shadow_depth_ceiling_m, _ = _sentinels()
    with tempfile.TemporaryDirectory(prefix="ptm_sentinel_") as tmp:
        run = Path(tmp)
        (run / "statistics").mkdir()
        assert shadow_depth_ceiling_m(run) is None, "no sidecar must yield None"
        side = run / "statistics" / "shadow_depth.json"
        side.write_text("{ not json", encoding="utf-8")
        assert shadow_depth_ceiling_m(run) is None, "unreadable must yield None"
        side.write_text(json.dumps({"h_max_m": 5}), encoding="utf-8")
        assert shadow_depth_ceiling_m(run) == 5.0
        side.write_text(json.dumps({"h_max_m": "five"}), encoding="utf-8")
        assert shadow_depth_ceiling_m(run) is None, "a non-number must yield None"
        side.write_text(json.dumps({"h_max_m": -1}), encoding="utf-8")
        assert shadow_depth_ceiling_m(run) is None, "a nonpositive ceiling is not a ceiling"
        assert shadow_depth_ceiling_m(None) is None


def test_the_legend_names_both_sentinels_and_says_the_ceiling():
    """The user must be able to read magenta and gray without guessing."""
    fields, _, sentinel_legend, _, _ = _sentinels()
    per_step = sentinel_legend("shadow_depth", 5.0)
    for needle in ("0 m", "magenta", "Gray", "5 m", "deeper"):
        assert needle.lower() in per_step.lower(), (
            f"the per-step shadow-depth legend does not mention {needle!r}: "
            f"{per_step!r}")
    assert "NOT 5 m" in per_step, (
        "the legend must say the sentinel is NOT the ceiling value; a user who "
        "reads magenta as '5 m' has read a number the engine refused to write")
    for key in sorted(fields - {"shadow_depth"}):
        stat = sentinel_legend(key, 5.0)
        assert "resolved" in stat.lower() and "gray" in stat.lower(), (
            f"the legend for {key} must say what gray means there: {stat!r}")
    assert "1.0" in sentinel_legend("shadow_unresolved_frac", 5.0)
    assert sentinel_legend("shadow_lit_frac", 5.0)
    assert sentinel_legend("temperature_max", 5.0) == "", (
        "a field with no sentinel must get no legend, or the pane becomes "
        "noise the user learns to ignore")
    assert "the search ceiling" in sentinel_legend("shadow_depth", None), (
        "with no sidecar the legend must say 'the search ceiling' rather than "
        "invent a height")


def test_no_single_off_scale_color_is_safe_and_the_code_knows_it():
    """⭐ The assumption that was FALSE, pinned as a measurement.

    The first version of this feature asserted in a comment that magenta is
    absent from every colormap the toolbar offers. It is not: magenta is
    EXACTLY the top of `cool`. This arm re-derives that against the real
    colormap list, so the claim can never quietly become a comment again.
    """
    try:
        from gui_panels.visualization_panel import (
            OFF_SCALE_ABOVE_CANDIDATES, OFF_SCALE_MIN_RGB_DISTANCE,
            VisualizationPanel, off_scale_above_color)
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"matplotlib is unavailable here ({exc})")

    def _closest(cmap_name, rgb):
        ramp = plt.get_cmap(cmap_name)(np.linspace(0.0, 1.0, 256))[:, :3]
        return float(np.sqrt(((ramp - np.asarray(rgb)) ** 2).sum(axis=1)).min())

    magenta = OFF_SCALE_ABOVE_CANDIDATES[0][1]
    assert _closest("cool", magenta) < 0.01, (
        "`cool` no longer ends at magenta, so the premise of "
        "`off_scale_above_color` has changed and its candidate order should "
        "be re-derived")

    # Whatever it picks, the picked color must actually clear the ramp.
    by_name = dict(OFF_SCALE_ABOVE_CANDIDATES)
    for cmap_name in VisualizationPanel.AVAILABLE_COLORMAPS:
        name, warning = off_scale_above_color(cmap_name)
        assert name in by_name, f"{cmap_name}: unknown color {name!r}"
        distance = _closest(cmap_name, by_name[name])
        if warning:
            # It may only fall back when EVERY candidate collides.
            for _cand, rgb in OFF_SCALE_ABOVE_CANDIDATES:
                assert _closest(cmap_name, rgb) < OFF_SCALE_MIN_RGB_DISTANCE, (
                    f"{cmap_name}: warned about a collision while a clear "
                    f"candidate existed")
            assert "colormap" in warning and cmap_name in warning, (
                f"{cmap_name}: the warning must name the colormap: {warning!r}")
        else:
            assert distance >= OFF_SCALE_MIN_RGB_DISTANCE, (
                f"{cmap_name}: chose {name!r}, which sits {distance:.3f} from "
                f"the ramp -- inside the {OFF_SCALE_MIN_RGB_DISTANCE} floor, "
                f"and reported no warning")
    # ⛔ The collision warning must be REACHABLE, not defensive decoration.
    # None of the 39 offered colormaps happens to swallow every candidate
    # today -- measured just above, not assumed -- so the branch is exercised
    # against a colormap built to contain all of them.
    from matplotlib.colors import ListedColormap
    everything = ListedColormap([rgb for _n, rgb in OFF_SCALE_ABOVE_CANDIDATES],
                                name="ptm_test_every_candidate")
    try:
        import matplotlib
        matplotlib.colormaps.register(everything, force=True)
        name, warning = off_scale_above_color("ptm_test_every_candidate")
    finally:
        try:
            matplotlib.colormaps.unregister("ptm_test_every_candidate")
        except Exception:                          # pragma: no cover
            pass
    assert warning and name == OFF_SCALE_ABOVE_CANDIDATES[0][0], (
        "a colormap containing every candidate must fall back to the first "
        f"and SAY SO; got name={name!r} warning={warning!r}")
    # An unsampleable colormap name must also warn rather than pretend.
    name, warning = off_scale_above_color("not_a_real_colormap")
    assert name == OFF_SCALE_ABOVE_CANDIDATES[0][0] and warning, (
        "an unknown colormap must fall back WITH a warning, not silently; "
        f"got name={name!r} warning={warning!r}")


def test_the_legend_names_the_color_that_is_actually_drawn():
    """A legend that says 'magenta' while the view draws cyan is worse than none."""
    try:
        from gui_panels.visualization_panel import (
            off_scale_above_color, sentinel_legend)
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    for cmap_name in ("cividis_r", "cool", "hsv"):
        above, warning = off_scale_above_color(cmap_name)
        text = sentinel_legend("shadow_depth", 5.0, above_color=above,
                               warning=warning)
        assert above.capitalize() in text or above in text, (
            f"under {cmap_name} the sentinel is drawn {above!r} and the legend "
            f"does not say so: {text!r}")
        if warning:
            assert warning in text, (
                f"under {cmap_name} the collision warning was dropped from the "
                f"legend: {text!r}")


# --- 3. A synthetic run directory -------------------------------------------

def _write_faces(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype="<f4")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<I", arr.size))
        handle.write(arr.tobytes())


N_FACES = 8


def _synthetic_run(root):
    """A run directory holding one file for every key the registry knows.

    Small and self-contained: the defect is about which files the viewer looks
    for, so eight facets prove it exactly as well as 2,397,890 do.
    """
    from ptm.results import STATISTICS_FIELDS, STAT_TYPES
    run = Path(root)
    plain = np.linspace(1.0, 2.0, N_FACES)
    for base, info in STATISTICS_FIELDS.items():
        if info.get("scalar"):
            _write_faces(run / "statistics" / f"{base}.bin", plain)
            continue
        for stat in STAT_TYPES:
            _write_faces(run / "statistics" / f"{base}_{stat}.bin", plain)
        for step in range(2):
            _write_faces(run / base / f"{base}_{step:04d}.bin", plain)
    # The shadow-depth series, with the real sentinel encoding: two facets lit
    # at the surface, two resolved, two above the ceiling, two not evaluated.
    depth = np.array([0.0, 0.0, 0.5, 4.5, np.inf, np.inf, np.nan, np.nan])
    for step in range(2):
        _write_faces(run / "shadow_depth" / f"shadow_depth_{step:04d}.bin", depth)
    # The statistics exclude zero and unresolved epochs, so NaN, never +inf.
    stat_depth = np.array([np.nan, np.nan, 0.5, 4.5, np.nan, np.nan, np.nan, np.nan])
    for stat in ("avg", "max", "min"):
        _write_faces(run / "statistics" / f"shadow_depth_{stat}.bin", stat_depth)
    (run / "statistics" / "shadow_depth.json").write_text(
        json.dumps({"ptm3d_shadow_depth_schema": 1, "h_max_m": 5,
                    "units": "meters"}), encoding="utf-8")
    (run / "time_steps.txt").write_text(
        "# time\tlabel\n0.0\tstep 0\n3600.0\tstep 1\n", encoding="utf-8")
    return run


# --- 4. The real widget ------------------------------------------------------

_APP = None
_SCRATCH = None


def _app():
    global _APP, _SCRATCH
    if _APP is not None:
        return _APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    # Redirect QSettings before any panel touches it, or this writes the real
    # com.ptm3d.* preferences.
    _SCRATCH = tempfile.mkdtemp(prefix="ptm_field_registry_test_")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, _SCRATCH)
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _loaded_panel(run_dir):
    """A real VisualizationPanel with `run_dir` loaded, shown and laid out."""
    app = _app()
    try:
        from gui_panels.visualization_panel import VisualizationPanel
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    panel = VisualizationPanel()
    panel.resize(1400, 900)
    panel.show()
    app.processEvents()
    panel.load_results(str(run_dir))
    app.processEvents()
    return panel


def _menu(panel):
    combo = panel.field_combo
    return [combo.itemData(i) for i in range(combo.count())]


def _rect_in(child, ancestor):
    """`child`'s rectangle in `ancestor`'s coordinates.

    ⛔ The question `isVisibleTo` does not answer. `isVisibleTo` is True for a
    widget scrolled or laid out entirely off the visible area; this is
    geometry, computed from Qt, so an arm cannot inherit the blind spot of the
    code it is checking.
    """
    from PyQt6.QtCore import QRect
    top_left = child.mapTo(ancestor, child.rect().topLeft())
    return QRect(top_left, child.size())


def _drop_down_arm(run_dir, label):
    panel = _loaded_panel(run_dir)
    combo = panel.field_combo
    keys = _menu(panel)

    assert combo.isVisibleTo(panel), (
        f"[{label}] the field drop-down is not visible to the panel; nothing "
        f"below this is meaningful")
    assert combo.isEnabled(), f"[{label}] the field drop-down is disabled"
    inside = _rect_in(combo, panel)
    assert panel.rect().contains(inside), (
        f"[{label}] the field drop-down sits at {inside} which is outside the "
        f"panel's own {panel.rect()}; it is 'visible' and unreachable")
    assert inside.width() > 0 and inside.height() > 0, (
        f"[{label}] the field drop-down has collapsed to {inside}")

    # ⛔ Only the products this run ACTUALLY WROTE. `shadow_depth.write_frames`
    # can be off, and `ptm_merge_chunks` never writes a merged frames directory
    # at all -- it merges the statistics and the sidecar only -- so demanding
    # the per-epoch series on every run with shadow-depth statistics fails on
    # the most common real shape, a merged chunked window, with nothing wrong.
    expected = ["shadow_depth_avg", "shadow_depth_max", "shadow_depth_min",
                "shadow_lit_frac", "shadow_unresolved_frac"]
    if (Path(run_dir) / "shadow_depth").is_dir():
        expected.insert(0, "shadow_depth")
    for key in expected:
        assert key in keys, (
            f"[{label}] '{key}' is on disk in {run_dir} but the field "
            f"drop-down does not offer it. Menu: {keys}")

    # Every entry carries a human label, not the raw key.
    from gui_panels.visualization_panel import VisualizationPanel
    for i, key in enumerate(keys):
        text = combo.itemText(i)
        assert text and text != key, (
            f"[{label}] '{key}' is listed as its raw file-name key {text!r}")
        assert text == VisualizationPanel.FIELD_INFO[key][0], (
            f"[{label}] '{key}' is labeled {text!r}, FIELD_INFO says "
            f"{VisualizationPanel.FIELD_INFO[key][0]!r}")
    return panel, keys


def test_the_drop_down_offers_every_shadow_depth_product():
    with tempfile.TemporaryDirectory(prefix="ptm_field_registry_run_") as tmp:
        run = _synthetic_run(tmp)
        _drop_down_arm(run, "synthetic")


def test_selecting_a_shadow_field_reads_the_file_that_exists():
    """The menu entry must route to the file the engine actually wrote.

    ⛔ The half of this bug that a "does the menu contain it" arm cannot see:
    a scalar statistic missing from the stats rule is listed in the menu and
    then looked for under `shadow_lit_frac/shadow_lit_frac_0000.bin`, which no
    run writes. The user picks it and the view silently does not change.
    """
    with tempfile.TemporaryDirectory(prefix="ptm_field_registry_run_") as tmp:
        run = _synthetic_run(tmp)
        panel = _loaded_panel(run)
        view = panel.view_3d
        for key in ("shadow_lit_frac", "shadow_unresolved_frac",
                    "shadow_depth_avg", "ice_depth"):
            data = view.load_field_data(key, 0, N_FACES)
            assert data is not None and len(data) == N_FACES, (
                f"'{key}' did not resolve to a readable file under {run}; "
                f"got {data!r}")
        series = view.load_field_data("shadow_depth", 0, N_FACES)
        assert series is not None and len(series) == N_FACES, (
            "the per-step shadow_depth series did not resolve")
        assert np.isposinf(series).sum() == 2, (
            "the reader must hand the +inf sentinel through untouched -- a "
            f"substituted number is a number the engine refused to write; got "
            f"{series!r}")


def test_the_color_scale_skips_the_sentinels():
    """Driven through the widget's own method, not the pure function."""
    import types
    with tempfile.TemporaryDirectory(prefix="ptm_field_registry_run_") as tmp:
        run = _synthetic_run(tmp)
        panel = _loaded_panel(run)
        view = panel.view_3d
        # The headless canvas has no mesh; the range reader needs only a face
        # count, so supply exactly that and nothing else.
        view.mesh = types.SimpleNamespace(n_cells=N_FACES)

        view.current_field = "shadow_depth"
        view.global_data_range = None
        assert view.compute_global_data_range() == (0.0, 5.0), (
            "the per-step shadow-depth color range must be [0, ceiling]; it "
            f"is {view.compute_global_data_range()}. A +inf in the range is "
            "the flat-color defect.")

        view.current_field = "shadow_depth_avg"
        assert view.compute_global_data_range() == (0.0, 5.0), (
            "the shadow-depth statistics must use the same [0, ceiling] scale "
            f"so the two views are comparable; got "
            f"{view.compute_global_data_range()}")

        view.current_field = "temperature"
        rng = view.compute_global_data_range()
        assert rng is not None and np.isfinite(rng).all(), (
            f"an ordinary field's range must be unchanged and finite: {rng}")
        assert rng == (1.0, 2.0), f"temperature range moved: {rng}"


def test_unticking_fixed_scale_does_not_restore_the_flat_color_defect():
    """⛔ The reachable path that put +inf back on an autoscaled bar.

    `on_fixed_scale_changed` seeded the two spin boxes and left
    `color_range_min/max` at None, so `update_display` passed no clim and
    PyVista autoscaled to (0, inf) -- at which range VTK draws +inf in the NaN
    color, making "deeper than we searched" and "not evaluated" the same pixel
    while the legend pane on screen said they were different colors.
    """
    import types
    with tempfile.TemporaryDirectory(prefix="ptm_field_registry_run_") as tmp:
        run = _synthetic_run(tmp)
        panel = _loaded_panel(run)
        app = _app()
        view = panel.view_3d
        view.mesh = types.SimpleNamespace(n_cells=N_FACES)

        from gui_panels.visualization_panel import SHADOW_SENTINEL_FIELDS
        for field in sorted(SHADOW_SENTINEL_FIELDS):
            view.current_field = field
            view.global_data_range = None
            view.fixed_color_scale = True
            fixed = view.effective_clim()
            assert fixed is not None and np.isfinite(fixed).all(), (
                f"{field}: the fixed-scale range is {fixed}")

            # The exact post-toggle state the defect left behind.
            view.fixed_color_scale = False
            view.color_range_min = None
            view.color_range_max = None
            manual = view.effective_clim()
            assert manual is not None and np.isfinite(manual).all(), (
                f"{field}: unticking 'Fixed Scale' before touching a spin box "
                f"yields clim={manual}, so PyVista autoscales over the "
                f"sentinel and +inf is drawn in the NaN color")

        # And the toggle itself must adopt the range it puts in the boxes.
        view.current_field = "shadow_depth"
        view.fixed_color_scale = True
        view.global_data_range = view.compute_global_data_range()
        assert view.global_data_range == (0.0, 5.0)
        from PyQt6.QtCore import Qt as _Qt
        view.on_fixed_scale_changed(_Qt.CheckState.Unchecked.value)
        app.processEvents()
        assert not view.fixed_color_scale, "the toggle did not take"
        assert (view.color_range_min, view.color_range_max) == (0.0, 5.0), (
            "switching to a manual scale displayed a range in the spin boxes "
            "that the view was not using: "
            f"{(view.color_range_min, view.color_range_max)}")

        # An ordinary field is untouched: no clim invented where there was none.
        view.current_field = "temperature"
        view.fixed_color_scale = False
        view.color_range_min = None
        view.color_range_max = None
        assert view.effective_clim() is None, (
            "a field with no sentinel must keep PyVista's autoscale when the "
            "manual range is unset; the fix must not widen to every field")


def test_the_legend_pane_appears_only_for_sentinel_fields():
    with tempfile.TemporaryDirectory(prefix="ptm_field_registry_run_") as tmp:
        run = _synthetic_run(tmp)
        panel = _loaded_panel(run)
        app = _app()

        # The default selection is temperature: no sentinel, no pane.
        assert panel.field_combo.currentData() == "temperature", (
            f"the default field moved to {panel.field_combo.currentData()!r}")
        assert not panel.field_note.isVisibleTo(panel), (
            "the legend pane is showing for a field with no sentinel")

        # ⛔ setCurrentIndex, not a direct call to the handler: the wiring is
        # what is under test, and a handler called by hand proves nothing
        # about whether the signal reaches it.
        idx = panel.field_combo.findData("shadow_depth")
        assert idx >= 0
        panel.field_combo.setCurrentIndex(idx)
        app.processEvents()
        assert panel.field_note.isVisibleTo(panel), (
            "the legend pane did not appear for shadow_depth; magenta and "
            "gray are then unlabeled on screen")
        text = panel.field_note.text()
        assert "5 m" in text and "magenta" in text.lower(), (
            f"the legend does not name the ceiling and the off-scale color: "
            f"{text!r}")
        inside = _rect_in(panel.field_note, panel)
        assert panel.rect().contains(inside), (
            f"the legend pane sits at {inside}, outside the panel's "
            f"{panel.rect()}")

        idx = panel.field_combo.findData("temperature_max")
        panel.field_combo.setCurrentIndex(idx)
        app.processEvents()
        assert not panel.field_note.isVisibleTo(panel), (
            "the legend pane stayed up after switching to a field with no "
            f"sentinel; it says {panel.field_note.text()!r}")


# --- 5. The REAL main window ------------------------------------------------
#
# The panel's own combo (section 4) is a sub-widget. The drop-down a user
# actually looks at sits on the main toolbar and is filled by
# `PTM3DMainWindow._refresh_field_combo`, which reads the SAME two class
# attributes -- so it inherited the same defect and must be shown to inherit
# the fix. Three rounds of a flux-cube defect got past green suites because
# every harness drove a sub-widget and never built the real window.

_WINDOW = None


def _real_window():
    """The real PTM3DMainWindow, built once. Skips if it cannot be built."""
    global _WINDOW
    if _WINDOW is not None:
        return _WINDOW
    _app()
    # ⛔ IF THE CANVAS IS LIVE, DO NOT BUILD THE WINDOW -- `pyvistaqt`'s
    # interactor needs a real GL context and SEGFAULTS the whole process under
    # the offscreen platform plugin, taking every other arm with it and
    # reporting nothing. A named refusal is recoverable; rc=139 is not.
    try:
        from gui_panels import visualization_panel as _vp
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the visualization panel is unavailable here ({exc})")
    if getattr(_vp, "PYVISTA_AVAILABLE", False):
        raise Skip(
            "the 3D canvas is live in this process, so building the real "
            "window would segfault under the offscreen platform plugin. Set "
            "PTM_GUI_DISABLE_3D=1 in the environment before the run.")
    try:
        from ptm3d_gui import PTM3DMainWindow
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the main window module is unavailable here ({exc})")
    _WINDOW = PTM3DMainWindow()
    _WINDOW.resize(1600, 1000)
    _WINDOW.show()
    _app().processEvents()
    return _WINDOW


def test_the_toolbar_drop_down_offers_every_shadow_depth_product():
    """The window's own Field selector, driven through its own refresh."""
    win = _real_window()
    app = _app()
    with tempfile.TemporaryDirectory(prefix="ptm_field_registry_run_") as tmp:
        run = _synthetic_run(tmp)
        win._refresh_field_combo(str(run))
        app.processEvents()
        combo = win.field_combo
        keys = [combo.itemData(i) for i in range(combo.count())]
        assert combo.isEnabled(), (
            "the toolbar field selector is disabled after loading a run that "
            f"has {combo.count()} fields on disk")
        assert combo.isVisibleTo(win), (
            "the toolbar field selector is not visible to the main window")
        inside = _rect_in(combo, win)
        assert win.rect().contains(inside), (
            f"the toolbar field selector sits at {inside}, outside the "
            f"window's own {win.rect()}")
        for key in ("shadow_depth", "shadow_depth_avg", "shadow_depth_max",
                    "shadow_depth_min", "shadow_lit_frac",
                    "shadow_unresolved_frac"):
            assert key in keys, (
                f"'{key}' is on disk in {run} but the main toolbar's Field "
                f"drop-down does not offer it. Menu: {keys}")


# --- 6. The real run, when the caller has one -------------------------------

def test_the_drop_down_offers_them_on_a_real_run():
    if not REAL_RUN:
        raise Skip("PTM_TEST_RUN_DIR is not set; section 4's synthetic "
                   "fixture is what ran")
    run = Path(REAL_RUN).expanduser()
    if not (run / "statistics" / "shadow_depth_avg.bin").is_file():
        raise Skip(f"{run} has no shadow-depth statistics")
    _, keys = _drop_down_arm(run, str(run))
    from ptm.results import depth_of_field, detect_available_statistics
    detected = detect_available_statistics(run / "statistics")
    for base, stat in detected:
        # ⛔ Subsurface temperature maps are EXCLUDED, and not as a
        # convenience: `ptm.results` states outright that they cannot live in
        # a static catalog because the field name carries the depth, so the
        # set is a property of the run. Asserting them here would fail on a
        # correct tree the moment anyone sets `output.map_depths_m`.
        if depth_of_field(base) is not None:
            continue
        key = base if stat is None else f"{base}_{stat}"
        assert key in keys, (
            f"`detect_available_statistics` reports '{key}' in {run} and the "
            f"field drop-down does not offer it -- the two registries have "
            f"drifted again. Menu: {keys}")


TESTS = [
    test_every_shared_registry_field_is_viewable_in_the_gui,
    test_the_gate_can_fail,
    test_the_scalar_statistics_lists_agree,
    test_is_stats_field_classifies_every_registry_key,
    test_a_plain_min_max_over_a_shadow_depth_array_is_the_defect,
    test_finite_data_range_refuses_to_invent_a_range,
    test_the_color_scale_bottom_is_pinned_at_exactly_zero,
    test_the_search_ceiling_comes_from_the_engine_or_not_at_all,
    test_the_legend_names_both_sentinels_and_says_the_ceiling,
    test_no_single_off_scale_color_is_safe_and_the_code_knows_it,
    test_the_legend_names_the_color_that_is_actually_drawn,
    test_the_drop_down_offers_every_shadow_depth_product,
    test_selecting_a_shadow_field_reads_the_file_that_exists,
    test_the_color_scale_skips_the_sentinels,
    test_unticking_fixed_scale_does_not_restore_the_flat_color_defect,
    test_the_legend_pane_appears_only_for_sentinel_fields,
    test_the_toolbar_drop_down_offers_every_shadow_depth_product,
    test_the_drop_down_offers_them_on_a_real_run,
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
    print(f"\n[test_gui_field_registry] {total - failures - skipped}/{total} passed"
          + (f", {skipped} skipped" if skipped else "")
          + (f", {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
