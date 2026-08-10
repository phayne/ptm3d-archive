#!/usr/bin/env python3
"""The Results/readers/plots layer, against a run directory built in a tempdir.

No engine, no SPICE, no GPU, no network: every check below writes the exact
byte layouts the engine writes and then reads them back. That is the point --
the formats are the contract, and three of them are documented nowhere in
``docs/``, which is why the tree carries 44 independent ``mesh.ply`` parsers
and 40 per-face ``.bin`` readers.

What is asserted, in the order it would break:

1. the readers round-trip the byte layouts, including the albedo map's
   *different* one (``size_t`` + float64, not ``int32`` + float32);
2. the mesh length unit is decided the way the engine decides it -- the same
   run written in metres and in kilometres reads back the same geometry, and
   the cases the engine calls fatal raise here too;
3. ``Results`` discovers fields and statistics from the directory rather than
   from a hard-coded list;
4. the two flux-unit spellings stay distinct all the way to an axis label;
5. every canned plot draws on the Agg backend.

Run directly (no pytest needed)::

    python3 src/python/tests/test_results_api.py
"""

from __future__ import annotations

import shutil
import struct
import sys
import tempfile
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

import numpy as np                                              # noqa: E402

from ptm import readers                                         # noqa: E402
from ptm.readers import MeshUnitError                           # noqa: E402
from ptm.results import Results                                 # noqa: E402

MOON_RADIUS_KM = 1737.4


# --- a synthetic run directory ----------------------------------------------

def _icosphere(subdivisions=2):
    """A unit icosphere: ``(vertices, faces)``. Closed, pure triangles."""
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
             (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
             (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [tuple(np.array(v) / np.linalg.norm(v)) for v in verts]
    for _ in range(subdivisions):
        midpoint, new_faces = {}, []

        def _mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                m = np.array(verts[a]) + np.array(verts[b])
                verts.append(tuple(m / np.linalg.norm(m)))
                midpoint[key] = len(verts) - 1
            return midpoint[key]

        for a, b, c in faces:
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def _write_ply(path, vertices, faces):
    """``mesh.ply`` exactly as the engine writes it.

    Vertex ``6 x float64`` (xyz then three zero normals); face ``uchar(3)`` +
    ``3 x int32`` + ``float32`` latitude + ``float32`` longitude in degrees.
    """
    centers = vertices[faces].mean(axis=1)
    radius = np.linalg.norm(centers, axis=1)
    lat = np.degrees(np.arcsin(centers[:, 2] / radius))
    lon = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    with open(path, "wb") as handle:
        handle.write(
            b"ply\nformat binary_little_endian 1.0\n"
            b"element vertex %d\n"
            b"property double x\nproperty double y\nproperty double z\n"
            b"property double nx\nproperty double ny\nproperty double nz\n"
            b"element face %d\n"
            b"property list uchar int vertex_indices\n"
            b"property float latitude\nproperty float longitude\n"
            b"end_header\n" % (len(vertices), len(faces)))
        block = np.zeros((len(vertices), 6), dtype="<f8")
        block[:, :3] = vertices
        handle.write(block.tobytes())
        for i, (a, b, c) in enumerate(faces):
            handle.write(struct.pack("<B3iff", 3, a, b, c, lat[i], lon[i]))
    return lat, lon


def _write_bin(path, values):
    """A per-face ``.bin``: ``int32`` count then that many little-endian float32."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype="<f4")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<I", values.size))
        handle.write(values.tobytes())


def make_run(root, unit="km", n_steps=8, extra_fields=()):
    """Build a complete run directory and return its path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vertices, faces = _icosphere(2)
    scale = MOON_RADIUS_KM * (1000.0 if unit == "m" else 1.0)
    lat, lon = _write_ply(root / "mesh.ply", vertices * scale, faces)
    n_faces = len(faces)

    # A field with real latitude structure, so a map or a globe of it is
    # recognizable rather than noise. No physics: this is a cosine, not a model.
    base = 100.0 + 250.0 * np.clip(np.cos(np.radians(lat)), 0.0, None)
    for step in range(n_steps):
        phase = 2 * np.pi * step / n_steps
        lit = np.clip(np.cos(np.radians(lon) - phase), 0.0, None)
        _write_bin(root / "temperature" / f"temperature_{step:04d}.bin",
                   base * (0.4 + 0.6 * lit))
        _write_bin(root / "direct_solar" / f"direct_solar_{step:04d}.bin",
                   1361.0 * lit * np.clip(np.cos(np.radians(lat)), 0.0, None))
        for name in extra_fields:
            _write_bin(root / name / f"{name}_{step:04d}.bin",
                       np.full(n_faces, 1.0 + step, dtype=np.float32))

    stats = root / "statistics"
    stack = np.stack([
        np.frombuffer((root / "temperature" /
                       f"temperature_{s:04d}.bin").read_bytes(),
                      dtype="<f4", offset=4)
        for s in range(n_steps)])
    for moment, func in (("avg", np.mean), ("max", np.max), ("min", np.min)):
        _write_bin(stats / f"temperature_{moment}.bin", func(stack, axis=0))
    # Written by every run, and absent from STATISTICS_FIELDS.
    _write_bin(stats / "effective_albedo.bin", np.full(n_faces, 0.12))
    for name in extra_fields:
        _write_bin(stats / f"{name}_avg.bin", np.full(n_faces, 2.0))

    et0 = 8.28e8
    lines = ["# Time steps metadata",
             "# Column 1: Time elapsed from start (seconds)",
             "# Column 2: UTC time"]
    for step in range(n_steps):
        lines.append(f"{et0 + step * 3600.0:.6f}\t2026-04-{step + 1:02d}T00:00:00.000")
    (root / "time_steps.txt").write_text("\n".join(lines) + "\n")

    rows = ["time,sun_x,sun_y,sun_z,primary_x,primary_y,primary_z"]
    for step in range(n_steps):
        phase = 2 * np.pi * step / n_steps
        rows.append(f"2026-04-{step + 1:02d}T00:00:00.000,"
                    f"{1.5e8 * np.cos(phase)},{1.5e8 * np.sin(phase)},0.0,"
                    f"{4.0e5 * np.cos(phase)},{4.0e5 * np.sin(phase)},0.0")
    (root / "body_positions.csv").write_text("\n".join(rows) + "\n")

    components = ",".join(("temperature", "direct_solar") + tuple(extra_fields))
    (root / "metadata.txt").write_text(
        f"faces={n_faces}\ntime_steps={n_steps}\n"
        f"successful_timesteps={n_steps}\nfailed_timesteps=0\n"
        f"is_custom_mesh=false\ncomponents={components}\n")
    return root


# --- 1. byte layouts ---------------------------------------------------------

def test_time_steps_column_one_is_et_and_is_rebased(tmp):
    run = Results(make_run(tmp / "km_run"))
    times = run.times()
    assert times.et[0] > 8.0e8, (
        "column 1 of time_steps.txt is SPICE ET (seconds past J2000); a reader "
        "that treats it as elapsed time gets an axis starting at 8.3e8")
    assert times.elapsed_s[0] == 0.0
    assert np.allclose(np.diff(times.elapsed_s), 3600.0)
    assert len(times.utc) == len(times.et)


def test_albedo_map_is_a_different_binary_format(tmp):
    values = np.linspace(0.05, 0.4, 320)
    path = tmp / "albedo.bin"
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", values.size))
        handle.write(values.astype("<f8").tobytes())
    back = readers.read_albedo_map(path, expected_count=320)
    assert np.allclose(back, values)

    # The statistics layout must not be readable as an albedo map: silently
    # returning garbage is the failure this reader exists to prevent.
    stats_like = tmp / "stats_like.bin"
    _write_bin(stats_like, values.astype(np.float32))
    try:
        readers.read_albedo_map(stats_like)
    except ValueError as exc:
        assert "statistics" in str(exc)
    else:
        raise AssertionError("an int32+float32 file was accepted as an albedo map")


def test_body_positions_and_subsolar_point(tmp):
    run = Results(make_run(tmp / "km_run"))
    positions = run.body_positions()
    assert positions.shape[0] == 8
    assert "sun_x" in positions.dtype.names
    lat, lon = run.subsolar_latlon(0)
    assert abs(lat) < 1e-9 and abs(lon) < 1e-9
    lat, lon = run.subsolar_latlon(2)          # a quarter turn
    assert abs(lat) < 1e-9 and abs(lon - 90.0) < 1e-9


def test_find_nearest_face_is_longitude_convention_safe(tmp):
    run = Results(make_run(tmp / "km_run"))
    index_a, sep_a = run.face_at_verbose(30.0, -170.0)
    index_b, sep_b = run.face_at_verbose(30.0, 190.0)   # the same meridian
    assert index_a == index_b, (
        "a target on [0, 360) and one on [-180, 180] naming the same point "
        "must select the same face")
    assert abs(sep_a - sep_b) < 1e-9


# --- 2. the mesh length unit -------------------------------------------------

def test_same_geometry_in_metres_and_kilometres(tmp):
    """The trap, stated as a test: one body, two units, no unit in the file."""
    km = Results(make_run(tmp / "u_km", unit="km"))
    m = Results(make_run(tmp / "u_m", unit="m"))
    assert km.mesh_units == "km" and m.mesh_units == "m"
    assert np.allclose(km.face_centers_m, m.face_centers_m, rtol=1e-9)
    assert np.allclose(km.face_areas_m2, m.face_areas_m2, rtol=1e-9)
    assert np.allclose(km.lat, m.lat) and np.allclose(km.lon, m.lon)
    # Both must corroborate against a real radius rather than guess.
    assert km.mesh_unit_report.corroborated
    assert m.mesh_unit_report.corroborated


def test_unit_inference_refuses_what_the_engine_refuses():
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(200, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Straddling the 10,000 discriminator: consumers classify from one face.
    straddle = directions * np.linspace(5e3, 5e4, 200)[:, None]
    _expect_refusal(straddle, "straddle")

    # Inside the ambiguous band with no known radius to confirm it.
    ambiguous = directions * 9.0e3
    _expect_refusal(ambiguous, "ambiguous band")

    # Face centers rescaled away from the vertices (the mesh.scale_factor bug).
    centers = directions * MOON_RADIUS_KM
    _expect_refusal(centers, "internally inconsistent",
                    vertices=directions * MOON_RADIUS_KM * 1000.0)


def _expect_refusal(centers, fragment, vertices=None):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            readers.infer_mesh_length_unit(centers, vertices=vertices)
    except MeshUnitError as exc:
        assert fragment in str(exc), f"expected {fragment!r} in: {exc}"
    else:
        raise AssertionError(f"no refusal for the {fragment!r} case")


def test_magnitude_only_verdict_warns():
    directions = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = readers.infer_mesh_length_unit(directions * 3.0)
    assert report.unit == "km" and not report.corroborated
    assert any("magnitude alone" in str(w.message) for w in caught), (
        "a unit decided on magnitude alone must say so")


def test_forced_mesh_units_are_honoured(tmp):
    run = Results(make_run(tmp / "u_km"), mesh_units="m")
    assert run.mesh_units == "m"
    assert not run.mesh_unit_report.corroborated


# --- 3. discovery ------------------------------------------------------------

def test_fields_and_statistics_are_discovered_not_listed(tmp):
    run = Results(make_run(tmp / "two_band", extra_fields=("indirect_solar",
                                                           "indirect_ir")))
    assert set(run.step_fields) == {"temperature", "direct_solar",
                                    "indirect_solar", "indirect_ir"}, (
        "per-step fields must come from the directory: a two-band radiosity "
        "run writes indirect_solar and indirect_ir, and a hard-coded list "
        "drops them")
    stats = dict(run.stats)
    assert ("effective_albedo", None) in run.stats, (
        "effective_albedo.bin is written by every run and is not in "
        "STATISTICS_FIELDS, so it must be discovered")
    assert stats.get("indirect_solar") is not None


def test_metadata_face_count_mismatch_is_fatal(tmp):
    root = make_run(tmp / "bad_meta")
    text = (root / "metadata.txt").read_text().replace("faces=320", "faces=321")
    (root / "metadata.txt").write_text(text)
    try:
        Results(root).mesh
    except ValueError as exc:
        assert "inconsistent" in str(exc)
    else:
        raise AssertionError("a metadata/mesh face-count mismatch was accepted")


def test_truncated_field_file_raises(tmp):
    root = make_run(tmp / "truncated")
    target = root / "temperature" / "temperature_0000.bin"
    target.write_bytes(target.read_bytes()[:-40])
    try:
        Results(root).field("temperature", 0)
    except ValueError as exc:
        assert "readable per-face binary" in str(exc)
    else:
        raise AssertionError("a truncated per-face binary was accepted")


def test_series_and_local_solar_time(tmp):
    run = Results(make_run(tmp / "km_run"))
    hours, values = run.series("temperature", lat=0.0, lon=0.0)
    assert hours.shape == values.shape == (8,)
    assert hours[0] == 0.0 and np.isclose(hours[-1], 7.0)
    lst, mean, p25, p75, n = run.local_solar_time_curve(n_bins=8, lat_max=30.0)
    assert lst.shape == (8,) and n > 0
    assert np.nanmax(mean) > np.nanmin(mean)


def test_provenance_says_what_is_missing(tmp):
    prov = Results(make_run(tmp / "km_run")).provenance()
    assert prov["n_faces"] == 320
    assert prov["config_sha256"] is None
    assert prov["engine"] == {}, "a run with no provenance.txt reports no engine"
    joined = " ".join(prov["not_recorded"])
    assert "engine binary md5" in joined and "config" in joined, (
        "provenance must name what the run does not record rather than "
        "inferring it")


def test_provenance_reads_the_engine_record(tmp):
    """A run WITH provenance.txt stops reporting the fields as missing.

    The engine has written ``provenance.txt`` since schema
    ``ptm3d-provenance-1``. This is the reader's half of that contract: the
    field names below are the ones ``run_provenance.cpp`` writes, and if either
    side renames one, provenance silently goes back to ``not_recorded`` --
    which looks like an old run rather than like a bug. Hence an explicit test
    rather than trust.
    """
    root = make_run(tmp / "km_run_prov")
    (root / "config_used.yaml").write_text("simulation:\n  body: Moon\n")
    (root / "provenance.txt").write_text(
        "schema=ptm3d-provenance-1\n"
        "run_started_utc=2026-08-09T12:00:00Z\n"
        "engine_version=1.1\n"
        "engine_git_sha=abcdef0123\n"
        "engine_build_type=Release\n"
        "engine_gpu_enabled=false\n"
        "engine_binary=/somewhere/ptm_thermal_sim\n"
        "engine_binary_md5=0123456789abcdef0123456789abcdef\n"
        "engine_shared_libraries=false\n"
        "command_line=ptm_thermal_sim --config x.yaml --albedo 0.2\n"
        "body=Moon\n"
        "generic_body=false\n"
        "mesh_source=icosphere, resolution 2\n"
        "mesh_faces=320\n"
        "mesh_length_unit=kilometers\n"
        "config_copy=config_used.yaml\n"
        "config_md5=00112233445566778899aabbccddeeff\n")

    prov = Results(root).provenance()
    assert prov["engine_version"] == "1.1"
    assert prov["engine_git_sha"] == "abcdef0123"
    assert prov["engine_binary_md5"] == "0123456789abcdef0123456789abcdef"
    assert prov["body"] == "Moon"
    assert "--albedo 0.2" in prov["command_line"]
    # Compare by identity, not by string: Results resolves its path, and on
    # macOS the temp directory is reached through the /var -> /private/var
    # symlink, so the two spellings differ while naming one file.
    assert Path(prov["config_file"]).samefile(root / "config_used.yaml")
    assert prov["config_sha256"] is not None
    assert prov["config_md5_recorded"] == "00112233445566778899aabbccddeeff"
    assert prov["started_utc"] == "2026-08-09T12:00:00Z"

    joined = " ".join(prov["not_recorded"])
    assert "engine binary md5" not in joined
    assert "predates provenance.txt" not in joined
    # 'kilometers' and 'km' are the same unit spelled two ways; a vocabulary
    # difference must not be reported as a disagreement.
    assert "agreed mesh length unit" not in joined
    assert prov["mesh_units_recorded_by_engine"] == "kilometers"

    # A build with shared project libraries must say the executable md5 is not
    # the whole engine -- silence there is how a dylib-only change gets missed.
    (root / "provenance.txt").write_text(
        (root / "provenance.txt").read_text().replace(
            "engine_shared_libraries=false", "engine_shared_libraries=true"))
    shared = " ".join(Results(root).provenance()["not_recorded"])
    assert "shared project libraries" in shared


# --- 4. the two unit spellings ------------------------------------------------

def test_units_style_survives_to_the_axis_label(tmp):
    root = make_run(tmp / "km_run")
    ascii_run = Results(root, units_style="ascii")
    unicode_run = Results(root, units_style="unicode")
    assert ascii_run.units_of("illumination") == "W/m^2"
    assert unicode_run.units_of("illumination") == "W/m²"
    assert ascii_run.axis_label("illumination", "avg").endswith("(W/m^2)")
    assert unicode_run.axis_label("illumination", "avg").endswith("(W/m²)")
    assert ascii_run.units_of("temperature") == unicode_run.units_of("temperature")
    try:
        Results(root, units_style="latex")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown units_style was accepted")


# --- 5. the plots draw --------------------------------------------------------

def test_every_canned_plot_draws(tmp):
    import matplotlib
    matplotlib.use("Agg")
    from ptm import plots

    run = Results(make_run(tmp / "km_run"))
    out = tmp / "figs"
    out.mkdir(exist_ok=True)
    run.plot_diurnal(lat=0, lon=0, save=out / "diurnal.png")
    run.plot_globes("temperature", "max", save=out / "globes.png")
    run.plot_histogram("temperature", "max", save=out / "hist.png")
    try:
        run.plot_map("temperature", "max", save=out / "map.png")
    except RuntimeError as exc:
        assert "scipy" in str(exc)          # optional dependency, named clearly
    for name in ("diurnal", "globes", "hist"):
        assert (out / f"{name}.png").stat().st_size > 5000, f"{name} is blank"

    norm = plots.flux_norm(np.array([100.0, 200.0, 390.0]))
    assert plots.clip_extend(np.array([50.0, 400.0]), norm) == "both"


def test_globes_refuse_a_site_patch(tmp):
    import matplotlib
    matplotlib.use("Agg")
    from ptm import plots

    root = make_run(tmp / "patch_run")
    run = Results(root)
    full = plots.mesh_angular_radius(run.face_unit_vectors)
    assert full > 170.0, "a closed icosphere covers the whole sphere"

    # Keep only the faces near one pole, in place, so the same Results object
    # now describes a site patch of the body it described a moment ago.
    keep = np.where(run.lat > 80.0)[0]
    mesh = run.mesh
    mesh["face_centers"] = mesh["face_centers"][keep]
    mesh["face_latitudes"] = mesh["face_latitudes"][keep]
    mesh["face_longitudes"] = mesh["face_longitudes"][keep]
    assert plots.mesh_angular_radius(run.face_unit_vectors) < 30.0
    try:
        plots.plot_globes(run, "temperature", "max")
    except ValueError as exc:
        assert "site patch" in str(exc)
    else:
        raise AssertionError("a globe of a site patch was drawn")


def test_scatter_map_catches_transposed_lat_lon(tmp):
    """(ax, lat, lon, values) drawn as (lon, lat) is a plausible wrong figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ptm import plots

    run = Results(make_run(tmp / "sm_run"))
    values = run.stat("temperature", "max")
    fig, ax = plt.subplots()
    plots.scatter_map(ax, run.lat, run.lon, values)          # correct order
    try:
        plots.scatter_map(ax, run.lon, run.lat, values)      # transposed
    except ValueError as exc:
        assert "transposed" in str(exc), exc
    else:
        raise AssertionError("a transposed lat/lon was drawn without complaint")
    plt.close(fig)


def test_globe_grid_rows_views_and_per_view_fields(tmp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ptm import plots

    run = Results(make_run(tmp / "gg_run"))
    out = tmp / "figs2"
    out.mkdir(exist_ok=True)
    tmax = run.stat("temperature", "max")
    delta = tmax - run.stat("temperature", "min")

    fig = run.globe_grid(
        rows=[{"what": "temperature", "moment": "max"},
              {"what": delta, "label": "range (K)", "cmap": plots.CMAP_DIFF,
               "norm": plots.sym_norm(delta)}],
        views=[(90, 0, "north"), (-90, 0, "south")],
        panel_labels=True, suptitle="grid", save=out / "grid.png")
    assert len(fig.axes) == 4 + 2, "expected 4 panels and 2 colorbars"
    assert (out / "grid.png").stat().st_size > 5000
    plt.close(fig)

    # One row, a different field per view -- the two-output-steps comparison.
    fig = run.globe_grid(
        rows=[{"what": [run.field("temperature", 0), run.field("temperature", 4)],
               "label": "T (K)"}],
        views=[(0, 0, "step 0"), (0, 0, "step 4")], save=out / "steps.png")
    assert len(fig.axes) == 2 + 1
    plt.close(fig)

    # A per-view list of the wrong length is a mistake, not a broadcast.
    try:
        run.globe_grid(rows=[{"what": [tmax, tmax, tmax]}],
                       views=[(0, 0), (0, 180)])
    except ValueError as exc:
        assert "one entry per view" in str(exc), exc
    else:
        raise AssertionError("a mismatched per-view list was accepted")
    plt.close("all")


def test_flux_scale_default_is_peak_temperature_only(tmp):
    """A day/night snapshot must not default to the T**4 colour scale.

    On the Moon quickstart it puts 51% of the body -- everything below 150 K --
    into the bottom 2.1% of the bar, one flat black. The rule is the same one
    plot_map, plot_globes and globe_grid all read.
    """
    from ptm import plots
    assert plots._flux_by_default("temperature", "max", None) is True
    assert plots._flux_by_default("temperature", "max", 5) is False
    assert plots._flux_by_default("temperature", "min", None) is False
    assert plots._flux_by_default("temperature", "avg", None) is False
    assert plots._flux_by_default("direct_solar", "max", None) is False
    assert plots._flux_by_default(np.zeros(3), "max", None) is False


def test_shape_draws_real_facets_and_culls(tmp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ptm import plots

    run = Results(make_run(tmp / "shape_run"))
    out = tmp / "figs3"
    out.mkdir(exist_ok=True)

    # Culling keeps about half a closed body, and never all of it.
    mesh = run.mesh
    normals = plots._face_normals(mesh["vertices"], mesh["faces"])
    visible = normals @ plots._view_direction(20.0, 35.0) > 0
    assert 0.3 < visible.mean() < 0.7, visible.mean()

    fig = run.plot_shape(what="temperature", moment="max",
                         save=out / "shape.png")
    assert (out / "shape.png").stat().st_size > 5000, "shape figure is blank"
    plt.close(fig)

    # Several views drop the 3-D boxes, one view keeps them.
    fig = run.plot_shape(what="temperature", moment="max",
                         views=[(20, 35), (20, 215)])
    assert fig.axes[0].get_zlabel() == "", \
        "a multi-view shape kept its overprinting axis labels"
    plt.close(fig)
    fig = run.plot_shape(what="temperature", moment="max")
    assert "km" in fig.axes[0].get_zlabel(), \
        f"a single view lost its length scale: {fig.axes[0].get_zlabel()!r}"
    plt.close(fig)

    # The vertices are drawn where they are: the drawn extent must match the
    # mesh's own, not a unit sphere.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plots.shape(ax, mesh["vertices"], mesh["faces"],
                run.stat("temperature", "max"))
    span = float(np.abs(mesh["vertices"]).max())
    assert abs(ax.get_zlim()[1] - span) < 1e-6, (ax.get_zlim(), span)
    plt.close(fig)


TESTS = [
    test_time_steps_column_one_is_et_and_is_rebased,
    test_albedo_map_is_a_different_binary_format,
    test_body_positions_and_subsolar_point,
    test_find_nearest_face_is_longitude_convention_safe,
    test_same_geometry_in_metres_and_kilometres,
    test_unit_inference_refuses_what_the_engine_refuses,
    test_magnitude_only_verdict_warns,
    test_forced_mesh_units_are_honoured,
    test_fields_and_statistics_are_discovered_not_listed,
    test_metadata_face_count_mismatch_is_fatal,
    test_truncated_field_file_raises,
    test_series_and_local_solar_time,
    test_provenance_says_what_is_missing,
    test_provenance_reads_the_engine_record,
    test_units_style_survives_to_the_axis_label,
    test_every_canned_plot_draws,
    test_globes_refuse_a_site_patch,
    test_scatter_map_catches_transposed_lat_lon,
    test_globe_grid_rows_views_and_per_view_fields,
    test_flux_scale_default_is_peak_temperature_only,
    test_shape_draws_real_facets_and_culls,
]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ptm_results_api_"))
    failures = 0
    try:
        for test in TESTS:
            needs_tmp = test.__code__.co_argcount == 1
            try:
                test(tmp) if needs_tmp else test()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {test.__name__}\n     {exc}")
            except Exception as exc:                   # noqa: BLE001
                failures += 1
                print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {test.__name__}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(TESTS)
    print(f"\n[test_results_api] {total - failures}/{total} passed")
    if failures:
        print("[test_results_api] FAIL")
        return 1
    print("[test_results_api] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
