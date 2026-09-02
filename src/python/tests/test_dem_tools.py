#!/usr/bin/env python3
"""DEM tools: the binary does what the GUI asks, and the GUI asks for flags that exist.

Covers the compiled DTM converter (``src/cpp/tools/dtm_to_ply_delaunay.cpp``)
and the GUI wrapper that runs it (``gui/dem_core/dem_runner.py``). See
``docs/dem-to-mesh-workflow.md``.

This file **builds nothing**. Three groups of checks, with different needs:

* **Argv coverage** (no binaries, no GDAL, no Qt -- always runs). Every
  ``--flag`` the GUI emits must be one the C++ ``main()`` actually parses. This
  is a source-level cross-check because the failure mode it guards against is
  silent: the tool ignores an argument it does not recognize, so a renamed or
  dropped flag changes the mesh with nothing in the log. ``--south-pole-coords``
  is known-dead and asserted to still carry its warning, so that implementing it
  forces a deliberate edit here.

* **Synthetic-DTM smoke** (needs ``build/bin/dtm_to_ply_delaunay`` and the GDAL
  Python bindings). Generates its own south-polar lunar GeoTIFF -- no fixture
  data -- converts it with and without ROI flags, and *inspects the meshes*. A
  zero exit code proves nothing here: what is checked is that faces are real
  triangles with in-range indices, that vertex radii match ``body_radius +
  elevation``, that ``--roi-radius`` genuinely bounds the extent, that
  ``--roi-max-points`` is honored, and that ``--roi-core-frac`` /
  ``--density-exp`` really do grade the point density from core to rim.

* **Refusal, and staying removed.** A request that cannot be honored must fail
  rather than write an empty mesh. And ``dtm_advanced_processor`` -- the
  adaptive refiner that was never implemented, wrote a 432-byte 0-vertex PLY,
  printed "Processing completed successfully!" and exited 0 -- was **removed
  from the repository** by Paul's ruling. A tripwire here keeps it removed: a
  button that shells out to a binary that does not exist is the defect class
  this whole change closes, so it must not come back by accident.

Missing prerequisites SKIP, naming exactly what is missing; a skip is printed
and reported, never counted as a pass. Run directly::

    python3 src/python/tests/test_dem_tools.py

or under pytest (use ``-rs`` to see skip reasons)::

    python3 -m pytest src/python/tests/test_dem_tools.py -rs
"""

from __future__ import annotations

import atexit
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CONVERT_BIN = REPO_ROOT / "build" / "bin" / "dtm_to_ply_delaunay"
CONVERT_SRC = REPO_ROOT / "src" / "cpp" / "tools" / "dtm_to_ply_delaunay.cpp"
RUNNER_PY = REPO_ROOT / "gui" / "dem_core" / "dem_runner.py"

# Removed by Paul's ruling (never implemented). See
# test_the_unimplemented_refiner_stays_removed.
REMOVED_REFINER_SRC = REPO_ROOT / "src" / "cpp" / "tools" / "dtm_advanced_processor.cpp"

MOON_RADIUS_M = 1737400.0

# Synthetic DTM geometry, shared by every smoke arm.
_DTM_N = 400            # pixels per side
_DTM_PX_M = 20.0        # meters per pixel
_ROI_LAT = -89.95
_ROI_LON = 0.0
_ROI_RADIUS_KM = 2.0
_ROI_MAX_POINTS = 8000
_ROI_CORE_FRAC = 0.3

# South-polar-stereographic CRS on a 1737400 m sphere, as a literal so the
# fixture never depends on a working proj.db. See _write_synthetic_dtm().
_POLAR_STEREO_WKT = (
    'PROJCS["unknown",GEOGCS["unknown",DATUM["unknown",'
    'SPHEROID["unknown",1737400,0]],PRIMEM["Reference meridian",0],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],PARAMETER["false_easting",0],'
    # The OGC WKT linear-unit name is the British spelling. It is a format
    # literal, not prose -- GDAL will not parse the American one here.
    'PARAMETER["false_northing",0],UNIT["metre",1],'  # american-english: allow
    'AXIS["Easting",NORTH],AXIS["Northing",NORTH]]'
)


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


_SCRATCH: Path | None = None


def _cleanup_scratch():
    if _SCRATCH is not None:
        shutil.rmtree(_SCRATCH, ignore_errors=True)


def _scratch() -> Path:
    """Temporary working directory, removed at interpreter exit.

    Registered with atexit rather than cleaned up only in main(), so a pytest
    run does not leave the synthetic DTM and its meshes behind in /tmp.
    """
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = Path(tempfile.mkdtemp(prefix="ptm_dem_tools_"))
        atexit.register(_cleanup_scratch)
    return _SCRATCH


# ---------------------------------------------------------------------------
# Flag inventories, read from source
# ---------------------------------------------------------------------------

def _implemented_flags(cpp_path: Path) -> set[str]:
    """Flags a tool's main() actually compares argv against."""
    if not cpp_path.is_file():
        raise Skip(f"source not found: {cpp_path}")
    text = cpp_path.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r'arg\s*==\s*"(--[a-z0-9-]+)"', text))


def _emitted_flags(runner_path: Path) -> set[str]:
    """Flags DEMConvertThread puts on dtm_to_ply_delaunay's command line.

    Scoped to the thread class rather than the whole module, so a flag named
    only in a docstring or a help string is never credited as emitted. Both
    ``cmd.extend([...])`` and ``cmd.append(...)`` forms count.
    """
    if not runner_path.is_file():
        raise Skip(f"runner not found: {runner_path}")
    text = runner_path.read_text(encoding="utf-8", errors="replace")
    body = text[text.index("class DEMConvertThread"):]
    return set(re.findall(r'"(--[a-z0-9-]+)"', body))


def test_every_flag_the_gui_emits_is_implemented_by_the_converter():
    emitted = _emitted_flags(RUNNER_PY)
    implemented = _implemented_flags(CONVERT_SRC)
    assert emitted, "parsed no flags out of DEMConvertThread -- parser is broken"
    unknown = sorted(emitted - implemented)
    assert not unknown, (
        f"DEMConvertThread emits flags dtm_to_ply_delaunay does not parse: "
        f"{unknown}. The tool IGNORES unknown arguments, so this would change "
        f"the mesh silently. Implemented: {sorted(implemented)}"
    )


def test_the_dead_flag_still_announces_itself():
    """--south-pole-coords is parsed and never used.

    It is wired to a GUI checkbox, so a user can switch it on and get no
    effect. The only honest state is a warning. If someone implements it, this
    arm fails and forces the warning (and docs/dem-to-mesh-workflow.md) to be
    updated deliberately rather than left behind as a lie.
    """
    convert = CONVERT_SRC.read_text(encoding="utf-8", errors="replace")
    assert "--south-pole-coords is accepted but NOT IMPLEMENTED" in convert, (
        "dtm_to_ply_delaunay must warn that --south-pole-coords is a no-op "
        "(use_south_pole_coords_ is stored and never read), or implement it "
        "and update this test and docs/dem-to-mesh-workflow.md"
    )


def test_the_unimplemented_refiner_stays_removed():
    """dtm_advanced_processor was removed; nothing may reach for it again.

    Paul's ruling. It was never implemented -- three comment stubs in main(),
    an elevation callback returning 0.0 for every query -- and it wrote a
    432-byte 0-vertex PLY while printing "Processing completed successfully!"
    and exiting 0. A GUI button that shells out to a binary that does not exist
    is the defect class this change closes, so the button went with the binary.

    This arm is a tripwire, not a style check: reintroducing the source, the
    CMake target or the GUI thread without also implementing the pipeline puts
    the silent failure straight back.
    """
    assert not REMOVED_REFINER_SRC.exists(), (
        f"{REMOVED_REFINER_SRC} is back. If the adaptive refiner has actually "
        f"been implemented (initial mesh from the DTM, a real getElevation), "
        f"delete this arm deliberately and add real coverage. Do not restore "
        f"the unimplemented version."
    )

    cmake = (REPO_ROOT / "src" / "cpp" / "tools" / "CMakeLists.txt").read_text(
        encoding="utf-8", errors="replace")
    assert "dtm_advanced_processor" not in cmake, (
        "src/cpp/tools/CMakeLists.txt still names dtm_advanced_processor"
    )

    runner = RUNNER_PY.read_text(encoding="utf-8", errors="replace")
    assert "dtm_advanced_processor" not in runner, (
        "gui/dem_core/dem_runner.py still names dtm_advanced_processor"
    )
    assert "DEMRefineThread" not in runner, (
        "gui/dem_core/dem_runner.py still defines DEMRefineThread"
    )

    panel = (REPO_ROOT / "gui" / "dem_panels" / "dem_config_panel.py").read_text(
        encoding="utf-8", errors="replace")
    for needle in ("DEMRefineThread", "_create_refine_page", "_run_refine",
                   '"Refine"'):
        assert needle not in panel, (
            f"gui/dem_panels/dem_config_panel.py still carries {needle!r} -- "
            f"the Refine page must stay gone"
        )


def test_the_nav_rail_still_offers_the_three_surviving_pages():
    """Removing a page must not leave the rail and the stack out of step.

    The rail is a plain QListWidget whose row index is fed straight to
    QStackedWidget.setCurrentIndex, so a name list and a widget list that
    disagree in length silently mis-route clicks rather than raising.
    """
    panel = (REPO_ROOT / "gui" / "dem_panels" / "dem_config_panel.py").read_text(
        encoding="utf-8", errors="replace")

    match = re.search(r"pages = \[([^\]]*)\]", panel)
    assert match, "could not find the nav rail's page list in dem_config_panel.py"
    names = re.findall(r'"([^"]+)"', match.group(1))
    assert names == ["Preview", "Convert", "Inspect"], (
        f"nav rail pages are {names}; expected Preview/Convert/Inspect"
    )

    init = panel[panel.index("def _init_ui"):panel.index("def _create_convert_page")]
    added = init.count("self._stack.addWidget(")
    assert added == len(names), (
        f"the rail lists {len(names)} pages but _init_ui adds {added} widgets "
        f"to the stack; clicks would be routed to the wrong page"
    )


def test_the_runner_prefers_the_in_repo_build_directory():
    """build/bin must be searched before the legacy standalone dtm/ directory.

    The legacy binaries predate the port and can be built for another CPU
    architecture; finding one first means the GUI runs code that is not in this
    checkout.
    """
    text = RUNNER_PY.read_text(encoding="utf-8", errors="replace")
    block = text[text.index("def _find_executable"):text.index("def _missing_executable_message")]
    build_at = block.index('"build"')
    legacy_at = block.index('"dtm"')
    assert build_at < legacy_at, (
        "_find_executable searches the legacy dtm/ directory before "
        "build/bin/; the in-repo build must win"
    )


def test_the_missing_binary_message_names_the_build_target_and_gdal():
    text = RUNNER_PY.read_text(encoding="utf-8", errors="replace")
    block = text[text.index("def _missing_executable_message"):text.index("def _require_posix_pipes")]
    for needle in ("cmake --build", "GDAL", "INSTALL.md"):
        assert needle in block, (
            f"the missing-executable message must mention {needle!r}; "
            f"'cd dtm && make' is not a build path in this repository"
        )


# ---------------------------------------------------------------------------
# Synthetic DTM + PLY reader
# ---------------------------------------------------------------------------

def _write_synthetic_dtm(path: Path) -> None:
    """A south-polar-stereographic lunar GeoTIFF: sloping plain + bowl crater."""
    try:
        from osgeo import gdal, osr
    except ImportError as exc:
        raise Skip(
            "GDAL Python bindings (osgeo) not importable, so the synthetic DTM "
            f"cannot be written: {exc}. Install with "
            "'conda install -c conda-forge gdal'."
        )
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover
        raise Skip(f"numpy not importable: {exc}")

    n, px = _DTM_N, _DTM_PX_M
    gt = (-n / 2 * px, px, 0.0, n / 2 * px, 0.0, -px)

    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    r = np.hypot(xx - n / 2.0, yy - n / 2.0) * px
    z = 0.02 * (xx * px) + 50.0 * np.sin(2 * np.pi * xx / 97.0) * np.cos(
        2 * np.pi * yy / 89.0
    )
    z = z + np.where(r < 1500.0, -300.0 * (1.0 - (r / 1500.0) ** 2), 0.0)
    z = z.astype(np.float32)

    # The CRS is not optional here. Without it the converter falls back to
    # treating projected meters as degrees, and every geometric check below
    # would silently measure the WRONG code path -- the polar-stereographic
    # inverse is never entered, and GDAL accepts SetProjection('') without
    # complaint.
    #
    # The WKT is a LITERAL rather than built with ImportFromProj4, because that
    # call needs a working proj.db and returns 5 with an empty WKT on any host
    # whose PROJ database does not match its PROJ library -- a common state, and
    # one that produced a projection-less fixture that then blamed the tool.
    # A literal needs no database. This is byte-for-byte what GDAL 3.4.1 wrote
    # from the equivalent +proj=stere definition.
    srs = osr.SpatialReference()
    rc = srs.ImportFromWkt(_POLAR_STEREO_WKT)
    wkt = srs.ExportToWkt() if rc == 0 else ""
    if rc != 0 or "Polar_Stereographic" not in wkt:
        raise Skip(
            f"GDAL could not parse the south-polar-stereographic WKT "
            f"(ImportFromWkt returned {rc}, round-tripped WKT is {len(wkt)} "
            f"chars). Without a CRS the converter takes the unprojected "
            f"fallback path, so these checks would measure the wrong code."
        )

    ds = gdal.GetDriverByName("GTiff").Create(str(path), n, n, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    ds.SetProjection(wkt)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(z)
    ds.FlushCache()
    del ds

    # Read it back: SetProjection() does not report failure either.
    check = gdal.Open(str(path))
    written = check.GetProjection() or ""
    del check
    if "Polar_Stereographic" not in written:
        raise Skip(
            f"the synthetic GeoTIFF at {path} came back without a polar "
            f"stereographic CRS ({len(written)} chars of WKT); GDAL accepted "
            f"the projection but did not store it."
        )


class Mesh:
    """The subset of a dtm_to_ply_delaunay PLY these checks need."""

    __slots__ = ("n_vertices", "n_faces", "lat", "lon", "elev", "radius",
                 "face_indices", "face_counts")


def _read_ply(path: Path) -> Mesh:
    raw = path.read_bytes()
    marker = b"end_header\n"
    at = raw.find(marker)
    if at < 0:
        raise AssertionError(f"{path} has no PLY header terminator")
    header = raw[:at].decode("ascii", "replace")
    body_at = at + len(marker)

    def count(kind: str) -> int:
        for line in header.splitlines():
            if line.startswith(f"element {kind} "):
                return int(line.split()[-1])
        raise AssertionError(f"{path} header has no 'element {kind}'")

    m = Mesh()
    m.n_vertices = count("vertex")
    m.n_faces = count("face")

    # Vertex record: 3x float64 xyz, 3x float32 normal, 3x float64 lat/lon/elev
    vsize = 3 * 8 + 3 * 4 + 3 * 8
    m.lat, m.lon, m.elev, m.radius = [], [], [], []
    for i in range(m.n_vertices):
        off = body_at + i * vsize
        x, y, z = struct.unpack_from("<3d", raw, off)
        lat, lon, elev = struct.unpack_from("<3d", raw, off + 3 * 8 + 3 * 4)
        m.lat.append(lat)
        m.lon.append(lon)
        m.elev.append(elev)
        m.radius.append(math.sqrt(x * x + y * y + z * z))

    # Face record: uchar count + 3x int32
    foff = body_at + m.n_vertices * vsize
    m.face_indices, m.face_counts = [], []
    for i in range(m.n_faces):
        off = foff + i * (1 + 12)
        (cnt,) = struct.unpack_from("<B", raw, off)
        a, b, c = struct.unpack_from("<3i", raw, off + 1)
        m.face_counts.append(cnt)
        m.face_indices.append((a, b, c))

    expected = foff + m.n_faces * 13
    assert len(raw) == expected, (
        f"{path}: file is {len(raw)} bytes, header implies {expected} "
        f"({m.n_vertices} vertices, {m.n_faces} faces). A length check is the "
        f"only thing that catches a record-size mistake -- a wrong stride still "
        f"decodes to plausible numbers."
    )
    return m


def _geodesic_km(lat1, lon1, lat2, lon2, radius_m=MOON_RADIUS_M):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2.0 * radius_m * math.asin(math.sqrt(a)) / 1000.0


def _run_convert(args, out_name):
    if not CONVERT_BIN.is_file() or not os.access(CONVERT_BIN, os.X_OK):
        raise Skip(
            f"missing binary: {CONVERT_BIN}. Build it with "
            f"'cmake --build build --target dtm_to_ply_delaunay -j' "
            f"(needs GDAL at configure time; see INSTALL.md 'DEM / DTM tools')."
        )
    work = _scratch()
    dtm = work / "synthetic_dtm.tif"
    if not dtm.is_file():
        _write_synthetic_dtm(dtm)
    out = work / out_name
    if out.exists():
        out.unlink()
    proc = subprocess.run(
        [str(CONVERT_BIN), str(dtm), str(out)] + [str(a) for a in args],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"dtm_to_ply_delaunay exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )
    assert out.is_file(), f"exit 0 but no output at {out}"
    return out, proc


def test_a_whole_dtm_converts_to_a_sane_mesh():
    out, proc = _run_convert(
        ["--downsample", 2, "--body-radius", MOON_RADIUS_M], "full.ply"
    )
    # Prove the projected code path was taken. If the CRS were missing the
    # tool would treat projected meters as degrees and every check below would
    # be measuring the unprojected fallback instead.
    assert "Detected POLAR STEREOGRAPHIC projection" in proc.stdout, (
        "the converter did not recognize the fixture's CRS, so this ran the "
        "unprojected fallback path:\n" + proc.stdout[:1500]
    )
    m = _read_ply(out)

    assert m.n_vertices > 1000, f"only {m.n_vertices} vertices from a 400x400 DTM"
    assert m.n_faces > m.n_vertices, (
        f"{m.n_faces} faces for {m.n_vertices} vertices; a Delaunay "
        f"triangulation of an interior-dominated point set has roughly 2N"
    )
    assert set(m.face_counts) == {3}, "PLY must contain triangles only"

    lo = min(min(f) for f in m.face_indices)
    hi = max(max(f) for f in m.face_indices)
    assert lo >= 0 and hi < m.n_vertices, (
        f"face indices span [{lo}, {hi}] against {m.n_vertices} vertices"
    )

    # Vertices sit at body_radius + elevation, so the radius spread must match
    # the elevation spread -- catches a unit or datum slip.
    r_span = max(m.radius) - min(m.radius)
    e_span = max(m.elev) - min(m.elev)
    assert abs(r_span - e_span) < 1.0, (
        f"radius spread {r_span:.2f} m does not match elevation spread "
        f"{e_span:.2f} m; body radius or units are wrong"
    )
    assert MOON_RADIUS_M - 2000 < min(m.radius) < MOON_RADIUS_M + 2000, (
        f"vertex radii near {min(m.radius):.0f} m, expected ~{MOON_RADIUS_M:.0f}"
    )
    assert all(lat <= -89.0 for lat in m.lat), (
        "a south-polar DTM must not produce vertices outside the polar cap"
    )


def test_roi_flags_actually_restrict_the_extent_and_grade_the_density():
    full, _ = _run_convert(
        ["--downsample", 2, "--body-radius", MOON_RADIUS_M], "full.ply"
    )
    roi, _ = _run_convert(
        [
            "--downsample", 1,
            "--body-radius", MOON_RADIUS_M,
            "--roi-center", _ROI_LAT, _ROI_LON,
            "--roi-radius", _ROI_RADIUS_KM,
            "--roi-max-points", _ROI_MAX_POINTS,
            "--roi-core-frac", _ROI_CORE_FRAC,
        ],
        "roi.ply",
    )
    mf = _read_ply(full)
    mr = _read_ply(roi)

    dist = [_geodesic_km(_ROI_LAT, _ROI_LON, la, lo) for la, lo in zip(mr.lat, mr.lon)]
    assert max(dist) <= _ROI_RADIUS_KM + 1e-6, (
        f"--roi-radius {_ROI_RADIUS_KM} km did not bound the mesh: farthest "
        f"vertex is {max(dist):.4f} km out"
    )

    # The cutoff has to actually cut: the full mesh must reach past it.
    dist_full = [_geodesic_km(_ROI_LAT, _ROI_LON, la, lo)
                 for la, lo in zip(mf.lat, mf.lon)]
    assert max(dist_full) > _ROI_RADIUS_KM * 1.5, (
        "the unrestricted mesh does not extend past the ROI radius, so this "
        "test could not tell a working ROI from a no-op"
    )

    assert mr.n_vertices <= _ROI_MAX_POINTS, (
        f"--roi-max-points {_ROI_MAX_POINTS} exceeded: {mr.n_vertices} vertices"
    )

    # Density grading: --roi-core-frac 0.3 of 2 km protects the inner 0.6 km at
    # full resolution, and --density-exp (default 2.0) thins outward from there.
    core_r = _ROI_RADIUS_KM * _ROI_CORE_FRAC
    rim_lo, rim_hi = _ROI_RADIUS_KM * 0.75, _ROI_RADIUS_KM
    core_n = sum(1 for d in dist if d < core_r)
    rim_n = sum(1 for d in dist if rim_lo <= d < rim_hi)
    core_density = core_n / (math.pi * core_r ** 2)
    rim_density = rim_n / (math.pi * (rim_hi ** 2 - rim_lo ** 2))
    assert core_n > 0, "the protected core is empty"
    assert core_density > 10.0 * max(rim_density, 1e-9), (
        f"density is not graded: core {core_density:.1f} verts/km^2 vs rim "
        f"{rim_density:.1f} verts/km^2. --roi-core-frac / --density-exp are "
        f"the substitute for the unimplemented refiner, so they must bite."
    )


def test_the_same_command_twice_gives_the_same_mesh():
    """Thinning is probabilistic but seeded (rng_(42)); reruns must match."""
    args = [
        "--downsample", 1,
        "--body-radius", MOON_RADIUS_M,
        "--roi-center", _ROI_LAT, _ROI_LON,
        "--roi-radius", _ROI_RADIUS_KM,
        "--roi-max-points", _ROI_MAX_POINTS,
        "--roi-core-frac", _ROI_CORE_FRAC,
    ]
    a, _ = _run_convert(args, "repeat_a.ply")
    b, _ = _run_convert(args, "repeat_b.ply")
    assert a.read_bytes() == b.read_bytes(), (
        "two identical invocations produced different meshes; the fixed RNG "
        "seed is what makes DEM-derived meshes reproducible"
    )


def test_conversion_refuses_instead_of_writing_an_empty_mesh():
    """An ROI that misses the raster must fail, not produce a 360-byte PLY.

    A wrong latitude sign on a polar DEM is the obvious way in. The tool used
    to sample zero points, triangulate nothing, write a 0-vertex / 0-face PLY,
    print "Success!" and exit 0 -- and the GUI, which keys on the exit code,
    would load that empty mesh and emit it downstream as a converted result.
    """
    if not CONVERT_BIN.is_file() or not os.access(CONVERT_BIN, os.X_OK):
        raise Skip(
            f"missing binary: {CONVERT_BIN}. Build it with "
            f"'cmake --build build --target dtm_to_ply_delaunay -j'."
        )
    work = _scratch()
    dtm = work / "synthetic_dtm.tif"
    if not dtm.is_file():
        _write_synthetic_dtm(dtm)
    out = work / "should_not_exist.ply"
    if out.exists():
        out.unlink()

    proc = subprocess.run(
        [
            str(CONVERT_BIN), str(dtm), str(out),
            "--body-radius", str(MOON_RADIUS_M),
            # NORTHERN latitude against a south-polar raster: no points at all.
            "--roi-center", "45.0", "30.0",
            "--roi-radius", "1.0",
        ],
        capture_output=True, text=True, timeout=600,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "dtm_to_ply_delaunay exited 0 on an ROI that selects no points.\n"
        + combined[-1500:]
    )
    assert not out.exists(), (
        f"an empty mesh was written to {out}; callers cannot distinguish it "
        f"from a real one without opening it"
    )
    assert "empty" in combined.lower(), (
        "the refusal must say the mesh was empty:\n" + combined[-1500:]
    )
    assert "nan" not in combined.lower(), (
        "the sliver statistic divided by zero faces:\n" + combined[-1500:]
    )


TESTS = [
    # 1. flag coverage and wiring -- no binaries, no GDAL, no Qt
    test_every_flag_the_gui_emits_is_implemented_by_the_converter,
    test_the_dead_flag_still_announces_itself,
    test_the_unimplemented_refiner_stays_removed,
    test_the_nav_rail_still_offers_the_three_surviving_pages,
    test_the_runner_prefers_the_in_repo_build_directory,
    test_the_missing_binary_message_names_the_build_target_and_gdal,
    # 2. synthetic-DTM smoke -- needs build/bin + osgeo
    test_a_whole_dtm_converts_to_a_sane_mesh,
    test_roi_flags_actually_restrict_the_extent_and_grade_the_density,
    test_the_same_command_twice_gives_the_same_mesh,
    # 3. a request that cannot be honored must fail, never write an empty mesh
    test_conversion_refuses_instead_of_writing_an_empty_mesh,
]


def _pytest_wrap():
    """Turn Skip into a real pytest skip, so it is reported and never a pass."""
    try:
        import pytest
    except ImportError:
        return
    def make(fn):
        def wrapper():
            try:
                fn()
            except Skip as exc:
                pytest.skip(str(exc))
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    for fn in TESTS:
        globals()[fn.__name__] = make(fn)


_pytest_wrap()


def main():
    failures = skipped = 0
    # TESTS holds the unwrapped originals (it is built before _pytest_wrap
    # rebinds the module globals), so Skip arrives here as an exception.
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
    print(f"\n[test_dem_tools] {total - failures - skipped}/{total} passed, "
          f"{skipped} skipped")
    if _SCRATCH:
        shutil.rmtree(_SCRATCH, ignore_errors=True)
    if skipped == total:
        print("[test_dem_tools] SKIPPED ENTIRELY -- proved NOTHING here.")
    if skipped:
        print(f"[test_dem_tools] {skipped} check(s) could not run -- see the "
              f"SKIP lines above; they are not passes.")
    if failures:
        print("[test_dem_tools] FAIL")
        return 1
    print("[test_dem_tools] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
