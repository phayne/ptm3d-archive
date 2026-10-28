#!/usr/bin/env python3
"""``psr_mask.tif`` must not answer "not permanently shadowed" for a facet that
has no value.

⛔⛔ THE DEFECT THIS EXISTS TO STOP -- MEASURED, NOT INFERRED.

``scripts/ptm3d_stats_to_cog.py`` derived the permanent-shadow layer with

    psr = dmax < args.psr_threshold

over ``direct_solar_max``. In IEEE arithmetic ``NaN < 1.0`` is ``False``, so a
facet carrying NO value -- never evaluated, or a zero-contribution face in a
merged product -- fell out of that comparison as ``False`` and was written to
the product as **0 = not permanently shadowed**. That inverts the exact
quantity the layer exists to report, silently, for the population a PSR map is
read for. An unresolved facet is neither class; it is no data.

Section 1 is the falsifying arm. It builds a synthetic polar patch with a known
permanently-shadowed block, poisons a known subset of that block with ``NaN``,
runs the real converter, and reads the raster back with **rasterio** -- a
different GDAL build from the ``osgeo.gdal`` the writer uses. On the unfixed
converter those pixels come back holding a finite value below 0.5, i.e. "lit".
They must come back as nodata.

Section 2 is the do-no-harm arm: with no non-finite face the two classes are
still 1 and 0 and the raster carries no nodata over the patch, so Section 1
cannot be passing because the layer went blank.

Section 3 checks the nodata value itself cannot be read as either class by a
consumer that ignores the nodata declaration, under BOTH nodata conventions the
converter offers.

Section 4 is the IDW arm. ``--interpolation idw`` drops a non-finite facet
instead of spreading it, so the NaN alone is NOT enough: the pixel is refilled
from the resolved facets around it and the class comes straight back. Measured
on this file's own fixture before the stamp was added: 100 of 100 unresolved
facets held 0.0 while the converter's console line said "100 no data".

Section 5 is the unit arm over the classification itself, including the two
infinities. It runs on numpy alone -- no rasterio, no GDAL -- so the rule stays
enforceable where the raster stack is not installed.

Section 6 covers the two ways the answer could be preserved or discarded
without anyone noticing: ``--resume`` over a mask the old rule wrote, and a
non-finite ``--psr-threshold``.

⛔ Every fixture here is synthetic. No site identifier and no real site
coordinate appears in this file, under the 2026-09-03 standing ruling.

Run directly::

    python3 \
        src/python/tests/test_psr_mask_nodata.py
"""

from __future__ import annotations

import importlib.util
import math
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

CONVERTER = REPO_ROOT / "scripts" / "ptm3d_stats_to_cog.py"

#: Lunar radius, in meters. A body constant, not a site. Passed to the
#: converter explicitly so the fixture never depends on body auto-detection.
BODY_RADIUS_M = 1737400.0

#: Where the synthetic patch sits. INVENTED round numbers, south of the 70 deg
#: polar threshold so the converter takes its polar stereographic branch, and
#: away from the pole itself so the meridian is a real number rather than a
#: wrap artifact. Under the 2026-09-03 standing ruling no real site coordinate
#: may appear in the tracked tree.
FIXTURE_LAT_DEG = -80.0
FIXTURE_LON_DEG = 20.0

#: Grid side of the synthetic patch, its half-width in meters, and the pixel
#: size the converter is driven at. The pixel is an order of magnitude finer
#: than the face spacing (~51 m), so the pixel over a facet centroid is
#: dominated by that facet and the barycentric blend with its neighbors cannot
#: be what decides the assertion.
PATCH_N = 60
PATCH_HALF_M = 1500.0
PIXEL_M = 5.0

#: Flux written on a lit facet and on a permanently shadowed one, in W/m^2,
#: either side of the converter's default 1.0 threshold by a wide margin.
LIT_W_M2 = 500.0
SHADOWED_W_M2 = 0.0

#: How many of the shadowed facets are poisoned with the sentinel.
N_POISONED = 100


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


FAILURES: list = []
SKIPS: list = []
PASSES: list = []


def check(name, fn, *args):
    try:
        detail = fn(*args)
    except Skip as exc:
        SKIPS.append(f"{name}: {exc}")
        print(f"SKIP {name} :: {exc}", flush=True)
    except AssertionError as exc:
        FAILURES.append(f"{name}: {exc}")
        print(f"FAIL {name} :: {exc}", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name} :: {type(exc).__name__}: {exc}", flush=True)
    else:
        PASSES.append(name)
        print(f"PASS {name} :: {detail or ''}", flush=True)


def _need(module):
    try:
        return __import__(module)
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"{module} is not importable here ({exc})")


def _rule():
    """``ptm_cog.psr_mask_values``, on numpy alone.

    Imported from the module rather than from the converter script: the script
    exits 2 at import without rasterio, which would have made the one check
    here that needs nothing installed skip everywhere it matters most.
    """
    _need("numpy")
    # ⛔ NOT a Skip. This arm is the one that runs where the raster stack is
    # not installed -- CI -- so a Skip here would leave that step green having
    # enforced nothing, which is precisely how the hygiene gate went quiet in
    # 2026-09. numpy is the only thing that could legitimately be missing.
    try:
        import ptm_cog
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"ptm_cog could not be imported even though numpy is present, so "
            f"the PSR rule cannot be checked at all: {type(exc).__name__}: "
            f"{exc}")
    fn = getattr(ptm_cog, "psr_mask_values", None)
    assert fn is not None, (
        "ptm_cog has no psr_mask_values(); the PSR rule is still a bare "
        "comparison somewhere and an unresolved facet is still forced into "
        "one of the two classes")
    return fn


def _converter():
    """The real converter module, imported from its path.

    Imported rather than shelled out to, so the check drives the same code the
    campaign tooling drives and a traceback points at a line.
    """
    _need("rasterio")
    if not CONVERTER.is_file():
        raise Skip(f"{CONVERTER} does not exist in this checkout")
    spec = importlib.util.spec_from_file_location(
        "ptm3d_stats_to_cog_under_test", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as exc:      # the module exits 2 without rasterio
        raise Skip(f"the converter refused to import here (exit {exc.code})")
    return mod


# ---------------------------------------------------------------------------
# The synthetic run
# ---------------------------------------------------------------------------

def _patch(root, poison=True):
    """A results directory holding one statistics map on a polar patch.

    ``direct_solar_max`` is LIT_W_M2 everywhere except a contiguous central
    block at SHADOWED_W_M2 -- the permanently shadowed population. When
    ``poison`` is set, N_POISONED facets INSIDE that block are overwritten with
    ``NaN``: facets that are known to be shadowed and that the product must
    nonetheless decline to classify, because their value is absent.

    Returns ``(mesh_path, stats_dir, lat, lon, shadowed_idx, poisoned_idx,
    interior_lit_idx)`` -- the last being facets that are lit, far from
    the shadowed block AND far from the patch rim, so neither the
    barycentric blend across the class boundary nor the edge of the mesh
    can be what a do-no-harm assertion about them is measuring.
    """
    np = _need("numpy")
    root = Path(root)
    (root / "statistics").mkdir(parents=True, exist_ok=True)

    u = np.linspace(-PATCH_HALF_M, PATCH_HALF_M, PATCH_N)
    X, Y = np.meshgrid(u, u)
    elev = 120.0 * np.sin(X / 700.0) * np.cos(Y / 900.0)

    # Local east/north/up at the synthetic center, rotated into the
    # body-centered frame, so the patch is a real piece of a sphere and its
    # latitudes and longitudes form a narrow band as a site mesh's do.
    lat0 = math.radians(FIXTURE_LAT_DEG)
    lon0 = math.radians(FIXTURE_LON_DEG)
    east = np.array([-math.sin(lon0), math.cos(lon0), 0.0])
    north = np.array([-math.sin(lat0) * math.cos(lon0),
                      -math.sin(lat0) * math.sin(lon0), math.cos(lat0)])
    up = np.array([math.cos(lat0) * math.cos(lon0),
                   math.cos(lat0) * math.sin(lon0), math.sin(lat0)])
    verts = (up * BODY_RADIUS_M
             + X.ravel()[:, None] * east
             + Y.ravel()[:, None] * north
             + elev.ravel()[:, None] * up)

    tris = []
    for j in range(PATCH_N - 1):
        for i in range(PATCH_N - 1):
            a = j * PATCH_N + i
            tris.append((a, a + PATCH_N, a + 1))
            tris.append((a + 1, a + PATCH_N, a + PATCH_N + 1))
    tris = np.asarray(tris, dtype=np.int64)
    nf = len(tris)

    cen = verts[tris].mean(axis=1)
    rad = np.linalg.norm(cen, axis=1)
    lat = np.degrees(np.arcsin(cen[:, 2] / rad))
    lon = np.degrees(np.arctan2(cen[:, 1], cen[:, 0]))

    # The shadowed block: facets whose centroid sits inside a square well away
    # from the patch boundary, so every poisoned facet is surrounded by other
    # shadowed facets and the barycentric neighborhood of its pixel is entirely
    # inside the block.
    local_e = (cen - up * BODY_RADIUS_M) @ east
    local_n = (cen - up * BODY_RADIUS_M) @ north
    block = (np.abs(local_e) < 700.0) & (np.abs(local_n) < 700.0)
    shadowed_idx = np.flatnonzero(block)
    assert len(shadowed_idx) > 4 * N_POISONED, (
        f"the fixture's shadowed block holds only {len(shadowed_idx)} facets")

    dmax = np.full(nf, LIT_W_M2, dtype="<f4")
    dmax[shadowed_idx] = SHADOWED_W_M2

    # Poison from the INTERIOR of the block (every 7th, starting well inside),
    # not its rim, so an unresolved pixel can never be explained by the
    # block boundary.
    interior = shadowed_idx[(np.abs(local_e[shadowed_idx]) < 450.0)
                            & (np.abs(local_n[shadowed_idx]) < 450.0)]
    poisoned_idx = interior[::max(1, len(interior) // N_POISONED)][:N_POISONED]
    if poison:
        dmax[poisoned_idx] = np.nan
    else:
        poisoned_idx = poisoned_idx[:0]

    with open(root / "statistics" / "direct_solar_max.bin", "wb") as fh:
        fh.write(struct.pack("<I", nf))
        fh.write(dmax.astype("<f4").tobytes())

    mesh_path = root / "mesh.ply"
    with open(mesh_path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(verts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write(f"element face {nf}\n")
        fh.write("property list uchar int vertex_indices\n")
        fh.write("end_header\n")
        for v in verts:
            fh.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for t in tris:
            fh.write(f"3 {t[0]} {t[1]} {t[2]}\n")

    lit_interior = np.flatnonzero(
        (np.maximum(np.abs(local_e), np.abs(local_n)) > 950.0)
        & (np.maximum(np.abs(local_e), np.abs(local_n)) < 1250.0))

    return (mesh_path, root / "statistics", lat, lon, shadowed_idx,
            poisoned_idx, lit_interior)


def _convert(mod, mesh_path, stats_dir, out_dir, extra=()):
    """Drive the converter's own CLI entry point."""
    argv = sys.argv
    sys.argv = ["ptm3d_stats_to_cog.py",
                "--mesh", str(mesh_path),
                "--stats", str(stats_dir),
                "--output", str(out_dir),
                "--body-radius", repr(BODY_RADIUS_M),
                "--resolution-m", repr(PIXEL_M),
                *extra]
    try:
        mod.main()
    except SystemExit as exc:
        if exc.code:
            # `sys.exit("message")` carries the message as the code, and the
            # message is what says WHICH refusal this was.
            raise AssertionError(f"the converter exited {exc.code}")
    finally:
        sys.argv = argv
    out = Path(out_dir) / "psr_mask.tif"
    assert out.is_file(), f"the converter wrote no {out}"
    return out


def _forward_south_polar_stereo(lat_deg, lon_deg, center_lon_deg, radius):
    """Forward south-polar stereographic, written out here on purpose.

    NOT ``ptm.results._forward_polar_stereo``: that is the code the raster was
    built with, and checking a projection against itself proves nothing.
    """
    rho = 2.0 * radius * math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0)
    dlon = math.radians(lon_deg - center_lon_deg)
    return rho * math.sin(dlon), rho * math.cos(dlon)


def _pixels_at(ds, lat, lon, idx):
    """(row, col) of each facet in ``idx``, from its own coordinates."""
    from rasterio.transform import rowcol
    out = []
    for f in idx:
        x, y = _forward_south_polar_stereo(float(lat[f]), float(lon[f]),
                                           0.0, BODY_RADIUS_M)
        r, c = rowcol(ds.transform, x, y)
        assert 0 <= r < ds.height and 0 <= c < ds.width, (
            f"facet {f} projects to ({r}, {c}), outside the "
            f"{ds.height}x{ds.width} raster")
        out.append((int(r), int(c)))
    return out


# ---------------------------------------------------------------------------
# Section 1 -- the falsifying arm
# ---------------------------------------------------------------------------

def check_an_unresolved_facet_is_not_stamped_lit(tmp):
    """A facet with no direct_solar_max must not read "not PSR"."""
    np = _need("numpy")
    rasterio = _need("rasterio")
    _need("scipy")
    mod = _converter()

    mesh_path, stats_dir, lat, lon, shadowed, poisoned, _lit = _patch(
        Path(tmp) / "poisoned")
    assert len(poisoned) == N_POISONED, (
        f"the fixture poisoned {len(poisoned)} facets, not {N_POISONED}")
    out = _convert(mod, mesh_path, stats_dir, Path(tmp) / "poisoned_cog")

    with rasterio.open(out) as ds:
        pix = _pixels_at(ds, lat, lon, poisoned)
        band = ds.read(1, masked=True)
        nodata = ds.nodata

    lit = [(r, c) for (r, c) in pix
           if not np.ma.is_masked(band[r, c]) and float(band[r, c]) < 0.5]
    assert not lit, (
        f"{len(lit)} of {len(pix)} pixels over facets with NO "
        f"direct_solar_max value read as NOT permanently shadowed "
        f"(first: {lit[0]} = "
        f"{float(band[lit[0][0]][lit[0][1]]):.4f}); an absent value was "
        f"published as a measurement of sunlight")
    masked = sum(1 for (r, c) in pix if np.ma.is_masked(band[r, c]))
    assert masked == len(pix), (
        f"only {masked} of {len(pix)} unresolved facets came back as nodata")
    return (f"{len(pix)}/{len(pix)} unresolved facets are nodata "
            f"({nodata:.7e}); 0 read as lit")


# ---------------------------------------------------------------------------
# Section 2 -- do no harm
# ---------------------------------------------------------------------------

def check_a_resolved_patch_still_carries_both_classes(tmp):
    """With nothing non-finite, the mask is still a mask.

    Without this, Section 1 would also pass on a converter that wrote an empty
    raster, or that stamped the whole shadowed block out.
    """
    np = _need("numpy")
    rasterio = _need("rasterio")
    _need("scipy")
    mod = _converter()

    mesh_path, stats_dir, lat, lon, shadowed, poisoned, lit = _patch(
        Path(tmp) / "clean", poison=False)
    assert len(poisoned) == 0
    out = _convert(mod, mesh_path, stats_dir, Path(tmp) / "clean_cog")

    with rasterio.open(out) as ds:
        pix = _pixels_at(ds, lat, lon, shadowed[::13])
        band = ds.read(1, masked=True)

    bad = [(r, c) for (r, c) in pix
           if np.ma.is_masked(band[r, c]) or float(band[r, c]) < 0.5]
    assert not bad, (
        f"{len(bad)} of {len(pix)} genuinely shadowed facets did not read as "
        f"permanently shadowed on a source with no non-finite face")
    # And the lit population is still 0, not blank.
    lit_faces = lit[::37]
    with rasterio.open(out) as ds:
        lpix = _pixels_at(ds, lat, lon, lit_faces)
        band = ds.read(1, masked=True)
    blank = [(r, c) for (r, c) in lpix
             if np.ma.is_masked(band[r, c]) or float(band[r, c]) > 0.5]
    assert not blank, (
        f"{len(blank)} of {len(lpix)} lit facets did not read as not "
        f"permanently shadowed")
    return (f"{len(pix)} shadowed facets read 1, {len(lpix)} lit facets read "
            f"0, none blank")


# ---------------------------------------------------------------------------
# Section 3 -- the nodata value cannot be read as a class
# ---------------------------------------------------------------------------

def check_nodata_cannot_be_mistaken_for_either_class(tmp):
    """A reader that ignores the nodata declaration must not silently receive
    "PSR" or "not PSR"."""
    np = _need("numpy")
    rasterio = _need("rasterio")
    _need("scipy")
    mod = _converter()

    mesh_path, stats_dir, lat, lon, shadowed, poisoned, _lit = _patch(
        Path(tmp) / "raw")
    seen = []
    for tag, extra in (("default", ()),
                       ("nan", ("--nodata-value", "nan"))):
        out = _convert(mod, mesh_path, stats_dir,
                       Path(tmp) / f"raw_cog_{tag}", extra=extra)
        with rasterio.open(out) as ds:
            pix = _pixels_at(ds, lat, lon, poisoned)
            raw = ds.read(1)          # NOT masked: the careless reader
            declared = ds.nodata
        vals = np.array([raw[r, c] for (r, c) in pix], dtype=np.float64)
        assert declared is not None, (
            f"[{tag}] the raster declares no nodata at all")
        confusable = np.isfinite(vals) & ((np.abs(vals) < 0.5)
                                          | (np.abs(vals - 1.0) < 0.5))
        assert not confusable.any(), (
            f"[{tag}] {int(confusable.sum())} unresolved pixels hold a raw "
            f"value a 0/1 consumer would read as one of the two classes "
            f"(e.g. {vals[confusable][0]!r})")
        seen.append(f"{tag}: raw={vals[0]!r} declared={declared!r}")
    return "; ".join(seen)


# ---------------------------------------------------------------------------
# Section 4 -- the inverse-distance path
# ---------------------------------------------------------------------------

def check_idw_cannot_refill_an_unresolved_pixel(tmp):
    """`--interpolation idw` must not put the class back.

    ⛔ The NaN alone does not survive this path. `MeshRegridder.regrid` zeroes
    the weight of every non-finite neighbor, so the unresolved facet is DROPPED
    and its pixel is filled from the resolved facets around it. On this exact
    fixture, before `export_field` was given the unresolved population to stamp
    with, 100 of 100 unresolved facets came back holding 0.0 while the
    converter's own console line said "100 no data".
    """
    np = _need("numpy")
    rasterio = _need("rasterio")
    _need("scipy")
    mod = _converter()

    mesh_path, stats_dir, lat, lon, shadowed, poisoned, _lit = _patch(
        Path(tmp) / "idw")
    _relight(stats_dir, len(lat), poisoned)
    out = _convert(mod, mesh_path, stats_dir, Path(tmp) / "idw_cog",
                   extra=("--interpolation", "idw"))
    with rasterio.open(out) as ds:
        pix = _pixels_at(ds, lat, lon, poisoned)
        band = ds.read(1, masked=True)
    classed = [(r, c) for (r, c) in pix if not np.ma.is_masked(band[r, c])]
    assert not classed, (
        f"{len(classed)} of {len(pix)} unresolved facets were refilled with a "
        f"class by the inverse-distance path (first: {classed[0]} = "
        f"{float(band[classed[0][0]][classed[0][1]]):.4f})")
    return f"{len(pix)}/{len(pix)} unresolved facets stayed nodata under idw"


# ---------------------------------------------------------------------------
# Section 5 -- the classification itself
# ---------------------------------------------------------------------------

def check_the_classifier_refuses_every_non_finite_input():
    """Unit arm. Both infinities are no data too: neither is a flux."""
    np = _need("numpy")
    fn = _rule()
    src = np.array([0.0, 0.5, 1.0, 2.0, np.nan, np.inf, -np.inf])
    got = fn(src, 1.0)
    assert list(got[:4]) == [1.0, 1.0, 0.0, 0.0], (
        f"the finite classification changed: {list(got[:4])}")
    assert bool(np.isnan(got[4])), "NaN was classified rather than refused"
    assert bool(np.isnan(got[5])), "+inf was classified rather than refused"
    assert bool(np.isnan(got[6])), "-inf was classified rather than refused"
    return f"{got.tolist()} for {src.tolist()} at threshold 1.0"


def _relight(stats_dir, n_faces, poisoned):
    """Make the whole patch LIT except the poisoned facets.

    The poisoned facets' resolved NEIGHBORS then all say "not permanently
    shadowed", so an interpolation that refills their pixels from those
    neighbors reinstates the original defect exactly -- 0 where the truth is
    "no value". With the block shadowed instead, the same leak would produce
    1 and could be mistaken for a correct answer.
    """
    np = _need("numpy")
    d = np.full(n_faces, LIT_W_M2, dtype="<f4")
    d[poisoned] = np.nan
    with open(Path(stats_dir) / "direct_solar_max.bin", "wb") as fh:
        fh.write(struct.pack("<I", n_faces))
        fh.write(d.tobytes())


# ---------------------------------------------------------------------------
# Section 6 -- the answer cannot be preserved or discarded quietly
# ---------------------------------------------------------------------------

def check_resume_does_not_keep_a_mask_written_by_the_old_rule(tmp):
    """`--resume` over an existing psr_mask.tif must re-export it when the
    source has facets with no value; a mask the old rule wrote reports them as
    "not permanently shadowed" and would otherwise be kept forever."""
    np = _need("numpy")
    rasterio = _need("rasterio")
    _need("scipy")
    mod = _converter()

    mesh_path, stats_dir, lat, lon, shadowed, poisoned, _lit = _patch(
        Path(tmp) / "resume")
    _relight(stats_dir, len(lat), poisoned)
    out_dir = Path(tmp) / "resume_cog"
    out = _convert(mod, mesh_path, stats_dir, out_dir)

    # Stand in for a mask the OLD rule wrote: every pixel a class, no nodata.
    with rasterio.open(out) as ds:
        profile = ds.profile
        stale = np.zeros((ds.height, ds.width), dtype=np.float32)
    with rasterio.open(out, "w", **profile) as ds:
        ds.write(stale, 1)

    import contextlib
    import io as _io
    log = _io.StringIO()
    with contextlib.redirect_stdout(log):
        out = _convert(mod, mesh_path, stats_dir, out_dir, extra=("--resume",))
    assert "redo  psr_mask" in log.getvalue(), (
        "--resume did not re-export a psr_mask.tif written by the old rule")
    with rasterio.open(out) as ds:
        pix = _pixels_at(ds, lat, lon, poisoned)
        band = ds.read(1, masked=True)
    classed = [(r, c) for (r, c) in pix if not np.ma.is_masked(band[r, c])]
    assert not classed, (
        f"--resume kept a psr_mask.tif that reports {len(classed)} of "
        f"{len(pix)} unresolved facets as a class")

    # ...and only once. A correct mask declares the same unresolved population
    # in PTM_FACES_NAN, so the next resume must SKIP it -- the campaign watcher
    # resumes on every pass, and re-exporting a multi-million-face mask forever
    # would be a defect introduced by the fix.
    log = _io.StringIO()
    with contextlib.redirect_stdout(log):
        _convert(mod, mesh_path, stats_dir, out_dir, extra=("--resume",))
    assert "skip  psr_mask" in log.getvalue(), (
        "--resume re-exported a mask that already declares its unresolved "
        "population; the fix made --resume non-idempotent")
    return (f"--resume re-exported the stale mask ({len(pix)}/{len(pix)} now "
            f"nodata) and skipped the corrected one")


def check_a_non_finite_threshold_or_radius_is_refused(tmp):
    """Every comparison against a non-finite threshold or clip radius is
    False, so either would have written a silent "nothing here is permanently
    shadowed" map from a fully resolved run."""
    _need("numpy")
    _need("rasterio")
    _need("scipy")
    mod = _converter()
    mesh_path, stats_dir, lat, lon, shadowed, poisoned, _lit = _patch(
        Path(tmp) / "thresh", poison=False)
    seen = []
    for flag, value, want in (("--psr-threshold", "nan", "psr-threshold"),
                              ("--psr-clip-radius-km", "nan", "clip-radius"),
                              ("--psr-clip-radius-km", "-1", "clip-radius")):
        out_dir = Path(tmp) / f"thresh_cog_{want}_{value}"
        try:
            _convert(mod, mesh_path, stats_dir, out_dir,
                     extra=(flag, value))
        except AssertionError as exc:
            # ⛔ On the MESSAGE, not merely on "it raised": `_convert` also
            # raises when no psr_mask.tif appears, so accepting any
            # AssertionError would pass a build that took the flag and then
            # failed to write the layer for an unrelated reason.
            assert want in str(exc), (
                f"{flag} {value} was refused, but for the wrong reason: {exc}")
            seen.append(f"{flag} {value}")
            continue
        raise AssertionError(
            f"{flag} {value} was accepted; every comparison against it is "
            f"False, so the converter wrote a mask in which nothing is "
            f"permanently shadowed, from a fully resolved run")
    return "refused: " + ", ".join(seen)


def main():
    tmp = tempfile.mkdtemp(prefix="ptm_psr_nodata_gate_")
    check("test_an_unresolved_facet_is_not_stamped_lit",
          check_an_unresolved_facet_is_not_stamped_lit, tmp)
    check("test_a_resolved_patch_still_carries_both_classes",
          check_a_resolved_patch_still_carries_both_classes, tmp)
    check("test_nodata_cannot_be_mistaken_for_either_class",
          check_nodata_cannot_be_mistaken_for_either_class, tmp)
    check("test_idw_cannot_refill_an_unresolved_pixel",
          check_idw_cannot_refill_an_unresolved_pixel, tmp)
    check("test_the_classifier_refuses_every_non_finite_input",
          check_the_classifier_refuses_every_non_finite_input)
    check("test_resume_does_not_keep_a_mask_written_by_the_old_rule",
          check_resume_does_not_keep_a_mask_written_by_the_old_rule, tmp)
    check("test_a_non_finite_threshold_or_radius_is_refused",
          check_a_non_finite_threshold_or_radius_is_refused, tmp)

    total = len(PASSES) + len(FAILURES) + len(SKIPS)
    print(f"\n[test_psr_mask_nodata] {len(PASSES)}/{total} passed"
          + (f", {len(SKIPS)} skipped" if SKIPS else "")
          + (f", {len(FAILURES)} failed" if FAILURES else ""), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
