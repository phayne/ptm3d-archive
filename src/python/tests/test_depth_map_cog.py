#!/usr/bin/env python3
"""Subsurface temperature maps reach a georeferenced raster, carrying their depth.

The engine writes one per-facet file per depth per statistic, named
``temperature_z<meters, '.' -> 'p'>m_<stat>.bin``, plus a sidecar
``statistics/depth_map_z.json`` recording, for each one, the depth that was
ASKED FOR and the depth that was actually SAMPLED. Those two differ whenever
the run moved a request to the nearest depth it solves on, which is the
default.

Three things have to hold for that to survive into a raster, and each of them
has failed somewhere in this project before:

* **The exporter has to find the files at all.** ``STATISTICS_FIELDS`` is a
  fixed table and cannot hold these -- the name carries the depth, so the set
  belongs to the run -- so the enumeration is by discovery. Without it a run
  that asked for depth maps would export only its surface products and say
  nothing about the rest.

* **The header is uint32.** A per-facet statistics file is a 4-byte
  little-endian UNSIGNED facet count followed by that many float32. Reading it
  as signed, or reading a file whose length does not match its header, does not
  raise -- it lands in a neighboring value and returns numbers. Section 2
  checks the length as well as the count, because a count alone passes a
  truncated file.

* **The depth has to travel with the pixels.** A GeoTIFF outlives the run
  directory it came from. A raster holding 0.298 m, named for a request of
  0.30 m, with nothing on the band saying which is which, is a number that will
  be quoted as the depth requested.

Needs numpy for the arithmetic; ``rasterio``/``osgeo`` and a mesh for the
raster arms, which skip without them. Run directly::

    python3 src/python/tests/test_depth_map_cog.py
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

#: Shipped with the repository, 12,800 facets, and the same mesh
#: ``examples/validation/baseline_crater80N_*.yaml`` runs. No site identifier.
CRATER = REPO_ROOT / "data" / "meshes" / "lunar_crater_80N.ply"

EXPORTER = REPO_ROOT / "scripts" / "ptm3d_stats_to_cog.py"

#: The two depths the fixture writes. The second is the interesting one: it was
#: REQUESTED at 0.30 m and SAMPLED at 0.298 m, so a reader who sees only the
#: file name learns the wrong number by 2 mm.
FIXTURE_DEPTHS = ((0.05, 0.05), (0.30, 0.298))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:                       # pragma: no cover
        raise Skip(f"numpy is unavailable here ({exc})")
    return np


def _write_stat_file(path, values):
    """A per-facet statistics binary, in the engine's own format.

    ``<I`` -- little-endian UNSIGNED 32-bit -- then that many float32. Written
    here rather than borrowed from a reader, so this fixture cannot inherit a
    reader's mistake about the header.
    """
    np = _numpy()
    values = np.asarray(values, dtype="<f4")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<I", values.size))
        handle.write(values.tobytes())


def _fixture(directory, with_manifest=True):
    """A run directory carrying surface products AND two depth maps."""
    np = _numpy()
    if not CRATER.is_file():
        raise Skip(f"{CRATER} is not in this checkout")
    from ptm.results import read_ply_mesh_minimal
    mesh = read_ply_mesh_minimal(CRATER)
    n_faces = mesh["n_faces"]
    stats = Path(directory) / "statistics"
    stats.mkdir(parents=True, exist_ok=True)

    # A smooth, deterministic field so a raster arm can compare numbers.
    lat = np.asarray(mesh["face_latitudes"], dtype=np.float64)
    base = 100.0 + 40.0 * np.cos(np.radians(lat * 4.0))

    for stat, offset in (("avg", 0.0), ("max", 12.0), ("min", -9.0)):
        _write_stat_file(stats / f"temperature_{stat}.bin", base + offset)
        for _requested, resolved in FIXTURE_DEPTHS:
            name = "temperature_z" + ("%.3f" % resolved).replace(".", "p") + "m"
            _write_stat_file(stats / f"{name}_{stat}.bin",
                             base + offset * resolved)

    if with_manifest:
        entries = []
        for requested, resolved in FIXTURE_DEPTHS:
            tag = ("%.3f" % resolved).replace(".", "p")
            entries.append({
                "requested_m": requested,
                "resolved_m": resolved,
                "residual_m": round(resolved - requested, 6),
                "node_j": 3,
                "weight_on_j_plus_1": 0.0,
                "on_node": True,
                "field": f"temperature_z{tag}m",
            })
        with open(stats / "depth_map_z.json", "w") as handle:
            json.dump({"schema": "ptm3d-depth-map/1",
                       "snap_to_node": True,
                       "n_faces": n_faces,
                       "statistics": ["avg", "max", "min"],
                       "depths": entries}, handle, indent=1)
    return stats, n_faces


# ==========================================================================
# 1. Discovery -- the exporter can see the files
# ==========================================================================

def test_a_depth_field_name_is_read_back_to_a_depth():
    from ptm.results import depth_of_field
    for name, meters in (("temperature_z0p000m", 0.0),
                         ("temperature_z0p050m", 0.05),
                         ("temperature_z0p298m", 0.298),
                         ("temperature_z12p346m", 12.346)):
        assert depth_of_field(name) == meters, (
            f"{name} read back as {depth_of_field(name)}, not {meters}")
    for name in ("temperature_avg", "temperature", "ice_depth",
                 "temperature_z0p05m",        # two decimals, not the rule
                 "xtemperature_z0p050m"):     # not anchored
        assert depth_of_field(name) is None, (
            f"{name} was mistaken for a subsurface temperature map")


def test_the_two_python_mirrors_of_the_naming_rule_agree():
    """⛔ THERE ARE TWO, and nothing else compares them.

    The panel writes a depth name FORWARD (``gui_core.depth_maps.field_name``,
    so it can show what a run will produce before the run exists) and the
    raster exporter reads one BACKWARD (``ptm.results.DEPTH_FIELD_RE``, so it
    can label a file it finds). Each is separately anchored to the engine's
    header -- but only in its own suite, and each suite is blind to the other
    module. Measured: changing the panel to two decimals fails two arms in the
    GUI suite and NONE here; changing the pattern here fails eleven here and
    none there. Drift them the same way and both suites stay green while the
    application names files nothing on disk matches.

    Skips rather than fails where the GUI package is unavailable: this arm is
    about two modules agreeing, and it cannot say anything when only one of
    them can be imported.
    """
    gui_root = REPO_ROOT / "gui"
    if not gui_root.is_dir():                        # pragma: no cover
        raise Skip("this checkout has no gui/ directory")
    sys.path.insert(0, str(gui_root))
    try:
        try:
            from gui_core import depth_maps as dm
        except ImportError as exc:
            raise Skip(f"the GUI package is unavailable here ({exc})")
    finally:
        sys.path.pop(0)
    from ptm.results import depth_of_field

    for meters in (0.0, 0.001, 0.05, 0.298, 0.5, 1.25, 12.3456, 99.0):
        name = dm.field_name(meters)
        back = depth_of_field(name)
        assert back is not None, (
            f"the panel would write {name!r} and the raster exporter does not "
            f"recognize it as a subsurface temperature map at all")
        assert abs(back - round(meters, 3)) < 5e-7, (
            f"{meters} m is written as {name!r} and read back as {back}")


def test_the_depth_maps_are_discovered_from_a_run_directory():
    with tempfile.TemporaryDirectory() as scratch:
        stats, _n = _fixture(scratch)
        from ptm.results import discover_depth_fields
        found = discover_depth_fields(stats)
    assert set(found) == {"temperature_z0p050m", "temperature_z0p298m"}, (
        f"discovery found {sorted(found)}")
    assert found["temperature_z0p298m"]["depth_m"] == 0.298
    assert found["temperature_z0p298m"]["units"] == "K"
    assert found["temperature_z0p298m"]["depth_requested_m"] == 0.30, (
        "the manifest's requested depth did not reach the field record, so "
        "nothing downstream can tell a moved depth from a requested one")


def test_discovery_finds_nothing_in_a_run_that_asked_for_no_depths():
    """The control. Every existing run in the archive is this one."""
    with tempfile.TemporaryDirectory() as scratch:
        stats = Path(scratch) / "statistics"
        stats.mkdir()
        _write_stat_file(stats / "temperature_avg.bin", [1.0, 2.0, 3.0])
        from ptm.results import discover_depth_fields, detect_available_statistics
        assert discover_depth_fields(stats) == {}
        available = detect_available_statistics(stats)
    assert available == [("temperature", "avg")], (
        f"a run with no depth maps reported {available}")


def test_the_statistics_listing_includes_the_depth_maps():
    with tempfile.TemporaryDirectory() as scratch:
        stats, _n = _fixture(scratch)
        from ptm.results import detect_available_statistics
        available = detect_available_statistics(stats)
    names = {f"{base}_{stat}" for base, stat in available}
    for depth_tag in ("z0p050m", "z0p298m"):
        for stat in ("avg", "max", "min"):
            field = f"temperature_{depth_tag}_{stat}"
            assert field in names, (
                f"{field} is on disk and was not listed, so the exporter "
                f"would skip it in silence")
    assert "temperature_avg" in names, "the surface products stopped being listed"


def test_a_missing_manifest_costs_the_requested_depth_and_nothing_else():
    """No manifest is not an error, and it must not become an invented depth."""
    with tempfile.TemporaryDirectory() as scratch:
        stats, _n = _fixture(scratch, with_manifest=False)
        from ptm.results import discover_depth_fields, depth_entries
        assert depth_entries(stats) == {}
        found = discover_depth_fields(stats)
    assert set(found) == {"temperature_z0p050m", "temperature_z0p298m"}
    assert found["temperature_z0p298m"]["depth_m"] == 0.298
    assert "depth_requested_m" not in found["temperature_z0p298m"], (
        "a requested depth appeared with no manifest to take it from")


def test_the_exporter_enumerates_the_depth_maps():
    with tempfile.TemporaryDirectory() as scratch:
        stats, _n = _fixture(scratch)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_ptm3d_stats_to_cog", EXPORTER)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except SystemExit as exc:                # rasterio missing
                raise Skip(f"the exporter cannot be imported here ({exc})")
            pairs = dict(module._expected_field_files(stats))
        finally:
            sys.path.pop(0)
    for stat in ("avg", "max", "min"):
        field = f"temperature_z0p298m_{stat}"
        assert pairs.get(field) == f"{field}.bin", (
            f"the exporter does not know about {field}")
    assert "temperature_avg" in pairs, "the fixed catalog was lost"
    assert "ice_depth" in pairs


# ==========================================================================
# 2. The header discipline
# ==========================================================================

def test_the_facet_count_is_read_as_an_unsigned_32_bit_header():
    """A per-facet file is `uint32` then that many float32, little-endian.

    Two separate claims, and this arm makes both rather than one:

    * the readers SPELL the header unsigned. Checked against the source,
      because a header above 2^31 needs an 8 GB file and cannot be built here,
      so no round trip at this scale can tell `<I` from `<i`;
    * a header the readers cannot reconcile with the file's own length is
      refused rather than acted on -- which is what actually happens if the
      spelling is ever changed, and is the behavior a caller depends on.
    """
    np = _numpy()
    source = (REPO_ROOT / "src" / "python" / "ptm" / "results.py").read_text()
    assert source.count("struct.unpack('<I', header)") >= 2, (
        "a per-facet statistics header is no longer read as an unsigned "
        "little-endian 32-bit count in both readers")
    assert "struct.unpack('<i'" not in source

    from ptm.results import read_binary_data
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "temperature_z0p050m_avg.bin"
        values = np.arange(64, dtype="<f4")
        _write_stat_file(path, values)
        got = read_binary_data(path, 64)
        assert got is not None and len(got) == 64
        assert np.array_equal(got, values)

        # The header a signed reader would report as negative. Whatever it is
        # read as, it does not match this file, and the reader must say so
        # instead of returning the bytes that follow it.
        with open(path, "r+b") as handle:
            handle.seek(0)
            handle.write(struct.pack("<I", 0x80000001))
        assert read_binary_data(path, 64) is None, (
            "a file whose header claims 2,147,483,649 facets was accepted for "
            "a 64-facet mesh")


def test_a_truncated_depth_file_is_refused():
    """A count alone passes a truncated file -- the LENGTH is what catches it."""
    np = _numpy()
    from ptm.results import detect_available_statistics
    with tempfile.TemporaryDirectory() as scratch:
        stats, n_faces = _fixture(scratch)
        victim = stats / "temperature_z0p050m_max.bin"
        full = victim.read_bytes()
        victim.write_bytes(full[:len(full) - 4 * 100])
        names = {f"{b}_{s}" for b, s in detect_available_statistics(stats)}
    assert "temperature_z0p050m_max" not in names, (
        "a depth file 100 values short of its own header was reported as "
        "readable")
    assert "temperature_z0p050m_avg" in names, (
        "truncating one file suppressed its siblings")


def test_an_all_nan_depth_file_is_not_offered():
    np = _numpy()
    from ptm.results import detect_available_statistics
    with tempfile.TemporaryDirectory() as scratch:
        stats, n_faces = _fixture(scratch)
        _write_stat_file(stats / "temperature_z0p050m_min.bin",
                         np.full(n_faces, np.nan, dtype="<f4"))
        names = {f"{b}_{s}" for b, s in detect_available_statistics(stats)}
    assert "temperature_z0p050m_min" not in names, (
        "a depth map with no data in it was offered for export")


# ==========================================================================
# 3. The band metadata
# ==========================================================================

def test_the_band_tags_name_the_depth_and_the_one_that_was_asked_for():
    with tempfile.TemporaryDirectory() as scratch:
        stats, _n = _fixture(scratch)
        from ptm.results import depth_entries
        from ptm_cog import depth_band_tags
        entries = depth_entries(stats)
        tags = depth_band_tags("temperature_z0p298m_max", entries)
    assert tags["depth_m"] == "0.298"
    assert tags["depth_requested_m"] == "0.3"
    assert tags["depth_residual_m"] == "-0.002", (
        "the residual is the whole point: without it a raster holding 0.298 m "
        "and a request of 0.30 m are indistinguishable")
    assert tags["depth_on_solved_node"] == "true"


def test_the_band_depth_is_the_measured_one_not_the_rounded_file_name():
    """A file name carries three decimals; the run knows more than that.

    MEASURED on a real lunar run: a 0.05 m request resolved to 0.052724 m and
    the file was named ``z0p053m`` -- 276 um coarser. Reporting the name's
    value on the band would hand a reader a depth 276 um from the one the
    pixels hold, which is the kind of number that gets quoted.
    """
    from ptm_cog import depth_band_tags
    entries = {"temperature_z0p053m": {"requested_m": 0.05,
                                       "resolved_m": 0.0527243,
                                       "residual_m": 0.0027243,
                                       "on_node": True}}
    tags = depth_band_tags("temperature_z0p053m_avg", entries)
    assert tags["depth_m"] == "0.0527243", (
        f"the band reports {tags['depth_m']!r}, which is the file name's "
        f"rounding and not the depth the run sampled")
    # ...and with no manifest, the name is all there is, and that is what is
    # reported -- not a fabricated extra digit.
    assert depth_band_tags("temperature_z0p053m_avg")["depth_m"] == "0.053"


def test_a_surface_field_gets_no_depth_tags():
    from ptm_cog import depth_band_tags
    for field in ("temperature_avg", "direct_solar_max", "ice_depth",
                  "psr_mask"):
        assert depth_band_tags(field) == {}, (
            f"{field} was given depth metadata it has no business carrying")


def test_the_band_description_carries_the_depth():
    from ptm_cog import field_metadata
    units, file_desc, band_desc = field_metadata("temperature_z0p298m_max")
    assert units == "K"
    assert "0.298 m" in band_desc and "Maximum" in band_desc, (
        f"the band is described as {band_desc!r}, which does not say what "
        f"depth it holds")
    units, _f, band_desc = field_metadata("temperature_max")
    assert units == "K" and "depth" not in band_desc.lower(), (
        "a surface product gained a depth in its description")


def test_extra_tags_cannot_overwrite_a_computed_statistic():
    """A depth quietly replacing STATISTICS_MEAN would be a wrong number in a
    field name every reader trusts."""
    try:
        from ptm_cog import export_field, GDAL_AVAILABLE
    except ImportError as exc:                       # pragma: no cover
        raise Skip(f"the raster writer is unavailable here ({exc})")
    if not GDAL_AVAILABLE:
        raise Skip("osgeo.gdal is unavailable here")

    class _Regridder:                                # never reached
        pass

    try:
        export_field("/dev/null/not-written", [0.0], _Regridder(),
                     "temperature_z0p050m_avg",
                     extra_tags={"STATISTICS_MEAN": "0.05"})
    except ValueError as exc:
        assert "STATISTICS_MEAN" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError(
            "extra band metadata was allowed to collide with a computed "
            "statistics tag")


# ==========================================================================
# 4. End to end -- the raster on disk
# ==========================================================================

def _run_exporter(stats, out_dir):
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise Skip(f"rasterio is unavailable here ({exc})")
    try:
        from osgeo import gdal  # noqa: F401
    except ImportError as exc:
        raise Skip(f"osgeo.gdal is unavailable here ({exc})")
    if not CRATER.is_file():
        raise Skip(f"{CRATER} is not in this checkout")
    result = subprocess.run(
        [sys.executable, str(EXPORTER),
         "--mesh", str(CRATER), "--stats", str(stats),
         "--output", str(out_dir), "--resolution-m", "6"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"the exporter exited {result.returncode}:\n{result.stdout[-3000:]}")
    return result.stdout


def test_a_depth_map_opens_as_a_georeferenced_raster_carrying_its_depth():
    from osgeo import gdal  # noqa: F401  (skip handled in _run_exporter)
    with tempfile.TemporaryDirectory() as scratch:
        stats, _n = _fixture(scratch)
        out_dir = Path(scratch) / "cog"
        _run_exporter(stats, out_dir)
        written = sorted(p.name for p in out_dir.glob("*.tif"))
        assert "temperature_z0p298m_max.tif" in written, (
            f"the depth maps were not exported: {written}")
        assert "temperature_max.tif" in written, (
            "the surface products stopped being exported")

        from osgeo import gdal
        dataset = gdal.Open(str(out_dir / "temperature_z0p298m_max.tif"))
        assert dataset is not None, "the depth raster does not open"
        assert dataset.GetProjection(), (
            "the depth raster carries no projection, so it is not "
            "georeferenced")
        transform = dataset.GetGeoTransform()
        assert transform and transform[1] > 0, (
            f"the depth raster has no usable geotransform: {transform}")
        band = dataset.GetRasterBand(1)
        metadata = band.GetMetadata()
        assert metadata.get("depth_m") == "0.298", (
            f"the band does not carry its own depth: {metadata}")
        assert metadata.get("depth_requested_m") == "0.3", (
            f"the band does not carry the depth that was requested: "
            f"{metadata}")
        assert metadata.get("depth_residual_m") == "-0.002"
        assert "0.298 m" in band.GetDescription()
        assert dataset.GetMetadataItem("units") == "K"
        assert float(metadata["STATISTICS_VALID_PERCENT"]) > 50.0, (
            "the depth raster is mostly empty; the values did not reach it")
        dataset = None


def test_the_exporter_says_how_many_depth_bands_it_found():
    with tempfile.TemporaryDirectory() as scratch:
        stats, _n = _fixture(scratch)
        out_dir = Path(scratch) / "cog"
        output = _run_exporter(stats, out_dir)
    assert "subsurface temperature maps: 6 band(s)" in output, (
        f"the exporter did not report the depth bands it wrote:\n"
        f"{output[-2000:]}")


def test_without_a_manifest_the_raster_still_says_which_depth_it_holds():
    """A run directory that lost its sidecar still yields an honest raster:
    the depth in the file name, and no claim about what was requested."""
    with tempfile.TemporaryDirectory() as scratch:
        stats, _n = _fixture(scratch, with_manifest=False)
        out_dir = Path(scratch) / "cog"
        output = _run_exporter(stats, out_dir)
        from osgeo import gdal
        dataset = gdal.Open(str(out_dir / "temperature_z0p298m_avg.tif"))
        metadata = dataset.GetRasterBand(1).GetMetadata()
        dataset = None
    assert metadata.get("depth_m") == "0.298"
    assert "depth_requested_m" not in metadata, (
        "a requested depth was reported with no record to take it from")
    assert "no depth_map_z.json" in output, (
        f"the exporter did not say the sidecar was missing:\n"
        f"{output[-2000:]}")


TESTS = [
    test_a_depth_field_name_is_read_back_to_a_depth,
    test_the_two_python_mirrors_of_the_naming_rule_agree,
    test_the_depth_maps_are_discovered_from_a_run_directory,
    test_discovery_finds_nothing_in_a_run_that_asked_for_no_depths,
    test_the_statistics_listing_includes_the_depth_maps,
    test_a_missing_manifest_costs_the_requested_depth_and_nothing_else,
    test_the_exporter_enumerates_the_depth_maps,
    test_the_facet_count_is_read_as_an_unsigned_32_bit_header,
    test_a_truncated_depth_file_is_refused,
    test_an_all_nan_depth_file_is_not_offered,
    test_the_band_tags_name_the_depth_and_the_one_that_was_asked_for,
    test_the_band_depth_is_the_measured_one_not_the_rounded_file_name,
    test_a_surface_field_gets_no_depth_tags,
    test_the_band_description_carries_the_depth,
    test_extra_tags_cannot_overwrite_a_computed_statistic,
    test_a_depth_map_opens_as_a_georeferenced_raster_carrying_its_depth,
    test_the_exporter_says_how_many_depth_bands_it_found,
    test_without_a_manifest_the_raster_still_says_which_depth_it_holds,
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
    print(f"\n[test_depth_map_cog] {total - failures - skipped}/{total} passed"
          + (f", {skipped} skipped" if skipped else "")
          + (f", {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
