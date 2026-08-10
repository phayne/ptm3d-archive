"""The run-directory readers that had no shared implementation anywhere.

``ptm.results`` already carries the two readers that were forked most widely --
``read_binary_data`` (the per-face ``.bin``) and ``read_ply_mesh_minimal``
(``mesh.ply``). The functions here are the rest of what a run directory
contains, each of which was being re-derived at every call site:

============================  =========================================
:func:`read_time_steps`       ``time_steps.txt``, with the J2000-ET rebase
:func:`read_body_positions`   ``body_positions.csv``
:func:`subsolar_latlon`       body-fixed sub-solar point from the sun vector
:func:`read_albedo_map`       the *other* binary format (size_t + float64)
:func:`find_nearest_face`     nearest face to a lat/lon
:func:`infer_mesh_length_unit`  metres or kilometres, by the engine's own rule
:func:`local_solar_time_bins`   noon-referenced binning of a per-step field
============================  =========================================

Nothing here computes physics. These read files, convert units and select
indices; the temperatures and fluxes were computed by ``ptm_thermal_sim``.
"""

from __future__ import annotations

import csv
import struct
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np

__all__ = [
    "DepthSeries",
    "MeshUnitError",
    "MeshUnitReport",
    "TimeAxis",
    "find_nearest_face",
    "infer_mesh_length_unit",
    "local_solar_time_bins",
    "read_albedo_map",
    "read_body_positions",
    "read_depth_series",
    "read_time_steps",
    "subsolar_latlon",
]


# --- time_steps.txt ---------------------------------------------------------

#: ``et`` seconds past J2000 (SPICE ET, what the file stores); ``elapsed_s``
#: seconds since the first output step; ``utc`` the file's own UTC labels.
TimeAxis = namedtuple("TimeAxis", "et elapsed_s utc")


def read_time_steps(path):
    """Read ``time_steps.txt`` into a :class:`TimeAxis`.

    Column 1 is **seconds past the J2000 epoch** -- SPICE ET -- even though the
    file's own header comment says "Time elapsed from start (seconds)". The
    header is wrong and has been wrong for every run in the tree;
    ``docs/tutorials/06_interpreting_outputs.md`` documents the trap. Eleven
    separate parsers in the repo re-discover it, several of them by plotting a
    time axis that starts at 8.3e8. ``elapsed_s`` is the rebased column, so a
    caller that wants elapsed time never has to know.
    """
    path = Path(path)
    et, utc = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        et.append(float(parts[0]))
        utc.append(parts[1] if len(parts) > 1 else "")
    if not et:
        raise ValueError(f"{path}: no time-step rows (only comments?)")
    et = np.asarray(et, dtype=np.float64)
    return TimeAxis(et=et, elapsed_s=et - et[0], utc=utc)


# --- locations/<name>_depths.csv ---------------------------------------------

#: ``location`` the name in the file's header comment; ``utc`` the time labels;
#: ``depths_m`` the depth of each column in metres; ``temperature`` the
#: ``(n_times, n_depths)`` block in kelvin; ``n_equilibration`` how many leading
#: rows were dropped as equilibration (0 from the raw reader --
#: :meth:`ptm.results.Results.depth_series` is what fills it in).
DepthSeries = namedtuple(
    "DepthSeries", "location utc depths_m temperature n_equilibration")


def read_depth_series(path):
    """Read a subsurface temperature profile written by ``output.depths_file``.

    A run configured with ``output.locations_file`` and ``output.depths_file``
    writes ``locations/<name>_depths.csv``: a ``# Location: <name>`` comment,
    then a header ``time,T(z=0.000 m),T(z=0.025 m),...``, then one row per
    step. The depths are recoverable only by parsing the *column headings*,
    which is why the ALMA workflow -- the one consumer of this product in the
    tree -- carries its own copy of that parse.

    Two properties of this file are not shared by any other output product,
    and both are silent:

    * **It covers the equilibration period as well as the output window**,
      where ``statistics/`` and the per-step binaries cover the output window
      only. Measured on ``quickstart_europa_diurnal`` with
      ``equilibration_orbits: 20``: 4,032 rows against 192 in
      ``time_steps.txt``, the last 192 of which match it exactly.
      :meth:`ptm.results.Results.depth_series` trims to the output window by
      matching those labels; this function returns the file as written.
    * **The depths are printed to three decimals**, so two requested depths
      closer than a millimetre become the same heading. That is data loss in
      the file, not in the parse, and it is reported here rather than passed
      on quietly.
    """
    path = Path(path)
    location = path.stem
    header = None
    rows, utc = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            _, _, tail = line.partition("Location:")
            if tail.strip():
                location = tail.strip()
            continue
        if header is None:
            header = [c.strip() for c in line.split(",")]
            continue
        parts = line.split(",")
        utc.append(parts[0])
        rows.append([float(v) for v in parts[1:]])
    if header is None or not rows:
        raise ValueError(f"{path}: no depth-series rows found")

    depths = []
    for column in header[1:]:
        # 'T(z=0.025 m)' -> 0.025
        _, _, tail = column.partition("z=")
        token = tail.replace("m)", "").replace(")", "").strip()
        try:
            depths.append(float(token))
        except ValueError:
            raise ValueError(
                f"{path}: cannot read a depth out of the column heading "
                f"{column!r}; expected the form 'T(z=0.025 m)'")
    temperature = np.asarray(rows, dtype=np.float64)
    if temperature.shape[1] != len(depths):
        raise ValueError(
            f"{path}: {temperature.shape[1]} data columns but {len(depths)} "
            f"depth headings")
    depths = np.asarray(depths, dtype=np.float64)
    repeated = int(np.sum(np.diff(depths) <= 0.0))
    if repeated:
        import warnings
        warnings.warn(
            f"{path}: {repeated} of {depths.size} depth headings do not "
            f"increase -- the file prints depths to three decimals, so "
            f"requested depths closer than 1 mm collapse onto one label. "
            f"Space the entries in output.depths_file further apart.",
            stacklevel=2)
    return DepthSeries(location=location, utc=utc, depths_m=depths,
                       temperature=temperature, n_equilibration=0)


# --- body_positions.csv -----------------------------------------------------

def read_body_positions(path):
    """Read ``body_positions.csv`` into a structured array.

    The engine writes one row per output step: a UTC ``time`` label, then the
    Sun and primary position vectors in kilometres, **in the body-fixed frame**
    -- which is why :func:`subsolar_latlon` is a coordinate conversion rather
    than an inference. Columns are taken from the header, so a run that gains a
    column keeps working.
    """
    path = Path(path)
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no rows")
    names = list(rows[0].keys())
    numeric = [n for n in names if n != "time"]
    dtype = [("time", "U32")] + [(n, "<f8") for n in numeric]
    out = np.zeros(len(rows), dtype=dtype)
    out["time"] = [r.get("time", "") for r in rows]
    for name in numeric:
        out[name] = [float(r[name]) for r in rows]
    return out


def subsolar_latlon(positions, step=0):
    """Body-fixed sub-solar latitude and longitude (degrees) at ``step``.

    ``positions`` is what :func:`read_body_positions` returned. The Sun vector
    is already in the body-fixed frame, so this is ``asin``/``atan2`` on a
    recorded vector and nothing more -- no illumination is examined and no
    physical direction is inferred.

    Cross-checked against the flux-weighted estimate the tutorial notebooks
    use (the ``direct_solar``-weighted mean of the face directions) on
    ``quickstart_moon_diurnal``: the two agree to better than 0.01 degrees at
    steps 0, 60, 120 and 180, against a 5.7-degree mesh face pitch.

    Longitude is returned on [-180, 180], the same convention ``mesh.ply``
    uses for its per-face longitudes.
    """
    row = positions[int(step)]
    vec = np.array([row["sun_x"], row["sun_y"], row["sun_z"]], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if not norm > 0:
        raise ValueError(f"body_positions row {step}: sun vector has zero length")
    vec /= norm
    return (float(np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0)))),
            float(np.degrees(np.arctan2(vec[1], vec[0]))))


# --- the other binary format ------------------------------------------------

def read_albedo_map(path, expected_count=None):
    """Read a per-face albedo map: ``size_t`` count (8 bytes) + ``n`` float64.

    This is **not** the ``statistics/*.bin`` layout. Those are an ``int32``
    count and float32 values; the albedo maps that
    ``src/cpp/tools/albedo_map_generator.cpp`` writes (:826, :835) use an
    8-byte count and float64. The difference is documented in no file under
    ``docs/``, which is why a reader that assumes the statistics layout reads
    an albedo map as garbage rather than failing.

    Raises on a truncated file, and on a face-count mismatch when
    ``expected_count`` is given, instead of returning a short array.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: {len(raw)} bytes, too short for the 8-byte count")
    n_faces = struct.unpack_from("<Q", raw, 0)[0]
    expected_bytes = 8 + 8 * n_faces
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path}: header says {n_faces} faces ({expected_bytes} bytes), file is "
            f"{len(raw)} bytes. If it is 4 + 4n bytes this is a statistics/*.bin "
            f"(int32 + float32), not an albedo map")
    values = np.frombuffer(raw, dtype="<f8", count=n_faces, offset=8)
    if expected_count is not None and n_faces != expected_count:
        raise ValueError(
            f"{path}: {n_faces} values but the mesh has {expected_count} faces")
    return values


# --- face selection ---------------------------------------------------------

def find_nearest_face(target_lat, target_lon, latitudes, longitudes):
    """Index of the face nearest ``(target_lat, target_lon)``, and how far.

    Lifted from ``src/python/extract_timeseries.py:196``, which is the one
    implementation of this in the tree and reached three importers, all in
    ``scripts/jwst/``, because it sits under a 170-line argparse ``main``.

    Returns ``(index, separation_degrees)``. The great-circle form is
    insensitive to the longitude convention -- ``sin^2(dlon/2)`` is unchanged
    by a 360-degree shift -- so a mesh on [-180, 180] and a target on
    [0, 360) compare correctly.
    """
    target_lon = target_lon % 360.0

    lat1 = np.radians(target_lat)
    lon1 = np.radians(target_lon)
    lat2 = np.radians(np.asarray(latitudes, dtype=np.float64))
    lon2 = np.radians(np.asarray(longitudes, dtype=np.float64))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    angular_dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    nearest_idx = int(np.argmin(angular_dist))
    return nearest_idx, float(np.degrees(angular_dist[nearest_idx]))


# --- mesh length unit -------------------------------------------------------

class MeshUnitError(ValueError):
    """The mesh's length unit cannot be determined safely.

    Raised in exactly the cases where ``ptm_thermal_sim`` prints
    ``FATAL [mesh-units]`` and refuses to run.
    """


#: ``unit`` is 'm' or 'km'; ``corroborated`` is True when a body radius agrees
#: with the magnitude heuristic, False when the magnitude alone decided it.
#: ``body`` is the closest-fitting known radius under the chosen unit -- it
#: corroborates the UNIT and is not an identification of the body.
MeshUnitReport = namedtuple("MeshUnitReport", "unit median lo hi corroborated body")

#: Equatorial radii in kilometres, used only to corroborate the unit.
_BODY_RADII_KM = {
    "Europa": 1560.8, "Moon": 1737.4, "Mercury": 2439.7, "Mars": 3389.5,
    "Enceladus": 252.1, "Titan": 2574.73, "Ceres": 473.0,
}

# The constants below are the engine's, from inspectMeshLengthUnit()
# (src/cpp/main/ptm_thermal_sim.cpp:8195-8215). They are copied deliberately:
# the point of this function is to reach the SAME verdict the run reached, not
# a better one. A cleverer rule here would make Results disagree with the
# engine about the mesh it is reading.
_DISCRIMINATOR = 10000.0
_AMBIGUOUS_FACTOR = 50.0
_RADIUS_TOL_LO, _RADIUS_TOL_HI = 0.5, 2.0


def infer_mesh_length_unit(face_centers, vertices=None, body_radius_km=None):
    """Decide whether a mesh is in metres or kilometres, as the engine does.

    ``mesh.ply`` records no length unit. The same file is metres for a
    topographic run and kilometres for a plain icosphere -- on the same body,
    with the same header. Every consumer in the tree re-derives the unit from
    ``|face_center| > 10000``, and getting it backwards is a 1000x error in
    every length (that was PTM-36: Jupiter's shadow shrank 1000x and every
    Europa eclipse vanished).

    This is a port of ``inspectMeshLengthUnit`` in
    ``src/cpp/main/ptm_thermal_sim.cpp:8206``, including its four refusals:
    face centers inconsistent with the vertices, a mesh straddling the
    discriminator, a body radius that contradicts the heuristic, and a median
    inside the ambiguous band with nothing to corroborate it. Those raise
    :class:`MeshUnitError`; a verdict reached on magnitude alone warns.

    ``body_radius_km`` is the run's body radius when the caller knows it. The
    run directory does not record the body, so when it is omitted the known
    radii above are tried instead: a unit is corroborated when some known body
    is within a factor of two of the median under that interpretation, and
    only that interpretation.
    """
    centers = np.asarray(face_centers, dtype=np.float64)
    if centers.size == 0:
        raise MeshUnitError("mesh has no faces")

    mag = np.linalg.norm(centers, axis=1)
    median = float(np.median(mag))
    lo, hi = float(mag.min()), float(mag.max())
    meters = median > _DISCRIMINATOR

    # 0. Face centers are vertex averages, so their magnitudes cannot
    #    legitimately differ by a unit scale. When they do, something rescaled
    #    one array and not the other (PTM-65, mesh.scale_factor).
    if vertices is not None and len(vertices) and median > 0:
        vmedian = float(np.median(np.linalg.norm(
            np.asarray(vertices, dtype=np.float64), axis=1)))
        if vmedian > 0:
            ratio = median / vmedian
            if ratio < 0.5 or ratio > 2.0:
                raise MeshUnitError(
                    f"mesh is internally inconsistent: median |face_center| = "
                    f"{median:g} but median |vertex| = {vmedian:g} (ratio "
                    f"{ratio:g}). Face centers are vertex averages, so these "
                    f"cannot legitimately differ.")

    # 1. Straddling the discriminator: different call sites would classify the
    #    same mesh differently.
    if lo <= _DISCRIMINATOR < hi:
        raise MeshUnitError(
            f"mesh face centers straddle the {_DISCRIMINATOR:g} unit "
            f"discriminator (min {lo:g}, max {hi:g}); consumers classify the "
            f"whole mesh from one face and would disagree about this one")

    # 2/3. Corroborate against a body radius.
    if body_radius_km is not None and body_radius_km > 0:
        candidates = {"body": float(body_radius_km)}
    else:
        candidates = dict(_BODY_RADII_KM)

    def _match(scale):
        """The best-fitting candidate radius under this interpretation, if any.

        Best-fitting, not first-fitting: the tolerance is a factor of two, so
        several bodies accept the same mesh and dict order would otherwise
        decide which one gets named. A Moon mesh reported as 'consistent with
        Europa' is technically true and reads as a mistake.
        """
        best, best_error = None, None
        for name, radius in candidates.items():
            ratio = (median / scale) / radius
            if _RADIUS_TOL_LO <= ratio <= _RADIUS_TOL_HI:
                error = abs(np.log(ratio))
                if best_error is None or error < best_error:
                    best, best_error = name, error
        return best

    km_body = _match(1.0)
    m_body = _match(1000.0)
    if km_body is not None and m_body is not None:
        km_body = m_body = None          # both fit: corroborates nothing
    says_km, says_m = km_body is not None, m_body is not None

    if (says_km or says_m) and (says_m != meters):
        raise MeshUnitError(
            f"the body radius contradicts the magnitude heuristic: median "
            f"|face_center| = {median:g} matches {km_body or m_body} in "
            f"{'KILOMETERS' if says_km else 'METERS'}, but the "
            f"|face_center| > {_DISCRIMINATOR:g} rule every consumer uses says "
            f"{'METERS' if meters else 'KILOMETERS'}. Running on this would "
            f"scale every SPICE-derived length by 1000x the wrong way (PTM-36)")

    in_band = (_DISCRIMINATOR / _AMBIGUOUS_FACTOR < median
               < _DISCRIMINATOR * _AMBIGUOUS_FACTOR)
    if in_band and not (says_km or says_m):
        raise MeshUnitError(
            f"median |face_center| = {median:g} sits in the ambiguous band "
            f"({_DISCRIMINATOR / _AMBIGUOUS_FACTOR:g} .. "
            f"{_DISCRIMINATOR * _AMBIGUOUS_FACTOR:g}) around the "
            f"{_DISCRIMINATOR:g} discriminator and no body radius confirms "
            f"either unit. Pass mesh_units='m' or 'km', or body_radius_km=..., "
            f"to say which it is")

    unit = "m" if meters else "km"
    if not (says_km or says_m):
        warnings.warn(
            f"mesh length unit inferred as {unit.upper()} from magnitude alone "
            f"(median |face_center| = {median:g}); no body radius corroborates "
            f"it. If this mesh is not body-centered, check it.",
            stacklevel=2)
    return MeshUnitReport(unit=unit, median=median, lo=lo, hi=hi,
                          corroborated=bool(says_km or says_m),
                          body=(km_body or m_body))


# --- local solar time -------------------------------------------------------

def local_solar_time_bins(values, noon_reference, n_bins=48):
    """Bin a per-step, per-face field by local solar time.

    ``values`` and ``noon_reference`` are both ``(n_steps, n_faces)``. Each
    face's local noon is the step at which ``noon_reference`` peaks for that
    face -- in practice ``direct_solar`` -- and every sample is placed on a
    24-hour local-solar-time axis relative to it. One rotation of output is
    assumed to span the ``n_steps`` given, so the caller selects the steps.

    Returns ``(centers, mean, p25, p75)``, each of length ``n_bins``. Bins with
    no samples are NaN rather than an error, so a partial rotation still plots.

    This is the reduction the asteroid notebooks carry verbatim in two copies.
    Those notebooks pass the *second* of two output rotations, on the grounds
    that it is the settled one; that choice is the caller's, not this
    function's, so nothing here slices the array.
    """
    values = np.asarray(values, dtype=np.float64)
    noon_reference = np.asarray(noon_reference, dtype=np.float64)
    if values.shape != noon_reference.shape:
        raise ValueError(f"values {values.shape} and noon_reference "
                         f"{noon_reference.shape} must have the same shape")
    n_steps = values.shape[0]
    if n_steps < 2:
        raise ValueError("need at least two steps to bin by local solar time")

    noon_step = np.argmax(noon_reference, axis=0)
    steps = np.arange(n_steps)[:, None]
    lst = ((steps - noon_step[None, :]) * (24.0 / n_steps) + 12.0) % 24.0

    lst_flat, val_flat = lst.ravel(), values.ravel()
    edges = np.linspace(0.0, 24.0, n_bins + 1)
    idx = np.clip(np.digitize(lst_flat, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Group by sorting once, rather than testing `idx == b` for every bin: on a
    # 140k-face patch over 240 steps that is 34M samples, and the masked form
    # walks all of them once per bin.
    order = np.argsort(idx, kind="stable")
    bounds = np.searchsorted(idx[order], np.arange(n_bins + 1))
    sorted_values = val_flat[order]

    mean = np.full(n_bins, np.nan)
    p25 = np.full(n_bins, np.nan)
    p75 = np.full(n_bins, np.nan)
    for b in range(n_bins):
        group = sorted_values[bounds[b]:bounds[b + 1]]
        if group.size:
            mean[b] = group.mean()
            p25[b], p75[b] = np.percentile(group, [25, 75])
    return centers, mean, p25, p75
