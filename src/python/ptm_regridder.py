"""Compatibility shim -- the implementation moved to ``ptm.results``.

Scripts that reach this module by putting ``src/python`` on ``sys.path``
(``from ptm_regridder import read_ply_mesh_minimal``) keep working unchanged.
New code should import from the package instead::

    from ptm.results import read_ply_mesh_minimal

There is no second copy of anything here: every name below is the one object
defined in ``ptm/results.py``.
"""

from ptm.results import (  # noqa: F401
    BODY_RADII,
    DEPTH_FIELD_RE,
    DEPTH_MANIFEST_NAME,
    FIELD_INFO,
    NICE_RESOLUTIONS,
    POLAR_LAT_THRESHOLD,
    SCIPY_AVAILABLE,
    STAT_LABELS,
    STAT_TYPES,
    STATISTICS_FIELDS,
    GridResolutionEstimator,
    MeshRegridder,
    _forward_polar_stereo,
    _inverse_polar_stereo,
    choose_projection,
    depth_entries,
    depth_of_field,
    detect_available_statistics,
    detect_body_from_mesh,
    discover_depth_fields,
    field_catalog,
    field_info,
    read_binary_data,
    read_depth_manifest,
    read_ply_mesh_minimal,
)

__all__ = [
    "BODY_RADII",
    "DEPTH_FIELD_RE",
    "DEPTH_MANIFEST_NAME",
    "FIELD_INFO",
    "NICE_RESOLUTIONS",
    "POLAR_LAT_THRESHOLD",
    "SCIPY_AVAILABLE",
    "STAT_LABELS",
    "STAT_TYPES",
    "STATISTICS_FIELDS",
    "GridResolutionEstimator",
    "MeshRegridder",
    "choose_projection",
    "depth_entries",
    "depth_of_field",
    "detect_available_statistics",
    "detect_body_from_mesh",
    "discover_depth_fields",
    "field_catalog",
    "field_info",
    "read_binary_data",
    "read_depth_manifest",
    "read_ply_mesh_minimal",
]
