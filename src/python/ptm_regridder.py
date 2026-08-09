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
    detect_available_statistics,
    detect_body_from_mesh,
    field_catalog,
    read_binary_data,
    read_ply_mesh_minimal,
)

__all__ = [
    "BODY_RADII",
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
    "detect_available_statistics",
    "detect_body_from_mesh",
    "field_catalog",
    "read_binary_data",
    "read_ply_mesh_minimal",
]
