"""PTM3D Python layer.

An importable home for the pieces of PTM3D's Python tooling that were
previously reachable only by putting ``src/python`` or ``gui`` on
``sys.path``. Nothing here computes physics: the C++ engine
(``ptm_thermal_sim``) does that, and this package reads what it wrote and
launches it.

Modules
-------
``ptm.results``        readers for a run directory: per-face ``.bin``,
                       ``mesh.ply``, statistics discovery, regridding
``ptm.run``            launching the engine: executable discovery, the output
                       severity classifier, the CPU-clock stall watchdog,
                       process-group teardown
``ptm.paths``          repository and executable location
``ptm.engine_config``  the config keys and defaults the engine actually reads,
                       derived from ``src/cpp/tools/config_parser.{h,cpp}``

Install with ``pip install -e .`` from the repository root.
"""

__version__ = "0.1.0"

__all__ = ["__version__", "engine_config", "paths", "results", "run"]
