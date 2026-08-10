"""PTM3D Python layer.

An importable home for the pieces of PTM3D's Python tooling that were
previously reachable only by putting ``src/python`` or ``gui`` on
``sys.path``. Nothing here computes physics: the C++ engine
(``ptm_thermal_sim``) does that, and this package reads what it wrote and
launches it.

The short version::

    from ptm import paths, Results
    from ptm import run as engine

    paths.preflight(["examples/quickstart_moon_diurnal.yaml"],
                    engine="ptm_thermal_sim").raise_if_missing()
    done = engine.run_engine("examples/quickstart_moon_diurnal.yaml", echo=True)
    run = done.results()
    run.plot_globes("temperature", "max", save="globes.png")

or, for a grid of runs::

    from ptm import Sweep
    sweep = Sweep("examples/quickstart_asteroid_diurnal.yaml",
                  grid={"thermal_inertia": [50, 100, 200, 400]})
    sweep.run()
    sweep.plot_curves(save="family.png")

or, from a shell, ``ptm info <run_dir>``,
``ptm plot <run_dir> map --save map.png`` and
``ptm sweep <config> --vary thermal_inertia=50,100,200``.

Modules
-------
``ptm.results``        a run directory: the :class:`~ptm.results.Results`
                       object, plus the per-face ``.bin`` and ``mesh.ply``
                       readers, statistics discovery and regridding
``ptm.readers``        the rest of a run directory: ``time_steps.txt`` with the
                       J2000-ET rebase, ``body_positions.csv``, the albedo-map
                       binary, nearest-face selection, the mesh length unit
``ptm.sweep``          a base config plus a grid: :class:`~ptm.sweep.Sweep`,
                       resumable, provenance-recording, serial by default
``ptm.plots``          the orthographic globe renderer and the shape-model
                       renderer, the flux-linear colour scale, the comparison
                       grid, five canned figures and two sweep figures
``ptm.cli``            the ``ptm`` command
``ptm.run``            launching the engine: :func:`~ptm.run.run_engine` and
                       the callback-shaped ``EngineProcess`` under it, plus
                       executable discovery, engine-flag detection, the output
                       severity classifier, the CPU-clock stall watchdog,
                       process-group teardown, and
                       :func:`~ptm.run.binary_provenance`
``ptm.paths``          repository and executable location, and
                       :func:`~ptm.paths.preflight`
``ptm.engine_config``  the config keys and defaults the engine actually reads,
                       derived from ``src/cpp/tools/config_parser.{h,cpp}``

Two further packages ship with this distribution and are importable the same
way: ``damit`` (DAMIT shape search and conversion) and ``horizons`` (the JPL
HORIZONS client, SPK generator and body-config writer). They are the small-body
preset half of the workflow.

Install with ``pip install -e .`` from the repository root.
"""

__version__ = "0.4.0"

__all__ = ["Results", "Sweep", "__version__", "cli", "engine_config", "paths",
           "plots", "readers", "results", "run", "sweep"]

_LAZY = {"Results": ("results", "Results"), "Sweep": ("sweep", "Sweep")}


def __getattr__(name):
    """Expose ``ptm.Results`` and ``ptm.Sweep`` without importing numpy at
    ``import ptm``.

    ``ptm.paths`` and ``ptm.engine_config`` are usable with no third-party
    package installed at all, and several scripts rely on that, so the heavy
    imports stay lazy.
    """
    target = _LAZY.get(name)
    if target is not None:
        import importlib
        return getattr(importlib.import_module(f".{target[0]}", __name__),
                       target[1])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
