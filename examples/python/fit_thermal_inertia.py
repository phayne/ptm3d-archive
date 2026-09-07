#!/usr/bin/env python3
"""Sweep -> lookup table -> fit: thermal inertia from a brightness-temperature curve.

    python3 examples/python/fit_thermal_inertia.py

Six small CPU-only Europa runs (about 18 s each on a 48-core box), one
brightness-temperature curve per run, and one number with an error bar at the
end. This is the shape every fit in this repository already has, and the reason
:class:`ptm.Sweep` is the load-bearing piece rather than an optimizer: the
objective below is evaluated a few thousand times against **five** engine runs.
Re-running the engine inside the optimizer would take months.

The three parts, in the order they matter
-----------------------------------------

1. **The sweep** builds the lookup table -- five thermal inertias, five
   subsurface temperature profiles ``T(z, t)``.

2. **The observable conversion is yours, not the library's.** ``observable``
   below is an ordinary function of a :class:`ptm.Results`; nothing in ``ptm``
   knows what a brightness temperature is. Here it is the repository's
   depth-weighted emission model,
   ``scripts/emission/depth_weighted_emission.radiative_transfer_1d`` over
   ``scripts/emission/dielectric_mixing.py`` -- Maxwell-Garnett mixing for
   porous ice, Fresnel surface emissivity, and the emission integral
   ``Tb = eps * INT T(z) kappa(z) exp(-tau(z)) dz``. Swap it for an infrared
   band, a bolometer, or anything else and nothing else in this script changes.

   Why a millimeter observable is worth the trouble: at these frequencies the
   regolith is semi-transparent, so ``Tb`` is a weighted average over the top
   few decimeters rather than a reading of the skin. That depth weighting is
   what makes the curve sensitive to thermal inertia in the first place.

3. **The fit** is a lookup table plus interpolation, not an engine in a loop.

What is real and what is synthetic
----------------------------------

The five model runs are real PTM3D output. The "observations" are **not**
measured data: they are a sixth engine run at a thermal inertia deliberately
placed *between* two grid points, pushed through the same observable and given
Gaussian noise, so that the answer is known and the recovery is checkable. A
fit against real data replaces ``observed()`` and nothing else.

Needs scipy (``pip install 'ptm[regrid]'`` brings it in). numpy at the
repository floor (>=1.20) is enough: the emission integral binds
``numpy.trapezoid`` or, before numpy 2.0 renamed it, ``numpy.trapz``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize_scalar
except ImportError:                                            # pragma: no cover
    raise SystemExit(
        "this example needs scipy for the interpolation and the 1-D minimizer.\n"
        "    pip install 'ptm[regrid]'      # the extra this package already has\n"
        "or  pip install scipy")

from ptm import Results, Sweep
from ptm.paths import find_repo_root

# --- the observable: the repository's depth-weighted emission model ----------

REPO = find_repo_root(Path(__file__).parent)
if REPO is None:
    raise SystemExit("run this from inside a PTM3D checkout")
sys.path.insert(0, str(REPO / "scripts" / "emission"))
try:
    from depth_weighted_emission import radiative_transfer_1d
except ImportError as exc:                                     # pragma: no cover
    raise SystemExit(
        f"could not import scripts/emission/depth_weighted_emission.py: {exc}")

FREQ_GHZ = 233.0     # observing frequency, 1.3 mm
POROSITY = 0.5       # regolith porosity fed to the Maxwell-Garnett mixing rule
EMISSIVITY = "fresnel"
SIGMA_TB = 1.5       # K, the assumed observational uncertainty per sample


def observable(results):
    """A PTM3D run -> the curve a millimeter radiometer would produce.

    ``(elapsed_hours, brightness_temperature_K)``. This is the user-supplied
    conversion: it reads the subsurface profile the run wrote and integrates
    the repository's depth-weighted emission model over it at each output
    time.
    """
    profile = results.depth_series()
    tb = np.array([
        radiative_transfer_1d(profile.depths_m, temperature_column,
                              FREQ_GHZ, porosity=POROSITY,
                              emissivity=EMISSIVITY)
        for temperature_column in profile.temperature])
    hours = results.elapsed_hours[:len(tb)]
    return hours, tb


# --- the fit: a lookup table, interpolated ------------------------------------

class LookupFit:
    """chi-squared over a sweep's lookup table, interpolated between points.

    The table is interpolated in ``log(TI)`` because thermal inertia is a scale
    parameter and the grid is geometric; the observable is smooth in the
    logarithm and kinked in the value.
    """

    def __init__(self, thermal_inertia, curves_y, sigma):
        order = np.argsort(thermal_inertia)
        self.ti = np.asarray(thermal_inertia, dtype=float)[order]
        self.table = np.asarray(curves_y, dtype=float)[order]
        self.sigma = float(sigma)
        if self.ti.size < 4:
            raise ValueError(
                f"a cubic interpolation over the grid needs 4 points; got "
                f"{self.ti.size}")
        spread = float(np.ptp(self.table.mean(axis=1)))
        if spread < 1.0:
            # The guard scripts/ema_ti_fit.py carries, and it earned its place:
            # it caught a solver build on which thermal inertia reached the
            # column not at all, and the fit would otherwise have returned a
            # confident number off a flat objective.
            raise RuntimeError(
                f"the curve mean varies by only {spread:.3g} K across the "
                f"whole thermal-inertia axis. The observable is not sensitive "
                f"to this parameter (or the parameter is not reaching the "
                f"solver); refusing to fit.")
        self._interp = interp1d(np.log(self.ti), self.table, axis=0,
                                kind="cubic", assume_sorted=True)

    def model(self, thermal_inertia):
        return self._interp(np.log(thermal_inertia))

    def chi2(self, thermal_inertia, observed):
        residual = self.model(thermal_inertia) - observed
        return float(np.sum((residual / self.sigma) ** 2))

    def fit(self, observed):
        """``(best_ti, minus_sigma, plus_sigma, chi2_min, dof)``.

        The error bar is the ``delta chi-squared = 1`` interval, scanned rather
        than taken from a curvature estimate, because the objective is only
        piecewise-smooth through the interpolant.
        """
        lo, hi = float(self.ti[0]), float(self.ti[-1])
        best = minimize_scalar(lambda t: self.chi2(t, observed),
                               bounds=(lo, hi), method="bounded",
                               options={"xatol": 1e-3})
        scan = np.geomspace(lo, hi, 2001)
        chi2 = np.array([self.chi2(t, observed) for t in scan])
        inside = scan[chi2 <= best.fun + 1.0]
        low = float(inside.min()) if inside.size else float(best.x)
        high = float(inside.max()) if inside.size else float(best.x)
        dof = observed.size - 1
        return float(best.x), low, high, float(best.fun), dof


# --- driver --------------------------------------------------------------------

def build_inputs(outdir):
    """The depth grid and the one location the runs record a profile at."""
    outdir.mkdir(parents=True, exist_ok=True)
    locations = outdir / "location.csv"
    locations.write_text("name,lat,lon\nequator,0.0,0.0\n")
    # 0 to 2 m, log-spaced: the 233 GHz penetration depth in porous ice is
    # decimeters, so the grid has to resolve the top few centimeters and still
    # reach optical depth. Rounded to millimeters and de-duplicated because the
    # engine prints depth column headings to three decimals, and two entries
    # closer than that come back as one ambiguous column.
    depths = np.unique(np.round(
        np.concatenate([[0.0], np.geomspace(3e-3, 2.0, 48)]), 3))
    depths_file = outdir / "depths.txt"
    depths_file.write_text("\n".join(f"{d:.6f}" for d in depths) + "\n")
    return locations, depths_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="output/europa_ti_fit_sweep")
    parser.add_argument("--truth", type=float, default=65.0,
                        help="the thermal inertia the synthetic observations "
                             "are drawn from; deliberately off-grid")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--figure", default=None,
                        help="write the curves-and-residuals figure here")
    parser.add_argument("--exe", default=None, help="engine binary")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    locations, depths_file = build_inputs(outdir)
    # thermal_b, and it is REQUIRED, not tuning. Since 2026-09-07 a requested
    # depth below the deepest solved node is refused rather than extrapolated
    # (docs/configuration.md, "Per-location depth series"), and this script asks
    # for 2.0 m. In constant-thermal-inertia mode the grid bottom is
    # b * delta_0 with delta_0 proportional to TI, so the SHALLOWEST grid in the
    # sweep is the smallest TI: measured on this exact base config, TI = 30 at
    # the default b = 15 reaches 1.4998 m, which would refuse. b = 21 puts it at
    # ~2.10 m with margin, and every larger TI deeper still. Lower the smallest
    # grid point in `grid=` below and this number has to come up with it.
    fixed = {"output.locations_file": str(locations),
             "output.depths_file": str(depths_file),
             "thermal.thermal_b": 21}
    base = REPO / "examples" / "quickstart_europa_diurnal.yaml"

    # 1. the sweep -- the lookup table.
    #
    # constant_thermal_inertia, not thermal_inertia, and the choice is not
    # incidental: they are different physics. This one solves for cp and
    # collapses the column so TI is exactly constant in temperature and depth,
    # and it reaches any engine (--constant-thermal-inertia, or the config key
    # of the same name). The other scales ks/kd and preserves the depth
    # contrast, and needs an engine with --thermal-inertia. At TI=200 on the
    # asteroid quickstart they differ by 2.9 K in diurnal amplitude. Swap the
    # key below to fit the other convention; nothing else changes.
    sweep = Sweep(base, grid={"constant_thermal_inertia": [30, 50, 80, 120, 200]},
                  outdir=outdir, overrides=fixed, exe=args.exe)
    sweep.run()

    curves = sweep.curves(of=observable)
    x = curves[0].x
    for curve in curves[1:]:
        if not np.allclose(curve.x, x):
            raise SystemExit(
                "the grid points do not share a time axis; every point must "
                "run the same schedule for a lookup table to mean anything")
    table = np.array([c.y for c in curves])
    thermal_inertia = sweep.values("constant_thermal_inertia")

    # 2. the "observations" -- a sixth run, off-grid, plus noise
    truth_dir = outdir / f"truth_ti-{args.truth:g}"
    truth = Sweep(base, grid={"constant_thermal_inertia": [args.truth]},
                  outdir=truth_dir, overrides=fixed, exe=args.exe, quiet=True)
    truth.run()
    _, clean = observable(truth.points[0].results)
    rng = np.random.default_rng(args.seed)
    observed = clean + rng.normal(0.0, SIGMA_TB, size=clean.shape)

    # 3. the fit
    fit = LookupFit(thermal_inertia, table, SIGMA_TB)
    best, low, high, chi2, dof = fit.fit(observed)

    print()
    print(f"{FREQ_GHZ:g} GHz, porosity {POROSITY}, "
          f"emissivity model {EMISSIVITY!r}, sigma {SIGMA_TB} K")
    print(f"grid              {thermal_inertia.tolist()} SI "
          f"({len(x)} samples per curve)")
    print(f"table spread      {np.ptp(table.mean(axis=1)):.3f} K in the curve "
          f"mean, {np.ptp(np.ptp(table, axis=1)):.3f} K in its amplitude")
    print(f"best-fit TI       {best:.1f}  (+{high - best:.1f} / "
          f"-{best - low:.1f}) SI   [delta chi2 = 1]")
    print(f"truth             {args.truth:.1f} SI  -> "
          f"{'recovered' if low <= args.truth <= high else 'MISSED'}")
    print(f"chi2 / dof        {chi2:.1f} / {dof} = {chi2 / dof:.2f}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        sweep.plot_residuals(
            x, observed, obs_sigma=SIGMA_TB, of=observable,
            xlabel="Elapsed time (h)",
            ylabel=r"$T_\mathrm{b}$ (K)",
            obs_label=f"observed (TI = {args.truth:g} SI)",
            legend_title="Thermal inertia (SI)",
            save=args.figure)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
