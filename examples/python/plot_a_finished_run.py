#!/usr/bin/env python3
"""From a finished run directory to four figures.

    python3 examples/python/plot_a_finished_run.py output/quickstart_moon_diurnal

This is the same set of pictures tutorial notebook 01 draws, written against
``ptm.Results``. The notebook spends 246 non-blank lines of code getting from
the run directory to its orthographic globe figure -- 60 for a ``mesh.ply``
parser and a per-face ``.bin`` reader, 69 for a matplotlib style block and the
``T**4`` color scale, 84 for the globe renderer, 33 for the figure itself.
The equivalent below is the three lines under "the whole thing".

Everything else here is choosing which pictures to draw.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from ptm import Results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="a finished PTM3D output directory")
    parser.add_argument("--outdir", default="figures", help="where to write")
    parser.add_argument("--field", default="temperature")
    parser.add_argument("--units", default="ascii", choices=("ascii", "unicode"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- the whole thing ---------------------------------------------------
    run = Results(args.run_dir, units_style=args.units)
    run.plot_globes(args.field, "max", save=outdir / "globes.png")
    run.plot_diurnal(field=args.field, lat=0, lon=0, save=outdir / "diurnal.png")
    # -----------------------------------------------------------------------

    run.plot_map(args.field, "max", save=outdir / "map.png")
    run.plot_histogram(args.field, "max", save=outdir / "histogram.png")

    # A snapshot, with the sub-solar point read from body_positions.csv.
    mid = len(run.times().et) // 2
    run.plot_globes(args.field, step=mid, mark_subsolar=True,
                    save=outdir / "globes_snapshot.png")

    provenance = run.provenance()
    print(f"{run}\n  mesh in {run.mesh_units} "
          f"(radius {provenance['mesh_median_radius_m'] / 1e3:.4g} km)")
    print(f"  wrote 5 figures to {outdir}/")
    print("  the run does not record: "
          + "; ".join(provenance["not_recorded"]))


if __name__ == "__main__":
    main()
