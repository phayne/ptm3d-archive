#!/usr/bin/env python3
"""The 40-file Europa product, as one sweep -- checked against the 40 files.

    python3 examples/python/sweep_replaces_45_configs.py
    python3 examples/python/sweep_replaces_45_configs.py --dry-run   # needs an engine

``configs/europa_lat{00,15,30,45,60,75,80,85}_{agenor,cilix,pwyll,rhad,ridges}.yaml``
is a latitude x DEM cartesian product materialized as one file per point. The
friction audit measured the pairwise differences at two lines (latitude) and
three (DEM). This script does three things, in increasing order of how much
they are worth:

1. **Measures** the cluster on the current checkout -- files, lines, and how
   many of those lines are distinct.
2. **Builds** the :class:`ptm.Sweep` that produces the same runs, taking the
   per-DEM paths *out of the shipped configs* rather than retyping them.
3. **Checks** that the sweep's generated config for every one of the 40 points
   equals the shipped file for that point, key by key. Anything that does not
   match is printed. Without step 3 this script would be a claim; with it, it
   is a measurement.

Nothing is deleted and nothing is run. The runs themselves need the Europa
DEMs, which are not distributed with the repository -- ``--dry-run`` prints the
engine command for each point without launching any.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import re
from pathlib import Path

import yaml

from ptm.engine_config import flatten_config, get_dotted
from ptm.paths import find_repo_root

REPO = find_repo_root(Path(__file__).parent)
if REPO is None:
    raise SystemExit("run this from inside a PTM3D checkout")

CONFIG_DIR = REPO / "configs"
PATTERN = re.compile(r"europa_lat(\d+)_([a-z]+)\.yaml$")

#: The keys that differ across the product, and nothing else does.
VARYING = ("mesh.latitude_override", "mesh.import_file",
           "radiation.view_factor_file", "output.directory")


def discover():
    """``{(latitude, dem): path}`` for the shipped cluster."""
    found = {}
    for path in sorted(CONFIG_DIR.glob("europa_lat*_*.yaml")):
        match = PATTERN.search(path.name)
        if match:
            found[(int(match.group(1)), match.group(2))] = path
    return found


def axes_from(cluster):
    """The latitude and DEM axes, in the order the filenames give them."""
    latitudes = sorted({lat for lat, _ in cluster})
    dems = []
    for _, dem in sorted(cluster):
        if dem not in dems:
            dems.append(dem)
    return latitudes, dems


def dem_paths_from(cluster, dems):
    """``{dem: (import_file, view_factor_file)}``, read from the shipped files.

    Read, not retyped: the stems are not uniform (``Pwyll_DEM_750m_SOCET`` next
    to ``Cilix_DEM_350m_SOCET`` next to ``Ridges_DEM_60m_SOCET``), and a
    plausible guess at them would produce a sweep that looks equivalent and
    runs a different DEM.
    """
    out = {}
    for dem in dems:
        path = next(p for (_, d), p in sorted(cluster.items()) if d == dem)
        config = yaml.safe_load(path.read_text())
        out[dem] = (get_dotted(config, "mesh.import_file"),
                    get_dotted(config, "radiation.view_factor_file"))
    return out


# --- the replacement ------------------------------------------------------------

def build_sweep(cluster, outdir, **kwargs):
    """The sweep that produces the same runs as the files in ``cluster``."""
    from ptm import Sweep
    latitudes, dems = axes_from(cluster)
    dem_paths = dem_paths_from(cluster, dems)
    return Sweep(
        cluster[(latitudes[0], dems[0])],
        grid={"mesh.latitude_override": latitudes, "dem": dems},
        label_axes=("dem",),
        derived={"mesh.import_file": lambda p: dem_paths[p["dem"]][0],
                 "radiation.view_factor_file": lambda p: dem_paths[p["dem"]][1]},
        name_from=lambda p: f"lat{p['mesh.latitude_override']:02d}_{p['dem']}",
        outdir=outdir, **kwargs)


REPLACEMENT_SOURCE = """\
Sweep("configs/europa_lat00_agenor.yaml",
      grid={"mesh.latitude_override": latitudes,   # [0, 15, 30, 45, 60, 75, 80, 85]
            "dem": dems},                          # ["agenor", ... , "ridges"]
      label_axes=("dem",),
      derived={"mesh.import_file":         lambda p: dem_paths[p["dem"]][0],
               "radiation.view_factor_file": lambda p: dem_paths[p["dem"]][1]},
      name_from=lambda p: f"lat{p['mesh.latitude_override']:02d}_{p['dem']}")"""


# --- the measurement ---------------------------------------------------------------

def measure(cluster, latitudes, dems):
    texts = {p: p.read_text().splitlines() for p in cluster.values()}
    total = sum(len(v) for v in texts.values())
    distinct = len({line for lines in texts.values() for line in lines})

    def changed(a, b):
        # max(removed, added), not their sum: a one-line substitution appears
        # in a unified diff as one '-' and one '+', and counting both doubles
        # every number in the audit's table.
        diff = list(difflib.unified_diff(texts[a], texts[b], n=0, lineterm=""))
        removed = sum(1 for x in diff
                      if x.startswith("-") and not x.startswith("---"))
        added = sum(1 for x in diff
                    if x.startswith("+") and not x.startswith("+++"))
        return max(removed, added)

    lat_pairs = [changed(cluster[(a, d)], cluster[(b, d)])
                 for d in dems for a, b in zip(latitudes, latitudes[1:])
                 if (a, d) in cluster and (b, d) in cluster]
    dem_pairs = [changed(cluster[(lat, a)], cluster[(lat, b)])
                 for lat in latitudes for a, b in zip(dems, dems[1:])
                 if (lat, a) in cluster and (lat, b) in cluster]
    return total, distinct, lat_pairs, dem_pairs


# --- the check ------------------------------------------------------------------------

def verify(sweep, cluster):
    """Every generated config against its shipped file, key by key.

    Returns ``(n_keys_compared, differences)``. ``output.directory`` is
    excluded by construction: pointing every run at its own directory under one
    output root is the whole point.

    ``radiation.view_factor_file`` is compared *the way the engine resolves
    it*: a relative value there is taken against the YAML file's own directory,
    not the working directory (``config_parser.cpp:507-517``). That is what
    makes the comparison meaningful, and it is also what makes it interesting
    -- see the note this prints.
    """
    differences, compared = [], 0
    for point in sweep:
        lat = point.params["mesh.latitude_override"]
        shipped_path = cluster[(lat, point.params["dem"])]
        flat_shipped = flatten_config(yaml.safe_load(shipped_path.read_text()))
        flat_generated = flatten_config(point.config())

        for key in sorted(set(flat_shipped) | set(flat_generated)):
            if key == "output.directory":
                continue
            compared += 1
            want, got = flat_shipped.get(key), flat_generated.get(key)
            if key == "radiation.view_factor_file" and want is not None:
                if not Path(want).is_absolute():
                    want = str(Path(shipped_path.parent / want))
                want = str(Path(want))
                got = str(Path(got)) if got else got
            if want != got:
                differences.append((shipped_path.name, key, want, got))
    return compared, differences


def report_differences(differences):
    """Print the differences, separating the one that is a finding."""
    vf = [d for d in differences if d[1] == "radiation.view_factor_file"]
    other = [d for d in differences if d[1] != "radiation.view_factor_file"]

    if other:
        print(f"CHECK FAILED       {len(other)} unexpected difference(s):")
        for name, key, want, got in other[:20]:
            print(f"  {name}: {key}: shipped {want!r} != generated {got!r}")
    if vf:
        shipped_dir = Path(vf[0][2]).parent
        print(f"note               "
              f"radiation.view_factor_file differs on {len(vf)}/{len(vf)} "
              f"points, and the difference is a finding about the shipped "
              f"files, not about the sweep:")
        print(f"  shipped   {vf[0][2]}")
        print(f"  generated {vf[0][3]}")
        print(f"  The shipped value is relative, and the engine resolves a "
              f"relative view_factor_file against the YAML's own directory "
              f"(config_parser.cpp:507-517), i.e. under configs/. That "
              f"directory does not exist in this checkout "
              f"({'it does' if shipped_dir.exists() else 'confirmed absent'}"
              f"), so those runs would rebuild the view-factor matrix instead "
              f"of loading the cached one. The sweep anchors a path supplied "
              f"through derived= to the working directory instead, which is "
              f"where data/external/ actually lives.")
    return bool(other)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the engine command for each point")
    parser.add_argument("--outdir", default="output/europa_lat_sweep")
    args = parser.parse_args()

    cluster = discover()
    if not cluster:
        raise SystemExit("no configs/europa_lat*_*.yaml in this checkout")
    latitudes, dems = axes_from(cluster)
    total, distinct, lat_pairs, dem_pairs = measure(cluster, latitudes, dems)
    replacement_lines = len(REPLACEMENT_SOURCE.splitlines())

    print(f"the cluster        {len(cluster)} files "
          f"({len(latitudes)} latitudes x {len(dems)} DEMs), {total} lines")
    print(f"distinct lines     {distinct} "
          f"({100.0 * distinct / total:.1f}% of the total)")
    print(f"latitude-pair diff {sorted(set(lat_pairs))} changed lines")
    print(f"DEM-pair diff      {sorted(set(dem_pairs))} changed lines")
    print(f"the replacement    {replacement_lines} lines of Python:")
    print()
    for line in REPLACEMENT_SOURCE.splitlines():
        print(f"    {line}")
    print()
    print(f"ratio              {total} YAML lines -> {replacement_lines} "
          f"Python lines ({total / replacement_lines:.0f}x)")

    sweep = build_sweep(cluster, Path(args.outdir).expanduser().resolve(),
                        quiet=True)
    compared, differences = verify(sweep, cluster)
    print()
    print(f"check              {compared} keys compared across "
          f"{len(sweep)} points; {compared - len(differences)} equal, "
          f"{len(differences)} different")
    failed = report_differences(differences) if differences else False
    print()
    print("Nothing here deletes anything. The 40 files still work; this is the "
          "measurement that says what retiring them would cost.")

    if args.dry_run:
        sweep.run(dry_run=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
