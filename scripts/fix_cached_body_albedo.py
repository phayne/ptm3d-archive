#!/usr/bin/env python3
"""Rewrite cached ``body_config.yaml`` albedos from geometric p_V to Bond.

WHAT WAS WRONG
--------------

``generic_body.surface.albedo`` is a BOND albedo to the engine: it computes
``1 - albedo`` and multiplies the solar constant by it
(``thermal_model.cpp:2906``).  ``BodyConfigWriter._default_surface`` wrote
SBDB's GEOMETRIC albedo p_V into that field, emitted a ``logger.warning`` about
it, and wrote it anyway.  Every ``body_config.yaml`` in
``data/horizons_cache/bodies`` was produced that way, and the Asteroid card
copied the value straight into a run configuration.

A warning in a log is not a guard.  On 2026-08-27 a generated (101955) Bennu
configuration reached a run carrying ``surface.albedo: 0.044``, which is
Bennu's p_V; its Bond albedo is 0.0118.

WHAT THIS DOES
--------------

For each cached body it converts ``surface.albedo`` with
``horizons.albedo.bond_albedo_from_pv`` -- A_B = p_V x (0.290 + 0.684 G),
Bowell et al. 1989 -- and rewrites the file with:

    albedo:                    the Bond albedo
    albedo_type:               "bond"          <- so no later reader guesses
    albedo_geometric_pv:       the p_V it came from
    albedo_slope_parameter_g:  the G used
    albedo_provenance:         one line saying which, and whether G was assumed

G comes from SBDB when ``--online`` is given (the default), and from the IAU
default 0.15 when it is not, or when SBDB publishes none.  Which one was used
is recorded per body; it is never silently substituted.

IDEMPOTENT.  A file that already says ``albedo_type: bond`` is left alone and
reported as already done.

⚠️ THIS CHANGES WHAT A CACHED BODY MEANS, and one consumer wants the OTHER
convention.  ``scripts/ema_campaign_driver.py`` reads the same field
(``ema_campaign_driver.py:369``: ``albedo=float(gb['surface']['albedo'])``) and
uses it as the GEOMETRIC albedo p -- deliberately; its comment at lines 79-83
says so, and its Hapke single-scattering albedos ``w`` are calibrated to
reproduce those p values.  Running this over the EMA campaign's cache would
silently change the number that campaign is built on.  So:

  * the default target is the repository's own cache
    (``<repo>/data/horizons_cache``), which is what the GUI reads;
  * any other target must be named explicitly with ``--cache-dir``;
  * ``--dry-run`` prints the table and writes nothing.

Usage::

    python3 scripts/fix_cached_body_albedo.py --dry-run
    python3 scripts/fix_cached_body_albedo.py
    python3 scripts/fix_cached_body_albedo.py --offline      # IAU default G
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "python"))

from horizons.albedo import bond_albedo_from_pv          # noqa: E402

SBDB_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"


def sbdb_slope_parameter(query, timeout=20):
    """SBDB's fitted G for this body, or ``None`` if it publishes none.

    ``None`` is a real answer here and is NOT the same as 0.15: the IAU default
    is an assumption and the caller has to be able to record that it made one.
    """
    url = SBDB_API_URL + "?" + urllib.parse.urlencode(
        {"sstr": str(query), "phys-par": "1"})
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for parameter in payload.get("phys_par", []) or []:
        if str(parameter.get("name")) == "G":
            try:
                return float(parameter.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def body_query(generic, directory_name):
    """What to ask SBDB about: the asteroid number, which is unambiguous."""
    name = str(generic.get("name") or directory_name).strip()
    head = name.replace("_", " ").split()
    if head and head[0].isdigit():
        return head[0]
    return name


def rewrite(path, online=True, dry_run=False):
    """One cached config. Returns a row dict for the report."""
    import yaml

    text = path.read_text()
    data = yaml.safe_load(text) or {}
    generic = data.get("generic_body")
    if not isinstance(generic, dict):
        return {"body": path.parent.name, "status": "no generic_body section"}
    surface = generic.get("surface")
    if not isinstance(surface, dict):
        return {"body": path.parent.name, "status": "no surface block"}

    row = {"body": str(generic.get("name") or path.parent.name),
           "path": str(path)}
    declared = str(surface.get("albedo_type") or "").strip().lower()
    if declared == "bond":
        row["status"] = "already bond, left alone"
        row["albedo"] = surface.get("albedo")
        return row

    p_v = surface.get("albedo")
    row["p_v"] = p_v
    slope, slope_source = None, "IAU default (offline)"
    if online:
        query = body_query(generic, path.parent.name)
        try:
            slope = sbdb_slope_parameter(query)
        except Exception as exc:                             # noqa: BLE001
            row["status"] = f"SBDB lookup failed ({exc}); NOT rewritten"
            return row
        slope_source = ("SBDB" if slope is not None
                        else "IAU default (SBDB publishes none)")

    conversion = bond_albedo_from_pv(p_v, slope)
    if conversion is None:
        row["status"] = f"albedo {p_v!r} is not a usable p_V; NOT rewritten"
        return row

    row.update({
        "bond": round(conversion["bond"], 6),
        "G": conversion["slope_parameter"],
        "G_source": slope_source,
        "q": round(conversion["q"], 4),
        "status": "converted" if not dry_run else "would convert",
    })
    if dry_run:
        return row

    surface["albedo"] = round(conversion["bond"], 6)
    surface["albedo_type"] = "bond"
    surface["albedo_geometric_pv"] = conversion["geometric_albedo"]
    surface["albedo_slope_parameter_g"] = conversion["slope_parameter"]
    surface["albedo_provenance"] = (
        conversion["provenance"]
        + f" Rewritten from the geometric value by "
          f"scripts/fix_cached_body_albedo.py on "
          f"{datetime.now().strftime('%Y-%m-%d')}; G from {slope_source}.")

    # Written to a scratch name and moved into place, so an interrupted run
    # cannot leave a half-written body_config.yaml where a whole one was.
    scratch = path.with_suffix(".yaml.part")
    scratch.write_text(yaml.dump(data, default_flow_style=False,
                                 sort_keys=False))
    scratch.replace(path)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--cache-dir", default=str(REPO / "data" / "horizons_cache"),
        help="the horizons_cache to rewrite (default: this repository's)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the table and write nothing")
    parser.add_argument("--offline", action="store_true",
                        help="do not ask SBDB for G; assume the IAU default")
    args = parser.parse_args(argv)

    bodies = Path(args.cache_dir) / "bodies"
    if not bodies.is_dir():
        parser.error(f"{bodies} is not a directory")

    rows = [rewrite(path, online=not args.offline, dry_run=args.dry_run)
            for path in sorted(bodies.glob("*/body_config.yaml"))]

    print(f"{'body':<32} {'p_V':>8} {'->':^4} {'A_B':>10} {'G':>8}  "
          f"{'G source':<34} status")
    for row in rows:
        print(f"{row.get('body', '?'):<32} "
              f"{row.get('p_v', ''):>8} {'->':^4} {row.get('bond', ''):>10} "
              f"{row.get('G', ''):>8}  {row.get('G_source', ''):<34} "
              f"{row.get('status', '')}")
    changed = sum(1 for r in rows if r.get("status") == "converted")
    print(f"\n{changed} of {len(rows)} cached bodies rewritten in "
          f"{args.cache_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
