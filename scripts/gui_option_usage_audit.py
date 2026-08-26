#!/usr/bin/env python3
"""How many of the engine's config keys does a body actually vary?

This is the measurement behind ``docs/design/gui-body-first-redesign.md``. It
clusters every runnable config in ``configs/`` and ``examples/`` by planetary
body and, for each cluster, counts the keys that take more than one value
across that cluster's files. Keys that take exactly one value are candidate
body defaults; keys that vary are the candidate exposed set for that body's
surface in the GUI.

Two rules make the count mean what it says.

1. **Every config is completed from the engine's own default table**
   (``ptm.engine_defaults.ENGINE_DEFAULTS``, 184 keys) before anything is
   compared. Omitting a key and writing its default are the same run, so
   counting "this file does not mention the key" as variation would inflate
   every cluster -- measured, it inflated Moon-local from 30 varying science
   keys to 85.

2. **Bookkeeping keys are counted separately.** ``output.directory`` differs
   between any two runs by construction, and a view-factor cache path is
   derived from the mesh. Neither is a scientific choice, and every surface has
   to expose them whatever the body, so folding them in would overstate the
   body-specific set.

The ``examples/validation/`` configs are deliberate solver-permutation arms --
their whole purpose is to differ in ``radiation.use_hmatrix``,
``thermal.use_gpu`` and friends -- so the second arm reports the corpus without
them. That is the number a GUI surface should be sized against.

Usage::

    python3 scripts/gui_option_usage_audit.py [repo_root]
    python3 scripts/gui_option_usage_audit.py --keys      # list the keys too

Nothing is written and nothing is run.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import sys

import yaml

#: Keys that are not a scientific option of the body's surface. Three kinds:
#: output paths, which differ between any two runs by construction; machine
#: settings (thread counts, verbosity); and caches derived from the mesh.
#: ``simulation.body`` and ``generic_body.name`` are here as LABELS -- inside a
#: cluster the body is fixed by definition, and in the generic-body cluster the
#: two keys carry the same string as each other.
BOOKKEEPING = frozenset({
    "advanced.verbose",
    "generic_body.name",
    "illumination.bvh_cache_file",
    "illumination.positions_file",
    "output.complete_vtk_file",
    "output.depths_file",
    "output.directory",
    "output.location_data",
    "output.locations_file",
    "output.series_output",
    "output.thermal_vtk_file",
    "radiation.hmatrix_threads",
    "radiation.view_factor_file",
    "simulation.body",
    "thermal.threads",
})

#: Clusters, in the order the design document reports them.
CLUSTER_ORDER = ("Moon-global", "Moon-local", "Europa-global", "Europa-local",
                 "Mercury", "Asteroid/generic", "Unclassified")


def flatten(node, prefix=""):
    """Dotted leaf keys. A sequence is one leaf, compared as its JSON text."""
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            child = "%s.%s" % (prefix, key) if prefix else str(key)
            out.update(flatten(value, child))
    elif isinstance(node, list):
        out[prefix] = json.dumps(node, sort_keys=True, default=str)
    else:
        out[prefix] = node
    return out


def norm(value):
    """A comparable spelling. ``1`` and ``1.0`` are the same value here.

    Datetimes are rendered back to ISO text: PyYAML autotypes an UNQUOTED
    ``start_time: 2025-01-01T00:00:00`` into a ``datetime``, whose ``str()`` uses
    a space rather than a ``T`` and so could never equal the quoted spelling every
    other config uses. One unquoted timestamp would make ``simulation.start_time``
    count as varying everywhere -- the inflation rule 1 exists to prevent.
    """
    if value is None:
        return "null"                       # distinct from the string "none"
    if isinstance(value, bool):
        return "bool:%s" % value
    if isinstance(value, (int, float)):
        return "num:%.12g" % float(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return "str:%s" % value.isoformat()
    return "str:%s" % value


def classify(cfg, default_body=""):
    """Body cluster for one config.

    ``local`` means the run uses an IMPORTED mesh, and nothing else. That is
    exactly the engine's own ``is_custom_mesh``, which is assigned from the
    mesh file alone::

        thermal_config.custom_properties["is_custom_mesh"]
            = (mesh_file.empty()) ? "0" : "1";        # ptm_thermal_sim.cpp:17036

    ``mesh.use_topographic`` deliberately does NOT count. A topographic run is
    still the procedural icosphere at ``mesh.resolution``
    (``ptm_thermal_sim.cpp:15115``) with LOLA relief applied to its vertices, so
    it keeps the ``20 * 4^level`` face count and the per-face albedo binary
    still matches. Treating it as local put ``examples/moon_global_topo_res5.yaml``
    -- whose own header calls it "the GUI's default global-Moon setup" -- in the
    local cluster, one key away from its documented flat twin.
    """
    body = str((cfg.get("simulation") or {}).get("body", "")).strip().lower()
    if not body:
        # Rule 1 applies to the body key too: omitting it and writing the
        # engine default are the same run.
        body = str(default_body).strip().lower()
    mesh = cfg.get("mesh") or {}
    local = bool(str(mesh.get("import_file", "") or "").strip())
    if body == "moon":
        return "Moon-local" if local else "Moon-global"
    if body == "europa":
        return "Europa-local" if local else "Europa-global"
    if body == "mercury":
        return "Mercury"
    if (cfg.get("generic_body") or {}) or body:
        return "Asteroid/generic"
    return "Unclassified"


def collect(root):
    """Every runnable config under configs/ and examples/, path-sorted."""
    found = []
    tops = [t for t in ("configs", "examples")
            if os.path.isdir(os.path.join(root, t))]
    if not tops:
        # An empty walk would still print a complete, authoritative-looking
        # report over zero files. Refuse instead.
        raise SystemExit(
            "no configs/ or examples/ directory under %s -- is that a PTM3D "
            "checkout?" % root)
    for top in tops:
        for dirpath, _dirnames, filenames in os.walk(os.path.join(root, top)):
            for name in sorted(filenames):
                if not name.endswith((".yaml", ".yml")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path) as handle:
                        cfg = yaml.safe_load(handle)
                except Exception as exc:            # noqa: BLE001
                    print("  skipped %s (%s)" % (path, exc), file=sys.stderr)
                    continue
                if not isinstance(cfg, dict):
                    continue
                # A camera or survey template is not a run.
                if "simulation" not in cfg and "generic_body" not in cfg:
                    continue
                found.append((os.path.relpath(path, root), cfg))
    return sorted(found)


def complete(flat, defaults):
    """All engine keys for one config; absent means the engine default."""
    full = {k: norm(flat.get(k, d)) for k, d in defaults.items()}
    return full, sorted(set(flat) - set(defaults))


def summarize(members, defaults):
    values = collections.defaultdict(set)
    for _path, full in members:
        for key, value in full.items():
            values[key].add(value)
    vary = sorted(k for k, s in values.items() if len(s) > 1)
    return (values,
            [k for k in vary if k not in BOOKKEEPING],
            [k for k in vary if k in BOOKKEEPING])


def line(label, members, defaults, values_out=None):
    values, science, book = summarize(members, defaults)
    total = len(defaults)
    print("%-34s n=%-3d  VARY %-4d (science %-3d, bookkeeping %-2d)  "
          "CONSTANT %d of %d"
          % (label, len(members), len(science) + len(book), len(science),
             len(book), total - len(science) - len(book), total))
    if values_out is not None:
        values_out.extend((k, len(values[k])) for k in science)
    return science


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=os.getcwd())
    parser.add_argument("--keys", action="store_true",
                        help="list each cluster's varying science keys")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    src_python = os.path.join(root, "src", "python")
    if not os.path.isdir(src_python):
        raise SystemExit("no src/python under %s -- is that a PTM3D checkout?"
                         % root)
    sys.path.insert(0, src_python)
    import ptm
    from ptm.engine_defaults import ENGINE_DEFAULTS
    # The insert above is a no-op if an installed or otherwise-importable `ptm`
    # wins, and on this machine that resolves to a DIFFERENT checkout -- which
    # would measure this tree's configs against another tree's key table and
    # print a confident, wrong report. Same trap as pytest in a worktree.
    if not os.path.realpath(ptm.__file__).startswith(os.path.realpath(root)):
        raise SystemExit(
            "imported ptm from %s, which is outside %s. Set PYTHONPATH=%s"
            % (ptm.__file__, root, src_python))

    configs = collect(root)
    clusters = collections.defaultdict(list)
    outside = set()
    default_body = ENGINE_DEFAULTS.get("simulation.body", "")
    for path, cfg in configs:
        full, unknown = complete(flatten(cfg), ENGINE_DEFAULTS)
        outside |= set(unknown)
        clusters[classify(cfg, default_body)].append((path, full))

    print("CORPUS: %d runnable configs under configs/ + examples/" % len(configs))
    print("ENGINE KEY SURFACE: %d keys" % len(ENGINE_DEFAULTS))
    if outside:
        print("KEYS USED BY SHIPPED CONFIGS THAT THE PARSER DOES NOT READ "
              "(%d):" % len(outside))
        for key in sorted(outside):
            print("    %s" % key)
    print()

    def ordered():
        return [c for c in CLUSTER_ORDER if c in clusters] + \
            [c for c in sorted(clusters) if c not in CLUSTER_ORDER]

    print("--- ARM A: every shipped config ---")
    for cluster in ordered():
        line(cluster, clusters[cluster], ENGINE_DEFAULTS)
    print()

    print("--- ARM B: without examples/validation/ (solver-permutation arms) ---")
    trimmed = {}
    for cluster in ordered():
        members = [m for m in clusters[cluster]
                   if not m[0].startswith("examples/validation/")]
        trimmed[cluster] = members
        if members:
            line(cluster, members, ENGINE_DEFAULTS)
    print()

    every = [m for members in trimmed.values() for m in members]
    _values, science, book = summarize(every, ENGINE_DEFAULTS)
    print("--- ARM C: the corpus as a whole ---")
    print("Keys that take a second value ANYWHERE in %d configs: %d of %d"
          % (len(every), len(science) + len(book), len(ENGINE_DEFAULTS)))
    print("Keys that never take a second value anywhere:        %d of %d"
          % (len(ENGINE_DEFAULTS) - len(science) - len(book),
             len(ENGINE_DEFAULTS)))
    print()

    print("--- ARM D: per body, global and local clusters unioned ---")
    for body, parts in (("Moon", ("Moon-global", "Moon-local")),
                        ("Europa", ("Europa-global", "Europa-local")),
                        ("Mercury", ("Mercury",)),
                        ("Asteroid", ("Asteroid/generic",))):
        members = [m for part in parts for m in trimmed.get(part, [])]
        if members:
            line(body, members, ENGINE_DEFAULTS)
        elif any(clusters.get(part) for part in parts):
            # Configs exist, but all of them are validation arms.
            print("%-34s n=0    only examples/validation/ arms" % body)
        else:
            print("%-34s n=0    NO SHIPPED CONFIGS" % body)

    if args.keys:
        print()
        print("--- varying science keys, per cluster (arm B) ---")
        for cluster in ordered():
            if not trimmed.get(cluster):
                continue
            print()
            keys = []
            line(cluster, trimmed[cluster], ENGINE_DEFAULTS, keys)
            for key, count in keys:
                print("        %-52s %d values" % (key, count))


if __name__ == "__main__":
    main()
