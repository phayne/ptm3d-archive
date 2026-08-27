"""
Built-in dataset catalog for the PTM3D GUI.

Lets the user pick "Built-in (recommended)" topography / albedo instead of
navigating to files by hand. Resolution follows the SAME contract as the
engine (config_parser.cpp, "PTM_DATA_ROOT resolution contract"):

    absolute config path wins  ->  $PTM_DATA_ROOT for "data/..." references
    ->  the repository data/ directory (project root; the GUI uses the
        executable-derived root that gui_core.preflight computes, which is
        the directory the engine subprocess runs in).

Dataset names for fetch-on-missing are NOT hardcoded: they are derived from
data/data_manifest.json by matching each dataset's files[].unpack_to prefix
against the expected on-disk path (e.g. "data/topo/moon/" -> moon-topo).

Engine file-naming ground truth:
  topography (mesh_utils.cpp:checkForTopographicMesh):
      <topo_dir>/<body_lc>/topo_<body_lc>_ico<N>.bin   (modern)
      <topo_dir>/topo_<body_lc>_ico<N>.bin             (modern, flat)
      <topo_dir>/<Body>_Topo_res<N>.bin                (legacy variants)
  albedo (albedo_manager.cpp:tryLoadBestMap):
      <albedo_dir>/<Body>_Albedo_res<N>.bin, tried res 8 -> 3, first hit
      wins (FITS variants are tried first; built-in mode relies on the
      binary per-face maps that ship in the albedo-maps dataset).

Only bodies listed in the catalogs below ship built-in data today:
topography = Moon; albedo = Moon + Europa.
"""

import json
import os
from pathlib import Path

from gui_core.preflight import is_dataless, resolve_data_path

# Canonical RELATIVE references written into the YAML when Built-in mode is
# selected. Keeping them relative makes the saved config portable: the
# engine resolves them per its own contract on whatever machine runs it.
TOPO_CONFIG_VALUE = "data/topo"
ALBEDO_CONFIG_VALUE = "data/albedo_maps"

# Bodies with shipped built-in data (lowercase). Kept as explicit sets so the
# UI can gray out "Built-in" for bodies without data; the fetch DATASET NAME
# still comes from the manifest, not from here.
TOPO_BODIES = {"moon"}
ALBEDO_BODIES = {"moon", "europa"}

MANIFEST_RELPATH = Path("data") / "data_manifest.json"
PLACEHOLDER = "ZENODO_PLACEHOLDER"


def resolve_data_reference(value, project_root, env=None):
    """Resolve a config data-path reference exactly like the engine
    (single implementation lives in gui_core.preflight.resolve_data_path)."""
    return resolve_data_path(value, project_root, env)


def manifest_path(project_root, env=None):
    """The manifest file the GUI (and fetch) should use: the PTM_DATA_ROOT
    copy when present, else the repository copy."""
    primary = resolve_data_reference(
        str(MANIFEST_RELPATH).replace(os.sep, "/"), project_root, env)
    if primary.exists():
        return primary
    return Path(project_root) / MANIFEST_RELPATH


def read_manifest(project_root, env=None):
    """Load data_manifest.json (under PTM_DATA_ROOT if set, else the repo
    data/). Returns {} when unavailable/unparseable -- callers degrade to
    'dataset name unknown' rather than crashing the GUI."""
    try:
        with open(manifest_path(project_root, env)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def base_url_is_placeholder(manifest):
    return PLACEHOLDER in str(manifest.get("base_url", PLACEHOLDER))


def dataset_for_data_path(manifest, rel_path):
    """Map a 'data/...' path to the manifest dataset that provides it, via
    files[].unpack_to prefixes. Returns dataset name or ''."""
    rel = str(rel_path).replace("\\", "/")
    if not rel.endswith("/"):
        rel_dir = rel + "/"
    else:
        rel_dir = rel
    best = ("", 0)
    for name, ds in (manifest.get("datasets", {}) or {}).items():
        for fe in ds.get("files", []) or []:
            prefix = str(fe.get("unpack_to", "")).replace("\\", "/")
            if not prefix:
                continue
            if rel.startswith(prefix) or rel_dir.startswith(prefix):
                if len(prefix) > best[1]:
                    best = (name, len(prefix))
    return best[0]


# ---------------------------------------------------------------------------
# Per-dataset status
# ---------------------------------------------------------------------------

STATE_OK = "ok"                  # file present and hydrated
STATE_MISSING = "missing"        # not on disk (fetch can provide it)
STATE_DATALESS = "dataless"      # on disk but evicted cloud placeholder
STATE_UNSUPPORTED = "unsupported"  # body ships no built-in data of this kind


class DatasetStatus:
    def __init__(self, state, resolved, detail="", dataset=""):
        self.state = state          # one of the STATE_* constants
        self.resolved = resolved    # Path the engine would load (or look for)
        self.detail = detail        # human-readable note for the UI label
        self.dataset = dataset      # manifest dataset name for fetching

    @property
    def available(self):
        return self.state == STATE_OK


def _classify(path):
    if not path.exists():
        return STATE_MISSING
    if is_dataless(path):
        return STATE_DATALESS
    return STATE_OK


def _proper(body):
    b = str(body).strip()
    return b[:1].upper() + b[1:].lower() if b else b


def topo_status(body, resolution, project_root, env=None, manifest=None):
    """Status of built-in topography for (body, resolution)."""
    body_lc = str(body).strip().lower()
    if body_lc not in TOPO_BODIES:
        return DatasetStatus(
            STATE_UNSUPPORTED, None,
            f"No built-in topography for {_proper(body)} "
            "(only the Moon ships topography today)")
    root_dir = resolve_data_reference(TOPO_CONFIG_VALUE, project_root, env)
    fname = f"topo_{body_lc}_ico{int(resolution)}.bin"
    candidates = [
        root_dir / body_lc / fname,                                  # modern
        root_dir / fname,                                            # flat
        root_dir / f"{_proper(body)}_Topo_res{int(resolution)}.bin",  # legacy
    ]
    manifest = manifest if manifest is not None else read_manifest(project_root, env)
    dataset = dataset_for_data_path(
        manifest, f"{TOPO_CONFIG_VALUE}/{body_lc}/{fname}")
    for cand in candidates:
        state = _classify(cand)
        if state != STATE_MISSING:
            detail = ("" if state == STATE_OK
                      else "present but DATALESS (cloud placeholder) -- "
                           "hydrate or re-fetch")
            return DatasetStatus(state, cand, detail, dataset)
    return DatasetStatus(
        STATE_MISSING, candidates[0],
        f"not on disk (dataset '{dataset or 'unknown'}')", dataset)


#: The latitude/longitude maps the engine interpolates onto an ARBITRARY mesh,
#: in the order AlbedoManager::tryLoadBestMap tries them. Imported from the
#: generated engine table rather than retyped, so a change to the engine's
#: naming shows up here.
try:
    from ptm.body_table import ALBEDO_FITS_PATTERNS
except ImportError:                                   # pragma: no cover
    ALBEDO_FITS_PATTERNS = ("{Body}-0.5deg-BondAlbedo.fits",)


def albedo_status(body, resolution, project_root, env=None, manifest=None,
                  import_file=None):
    """Status of the built-in albedo map for (body, resolution, mesh).

    Mirrors the engine: prefers <Body>_Albedo_res<meshres>.bin, but the
    engine will fall back to the highest available res 8 -> 3, so any
    resolution present counts as available (with a note).

    ``import_file`` is the mesh, and it changes the answer. The engine picks
    the albedo branch on ``is_custom_mesh``, which is the mesh file alone
    (``ptm_thermal_sim.cpp:17036``), and the per-face binary is matched to its
    icosphere BY FACE COUNT -- the load is refused outright on a size mismatch
    (``:2471``). So on an imported mesh the per-face map cannot apply at all,
    and what does apply is a latitude/longitude FITS map, which the engine
    interpolates onto any mesh (``:2329``).

    Ignoring the mesh is why this function used to promise a per-face map for a
    lunar crater configuration that the engine would then reject, falling back
    to a single albedo and saying so only on stdout (BUGS.md
    GUI-ALBEDO-STATUS-IGNORES-MESH). Callers that genuinely have no mesh in
    hand may still omit the argument; callers that do have one must pass it.
    """
    body_lc = str(body).strip().lower()
    if body_lc not in ALBEDO_BODIES:
        return DatasetStatus(
            STATE_UNSUPPORTED, None,
            f"No built-in albedo map for {_proper(body)} "
            "(Moon and Europa ship albedo maps today)")
    root_dir = resolve_data_reference(ALBEDO_CONFIG_VALUE, project_root, env)
    proper = _proper(body)

    if str(import_file or "").strip():
        manifest = manifest if manifest is not None else read_manifest(
            project_root, env)
        for pattern in ALBEDO_FITS_PATTERNS:
            candidate = root_dir / pattern.format(Body=proper)
            state = _classify(candidate)
            dataset = dataset_for_data_path(
                manifest, f"{ALBEDO_CONFIG_VALUE}/{candidate.name}")
            if state == STATE_OK:
                return DatasetStatus(
                    STATE_OK, candidate,
                    "a latitude/longitude map, interpolated onto the imported "
                    "mesh (the per-face map is matched by face count and "
                    "cannot apply to an imported mesh)", dataset)
            if state == STATE_DATALESS:
                return DatasetStatus(
                    STATE_DATALESS, candidate,
                    "present but DATALESS (cloud placeholder) -- hydrate or "
                    "re-fetch", dataset)
        first = root_dir / ALBEDO_FITS_PATTERNS[0].format(Body=proper)
        return DatasetStatus(
            STATE_MISSING, first,
            "this run uses an imported mesh, so the per-face albedo map "
            "cannot apply -- it is matched to a procedural sphere by face "
            "count. A latitude/longitude map would apply and is not on disk, "
            "so every face takes the single Bond albedo from Properties.",
            dataset_for_data_path(manifest,
                                  f"{ALBEDO_CONFIG_VALUE}/{first.name}"))
    exact = root_dir / f"{proper}_Albedo_res{int(resolution)}.bin"
    manifest = manifest if manifest is not None else read_manifest(project_root, env)
    dataset = dataset_for_data_path(
        manifest, f"{ALBEDO_CONFIG_VALUE}/{exact.name}")
    state = _classify(exact)
    if state == STATE_OK:
        return DatasetStatus(STATE_OK, exact, "", dataset)
    if state == STATE_DATALESS:
        return DatasetStatus(
            STATE_DATALESS, exact,
            "present but DATALESS (cloud placeholder) -- hydrate or re-fetch",
            dataset)
    # engine fallback order: res 8 down to 3, first existing wins
    for res in range(8, 2, -1):
        cand = root_dir / f"{proper}_Albedo_res{res}.bin"
        st = _classify(cand)
        if st == STATE_OK:
            return DatasetStatus(
                STATE_OK, cand,
                f"res{int(resolution)} map not present; engine will load "
                f"res{res} (its standard fallback)", dataset)
        if st == STATE_DATALESS:
            return DatasetStatus(
                STATE_DATALESS, cand,
                "present but DATALESS (cloud placeholder) -- hydrate or "
                "re-fetch", dataset)
    return DatasetStatus(
        STATE_MISSING, exact,
        f"not on disk (dataset '{dataset or 'unknown'}')", dataset)


def fetch_command(dataset, project_root, base_url=None, env=None):
    """The exact fetch invocation (list form) for a dataset. Passes the
    SAME manifest the GUI read (--manifest), so a PTM_DATA_ROOT-local
    manifest override is honored by the fetch too."""
    import sys as _sys
    env = env if env is not None else os.environ
    cmd = [_sys.executable,
           str(Path(project_root) / "scripts" / "fetch_data.py"),
           "--manifest", str(manifest_path(project_root, env)),
           "--dataset", dataset]
    if base_url:
        cmd += ["--base-url", base_url]
    data_root = env.get("PTM_DATA_ROOT", "")
    if data_root:
        cmd += ["--dest", data_root]
    return cmd
