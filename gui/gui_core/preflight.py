"""
Pre-flight validation for PTM3D GUI simulation launches.

Motivation (GUI review 2026-07-29, blockers B1/B2/B4): the GUI previously
launched ptm_thermal_sim with no path validation at all. A config pointing at
cloud-synced (OneDrive) directories containing dehydrated ("dataless") files
hangs the engine indefinitely in a file read, indistinguishable from
computation. This module performs cheap, stat-only checks BEFORE launch:

  * resolve every path field to an absolute path against the project root
  * existence / readability of every referenced file and directory
  * dataless scan: st_blocks == 0 while st_size > 0 means a cloud placeholder
    whose content is NOT on local disk -- reading it blocks on network fetch
    (stat itself never triggers hydration, so this scan is safe)
  * cloud-path detection (/Library/CloudStorage/) with explicit user override
  * output directory free-disk check
  * radiation view-factor guard: file exists and its face count matches the
    mesh (PTM-19: an accidental VF rebuild from the current mesh is forbidden)
  * executable discovery with md5 + mtime reported to the console

All filesystem probing is stat/os.scandir only -- nothing here opens data
files for reading except the first 24 bytes of a .vf/.hmat header and a PLY
text header, both of which are checked for datalessness first.

Env-completeness checking follows scripts/ptm_run_guard.py (Phase R, R3a):
if that module is importable it is reused; otherwise a compatible local
fallback is used so the GUI works on branches that predate it.
"""

import hashlib
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Recommended engine environment (the "recommended-ON" physics fix set).
#
# These env-gated fixes are default-OFF in the engine for byte-identical
# reproducibility of old runs, but every production campaign runs with them
# ON. GUI runs get them by default; the Engine Flags table in the
# configuration panel can override or extend them (config key "engine_env").
#
#   PTM_ILLUM_TRUE_NORMALS=1  flipped-normal overhang shadowing fix (PTM-13
#                             Phase-2, commit dd09cb8; Fermi gate PASS)
#   PTM_ILLUM_RISE_HABS=0     grazing rise-floor leak fix: h_abs = 0 kills
#                             100% of leaks with zero shadow acne (same gate)
#   PTM_ILLUM_SEAL_HOLES=1    PTM-18 interior mesh-hole penumbra flash fix
#   PTM_EQ_PERFACE_FLUX=1     per-face flux equilibration
#
# Unknown variables are ignored by older engine builds, so this set is safe
# to inject unconditionally.
# --------------------------------------------------------------------------
RECOMMENDED_ENGINE_ENV = {
    "PTM_ILLUM_TRUE_NORMALS": "1",
    "PTM_ILLUM_RISE_HABS": "0",
    "PTM_ILLUM_SEAL_HOLES": "1",
    "PTM_EQ_PERFACE_FLUX": "1",
}

CLOUD_PATH_MARKERS = ("/Library/CloudStorage/",)

# Config fields that hold a single file path: (dotted key, required-if callable)
_FILE_FIELDS = [
    "mesh.import_file",
    "illumination.albedo_map",
    "illumination.positions_file",
    "thermal.thermal_inertia_map",
    "thermal.h_parameter_map",
    "thermal.heat_flow_mask_file",
    "thermal.initial_state_file",
    "output.locations_file",
    "output.depths_file",
    "generic_body.config_file",
    "generic_body.spice_metakernel",
]

# Config fields that hold a directory that will be READ during the run.
_DATA_DIR_FIELDS = [
    "mesh.topo_directory",
    "illumination.albedo_maps_directory",
    "thermal.property_maps_directory",
]

# Caps so a huge directory can't freeze the GUI thread. stat-only, so even
# the capped scan is fast (~10k stats/s on local disk).
_SCAN_MAX_ENTRIES = 20000
_SCAN_MAX_SECONDS = 10.0
_MIN_DISK_ERROR_GB = 1.0
_MIN_DISK_WARN_GB = 10.0


@dataclass
class PreflightReport:
    """Result of a pre-flight run. errors block launch; cloud_paths and
    vf_rebuild_warning block unless the user explicitly overrides."""
    errors: list = field(default_factory=list)       # hard blockers
    dataless: list = field(default_factory=list)     # (field, path) cloud placeholders
    cloud_paths: list = field(default_factory=list)  # (field, path) on cloud storage
    warnings: list = field(default_factory=list)     # non-blocking
    info: list = field(default_factory=list)         # console notes (exe hash etc.)
    vf_rebuild_warning: str = ""                     # PTM-19 guard text, if triggered
    executable: Path = None
    project_root: Path = None

    @property
    def ok(self):
        return not self.errors and not self.dataless

    def summary_lines(self):
        out = []
        for e in self.errors:
            out.append(f"ERROR: {e}")
        for f_, p in self.dataless:
            out.append(f"DATALESS: {f_}: {p}")
        for f_, p in self.cloud_paths:
            out.append(f"CLOUD: {f_}: {p}")
        if self.vf_rebuild_warning:
            out.append(f"VF-GUARD: {self.vf_rebuild_warning}")
        for w in self.warnings:
            out.append(f"WARNING: {w}")
        for i in self.info:
            out.append(f"INFO: {i}")
        return out


def is_dataless(path):
    """True if `path` is a cloud placeholder: nonzero logical size but zero
    blocks on local disk. stat() does not trigger hydration."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_blocks == 0 and st.st_size > 0


def is_cloud_path(path):
    p = str(Path(path).resolve()) if path else ""
    return any(marker in p for marker in CLOUD_PATH_MARKERS)


def scan_dir_for_dataless(directory, max_entries=_SCAN_MAX_ENTRIES,
                          max_seconds=_SCAN_MAX_SECONDS):
    """Walk `directory` (stat-only) and return (dataless_files, truncated)."""
    bad = []
    n = 0
    t0 = time.monotonic()
    truncated = False
    stack = [Path(directory)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    n += 1
                    if n > max_entries or (time.monotonic() - t0) > max_seconds:
                        truncated = True
                        return bad, truncated
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            if st.st_blocks == 0 and st.st_size > 0:
                                bad.append(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue
    return bad, truncated


def get_config_value(config, dotted_key, default=None):
    cur = config
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_path(value, project_root):
    """Resolve a config path against the project root (the directory the
    engine subprocess will run in)."""
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = Path(project_root) / p
    return p


def resolve_data_path(value, project_root, env=None):
    """Resolve a DATA-path config value exactly like the engine's
    PTM_DATA_ROOT contract (config_parser.cpp): an absolute path wins; a
    relative "data/..." reference resolves under $PTM_DATA_ROOT when set
    (PTM_DATA_ROOT points AT the data directory); otherwise under the
    project root. Non-"data/..." relative values fall back to the project
    root like resolve_path."""
    env = env if env is not None else os.environ
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    value_s = str(value).replace("\\", "/")
    data_root = env.get("PTM_DATA_ROOT", "")
    if data_root and (value_s == "data" or value_s.startswith("data/")):
        rest = value_s[5:] if value_s.startswith("data/") else ""
        return (Path(data_root) / rest) if rest else Path(data_root)
    return Path(project_root) / p


def find_project_root(executable=None):
    """Derive the repo/project root from the engine executable location
    (<root>/build/bin/ptm_thermal_sim). Fallback: two levels above gui_core."""
    if executable:
        exe = Path(executable).resolve()
        if exe.parent.name == "bin" and exe.parent.parent.name.startswith("build"):
            return exe.parent.parent.parent
        return exe.parent
    return Path(__file__).resolve().parent.parent.parent


def executable_fingerprint(exe_path):
    """md5 + mtime of the engine binary, for the console banner."""
    exe_path = Path(exe_path)
    h = hashlib.md5()
    with open(exe_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    mtime = time.strftime("%Y-%m-%d %H:%M:%S",
                          time.localtime(exe_path.stat().st_mtime))
    return h.hexdigest(), mtime


def read_vf_face_count(vf_path):
    """Read the face count from a view-factor file header without loading it.

    Two on-disk formats (radiation_exchange.cpp:loadViewFactors):
      H-matrix: u32 magic 0x484D4154 'HMAT', u32 version, u64 n_rows, u64 n_cols
      Sparse:   u32 magic 0x5646464D 'VFFM', u32 version, u64 nrows, u64 ncols, u64 nnz
    Returns (n_faces, format_name) or (None, reason).
    """
    try:
        with open(vf_path, "rb") as f:
            header = f.read(24)
    except OSError as e:
        return None, f"unreadable ({e})"
    if len(header) < 24:
        return None, "file too short for a VF header"
    magic, _version = struct.unpack("<II", header[:8])
    n_rows, _n_cols = struct.unpack("<QQ", header[8:24])
    if magic == 0x484D4154:
        return int(n_rows), "H-matrix"
    if magic == 0x5646464D:
        return int(n_rows), "sparse"
    return None, "unrecognized magic (not a .vf/.hmat file)"


def expected_face_count(config):
    """Expected mesh face count, when derivable. Icosphere: 20 * 4^res.
    For an imported PLY, parse the text header (safe: datalessness is checked
    by the caller first). Returns int or None."""
    mesh = config.get("mesh", {}) if isinstance(config, dict) else {}
    import_file = str(mesh.get("import_file", "") or "")
    use_topo = bool(mesh.get("use_topographic", False))
    if import_file:
        if import_file.lower().endswith(".ply"):
            return _ply_face_count(import_file)
        return None
    if use_topo:
        return None  # topo tile face count is not derivable here
    res = mesh.get("resolution", None)
    try:
        res = int(res)
    except (TypeError, ValueError):
        return None
    if 0 <= res <= 12:
        return 20 * (4 ** res)
    return None


def _ply_face_count(ply_path):
    """Face count from a PLY header (text even for binary PLY)."""
    try:
        with open(ply_path, "rb") as f:
            head = f.read(65536)
    except OSError:
        return None
    try:
        text = head.decode("ascii", errors="replace")
    except Exception:
        return None
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "face":
            try:
                return int(parts[2])
            except ValueError:
                return None
        if line.strip() == "end_header":
            break
    return None


def check_env_completeness(env, required=None):
    """R3a env-completeness gate. Reuses scripts/ptm_run_guard.py when
    available; otherwise a compatible local check. Returns a list of
    problem strings (empty = complete)."""
    required = required if required is not None else RECOMMENDED_ENGINE_ENV
    try:
        sys.path.insert(0, str(find_project_root() / "scripts"))
        import ptm_run_guard  # noqa: F401
        if hasattr(ptm_run_guard, "check_env_completeness"):
            # ptm_run_guard signature (Phase R): (required, env=None) --
            # required first, env second.
            return list(ptm_run_guard.check_env_completeness(required, env))
    except Exception:
        pass
    finally:
        if sys.path and sys.path[0].endswith("scripts"):
            sys.path.pop(0)
    problems = []
    for key, want in required.items():
        got = env.get(key)
        if got is None:
            problems.append(f"required engine env {key} is unset (want {want})")
        elif str(got) != str(want):
            problems.append(f"engine env {key}={got} differs from required {want}")
    return problems


def build_engine_env(config, base_env=None):
    """Compose the subprocess environment: inherited env + recommended-ON
    set + the user's Engine Flags table (config['engine_env']), which wins.
    Returns (env, applied) where applied lists 'KEY=VAL' lines actually set."""
    env = dict(base_env if base_env is not None else os.environ)
    table = dict(RECOMMENDED_ENGINE_ENV)
    user_table = config.get("engine_env", {}) if isinstance(config, dict) else {}
    if isinstance(user_table, dict):
        for k, v in user_table.items():
            k = str(k).strip()
            if not k:
                continue
            if v is None or str(v).strip() == "":
                table.pop(k, None)  # blank value = do not set / unset default
            else:
                table[k] = str(v).strip()
    applied = []
    for k in sorted(table):
        env[k] = table[k]
        applied.append(f"{k}={table[k]}")
    env["PYTHON_EXECUTABLE"] = sys.executable
    return env, applied


def absolutize_config_paths(config, project_root):
    """Deep-copy `config` with every known path field made absolute against
    `project_root`. Used when the run YAML is written somewhere other than
    the user's project directory (e.g. beside the output dir), because the
    engine resolves radiation.view_factor_file relative to the YAML's own
    directory (config_parser.cpp)."""
    from copy import deepcopy
    cfg = deepcopy(config)
    path_fields = _FILE_FIELDS + _DATA_DIR_FIELDS + [
        "radiation.view_factor_file",
        "output.directory",
    ]
    data_fields = set(_FILE_FIELDS + _DATA_DIR_FIELDS)
    for dotted in path_fields:
        val = get_config_value(cfg, dotted)
        if not val:
            continue
        parts = dotted.split(".")
        parent = cfg
        for k in parts[:-1]:
            parent = parent.setdefault(k, {})
        # Data-path keys follow the engine's PTM_DATA_ROOT contract;
        # output/VF keys resolve against the project root as before.
        resolver = resolve_data_path if dotted in data_fields else resolve_path
        parent[parts[-1]] = str(resolver(val, project_root))
    return cfg


def run_preflight(config, find_executable, project_root=None):
    """Run all pre-flight checks. `find_executable` is a callable returning
    the engine binary path or None (SimulationRunner.find_ptm_executable).
    Returns a PreflightReport; launch only if report.ok and the user
    accepted any cloud-path / VF-rebuild overrides."""
    report = PreflightReport()

    # -- Executable ---------------------------------------------------------
    exe = find_executable() if callable(find_executable) else find_executable
    if not exe or not Path(exe).exists():
        report.errors.append(
            "ptm_thermal_sim executable not found. Build the project first "
            "(cd build && make ptm_thermal_sim)."
        )
        return report  # nothing else is meaningful without the exe
    report.executable = Path(exe)
    if project_root is None:
        project_root = find_project_root(exe)
    report.project_root = Path(project_root)
    try:
        md5, mtime = executable_fingerprint(exe)
        report.info.append(f"engine: {exe} (md5 {md5}, built {mtime})")
    except OSError as e:
        report.warnings.append(f"could not fingerprint executable: {e}")

    # -- Individual file fields ---------------------------------------------
    # Data paths follow the engine's PTM_DATA_ROOT contract (absolute wins ->
    # $PTM_DATA_ROOT for "data/..." -> repo data/), so the preflight looks in
    # exactly the place the engine will.
    for dotted in _FILE_FIELDS:
        val = get_config_value(config, dotted)
        if not val:
            continue
        p = resolve_data_path(val, project_root)
        if not p.exists():
            report.errors.append(f"{dotted}: file not found: {p}")
            continue
        if not os.access(p, os.R_OK):
            report.errors.append(f"{dotted}: file not readable: {p}")
            continue
        if is_cloud_path(p):
            report.cloud_paths.append((dotted, str(p)))
        if is_dataless(p):
            report.dataless.append((dotted, str(p)))

    # -- Data directories (existence + dataless scan) ------------------------
    # Requirement gating mirrors the engine (config_parser.cpp): each data
    # directory is REQUIRED only when its feature is actually in use, and
    # the property-map directory NEVER hard-fails (the engine falls back to
    # uniform body defaults).
    mesh = config.get("mesh", {})
    illum = config.get("illumination", {})
    therm = config.get("thermal", {})
    for dotted in _DATA_DIR_FIELDS:
        val = get_config_value(config, dotted)
        if not val:
            continue
        # topo dir only matters if topographic meshes are enabled
        if dotted == "mesh.topo_directory" and not mesh.get("use_topographic", False):
            continue
        # albedo dir only matters if map loading is enabled
        if dotted == "illumination.albedo_maps_directory" and \
                illum.get("use_albedo_map", True) is False:
            continue
        soft = False
        if dotted == "thermal.property_maps_directory":
            if not therm.get("use_default_property_map", False):
                continue  # feature off: directory is irrelevant
            soft = True   # engine falls back to uniform defaults
        d = resolve_data_path(val, project_root)
        if not d.exists():
            if soft:
                report.warnings.append(
                    f"{dotted}: directory not found: {d} -- the engine will "
                    "fall back to uniform body defaults")
            else:
                report.errors.append(f"{dotted}: directory not found: {d}")
            continue
        if not d.is_dir():
            report.errors.append(f"{dotted}: not a directory: {d}")
            continue
        if is_cloud_path(d):
            report.cloud_paths.append((dotted, str(d)))
        bad, truncated = scan_dir_for_dataless(d)
        for b in bad[:20]:
            report.dataless.append((dotted, b))
        if len(bad) > 20:
            report.dataless.append((dotted, f"... and {len(bad) - 20} more"))
        if truncated:
            report.warnings.append(
                f"{dotted}: dataless scan truncated at {_SCAN_MAX_ENTRIES} entries/"
                f"{_SCAN_MAX_SECONDS:.0f}s -- large directory, scan incomplete"
            )

    # -- Output directory + disk space ---------------------------------------
    out_dir = resolve_path(get_config_value(config, "output.directory", "output"),
                           project_root)
    if is_cloud_path(out_dir):
        report.cloud_paths.append(("output.directory", str(out_dir)))
    probe = out_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free_gb = shutil.disk_usage(probe).free / 1e9
        if free_gb < _MIN_DISK_ERROR_GB:
            report.errors.append(
                f"output.directory: only {free_gb:.1f} GB free at {probe} "
                f"(< {_MIN_DISK_ERROR_GB:.0f} GB)")
        elif free_gb < _MIN_DISK_WARN_GB:
            report.warnings.append(
                f"output.directory: {free_gb:.1f} GB free at {probe} "
                f"(< {_MIN_DISK_WARN_GB:.0f} GB) -- long runs may fill the disk")
        else:
            report.info.append(f"output: {out_dir} ({free_gb:.0f} GB free)")
    except OSError as e:
        report.warnings.append(f"could not check disk space for {out_dir}: {e}")

    # -- Radiation / view-factor guard (PTM-19) ------------------------------
    rad = config.get("radiation", {})
    if rad.get("enabled", False):
        vf_val = rad.get("view_factor_file", "")
        if not vf_val:
            report.vf_rebuild_warning = (
                "Radiation is enabled with NO view-factor file. The engine will "
                "BUILD view factors from the current mesh. PTM-19: for meshes "
                "whose stored VF was built with different winding conventions "
                "(e.g. the moon3d meshes), a rebuild produces WRONG physics. "
                "Point radiation.view_factor_file at the existing .vf unless you "
                "are certain a rebuild is intended.")
        else:
            vf_path = resolve_path(vf_val, project_root)
            if not vf_path.exists():
                report.errors.append(
                    f"radiation.view_factor_file: not found: {vf_path}")
            elif is_dataless(vf_path):
                report.dataless.append(("radiation.view_factor_file", str(vf_path)))
            else:
                if is_cloud_path(vf_path):
                    report.cloud_paths.append(
                        ("radiation.view_factor_file", str(vf_path)))
                n_vf, fmt = read_vf_face_count(vf_path)
                if n_vf is None:
                    report.errors.append(
                        f"radiation.view_factor_file: {fmt}: {vf_path}")
                else:
                    n_mesh = expected_face_count(config)
                    if n_mesh is not None and n_mesh != n_vf:
                        report.errors.append(
                            f"view-factor face count mismatch: VF file has "
                            f"{n_vf} faces ({fmt}) but the mesh will have "
                            f"{n_mesh}. Wrong VF file for this mesh.")
                    else:
                        report.info.append(
                            f"view factors: {vf_path.name} ({fmt}, {n_vf} faces"
                            + ("" if n_mesh is None else ", matches mesh") + ")")

    # -- Engine env completeness (R3a) ---------------------------------------
    env, applied = build_engine_env(config)
    problems = check_env_completeness(env)
    if problems:
        # build_engine_env always applies the recommended set, so problems
        # here mean the user's Engine Flags table removed/changed one.
        for p in problems:
            report.warnings.append(f"engine env: {p}")
    report.info.append("engine env: " + " ".join(applied))

    return report


# --------------------------------------------------------------------------
# GUI confirmation dialog (kept here so ptm3d_gui and the dashboard thermal
# adapter share one implementation). Imports Qt lazily so the logic above is
# testable without a QApplication.
# --------------------------------------------------------------------------

def confirm_preflight(parent, report):
    """Show the preflight outcome. Returns True if launch should proceed.

    errors / dataless  -> blocking dialog, no override (dataless files WILL
                          hang the engine; hydrate or relocate first)
    cloud paths        -> warning with an explicit override checkbox
    VF rebuild (PTM-19)-> warning with an explicit override checkbox
    """
    from PyQt6.QtWidgets import QMessageBox, QCheckBox

    if report.errors or report.dataless:
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Pre-flight failed")
        lines = [f"- {e}" for e in report.errors]
        if report.dataless:
            lines.append("")
            lines.append("Cloud placeholder (dataless) files -- reading these "
                         "hangs the simulation until the cloud provider "
                         "downloads them:")
            lines += [f"- {f_}: {p}" for f_, p in report.dataless[:15]]
            lines.append("")
            lines.append("Hydrate these files (Finder: right-click > \"Always "
                         "Keep on This Device\", wait for download) or point "
                         "the config at a local copy of the data.")
        msg.setText("The configuration cannot run as-is:")
        msg.setInformativeText("\n".join(lines))
        msg.exec()
        return False

    if report.cloud_paths:
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Cloud-synced paths")
        listing = "\n".join(f"- {f_}: {p}" for f_, p in report.cloud_paths[:15])
        msg.setText("Some paths are on cloud-synced storage "
                    "(/Library/CloudStorage/). Files there can be evicted at "
                    "any time; a future read of an evicted file will hang the "
                    "simulation.")
        msg.setInformativeText(listing + "\n\nRecommended: copy the data to a "
                               "local directory and update the config.")
        override = QCheckBox("Run anyway -- I verified these files are fully "
                             "available offline")
        msg.setCheckBox(override)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok |
                               QMessageBox.StandardButton.Cancel)
        msg.exec()
        if not override.isChecked():
            return False

    if report.vf_rebuild_warning:
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("View-factor rebuild")
        msg.setText(report.vf_rebuild_warning)
        override = QCheckBox("Proceed and let the engine build view factors "
                             "from the current mesh")
        msg.setCheckBox(override)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok |
                               QMessageBox.StandardButton.Cancel)
        msg.exec()
        if not override.isChecked():
            return False

    return True
