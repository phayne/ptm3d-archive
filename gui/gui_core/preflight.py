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
#   PTM_ILLUM_RISE_EPS=1      enables the rise-epsilon grazing-incidence path
#                             (traceOccludedRiseEps). It is the GATE: without
#                             it, RISE_KEPS and RISE_HABS below are never read
#                             (illumination_model.cpp:616, :1419).
#   PTM_ILLUM_RISE_KEPS=0.2   the rise-epsilon scale factor validated with it
#   PTM_ILLUM_RISE_HABS=0     grazing rise-floor leak fix: h_abs = 0 kills
#                             100% of leaks with zero shadow acne (same gate)
#   PTM_ILLUM_SEAL_HOLES=1    PTM-18 interior mesh-hole penumbra flash fix
#
# This is the [recommended-ON] set of docs/ENVIRONMENT.md MINUS one variable:
#
#   PTM_EQ_PERFACE_FLUX       deliberately NOT injected (PTM-276). On the CPU
#                             path the engine itself reports it has "NO effect
#                             on this run"; on the GPU path with
#                             thermal.spinup_seed: legacy it is LIVE and
#                             silently overrides the one setting whose whole
#                             purpose is digit-for-digit reproduction of a
#                             pre-fix baseline. A config that wants it can
#                             still ask, in its own engine_env: block.
#
# ENV_SET_CHANGED_PHYSICS below records that this table changed on 2026-08-25
# and that GUI-run physics moved with it.
#
# Unknown variables are ignored by older engine builds, so this set is safe
# to inject unconditionally.
# --------------------------------------------------------------------------
RECOMMENDED_ENGINE_ENV = {
    "PTM_ILLUM_TRUE_NORMALS": "1",
    "PTM_ILLUM_RISE_EPS": "1",
    "PTM_ILLUM_RISE_KEPS": "0.2",
    "PTM_ILLUM_RISE_HABS": "0",
    "PTM_ILLUM_SEAL_HOLES": "1",
}

#: Variables in the docs/ENVIRONMENT.md [recommended-ON] set that the GUI does
#: NOT inject, each with the reason. Pinned by
#: ``src/python/tests/test_gui_engine_env.py``: a variable may be left out only
#: by appearing here, so the omission is a decision on the record rather than a
#: table that quietly fell behind the documentation.
DELIBERATELY_NOT_INJECTED = {
    "PTM_EQ_PERFACE_FLUX":
        "Inert on the processor path (the engine says so), and on the graphics "
        "cards it overrides an explicit thermal.spinup_seed: legacy. Add it "
        "per configuration if you want it.",
}

#: One-line banner for the console: injecting the rise-epsilon gate changed the
#: physics of every GUI run relative to every GUI run before 2026-08-25.
ENV_SET_CHANGED_PHYSICS = (
    "engine environment changed 2026-08-25: PTM_ILLUM_RISE_EPS=1 and "
    "PTM_ILLUM_RISE_KEPS=0.2 are now injected, so the validated grazing-leak "
    "fix APPLIES. Every GUI run before this date silently ran without it "
    "(PTM_ILLUM_RISE_HABS alone is inert). Illumination -- and therefore "
    "temperatures -- differ from earlier GUI runs of the same config. "
    "PTM_EQ_PERFACE_FLUX is no longer injected."
)

CLOUD_PATH_MARKERS = ("/Library/CloudStorage/",)

# Config fields that hold a single file path: (dotted key, required-if callable)
_FILE_FIELDS = [
    "mesh.import_file",
    # The occluder gets the same existence, dataless and cloud checks as the
    # core mesh: it is read at BVH build time, so an evicted cloud placeholder
    # here hangs the run in exactly the same way.
    "mesh.occluder_file",
    "illumination.albedo_map",
    "illumination.positions_file",
    "thermal.thermal_inertia_map",
    "thermal.h_parameter_map",
    "thermal.heat_flow_mask_file",
    "thermal.initial_state_file",
    # An explicit 'QIND' file from a prior pass-1 run. It is READ at
    # equilibration time and always wins over eq_indirect_source, so an
    # evicted cloud placeholder here changes the deep answer, not just the
    # speed -- the same reason the occluder joined this list in Stage 1.
    "thermal.eq_indirect_flux_file",
    "output.locations_file",
    "output.depths_file",
    "generic_body.config_file",
    "generic_body.spice_metakernel",
]

#: Data-path fields that follow the PTM_DATA_ROOT contract but must NOT be
#: existence-checked, because a first run is supposed to find them missing.
#: (config_parser.cpp resolves illumination.bvh_cache_file with resolveDataPath
#: and `required=false`.)
_DATA_PATH_NO_EXISTENCE_CHECK = [
    "illumination.bvh_cache_file",
]

#: Values that are SENTINELS, not paths, and must survive absolutization
#: untouched. config_parser.h:435 and illumination_model.cpp:781/795 document
#: `bvh_cache_file: "off"` as "disable the BVH cache"; the engine's
#: resolveDataPath returns it verbatim (not absolute, not `data/`, does not
#: exist). Absolutizing it produced "<project_root>/off", which would ENABLE
#: caching into a file literally named `off` -- the opposite of what was asked.
#: Reproduced.
_PATH_SENTINELS = {
    "illumination.bvh_cache_file": {"off"},
}


def is_path_sentinel(dotted, value):
    """True when `value` is a documented sentinel for `dotted`, not a path.

    EXACT, because the engine's test is exact: `bvh_cache_file_ != "off"`
    (illumination_model.cpp:781, :795). A case-insensitive match here would
    make the GUI skip `"OFF"` as "caching disabled" while the engine read it as
    a relative path and wrote a cache to <repo>/OFF -- the disclosure and the
    run disagreeing about the same file.
    """
    return str(value) in _PATH_SENTINELS.get(dotted, ())


#: EXACTLY the keys the engine resolves through resolveDataPath
#: (config_parser.cpp, the "Apply the PTM_DATA_ROOT resolution contract" block).
#: Transcribed from that list, not inferred from "it looks like a path".
#:
#: Being on _FILE_FIELDS is about the pre-flight EXISTENCE check; being here is
#: about how a relative `data/...` value RESOLVES, and the two sets are not the
#: same. thermal.eq_indirect_flux_file, thermal.initial_state_file,
#: illumination.positions_file and generic_body.config_file are read by the
#: engine as plain paths relative to the process working directory, so
#: resolving them under PTM_DATA_ROOT made the GUI's run YAML point somewhere
#: a CLI run of the same config would not look -- silently, and for the
#: indirect-flux file that changes the deep answer.
_ENGINE_DATA_ROOT_FIELDS = [
    "mesh.import_file",
    "mesh.occluder_file",
    "mesh.topo_directory",
    "thermal.thermal_inertia_map",
    "thermal.h_parameter_map",
    "thermal.heat_flow_mask_file",
    "thermal.property_maps_directory",
    "illumination.albedo_map",
    "illumination.albedo_maps_directory",
    "illumination.bvh_cache_file",
    "output.locations_file",
    "output.depths_file",
    "generic_body.spice_metakernel",
    # The engine's key is generic_body.mesh_file (config_parser.cpp:881); the
    # GUI writes the same value at generic_body.shape.mesh_file. Listed under
    # BOTH spellings so this table stays literally exact against that block,
    # whichever one a future writer emits.
    "generic_body.mesh_file",
    "generic_body.shape.mesh_file",
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
    beaming_warning: str = ""                        # facet-scale rule text, if triggered
    vf_cost_line: str = ""                           # advisory VF cost headline
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
        if self.beaming_warning:
            out.append(f"BEAM-GUARD: {self.beaming_warning}")
        if self.vf_cost_line:
            out.append(f"VF-COST: {self.vf_cost_line}")
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


# --------------------------------------------------------------------------
# Pre-run view-factor cost estimate (advisory: warns, never blocks).
# --------------------------------------------------------------------------

# 8 bytes per stored nonzero: int32 column index + float32 value.
_VF_BYTES_PER_NNZ = 8
# Radiosity-SpMV throughput (nonzeros/second) from 32/48 min per window at
# 2.48e9 nnz on the reference Mac Pro. Order-of-magnitude only.
_VF_THROUGHPUT_GPU = 1.3e6
_VF_THROUGHPUT_CPU = 8.6e5
# Single-GPU SpMV hard wall: a signed-int32 byte offset into a 4-byte-element
# OpenCL buffer caps one buffer at 2^31 bytes = 536,870,912 nonzeros.
VF_SINGLE_GPU_NNZ_WALL = 536_870_912
# Partners-per-face heuristic (the dominant uncertainty): smooth/convex 5-20,
# local-rough DEM 30-80, global/high-roughness lunar 100-150. Default 105.
_VF_PARTNERS_DEFAULT = 105
_VF_PARTNERS_LOW = 20
_VF_PARTNERS_HIGH = 150


def read_vf_nnz(vf_path):
    """True stored-nonzero count from a sparse (VFFM) VF header, or None.

    Only the sparse format stores nnz in the header (u64 at byte offset 24);
    the H-matrix format stores compressed blocks with no single nnz field."""
    try:
        with open(vf_path, "rb") as f:
            header = f.read(32)
    except OSError:
        return None
    if len(header) < 32:
        return None
    magic, _version = struct.unpack("<II", header[:8])
    if magic != 0x5646464D:  # 'VFFM' sparse
        return None
    (nnz,) = struct.unpack("<Q", header[24:32])
    return int(nnz)


def _fmt_duration(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.1f} h"


def _fmt_disk_gb(gb):
    if gb < 1.0:
        return f"{gb * 1000.0:.0f} MB"
    if gb >= 1000.0:
        return f"{gb / 1000.0:.2f} TB"
    return f"{gb:.1f} GB"


@dataclass
class VFCostEstimate:
    """Advisory view-factor disk/solve estimate. See estimate_vf_cost."""
    available: bool = False
    mode: str = "unavailable"   # "header-exact" | "heuristic" | "unavailable"
    n_faces: int = None
    nnz_mid: int = None
    nnz_low: int = None
    nnz_high: int = None
    disk_gb_mid: float = None
    disk_gb_low: float = None
    disk_gb_high: float = None
    wall_s_mid: float = None
    throughput_label: str = ""
    free_gb: float = None
    lines: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def headline(self):
        if not self.available:
            return "View-factor cost: not estimable for this config"
        if self.mode == "header-exact":
            return (f"View-factor cost ~{_fmt_disk_gb(self.disk_gb_mid)} disk, "
                    f"~{_fmt_duration(self.wall_s_mid)} solve/window "
                    f"({self.throughput_label}; nnz exact from VF header)")
        return (f"View-factor cost ~{_fmt_disk_gb(self.disk_gb_mid)} disk "
                f"({_fmt_disk_gb(self.disk_gb_low)}-{_fmt_disk_gb(self.disk_gb_high)}), "
                f"~{_fmt_duration(self.wall_s_mid)} solve/window "
                f"({self.throughput_label}; approximate, order-of-magnitude)")


def estimate_vf_cost(config, project_root=None):
    """Advisory pre-run estimate of view-factor disk + radiosity-solve cost.

    Radiosity coupling stores a sparse face-to-face matrix; its size scales as
    ``nnz = n_faces * (partners per face)``. Disk = 8 bytes/nnz (int32 column +
    float32 value). Solve wall time = nnz / throughput. The view-factor *build*
    time is reported as a SEPARATE, "highly variable" line and never folded in.

    Numbers are order-of-magnitude from the partners/face heuristic UNLESS an
    existing sparse VF file is configured, in which case the true nnz is read
    from its header (exact). Advisory only: warns, never blocks. Returns a
    VFCostEstimate.
    """
    est = VFCostEstimate()
    rad = config.get("radiation", {}) if isinstance(config, dict) else {}
    if not rad.get("enabled", False):
        est.lines = ["Radiation disabled - no view-factor cost."]
        return est

    use_gpu = bool(rad.get("use_gpu", True))
    est.throughput_label = "GPU" if use_gpu else "CPU"
    throughput = _VF_THROUGHPUT_GPU if use_gpu else _VF_THROUGHPUT_CPU

    # Prefer the true nnz from an existing sparse VF file.
    nnz_true = None
    vf_val = str(rad.get("view_factor_file", "") or "")
    if vf_val and project_root is not None:
        try:
            vf_path = resolve_path(vf_val, project_root)
            if vf_path.exists():
                nnz_true = read_vf_nnz(vf_path)
        except OSError:
            nnz_true = None

    if nnz_true is not None:
        est.available = True
        est.mode = "header-exact"
        est.nnz_mid = est.nnz_low = est.nnz_high = nnz_true
    else:
        n_faces = expected_face_count(config)
        if not n_faces:
            est.lines = [
                "View-factor cost: mesh face count not derivable from this "
                "config (imported non-PLY mesh or topography tile) and no "
                "existing VF file to read -- cannot estimate."]
            return est
        est.available = True
        est.mode = "heuristic"
        est.n_faces = int(n_faces)
        est.nnz_mid = int(n_faces * _VF_PARTNERS_DEFAULT)
        est.nnz_low = int(n_faces * _VF_PARTNERS_LOW)
        est.nnz_high = int(n_faces * _VF_PARTNERS_HIGH)

    est.disk_gb_mid = est.nnz_mid * _VF_BYTES_PER_NNZ / 1e9
    est.disk_gb_low = est.nnz_low * _VF_BYTES_PER_NNZ / 1e9
    est.disk_gb_high = est.nnz_high * _VF_BYTES_PER_NNZ / 1e9
    est.wall_s_mid = est.nnz_mid / throughput

    lines = []
    if est.mode == "header-exact":
        lines.append(
            f"Existing VF file: {est.nnz_mid:,} nonzeros (from header, exact).")
        lines.append(
            f"  Disk (already on disk): ~{_fmt_disk_gb(est.disk_gb_mid)} "
            f"(8 bytes/nonzero).")
    else:
        lines.append(
            f"Mesh faces: {est.n_faces:,}. Assuming ~{_VF_PARTNERS_DEFAULT} "
            f"partners/face (band {_VF_PARTNERS_LOW}-{_VF_PARTNERS_HIGH}; "
            f"smooth-convex can be 5-20).")
        lines.append(
            f"  Nonzeros: ~{est.nnz_mid:,} "
            f"({est.nnz_low:,}-{est.nnz_high:,}).")
        lines.append(
            f"  Disk: ~{_fmt_disk_gb(est.disk_gb_mid)} "
            f"({_fmt_disk_gb(est.disk_gb_low)}-{_fmt_disk_gb(est.disk_gb_high)}) "
            f"at 8 bytes/nonzero.")
    lines.append(
        f"  Radiosity solve: ~{_fmt_duration(est.wall_s_mid)}/window "
        f"({est.throughput_label}, ~{throughput:.1e} nonzeros/s).")
    lines.append(
        "  View-factor BUILD time is SEPARATE and HIGHLY VARIABLE (minutes to "
        "hours; depends on mesh + occlusion) -- not included above.")
    if est.mode == "header-exact":
        lines.append(
            "Approximate - order-of-magnitude (nnz exact; disk/time from "
            "throughput).")
    else:
        lines.append(
            "Approximate - order-of-magnitude; disk can be +/-3-5x from the "
            "partners/face heuristic.")

    # Threshold warnings (warn, never block).
    if est.nnz_mid > VF_SINGLE_GPU_NNZ_WALL:
        est.warnings.append(
            f"estimated nonzeros (~{est.nnz_mid:,}) exceed the single-GPU "
            f"radiosity wall ({VF_SINGLE_GPU_NNZ_WALL:,} = 2^31-byte SpMV "
            f"offset) -- single-GPU radiosity needs the chunked/int64 path or "
            f"multi-GPU, otherwise it cannot allocate.")
    if project_root is not None:
        try:
            probe = project_root
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            est.free_gb = shutil.disk_usage(str(probe)).free / 1e9
            if est.disk_gb_high > est.free_gb:
                est.warnings.append(
                    f"estimated view-factor disk (up to ~{est.disk_gb_high:.1f} "
                    f"GB) may exceed free space (~{est.free_gb:.1f} GB).")
        except OSError:
            pass

    est.lines = lines
    return est


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


def engine_env_plan(config):
    """Decide WHICH variables a config asks for. Returns (table, source).

    A config that carries an ``engine_env:`` mapping is taken at its word: the
    block IS the environment, so ``engine_env: {}`` means "no fix flags" and a
    block naming one variable does not silently gain the other four. A config
    with no ``engine_env:`` key at all has not expressed an opinion, and gets
    the recommended-ON set.

    Before this distinction existed, ``load_from_yaml`` merged the loaded file
    into DEFAULT_CONFIG, so every config came back carrying the recommended set
    whatever it said -- there was no way to write "this run wants nothing".
    A block that omits a recommended variable still gets a preflight warning
    from ``check_env_completeness``; honoring it is not the same as endorsing it.
    """
    if not isinstance(config, dict) or "engine_env" not in config:
        return dict(RECOMMENDED_ENGINE_ENV), "recommended-ON default set"
    block = config.get("engine_env")
    if not isinstance(block, dict):
        # `engine_env:` with no mapping under it. Do not read a truncated file
        # as a request for an empty environment; say so and use the default.
        return (dict(RECOMMENDED_ENGINE_ENV),
                "engine_env: present but not a mapping -- ignored, "
                "recommended-ON default set applied")
    table = {}
    for k, v in block.items():
        k = str(k).strip()
        if not k:
            continue
        if v is None or str(v).strip() == "":
            continue  # blank value = do not set
        table[k] = str(v).strip()
    return table, "engine_env: block in the config"


#: Environment variables that OVERRIDE a config key the GUI now writes. An
#: inherited one silently beats the control the user just set, which is the
#: exact failure PTM_EQ_PERFACE_FLUX was dropped for -- so they are managed too,
#: and removed unless a config's engine_env: block asks for them by name.
#:
#:   PTM_SPINUP_SEED          ptm_thermal_sim.cpp:15688  thermal.spinup_seed
#:   PTM_SEED_ANCHOR          :15713                     thermal.seed_anchor
#:   PTM_ANALYTIC_SEED_CYCLES :15731                     thermal.analytic_seed_cycles
#:   PTM_COUPLING_MODE        :14143                     radiation.coupling_mode
#:
#: The seed anchor is the sharpest: `warm` reads +45 K on the lit population of
#: the shipped lunar 80N crater, and an exported PTM_SEED_ANCHOR would apply it
#: while the Thermal tab said `lumped`.
#:   PTM_HMATRIX_THREADS      hmatrix_view_factor.cpp:1421  radiation.hmatrix_threads
#:                            ("Env override (highest priority)" in its own
#:                            comment; the GUI has had a control for this key
#:                            since before this branch, and EngineFlagsTab's
#:                            docstring names this variable as THE example extra
#:                            flag, so users do have it exported)
#:
#: NOT managed, deliberately: PTM_RAD_EMISSIVITY and PTM_SOLAR_ALBEDO
#: (radiation_exchange.cpp:208, :235). Neither overrides a config key the GUI
#: writes -- they force a radiosity-internal quantity that has no config
#: equivalent -- and both announce themselves; PTM_RAD_EMISSIVITY additionally
#: makes the engine's own [kirchhoff] check FAIL the run. Removing a variable
#: nobody's control claims to own would be overreach.
CONFIG_OVERRIDING_ENGINE_ENV = (
    "PTM_SPINUP_SEED",
    "PTM_SEED_ANCHOR",
    "PTM_ANALYTIC_SEED_CYCLES",
    "PTM_COUPLING_MODE",
    "PTM_HMATRIX_THREADS",
)

#: The variables this module OWNS: it sets them when the plan asks for them and
#: REMOVES them when the plan does not. Nothing outside this set is ever taken
#: out of the inherited environment.
MANAGED_ENGINE_ENV = tuple(sorted(
    set(RECOMMENDED_ENGINE_ENV) | set(DELIBERATELY_NOT_INJECTED)
    | set(CONFIG_OVERRIDING_ENGINE_ENV)))


def build_engine_env(config, base_env=None):
    """Compose the subprocess environment: inherited env, then the managed
    variables forced to exactly what ``engine_env_plan`` says.

    Setting is not enough. The launching shell may already export
    ``PTM_EQ_PERFACE_FLUX=1``, and merely declining to add it would leave the
    child running with it -- so a config that says "no fix flags", or a
    ``thermal.spinup_seed: legacy`` run that must not be overridden (PTM-276),
    would silently get it anyway while this function reported an empty
    environment. Managed variables the plan omits are therefore REMOVED from the
    inherited environment, and the removal is reported.

    Returns (env, applied); `applied` lists what was set AND what was removed,
    because a run's disclosure has to describe the environment the engine
    actually sees.
    """
    env = dict(base_env if base_env is not None else os.environ)
    table, _source = engine_env_plan(config)
    applied = []
    for k in sorted(table):
        env[k] = table[k]
        applied.append(f"{k}={table[k]}")
    for k in MANAGED_ENGINE_ENV:
        if k not in table and k in env:
            inherited = env.pop(k)
            applied.append(f"{k}=(removed; inherited {inherited!r})")
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
        # An OUTPUT path, so it is absolutized but not existence-checked: it is
        # the file this run will write, not one it must find.
        "thermal.final_state_file",
        "output.directory",
    ] + _DATA_PATH_NO_EXISTENCE_CHECK
    # The engine resolves these through the PTM_DATA_ROOT contract
    # (resolveDataPath), not against the project root. Getting this wrong is
    # silent: the GUI would write <project_root>/cache/site.bvh into the run
    # YAML while the engine alone would have used $PTM_DATA_ROOT/cache/site.bvh,
    # and the BVH cache would be written and read in different places.
    data_fields = set(_ENGINE_DATA_ROOT_FIELDS)
    for dotted in path_fields:
        val = get_config_value(cfg, dotted)
        if not val:
            continue
        if is_path_sentinel(dotted, val):
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


#: Wall-clock ceiling for the engine's own config check. Measured: 0.13-0.20 s
#: on this machine, because the mode parses and exits. This call is SYNCHRONOUS
#: on the Qt thread, so the ceiling is also how long the window can freeze --
#: it was 60 s, which is a minute of dead GUI on a stalled network or cloud
#: mount. 15 s leaves ~75x headroom over the measured time and bounds the
#: freeze. The caller shows a wait cursor for the duration.
VALIDATE_TIMEOUT_S = 15.0
#: Ceiling for the `--help` capability probe. It is a separate synchronous
#: call, so the WORST-CASE freeze is this plus VALIDATE_TIMEOUT_S -- giving the
#: probe the full validation ceiling made the documented 15 s bound a 30 s one.
#: --help prints and exits, so 5 s is already generous.
PROBE_TIMEOUT_S = 5.0


def engine_validate_config(config, executable, project_root=None, env=None):
    """Hand `config` to the engine's own parser and report what IT says.

    Item 2.4 / DP-6. ``ConfigManager.validate`` is a second, weaker copy of the
    engine's rules, and the two have already disagreed: the first draft of the
    ``eq_indirect_source`` check invented a valid set (``auto|none|file|
    internal``) that would have refused ``converged``, the one non-default value
    the engine accepts. The fix is to stop keeping a second copy of the answer.

    Returns ``(status, text)`` where status is:
      ``"accepted"``     rc 0 -- well formed as far as this mode can tell
      ``"refused"``      rc 1 -- the engine will not run it, and `text` says why
      ``"unavailable"``  no binary, no ``--validate-config`` support, or the
                         call failed. NEVER silently treated as acceptance.

    The YAML is written to a temporary file with every path absolutized, so it
    resolves exactly as the run YAML would; the temporary file is removed
    afterward, whatever happens.
    """
    import shutil
    import subprocess
    import tempfile
    import yaml as _yaml

    if not executable:
        return ("unavailable",
                "No ptm_thermal_sim was found, so the engine's own checks "
                "could not be run. Only the GUI's checks applied. Build it "
                "with: cmake --build build --target ptm_thermal_sim")
    executable = Path(executable)
    if not executable.exists():
        return ("unavailable",
                f"{executable} does not exist, so the engine's own checks "
                f"could not be run. Only the GUI's checks applied.")

    # CAPABILITY PROBE FIRST, and it is not a nicety.
    # The engine's argument loop has no unknown-argument handler, so a binary
    # that predates --validate-config IGNORES it and runs the simulation.
    # MEASURED against ~/projects/ptm-postmerge-2026-08-24 (md5
    # 9d657590009ade0c0d8f2d07d2faa7a2): it ran a complete Moon ico2 job, wrote
    # the whole output directory and exited 0. On a production config, pressing
    # Ctrl+T would have started a multi-hour GPU run.
    #
    # --help exits without touching anything, so asking it first is free.
    try:
        probe = subprocess.run([str(executable), "--help"], capture_output=True,
                               text=True, timeout=PROBE_TIMEOUT_S)
    except Exception as exc:                           # noqa: BLE001
        return ("unavailable",
                f"{executable} could not be asked for its options ({exc}), so "
                f"the engine's checks were not run.")
    if "--validate-config" not in (probe.stdout or "") + (probe.stderr or ""):
        return ("unavailable",
                f"{executable} does not support --validate-config, so nothing "
                f"was checked by the engine -- only the GUI's own rules "
                f"applied.\n"
                f"  The flag was NOT passed to it. A binary that predates the "
                f"flag ignores it and RUNS THE SIMULATION: measured, a "
                f"pre-flag build ran a complete job and exited 0 when handed "
                f"--validate-config.\n"
                f"  Rebuild from a tree that carries the flag: cmake --build "
                f"build --target ptm_thermal_sim")

    root = Path(project_root) if project_root else find_project_root(executable)
    tmpdir = tempfile.mkdtemp(prefix="ptm3d_validate_")
    try:
        path = Path(tmpdir) / "config.yaml"
        # Belt and braces: point output.directory at the temporary directory.
        # Nothing in --validate-config reads it, and if a binary ever ignores
        # the flag again, whatever it writes lands here and is deleted with the
        # rest. The user's own output directory is never a target of a CHECK.
        checked = absolutize_config_paths(config, root)
        # isinstance, not setdefault: `output:` as a scalar or null makes
        # setdefault return the non-mapping and dict() raise TypeError -- out
        # of the Check Configuration slot, which is the process-abort class
        # this codebase guards against everywhere else.
        node = checked.get("output")
        checked["output"] = dict(node) if isinstance(node, dict) else {}
        checked["output"]["directory"] = str(Path(tmpdir) / "output")
        with open(path, "w") as handle:
            _yaml.dump(checked, handle,
                       default_flow_style=False, sort_keys=False)
        # build_engine_env returns (env, applied). The SAME environment the
        # launch would apply: validating under a different one answers a
        # different question, and PTM_ILLUM_RISE_EPS is exactly the variable
        # that turned out to matter.
        child_env, _applied = build_engine_env(config, env)
        try:
            proc = subprocess.run(
                [str(executable), "--config", str(path), "--validate-config"],
                capture_output=True, text=True, cwd=str(root),
                env=child_env,
                timeout=VALIDATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return ("unavailable",
                    f"The engine did not answer within {VALIDATE_TIMEOUT_S:.0f} s. "
                    f"--validate-config parses and exits, so this is not a slow "
                    f"config -- treat it as 'not checked', not as 'accepted'.")
        except Exception as exc:                       # noqa: BLE001
            return ("unavailable",
                    f"The engine's config check could not be run ({exc}). Only "
                    f"the GUI's checks applied.")

        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode < 0:
            # A signal, not a verdict. --validate-config returns 0 or 1; a
            # negative code is a crash IN the parser (this codebase has a
            # documented -mavx2 / Eigen ABI SIGABRT class), and reporting it
            # as "the engine REFUSED this configuration" blames the user's YAML
            # for an engine fault.
            import signal as _signal
            try:
                name = _signal.Signals(-proc.returncode).name
            except (ValueError, AttributeError):
                name = f"signal {-proc.returncode}"
            return ("unavailable",
                    f"The engine CRASHED while parsing the configuration "
                    f"({name}). That is a fault in the binary, not a verdict on "
                    f"the config -- nothing was validated.\n" + text)
        if proc.returncode > 1:
            return ("unavailable",
                    f"The engine exited {proc.returncode}, which is neither "
                    f"accept (0) nor refuse (1). Treat it as not checked.\n"
                    + text)
        if proc.returncode == 0 and "[validate-config]" in text:
            return ("accepted", text)
        if proc.returncode == 0:
            # rc 0 with no banner means this binary predates --validate-config
            # and ignored the unknown flag. It did NOT validate anything, and
            # reporting success here would be the exact silence this whole
            # stage is about.
            return ("unavailable",
                    "This ptm_thermal_sim does not support --validate-config "
                    "(it exited 0 without the [validate-config] banner), so "
                    "nothing was checked by the engine. Rebuild from a tree "
                    "that carries the flag, or rely on the GUI's own checks.")
        return ("refused", text)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_preflight(config, find_executable, project_root=None,
                  config_manager=None):
    """Run all pre-flight checks. `find_executable` is a callable returning
    the engine binary path or None (SimulationRunner.find_ptm_executable).
    Returns a PreflightReport; launch only if report.ok and the user
    accepted any cloud-path / VF-rebuild overrides.

    `config_manager` is optional and is used for the one check that cannot be
    made from the configuration dictionary alone -- see the body-identity
    check below, which needs to know whether `simulation.body` was chosen or
    is DEFAULT_CONFIG showing through.
    """
    report = PreflightReport()

    # -- Two bodies in one configuration ------------------------------------
    # Said HERE because this is the screen the user reads before a run starts.
    # The correction itself happens in ConfigManager._config_for_yaml, which
    # leaves the contradicting `generic_body:` section out of the file the
    # engine is handed; without this line the only trace would be a log
    # warning, and a run that quietly drops a body definition is the same
    # class of silence as a run that quietly uses one.
    if config_manager is not None:
        try:
            conflict = config_manager.body_identity_conflict()
        except Exception as exc:                             # noqa: BLE001
            conflict = ""
            report.warnings.append(
                f"could not check whether this configuration names one body "
                f"or two: {exc}")
        if conflict:
            report.warnings.append("body identity: " + conflict)

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
        if is_path_sentinel(dotted, val):
            continue
        # The SAME resolver absolutize_config_paths uses for this key. They
        # disagreed: the pre-flight stat'd $PTM_DATA_ROOT/... for four keys the
        # engine reads relative to the working directory, so a launch could be
        # blocked for a file that exists, or waved through for one that does
        # not -- and for thermal.eq_indirect_flux_file that changes the deep
        # answer rather than merely failing.
        resolver = (resolve_data_path if dotted in _ENGINE_DATA_ROOT_FIELDS
                    else resolve_path)
        p = resolver(val, project_root)
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
                "Radiation exchange is on but no view-factor file is set. "
                "PTM3D will build the view-factor matrix (which facets see which, "
                "for facet-to-facet radiation) from the current mesh before the "
                "run starts. For large meshes this can take a long time and a "
                "lot of disk space — see the estimate below. If you already have "
                "a view-factor file built for this exact mesh, set "
                "radiation.view_factor_file to it to reuse it and skip the build.")
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

    # -- View-factor cost estimate (advisory: warns, never blocks) -----------
    if rad.get("enabled", False):
        est = estimate_vf_cost(config, project_root)
        if est.available:
            report.vf_cost_line = est.headline()
            report.info.extend(est.lines)
            report.warnings.extend(est.warnings)

    # -- Beaming facet-scale guard -------------------------------------------
    # The configuration tab shows the rule, but a label on a tab is not a gate:
    # a user can enable beaming on a lunar icosphere and then import a
    # millimeter-scale mesh, and nothing outside that tab would ever mention it
    # again. The tab deliberately does NOT rewrite the setting -- silently
    # changing someone's physics is the failure this stage exists to remove --
    # so the launch is where it has to be asked. Same shape as the PTM-19
    # view-factor guard: a warning with an explicit override.
    if rad.get("beaming", False) and not rad.get("enabled", False):
        # Inert, and the run would say nothing about it. Warn rather than let
        # a Lambertian result be read as a beaming one.
        report.warnings.append(
            "beaming: radiation.beaming is ON but radiation.enabled is off, so "
            "the engine ignores it entirely -- it reads that key only on the "
            "radiation path. This run will be Lambertian and will not say so.")
    elif rad.get("beaming", False):
        try:
            from gui_core.facet_scale import beaming_policy
            policy = beaming_policy(config, project_root)
            if not policy.allowed:
                report.beaming_warning = (
                    "radiation.beaming is ON and this mesh does not meet the "
                    "facet-scale rule, so the sub-facet roughness model would "
                    "be applied on top of roughness the mesh already resolves "
                    "(double-counting). " + policy.reason)
            elif policy.needs_acknowledgment:
                report.warnings.append("beaming: " + policy.reason)
            else:
                report.info.append("beaming: " + policy.reason)
        except Exception as e:                       # advisory, never a crash
            report.warnings.append(
                f"beaming: the facet-scale rule could not be evaluated ({e}); "
                f"confirm this mesh is far coarser than the roughness scale.")

    # -- Engine env completeness (R3a) ---------------------------------------
    env, applied = build_engine_env(config)
    _table, env_source = engine_env_plan(config)
    problems = check_env_completeness(env)
    if problems:
        # A problem here means this config's own engine_env: block dropped or
        # changed a recommended variable. Honoring the block is the point; not
        # saying so would be the old silence.
        for p in problems:
            report.warnings.append(f"engine env: {p}")
    report.info.append(f"engine env source: {env_source}")
    report.info.append("engine env: " + (" ".join(applied) or "(none)"))
    # A physics change is not a footnote. Warn, not info, so the console
    # classifier colors it and a colleague comparing against an older GUI run
    # of the same config is told why the numbers moved.
    if env.get("PTM_ILLUM_RISE_EPS") == "1":
        report.warnings.append(ENV_SET_CHANGED_PHYSICS)

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

    if report.beaming_warning:
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Beaming on a fine mesh")
        msg.setText(report.beaming_warning)
        msg.setInformativeText(
            "The configuration is left exactly as you set it either way -- "
            "nothing here rewrites your physics. Cancel to turn beaming off on "
            "the Radiation page, or confirm to run as configured.")
        override = QCheckBox("Run anyway -- this mesh does not resolve "
                             "sub-facet slopes")
        msg.setCheckBox(override)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok |
                               QMessageBox.StandardButton.Cancel)
        # BOTH the button and the box. The two older dialogs below consult only
        # the checkbox, so Cancel with the box ticked launches anyway -- which
        # is survivable there because they do not mention the buttons, but this
        # one's own text tells the user to press Cancel. (The same bug in the
        # cloud-path and view-factor dialogs is pre-existing and out of scope
        # for this branch; it is owed as a ledger row.)
        clicked = msg.exec()
        if clicked != QMessageBox.StandardButton.Ok or not override.isChecked():
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
        text = report.vf_rebuild_warning
        if report.vf_cost_line:
            text += f"\n\nEstimated {report.vf_cost_line}"
        msg.setText(text)
        override = QCheckBox("Proceed and let the engine build view factors "
                             "from the current mesh")
        msg.setCheckBox(override)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok |
                               QMessageBox.StandardButton.Cancel)
        msg.exec()
        if not override.isChecked():
            return False

    return True
