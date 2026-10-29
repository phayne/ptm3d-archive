"""Find the view-factor matrices that already sit next to a mesh.

A user who picks a terrain mesh has, very often, the matrix for it already on
disk -- built by a previous run, copied from a colleague, or delivered with the
DEM. Nothing in this project told them so. The View-factor cache box stayed
empty, the run rebuilt a multi-gigabyte matrix that was already there, and on a
full core+ring lunar mesh that rebuild is the reliable GPU-wedge trigger the
whole reuse doctrine exists to prevent.

So: when a mesh is chosen, LOOK. In the mesh's own directory, and in the
directory the configuration's ``radiation.view_factor_file`` already points at,
because a user who set that key once usually keeps their matrices together.

Three rules this module follows
-------------------------------

**Offer, never impose.** Nothing here writes a configuration. It returns a
ranked list and the caller shows it; one click wires one candidate. A matrix
selected for somebody silently is a matrix nobody checked, and the mesh a
matrix was built for is not recoverable from its name.

**Refuse with the reason, never with a shrug.** A file whose row count
disagrees with the mesh is REFUSED and says by how much. That is the
wrong-mesh-same-size trap from the other side: two A3GT sites at the same DEM
posting produce meshes of DIFFERENT face counts, so the count catches most
mismatches -- but two matrices that happen to agree on the count do not agree
on the geometry, and only a ``.csr`` provenance sidecar can say which mesh a
matrix was built for. That is why the sidecar outranks everything.

**Cheap.** This runs from a Qt slot when a path changes. Every probe is
``os.stat`` first, then at most 32 bytes of header, one 8-byte read from the
end of an H-matrix file, and one small JSON sidecar. Nothing is read whole. A
cloud placeholder (``st_blocks == 0`` with ``st_size > 0``) is refused BEFORE
any read, because reading one blocks the window while OneDrive hydrates it --
which on a 19.9 GB matrix is not a pause, it is a hang.

What a rank does and does not claim
-----------------------------------

The ranks are about IDENTITY EVIDENCE, not about physics quality:

===== ============================ ==============================================
rank  tier                          what has actually been checked
===== ============================ ==============================================
0     ``provenance``                a ``.csr`` whose ``.provenance.json`` sidecar
                                    NAMES this mesh, whose recorded ``geomhash``
                                    equals the one in the matrix header, and
                                    whose face count equals the mesh's
1     ``vffm-v2``                   sparse VFFM, on-disk version 2, rows == faces
2     ``hmat``                      H-matrix (``HMAT``/v2), rows == faces
3     ``csr-rows``                  ``.csr``, rows == faces, NO sidecar
4     ``vffm-v1``                   sparse VFFM version 1, rows == faces
5     ``empty``                     rows == faces and nnz == 0 -- a base stub
===== ============================ ==============================================

**The mesh's own geometry hash is never recomputed here.** It is
``sum(A_i) + sum((i+1) * A_i) * 1e-9`` over every face area
(``radiation_exchange.cpp:283``), so computing it means reading the whole mesh
and re-deriving 2.4 million triangle areas in the engine's own face order. That
is not a keystroke's worth of work, and a second implementation of it would be
a second place for it to be subtly wrong. The engine computes it at load and
ABORTS on a mismatch (``radiation_exchange.cpp:298``), so a foreign matrix is
caught -- loudly, before any physics. Rank 0 says the sidecar and the matrix
agree with each other and name this mesh; it does not say the mesh was
re-measured.

**HMAT carries no physics-generation stamp.** The sparse VFFM header does
(``view_factor.cpp:517-525``: same bytes, version 1 vs 2, different near-field
estimator), and the engine announces a v1 load. The H-matrix header has magic,
version, dimensions and block counts, and its version is a FORMAT number that
has been 2 since the format existed -- so an HMAT file that loads tells you
nothing about which estimator generation built it. Every HMAT candidate says
so. That gap is the owed ledger row named in this module's tests.

Formats, as their writers define them
-------------------------------------

* sparse VFFM -- ``view_factor.cpp:527`` --
  ``u32 magic 'VFFM' (0x5646464D), u32 version, u64 nrows, u64 ncols, u64 nnz``
* H-matrix -- ``hmatrix_view_factor.cpp:5217`` --
  ``u32 magic 'HMAT' (0x484D4154), u32 version, u64 n_rows, u64 n_cols,
  u64 n_root_blocks``, blocks, then six ``u64`` statistics of which the LAST is
  ``total_nnz`` -- hence the 8-byte read from the end.
* replacement CSR -- ``radiation_exchange.cpp:276`` --
  ``i32 n, f64 geomhash, i64 indptr[n+1], i64 nnz, i32 cols[nnz],
  f32 data[nnz]``. It has NO magic number, so a ``.csr`` is identified by its
  suffix and then by whether its declared layout accounts for its file size
  EXACTLY -- the same test the engine's truncation check performs.

No Qt, so all of this is measurable without a window.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DiscoveryResult",
    "RANK_LABELS",
    "VfCandidate",
    "candidate_wiring",
    "discover_view_factors",
    "known_defective_view_factors",
    "mesh_face_count",
    "probe_view_factor",
    "superseded_view_factors",
]

#: Suffixes worth opening. THE SUFFIX SELECTS A CANDIDATE; THE MAGIC NUMBER
#: DECIDES WHAT IT IS, and in this project those two disagree routinely.
#: ``~/projects/a3gt/campaign/base_vf/*.hmat`` are SPARSE (``VFFM``) version-1
#: files with zero nonzeros -- measured 2026-08-31 on six of them -- while the
#: current empty stubs, named ``.vf``, are H-matrix version 2. Anything here
#: that trusted the extension would report both backwards.
#:
#: (Those legacy files are the ones the stale-format fallback reports as "the old MFFV/v1 format".
#: ``MFFV`` is not a third magic number: it is ``VFFM``'s four bytes as they sit
#: on disk, little-endian. What made them a trap was the version, and the
#: current engine reads versions 1 and 2 deliberately -- ``view_factor.cpp:525``
#: -- so this module reports them rather than refusing them.)
VF_SUFFIXES = (".vf", ".hmat", ".csr")

SPARSE_VF_MAGIC = 0x5646464D          # 'VFFM', view_factor.cpp:515
HMATRIX_MAGIC = 0x484D4154            # 'HMAT', hmatrix_view_factor.cpp:4632

FORMAT_VFFM = "sparse VFFM"
FORMAT_HMAT = "H-matrix"
FORMAT_CSR = "replacement CSR"

RANK_PROVENANCE = 0
RANK_VFFM_V2 = 1
RANK_HMAT = 2
RANK_CSR_ROWS = 3
RANK_VFFM_V1 = 4
RANK_EMPTY = 5
RANK_REFUSED = 99

RANK_LABELS = {
    RANK_PROVENANCE: "provenance-verified",
    RANK_VFFM_V2: "current-generation sparse matrix",
    RANK_HMAT: "H-matrix, format-compatible",
    RANK_CSR_ROWS: "replacement CSR, unverified provenance",
    RANK_VFFM_V1: "superseded near-field generation",
    RANK_EMPTY: "empty base stub",
    RANK_REFUSED: "refused",
}

def known_defective_view_factors():
    """``{file name: why}`` -- matrices this project knows are defective.

    The table lives in ``gui_core.curated_sites`` beside the sites it belongs
    to, because it is registry data ("which artifacts does this project
    distrust") and this module's job is reading headers. Fetched through a
    function, and guarded, for the same two reasons as
    :func:`superseded_view_factors`: the format probes stay usable with no
    registry at all, and a broken registry cannot take the search down.
    """
    try:
        from gui_core.curated_sites import DEFECTIVE_VIEW_FACTORS
    except Exception as exc:                              # noqa: BLE001
        logger.info("[vf-discovery] the defective-matrix record could not be "
                    "consulted (%s); those files will not be refused.", exc)
        return {}
    return dict(DEFECTIVE_VIEW_FACTORS)

#: A radiance-PRUNED experimental matrix carries its own receipt, and for the
#: A3GT polar sites those receipts record FAILURE ("Diffuse polar terrain does
#: NOT prune safely": PSR-floor Tmax 3.3 K mean / 22.6 K max cold).
#:
#: TWO artifacts can say a matrix was pruned, and neither is guaranteed:
#: this receipt, whose key is ``radiance_prune_eps_Wm2`` (the same string
#: ``radiation_exchange.cpp:399`` greps for), and the ordinary provenance
#: sidecar, whose ``radiance_prune_eps`` is 0.0 for an unpruned matrix. Both
#: are read. Neither catches a pruned matrix that shipped with NO sidecar at
#: all -- ``polar_site_A_rad1e3_vf.csr`` is exactly that, 787,161,229 nonzeros
#: against the full matrix's 2,484,716,524 and nothing on disk beside it -- so
#: the registry's own superseded list below is what covers that case, and a
#: pruned matrix nobody has recorded anywhere is still offered. Said plainly
#: rather than claimed away: this catches the pruned matrices this project
#: knows about, not every possible one.
PRUNE_RECEIPT_SUFFIX = ".prune_receipt.json"

def superseded_view_factors():
    """``{path: why}`` -- matrices the site registry records as NOT used.

    ``gui_core.curated_sites`` already carries, per site, every matrix in the
    same directory that a curated entry deliberately does not wire, each with
    its reason. Not consulting it here was a measured defect rather than a
    missed nicety: in the real lunar polar site A directory both
    ``polar_site_A_corrected_vf_divisor2.csr`` and ``..._v3.csr`` reach the
    provenance-verified tier -- their sidecars are identical on every field
    this module ranks on -- so an alphabetical tie-break put the SUPERSEDED
    one first and "Use this file" wired it.

    Imported here rather than at module scope so the format probes stay usable
    with no registry at all, and so a broken registry cannot take the search
    down with it.
    """
    try:
        from gui_core.curated_sites import CURATED_SITES
    except Exception as exc:                              # noqa: BLE001
        logger.info("[vf-discovery] the site registry could not be consulted "
                    "(%s); superseded matrices will not be demoted.", exc)
        return {}
    out = {}
    for entry in CURATED_SITES:
        for item in getattr(entry, "superseded", ()):
            out[str(item.path)] = f"{entry.key}: {item.why}"
    return out


#: A directory with more entries than this is scanned only this far. A mesh
#: directory is normally tens of files; a user who points at a data root with
#: a hundred thousand entries should get a bounded answer and a note, not a
#: frozen window.
SCAN_LIMIT = 4000


# ---------------------------------------------------------------------------
# 1. Probing one file
# ---------------------------------------------------------------------------

class VfCandidate:
    """One view-factor file, what it is, and whether it fits this mesh."""

    def __init__(self, path, fmt="", version=None, rows=None, cols=None,
                 nnz=None, geomhash=None, size_bytes=0, dataless=False,
                 sidecar=None, rank=RANK_REFUSED, refusal="", notes=(),
                 superseded=""):
        self.path = Path(path)
        self.fmt = fmt
        self.version = version
        self.rows = rows
        self.cols = cols
        #: None means "not knowable cheaply for this format", which is a
        #: different statement from zero and must not be rendered as one.
        self.nnz = nnz
        self.geomhash = geomhash
        self.size_bytes = int(size_bytes)
        self.dataless = bool(dataless)
        #: The parsed ``.provenance.json`` next to a CSR, or None.
        self.sidecar = sidecar
        self.rank = rank
        #: Non-empty means this file is NOT offered, and this is why.
        self.refusal = refusal
        #: Things true of this candidate that the user has to be told even
        #: though they do not stop it being used.
        self.notes = list(notes)
        #: Non-empty when the site registry records this exact file as one it
        #: deliberately does not wire. It is still OFFERED -- a superseded
        #: matrix is a real matrix and reproducing an older run is a legitimate
        #: thing to want -- but it sorts BELOW everything that is not
        #: superseded, whatever their formats, so it is never what a default
        #: selection picks up.
        self.superseded = superseded

    @property
    def ok(self):
        return not self.refusal

    @property
    def rank_label(self):
        return RANK_LABELS.get(self.rank, "")

    def headline(self):
        """One line: what this file is, in the terms that decided its rank."""
        if self.refusal:
            return f"{self.path.name} — REFUSED: {self.refusal}"
        bits = [self.fmt]
        if self.version is not None:
            bits.append(f"v{self.version}")
        if self.rows is not None:
            bits.append(f"{self.rows:,} rows")
        if self.nnz is not None:
            bits.append(f"{self.nnz:,} nonzeros")
        label = ("SUPERSEDED — " + self.rank_label if self.superseded
                 else self.rank_label)
        return f"{self.path.name} — {', '.join(bits)} — {label}"

    def __repr__(self):                                   # pragma: no cover
        return f"<VfCandidate {self.path.name} rank={self.rank}>"


def _stat(path):
    try:
        return os.stat(path)
    except OSError:
        return None


def _is_dataless(stat_result):
    """A cloud placeholder: a name with no bytes behind it on this disk.

    Uses the stat this module already took rather than
    ``preflight.is_dataless``, which takes its own -- one stat per file is the
    budget, and the answer must come from the same one the size came from.
    """
    return bool(stat_result) and stat_result.st_blocks == 0 and stat_result.st_size > 0


def _read_head(path, nbytes):
    try:
        with open(path, "rb") as handle:
            return handle.read(nbytes)
    except OSError:
        return b""


def _read_at(path, offset, nbytes):
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            return handle.read(nbytes)
    except OSError:
        return b""


def _load_sidecar(csr_path):
    """The ``.provenance.json`` beside a CSR, parsed, or None.

    Written by ``scripts/ptm12_diag/assemble_vf_haworth.py``; it records the
    mesh the sweep ran against, the face count, the geometry hash, the ray
    count and the coverage gate's own result. It is the only artifact in this
    project that says which MESH a matrix belongs to.
    """
    sidecar = Path(str(csr_path) + ".provenance.json")
    stat_result = _stat(sidecar)
    if stat_result is None or _is_dataless(stat_result):
        return None
    if stat_result.st_size > 1 << 20:
        # A provenance record is under a kilobyte. Something megabyte-sized
        # under that name is not one, and is not going to be parsed here.
        logger.info("[vf-discovery] %s is %d bytes, too large to be a "
                    "provenance sidecar; ignored.", sidecar,
                    stat_result.st_size)
        return None
    try:
        return json.loads(sidecar.read_text())
    except (OSError, ValueError) as exc:
        logger.info("[vf-discovery] %s could not be read as JSON (%s); the "
                    "matrix is still offered, without provenance.",
                    sidecar, exc)
        return None


def _prune_receipt(path):
    receipt = Path(str(path) + PRUNE_RECEIPT_SUFFIX)
    stat_result = _stat(receipt)
    if stat_result is None or _is_dataless(stat_result) or stat_result.st_size > 1 << 20:
        return None
    try:
        return json.loads(receipt.read_text())
    except (OSError, ValueError):
        return None


class Records:
    """The registry's two verdict tables, read once for a whole scan.

    Both are static for the life of a process, and both were being rebuilt per
    CANDIDATE -- once per file in the directory, each rebuild walking the whole
    site table. A scan of a directory of sixty per-site stubs did that sixty
    times for one answer.

    A plain object rather than a closure or a module-level cache: a closure
    over the scan would keep the enclosing frame alive for as long as any
    candidate it produced, and a module cache would hide a registry a test had
    replaced.
    """

    def __init__(self, defective=None, superseded=None):
        self.defective = (known_defective_view_factors()
                          if defective is None else dict(defective))
        self.superseded = (superseded_view_factors()
                           if superseded is None else dict(superseded))


def probe_view_factor(path, mesh_faces=None, mesh_path=None, records=None):
    """Identify one file and decide whether it fits ``mesh_faces``.

    ``mesh_faces`` of None means "identify it, do not judge the fit" -- the
    candidate comes back with its format and dimensions and no rank.

    ``records`` is the registry's verdict tables. Left out, they are read here;
    :func:`discover_view_factors` reads them once and passes them in.

    Reads at most 32 bytes of header, 8 bytes from the end (H-matrix), 8 bytes
    at one seek (CSR nnz), and one small JSON sidecar. Never the file.
    """
    path = Path(path)
    records = Records() if records is None else records
    stat_result = _stat(path)
    if stat_result is None:
        return VfCandidate(path, refusal="it is not on this disk")
    if _is_dataless(stat_result):
        return VfCandidate(
            path, size_bytes=stat_result.st_size, dataless=True,
            refusal=("it is a cloud placeholder — the name is here, the bytes "
                     "are not. Reading it would block while it downloads. "
                     "Right-click it in Finder and choose 'Always Keep on "
                     "This Device' first."))
    if stat_result.st_size < 24:
        return VfCandidate(path, size_bytes=stat_result.st_size,
                           refusal="it is too short to hold a view-factor "
                                   "header")

    defect = records.defective.get(path.name)
    if defect:
        return VfCandidate(path, size_bytes=stat_result.st_size,
                           refusal=defect)

    receipt = _prune_receipt(path)
    if receipt:
        # `radiance_prune_eps_Wm2` is the receipt's key -- the same string
        # `radiation_exchange.cpp:399` greps for. `radiance_prune_eps` is the
        # PROVENANCE sidecar's key for the same quantity, and reading one
        # schema with the other's name printed "eps=?" on every real receipt.
        eps = receipt.get("radiance_prune_eps_Wm2",
                          receipt.get("radiance_prune_eps", "?"))
        verdict = str(receipt.get("MEASURED_AGT_counter_example", "")
                      or receipt.get("verdict", "")).strip()
        return VfCandidate(
            path, size_bytes=stat_result.st_size,
            refusal=(f"it is a radiance-PRUNED experimental matrix "
                     f"(eps={eps} W/m^2) and its own receipt beside it records "
                     f"the outcome: {verdict or 'see the receipt file'}. A "
                     f"pruned matrix is not a cheaper version of the full "
                     f"one."))

    if path.suffix.lower() == ".csr":
        candidate = _probe_csr(path, stat_result, mesh_faces, mesh_path)
    else:
        candidate = _probe_magic(path, stat_result, mesh_faces)
    return _mark_superseded(candidate, records)


def _mark_superseded(candidate, records):
    """Tag a candidate the site registry records as deliberately NOT used."""
    if not candidate.ok:
        return candidate
    why = records.superseded.get(str(candidate.path))
    if why:
        candidate.superseded = why
        candidate.notes.insert(0, "THIS PROJECT'S OWN RECORD SAYS THIS FILE "
                                  "IS SUPERSEDED. " + why)
    return candidate


def _probe_magic(path, stat_result, mesh_faces):
    head = _read_head(path, 32)
    if len(head) < 24:
        return VfCandidate(path, size_bytes=stat_result.st_size,
                           refusal="its header could not be read")
    magic, version = struct.unpack("<II", head[:8])
    rows, cols = struct.unpack("<QQ", head[8:24])

    if magic == SPARSE_VF_MAGIC:
        nnz = (struct.unpack("<Q", head[24:32])[0] if len(head) >= 32 else None)
        notes = []
        if version < 2:
            notes.append(
                "on-disk version 1. The byte layout is the same as version 2 "
                "and the engine loads it — but the version is a PHYSICS stamp, "
                "not a format number: a v1 file's near-field entries came from "
                "the one-sided estimator retired on 2026-08-23, which reads "
                "about -22% low on edge- and vertex-touching pairs and -3.9% "
                "mass-weighted over the near field. The run reproduces THAT "
                "physics and its output does not say so.")
        elif version > 2:
            return VfCandidate(
                path, FORMAT_VFFM, version, rows, cols, nnz,
                size_bytes=stat_result.st_size,
                refusal=(f"its on-disk version is {version} and this build "
                         f"reads 1..2, so the engine would refuse it and "
                         f"rebuild the matrix instead"))
        rank = RANK_VFFM_V2 if version >= 2 else RANK_VFFM_V1
        return _judge(VfCandidate(path, FORMAT_VFFM, version, rows, cols, nnz,
                                  size_bytes=stat_result.st_size, rank=rank,
                                  notes=notes), mesh_faces)

    if magic == HMATRIX_MAGIC:
        if version != 2:
            return VfCandidate(
                path, FORMAT_HMAT, version, rows, cols,
                size_bytes=stat_result.st_size,
                refusal=(f"its H-matrix version is {version} and this build "
                         f"reads 2 only. The engine FAILS to load it and then "
                         f"rebuilds the matrix from scratch, the recompute "
                         f"wedge this check exists to prevent."))
        # `total_nnz_` is the last of six size_t statistics written after the
        # blocks (hmatrix_view_factor.cpp:5270-5275), so it is one 8-byte read
        # from the end of the file rather than a parse of every block.
        tail = _read_at(path, stat_result.st_size - 8, 8)
        nnz = struct.unpack("<Q", tail)[0] if len(tail) == 8 else None
        notes = [
            "the H-matrix header carries NO physics-generation stamp — magic, "
            "format version, dimensions and block counts, and its version has "
            "been 2 for the life of the format. So this file being loadable "
            "says it is FORMAT-COMPATIBLE with this mesh; which view-factor "
            "estimator generation built it is unverifiable from the file. The "
            "sparse format does carry that stamp; this one does not."]
        return _judge(VfCandidate(path, FORMAT_HMAT, version, rows, cols, nnz,
                                  size_bytes=stat_result.st_size,
                                  rank=RANK_HMAT, notes=notes), mesh_faces)

    return VfCandidate(
        path, size_bytes=stat_result.st_size,
        refusal=(f"its first four bytes are {head[:4]!r} (0x{magic:08x}), and "
                 f"the engine reads two magic numbers: 'VFFM' for the sparse "
                 f"format and 'HMAT' for the H-matrix. A file it cannot "
                 f"identify is not an error the run stops on — the loader "
                 f"returns false and the matrix is REBUILT from scratch, "
                 f"which on a site-scale surface is hours and tens of "
                 f"gigabytes."))


def _probe_csr(path, stat_result, mesh_faces, mesh_path):
    """A replacement CSR, which has no magic number and so must be proved.

    The proof is the layout: ``12 + 8*(n+1) + 8 + 4*nnz + 4*nnz`` has to equal
    the file size EXACTLY. That is the same arithmetic the engine's own
    truncation check performs after reading each array, and it is what
    distinguishes this format from any other file somebody named ``.csr``.
    """
    head = _read_head(path, 12)
    if len(head) < 12:
        return VfCandidate(path, size_bytes=stat_result.st_size,
                           refusal="its header could not be read")
    n, = struct.unpack("<i", head[:4])
    geomhash, = struct.unpack("<d", head[4:12])
    if n <= 0:
        return VfCandidate(path, FORMAT_CSR, size_bytes=stat_result.st_size,
                           refusal=(f"it declares {n} rows, which is not a "
                                    f"face count. This is not a PTM3D "
                                    f"replacement CSR."))
    nnz_offset = 12 + 8 * (n + 1)
    if stat_result.st_size < nnz_offset + 8:
        return VfCandidate(path, FORMAT_CSR, rows=n, geomhash=geomhash,
                           size_bytes=stat_result.st_size,
                           refusal=(f"it declares {n:,} rows, which needs at "
                                    f"least {nnz_offset + 8:,} bytes of index "
                                    f"array, and the file is "
                                    f"{stat_result.st_size:,} bytes. It is "
                                    f"truncated or it is not this format."))
    tail = _read_at(path, nnz_offset, 8)
    if len(tail) < 8:
        return VfCandidate(path, FORMAT_CSR, rows=n, geomhash=geomhash,
                           size_bytes=stat_result.st_size,
                           refusal="its nonzero count could not be read")
    nnz, = struct.unpack("<q", tail)
    expected = nnz_offset + 8 + 8 * nnz          # int32 column + float32 value
    if nnz < 0 or expected != stat_result.st_size:
        return VfCandidate(
            path, FORMAT_CSR, rows=n, nnz=nnz, geomhash=geomhash,
            size_bytes=stat_result.st_size,
            refusal=(f"its declared layout ({n:,} rows, {nnz:,} nonzeros) "
                     f"accounts for {expected:,} bytes and the file is "
                     f"{stat_result.st_size:,}. The engine reads this format "
                     f"array by array and aborts on the shortfall; it is "
                     f"truncated, or it is a different kind of .csr."))

    candidate = VfCandidate(path, FORMAT_CSR, rows=n, cols=n, nnz=nnz,
                            geomhash=geomhash, size_bytes=stat_result.st_size,
                            rank=RANK_CSR_ROWS)
    sidecar = _load_sidecar(path)
    candidate.sidecar = sidecar
    # The SECOND place a prune is recorded. A pruned matrix built through the
    # ordinary assembler carries it here rather than in a separate receipt, and
    # 0.0 is what every unpruned matrix records.
    if sidecar is not None:
        try:
            eps = float(sidecar.get("radiance_prune_eps", 0.0) or 0.0)
        except (TypeError, ValueError):
            eps = 0.0
        if eps > 0.0:
            candidate.refusal = (
                f"its own provenance records radiance_prune_eps = {eps} W/m^2: "
                f"this matrix had its small entries DISCARDED at assembly. On "
                f"the diffuse polar terrain these matrices were built for that "
                f"was measured to cool the permanently-shadowed floor by 3.3 K "
                f"on average and 22.6 K at worst. A pruned matrix is not a "
                f"cheaper version of the full one.")
            candidate.rank = RANK_REFUSED
            return candidate
    if sidecar is None:
        candidate.notes.append(
            "no .provenance.json sidecar beside it, so the only thing checked "
            "here is the face count. Which mesh this matrix was built for is "
            "not recoverable from the file; the engine recomputes the mesh's "
            "geometry hash at load and ABORTS if it disagrees, and that check "
            "— not this one — is what catches a foreign matrix.")
        return _judge(candidate, mesh_faces)

    built = sidecar.get("assembled_utc", "at an unrecorded time")
    assembler = Path(str(sidecar.get("assembler", "?"))).name
    candidate.notes.append(f"built {built} by {assembler}, "
                           f"{sidecar.get('nrays', '?')} rays/face, estimator "
                           f"{sidecar.get('estimator', 'unrecorded')}")
    return _judge(_rank_by_sidecar(candidate, sidecar, mesh_path), mesh_faces)


def _rank_by_sidecar(candidate, sidecar, mesh_path):
    """Promote a CSR to rank 0 only when the sidecar agrees with BOTH sides.

    Three conditions, and the ranking says which one failed rather than
    dropping the candidate to a lower tier with no explanation:

    1. the sidecar's ``geomhash`` equals the one in the matrix header,
    2. the sidecar's ``n_faces`` equals the matrix's declared rows,
    3. the sidecar's ``mesh`` resolves to the mesh being wired.

    (3) is what the whole tier is FOR, and it is also the one that can be
    absent -- a caller identifying a file without a mesh in hand passes
    ``mesh_path=None``, and then this stays at the rows-only tier rather than
    claiming a match against nothing.
    """
    recorded_hash = sidecar.get("geomhash")
    recorded_faces = sidecar.get("n_faces")
    recorded_mesh = str(sidecar.get("mesh", "") or "")

    try:
        hash_agrees = (recorded_hash is not None
                       and abs(float(recorded_hash) - float(candidate.geomhash))
                       <= 1e-6 * abs(float(candidate.geomhash)))
    except (TypeError, ValueError):
        hash_agrees = False
    if not hash_agrees:
        candidate.notes.append(
            f"its provenance sidecar records geomhash {recorded_hash!r} and "
            f"the matrix header carries {candidate.geomhash!r}. The two "
            f"describe different geometry, so the sidecar is not evidence "
            f"about this file and has not been used to rank it.")
        return candidate
    if recorded_faces is not None and int(recorded_faces) != int(candidate.rows):
        candidate.notes.append(
            f"its provenance sidecar records {int(recorded_faces):,} faces and "
            f"the matrix header declares {candidate.rows:,}. Not used to rank "
            f"it.")
        return candidate

    if not mesh_path:
        candidate.notes.append(
            "it has a provenance sidecar naming "
            f"{Path(recorded_mesh).name or 'a mesh'}, but no mesh was supplied "
            "to compare it against, so it is ranked on its face count only.")
        return candidate

    same, how = _same_mesh(recorded_mesh, mesh_path)
    if same:
        candidate.rank = RANK_PROVENANCE
        candidate.notes.append(
            f"its provenance sidecar names THIS mesh ({how}) and its recorded "
            f"geometry hash matches the matrix header. That is the strongest "
            f"identity evidence available without reading the mesh; the "
            f"engine still recomputes the hash from the face areas at load "
            f"and aborts on a mismatch.")
    else:
        candidate.notes.append(
            f"its provenance sidecar says it was built for "
            f"{recorded_mesh or 'an unrecorded mesh'}, which is not the mesh "
            f"being wired. The face count agrees, which is exactly the "
            f"wrong-mesh-same-size case: ranked on the count alone, and the "
            f"engine's geometry-hash gate is what will settle it.")
    return candidate


def _same_mesh(recorded, wired):
    """Is the mesh a sidecar names the mesh being wired? ``(bool, how)``.

    Three rungs, and ``how`` says which one answered, because they are not
    equally strong and a card that only printed "matched" would flatten them:

    1. **the same file** -- same device and inode, so symlinks and two
       spellings of one path both resolve;
    2. **an identical copy elsewhere** -- same file name and same byte length.
       This rung is not academic and was not anticipated: every real A3GT
       provenance sidecar records the mesh under
       ``<repo>/data/dem/a3gt_v2/``, and every production configuration loads
       it from ``~/projects/a3gt/meshes/``. Measured 2026-08-31, those two
       copies of ``polar_site_A_R3km_linear.ply`` are
       byte-identical (md5 ``a256fbc7…``) -- so rung 1 alone would have
       refused every genuine match this project has.
    3. **neither** -- the sidecar belongs to a different mesh, or to a copy
       this cannot recognize, and the candidate stays at the face-count tier.

    Rung 2 does NOT prove the bytes are identical; it proves the name and the
    length are, which two independently-built meshes of one site would not be.
    It is stated as what it is, and it is not the last line of defense: the
    engine recomputes the geometry hash from THIS mesh's own face areas at
    load and aborts on a mismatch (``radiation_exchange.cpp:298``).
    """
    if not recorded or not wired:
        return False, ""
    pa, pb = Path(recorded).expanduser(), Path(wired).expanduser()
    sa, sb = _stat(pa), _stat(pb)
    if sa is not None and sb is not None:
        if (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino):
            return True, "the same file"
        if pa.name == pb.name and sa.st_size == sb.st_size:
            return True, (f"a copy of it under another path — same name, same "
                          f"{sa.st_size:,} bytes, at {pa.parent}; the bytes "
                          f"themselves were not compared")
        return False, ""
    # One of the two is not on this disk, so neither rung above can be applied.
    # An identical spelling is the only thing left, and a same NAME alone is
    # not enough -- that is the wrong-mesh case, not evidence against it.
    try:
        if pa.resolve() == pb.resolve():
            return True, "the same path"
    except OSError:                                       # pragma: no cover
        pass
    return False, ""


def _judge(candidate, mesh_faces):
    """Apply the face-count gate and the empty-matrix demotion."""
    if mesh_faces is None:
        return candidate
    if candidate.rows is None:
        candidate.refusal = "its row count could not be read"
        candidate.rank = RANK_REFUSED
        return candidate
    if int(candidate.rows) != int(mesh_faces):
        candidate.rank = RANK_REFUSED
        candidate.refusal = (
            f"it is a {int(candidate.rows):,}-row matrix and this mesh has "
            f"{int(mesh_faces):,} faces. A view-factor matrix belongs to one "
            f"mesh; the engine refuses a mismatch rather than solving the "
            f"wrong geometry.")
        return candidate
    if candidate.nnz == 0:
        candidate.rank = RANK_EMPTY
        candidate.notes.append(
            "it is correctly dimensioned for this mesh and EMPTY (0 stored "
            "nonzeros). Loading it gives no facet-to-facet exchange at all — "
            "a shadowed crater floor is then warmed by nothing. Its purpose "
            "is to be a base the engine loads clean so it does not rebuild, "
            "while the real matrix arrives another way (the "
            "PTM_HMATRIX_VF_REPLACE CSR route). On its own it is radiation switched off with the "
            "switch left on.")
    return candidate


# ---------------------------------------------------------------------------
# 2. Scanning
# ---------------------------------------------------------------------------

def mesh_face_count(mesh_path):
    """Faces in a mesh, or None -- from the PLY header, nothing else read.

    Delegates to the pre-flight's PLY reader so the count this ranks against
    and the count the launch check refuses against are one implementation.
    Any other mesh format returns None, and then discovery identifies files
    without judging their fit rather than guessing a count.
    """
    path = Path(mesh_path)
    stat_result = _stat(path)
    if stat_result is None or _is_dataless(stat_result):
        return None
    if path.suffix.lower() != ".ply":
        return None
    try:
        from gui_core.preflight import _ply_face_count
    except ImportError:                                   # pragma: no cover
        return None
    return _ply_face_count(str(path))


class DiscoveryResult:
    """What a scan found: the offers, the refusals, and where it looked."""

    def __init__(self, mesh_path, mesh_faces, directories, candidates,
                 truncated=()):
        self.mesh_path = str(mesh_path or "")
        self.mesh_faces = mesh_faces
        self.directories = [str(d) for d in directories]
        self._all = list(candidates)
        #: Directories whose listing hit SCAN_LIMIT, so the answer is partial
        #: and says so instead of looking complete.
        self.truncated = [str(d) for d in truncated]

    @property
    def matches(self):
        """Offerable candidates, best evidence first, then by name.

        THE SUPERSEDED FLAG COMES BEFORE THE RANK, and it has to. Measured in
        the real lunar polar site A directory: ``..._corrected_vf_divisor2.csr`` and
        ``..._corrected_vf_v3.csr`` both reach the provenance-verified tier --
        their sidecars agree on the estimator, the ray count, the geometry hash
        and the mesh, and differ only in a timestamp and a nonzero count that
        nothing here ranks on -- so ordering by rank and then by path put the
        superseded one first, where a default selection picks it up.

        After that: rank, then the PATH. Never modification time. "The newest
        one" is a guess about which matrix is right, and this module does not
        make that guess -- the registry's record and the user do.
        """
        return sorted((c for c in self._all if c.ok),
                      key=lambda c: (bool(c.superseded), c.rank, str(c.path)))

    @property
    def refusals(self):
        return sorted((c for c in self._all if not c.ok), key=lambda c: str(c.path))

    @property
    def best(self):
        matches = self.matches
        return matches[0] if matches else None

    def summary(self):
        """One line for the card. Never claims a scan it did not finish."""
        where = (f"{len(self.directories)} director"
                 f"{'y' if len(self.directories) == 1 else 'ies'}")
        if self.mesh_faces is None:
            return (f"Searched {where}. This mesh's face count could not be "
                    f"read, so nothing found could be checked against it — "
                    f"any match below is a format identification only.")
        if not self._all:
            return (f"Searched {where}: no .vf, .hmat or .csr file beside this "
                    f"mesh.")
        parts = []
        if self.matches:
            parts.append(f"{len(self.matches)} match"
                         f"{'' if len(self.matches) == 1 else 'es'} for "
                         f"{self.mesh_faces:,} faces")
        if self.refusals:
            parts.append(f"{len(self.refusals)} refused")
        line = f"Searched {where}: " + ", ".join(parts) + "."
        if self.truncated:
            line += (f" {len(self.truncated)} of them held more than "
                     f"{SCAN_LIMIT:,} entries and were only read that far, so "
                     f"this list may be incomplete.")
        return line

    def __repr__(self):                                   # pragma: no cover
        return (f"<DiscoveryResult {len(self.matches)} matches, "
                f"{len(self.refusals)} refused>")


def discover_view_factors(mesh_path, view_factor_file="", project_root=None,
                          extra_dirs=()):
    """Rank the view-factor files sitting next to ``mesh_path``.

    Two places are searched, and no more: the mesh's own directory, and the
    directory ``view_factor_file`` points at when the configuration already has
    one. Not recursive -- a recursive walk of a DEM tree is not a keystroke's
    worth of work, and a matrix three directories away from its mesh is not
    "beside" it in any sense a user meant.

    Returns a :class:`DiscoveryResult`. It writes nothing.
    """
    mesh_path = str(mesh_path or "").strip()
    if not mesh_path:
        return DiscoveryResult("", None, [], [])

    from gui_core.preflight import resolve_data_path, resolve_path
    root = Path(project_root) if project_root else Path.cwd()
    mesh_abs = resolve_data_path(mesh_path, root)
    faces = mesh_face_count(mesh_abs)

    directories, seen = [], set()
    def _add(directory):
        if directory is None:
            return
        stat_result = _stat(directory)
        if stat_result is None or not os.path.isdir(directory):
            return
        key = (stat_result.st_dev, stat_result.st_ino)
        if key in seen:
            return
        seen.add(key)
        directories.append(Path(directory))

    _add(mesh_abs.parent)
    if str(view_factor_file or "").strip():
        # `radiation.view_factor_file` deliberately does NOT follow the
        # PTM_DATA_ROOT contract (`preflight._ENGINE_DATA_ROOT_FIELDS`), so it
        # gets `resolve_path`. Using the data-root resolver here would look in
        # a directory the run never opens.
        _add(resolve_path(str(view_factor_file), root).parent)
    for extra in extra_dirs:
        _add(Path(str(extra)))

    candidates, truncated = [], []
    records = Records()
    for directory in directories:
        found = 0
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    found += 1
                    if found > SCAN_LIMIT:
                        truncated.append(directory)
                        break
                    if not entry.is_file(follow_symlinks=True):
                        continue
                    if Path(entry.name).suffix.lower() not in VF_SUFFIXES:
                        continue
                    candidates.append(probe_view_factor(
                        Path(entry.path), mesh_faces=faces,
                        mesh_path=str(mesh_abs), records=records))
        except OSError as exc:
            logger.info("[vf-discovery] %s could not be listed (%s); skipped.",
                        directory, exc)
    return DiscoveryResult(mesh_abs, faces, directories, candidates, truncated)


# ---------------------------------------------------------------------------
# 3. Turning a choice into configuration keys
# ---------------------------------------------------------------------------

def candidate_wiring(candidate, stub_path=""):
    """``[(dotted key, value), ...]`` for wiring ONE candidate. Nothing else.

    A ``.vf``/``.hmat`` goes straight into ``radiation.view_factor_file``.

    A ``.csr`` does not, and cannot: the engine's own loader has no reader for
    that layout. It is installed through the environment variable
    ``PTM_HMATRIX_VF_REPLACE`` (``radiation_exchange.cpp:280``), and it needs a
    correctly-dimensioned base file in ``radiation.view_factor_file`` as well
    -- otherwise the loader finds no cache and rebuilds the multi-gigabyte
    matrix it was handed a replacement for. Without a ``stub_path`` this
    returns the env line alone and the caller must say the base is still
    missing; inventing one here would be a file this function wrote without
    being asked.
    """
    if candidate is None or not candidate.ok:
        return []
    if candidate.fmt != FORMAT_CSR:
        return [("radiation.view_factor_file", str(candidate.path))]
    wiring = [("engine_env.PTM_HMATRIX_VF_REPLACE", str(candidate.path))]
    if str(stub_path or "").strip():
        wiring.insert(0, ("radiation.view_factor_file", str(stub_path)))
    return wiring
