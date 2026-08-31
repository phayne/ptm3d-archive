"""Lunar sites whose mesh, occluder and view-factor matrix are a PROVEN set.

This is the second tier of the curated registry. ``curated_sources.py`` covers
bodies whose EPHEMERIS and SHAPE a live service correctly refuses to serve
(Bennu). This module covers the other half of the same problem on the Moon: a
site whose terrain mesh, shadow-only occluder ring and view-factor matrix only
mean anything TOGETHER, and where picking two of the three by hand produces a
run that completes and is wrong.

Why a registry rather than three file pickers
---------------------------------------------

The A3GT lunar-polar sites are the sharpest case in this project's history.
Every one of the following is a real, measured, silent failure from that
campaign, and a file picker offers all of them:

* **The occluder ring is not one ring.** A03MP026 has three
  (``Rext20km``, ``Rext27.3108km``, ``Rext81.6712km``). Its production runs use
  the 27.3108 km one as ``mesh.occluder_file``; its view-factor SWEEP used the
  81.6712 km one as a ray blocker. Those are different jobs -- run-time shadow
  BVH versus offline Monte-Carlo occlusion -- and the bigger ring is NOT the
  better choice for the first. Nothing on disk says which is which.
* **A matrix can be EMPTY and load clean.** The June ``view_factors.hmat``
  for this site has ``nnz=0``: dimensions match, the loader is happy, radiation
  exchange silently computes nothing (project map, "Trap directories").
* **A matrix can be for the wrong mesh and still be the right SIZE.** The DEM
  was regenerated on 12 June, after its own matrix: 2,397,890 faces against the
  matrix's 1,796,406. That one is caught, by the count. Two matrices that agree
  on the count are not.
* **A matrix can be a pruned experiment whose own receipt records failure.**
  Three sit in the same directory as the good one, and one of them says in its
  receipt: "MEASURED FAIL: PSR-floor Tmax mean 3.3 K / max 22.6 K cold ...
  Diffuse polar terrain does NOT prune safely."
* **Rebuilding, instead of loading, is the wedge.** A multi-gigabyte matrix
  rebuilt on a full core+ring lunar mesh hard-wedged the graphics cards on
  2026-07-03 and needed a reboot. That is what PTM-19 exists to prevent, and a
  wrong or missing cache path is how it happens by accident.

So the entries below are not conveniences. Each is a set that RAN, to
completion, in a campaign that was checked afterwards.

The three rules, the same as the Bennu tier's
---------------------------------------------

**Nothing is invented.** Every face count, nonzero count and geometry hash
below was read off the file it describes on 2026-08-31, and
``src/python/tests/test_gui_curated_sites.py`` re-reads them. A number that
drifts from its file fails the suite rather than being believed.

**Being curated never means being quiet.** A curated site says which matrix
generation it wires, which SUPERSEDED matrices sit in the same directories, and
what it refuses to point at and why.

**Local-only entries degrade, they do not pretend.** ``~/projects/a3gt`` is not
distributed with PTM3D -- 220 GB of DEM-derived meshes and matrices on one
machine. Every entry here is marked ``local_only`` and its status is computed
from the disk in front of the user, so a checkout somewhere else shows the
entry, says the artifacts are not here, and refuses to wire it. It does not
write paths that do not exist and let the launch check find out.

Why the ``v3`` matrices, and not the ones the project map names first
--------------------------------------------------------------------

The project map's A3GT row points at
``prune_test/vfr_full/A03MP026/A03MP026_corrected_vf.csr`` -- the July matrix,
19.9 GB, 2,484,716,524 nonzeros. It is real, it ran, and it is SUPERSEDED. On
2026-08-19/20 all three sites were re-swept and re-assembled with the FC-08
symmetrization-divisor fix, and the GEN-2 campaign ran those instead -- 9
windows plus a merge plus 17 COGs per site, all three verified afterwards
(``run_manifest_A03*.json`` in ``~/projects/agt_corrected_2026-08-19``, whose
``vf_replace`` field names the ``_v3`` file for every window of every site).

Curating the July matrix for A03MP026 while curating ``_v3`` for the other two
would ship three sites on two incompatible view-factor generations, from one
table, with nothing saying so. The July matrix also has NO provenance sidecar,
so the strongest identity check this project has cannot be applied to it. It is
recorded below as superseded, in the same way Bennu's solution-76 kernel is:
named, explained, and NOT wired.

No Qt, so the whole tier is measurable without a window.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CURATED_SITES",
    "CuratedSite",
    "CuratedSiteMesh",
    "CuratedSiteOccluder",
    "CuratedSiteViewFactors",
    "DEFECTIVE_VIEW_FACTORS",
    "NOT_CURATED",
    "SiteStatus",
    "SupersededArtifact",
    "curated_site",
    "curated_sites_for_body",
    "site_status",
    "site_wiring",
    "wiring_provenance",
]

#: The placement sentinel. ``mesh.latitude_override > -900.0`` makes the engine
#: ROTATE the loaded mesh so its vertex-mean direction lands on the given
#: latitude and longitude (``ptm_thermal_sim.cpp:15634`` ->
#: ``rotateMeshToLatLon`` at :1799). Every site here is a DEM mesh already built
#: in the lunar body frame at its own coordinates, so it must not be rotated --
#: see :class:`CuratedSiteMesh` for what happens if it is.
NOT_PLACED = -999.0


class CuratedSiteMesh:
    """The high-resolution core terrain, and where it already sits.

    ``latitude_deg``/``longitude_deg`` are the site's coordinates FOR DISPLAY
    and for naming the run. They are deliberately NOT written into
    ``mesh.latitude_override``/``longitude_override``, and that is the least
    obvious decision in this module:

    An A3GT core mesh is built from a south-polar-stereographic DEM into lunar
    body-frame Cartesian coordinates, so it is already at its own latitude and
    longitude. Measured 2026-08-31 from the vertex mean of each PLY: A03MP026
    lands at -84.2234, 33.5023 against a file name that says 84.223S, 33.502E.
    Writing the rounded name back as an override therefore asks the engine to
    rotate the mesh by about 0.0004 degrees -- past the 1e-10 radian no-op
    guard, so it rotates -- which at 1737 km is roughly a 12 m displacement.

    And only the CORE moves. ``rotateMeshToLatLon`` takes the core mesh alone;
    the occluder ring is loaded separately and is not rotated with it, so the
    two would come apart. The face AREAS are unchanged by a rotation, so the
    replacement CSR's geometry-hash gate stays green through all of it. It is
    silent by construction.

    Every production configuration for these sites -- July's
    ``sites_v3_corrected/*/yaml/*.yaml`` and August's GEN-2 ``configs/*/*.yaml``
    -- leaves both keys unset for exactly this reason. So does this registry:
    it writes the ``-999.0`` sentinel and shows the coordinates as text.
    """

    def __init__(self, path, faces, latitude_deg, longitude_deg,
                 core_radius_km, source, measured_latitude_deg=None,
                 measured_longitude_deg=None):
        self.path = str(path)
        self.faces = int(faces)
        self.latitude_deg = float(latitude_deg)
        self.longitude_deg = float(longitude_deg)
        self.core_radius_km = float(core_radius_km)
        #: How the terrain was derived. A LOLA-only fallback and an SfS product
        #: are not the same resolution class, and one of the three sites here
        #: is 200x coarser than the other two.
        self.source = source
        #: The vertex-mean direction actually measured from the PLY, kept so
        #: the test can re-measure it and so the card can show the difference
        #: rather than the file name's rounding.
        self.measured_latitude_deg = (float(measured_latitude_deg)
                                      if measured_latitude_deg is not None
                                      else None)
        self.measured_longitude_deg = (float(measured_longitude_deg)
                                       if measured_longitude_deg is not None
                                       else None)

    def coordinates(self):
        ns = "S" if self.latitude_deg < 0 else "N"
        return f"{abs(self.latitude_deg):.3f}{ns}, {self.longitude_deg:.3f}E"

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedSiteMesh {Path(self.path).name}>"


class CuratedSiteOccluder:
    """The shadow-only ring, which is an ANNULUS and not a second terrain.

    ``data/dem/a3gt_v2/<SITE>_ring_Rcore<N>_Rext<M>km.ply`` has zero vertices
    inside about ``R_core`` of the site center: it pairs with the core through
    ``mesh.occluder_file`` and the two are never merged. Loading one alone
    renders a doughnut with a hole where the site is.

    ``r_ext_km`` is the OUTER radius, and the entry records which ring the
    production runs used rather than the largest one available -- see the
    module docstring.
    """

    def __init__(self, path, faces, r_ext_km, role):
        self.path = str(path)
        self.faces = int(faces)
        self.r_ext_km = float(r_ext_km)
        self.role = role

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedSiteOccluder {Path(self.path).name}>"


class CuratedSiteViewFactors:
    """The matrix pair: a replacement CSR and the base stub it needs.

    THE STUB IS NOT OPTIONAL AND IS NOT COSMETIC. The engine has no reader for
    the replacement-CSR layout in its normal cache path; the CSR arrives
    through ``PTM_HMATRIX_VF_REPLACE`` (``radiation_exchange.cpp:280``) and the
    normal loader still runs. With ``radiation.view_factor_file`` empty or
    missing, that loader finds no cache and REBUILDS -- multi-gigabyte, on a
    full core+ring lunar mesh, which is the documented compute wedge. The stub
    is a correctly-dimensioned, current-format (``HMAT``/v2), ``nnz=0`` file
    that loads clean in milliseconds so the rebuild never starts.

    It must also be an H-matrix or empty stub rather than a real ``.vf``. The
    engine warns loudly if it is not: a real loaded matrix leaves TWO matrices
    live at once, and the Gauss-Seidel and GPU-decline paths read the
    uncorrected one while the banner names the corrected one
    (``radiation_exchange.cpp:370``).

    ``generation`` names the physics, not the file format. Two matrices on one
    mesh from different generations are different physics and the outputs do
    not say so.
    """

    def __init__(self, replace_csr, base_stub, n_faces, nnz, geomhash,
                 generation, provenance, campaign, gpu_fit,
                 stub_rows=None, stub_nnz=0):
        self.replace_csr = str(replace_csr)
        self.base_stub = str(base_stub)
        self.n_faces = int(n_faces)
        self.nnz = int(nnz)
        self.geomhash = float(geomhash)
        self.generation = generation
        #: The ``.provenance.json`` sidecar's path, which is what makes this
        #: matrix's mesh identity checkable at all.
        self.provenance = str(provenance)
        #: The run that proves the set, named so a claim can be re-checked.
        self.campaign = campaign
        #: Whether this matrix fit one graphics die in production, and what
        #: happened when it did not. Measured, not predicted.
        self.gpu_fit = gpu_fit
        self.stub_rows = int(stub_rows) if stub_rows is not None else int(n_faces)
        self.stub_nnz = int(stub_nnz)

    def disk_bytes(self):
        """The CSR's size implied by its own declared layout.

        ``12 + 8*(n+1) + 8 + 8*nnz``: the header, the index array, the nonzero
        count, and an int32 column plus a float32 value per entry. Computed
        rather than stored so it cannot disagree with ``n_faces``/``nnz``.
        """
        return 12 + 8 * (self.n_faces + 1) + 8 + 8 * self.nnz

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedSiteViewFactors {Path(self.replace_csr).name}>"


class SupersededArtifact:
    """A file in the same directory that this entry deliberately does NOT wire.

    Named because the alternative is a user finding it and reasonably assuming
    the bigger/newer/nearer file is the one meant. Every one of these is real
    and sits beside the curated matrix.
    """

    def __init__(self, path, why, exists_note=""):
        self.path = str(path)
        self.why = why
        #: For a file this project DELETED on purpose, so its absence is not
        #: read as "the disk is incomplete".
        self.exists_note = exists_note

    def __repr__(self):                                   # pragma: no cover
        return f"<SupersededArtifact {Path(self.path).name}>"


class CuratedSite:
    """One row of the lunar-site table."""

    def __init__(self, key, name, body, region, mesh, occluder, view_factors,
                 why_curated, local_only=True, local_root="", aliases=(),
                 notes=(), superseded=(), use_gpu=True):
        self.key = key
        self.name = name
        self.body = body
        self.region = region
        self.mesh = mesh
        self.occluder = occluder
        self.view_factors = view_factors
        self.why_curated = why_curated
        #: True when the artifacts live outside the distribution. Every entry
        #: today is; the flag exists so a future shipped site is not forced to
        #: claim it is machine-local.
        self.local_only = bool(local_only)
        #: The directory whose absence means "this is a different machine",
        #: as distinct from "one file went missing".
        self.local_root = str(local_root)
        self.aliases = tuple(aliases)
        self.notes = tuple(notes)
        self.superseded = tuple(superseded)
        #: The route the proven campaign ran. `radiation.use_gpu: true`
        #: promotes the replacement CSR to the fused two-band GPU radiosity
        #: pool; when the matrix does not fit, the engine's own
        #: pool -> single-GPU -> CPU cascade handles it and says so in the log.
        self.use_gpu = bool(use_gpu)

    def display(self):
        return f"{self.key} — {self.region} ({self.mesh.coordinates()})"

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedSite {self.key}>"


A3GT_ROOT = "/Users/paha3326/projects/a3gt"
A3GT_MESHES = A3GT_ROOT + "/meshes"
A3GT_BASE_VF = A3GT_ROOT + "/base_vf"
A3GT_VF_BUILD = A3GT_ROOT + "/vf_build"

_GEN2 = ("A3GT GEN-2, ~/projects/agt_corrected_2026-08-19 — 9 windows + merge "
         "+ 17 COGs per site, completed and checked 2026-08-22 "
         "(run_manifest_A03*.json names this CSR for every window)")

_WHY = ("An A3GT site is three files that only mean anything together: a "
        "high-resolution core DEM, a shadow-only occluder ring chosen from "
        "several, and a view-factor matrix built for THAT core by an offline "
        "referee sweep. Choosing them separately has produced, in this "
        "project, an empty matrix that loaded clean, a matrix for a "
        "superseded mesh, a pruned matrix whose own receipt records failure, "
        "and a multi-gigabyte rebuild that wedged the graphics cards. The set "
        "below is the one that ran.")


#: The curated lunar-site table. Three entries, all machine-local.
#:
#: Do NOT add a site because its files exist. Add one when a campaign has run
#: the set to completion and the numbers below can be read off the files.
CURATED_SITES = (
    CuratedSite(
        key="A03MP026",
        name="A03MP026 — Mons Mouton plateau",
        body="Moon",
        region="Mons Mouton plateau",
        aliases=("MP026", "A03MP026", "mons mouton"),
        local_only=True,
        local_root=A3GT_ROOT,
        why_curated=_WHY,
        mesh=CuratedSiteMesh(
            path=A3GT_MESHES + "/A03MP026_84.223S_33.502E_R3km_linear.ply",
            faces=2397890,
            latitude_deg=-84.223,
            longitude_deg=33.502,
            measured_latitude_deg=-84.2234,
            measured_longitude_deg=33.5023,
            core_radius_km=3.0,
            source="SfS-derived core DEM, 3 km region of interest, Delaunay "
                   "triangulated",
        ),
        occluder=CuratedSiteOccluder(
            path=(A3GT_MESHES
                  + "/A03MP026_84.223S_33.502E_ring_Rcore3_Rext27.3108km.ply"),
            faces=152765,
            r_ext_km=27.3108,
            role="run-time shadow occluder (mesh.occluder_file)",
        ),
        view_factors=CuratedSiteViewFactors(
            replace_csr=(A3GT_VF_BUILD
                         + "/A03MP026/A03MP026_corrected_vf_v3.csr"),
            base_stub=A3GT_BASE_VF + "/A03MP026_base_empty.vf",
            n_faces=2397890,
            nnz=2216190708,
            geomhash=28733826.165896364,
            generation="FC-08 corrected (symmetrization divisor = directions "
                       "TRIED, plus back-face cull), 200,000 rays/face, "
                       "assembled 2026-08-20",
            provenance=(A3GT_VF_BUILD
                        + "/A03MP026/A03MP026_corrected_vf_v3.csr"
                          ".provenance.json"),
            campaign=_GEN2,
            gpu_fit="fits one W6800X die; the GEN-2 windows ran GPU radiosity "
                    "at 35-40 min/window",
            stub_rows=2397890,
        ),
        notes=(
            "This site has THREE occluder rings on disk. The 27.3108 km one "
            "above is what every production configuration sets as "
            "mesh.occluder_file. The 81.6712 km one -- which the project map "
            "names in the same row as this matrix -- was the RAY BLOCKER for "
            "the offline view-factor sweep, a one-off from a June 16-window "
            "high-accuracy pass. Both are correct for their own job and "
            "swapping them changes the shadowing of the run.",
        ),
        superseded=(
            SupersededArtifact(
                path=(A3GT_ROOT + "/prune_test/vfr_full/A03MP026/"
                      "A03MP026_corrected_vf.csr"),
                why=("the July generation: 19.9 GB, 2,484,716,524 nonzeros, "
                     "built before the FC-08 symmetrization-divisor fix, so "
                     "its far field reads up to 2x high. It has no "
                     ".provenance.json sidecar, so its mesh identity cannot "
                     "be checked the way the curated matrix's can. It ran two "
                     "complete campaigns and is kept for comparison."),
            ),
            SupersededArtifact(
                path=(A3GT_VF_BUILD
                      + "/A03MP026/A03MP026_corrected_vf_divisor2.csr"),
                why=("the intermediate step: the divisor fix WITHOUT the "
                     "back-face cull, superseded by the _v3 file the same "
                     "day."),
            ),
            SupersededArtifact(
                path=A3GT_VF_BUILD + "/A03MP026/A03MP026_rad_demo.csr",
                why=("a radiance-PRUNED experiment whose own receipt beside "
                     "it records the outcome: 'MEASURED FAIL: PSR-floor Tmax "
                     "mean 3.3 K / max 22.6 K cold ... Diffuse polar terrain "
                     "does NOT prune safely'. A pruned matrix is not a "
                     "cheaper version of the full one."),
            ),
            SupersededArtifact(
                path=A3GT_VF_BUILD + "/A03MP026/A03MP026_rad1e3_vf.csr",
                why="a second radiance-pruned experiment from the same study.",
            ),
            SupersededArtifact(
                path=A3GT_VF_BUILD + "/A03MP026/A03MP026_corrected_vf.csr",
                why=("the July matrix's original build location. It was "
                     "DELETED under the rolling-VF disk policy after use; "
                     "only its 19 MB .rowsum.npy sidecar is still there."),
                exists_note="deleted on purpose — its absence is not a fault",
            ),
        ),
    ),
    CuratedSite(
        key="A03N1016",
        name="A03N1016 — Nobile Rim 1",
        body="Moon",
        region="Nobile Rim 1",
        aliases=("N1016", "A03N1016", "nobile rim 1"),
        local_only=True,
        local_root=A3GT_ROOT,
        why_curated=_WHY,
        mesh=CuratedSiteMesh(
            path=A3GT_MESHES + "/A03N1016_85.391S_32.258E_R3km_linear.ply",
            faces=1798135,
            latitude_deg=-85.391,
            longitude_deg=32.258,
            measured_latitude_deg=-85.3914,
            measured_longitude_deg=32.2587,
            core_radius_km=3.0,
            source="SfS-derived core DEM, 3 km region of interest, Delaunay "
                   "triangulated",
        ),
        occluder=CuratedSiteOccluder(
            path=(A3GT_MESHES
                  + "/A03N1016_85.391S_32.258E_ring_Rcore3_Rext37.9051km.ply"),
            faces=152531,
            r_ext_km=37.9051,
            role="run-time shadow occluder AND the offline sweep's ray "
                 "blocker — this site has one ring, used for both",
        ),
        view_factors=CuratedSiteViewFactors(
            replace_csr=(A3GT_VF_BUILD
                         + "/A03N1016/A03N1016_corrected_vf_v3.csr"),
            base_stub=A3GT_BASE_VF + "/A03N1016_base_empty.vf",
            n_faces=1798135,
            nnz=3539114614,
            geomhash=29017233.617904384,
            generation="FC-08 corrected (symmetrization divisor = directions "
                       "TRIED, plus back-face cull), 200,000 rays/face, "
                       "assembled 2026-08-20",
            provenance=(A3GT_VF_BUILD
                        + "/A03N1016/A03N1016_corrected_vf_v3.csr"
                          ".provenance.json"),
            campaign=_GEN2,
            gpu_fit="the GEN-2 windows fit one die at 26.55 GiB with 0.64 GiB "
                    "of margin — because the FC-08 fix SHRANK this matrix. "
                    "Its predecessor (30.9 GB, 3,865,673,362 nonzeros) did "
                    "not fit, and all nine of its windows ran the engine's "
                    "CPU radiosity fallback: 103-116 min/window against 35-40 "
                    "for a matrix that fits. The cascade is clean and says so "
                    "in the log; it is a cost, not a fault.",
            stub_rows=1798135,
        ),
        notes=(
            "Fewer faces than A03MP026 and 60% MORE nonzeros. Crater-rim "
            "terrain sees itself; a plateau does not. Matrix size and build "
            "cost follow concavity, not face count -- do not size this "
            "site's successors from their face counts.",
        ),
        superseded=(
            SupersededArtifact(
                path=(A3GT_VF_BUILD
                      + "/A03N1016/A03N1016_corrected_vf_divisor2.csr"),
                why=("the divisor fix without the back-face cull, superseded "
                     "the same day."),
            ),
            SupersededArtifact(
                path=A3GT_VF_BUILD + "/A03N1016/A03N1016_corrected_vf.csr",
                why=("the first build, before either FC-08 correction. This "
                     "is the 30.9 GB matrix that did not fit a die."),
            ),
        ),
    ),
    CuratedSite(
        key="A03SP010",
        name="A03SP010 — Slater Plain",
        body="Moon",
        region="Slater Plain",
        aliases=("SP010", "A03SP010", "slater plain"),
        local_only=True,
        local_root=A3GT_ROOT,
        why_curated=_WHY,
        mesh=CuratedSiteMesh(
            path=A3GT_MESHES + "/A03SP010_87.130S_127.826E_R3km_linear.ply",
            faces=8746,
            latitude_deg=-87.130,
            longitude_deg=127.826,
            measured_latitude_deg=-87.1302,
            measured_longitude_deg=127.8255,
            core_radius_km=3.0,
            source="LOLA-only fallback (80 m/px) — NO shape-from-shading "
                   "product covers this site",
        ),
        occluder=CuratedSiteOccluder(
            path=(A3GT_MESHES
                  + "/A03SP010_87.130S_127.826E_ring_Rcore3_Rext89.4015km.ply"),
            faces=151579,
            r_ext_km=89.4015,
            role="run-time shadow occluder AND the offline sweep's ray "
                 "blocker — this site has one ring, used for both",
        ),
        view_factors=CuratedSiteViewFactors(
            replace_csr=(A3GT_VF_BUILD
                         + "/A03SP010/A03SP010_corrected_vf_v3.csr"),
            base_stub=A3GT_BASE_VF + "/A03SP010_base_empty.vf",
            n_faces=8746,
            nnz=6868216,
            geomhash=28244229.58147998,
            generation="FC-08 corrected (symmetrization divisor = directions "
                       "TRIED, plus back-face cull), 200,000 rays/face, "
                       "assembled 2026-08-20",
            provenance=(A3GT_VF_BUILD
                        + "/A03SP010/A03SP010_corrected_vf_v3.csr"
                          ".provenance.json"),
            campaign=_GEN2,
            gpu_fit="55 MB — it fits anywhere; the nine GEN-2 windows took "
                    "about 2.3 minutes in total",
            stub_rows=8746,
        ),
        notes=(
            "⚠️ 8,746 faces against A03MP026's 2,397,890 — about 200x coarser, "
            "and it is not an omission. No high-resolution DEM exists for "
            "Slater Plain anywhere in this project: the site's own source "
            "raster is a 125x125-pixel LOLA window, re-derived byte-identical "
            "in 2026-08 to prove the point, and none of the shape-from-shading "
            "products covers 87.130S/127.826E. The engine's own documented "
            "LOLA fallback is what built this. Never present results from "
            "this site as being at the same resolution class as the other "
            "two.",
        ),
        superseded=(
            SupersededArtifact(
                path=(A3GT_VF_BUILD
                      + "/A03SP010/A03SP010_corrected_vf_divisor2.csr"),
                why=("the divisor fix without the back-face cull, superseded "
                     "the same day."),
            ),
            SupersededArtifact(
                path=A3GT_VF_BUILD + "/A03SP010/A03SP010_corrected_vf.csr",
                why="the first build, before either FC-08 correction.",
            ),
        ),
    ),
)


#: The five A3GT sites PTM-312 leaves without a usable matrix, and the ONE
#: sentence that explains both of the tables below.
_PTM312 = ("PTM-312: this site's view-factor matrix came from the offline "
           "pipeline BEFORE the FC-08 symmetrization-divisor fix and the "
           "back-face cull, so its far field reads up to 2x high. It could "
           "not be re-swept with the other ten -- the raw sweep files lived "
           "in /tmp and a reboot cleared them -- so there is no corrected "
           "matrix to point at. Nothing in the file says any of this: it is "
           "the current layout, its face count matches its mesh, and it "
           "loads clean.")

_ORPHANED_SITES = ("A03MP016", "A03MP037", "A03MP038", "A03MP039", "A03MP040")

#: A3GT sites whose meshes exist and which are DELIBERATELY not curated.
#:
#: This is documentation with a test behind it, not a feature. Picking one of
#: these meshes by hand does nothing special -- there is no site entry, and
#: whatever matrices are found next to it are ranked by
#: ``gui_core.vf_discovery`` like anybody else's.
NOT_CURATED = {site: _PTM312 for site in _ORPHANED_SITES}

#: Matrix FILES this project knows are defective, keyed by file name.
#:
#: The one case where a view-factor file is loadable, correctly dimensioned,
#: and known to be wrong -- so the only thing that can catch it is a record
#: like this one. ``gui_core.vf_discovery`` reads it and refuses these files
#: outright.
#:
#: It lives HERE, beside the sites it belongs to, rather than in the search:
#: this is registry data ("which artifacts does this project distrust, and
#: why"), and the search's job is reading headers. Keeping the two together
#: also keeps one sentence explaining both, which is what they share.
#:
#: Keyed by BASENAME rather than by absolute path deliberately. These matrices
#: are machine-local, so an absolute path here would be a path that exists on
#: one computer; the names are unique to the A3GT catalog and travel with the
#: file if somebody copies it. A denylist that only recognized the original
#: location would go quiet exactly when the file moved.
DEFECTIVE_VIEW_FACTORS = {f"{site}_corrected_vf.csr": _PTM312
                          for site in _ORPHANED_SITES}


def _fold(text):
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def curated_site(name="", body=""):
    """The site entry for this name, or None.

    Matched on the site key and its aliases, case-folded and
    punctuation-insensitive, and on nothing looser. An A3GT key is an exact
    catalog identifier; a fuzzy match here would wire one polar site's terrain
    under another's name, and the two are tens of kilometers apart with
    different illumination.
    """
    folded = _fold(name)
    if not folded:
        return None
    wanted_body = _fold(body)
    for entry in CURATED_SITES:
        if wanted_body and _fold(entry.body) != wanted_body:
            continue
        candidates = {_fold(entry.key), _fold(entry.name)}
        candidates.update(_fold(alias) for alias in entry.aliases)
        candidates.discard("")
        if folded in candidates:
            return entry
    return None


def curated_sites_for_body(body):
    """Every curated site on one body, in catalog order."""
    wanted = _fold(body)
    return tuple(e for e in CURATED_SITES if _fold(e.body) == wanted)


# ---------------------------------------------------------------------------
# Is this site's set actually on THIS disk?
# ---------------------------------------------------------------------------

class SiteArtifact:
    """One file of a curated set, and what the disk says about it."""

    def __init__(self, role, path, present, dataless, size_bytes,
                 expected_bytes=None, mismatch=""):
        self.role = role
        self.path = str(path)
        self.present = bool(present)
        self.dataless = bool(dataless)
        self.size_bytes = int(size_bytes)
        self.expected_bytes = expected_bytes
        #: What the file says about itself that disagrees with the registry.
        #: Non-empty means the entry and the disk have come apart, which is a
        #: harder failure than a missing file: one of them is a lie.
        self.mismatch = mismatch

    @property
    def ok(self):
        return self.present and not self.dataless and not self.mismatch

    def __repr__(self):                                   # pragma: no cover
        return f"<SiteArtifact {self.role} ok={self.ok}>"


class SiteStatus:
    """Whether a curated site can be wired on the machine in front of the user.

    ``ready`` is the only thing to branch on, and it is False for a missing
    file, for a cloud placeholder, and for a file whose own header disagrees
    with the registry. The last one matters most: a curated entry that has
    drifted from its files is worse than no entry, because it looks checked.
    """

    def __init__(self, entry, artifacts, machine_has_root):
        self.entry = entry
        self.artifacts = list(artifacts)
        #: False means "this is a different machine", which is a different
        #: message from "one of your files is missing".
        self.machine_has_root = bool(machine_has_root)
        self.missing = [a for a in self.artifacts
                        if not a.present and not a.dataless]
        self.dataless = [a for a in self.artifacts if a.dataless]
        self.mismatched = [a for a in self.artifacts if a.mismatch]

    @property
    def ready(self):
        """Everything present, readable, and consistent -- AND on this machine.

        ``machine_has_root`` is part of it rather than only a nicer message.
        A local-only entry whose working tree has been moved or renamed while
        individual files survive somewhere is a set nobody is maintaining: the
        matrices in that tree are rolled off disk on purpose as they are
        superseded, so a stale copy outliving its directory is exactly the
        state where the registry's numbers stop being a description of
        anything. Refusing is the safe direction and the message says which
        case it is.
        """
        return (bool(self.artifacts) and all(a.ok for a in self.artifacts)
                and (self.machine_has_root or not self.entry.local_only))

    def summary(self):
        """One line for the card, whichever state this is in."""
        entry = self.entry
        if not self.machine_has_root and entry.local_only:
            return (f"{entry.key}'s files are not on this computer. This site "
                    f"is curated from a local A3GT working tree "
                    f"({entry.local_root}) that PTM3D does not distribute -- "
                    f"hundreds of gigabytes of DEM-derived meshes and "
                    f"view-factor matrices. Nothing has been written to your "
                    f"configuration.")
        if self.mismatched:
            first = self.mismatched[0]
            return (f"{Path(first.path).name} disagrees with what this "
                    f"registry records about it: {first.mismatch} The entry "
                    f"and the file have come apart; the site was NOT wired.")
        if self.dataless:
            names = ", ".join(Path(a.path).name for a in self.dataless)
            return (f"{names} is a cloud placeholder — the name is on this "
                    f"disk, the bytes are not. Reading it would block the run "
                    f"while it downloads. Make it available offline first "
                    f"(Finder, right-click, 'Always Keep on This Device'). "
                    f"Nothing has been written to your configuration.")
        if self.missing:
            names = ", ".join(Path(a.path).name for a in self.missing)
            return (f"{entry.key} needs {names}, which "
                    f"{'is' if len(self.missing) == 1 else 'are'} not on this "
                    f"disk. Nothing has been written to your configuration.")
        total = sum(a.size_bytes for a in self.artifacts)
        # `generation` reads "FC-08 corrected (symmetrization divisor = ...,
        # plus back-face cull), 200,000 rays/face, assembled ...", so splitting
        # on a comma cuts INSIDE the parenthetical and leaves the line with an
        # unclosed bracket. Split before the parenthesis instead.
        generation = entry.view_factors.generation.split(" (")[0].split(",")[0]
        return (f"{entry.key}: mesh, occluder ring, base stub and the "
                f"{generation} matrix are all on this disk "
                f"({_human_bytes(total)} in total).")

    def __repr__(self):                                   # pragma: no cover
        return f"<SiteStatus {self.entry.key} ready={self.ready}>"


def _human_bytes(nbytes):
    """The other curated tier's formatter, so both read the same.

    ``curated_sources.human_bytes`` stops at GB, which is right for an 80 MB
    kernel and wrong here -- these matrices run to tens of gigabytes and one
    site's set is over 100 GB. It is extended rather than duplicated: a second
    size formatter is a second place for two cards to disagree about what
    "19.9 GB" means.
    """
    value = float(nbytes)
    if value < 1024.0 ** 3:
        from gui_core.curated_sources import human_bytes
        return human_bytes(nbytes)
    value /= 1024.0 ** 3
    for unit in ("GB", "TB", "PB"):
        if value < 1024.0 or unit == "PB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"                              # pragma: no cover


def _probe(role, path, expected_bytes=None, exact=False):
    """stat only. No file here is small enough to open casually."""
    try:
        stat_result = os.stat(path)
    except OSError:
        return SiteArtifact(role, path, present=False, dataless=False,
                            size_bytes=0, expected_bytes=expected_bytes)
    dataless = stat_result.st_blocks == 0 and stat_result.st_size > 0
    mismatch = ""
    if (exact and expected_bytes is not None
            and stat_result.st_size != expected_bytes):
        mismatch = (f"the registry's face and nonzero counts imply "
                    f"{expected_bytes:,} bytes and the file is "
                    f"{stat_result.st_size:,}.")
    return SiteArtifact(role, path, present=True, dataless=dataless,
                        size_bytes=stat_result.st_size,
                        expected_bytes=expected_bytes, mismatch=mismatch)


def site_status(entry):
    """Is this curated site's whole set on this disk? Local, stat-only.

    Four artifacts, and all four are required. A set missing any one of them
    is not a partially usable set: without the occluder the site is unshadowed
    by its surroundings, without the stub the run rebuilds the matrix, and
    without the matrix there is no facet-to-facet exchange at all.
    """
    root_present = os.path.isdir(entry.local_root) if entry.local_root else True
    vf = entry.view_factors
    artifacts = [
        _probe("core mesh", entry.mesh.path),
        _probe("occluder ring", entry.occluder.path),
        _probe("base stub", vf.base_stub),
        # The CSR is the one file whose size is fully determined by two numbers
        # this registry records, so an exact check costs nothing and catches a
        # truncated copy, a partial download and a stale generation at once.
        _probe("view-factor matrix", vf.replace_csr,
               expected_bytes=vf.disk_bytes(), exact=True),
    ]
    return SiteStatus(entry, artifacts, root_present)


# ---------------------------------------------------------------------------
# Turning a selection into configuration keys
# ---------------------------------------------------------------------------

def site_wiring(entry):
    """``[(dotted key, value), ...]`` for selecting this site. Nothing else.

    Returned rather than applied so the caller can show it, and so the whole
    of what a click does is one testable list.

    The two placement keys are set to the ``-999.0`` "not placed" sentinel ON
    PURPOSE, and clearing them is part of the wiring rather than an omission:
    a leftover latitude from a previously opened configuration would rotate
    this already-placed DEM, silently, and take the core away from its ring.
    See :class:`CuratedSiteMesh`.
    """
    vf = entry.view_factors
    return [
        ("simulation.body", entry.body),
        ("mesh.import_file", entry.mesh.path),
        ("mesh.occluder_file", entry.occluder.path),
        ("mesh.latitude_override", NOT_PLACED),
        ("mesh.longitude_override", NOT_PLACED),
        ("radiation.enabled", True),
        ("radiation.view_factor_file", vf.base_stub),
        ("radiation.use_gpu", bool(entry.use_gpu)),
        ("engine_env.PTM_HMATRIX_VF_REPLACE", vf.replace_csr),
    ]


def wiring_provenance(entry):
    """The lines a card shows so a selection is never a silent one.

    Everything a user would have had to know to assemble this set by hand,
    including what was NOT chosen.
    """
    vf = entry.view_factors
    lines = [
        f"{entry.name} — {entry.mesh.coordinates()}, {entry.mesh.core_radius_km:g} km core.",
        f"Terrain: {Path(entry.mesh.path).name} — {entry.mesh.faces:,} faces "
        f"({entry.mesh.source}).",
        f"Occluder: {Path(entry.occluder.path).name} — {entry.occluder.faces:,} "
        f"faces, outer radius {entry.occluder.r_ext_km:g} km, "
        f"{entry.occluder.role}.",
        f"View factors: {Path(vf.replace_csr).name} — {vf.nnz:,} nonzeros, "
        f"{vf.generation}. Loaded through PTM_HMATRIX_VF_REPLACE; the engine "
        f"recomputes this mesh's geometry hash at load and aborts if it "
        f"disagrees.",
        f"Base stub: {Path(vf.base_stub).name} — an empty {vf.stub_rows:,}-row "
        f"H-matrix so the normal loader finds a cache and never starts a "
        f"multi-gigabyte rebuild.",
        f"Route: {vf.gpu_fit}.",
        f"Proven by: {vf.campaign}.",
        "Placement: mesh.latitude_override and mesh.longitude_override are "
        "set to -999.0 (not placed). This DEM is already in the lunar body "
        "frame at its own coordinates; an override would rotate it -- and "
        "would rotate the core WITHOUT the ring.",
    ]
    for note in entry.notes:
        lines.append(note)
    for item in entry.superseded:
        suffix = f" ({item.exists_note})" if item.exists_note else ""
        lines.append(f"NOT used: {Path(item.path).name}{suffix} — {item.why}")
    if entry.local_only:
        lines.append(
            f"Local only: these files live in {entry.local_root}, which PTM3D "
            f"does not distribute. On another machine this entry is shown and "
            f"refuses rather than writing paths that are not there.")
    return lines
