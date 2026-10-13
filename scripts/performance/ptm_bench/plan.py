#!/usr/bin/env python3
"""Build the Moon3D R1 benchmark batch (C2/C3) and predict its cost. DRY RUN by
default: prints every run with predicted wall time and peak memory against the
plan's hard caps (PTM3D_changes_for_R1.md section 4.4), the GPU-safety block,
and per-arm totals; --write emits plan.json for run_bench.py.

Every prediction is a stated rate times a stated size, +-2x. Rates and their
sources are in RATES; nnz is predicted from a visible-partner model anchored on
measured matrices (MODEL below). Nothing here is a measurement of the batch.

  plan.py --manifest ~/projects/moon3d_R1/bench/meshes/manifest.json [--write plan.json]
"""
import argparse
import json
import math
from pathlib import Path

HOME = Path.home()

# --------------------------------------------------------------------------- rates
RATES = {
    # mc_ref_pp aggregate over 48 processes: Moon3D 509k 1.02e11 rays in 1961 s; r32c
    # 1.99e10 in 365 s (headline_table.json, 2026-08-24).
    "mc_rays_per_s_48": 5.3e7,
    # assemble_vf_haworth.py: 509k v4 7.45e8 nnz in 833 s; plan range 11-19 min / 1e9.
    "asm_s_per_nnz": 833 / 7.45e8,
    "asm_bytes_per_nnz": 48.0,            # vf-assembler-memory-fix-2026-09-13 (measured)
    # ---- MEASURED on the R1 binary family (afbd2bd GPU build), 10 clean Stage-3 Moon3D
    # 509k coupled lunations + 7 radoff (inventory.py, bench/inventory.json, 2026-10-13):
    # CPU radiosity (fused two-band Jacobi, 48 thr, rho(F) 0.78): 118.8 s per 600-step sweep
    # at 745,322,408 nnz (sd 11.9 s) -> 3.76e9 nnz x steps / s. The old assumption
    # (1.6e9 traversals/s, 4 per solve) was 9.4x pessimistic.
    "cpu_rad_nnz_steps_per_s": 600 * 745322408 / 118.8,
    # illumination (CPU Embree AVX-512, 48 thr): ~73 s per 600 steps at 509,337 faces.
    "illum_s_per_face_step": 73.0 / (600 * 509337),
    # GPU thermal kernel, whole mesh, N=1: ~45 s per 600 steps at 509,337 faces (kernel time,
    # host pauses excluded); re-thermal iteration 1 ~0.8x of it, iterations >= 2 ~0.28x.
    "gpu_kernel_s_per_face_step": 45.0 / (600 * 509337),
    "reth_iter1_frac": 0.8,
    "reth_steady_frac": 0.28,
    # everything else in a lunation (setup, export of 7 x 360 frames, statistics): radoff
    # wall 242-353 s - illum - kernel ~ 210 s at 509k -> 4.1e-4 s/face.
    "remainder_s_per_face_lunation": 210.0 / 509337,
    "csr_load_bytes_per_s": 1.0e9,        # ASSUMED (not separable in the Stage-3 logs)
    # CPU illumination: 10,082 faces in 15.8 s at 8 threads on a contended box -> x6 threads.
    "cpu_illum_s_per_face_48": 15.8 / 10082 / 6,
    # CPU SIMD thermal: 1.57 s per whole-mesh pass at 10,082 faces, 8 threads, contended.
    "cpu_thermal_s_per_face_pass_48": 1.57 / 10082 / 6,
    "gpu_n4_thermal_speedup": 1.95,       # perfbench 2026-08-15
    # bowl exact dense builder: 3,776 faces in 1.6 s at 8 threads (O(N^2) pairs).
    "bowl_exact_s_per_pair_48": 1.6 / 3776 ** 2 / 6,
    "bowl_solve_matvecs": 250,            # Picard (~50) x Jacobi (~5); radiosity_eq8 fp_iters
    "bowl_csr_bytes_per_nnz": 20.0,       # float32 read + double values + int cols
    "bowl_exact_bytes_per_nnz": 24.0,     # COO build -> CSR, double values (estimate)
    # Engine peak RSS per face (360 output + 240 spin-up steps), MEASURED 2026-10-12 with
    # PTM_TIMING_JSON: coupled 968 f 194 MB, 10,082 f 1,113 MB -> 0.10 MB/face + ~100 MB.
    # The streamed-thermal path (engaged above ~10 GB) is NOT measured; see PROBE rows.
    "engine_bytes_per_face": 1.0e5,
    "engine_bytes_base": 1.0e8,
    "repeat_max_wall_s": 900,             # timing repeats (n=2) only for rows <= 15 min
}
# Disk model (bytes), calibrated 2026-10-12 on the smoke runs: per-step frames 7 fields x
# 360 steps x (4N + 4) = 10.1 kB/face; statistics/ 87 B/face + mesh.ply 45 B/face; partner
# bins 6.2 B per stored nnz measured (8 B upper bound: D <= S); CSR 8 B/nnz + 8 (N + 1).
# The runner deletes bins after assembly (M2), each harness CSR after its last consumer
# (M3) and per-step frames after each engine row (M4); an unassembled sweep writes no bins (M1).
DISK = {"engine_kept_per_face": 150, "engine_kept_fixed": 5e7,
        "bin_per_nnz": 8, "csr_per_nnz": 8, "sweep_small_per_face": 60, "bowl_per_face": 100}

# Disk-guard overrides: explicit, per row, never global. run_bench.py requires free >=
# factor x predicted_disk_peak_bytes + reserve; the default factor is 2.0. Every override is
# written into the row and logged in the run ledger when the guard applies it.
DISK_GUARD_DEFAULT = 2.0
DISK_GUARD_OVERRIDES = {
    "D_haworth_radcore_r25_100m_n400973_nr200000_sweep":
        (1.8, "Paul ruling 2026-10-13: relax to 1.8x for the two largest rows only, so the "
              "batch fits in ~110 GiB after snapshot thinning"),
    "A_bowl_dd0.2_n100165_E2b_nr200000_k0.2_sweep":
        (1.8, "Paul ruling 2026-10-13: relax to 1.8x for the two largest rows only, so the "
              "batch fits in ~110 GiB after snapshot thinning"),
}
# Rows (with every row that consumes their CSR) moved to the FRONT of the batch, where free
# space is highest: the override rows have the thinnest margin.
FRONT_LOAD = list(DISK_GUARD_OVERRIDES)

CAPS = {
    "assembly_nnz": 6.9e9,                # 48 B/nnz on 330 GB usable (hard)
    # Planning cap: half the hard cap, because the nnz each row is planned on is a +-2x
    # model, and an assembler OOM at 330 GB takes the machine with it.
    "assembly_nnz_planning": 3.45e9,
    "ram_bytes": 330e9,
    "gpu_radiosity_nnz": 1e8,             # >1e8 nnz on a GPU is a known hard-wedge trigger
    "gpu_int32_wall_nnz": 5.37e8,
}

# --------------------------------------------------------------------------- nnz model
# Per-row stored partners after symmetrization = V (1 - exp(-2 lambda)), lambda = NR <sumF> / V:
# each direction finds a given visible partner with probability 1 - exp(-lambda) (uniform-F
# approximation), the stored pair is the union of both directions. Checked on the 3,776-face
# bowl: predicted 1940 / 2390 partners per face at NR 2e4 / 2e5, measured 2005 / 2310.
MODEL = {
    # family: (V at N0, N0, exponent of V in N, mean sumF)
    "bowl": (2394, 3776, 1.0, 0.105),           # exact dense matrix: 0.634 N
    "moon3d": (1463, 509337, 1.0, 0.1175),      # v4 509k saturated (detect-all)
    "fbm_bandlimited": (393, 10082, 0.64, 0.06),  # smoke NR 2e4 (968 and 10,082 faces)
    "fbm_extended": (393, 10082, 0.80, 0.07),     # ASSUMED steeper (new roughness each level)
    # terrain: ray-budget capped. V0 fitted so the model reproduces r32c v4 (99,665 f,
    # 5,697 partners/face, sumF 0.0213); it then predicts r50 at 6,386 vs 4,090 measured.
    "haworth": (10000, 99665, 1.0, 0.0213),
}


def nnz_per_face(family, n, nr):
    fam = ("haworth" if family.startswith("haworth") or family.startswith("sunlit")
           else "moon3d" if family.startswith("moon3d") else family)
    v0, n0, a, sf = MODEL[fam]
    V = v0 * (n / n0) ** a
    lam = nr * sf / V
    return V * (1 - math.exp(-2 * lam))


def hms(s):
    if s is None:
        return "-"
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.1f} m" if s >= 60 else f"{s:.0f} s"


def gb(b):
    return "-" if b is None else f"{b / 1e9:.1f} GB"


class Plan:
    def __init__(self, root, man):
        self.root, self.runs, self.rows = root, [], []
        self.man = man

    def disk(self, kind, run, nnz):
        """(peak bytes while the row runs, bytes it leaves behind, CSR bytes it leaves
        behind until that CSR's last consumer)."""
        n = (run or {}).get("n_faces", 0)
        if kind == "sweep":
            small = DISK["sweep_small_per_face"] * n + 1e6
            if (run or {}).get("assemble", True):
                csr = DISK["csr_per_nnz"] * nnz + 8 * (n + 1) + 20
                rowsum = 8 * n
                return small + DISK["bin_per_nnz"] * nnz + csr + rowsum, small + rowsum, csr
            return small, small, 0.0
        if kind == "engine":
            kept = DISK["engine_kept_per_face"] * n + DISK["engine_kept_fixed"]
            n_out = (run or {}).get("overrides", {}).get("simulation.timesteps", 360)
            frames = 7 * n_out * (4 * n + 4)
            return frames + kept, kept, 0.0
        b = DISK["bowl_per_face"] * n + 1e6
        return b, b, 0.0

    def add(self, arm, rid, kind, wall, mem, note="", run=None, gpu=False, repeats=1, nnz=None):
        status = "ok"
        if kind == "sweep" and run is not None and not str(run["mesh"]).endswith(".ply"):
            # PTM-724: mc_ref_pp reads binary PLY only; a .vtk here fails every attempt.
            raise SystemExit(f"FATAL: sweep row {rid} points at {run['mesh']}; mc_ref_pp "
                             "reads binary PLY only. Regenerate the manifest (make_meshes.py).")
        dpeak, dfinal, dcsr = self.disk(kind, run, nnz or 0)
        if mem is not None and mem > CAPS["ram_bytes"]:
            status = "OVER-RAM-CAP"
            if kind == "engine":
                status = "PROBE-FIRST"   # streamed thermal may fit; unmeasured -> not scheduled
                note += "; in-core model exceeds RAM, streamed path unmeasured: run after a 1e6 RSS check"
        if wall is not None and wall > RATES["repeat_max_wall_s"]:
            repeats = 1
        produces = (str(Path(self.root) / rid / "vf.csr")
                    if kind == "sweep" and (run or {}).get("assemble", True) else None)
        self.rows.append(dict(arm=arm, id=rid, kind=kind, wall=wall, mem=mem, note=note,
                              status=status, gpu=gpu, repeats=repeats, disk_peak=dpeak,
                              disk_final=dfinal, csr_bytes=dcsr, produces=produces,
                              consumes=(run or {}).get("csr")))
        if run is not None and status == "ok":
            run.update(id=rid, kind=kind, repeats=repeats, predicted_wall_s=wall,
                       predicted_peak_bytes=mem, predicted_disk_peak_bytes=dpeak,
                       predicted_disk_final_bytes=dfinal + dcsr)
            if rid in DISK_GUARD_OVERRIDES:
                f, why = DISK_GUARD_OVERRIDES[rid]
                run.update(disk_guard_factor=f, disk_guard_override_reason=why)
            self.runs.append(run)


def sweep_cost(n_src, nr):   # n_src = faces actually swept
    return n_src * nr / RATES["mc_rays_per_s_48"]


def asm_cost(nnz):
    return nnz * RATES["asm_s_per_nnz"], nnz * RATES["asm_bytes_per_nnz"]


def engine_mem(n, steps=600):
    return RATES["engine_bytes_base"] + n * RATES["engine_bytes_per_face"] * steps / 600


# --------------------------------------------------------------------------- short windows
# Coupled timing rows run a SHORT window, not a lunation (Paul 2026-10-13, <24 h batch):
# 30 output steps at the lunation cadence + 0.25 spin-up orbit (30 steps), coupling_mode
# fixed with 2 iterations. Iteration 1 carries the GPU equilibration replay (the costly,
# transient re-thermal: Stage 3 kernel 35 s vs 12.5 s steady at 509k), iteration 2 is the
# steady state; the eq-indirect pass adds one sweep + one whole-mesh solve, as in
# production. Stationarity evidence (inventory.py on the 10 clean Stage-3 coupled runs):
# within a sweep, elapsed is linear in steps to 3-12 %; per-sweep wall drifts +8-15 % from
# sweep 1 to 12; re-thermal kernel per iteration is flat from iteration 2 on (sd < 0.5 s).
WIN_OUT, WIN_EQ_ORBITS, WIN_ITERS = 30, 0.25, 2
WIN_STEPS = WIN_OUT + int(round(WIN_EQ_ORBITS * 120))
WINDOW = {"simulation.timesteps": WIN_OUT,
          "simulation.duration_days": round(WIN_OUT * 29.53 / 360, 6),
          "thermal.equilibration_orbits": WIN_EQ_ORBITS,
          "radiation.max_coupling_iterations": WIN_ITERS,
          "radiation.coupling_mode": "fixed"}


def radoff_cost(n, gpu_devices=None):
    """Full-lunation rad-off row (illumination is NOT stationary over a lunation, so the
    illumination/thermal curves keep whole lunations). Calibrated on Stage 3 radoff at
    509k (4 GPUs): wall 242-353 s = illum ~75 + kernel ~45 + remainder ~210 (export)."""
    illum = RATES["illum_s_per_face_step"] * n * 600
    if gpu_devices:
        th = RATES["gpu_kernel_s_per_face_step"] * n * 600 / (RATES["gpu_n4_thermal_speedup"]
                                                                 if gpu_devices == 4 else 1)
    else:
        th = 2 * RATES["cpu_thermal_s_per_face_pass_48"] * n
    return illum + th + RATES["remainder_s_per_face_lunation"] * n + 20


def window_cost(n, nnz):
    """Short coupled window: illumination + eq-indirect sweep + 2 iteration sweeps on
    WIN_STEPS steps, whole-mesh solve x2 (initial + eq-indirect), re-thermal iteration 1
    (transient) + 2 (steady), CSR load, small export."""
    ill = RATES["illum_s_per_face_step"] * n * WIN_STEPS
    rad = (1 + WIN_ITERS) * WIN_STEPS * nnz / RATES["cpu_rad_nnz_steps_per_s"]
    k = RATES["gpu_kernel_s_per_face_step"] * n * WIN_STEPS
    th = 2 * k + RATES["reth_iter1_frac"] * k + RATES["reth_steady_frac"] * k * (WIN_ITERS - 1)
    load = nnz * 8 / RATES["csr_load_bytes_per_s"]
    rem = 30 + RATES["remainder_s_per_face_lunation"] * n * WIN_OUT / 360
    return ill + rad + th + load + rem, nnz * 8 + engine_mem(n, WIN_STEPS)


def full_coupled_cost(n, nnz, sweeps=12):
    """Model check only: a full-lunation coupled run (used to verify the rates reproduce
    Stage 3, not scheduled)."""
    ill = RATES["illum_s_per_face_step"] * n * 600
    rad = sweeps * 600 * nnz / RATES["cpu_rad_nnz_steps_per_s"]
    k = RATES["gpu_kernel_s_per_face_step"] * n * 600
    th = k + RATES["reth_iter1_frac"] * k + RATES["reth_steady_frac"] * k * (sweeps - 2)
    return ill + rad + th + RATES["remainder_s_per_face_lunation"] * n + nnz * 8 / RATES["csr_load_bytes_per_s"]


ROW_SAMPLE_1_IN = 10   # row-sampled sweeps: seeded random 1-in-10 faces


def build(man, root, with_haworth_refinement=False):
    P = Plan(root, man)
    by = {}
    for m in man:
        by.setdefault(m["family"], []).append(m)
    P.dropped = []

    # ------------------------------------------------------------ arm A: bowl
    bowls = sorted([m for m in by.get("bowl", [])], key=lambda m: (m["dd"] != 0.2, m["n_faces"]))
    for m in bowls:
        n, dd = m["n_faces"], m["dd"]
        if dd != 0.2 and n > 3.1e4:
            continue                          # d/D 0.1 secondary: small rungs only
        tag = f"A_bowl_dd{dd:g}_n{n}"
        geo = dict(mesh=m["path"], n_faces=n, n_rings=m["n_rings"], n_az=m["n_az"],
                   n_flat=m["n_flat"], dd=dd)
        floor_list = str(Path(root) / "facelists" / f"bowl_n{n}_dd{dd:g}_floor.txt")
        n_floor = m.get("n_floor", int(0.12 * n))
        ref = None
        if n <= 3.1e4:
            nnz = 0.634 * n * n
            P.add("A", f"{tag}_E0_exact", "bowl",
                  0.634 * n * n * RATES["bowl_exact_s_per_pair_48"] + 1e-9 * nnz * RATES["bowl_solve_matvecs"],
                  nnz * RATES["bowl_exact_bytes_per_nnz"], "exact dense; same-N reference",
                  run=dict(estimator="exact", **geo))
            ref = str(Path(root) / f"{tag}_E0_exact" / "d")
        if n <= 1.01e5:
            P.add("A", f"{tag}_E1_sparse", "bowl", n * n * RATES["bowl_exact_s_per_pair_48"] * 2,
                  n * 500 * 24, "shipped 20/500/95% builder (O(N^2) scan, timed)",
                  run=dict(estimator="sparse_production", ref_dump=ref, **geo))
        mc = []
        if n <= 3.1e5:
            mc.append(("E2a", 20000, 0.2))
        if n <= 1.01e5:
            mc.append(("E2b", 200000, 0.2))
            mc.append(("E2c", int(20 * n), 0.2))
        if n <= 3.1e4:
            mc.append(("E2k", 200000, 0.01))    # keps attribution (2026-10-12 smoke)
        if 1.1e4 < n <= 3.1e4:
            P.dropped.append((f"{tag}_Eref_nr2e+06 (sweep + solve)", "same-N reference is E0 "
                              "exact at this N; Eref kept at N <= 1e4 for the 2e6 -> exact check"))
        if n <= 1.1e4:
            # high-NR reference at the same N. TRIMMED at N ~ 3e4 (2026-10-13): E0 exact
            # exists at every N <= 3e4 and is the better reference; Eref at N <= 1e4 is
            # kept to show the 2e6-ray estimator converging onto the exact matrix.
            mc.append(("Eref", 2000000, 0.2))
        for name, nr, keps in mc:
            nnz = n * nnz_per_face("bowl", n, nr)
            sid = f"{tag}_{name}_nr{nr:g}_k{keps:g}"
            asm_t, asm_m = asm_cost(nnz)
            if nnz > CAPS["assembly_nnz_planning"]:
                P.add("A", sid, "sweep", sweep_cost(n_floor, nr), None,
                      f"nnz {nnz:.2e} > planning cap: FLOOR-ONLY",
                      run=dict(nrays=nr, keps=keps, assemble=False, facelist=floor_list, **geo))
                continue
            P.add("A", sid + "_sweep", "sweep", sweep_cost(n, nr) + asm_t, asm_m,
                  f"nnz {nnz:.2e}", run=dict(nrays=nr, keps=keps, **geo), nnz=nnz)
            P.add("A", sid + "_solve", "bowl", 1e-9 * nnz * RATES["bowl_solve_matvecs"] + 60,
                  nnz * RATES["bowl_csr_bytes_per_nnz"], "CSR load + radiosity vs Eq. 8",
                  run=dict(estimator="csr", csr=str(Path(root) / (sid + "_sweep") / "vf.csr"),
                           ref_dump=ref, **geo))
        if n > 3.1e5:
            for nr in ((20000, 200000) if n <= 1.1e6 else (20000,)):
                P.add("A", f"{tag}_floor_nr{nr:g}", "sweep", sweep_cost(n_floor, nr), None,
                      "floor-only sweep (F_sky direct, no assembly)",
                      run=dict(nrays=nr, keps=0.2, assemble=False, facelist=floor_list, **geo))

    # ------------------------------------------------------------ arm B: fBm
    for fam in ("fbm_bandlimited", "fbm_extended"):
        short = fam[4:]
        for m in sorted(by.get(fam, []), key=lambda m: m["n_faces"]):
            n = m["n_faces"]
            tag = f"B_{short}_n{n}"
            geo = dict(mesh=m["path"], n_faces=n)
            # rad-off, full lunation: illumination + thermal curves (CPU, GPU N=1, GPU N=4)
            for nm, ov, ndev in (("radoff_cpu", {"radiation.enabled": False}, None),
                                 ("radoff_gpu1", {"radiation.enabled": False, "thermal.use_gpu": True,
                                                  "thermal.gpu_devices": 1}, 1),
                                 ("radoff_gpu4", {"radiation.enabled": False, "thermal.use_gpu": True,
                                                  "thermal.gpu_devices": 4}, 4)):
                if ndev == 4 and fam == "fbm_extended":
                    P.dropped.append((f"{tag}_{nm}", "N=4 scaling measured once (band-limited); "
                                      "the thermal kernel cost per face does not depend on roughness"))
                    continue
                P.add("B", f"{tag}_{nm}", "engine", radoff_cost(n, ndev), engine_mem(n),
                      "full lunation; illumination + thermal", gpu=bool(ndev), repeats=2,
                      run=dict(overrides=ov, **geo))
            nnz2e4 = n * nnz_per_face(fam, n, 20000)
            if n > 1.1e6:
                P.add("B", f"{tag}_nr2e4_rowsample", "sweep", sweep_cost(n / ROW_SAMPLE_1_IN, 20000), None,
                      f"1-in-{ROW_SAMPLE_1_IN} row sample: per-face sweep cost + partners/face -> nnz BRACKET",
                      run=dict(nrays=20000, assemble=False, row_sample_1_in=ROW_SAMPLE_1_IN, **geo))
                P.dropped.append((f"{tag}_nr20000_sweep (full)", f"replaced by a 1-in-{ROW_SAMPLE_1_IN} "
                                  "row sample: same per-face cost and partners/face, 10x cheaper"))
                continue
            asm_t, asm_m = asm_cost(nnz2e4)
            sid = f"{tag}_nr20000"
            P.add("B", sid + "_sweep", "sweep", sweep_cost(n, 20000) + asm_t, asm_m,
                  f"nnz {nnz2e4:.2e}", run=dict(nrays=20000, **geo), nnz=nnz2e4)
            w, mem = window_cost(n, nnz2e4)
            P.add("B", sid + "_window", "engine", w, mem,
                  f"SHORT coupled window ({WIN_OUT}+{WIN_STEPS - WIN_OUT} steps, {WIN_ITERS} it); "
                  "thermal GPU N=1, radiosity CPU", gpu=True, repeats=2,
                  run=dict(csr=str(Path(root) / (sid + "_sweep") / "vf.csr"),
                           overrides={**WINDOW, "thermal.use_gpu": True, "thermal.gpu_devices": 1,
                                      "radiation.use_gpu": False}, **geo))
            nnz2e5 = n * nnz_per_face(fam, n, 200000)
            if n > 1.1e5:
                P.add("B", f"{tag}_nr2e5_rowsample", "sweep", sweep_cost(n / ROW_SAMPLE_1_IN, 200000), None,
                      f"1-in-{ROW_SAMPLE_1_IN} row sample at NR 2e5 -> nnz BRACKET",
                      run=dict(nrays=200000, assemble=False, row_sample_1_in=ROW_SAMPLE_1_IN, **geo))
            else:
                asm_t, asm_m = asm_cost(nnz2e5)
                P.add("B", f"{tag}_nr200000_sweep", "sweep", sweep_cost(n, 200000) + asm_t, asm_m,
                      f"nnz {nnz2e5:.2e} (no consumer: nnz/saturation only)",
                      run=dict(nrays=200000, **geo), nnz=nnz2e5)

    # ------------------------------------------------------------ arm C: Moon3D
    for m in sorted(by.get("moon3d_dec", []), key=lambda m: m["n_faces"]):
        n = m["n_faces"]
        tag = f"C_moon3d_n{n}"
        geo = dict(mesh=m["path"], n_faces=n)
        P.add("C", f"{tag}_radoff_gpu1", "engine", radoff_cost(n, 1), engine_mem(n),
              "full lunation; illumination + thermal", gpu=True, repeats=2,
              run=dict(overrides={"radiation.enabled": False, "thermal.use_gpu": True,
                                  "thermal.gpu_devices": 1}, **geo))
        if n > 4e5:
            P.dropped.append((f"{tag}_nr2e5_sweep + _coupled", "509k point taken from the paper "
                              "mesh: v4 CSR sweep provenance (2026-08-24) + 10 clean Stage-3 coupled "
                              "lunations + one short calibration window (C_moon3d_509k_v4_window)"))
            continue
        nnz = n * nnz_per_face("moon3d", n, 200000)
        asm_t, asm_m = asm_cost(nnz)
        P.add("C", f"{tag}_nr2e5_sweep", "sweep", sweep_cost(n, 200000) + asm_t, asm_m,
              f"nnz {nnz:.2e} (nnz ~ N^2 regime test)", run=dict(nrays=200000, **geo), nnz=nnz)
        w, mem = window_cost(n, nnz)
        P.add("C", f"{tag}_window", "engine", w, mem, "SHORT coupled window; CPU radiosity",
              gpu=True, repeats=2,
              run=dict(csr=str(Path(root) / f"{tag}_nr2e5_sweep" / "vf.csr"),
                       overrides={**WINDOW, "thermal.use_gpu": True, "thermal.gpu_devices": 1,
                                  "radiation.use_gpu": False}, **geo))
    for m in by.get("moon3d_509k_v4_existing", []):
        n, nnz = 509337, 745322408
        w, mem = window_cost(n, nnz)
        P.add("C", "C_moon3d_509k_v4_window", "engine", w, mem,
              "CALIBRATION: same mesh/CSR/settings as Stage 3, with PTM_TIMING_JSON -> checks the "
              "log-derived components of the 10 reused Stage-3 lunations", gpu=True, repeats=2,
              run=dict(mesh=str(HOME / "dev/heat3d/moon3d_c_cartesian_NEW_500k_clean_native.ply"),
                       n_faces=n, csr=m["csr"],
                       overrides={**WINDOW, "thermal.use_gpu": True, "thermal.gpu_devices": 1,
                                  "radiation.use_gpu": False}))
    for m in by.get("moon3d_original", []):
        n = m.get("n_faces", 1698264)
        nnz = n * nnz_per_face("moon3d", n, 200000)
        P.add("C", "C_moon3d_orig_nr2e5_rowsample", "sweep", sweep_cost(n / ROW_SAMPLE_1_IN, 200000), None,
              f"1.7M original (PLY), 1-in-{ROW_SAMPLE_1_IN} row sample: no bins, no assembly; nnz "
              f"BRACKET [D, 2D] scaled from partners/face (model ~{nnz:.1e})",
              run=dict(mesh=m["path"], n_faces=n, nrays=200000, assemble=False, row_sample_1_in=ROW_SAMPLE_1_IN))
        P.dropped.append(("C_moon3d_orig_nr2e5_sweep (all rows, 1.8 h)",
                          f"1-in-{ROW_SAMPLE_1_IN} row sample gives the same per-face cost and partners/face"))
        P.add("C", "C_moon3d_orig_radoff_gpu1", "engine", radoff_cost(n, 1), engine_mem(n),
              "full lunation; the only >1e6 Moon3D illumination/thermal point", gpu=True, repeats=1,
              run=dict(mesh=m["path"], n_faces=n,
                       overrides={"radiation.enabled": False, "thermal.use_gpu": True,
                                  "thermal.gpu_devices": 1}))

    # ------------------------------------------------------------ arm D: Haworth
    # DEM-limited, spans < 1 decade (99.7k-900k): comparability with the paper matrices,
    # not an exponent family.
    existing = {"haworth_radcore_r32c": (HOME / "projects/moon3d/vf/resweep_v4_2026-08-24/haworthr32c/"
                                         "haworth_radcore_r32c_v4.csr", 567837786),
                "sunlit_radcore_r50": (HOME / "projects/moon3d/vf/resweep_v4_2026-08-24/sunlit_radcore_r50/"
                                       "sunlit_radcore_r50_v4.csr", 573040248)}
    for fam in ("haworth_radcore_r25", "haworth_radcore_r25_100m", "haworth_radcore_r32c",
                "sunlit_radcore_r50", "haworth_r15"):
        for m in by.get(fam, []):
            n = m["n_faces"]
            tag = f"D_{fam}_n{n}"
            geo = dict(mesh=m["path"], n_faces=n)
            P.add("D", f"{tag}_radoff_gpu1", "engine", radoff_cost(n, 1), engine_mem(n),
                  "full lunation", gpu=True, repeats=2,
                  run=dict(overrides={"radiation.enabled": False, "thermal.use_gpu": True,
                                      "thermal.gpu_devices": 1}, **geo))
            wov = {**WINDOW, "thermal.use_gpu": True, "thermal.gpu_devices": 1, "radiation.use_gpu": False}
            if fam in existing:
                csr, nnz = existing[fam]
                w, mem = window_cost(n, nnz)
                P.add("D", f"{tag}_window_v4", "engine", w, mem, "SHORT window on the paper matrix (v4)",
                      gpu=True, repeats=2, run=dict(csr=str(csr), overrides=wov, **geo))
                P.dropped.append((f"{tag}_coupled_v4 (full lunation)", "short window measures the "
                                  "same per-step / per-iteration costs; Stage 1 r50 runs are GPU-"
                                  "radiosity route, not comparable for radiosity"))
                continue
            nr = 20000 if fam == "haworth_r15" else 200000
            nnz = n * nnz_per_face("haworth", n, nr)
            if fam == "haworth_radcore_r25_100m" and not with_haworth_refinement:
                P.add("D", f"{tag}_nr{nr:g}_rowsample", "sweep", sweep_cost(n / ROW_SAMPLE_1_IN, nr), None,
                      f"1-in-{ROW_SAMPLE_1_IN} row sample: per-face cost + nnz BRACKET (full sweep is "
                      "--with-haworth-refinement)",
                      run=dict(nrays=nr, assemble=False, row_sample_1_in=ROW_SAMPLE_1_IN, **geo))
                P.dropped.append((f"{tag}_nr200000_sweep (full, 49 GB disk) + _coupled",
                                  "not needed for Arm A or for any exponent (Haworth spans <1 decade); "
                                  "Paul's 1.8x override for it is retained in --with-haworth-refinement"))
                continue
            asm_t, asm_m = asm_cost(nnz)
            P.add("D", f"{tag}_nr{nr:g}_sweep", "sweep", sweep_cost(n, nr) + asm_t, asm_m,
                  f"nnz {nnz:.2e}", run=dict(nrays=nr, **geo), nnz=nnz)
            w, mem = window_cost(n, nnz)
            P.add("D", f"{tag}_window", "engine", w, mem, "SHORT coupled window; CPU radiosity",
                  gpu=True, repeats=2,
                  run=dict(csr=str(Path(root) / f"{tag}_nr{nr:g}_sweep" / "vf.csr"), overrides=wov, **geo))

    # Every override must name a scheduled row: a typo would otherwise silently leave
    # the default in place for the row the ruling was about.
    ids = {r["id"] for r in P.runs}
    missing = [k for k in DISK_GUARD_OVERRIDES if k not in ids]
    by_design = [k for k in missing if k.startswith("D_haworth_radcore_r25_100m") and not with_haworth_refinement]
    if set(missing) - set(by_design):
        raise SystemExit(f"FATAL: disk-guard override(s) name no scheduled row: {sorted(set(missing) - set(by_design))}")
    for k in by_design:
        print(f"NOTE: disk-guard override for {k} is NOT applied: that row is not scheduled "
              "(run with --with-haworth-refinement to schedule it, override included)")
    # Front-load: each FRONT_LOAD row plus every row that consumes its CSR, in their
    # original relative order, then everything else.
    front = set()
    for rid in [x for x in FRONT_LOAD if x in ids]:
        front.add(rid)
        csr = str(Path(root) / rid / "vf.csr")
        front |= {r["id"] for r in P.runs if r.get("csr") == csr}
    P.runs = [r for r in P.runs if r["id"] in front] + [r for r in P.runs if r["id"] not in front]
    P.rows = [r for r in P.rows if r["id"] in front] + [r for r in P.rows if r["id"] not in front]
    P.front = front
    return P


GPU_SAFETY = """GPU safety (plan 4.5), enforced by run_bench.py on every GPU row:
  thermal.gpu_batch_size 1280, thermal.gpu_inter_batch_pause_ms 1500, PTM_RAD_RERUN_MAX_GPUS=1;
  coupled rows N=1 thermal, radiosity on the CPU (every arm matrix > 1e8 nnz except the smallest;
  >1e8 nnz on a GPU is a known hard-wedge trigger; 2^31 wall at 5.37e8); N=4 only on rad-off rows.
  Illumination on CPU Embree (multi-ISA ~/opt/embree-4.3.3-avx512). Coupled rows ALWAYS load a
  CSR through PTM_HMATRIX_VF_REPLACE over a make_empty_vf stub and must log 'REPLACED H-matrix'.
  GPU rows need --gpu-ok; the harness refuses a busy machine (foreign ptm_thermal_sim / mc_ref_pp
  or load > 2). Run detached (nohup caffeinate), one job at a time, gate on the ledger 'done'."""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=str(HOME / "projects/moon3d_R1/bench/runs"))
    ap.add_argument("--write", default=None, help="write plan.json for run_bench.py")
    ap.add_argument("--json-table", default=None, help="write the predicted table as JSON")
    ap.add_argument("--with-haworth-refinement", action="store_true",
                    help="schedule the full r25_100m NR 2e5 sweep (49 GB) + its coupled window")
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    P = build(man, a.root, a.with_haworth_refinement)
    tot = {}
    held = [r for r in P.rows if r["status"] != "ok"]
    # Disk: run the scheduled rows in order (repeats expanded) with the runner's own
    # deletion rules: M1/M2/M4 are inside each row's (peak, final); a harness CSR is
    # held from its sweep until its last consumer finishes (M3).
    ok_rows = [r for r in P.rows if r["status"] == "ok"]
    remaining = {}
    for r in ok_rows:
        if r["consumes"]:
            remaining[r["consumes"]] = remaining.get(r["consumes"], 0) + r["repeats"]
    alive, batch_peak, peak_at, csr_alive = 0.0, 0.0, None, {}
    for r in ok_rows:
        r["disk_before"] = alive
        for rep_ in range(r["repeats"]):
            if alive + r["disk_peak"] > batch_peak:
                batch_peak, peak_at = alive + r["disk_peak"], r["id"]
            alive += r["disk_final"]
            if r["produces"] and rep_ == 0:
                csr_alive[r["produces"]] = r["csr_bytes"]
                alive += r["csr_bytes"]
            for path in (r["consumes"], r["produces"]):
                if path in csr_alive and remaining.get(path, 0) - (path == r["consumes"]) <= 0:
                    alive -= csr_alive.pop(path)
            if r["consumes"] in remaining:
                remaining[r["consumes"]] -= 1
        r["disk_after"] = alive
    by_id = {r["id"]: r for r in ok_rows}
    for run in P.runs:   # the dry run subtracts what earlier rows leave on disk
        run["predicted_disk_retained_before_bytes"] = by_id[run["id"]]["disk_before"]
    print(f"{'arm':3s} {'id':58s} {'kind':6s} {'pred wall':>9s} {'x':>2s} {'peak mem':>9s} "
          f"{'disk pk':>8s} {'disk fin':>8s}  note")
    for r in P.rows:
        flag = "" if r["status"] == "ok" else f" [{r['status']}]"
        fin = r["disk_final"] + r["csr_bytes"]
        print(f"{r['arm']:3s} {r['id'][:58]:58s} {r['kind']:6s} {hms(r['wall']):>9s} "
              f"{r['repeats']:>2d} {gb(r['mem']):>9s} {gb(r['disk_peak']):>8s} {gb(fin):>8s}  "
              f"{r['note']}{' (GPU)' if r['gpu'] else ''}{flag}")
        if r["status"] != "ok":
            continue
        w = (r["wall"] or 0) * r["repeats"]
        t = tot.setdefault(r["arm"], [0.0, 0, 0.0])
        t[0] += w
        t[1] += 1
        if r["gpu"]:
            t[2] += w
    print()
    grand = 0.0
    for arm, (w, k, g) in sorted(tot.items()):
        grand += w
        print(f"arm {arm}: {k:3d} scheduled rows, predicted {hms(w)} (GPU-holding rows {hms(g)}), +-2x")
    print(f"TOTAL scheduled machine time {hms(grand)} = {grand / 86400:.1f} days (+-2x); worst "
          f"scheduled RAM peak {gb(max((r['mem'] or 0) for r in ok_rows))} vs "
          f"{gb(CAPS['ram_bytes'])} usable")
    worst_row = max(ok_rows, key=lambda r: r["disk_peak"])
    print(f"DISK: batch peak {gb(batch_peak)} (at {peak_at}); largest single-row peak "
          f"{gb(worst_row['disk_peak'])} ({worst_row['id']}); final footprint "
          f"after the batch {gb(alive)}. Without M1-M4 the same rows would leave "
          f"{gb(sum((r['disk_peak']) * r['repeats'] for r in ok_rows))}.")
    print(f"DISK GUARD: default free >= {DISK_GUARD_DEFAULT:g} x row peak + reserve; per-row "
          f"overrides ({len(DISK_GUARD_OVERRIDES)}):")
    for r in ok_rows:
        if r["id"] in DISK_GUARD_OVERRIDES:
            f = DISK_GUARD_OVERRIDES[r["id"]][0]
            print(f"  {r['id']}: factor {f:g}, peak {r['disk_peak'] / 2**30:.2f} GiB -> needs "
                  f"{(f * r['disk_peak']) / 2**30 + 15:.1f} GiB free at 15 GiB reserve (default "
                  f"would be {(2 * r['disk_peak']) / 2**30 + 15:.1f}); batch position "
                  f"{ok_rows.index(r) + 1}/{len(ok_rows)}, retained footprint before it "
                  f"{r['disk_before'] / 2**30:.2f} GiB")
    rest = [r for r in ok_rows if r["id"] not in DISK_GUARD_OVERRIDES]
    worst = max(rest, key=lambda r: r["disk_peak"])
    print(f"  largest non-override row: {worst['id']} peak {worst['disk_peak'] / 2**30:.2f} GiB -> "
          f"needs {2 * worst['disk_peak'] / 2**30 + 15:.1f} GiB free")
    prof = [r["disk_after"] for r in ok_rows]
    print(f"RETAINED FOOTPRINT (after each row, M1-M4 applied): max {max(prof) / 2**30:.2f} GiB, "
          f"median {sorted(prof)[len(prof) // 2] / 2**30:.2f} GiB, end {prof[-1] / 2**30:.2f} GiB; "
          f"batch peak (max over rows of retained + that row's peak) {batch_peak / 2**30:.2f} GiB")
    print(f"HELD (not scheduled): {len(held)} rows -- " + ", ".join(r["id"] for r in held))
    print(f"DROPPED / REPLACED vs the full-lunation plan ({len(P.dropped)}):")
    for rid, why in P.dropped:
        print(f"  - {rid}: {why}")
    chk = full_coupled_cost(509337, 745322408)
    print(f"MODEL CHECK: the recalibrated rates predict a full Moon3D 509k coupled lunation at "
          f"{chk:.0f} s; the 10 clean Stage-3 runs measured 1747-2114 s.")
    print()
    print(GPU_SAFETY)
    if a.write:
        # floor facelists for the floor-only bowl sweeps (mask == 1 faces)
        import numpy as np
        fl_dir = Path(a.root) / "facelists"
        fl_dir.mkdir(parents=True, exist_ok=True)
        for m in man:
            if m.get("family") == "bowl" and m.get("mask"):
                idx = np.nonzero(np.load(m["mask"]) == 1)[0]
                (fl_dir / f"bowl_n{m['n_faces']}_dd{m['dd']:g}_floor.txt").write_text(
                    "\n".join(map(str, idx)) + "\n")
        Path(a.write).write_text(json.dumps({"root": a.root, "rates": RATES, "caps": CAPS,
                                             "runs": P.runs}, indent=1))
        print(f"\nwrote {a.write} ({len(P.runs)} runnable rows)")
    if a.json_table:
        Path(a.json_table).write_text(json.dumps(P.rows, indent=1, default=str))


if __name__ == "__main__":
    main()
