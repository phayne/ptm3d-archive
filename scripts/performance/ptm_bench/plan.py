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
    # CPU radiosity: CSR traversal throughput, 48 threads. A3GT Y00 log 2.48e9 nnz at
    # 3.166 s/step (two-band fused). ASSUMPTION (Stage 3 probe will replace it).
    "cpu_nnz_traversals_per_s": 1.6e9,
    "traversals_per_solve": 4.0,          # fused two-band Jacobi, rho-dependent; 2 measured at rho 0.15
    "steps_per_sweep": 600,               # 360 output + 240 spin-up steps swept (smoke log: 3000 solves / 5 sweeps)
    "picard_sweeps_terrain": 5,           # measured 4-5 on terrain/fBm smokes
    "picard_sweeps_moon3d": 12,           # rho 0.78 -> rate 0.77/sweep; cap 16
    # Engine rad-off lunation on the GPU (illumination + thermal), Moon3D 509k: 115-124 s.
    "gpu_radoff_s_per_face": 120 / 509337,
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
    "repeat_max_wall_s": 1800,            # timing repeats (n=2) only for rows <= 30 min
}
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

    def add(self, arm, rid, kind, wall, mem, note="", run=None, gpu=False, repeats=1):
        status = "ok"
        if mem is not None and mem > CAPS["ram_bytes"]:
            status = "OVER-RAM-CAP"
            if kind == "engine":
                status = "PROBE-FIRST"   # streamed thermal may fit; unmeasured -> not scheduled
                note += "; in-core model exceeds RAM, streamed path unmeasured: run after a 1e6 RSS check"
        if wall is not None and wall > RATES["repeat_max_wall_s"]:
            repeats = 1
        self.rows.append(dict(arm=arm, id=rid, kind=kind, wall=wall, mem=mem, note=note,
                              status=status, gpu=gpu, repeats=repeats))
        if run is not None and status == "ok":
            run.update(id=rid, kind=kind, repeats=repeats, predicted_wall_s=wall,
                       predicted_peak_bytes=mem)
            self.runs.append(run)


def sweep_cost(n_src, nr):
    return n_src * nr / RATES["mc_rays_per_s_48"]


def asm_cost(nnz):
    return nnz * RATES["asm_s_per_nnz"], nnz * RATES["asm_bytes_per_nnz"]


def engine_mem(n):
    return RATES["engine_bytes_base"] + n * RATES["engine_bytes_per_face"]


def coupled_cost(n, nnz, sweeps, gpu_thermal=True):
    rad = sweeps * RATES["steps_per_sweep"] * RATES["traversals_per_solve"] * nnz / RATES["cpu_nnz_traversals_per_s"]
    base = n * RATES["gpu_radoff_s_per_face"] if gpu_thermal else n * (
        RATES["cpu_illum_s_per_face_48"] + 2 * RATES["cpu_thermal_s_per_face_pass_48"])
    reth = sweeps * n * (RATES["gpu_radoff_s_per_face"] * 0.8 if gpu_thermal
                         else RATES["cpu_thermal_s_per_face_pass_48"])
    mem = nnz * 8 + engine_mem(n)         # CSR in the solver + per-face series
    return base + rad + reth, mem


def build(man, root):
    P = Plan(root, man)
    by = {}
    for m in man:
        by.setdefault(m["family"], []).append(m)

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
            mc.append(("Eref", 2000000, 0.2))   # high-NR reference at the same N
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
                  f"nnz {nnz:.2e}", run=dict(nrays=nr, keps=keps, **geo))
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
        for m in sorted(by.get(fam, []), key=lambda m: m["n_faces"]):
            n = m["n_faces"]
            tag = f"B_{fam[4:]}_n{n}"
            geo = dict(mesh=m["path"], n_faces=n)
            cpu_t = n * (RATES["cpu_illum_s_per_face_48"] + 2 * RATES["cpu_thermal_s_per_face_pass_48"])
            P.add("B", f"{tag}_radoff_cpu", "engine", cpu_t, engine_mem(n), "illum + thermal, CPU",
                  run=dict(overrides={"radiation.enabled": False}, **geo), repeats=2)
            g1 = n * RATES["gpu_radoff_s_per_face"]
            P.add("B", f"{tag}_radoff_gpu1", "engine", g1, engine_mem(n), "GPU N=1", gpu=True,
                  run=dict(overrides={"radiation.enabled": False, "thermal.use_gpu": True,
                                      "thermal.gpu_devices": 1}, **geo), repeats=2)
            P.add("B", f"{tag}_radoff_gpu4", "engine", g1 / RATES["gpu_n4_thermal_speedup"],
                  engine_mem(n), "GPU N=4 (rad-off only; validated physics-neutral)", gpu=True,
                  run=dict(overrides={"radiation.enabled": False, "thermal.use_gpu": True,
                                      "thermal.gpu_devices": 4}, **geo), repeats=2)
            for nr in (20000, 200000):
                if nr == 200000 and n > 1.1e6:
                    continue
                nnz = n * nnz_per_face(fam, n, nr)
                sid = f"{tag}_nr{nr:g}"
                asm_t, asm_m = asm_cost(nnz)
                can_asm = nnz <= CAPS["assembly_nnz_planning"] and n <= 1.1e6
                P.add("B", sid + "_sweep", "sweep", sweep_cost(n, nr) + (asm_t if can_asm else 0),
                      asm_m if can_asm else None,
                      f"nnz {nnz:.2e}" + ("" if can_asm else " -> sweep only, nnz counted"),
                      run=dict(nrays=nr, assemble=can_asm, **geo), repeats=1)
                if can_asm and nr == 20000:
                    w, mem = coupled_cost(n, nnz, RATES["picard_sweeps_terrain"])
                    P.add("B", sid + "_coupled", "engine", w, mem,
                          "coupled; thermal GPU N=1, radiosity CPU" + (" (>1e8 nnz)" if nnz > 1e8 else ""),
                          gpu=True, repeats=2,
                          run=dict(csr=str(Path(root) / (sid + "_sweep") / "vf.csr"),
                                   overrides={"thermal.use_gpu": True, "thermal.gpu_devices": 1,
                                              "radiation.use_gpu": False}, **geo))

    # ------------------------------------------------------------ arm C: Moon3D
    for m in sorted(by.get("moon3d_dec", []), key=lambda m: m["n_faces"]):
        n = m["n_faces"]
        tag = f"C_moon3d_n{n}"
        geo = dict(mesh=m["path"], n_faces=n)
        nnz = n * nnz_per_face("moon3d", n, 200000)
        asm_t, asm_m = asm_cost(nnz)
        P.add("C", f"{tag}_nr2e5_sweep", "sweep", sweep_cost(n, 200000) + asm_t, asm_m,
              f"nnz {nnz:.2e} (nnz ~ N^2 regime test)", run=dict(nrays=200000, **geo))
        P.add("C", f"{tag}_radoff_gpu1", "engine", n * RATES["gpu_radoff_s_per_face"], engine_mem(n),
              "GPU N=1", gpu=True, repeats=2,
              run=dict(overrides={"radiation.enabled": False, "thermal.use_gpu": True}, **geo))
        w, mem = coupled_cost(n, nnz, RATES["picard_sweeps_moon3d"])
        P.add("C", f"{tag}_coupled", "engine", w, mem, "coupled; CPU radiosity", gpu=True, repeats=2,
              run=dict(csr=str(Path(root) / f"{tag}_nr2e5_sweep" / "vf.csr"),
                       overrides={"thermal.use_gpu": True, "radiation.use_gpu": False}, **geo))
    for m in by.get("moon3d_509k_v4_existing", []):
        n, nnz = 509337, 745322408
        w, mem = coupled_cost(n, nnz, RATES["picard_sweeps_moon3d"])
        P.add("C", "C_moon3d_509k_v4_coupled", "engine", w, mem,
              "paper-lineage mesh + v4 CSR (no re-sweep, PTM-19)", gpu=True, repeats=2,
              run=dict(mesh=str(HOME / "dev/heat3d/moon3d_c_cartesian_NEW_500k_clean_native.ply"),
                       n_faces=n, csr=m["csr"],
                       overrides={"thermal.use_gpu": True, "radiation.use_gpu": False}))
    for m in by.get("moon3d_original", []):
        n = m.get("n_faces", 1698274)
        nnz = n * nnz_per_face("moon3d", n, 200000)
        P.add("C", "C_moon3d_orig_nr2e5_sweep", "sweep", sweep_cost(n, 200000), None,
              f"1.7M original: nnz {nnz:.2e} counted from sweep bins, NO assembly",
              run=dict(mesh=m["path"], n_faces=n, nrays=200000, assemble=False))
        P.add("C", "C_moon3d_orig_radoff_gpu1", "engine", n * RATES["gpu_radoff_s_per_face"], engine_mem(n),
              "GPU N=1", gpu=True, repeats=2,
              run=dict(mesh=m["path"], n_faces=n,
                       overrides={"radiation.enabled": False, "thermal.use_gpu": True}))

    # ------------------------------------------------------------ arm D: Haworth
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
            P.add("D", f"{tag}_radoff_gpu1", "engine", n * RATES["gpu_radoff_s_per_face"], engine_mem(n),
                  "GPU N=1", gpu=True, repeats=2,
                  run=dict(overrides={"radiation.enabled": False, "thermal.use_gpu": True}, **geo))
            if fam in existing:
                csr, nnz = existing[fam]
                w, mem = coupled_cost(n, nnz, RATES["picard_sweeps_terrain"])
                P.add("D", f"{tag}_coupled_v4", "engine", w, mem, "paper matrix (v4), CPU radiosity",
                      gpu=True, repeats=2,
                      run=dict(csr=str(csr), overrides={"thermal.use_gpu": True,
                                                        "radiation.use_gpu": False}, **geo))
                continue
            nr = 20000 if fam == "haworth_r15" else 200000
            nnz = n * nnz_per_face("haworth", n, nr)
            can = nnz <= CAPS["assembly_nnz_planning"]
            asm_t, asm_m = asm_cost(nnz)
            P.add("D", f"{tag}_nr{nr:g}_sweep", "sweep", sweep_cost(n, nr) + (asm_t if can else 0),
                  asm_m if can else None, f"nnz {nnz:.2e}" + ("" if can else " -> sweep only"),
                  run=dict(nrays=nr, assemble=can, **geo))
            if can:
                w, mem = coupled_cost(n, nnz, RATES["picard_sweeps_terrain"])
                P.add("D", f"{tag}_coupled", "engine", w, mem, "coupled; CPU radiosity", gpu=True,
                      repeats=2, run=dict(csr=str(Path(root) / f"{tag}_nr{nr:g}_sweep" / "vf.csr"),
                                          overrides={"thermal.use_gpu": True,
                                                     "radiation.use_gpu": False}, **geo))
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
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    P = build(man, a.root)
    print(f"{'arm':3s} {'id':58s} {'kind':6s} {'pred wall':>9s} {'x':>2s} {'peak mem':>9s}  note")
    tot = {}
    held = [r for r in P.rows if r["status"] != "ok"]
    for r in P.rows:
        flag = "" if r["status"] == "ok" else f" [{r['status']}]"
        print(f"{r['arm']:3s} {r['id'][:58]:58s} {r['kind']:6s} {hms(r['wall']):>9s} "
              f"{r['repeats']:>2d} {gb(r['mem']):>9s}  {r['note']}{' (GPU)' if r['gpu'] else ''}{flag}")
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
    ok = [r for r in P.rows if r["status"] == "ok"]
    print(f"TOTAL scheduled machine time {hms(grand)} = {grand / 86400:.1f} days (+-2x); worst "
          f"scheduled peak {gb(max((r['mem'] or 0) for r in ok))} vs {gb(CAPS['ram_bytes'])} usable")
    print(f"HELD (not scheduled): {len(held)} rows -- " + ", ".join(r["id"] for r in held))
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
        Path(a.json_table).write_text(json.dumps(P.rows, indent=1))


if __name__ == "__main__":
    main()
