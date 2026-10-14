#!/usr/bin/env python3
"""PTM-727 gates G2-G6 (run by run_control_chain.sh after the build).

  G2  Moon, GPU persistent, coupled short window (10,082-face fBm + its CSR), switch OFF:
      new binary == batch binary, every data file byte-identical
  G3  Moon, CPU SIMD batch, coupled short window (968-face fBm + its CSR), switch OFF: same
  G4  generic body (Justitia, res 5), CPU, radiation off: same
  G5  Europa (res 5), CPU, radiation off: same
  G7  PTM-728 falsifying test (Fermi), MEASUREMENT not a gate: the G2 config run on the CPU
      SIMD path vs the GPU, switch OFF and switch series; median cold-trap (T_max < 110 K) and
      never-lit T_avg. PTM-728 predicts a material OFF gap (~2.3 K on Moon3D) that series closes.
  G8  Fermi R2: series on the 30-step (sub-cycle) window must REFUSE (rc != 0, named message).
  G6  switch ON (series) on ONE-LUNATION variants of the G2 and G3 configs (series needs a whole
      cycle, R2): logs 'seed_indirect: series armed' + the whole-cycle and fold (R1) lines, the
      eq-indirect re-solve's anchor census is no longer all fallback-radiation-coupled, and the
      temperature statistics DIFFER from OFF; switch 'mean' on G2 also runs and differs.
Run metadata (config_used.yaml, heartbeat.txt, provenance.txt) is excluded from byte
comparison: it carries paths, times and the binary SHA by design.

  gates.py <old_build_dir> <new_build_dir>
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import yaml

OLD, NEW = Path(sys.argv[1]), Path(sys.argv[2])
C = Path("/Users/paha3326/projects/moon3d_R1/runs/ptm727_control/gates")
WT = Path("/Users/paha3326/projects/moon3d_R1/ptm_bench")
B = Path("/Users/paha3326/projects/moon3d_R1/bench")
META = {"config_used.yaml", "heartbeat.txt", "provenance.txt"}


def tree(d):
    o = {}
    for r, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(r, f)
            o[os.path.relpath(p, d)] = hashlib.md5(open(p, "rb").read()).hexdigest()
    return o


def run(binary, cfg, out, env_extra, tag):
    c = yaml.safe_load(open(cfg))
    c["output"]["directory"] = str(out)
    p = C / f"{tag}.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False))
    env = {k: v for k, v in os.environ.items() if k != "PTM_TIMING_JSON"}
    env.update(env_extra)
    env["OMP_NUM_THREADS"] = "48"
    log = C / f"{tag}.log"
    with open(log, "w") as fh:
        rc = subprocess.run([str(binary / "bin/ptm_thermal_sim"), "--config", str(p)], cwd=WT,
                            env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"FATAL: {tag} rc={rc} (see {log})")
    return log.read_text(errors="replace")


def identical(name, cfg, env_extra):
    a, b = C / f"{name}_old", C / f"{name}_new"
    run(OLD, cfg, a, env_extra, f"{name}_old")
    run(NEW, cfg, b, env_extra, f"{name}_new")
    ta, tb = tree(a), tree(b)
    diff = sorted(k for k in ta if k not in META and ta.get(k) != tb.get(k))
    only = sorted(set(ta) ^ set(tb))
    ok = not diff and not only and len(ta) > 50
    print(f"GATE {name} {'PASS' if ok else 'FAIL'}: {len(ta)} files, {len(diff)} data files differ "
          f"{diff[:3]}, only-in-one {only[:3]}", flush=True)
    return ok


def variant(cfg, tag, sim=None, **thermal):
    c = yaml.safe_load(open(cfg))
    c.setdefault("thermal", {}).update(thermal)
    if sim:
        c.setdefault("simulation", {}).update(sim)
    p = C / f"{tag}_src.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False))
    return p


# series needs >= ONE whole cycle of output window (Fermi R2): G6/G7 run one lunation.
LUNATION = {"timesteps": 360, "duration_days": 29.53}


def acts(name, cfg, env_extra, mode, off_dir):
    """Runs OFF and ON on a one-lunation window (off_dir is ignored: the OFF reference is
    run here on the same window)."""
    on = C / f"{name}_{mode}"
    off = C / f"{name}_off_lunation"
    if not (off / "statistics/temperature_avg.bin").exists():
        run(NEW, variant(cfg, f"{name}_off_lunation", sim=LUNATION, seed_indirect="off"), off,
            env_extra, f"{name}_off_lunation")
    off_dir = off
    log = run(NEW, variant(cfg, f"{name}_{mode}", sim=LUNATION, seed_indirect=mode), on, env_extra,
              f"{name}_{mode}")
    armed = f"seed_indirect: {mode} armed" in log
    if mode == "series":
        armed = armed and "whole cycle(s)" in log and "(R1)" in log
    census = [ln for ln in log.splitlines() if "surface-anchor census" in ln]
    # series: the LAST census (the eq-indirect re-solve's) must have no face left on the
    # radiation-coupled fallback; mean keeps that fallback by design (warm bracket).
    left_fallback = mode == "mean" or (len(census) >= 2
                                       and "fallback-radiation-coupled" not in census[-1])
    s_on = (on / "statistics/temperature_avg.bin").read_bytes()
    s_off = (off_dir / "statistics/temperature_avg.bin").read_bytes()
    differs = s_on != s_off
    ok = armed and left_fallback and differs
    print(f"GATE {name}_{mode} {'PASS' if ok else 'FAIL'}: armed={armed} anchor_left_fallback="
          f"{left_fallback} stats_differ_from_off={differs} last_census='{census[-1][:160] if census else ''}'",
          flush=True)
    return ok


def stats(d):
    import numpy as np
    def rd(n):
        b = (d / "statistics" / f"{n}.bin").read_bytes()
        k = int(np.frombuffer(b[:4], "<u4")[0])
        return np.frombuffer(b[4:4 + 4 * k], "<f4").astype(float)
    return rd("temperature_max"), rd("temperature_avg"), rd("direct_solar_max")


def cpu_gpu_gap(ident, gpu_env):
    """G7 (PTM-728): same config, CPU SIMD vs GPU, OFF and series."""
    import numpy as np
    out = {}
    for mode in ("off", "series"):
        cfg = variant(ident, f"G7_cpu_{mode}", sim=LUNATION, use_gpu=False, seed_indirect=mode)
        run(NEW, cfg, C / f"G7_cpu_{mode}", gpu_env, f"G7_cpu_{mode}")
        gdir = C / ("G6_gpu_off_lunation" if mode == "off" else "G6_gpu_series")
        Tm_g, Ta_g, ds = stats(gdir)
        Tm_c, Ta_c, _ = stats(C / f"G7_cpu_{mode}")
        ct = Tm_g < 110.0
        nl = (ds <= 0.0) & ~ct
        out[mode] = (float(np.median(Ta_c[ct]) - np.median(Ta_g[ct])) if ct.any() else float("nan"),
                     float(np.median((Ta_c - Ta_g)[nl])) if nl.any() else float("nan"), int(ct.sum()), int(nl.sum()))
    for mode, (dct, dnl, nct, nnl) in out.items():
        print(f"MEASURE G7_ptm728 {mode:6s}: CPU - GPU cold-trap T_avg median {dct:+.3f} K (n_ct {nct}); "
              f"never-lit per-face median {dnl:+.3f} K (n {nnl})", flush=True)


def r2_refuses(cfg, env_extra):
    """R2 negative test: series on a SUB-cycle window (the 30-step identity window) must FATAL."""
    c = variant(cfg, "G8_r2_subcycle", seed_indirect="series")
    yaml_c = yaml.safe_load(open(c)); yaml_c["output"]["directory"] = str(C / "G8_r2_subcycle")
    p = C / "G8_r2_subcycle.yaml"; p.write_text(yaml.safe_dump(yaml_c, sort_keys=False))
    env = {k: v for k, v in os.environ.items() if k != "PTM_TIMING_JSON"}; env.update(env_extra)
    r = subprocess.run([str(NEW / "bin/ptm_thermal_sim"), "--config", str(p)], cwd=WT, env=env,
                       capture_output=True, text=True)
    ok = r.returncode != 0 and "at least ONE whole cycle" in (r.stdout + r.stderr)
    print(f"GATE G8_r2_subcycle {'PASS' if ok else 'FAIL'}: rc={r.returncode} "
          f"(series on a 30-step window must refuse)", flush=True)
    return ok


def main():
    C.mkdir(parents=True, exist_ok=True)
    ident = B / "launch/identity/cfg_off.yaml"            # 10,082-face fBm short window, GPU N=1
    csr10k = str(B / "smoke_harness/B_bl_n10082_sweep_nr2e4/vf.csr")
    csr968 = str(B / "smoke_harness/B_bl_n968_sweep_nr2e4/vf.csr")
    gpu_env = {"PTM_HMATRIX_VF_REPLACE": csr10k, "PTM_RAD_RERUN_MAX_GPUS": "1"}
    # G3 config: same window, CPU SIMD, 968 faces
    c3 = yaml.safe_load(open(ident))
    c3["mesh"]["import_file"] = str(B / "meshes/fbm_bandlimited_n968.ply")
    c3["thermal"]["use_gpu"] = False
    stub = C / "empty_stub_968.vf"
    subprocess.run([str(NEW / "bin/make_empty_vf"), "--faces", "968", str(stub)], check=True,
                   capture_output=True)
    c3["radiation"]["view_factor_file"] = str(stub)
    cfg3 = C / "g3_src.yaml"
    cfg3.write_text(yaml.safe_dump(c3, sort_keys=False))
    cpu_env = {"PTM_HMATRIX_VF_REPLACE": csr968}
    dn = B / "donoharm"
    results = [
        identical("G2_moon_gpu", ident, gpu_env),
        identical("G3_moon_cpu_simd", cfg3, cpu_env),
        identical("G4_generic_justitia", dn / "justitia_generic-test_new.yaml", {}),
        identical("G5_europa", dn / "europa_high_res_new.yaml", {}),
        acts("G6_gpu", ident, gpu_env, "series", C / "G2_moon_gpu_new"),
        acts("G6_cpu", cfg3, cpu_env, "series", C / "G3_moon_cpu_simd_new"),
        acts("G6_gpu", ident, gpu_env, "mean", C / "G2_moon_gpu_new"),
        r2_refuses(ident, gpu_env),
    ]
    cpu_gpu_gap(ident, gpu_env)   # G7: reported, never blocks (a measurement of PTM-728)
    print(f"GATES {'ALL PASS' if all(results) else 'FAILED'}: {sum(results)}/{len(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
