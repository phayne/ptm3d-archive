#!/usr/bin/env python3
"""Measure which thermophysical config keys actually reach the solver.

For every key that claims to set a material property, run the engine twice --
once at the body default, once at a deliberately different value -- and report
the temperature-field difference. A key whose max |dT| over all faces is
identically zero did not reach the solver, whatever the log said.

This exists because a log line that says "override applied" is not evidence:
BUGS.md PTM-53 and PTM-123/161 are both cases where the banner announced a
change the solver never saw.

Usage (from the repository root):

    python3 scripts/thermophys_channel_audit.py --list
    python3 scripts/thermophys_channel_audit.py --engine build-cpu/bin/ptm_thermal_sim \
        --workdir ~/projects/thermophys-audit --case ks_override

Every case is one entry in CASES below: a base config, a set of dotted YAML
overrides and/or extra CLI arguments, and the arm it is compared against.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]

MOON_BASE = "examples/quickstart_moon_diurnal.yaml"
AST_BASE = "examples/quickstart_asteroid_diurnal.yaml"


# --- config plumbing ---------------------------------------------------------

def set_dotted(config, dotted, value):
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def del_dotted(config, dotted):
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.get(part)
        if not isinstance(node, dict):
            return
    node.pop(parts[-1], None)


# --- running -----------------------------------------------------------------

def run_arm(engine, base, overrides, drops, extra_args, outdir, cfgdir, name,
            force=False):
    """Run one arm; return (outdir, log_text, returncode, seconds)."""
    outdir = Path(outdir)
    cfg_path = Path(cfgdir) / f"{name}.yaml"
    log_path = Path(cfgdir).parent / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((REPO / base).read_text()) or {}
    for key in drops or []:
        del_dotted(config, key)
    for key, value in (overrides or {}).items():
        set_dotted(config, key, value)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))

    stats = outdir / "statistics" / "temperature_max.bin"
    if stats.exists() and not force:
        return outdir, log_path.read_text() if log_path.exists() else "", 0, 0.0

    cmd = [str(engine), "--config", str(cfg_path), "--output", str(outdir)]
    cmd += list(extra_args or [])
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    dt = time.time() - t0
    log = proc.stdout + "\n===STDERR===\n" + proc.stderr
    log_path.write_text(log)
    return outdir, log, proc.returncode, dt


def read_bin(path, n=None):
    data = Path(path).read_bytes()
    count = struct.unpack("<I", data[:4])[0]
    if len(data) == 4 + count * 4:
        arr = np.frombuffer(data[4:], dtype="<f4")
    else:
        arr = np.frombuffer(data, dtype="<f4")
    return arr.astype(np.float64)


def field(outdir, which="max"):
    return read_bin(Path(outdir) / "statistics" / f"temperature_{which}.bin")


def md5(path):
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def compare(ref_dir, arm_dir):
    out = {}
    for which in ("max", "min", "avg"):
        a = field(ref_dir, which)
        b = field(arm_dir, which)
        d = b - a
        out[which] = {
            "max_abs_dT": float(np.max(np.abs(d))),
            "mean_dT": float(np.mean(d)),
            "ref_median": float(np.median(a)),
            "arm_median": float(np.median(b)),
        }
    # diurnal amplitude, median over faces
    amp_a = np.median(field(ref_dir, "max") - field(ref_dir, "min"))
    amp_b = np.median(field(arm_dir, "max") - field(arm_dir, "min"))
    out["amplitude_median"] = {"ref": float(amp_a), "arm": float(amp_b),
                               "d": float(amp_b - amp_a)}
    out["md5_ref"] = md5(Path(ref_dir) / "statistics" / "temperature_max.bin")
    out["md5_arm"] = md5(Path(arm_dir) / "statistics" / "temperature_max.bin")
    out["bytes_identical"] = out["md5_ref"] == out["md5_arm"]
    return out


# --- the cases ----------------------------------------------------------------
# Each: base config, overrides applied on top of the REFERENCE arm's config.
# "ref" names another case (or "base") whose run is the comparison arm.

def C(name, base, overrides=None, drops=None, args=(), ref="moon_base",
      note="", expect="LIVE"):
    return dict(name=name, base=base, overrides=overrides or {}, drops=drops or [],
                args=list(args), ref=ref, note=note, expect=expect)


CASES = [
    # ---- reference arms -----------------------------------------------------
    C("moon_base", MOON_BASE, ref=None, note="Moon ico3 CPU reference"),

    # ---- thermophysical: keys the parser READS on a built-in body -----------
    C("ks_override", MOON_BASE, {"thermophysical.ks_override": 0.00296},
      note="4x Moon ks (7.4e-4)"),
    C("kd_override", MOON_BASE, {"thermophysical.kd_override": 0.014},
      note="4x Moon kd (3.5e-3)"),
    C("H_override", MOON_BASE, {"thermophysical.H_override": 0.03},
      note="H 0.06 -> 0.03 m (Hayne+2017 Table A1 e-folding depth)"),
    C("albedo_override", MOON_BASE, {"thermophysical.albedo_override": 0.30},
      note="A0 0.12 -> 0.30"),
    C("emissivity_override", MOON_BASE, {"thermophysical.emissivity_override": 0.80},
      note="eps 0.95 -> 0.80"),
    C("constant_thermal_inertia", MOON_BASE,
      {"thermophysical.constant_thermal_inertia": 200.0},
      note="constant-TI mode, 200 SI"),
    C("conductivity_model", MOON_BASE,
      {"thermophysical.conductivity_model": "martinez2021"},
      note="Martinez & Siegler (2021) conduction law"),
    C("incidence_angle_albedo_off", MOON_BASE,
      {"thermophysical.incidence_angle_albedo": False},
      note="Hapke gate OFF (base has it ON)"),
    C("albedo_model_hapke", MOON_BASE,
      {"thermophysical.incidence_angle_albedo": False,
       "thermophysical.albedo_model": "hapke"},
      note="albedo_model: hapke == incidence_angle_albedo: true"),
    C("albedo_hapke_w", MOON_BASE, {"thermophysical.albedo_hapke_w": 0.90},
      note="explicit single-scattering w"),
    C("use_default_albedo_false", MOON_BASE,
      {"thermophysical.use_default_albedo": False},
      note="sets only the albedo_explicit flag"),

    # ---- thermophysical: plainly-named keys (PTM-67 suspects) ---------------
    C("plain_albedo", MOON_BASE, {"thermophysical.albedo": 0.30},
      note="plain name, not *_override", expect="DEAD"),
    C("plain_emissivity", MOON_BASE, {"thermophysical.emissivity": 0.80},
      expect="DEAD"),
    C("plain_ks", MOON_BASE, {"thermophysical.ks": 0.00296}, expect="DEAD"),
    C("plain_kd", MOON_BASE, {"thermophysical.kd": 0.014}, expect="DEAD"),
    C("plain_rhos", MOON_BASE, {"thermophysical.rhos": 2200.0}, expect="DEAD"),
    C("plain_rhod", MOON_BASE, {"thermophysical.rhod": 3600.0}, expect="DEAD"),
    C("plain_cp0", MOON_BASE, {"thermophysical.cp0": 1200.0}, expect="DEAD"),
    C("plain_H", MOON_BASE, {"thermophysical.H": 0.30}, expect="DEAD"),
    C("plain_Qb", MOON_BASE, {"thermophysical.Qb": 5.0}, expect="DEAD"),
    C("plain_chi", MOON_BASE, {"thermophysical.chi": 0.0}, expect="DEAD"),
    C("misspelled_key", MOON_BASE, {"thermophysical.ks_overide": 0.00296},
      note="typo: does the user get told?", expect="DEAD"),

    # ---- thermal: material keys --------------------------------------------
    C("geothermal_heat_flow", MOON_BASE, {"thermal.geothermal_heat_flow": 5.0},
      note="Qb 0.018 -> 5.0 W/m2 (floor 24.04 K -> 98.15 K if live)", expect="?"),
    C("geothermal_normalization", MOON_BASE,
      {"thermal.geothermal_heat_flow": 0.022},
      note="the lunar campaign value 0.022 vs the 0.018 default: the delta a "
           "config normalization would introduce IF the knob were live "
           "(floor 24.04 K -> 25.28 K)", expect="?"),
    C("geothermal_cli", MOON_BASE, args=["--geothermal-heat-flow", "5.0"],
      note="same via CLI", expect="?"),
    C("heat_flow_mask", MOON_BASE,
      {"thermal.heat_flow_mask_file": "HEATMASK", "thermal.heat_flow_mask_value": 5.0},
      note="per-face Qb map, all faces", expect="LIVE"),
    C("thermal_m", MOON_BASE, {"thermal.thermal_m": 20}, note="grid layers/skin"),
    C("thermal_n", MOON_BASE, {"thermal.thermal_n": 50}, note="PTM-165 suspect",
      expect="DEAD"),
    C("thermal_b", MOON_BASE, {"thermal.thermal_b": 30}, note="skin depths deep"),
    C("misspelled_section_key", MOON_BASE, {"thermal.equilibration_orbit": 8.0},
      note="typo outside thermophysical:", expect="DEAD"),

    # ---- CLI thermal-inertia channels ---------------------------------------
    C("cli_thermal_inertia", MOON_BASE, args=["--thermal-inertia", "200"],
      note="ks-scaling convention"),
    C("cli_constant_ti", MOON_BASE, args=["--constant-thermal-inertia", "200"],
      note="constant-in-T-and-depth convention"),
    C("cli_albedo", MOON_BASE, args=["--albedo", "0.30"],
      note="should match thermophysical.albedo_override 0.30"),

    # ---- per-face property maps on the Moon ---------------------------------
    C("thermal_inertia_map", MOON_BASE,
      {"thermal.thermal_inertia_map": "TIMAP"},
      note="uniform TI=200 map over all 1280 faces", expect="LIVE"),
    C("h_parameter_map", MOON_BASE, {"thermal.h_parameter_map": "HMAP"},
      note="uniform H=0.30 m map", expect="LIVE"),

    # ---- the per-face-H equivalence discriminator ---------------------------
    # A UNIFORM per-face H map states exactly the same physics as the scalar
    # H_override, so on any path that honors both they must agree. This is the
    # check that catches a silent no-op: "the map loaded" is a log line, "the
    # map changed the answer, by the right amount" is evidence.
    #
    # TWO tolerances, deliberately. The map is float32 on disk, so a YAML 0.03
    # and a map 0.03 are NOT the same number (0.03 has no exact binary32 form);
    # that pair can only be compared to round-off. 0.03125 = 2^-5 IS exact in
    # both, so that pair must be BYTE-IDENTICAL -- which is the stronger claim,
    # and the one that proves the two channels reach the same code rather than
    # merely landing nearby.
    C("h_map_0p06", MOON_BASE, {"thermal.h_parameter_map": "HMAP06"},
      note="uniform H = the Moon default 0.06 m; must equal moon_base to f32",
      expect="~SAME"),
    C("h_map_0p03", MOON_BASE, {"thermal.h_parameter_map": "HMAP03"},
      ref="H_override",
      note="uniform H=0.03 map vs scalar H_override=0.03 (f32 round-off only)",
      expect="~SAME"),
    C("H_override_exact", MOON_BASE, {"thermophysical.H_override": 0.03125},
      note="H 0.06 -> 0.03125 m (2^-5: exact in binary32 AND binary64)"),
    C("h_map_exact", MOON_BASE, {"thermal.h_parameter_map": "HMAPX"},
      ref="H_override_exact",
      note="uniform H=0.03125 map vs the same scalar -- must be BYTE-IDENTICAL",
      expect="IDENTICAL"),
    C("use_default_property_map", MOON_BASE,
      {"thermal.use_default_property_map": True},
      note="bundled Moon H map for ico3", expect="LIVE"),

    # ---- albedo map interaction ---------------------------------------------
    C("moon_map_base", MOON_BASE, {"illumination.use_albedo_map": True},
      note="Moon with the bundled albedo map (reference for the two below)"),
    C("map_use_default_albedo_false", MOON_BASE,
      {"illumination.use_albedo_map": True,
       "thermophysical.use_default_albedo": False},
      ref="moon_map_base", note="does use_default_albedo suppress the map?"),
    C("map_plus_albedo_override", MOON_BASE,
      {"illumination.use_albedo_map": True,
       "thermophysical.albedo_override": 0.30},
      ref="moon_map_base", note="scalar override vs a loaded map (rule 7 warning)"),

    # ---- GENERIC BODY: generic_body.surface.* -------------------------------
    C("ast_base", AST_BASE, ref=None, note="Quickstart Asteroid reference"),
    C("gb_albedo", AST_BASE, {"generic_body.surface.albedo": 0.30}, ref="ast_base"),
    C("gb_emissivity", AST_BASE, {"generic_body.surface.emissivity": 0.80},
      ref="ast_base"),
    C("gb_ks", AST_BASE, {"generic_body.surface.ks": 0.03333333}, ref="ast_base",
      note="4x ks"),
    C("gb_kd", AST_BASE, {"generic_body.surface.kd": 0.03333333}, ref="ast_base",
      note="4x kd only (ks unchanged)"),
    C("gb_rhos", AST_BASE, {"generic_body.surface.rhos": 3000.0}, ref="ast_base"),
    C("gb_rhod", AST_BASE, {"generic_body.surface.rhod": 3000.0}, ref="ast_base"),
    C("gb_cp0", AST_BASE, {"generic_body.surface.cp0": 1600.0}, ref="ast_base"),
    C("gb_H", AST_BASE, {"generic_body.surface.H": 0.30,
                         "generic_body.surface.kd": 0.03333333},
      ref="ast_base", note="H only matters when ks != kd, so kd is moved too"),
    C("gb_Qb", AST_BASE, {"generic_body.surface.Qb": 5.0}, ref="ast_base",
      note="Qb 0.018 -> 5.0 on a GENERIC body"),
    C("gb_chi", AST_BASE, {"generic_body.surface.chi": 2.7}, ref="ast_base",
      note="chi 0 -> 2.7"),
    C("gb_geothermal_thermal_section", AST_BASE,
      {"thermal.geothermal_heat_flow": 5.0}, ref="ast_base",
      note="thermal.geothermal_heat_flow on a GENERIC body", expect="?"),
    C("gb_plain_ks", AST_BASE, {"thermophysical.ks": 0.03333333}, ref="ast_base",
      note="plain thermophysical.ks on a generic body", expect="DEAD"),
    C("gb_ks_override", AST_BASE, {"thermophysical.ks_override": 0.03333333},
      ref="ast_base", note="*_override on a generic body"),

    # ---- the two TI conventions on the asteroid (phase 3a's TI pair) -------
    C("ast_ti200_flag", AST_BASE, args=["--thermal-inertia", "200"], ref="ast_base"),
    C("ast_ti200_const", AST_BASE, args=["--constant-thermal-inertia", "200"],
      ref="ast_base"),

    # ---- PTM-60: the gb_* forwarding still uses the 6-decimal setter --------
    # generic_body.surface.ks and thermophysical.ks_override request the SAME
    # physical value; only the second travels at full precision.
    C("prec_gb_ks", AST_BASE, {"generic_body.surface.ks": 1.2345e-5,
                               "generic_body.surface.kd": 1.2345e-5},
      ref="ast_base", note="ks 1.2345e-5 via generic_body.surface"),
    C("prec_ovr_ks", AST_BASE, {"thermophysical.ks_override": 1.2345e-5,
                                "thermophysical.kd_override": 1.2345e-5},
      ref="ast_base", note="the SAME ks via *_override (full precision)"),
    C("prec_gb_ks_tiny", AST_BASE, {"generic_body.surface.ks": 4e-7,
                                    "generic_body.surface.kd": 4e-7},
      ref="ast_base", note="ks 4e-7 -> to_string gives 0.000000", expect="?"),
    C("prec_ovr_ks_tiny", AST_BASE, {"thermophysical.ks_override": 4e-7,
                                     "thermophysical.kd_override": 4e-7},
      ref="ast_base", note="the SAME tiny ks via *_override", expect="?"),

    # ---- the OTHER CPU path: legacy scalar Model::run -----------------------
    C("scalar_base", MOON_BASE, args=["--no-simd-batch"], ref=None,
      note="Moon, legacy scalar CPU solver (reference)"),
    C("scalar_ks_override", MOON_BASE, {"thermophysical.ks_override": 0.00296},
      args=["--no-simd-batch"], ref="scalar_base"),
    C("scalar_albedo_override", MOON_BASE,
      {"thermophysical.albedo_override": 0.30}, args=["--no-simd-batch"],
      ref="scalar_base"),
    C("scalar_geothermal", MOON_BASE, {"thermal.geothermal_heat_flow": 5.0},
      args=["--no-simd-batch"], ref="scalar_base", expect="?"),
    C("scalar_hapke_w", MOON_BASE, {"thermophysical.albedo_hapke_w": 0.90},
      args=["--no-simd-batch"], ref="scalar_base", expect="?"),
    C("scalar_H_override", MOON_BASE, {"thermophysical.H_override": 0.03125},
      args=["--no-simd-batch"], ref="scalar_base",
      note="H 0.06 -> 0.03125 on the legacy CPU scalar column"),
    C("scalar_h_map_exact", MOON_BASE, {"thermal.h_parameter_map": "HMAPX"},
      args=["--no-simd-batch"], ref="scalar_H_override",
      note="uniform H=0.03125 map vs the same scalar, CPU scalar path",
      expect="IDENTICAL"),
]


def by_name():
    return {c["name"]: c for c in CASES}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="build-cpu/bin/ptm_thermal_sim")
    ap.add_argument("--workdir", default=str(Path.home() / "projects/thermophys-audit"))
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--tag", default="cpu")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="add thermal.use_gpu: true (1 device) to every case")
    args = ap.parse_args(argv)

    if args.list:
        for c in CASES:
            print(f"{c['name']:28s} expect={c['expect']:6s} {c['note']}")
        return 0

    work = Path(args.workdir)
    cfgdir = work / "configs" / args.tag
    outroot = work / "out" / args.tag
    cfgdir.mkdir(parents=True, exist_ok=True)
    outroot.mkdir(parents=True, exist_ok=True)

    # a heat-flow mask covering every face of the ico3 mesh
    mask = work / "heatmask_all_faces.txt"
    if not mask.exists():
        mask.write_text("\n".join(str(i) for i in range(1280)) + "\n")
    # uniform per-face property maps (binary float32, one value per face)
    timap = work / "ti_map_200_ico3.bin"
    if not timap.exists():
        timap.write_bytes(np.full(1280, 200.0, dtype="<f4").tobytes())
    hmap = work / "h_map_0p30_ico3.bin"
    if not hmap.exists():
        hmap.write_bytes(np.full(1280, 0.30, dtype="<f4").tobytes())
    # The equivalence pair: a uniform map at the body default (0.06 m) and one
    # at the swept value (0.03 m). float32 on disk, so the literals are chosen
    # to round-trip exactly in binary32 -- 0.03 does not, which is precisely why
    # the scalar arm below reads its target back out of the file rather than
    # trusting the YAML literal.
    hmap06 = work / "h_map_0p06_ico3.bin"
    if not hmap06.exists():
        hmap06.write_bytes(np.full(1280, 0.06, dtype="<f4").tobytes())
    hmap03 = work / "h_map_0p03_ico3.bin"
    if not hmap03.exists():
        hmap03.write_bytes(np.full(1280, 0.03, dtype="<f4").tobytes())
    hmapx = work / "h_map_0p03125_ico3.bin"
    if not hmapx.exists():
        hmapx.write_bytes(np.full(1280, 0.03125, dtype="<f4").tobytes())
    subs = {"HEATMASK": str(mask), "TIMAP": str(timap), "HMAP": str(hmap),
            "HMAP06": str(hmap06), "HMAP03": str(hmap03), "HMAPX": str(hmapx)}

    engine = Path(args.engine)
    if not engine.is_absolute():
        engine = REPO / engine

    todo = CASES if not args.case else [by_name()[n] for n in args.case]
    names = {c["name"] for c in todo}
    # always run the reference arms the selected cases need
    for c in list(todo):
        ref = c.get("ref")
        while ref and ref not in names:
            todo.insert(0, by_name()[ref])
            names.add(ref)
            ref = by_name()[ref].get("ref")

    results = {}
    for c in todo:
        ov = {k: subs.get(v, v) if isinstance(v, str) else v
              for k, v in c["overrides"].items()}
        if args.gpu:
            ov.setdefault("thermal.use_gpu", True)
            ov.setdefault("thermal.gpu_devices", 1)
        outdir = outroot / c["name"]
        _, log, rc, dt = run_arm(engine, c["base"], ov, c["drops"], c["args"],
                                 outdir, cfgdir, f"{args.tag}__{c['name']}",
                                 force=args.force)
        entry = {"rc": rc, "seconds": round(dt, 1), "note": c["note"],
                 "expect": c["expect"]}
        warn = [ln for ln in log.splitlines()
                if "WARNING" in ln or "FATAL" in ln or "no effect" in ln
                or "Thermo override" in ln or "Overriding planet" in ln
                or "Property override" in ln or "Constant thermal inertia" in ln
                or "Thermal inertia (requested)" in ln or "[property-map]" in ln
                or "conductivity" in ln.lower()]
        entry["log_lines"] = warn[:12]
        if rc == 0 and c["ref"]:
            entry.update(compare(outroot / c["ref"], outdir))
        results[c["name"]] = entry
        flag = ""
        if rc != 0:
            flag = "REFUSED(rc!=0)"
        elif c["ref"]:
            flag = ("IDENTICAL" if entry["bytes_identical"]
                    else f"dTmax={entry['max']['max_abs_dT']:.3f} K")
        print(f"{c['name']:28s} rc={rc} {dt:5.1f}s  {flag}")

    (work / f"results_{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {work}/results_{args.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
