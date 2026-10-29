#!/usr/bin/env python3
"""Unknown configuration keys and command-line arguments are reported, not dropped.

Runs `ptm_thermal_sim --validate-config` (parse only: no mesh, no SPICE, no
simulation) on small generated configs and checks four things:

  K1  a misspelled key (thermal.equilibration_orbit) and an unknown section are
      each named in a WARNING, and the run is still accepted (rc 0);
  K2  an unrecognized command-line flag and a stray positional argument are each
      named in a WARNING;
  K3  the shipped quickstart config, which uses only real keys, produces no
      unknown-key warning (no false alarm);
  K4  keys the parser reads directly rather than through its default helper
      (sequences, sub-maps) are not reported as unknown.

Usage: python3 scripts/test_unknown_config_keys.py [--binary build/bin/ptm_thermal_sim]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def validate(binary, cfg, extra=()):
    p = subprocess.run([str(binary), "--config", str(cfg), "--validate-config", *extra],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=str(REPO / "build/bin/ptm_thermal_sim"))
    a = ap.parse_args()
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{name}] {'PASS' if ok else 'FAIL'}" + (f"  {detail}" if detail and not ok else ""))
        fails += not ok

    with tempfile.TemporaryDirectory(prefix="ptm_unknown_keys_") as d:
        d = Path(d)
        typo = d / "typo.yaml"
        typo.write_text("simulation:\n  body: Moon\n"
                        "thermal:\n  equilibration_orbit: 5\n  run_thermal_mesh: true\n"
                        "visualization:\n  colormap: viridis\n")
        rc, out = validate(a.binary, typo)
        check("K1 typo key named", "unknown key: thermal.equilibration_orbit" in out, out[-400:])
        check("K1 unknown section named", "unknown key: visualization (whole section)" in out)
        check("K1 still accepted, with warnings", rc == 0 and "ACCEPTED WITH WARNINGS" in out)

        rc, out = validate(a.binary, REPO / "examples/quickstart_moon_diurnal.yaml",
                           ["--not-a-real-flag", "7"])
        check("K2 unknown flag named", "unrecognized command-line argument '--not-a-real-flag'" in out)
        check("K2 stray positional named", "unrecognized command-line argument '7'" in out)

        rc, out = validate(a.binary, REPO / "examples/quickstart_moon_diurnal.yaml")
        check("K3 clean config, no false alarm",
              rc == 0 and "unknown key:" not in out and "WITH WARNINGS" not in out, out[-400:])

        direct = d / "direct.yaml"
        direct.write_text("simulation:\n  body: Moon\n"
                          "thermal:\n  run_thermal_mesh: true\n  gpu_device_ids: [0]\n"
                          "  state_restore_allow_mismatch: [epoch]\n"
                          "  ice_stability:\n    grain_hop_length_um: 75.0\n"
                          "output:\n  map_depths_m: [0.1]\n"
                          "radiation:\n  beaming: false\n  gpu_device_ids: [0]\n"
                          "illumination:\n  gpu_device_ids: [0]\n")
        rc, out = validate(a.binary, direct)
        check("K4 directly-read keys not flagged", "unknown key:" not in out, out[-600:])

    print(f"[test_unknown_config_keys] {'PASS' if not fails else 'FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
