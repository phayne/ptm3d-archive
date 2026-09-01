#!/usr/bin/env python3
"""
Build a triangulated PLY mesh from a DEM file for a specified ROI.
Wraps the dtm_to_ply_delaunay C++ converter.
"""
import subprocess
import sys
from pathlib import Path

from survey_config import SurveyConfig, load_config
from utils import save_state, load_state


def find_converter():
    """Find the dtm_to_ply_delaunay binary.

    The in-repo build (build/bin/) comes first: dtm_to_ply_delaunay is built
    from src/cpp/tools/ by this repository's CMake, so that is the binary
    matching this checkout. The legacy standalone ``dtm/`` locations stay as
    fallbacks for machines that have not rebuilt yet -- those binaries predate
    the port and may target a different CPU architecture. See BUGS.md PTM-04.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "build" / "bin" / "dtm_to_ply_delaunay",
        # Legacy locations, kept so existing setups keep working. The first is
        # the original candidate; the second is the sibling `dtm/` directory
        # the tools actually lived in before the port.
        repo_root / "dtm" / "dtm_to_ply_delaunay",
        repo_root.parent / "dtm" / "dtm_to_ply_delaunay",
        Path("dtm/dtm_to_ply_delaunay"),
        Path("../dtm/dtm_to_ply_delaunay"),
    ]
    for p in candidates:
        if p.exists():
            print(f"[build_mesh] dtm_to_ply_delaunay: {p}")
            return str(p)
    return None


def build_mesh(cfg: SurveyConfig) -> bool:
    """Build the PLY mesh from DEM + ROI configuration."""
    converter = find_converter()
    if not converter:
        repo_root = Path(__file__).resolve().parents[2]
        print("ERROR: dtm_to_ply_delaunay not found. Build it from this "
              "repository:\n"
              f"    cd {repo_root}\n"
              "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
              "    cmake --build build --target dtm_to_ply_delaunay -j 8\n"
              "PREREQUISITE: GDAL must be installed and visible to CMake, or "
              "the target is skipped at configure time.\n"
              "See INSTALL.md ('DEM / DTM tools') and "
              "docs/dem-to-mesh-workflow.md.")
        return False

    mesh_path = cfg.mesh_path
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        converter,
        cfg.dem.file,
        str(mesh_path),
        "--roi-center", str(cfg.roi.center_lat), str(cfg.roi.center_lon),
        "--roi-radius", str(cfg.roi.radius_km),
        "--roi-max-points", str(cfg.roi.max_points),
        "--density-exp", str(cfg.roi.density_exp),
        "--roi-core-frac", str(cfg.roi.core_frac),
        "--body-radius", str(cfg.body_radius),
    ]
    if cfg.dem.south_pole_coords:
        cmd.append("--south-pole-coords")

    print(f"Building mesh: {cfg.dem.file} -> {mesh_path}")
    print(f"  ROI: ({cfg.roi.center_lat}, {cfg.roi.center_lon}), r={cfg.roi.radius_km} km")
    print(f"  Max points: {cfg.roi.max_points}, density_exp: {cfg.roi.density_exp}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"ERROR: Mesh generation failed:\n{result.stderr}")
        return False

    if not mesh_path.exists():
        print(f"ERROR: Mesh file not created: {mesh_path}")
        return False

    # Parse output for face/vertex counts
    nv = nf = 0
    for line in result.stdout.split('\n'):
        if 'Vertices:' in line and 'Faces:' in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == 'Vertices:':
                    nv = int(parts[i+1])
                if p == 'Faces:':
                    nf = int(parts[i+1])

    print(f"  Mesh created: {nv:,} vertices, {nf:,} faces")

    # Update state
    state = load_state(cfg.state_file)
    state['mesh'] = {'status': 'complete', 'path': str(mesh_path), 'vertices': nv, 'faces': nf}
    save_state(cfg.state_file, state)

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python build_mesh.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    errors = cfg.validate()
    if errors:
        print("Config errors:", errors)
        sys.exit(1)
    if not build_mesh(cfg):
        sys.exit(1)
