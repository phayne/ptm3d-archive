#!/usr/bin/env python3
"""
PTM3D Polar Survey Pipeline Orchestrator

Runs the full end-to-end pipeline or individual steps:
  1. mesh         - Build PLY mesh from DEM
  2. illumination - Run illumination survey + duration maps
  3. analyze      - Select target sites based on criteria
  4. thermal      - Run thermal + radiation simulations
  5. traverse     - Plan traverse route (optional)
  6. report       - Generate HTML report (TODO)

Usage:
  python3 scripts/survey/run_survey.py configs/my_survey.yaml
  python3 scripts/survey/run_survey.py configs/my_survey.yaml --step mesh
  python3 scripts/survey/run_survey.py configs/my_survey.yaml --status
"""
import argparse
import json
import sys
from pathlib import Path

from survey_config import load_config


STEPS = ['mesh', 'illumination', 'analyze', 'thermal', 'traverse', 'viewshed', 'plots', 'report']


def print_status(cfg):
    """Print current pipeline state."""
    state_file = cfg.state_file
    if not state_file.exists():
        print("No survey state found. Pipeline has not been started.")
        return

    with open(state_file) as f:
        state = json.load(f)

    print(f"=== Survey: {cfg.name} ===")
    print(f"Output: {cfg.output.base_dir}")
    print()

    step_names = {
        'mesh': '1. Mesh',
        'illumination': '2. Illumination',
        'duration_maps': '3. Duration Maps',
        'selection': '4. Site Selection',
        'thermal': '5. Thermal',
        'traverse': '6. Traverse',
        'viewshed': '7. Viewshed',
        'report': '8. Report',
    }

    for key, label in step_names.items():
        info = state.get(key, {})
        status = info.get('status', 'pending')
        icon = {'complete': '✓', 'partial': '◐', 'pending': '○'}.get(status, '?')
        detail = ''
        if key == 'mesh' and 'faces' in info:
            detail = f" ({info['faces']:,} faces)"
        elif key == 'illumination' and 'lunations_done' in info:
            n = len(info['lunations_done'])
            detail = f" ({n}/{cfg.illumination.num_lunations} lunations)"
        elif key == 'selection' and 'n_locations' in info:
            detail = f" ({info['n_locations']} sites)"
        elif key == 'thermal' and 'lunations_done' in info:
            detail = f" (lunations {info['lunations_done']})"
        elif key == 'traverse' and 'sites_visited' in info:
            detail = f" ({info['sites_visited']} sites, {info.get('total_distance_km', 0):.1f} km)"

        print(f"  {icon} {label}: {status}{detail}")


def run_step(cfg, step):
    """Run a single pipeline step."""
    if step == 'mesh':
        from build_mesh import build_mesh
        return build_mesh(cfg)
    elif step == 'illumination':
        from run_illumination import run_illumination_survey, compute_duration_maps
        ok = run_illumination_survey(cfg)
        if ok and cfg.illumination.compute_duration_maps:
            ok = compute_duration_maps(cfg)
        return ok
    elif step == 'analyze':
        from select_sites import select_sites
        return select_sites(cfg)
    elif step == 'thermal':
        from run_thermal import run_thermal
        return run_thermal(cfg)
    elif step == 'traverse':
        from plan_traverse import plan_traverse
        return plan_traverse(cfg)
    elif step == 'viewshed':
        from run_viewshed import run_viewshed
        return run_viewshed(cfg)
    elif step == 'plots':
        from plot_maps import plot_all
        plot_all(cfg)
        return True
    elif step == 'report':
        from generate_report import generate_report
        ok = generate_report(cfg, compact=False)
        if ok and cfg.output.report_format == 'compact_html':
            generate_report(cfg, compact=True)
        return ok
    else:
        print(f"Unknown step: {step}")
        return False


def run_pipeline(cfg, steps=None):
    """Run all (or selected) pipeline steps in order."""
    if steps is None:
        steps = STEPS

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  PTM3D Polar Survey: {cfg.name:<23s} ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"Body: {cfg.body} (R={cfg.body_radius/1000:.1f} km)")
    print(f"ROI: ({cfg.roi.center_lat}°, {cfg.roi.center_lon}°E), r={cfg.roi.radius_km} km")
    print(f"Time: {cfg.time.start}, {cfg.illumination.num_lunations} lunations")
    print()

    for step in steps:
        print(f"\n{'='*50}")
        ok = run_step(cfg, step)
        if not ok:
            print(f"\nPipeline stopped at step '{step}'.")
            return False

    print(f"\n{'='*50}")
    print("Pipeline complete.")
    print_status(cfg)
    return True


def main():
    parser = argparse.ArgumentParser(description='PTM3D Polar Survey Pipeline')
    parser.add_argument('config', help='YAML configuration file')
    parser.add_argument('--step', choices=STEPS, help='Run a single step')
    parser.add_argument('--status', action='store_true', help='Print pipeline status')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.status:
        print_status(cfg)
        return

    errors = cfg.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Ensure output directory exists
    Path(cfg.output.base_dir).mkdir(parents=True, exist_ok=True)

    if args.step:
        ok = run_step(cfg, args.step)
    else:
        ok = run_pipeline(cfg)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
