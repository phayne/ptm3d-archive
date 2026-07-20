#!/usr/bin/env python3
"""
Generate PTM3D simulation configs for each ALMA observation epoch.

Creates YAML configuration files that run PTM3D at the exact epochs when
ALMA observed Europa, so we can compare predicted vs. observed Tb.

Observation epochs:
  Trumbo et al. (2018): 4 observations, Band 6 (233 GHz), Nov 2015
  Thelen et al. (2024): 8 observations, Bands 3/6/7, 2016-2017

Usage:
  python setup_ptm3d_runs.py                   # Generate all configs
  python setup_ptm3d_runs.py --project trumbo   # Trumbo only
  python setup_ptm3d_runs.py --list              # List observations
"""

import argparse
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "alma"
OUTPUT_BASE = PROJECT_ROOT / "output" / "alma"

# ─────────────────────────────────────────────────────────────────────
# Observation catalog
# ─────────────────────────────────────────────────────────────────────

# Trumbo et al. (2018) — AJ 156, 161
# Project 2015.1.01302.S, Band 6 (233 GHz, 1.3 mm)
# Geometry from Table 1; best-fit emissivity 0.75, thermal inertia 95 SI
# NOTE: Trumbo reports spatially-resolved maps, not disk-averaged Tb.
# No published disk-averaged Tb values are available from this paper.
TRUMBO_OBS = [
    {
        "label": "T1",
        "epoch": "2015-11-17T14:43:00",
        "sub_obs_lon": 45.4,    # degrees West (Table 1: 44.9-45.9)
        "sub_obs_lat": -1.44,
        "ang_diam": 0.77,       # arcsec
        "band": 6,
        "freq_ghz": 233.0,
        "beam_mas": (57, 42),   # natural resolution from Table 1
        "reference": "Trumbo et al. (2018)",
        "hemisphere": "anti-Jovian",
        "tb_published": None,   # Not reported as disk-average
        "tb_error": None,
    },
    {
        "label": "T2",
        "epoch": "2015-11-25T11:33:00",
        "sub_obs_lon": 122.1,   # Table 1: 121.5-122.6
        "sub_obs_lat": -1.52,
        "ang_diam": 0.77,
        "band": 6,
        "freq_ghz": 233.0,
        "beam_mas": (59, 53),
        "reference": "Trumbo et al. (2018)",
        "hemisphere": "leading",
        "tb_published": None,
        "tb_error": None,
    },
    {
        "label": "T3",
        "epoch": "2015-11-26T08:30:00",
        "sub_obs_lon": 210.6,   # Table 1: 210-211.1
        "sub_obs_lat": -1.52,
        "ang_diam": 0.77,
        "band": 6,
        "freq_ghz": 233.0,
        "beam_mas": (77, 52),
        "reference": "Trumbo et al. (2018)",
        "hemisphere": "sub-Jovian",
        "tb_published": None,
        "tb_error": None,
    },
    {
        "label": "T4",
        "epoch": "2015-11-27T10:20:00",
        "sub_obs_lon": 319.5,   # Table 1: 318.1-320.9
        "sub_obs_lat": -1.54,
        "ang_diam": 0.77,
        "band": 6,
        "freq_ghz": 233.0,
        "beam_mas": (54, 53),
        "reference": "Trumbo et al. (2018)",
        "hemisphere": "trailing",
        "tb_published": None,
        "tb_error": None,
    },
]

# Thelen et al. (2024) — PSJ 5, 42
# Project 2016.1.00691.S, Bands 3/6/7
# Geometry from Table 1; disk-averaged Tb from Table 2
# Table 2 reports 7 hemispheric-averaged values (some Band 6 epochs combined)
#
# Published disk-averaged brightness temperatures (Table 2):
#   Band 3 leading  (103.3°W): 72.99 ± 3.65 K
#   Band 3 trailing (283.2°W): 85.27 ± 4.26 K
#   Band 6 leading  (80.63°W): 76.61 ± 3.83 K
#   Band 6 trailing (270.05°W): 79.96 ± 4.00 K
#   Band 6 trailing (287.02°W): 85.74 ± 5.40 K
#   Band 7 leading  (100.3°W): 86.09 ± 4.30 K
#   Band 7 trailing (226.7°W): 89.09 ± 5.52 K
#
# Global averages by band (Table 3):
#   Band 3: 78.19 ± 2.77 K
#   Band 6: 79.78 ± 2.46 K
#   Band 7: 87.22 ± 3.39 K

# Per-observation entries (individual epochs for PTM3D simulation).
# tb_published assigned from Table 2 where the sub-observer longitude
# matches; some observations were averaged in Thelen's Table 2.
THELEN_OBS = [
    {
        "label": "H1",
        "epoch": "2016-10-25T00:00:00",  # Approximate; refine with SPICE
        "sub_obs_lon": 226.7,   # Table 2 reports 226.7°W for trailing Band 7
        "sub_obs_lat": -1.5,
        "ang_diam": 0.678,
        "band": 7,
        "freq_ghz": 343.5,
        "beam_mas": (196, 151),
        "reference": "Thelen et al. (2024)",
        "hemisphere": "trailing",
        "tb_published": 89.09,
        "tb_error": 5.52,
    },
    {
        "label": "H2",
        "epoch": "2017-07-06T00:00:00",
        "sub_obs_lon": 100.3,   # Table 2 reports 100.3°W for leading Band 7
        "sub_obs_lat": -1.5,
        "ang_diam": 0.803,
        "band": 7,
        "freq_ghz": 343.5,
        "beam_mas": (139, 68),
        "reference": "Thelen et al. (2024)",
        "hemisphere": "leading",
        "tb_published": 86.09,
        "tb_error": 4.30,
    },
    {
        "label": "H3",
        "epoch": "2017-07-09T00:00:00",
        "sub_obs_lon": 270.05,  # Table 2 trailing Band 6
        "sub_obs_lat": -1.5,
        "ang_diam": 0.797,
        "band": 6,
        "freq_ghz": 233.0,
        "beam_mas": (280, 107),
        "reference": "Thelen et al. (2024)",
        "hemisphere": "trailing",
        "tb_published": 79.96,
        "tb_error": 4.00,
    },
    {
        "label": "H4",
        "epoch": "2017-07-30T00:00:00",
        "sub_obs_lon": 287.02,  # Table 2 trailing Band 6 (second entry)
        "sub_obs_lat": -1.5,
        "ang_diam": 0.754,
        "band": 6,
        "freq_ghz": 233.0,
        "beam_mas": (163, 78),
        "reference": "Thelen et al. (2024)",
        "hemisphere": "trailing",
        "tb_published": 85.74,
        "tb_error": 5.40,
    },
    {
        "label": "H5",
        "epoch": "2017-08-07T00:00:00",
        "sub_obs_lon": 80.63,   # Table 2 leading Band 6
        "sub_obs_lat": -1.5,
        "ang_diam": 0.737,
        "band": 6,
        "freq_ghz": 233.0,
        "beam_mas": (107, 82),
        "reference": "Thelen et al. (2024)",
        "hemisphere": "leading",
        "tb_published": 76.61,
        "tb_error": 3.83,
    },
    {
        "label": "H6",
        "epoch": "2017-08-16T00:00:00",
        "sub_obs_lon": 270.0,
        "sub_obs_lat": -1.5,
        "ang_diam": 0.722,
        "band": 6,
        "freq_ghz": 233.0,
        "beam_mas": (127, 79),
        "reference": "Thelen et al. (2024)",
        "hemisphere": "trailing",
        # No separate Table 2 entry; averaged with H3 or H4
        "tb_published": None,
        "tb_error": None,
    },
    {
        "label": "H7",
        "epoch": "2017-09-19T00:00:00",
        "sub_obs_lon": 103.3,   # Table 2 leading Band 3
        "sub_obs_lat": -1.5,
        "ang_diam": 0.684,
        "band": 3,
        "freq_ghz": 97.5,
        "beam_mas": (110, 86),
        "reference": "Thelen et al. (2024)",
        "hemisphere": "leading",
        "tb_published": 72.99,
        "tb_error": 3.65,
    },
    {
        "label": "H8",
        "epoch": "2017-09-28T00:00:00",
        "sub_obs_lon": 283.2,   # Table 2 trailing Band 3
        "sub_obs_lat": -1.5,
        "ang_diam": 0.677,
        "band": 3,
        "freq_ghz": 97.5,
        "beam_mas": (148, 85),
        "reference": "Thelen et al. (2024)",
        "hemisphere": "trailing",
        "tb_published": 85.27,
        "tb_error": 4.26,
    },
]

# Thelen Table 3: Global averages by band
THELEN_GLOBAL_AVG = {
    3: {"tb": 78.19, "tb_error": 2.77, "wavelength_mm": 3.05},
    6: {"tb": 79.78, "tb_error": 2.46, "wavelength_mm": 1.25},
    7: {"tb": 87.22, "tb_error": 3.39, "wavelength_mm": 0.88},
}

# Trumbo best-fit model parameters (Section 4)
TRUMBO_BEST_FIT = {
    "emissivity": 0.75,
    "thermal_inertia_SI": 95,   # J m^-2 K^-1 s^-1/2
    "bolometric_emissivity": 0.9,
}


def get_all_observations(project=None):
    """Return observation list filtered by project."""
    if project == "trumbo":
        return TRUMBO_OBS
    elif project == "thelen":
        return THELEN_OBS
    else:
        return TRUMBO_OBS + THELEN_OBS


def get_published_tb():
    """Return list of observations that have published Tb values.

    Only includes entries where tb_published is not None.
    """
    return [obs for obs in TRUMBO_OBS + THELEN_OBS
            if obs.get("tb_published") is not None]


def generate_config(obs, mesh_resolution=6, equilibration_orbits=10,
                    timesteps=288, grid_b=10):
    """Generate a PTM3D YAML config for one ALMA observation epoch.

    Parameters
    ----------
    obs : dict
        Observation metadata (epoch, geometry, etc.).
    mesh_resolution : int
        Icosahedral mesh subdivision level (6 = ~40K faces).
    equilibration_orbits : int
        Number of thermal equilibration orbits.
    timesteps : int
        Time steps per orbit (288 = 2.5 min resolution for Europa).
    grid_b : int
        Number of thermal skin depths to bottom boundary.
        Must be large enough for ALMA penetration depth.
        Band 3 needs grid_b >= 10 (probes ~20 cm).

    Returns
    -------
    dict
        YAML-compatible config dictionary.
    """
    label = obs["label"]
    output_dir = str(OUTPUT_BASE / f"alma_{label}")

    config = {
        "simulation": {
            "start_time": obs["epoch"],
            "duration_days": 3.55,       # ~1 Europa orbit (3.55 days)
            "timesteps": timesteps,
            "body": "Europa",
        },
        "mesh": {
            "resolution": mesh_resolution,
            "use_topographic": False,
            "topo_directory": "",
            "import_file": "",
            "scale_factor": 1.0,
            "export_basic": False,
        },
        "thermal": {
            "run_thermal_mesh": True,
            "threads": 0,  # auto
            "equilibration_orbits": equilibration_orbits,
            "compute_statistics": True,
            "thermal_inertia_map": "",
            "enable_shadow_flux": False,
            "crater_dd_ratio": 8.0,
            "integration_days": 3.55,
            "integration_steps": -1,
            "integration_offset_days": 0.0,
            "integration_offset_steps": 0,
            # Extended depth grid for ALMA subsurface probing
            "grid_m": 4,
            "grid_n": 10,
            "grid_b": grid_b,
        },
        "radiation": {
            "enabled": False,
        },
        "illumination": {
            "include_primary_reflection": True,
            "use_batch_processing": True,
            "albedo_map": "",
            "albedo_maps_directory": str(PROJECT_ROOT / "data" / "albedo_maps"),
            "positions_file": "",
        },
        "output": {
            "directory": output_dir,
            "format": "binary",
            "vtk_mode": "thermal",
            "include_primary_ir": False,
            "thermal_vtk_file": "",
            "complete_vtk_file": "",
            "no_maps": False,
            "locations_file": str(CONFIG_DIR / f"alma_{label}_locations.csv"),
            "depths_file": str(CONFIG_DIR / "alma_depths.txt"),
            "series_output": "",
            "location_data": "",
        },
        "visualization": {
            "create": False,
        },
        "advanced": {
            "verbose": True,
            "default_visible_albedo": -1.0,
        },
    }

    # Add ALMA-specific metadata as comments (preserved in YAML)
    config["_alma_metadata"] = {
        "observation_label": label,
        "reference": obs["reference"],
        "sub_obs_lon_deg_W": obs["sub_obs_lon"],
        "sub_obs_lat_deg": obs["sub_obs_lat"],
        "ang_diam_arcsec": obs["ang_diam"],
        "alma_band": obs["band"],
        "freq_ghz": obs["freq_ghz"],
        "beam_mas_maj_min": list(obs["beam_mas"]),
        "hemisphere": obs["hemisphere"],
    }

    return config


def write_configs(observations, mesh_resolution=6, dry_run=False):
    """Write YAML config files for all observations.

    Parameters
    ----------
    observations : list
        List of observation dictionaries.
    mesh_resolution : int
        Mesh subdivision level.
    dry_run : bool
        If True, print configs without writing.

    Returns
    -------
    list
        Paths to created config files.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    created = []

    for obs in observations:
        label = obs["label"]
        config = generate_config(obs, mesh_resolution=mesh_resolution)
        config_path = CONFIG_DIR / f"alma_{label}.yaml"

        if dry_run:
            print(f"\n  {'='*60}")
            print(f"  Config: {config_path.name}")
            print(f"  {'='*60}")
            print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        else:
            with open(config_path, "w") as f:
                f.write(f"# PTM3D config for ALMA observation {label}\n")
                f.write(f"# {obs['reference']}\n")
                f.write(f"# Epoch: {obs['epoch']}\n")
                f.write(f"# Band {obs['band']} ({obs['freq_ghz']:.1f} GHz), "
                        f"{obs['hemisphere']} hemisphere\n")
                f.write(f"# Sub-observer: lon={obs['sub_obs_lon']:.1f}°W, "
                        f"lat={obs['sub_obs_lat']:.1f}°\n\n")
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print(f"  Created: {config_path}")
            created.append(config_path)

    return created


def write_batch_script(observations):
    """Write a shell script to run all PTM3D simulations.

    Parameters
    ----------
    observations : list
        List of observation dictionaries.

    Returns
    -------
    Path
        Path to the batch script.
    """
    script_path = CONFIG_DIR / "run_all_alma.sh"
    ptm_bin = PROJECT_ROOT / "bin" / "ptm_thermal_sim"

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Run all PTM3D simulations for ALMA comparison\n")
        f.write(f"# Generated by setup_ptm3d_runs.py\n\n")
        f.write(f"PTM_BIN=\"{ptm_bin}\"\n")
        f.write(f"CONFIG_DIR=\"{CONFIG_DIR}\"\n\n")
        f.write("# Check that PTM3D binary exists\n")
        f.write('if [ ! -f "$PTM_BIN" ]; then\n')
        f.write('    echo "ERROR: PTM3D binary not found at $PTM_BIN"\n')
        f.write('    echo "Build with: cd build && cmake --build . --target ptm_thermal_sim -j4"\n')
        f.write('    exit 1\n')
        f.write('fi\n\n')

        f.write(f"DEPTHS_FILE=\"$CONFIG_DIR/alma_depths.txt\"\n\n")

        for obs in observations:
            label = obs["label"]
            f.write(f"echo \"\\n{'='*60}\"\n")
            f.write(f"echo \"Running {label}: {obs['reference']}, "
                    f"Band {obs['band']}, {obs['hemisphere']}\"\n")
            f.write(f"echo \"{'='*60}\"\n")
            f.write(f"$PTM_BIN --config $CONFIG_DIR/alma_{label}.yaml "
                    f"--locations $CONFIG_DIR/alma_{label}_locations.csv "
                    f"--depths $DEPTHS_FILE\n\n")

    script_path.chmod(0o755)
    print(f"\n  Batch script: {script_path}")
    return script_path


def list_observations(observations):
    """Print a summary table of all observations."""
    print(f"\n  {'Label':>6} {'Date':>12} {'Band':>6} {'Freq':>8} "
          f"{'Sub-obs lon':>12} {'Hemisphere':>12} {'Tb (K)':>12} {'Reference':>24}")
    print(f"  {'─'*105}")

    for obs in observations:
        date = obs["epoch"][:10]
        tb = obs.get("tb_published")
        tb_err = obs.get("tb_error")
        if tb is not None:
            tb_str = f"{tb:.1f}±{tb_err:.1f}"
        else:
            tb_str = "—"
        print(f"  {obs['label']:>6} {date:>12} {obs['band']:>6} "
              f"{obs['freq_ghz']:>7.1f}G {obs['sub_obs_lon']:>11.1f}° "
              f"{obs['hemisphere']:>12} {tb_str:>12} "
              f"{obs['reference']:>24}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PTM3D configs for ALMA observation epochs"
    )
    parser.add_argument("--project", choices=["trumbo", "thelen", "both"],
                        default="both",
                        help="Which project to generate (default: both)")
    parser.add_argument("--resolution", type=int, default=6,
                        help="Mesh resolution level (default: 6 = ~40K faces)")
    parser.add_argument("--list", action="store_true",
                        help="List all observations without generating configs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without writing files")
    parser.add_argument("--batch", action="store_true",
                        help="Also generate batch run script")
    args = parser.parse_args()

    observations = get_all_observations(
        args.project if args.project != "both" else None
    )

    if args.list:
        list_observations(observations)
        return

    print(f"\n  Generating PTM3D configs for {len(observations)} ALMA observations")
    print(f"  Mesh resolution: {args.resolution}")
    print(f"  Config directory: {CONFIG_DIR}")

    configs = write_configs(observations, args.resolution, args.dry_run)

    if args.batch and not args.dry_run:
        write_batch_script(observations)

    if not args.dry_run:
        print(f"\n  Created {len(configs)} config file(s)")
        print(f"\n  To run all simulations:")
        print(f"    bash {CONFIG_DIR / 'run_all_alma.sh'}")
        print(f"  Or individually:")
        print(f"    ./bin/ptm_thermal_sim --config {CONFIG_DIR}/alma_T1.yaml")


if __name__ == "__main__":
    main()
