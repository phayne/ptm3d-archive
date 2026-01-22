#!/usr/bin/env python3
"""
Plot thermal debug output from PTM3D.

Reads the thermal_debug.txt file generated when running with --thermal-debug
and creates diurnal thermal wave profile plots.

Usage:
    python plot_thermal_debug.py <thermal_debug.txt> [--output-dir DIR]
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path


def parse_thermal_debug(filepath):
    """
    Parse PTM3D thermal_debug.txt file.

    Returns:
        dict with:
            'planet': str
            'latitude': float
            'depths': array of depths in meters
            'timesteps': list of dicts with 't', 'local_hour', 'flux', 'profile'
            'planet_props': dict of planet properties
    """
    data = {
        'planet': '',
        'latitude': 0.0,
        'depths': [],
        'timesteps': [],
        'planet_props': {},
        'initial_profile': None
    }

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse header info
        if line.startswith('Planet:'):
            data['planet'] = line.split(':')[1].strip()
        elif line.startswith('Latitude:'):
            data['latitude'] = float(line.split(':')[1].replace('deg', '').strip())
        elif line.startswith('N_z'):
            data['n_layers'] = int(line.split(':')[1].strip())
        elif line.startswith('dt:'):
            data['dt'] = float(line.split(':')[1].replace('s', '').strip())

        # Parse planet properties
        elif line.startswith('day:'):
            data['planet_props']['day'] = float(line.split(':')[1].replace('s', '').strip())
        elif line.startswith('S (solar constant):'):
            data['planet_props']['S'] = float(line.split(':')[1].replace('W/m^2', '').strip())
        elif line.startswith('albedo:'):
            data['planet_props']['albedo'] = float(line.split(':')[1].strip())
        elif line.startswith('emissivity:'):
            data['planet_props']['emissivity'] = float(line.split(':')[1].strip())

        # Parse grid depths
        elif line.startswith('=== GRID DEPTHS'):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('==='):
                depth_line = lines[i].strip()
                if depth_line.startswith('z['):
                    # Parse "z[0] = 0 m" or "z[1] = 0.008882871507 m, dz = ..."
                    match = re.search(r'z\[\d+\]\s*=\s*([\d.e+-]+)\s*m', depth_line)
                    if match:
                        data['depths'].append(float(match.group(1)))
                elif depth_line.startswith('Total depth:'):
                    pass  # Skip
                i += 1
            continue

        # Parse initial profile
        elif line.startswith('=== INITIAL PROFILE'):
            i += 1
            profile = {'T': [], 'k': [], 'rho': [], 'cp': []}
            while i < len(lines) and not lines[i].strip().startswith('==='):
                prof_line = lines[i].strip()
                if prof_line.startswith('z['):
                    # Parse "z[0]: depth=0 m, T=273.0446884 K, k=..."
                    match = re.search(r'T=([\d.e+-]+)\s*K', prof_line)
                    if match:
                        profile['T'].append(float(match.group(1)))
                i += 1
            data['initial_profile'] = profile
            continue

        # Parse timestep data
        elif line.startswith('=== TIMESTEP'):
            # Parse "=== TIMESTEP 165 / 3960 | Local Hour: 16.325328 ==="
            match = re.search(r'TIMESTEP\s+(\d+)\s+/\s+\d+\s+\|\s+Local Hour:\s+([\d.]+)', line)
            if match:
                timestep_num = int(match.group(1))
                local_hour = float(match.group(2))

                # Read next lines for time and flux
                i += 1
                t_line = lines[i].strip()
                t_match = re.search(r't\s*=\s*([\d.e+-]+)\s*s', t_line)
                t = float(t_match.group(1)) if t_match else 0.0

                i += 1
                flux_line = lines[i].strip()
                flux_match = re.search(r'Qs.*=\s*([\d.e+-]+)', flux_line)
                flux = float(flux_match.group(1)) if flux_match else 0.0

                # Skip header lines
                i += 3  # Skip blank line and table header

                # Read profile data
                profile_T = []
                profile_depth = []
                while i < len(lines):
                    prof_line = lines[i].strip()
                    if not prof_line or prof_line.startswith('==='):
                        break
                    # Parse table row: "   0   0.000000 365.370208 3.012955e-03 ..."
                    parts = prof_line.split()
                    if len(parts) >= 3 and parts[0].isdigit():
                        try:
                            depth = float(parts[1])
                            T = float(parts[2])
                            profile_depth.append(depth)
                            profile_T.append(T)
                        except ValueError:
                            pass
                    i += 1

                data['timesteps'].append({
                    'timestep': timestep_num,
                    'local_hour': local_hour,
                    't': t,
                    'flux': flux,
                    'depths': np.array(profile_depth),
                    'T': np.array(profile_T)
                })
                continue

        i += 1

    data['depths'] = np.array(data['depths'])
    return data


def plot_diurnal_profiles(data, output_path, dpi=150):
    """
    Create the diurnal thermal wave profile plot.

    Shows temperature vs depth at different local times throughout
    the lunar day, illustrating the thermal wave propagation.
    """
    fig, ax = plt.subplots(figsize=(8, 10))

    # Get timesteps and sort by local hour
    timesteps = sorted(data['timesteps'], key=lambda x: x['local_hour'])

    # Select a subset of representative times (every ~3 hours)
    n_profiles = min(8, len(timesteps))
    indices = np.linspace(0, len(timesteps)-1, n_profiles, dtype=int)
    selected = [timesteps[i] for i in indices]

    # Color map based on local hour
    cmap = plt.cm.twilight
    norm = Normalize(vmin=0, vmax=24)

    for ts in selected:
        if len(ts['depths']) == 0 or len(ts['T']) == 0:
            continue
        color = cmap(norm(ts['local_hour']))
        label = f"{ts['local_hour']:.1f}h (Q={ts['flux']:.0f} W/m²)"
        ax.plot(ts['T'], ts['depths'] * 100, color=color, linewidth=2, label=label)  # Convert m to cm

    ax.invert_yaxis()
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Depth (cm)', fontsize=12)
    ax.set_title(f'Diurnal Thermal Wave Profiles\n{data["planet"]}, Latitude {data["latitude"]}°', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add colorbar for local time
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Local Solar Time (hours)', pad=0.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_surface_temperature(data, output_path, dpi=150):
    """
    Plot surface temperature vs local time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract surface temperatures
    timesteps = sorted(data['timesteps'], key=lambda x: x['local_hour'])
    local_hours = [ts['local_hour'] for ts in timesteps]
    surface_temps = [ts['T'][0] if len(ts['T']) > 0 else np.nan for ts in timesteps]
    fluxes = [ts['flux'] for ts in timesteps]

    # Plot temperature
    ax.plot(local_hours, surface_temps, 'b-', linewidth=2, label='Surface Temperature')
    ax.set_xlabel('Local Solar Time (hours)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')

    # Add flux on secondary axis
    ax2 = ax.twinx()
    ax2.plot(local_hours, fluxes, 'r--', linewidth=1.5, alpha=0.7, label='Solar Flux')
    ax2.set_ylabel('Solar Flux (W/m²)', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax.set_title(f'Surface Temperature Diurnal Cycle\n{data["planet"]}, Latitude {data["latitude"]}°', fontsize=14)
    ax.set_xlim(0, 24)
    ax.axvline(x=12, color='gray', linestyle=':', alpha=0.5, label='Noon')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=24, color='gray', linestyle=':', alpha=0.5)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_thermal_wave_heatmap(data, output_path, dpi=150):
    """
    Create a 2D heatmap showing temperature vs depth and local time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build 2D temperature array
    timesteps = sorted(data['timesteps'], key=lambda x: x['local_hour'])

    # Use depths from first timestep
    if not timesteps or len(timesteps[0]['depths']) == 0:
        print("No timestep data available for heatmap")
        return

    depths = timesteps[0]['depths']
    local_hours = np.array([ts['local_hour'] for ts in timesteps])

    # Build temperature matrix [n_times x n_depths]
    T_matrix = np.zeros((len(timesteps), len(depths)))
    for i, ts in enumerate(timesteps):
        if len(ts['T']) == len(depths):
            T_matrix[i, :] = ts['T']
        else:
            # Handle mismatched sizes
            n = min(len(ts['T']), len(depths))
            T_matrix[i, :n] = ts['T'][:n]

    # Create meshgrid
    H_mesh, Z_mesh = np.meshgrid(local_hours, depths * 100)  # Convert to cm

    # Plot heatmap
    im = ax.pcolormesh(H_mesh, Z_mesh, T_matrix.T, shading='auto', cmap='inferno')
    ax.invert_yaxis()

    ax.set_xlabel('Local Solar Time (hours)', fontsize=12)
    ax.set_ylabel('Depth (cm)', fontsize=12)
    ax.set_title(f'Subsurface Temperature\n{data["planet"]}, Latitude {data["latitude"]}°', fontsize=14)

    cbar = fig.colorbar(im, ax=ax, label='Temperature (K)')

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined_figure(data, output_path, dpi=150):
    """
    Create a combined 4-panel figure showing all thermal analysis.
    """
    fig = plt.figure(figsize=(14, 12))

    timesteps = sorted(data['timesteps'], key=lambda x: x['local_hour'])
    if not timesteps:
        print("No timestep data available")
        return

    depths = timesteps[0]['depths']
    local_hours = np.array([ts['local_hour'] for ts in timesteps])
    surface_temps = np.array([ts['T'][0] if len(ts['T']) > 0 else np.nan for ts in timesteps])
    fluxes = np.array([ts['flux'] for ts in timesteps])

    # Build temperature matrix
    T_matrix = np.zeros((len(timesteps), len(depths)))
    for i, ts in enumerate(timesteps):
        n = min(len(ts['T']), len(depths))
        T_matrix[i, :n] = ts['T'][:n]

    # Panel 1: Diurnal profiles
    ax1 = fig.add_subplot(2, 2, 1)
    n_profiles = 8
    indices = np.linspace(0, len(timesteps)-1, n_profiles, dtype=int)
    cmap = plt.cm.twilight
    norm = Normalize(vmin=0, vmax=24)

    for idx in indices:
        ts = timesteps[idx]
        if len(ts['T']) == 0:
            continue
        color = cmap(norm(ts['local_hour']))
        label = f"{ts['local_hour']:.1f}h"
        ax1.plot(ts['T'], depths * 100, color=color, linewidth=1.5, label=label)

    ax1.invert_yaxis()
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Depth (cm)')
    ax1.set_title('(a) Temperature Profiles at Different Local Times')
    ax1.legend(loc='lower right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    H_mesh, Z_mesh = np.meshgrid(local_hours, depths * 100)
    im = ax2.pcolormesh(H_mesh, Z_mesh, T_matrix.T, shading='auto', cmap='inferno')
    ax2.invert_yaxis()
    ax2.set_xlabel('Local Solar Time (hours)')
    ax2.set_ylabel('Depth (cm)')
    ax2.set_title('(b) Subsurface Temperature Heatmap')
    fig.colorbar(im, ax=ax2, label='T (K)')

    # Panel 3: Surface temperature and flux
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(local_hours, surface_temps, 'b-', linewidth=2, label='Surface T')
    ax3.set_xlabel('Local Solar Time (hours)')
    ax3.set_ylabel('Temperature (K)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_xlim(0, 24)

    ax3b = ax3.twinx()
    ax3b.plot(local_hours, fluxes, 'r--', linewidth=1.5, alpha=0.7, label='Flux')
    ax3b.set_ylabel('Solar Flux (W/m²)', color='r')
    ax3b.tick_params(axis='y', labelcolor='r')

    ax3.set_title('(c) Surface Temperature and Solar Flux')
    ax3.axvline(x=12, color='gray', linestyle=':', alpha=0.5)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Temperature amplitude vs depth
    ax4 = fig.add_subplot(2, 2, 4)
    T_min = T_matrix.min(axis=0)
    T_max = T_matrix.max(axis=0)
    T_mean = T_matrix.mean(axis=0)
    T_amp = (T_max - T_min) / 2

    ax4.plot(T_amp, depths * 100, 'b-', linewidth=2, label='Amplitude')
    ax4.invert_yaxis()
    ax4.set_xlabel('Temperature Amplitude (K)')
    ax4.set_ylabel('Depth (cm)')
    ax4.set_title('(d) Diurnal Temperature Amplitude vs Depth')
    ax4.grid(True, alpha=0.3)

    # Add skin depth annotation
    # Skin depth ~ sqrt(kappa * P / pi)
    if T_amp[0] > 0:
        # Find depth where amplitude drops to 1/e of surface
        e_folding = T_amp[0] / np.e
        skin_idx = np.argmin(np.abs(T_amp - e_folding))
        if skin_idx > 0:
            skin_depth = depths[skin_idx] * 100
            ax4.axhline(y=skin_depth, color='r', linestyle='--', alpha=0.7)
            ax4.annotate(f'Skin depth ≈ {skin_depth:.1f} cm',
                        xy=(T_amp[skin_idx], skin_depth),
                        xytext=(T_amp[0]*0.5, skin_depth*1.3),
                        fontsize=9, color='r',
                        arrowprops=dict(arrowstyle='->', color='r', alpha=0.7))

    fig.suptitle(f'Lunar Thermal Model Validation: {data["planet"]}, Latitude {data["latitude"]}°',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot PTM3D thermal debug output')
    parser.add_argument('input', help='Path to thermal_debug.txt file')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=150, help='Plot resolution')

    args = parser.parse_args()

    # Parse input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    print(f"Parsing: {input_path}")
    data = parse_thermal_debug(input_path)

    print(f"  Planet: {data['planet']}")
    print(f"  Latitude: {data['latitude']}°")
    print(f"  Depth layers: {len(data['depths'])}")
    print(f"  Timesteps with profiles: {len(data['timesteps'])}")

    if len(data['timesteps']) == 0:
        print("Error: No timestep data found in file")
        return 1

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    base_name = input_path.stem

    plot_diurnal_profiles(data, output_dir / f'{base_name}_profiles.png', args.dpi)
    plot_surface_temperature(data, output_dir / f'{base_name}_surface.png', args.dpi)
    plot_thermal_wave_heatmap(data, output_dir / f'{base_name}_heatmap.png', args.dpi)
    plot_combined_figure(data, output_dir / f'{base_name}_combined.png', args.dpi)

    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
