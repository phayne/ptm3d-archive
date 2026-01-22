#!/usr/bin/env python3
"""
Plot thermal model debug output showing temperature, conductivity, and heat capacity
profiles at different local times throughout a lunar day.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import re

def parse_debug_file(filename):
    """Parse the thermal debug output file."""
    with open(filename, 'r') as f:
        content = f.read()

    # Extract planet properties
    props = {}
    props['planet'] = re.search(r'Planet: (\w+)', content).group(1)
    props['latitude'] = float(re.search(r'Latitude: ([\d.-]+) deg', content).group(1))
    props['N_z'] = int(re.search(r'N_z \(depth layers\): (\d+)', content).group(1))
    props['day'] = float(re.search(r'day: ([\d.]+) s', content).group(1))

    # Parse timestep blocks
    timestep_pattern = r'=== TIMESTEP (\d+) / (\d+) \| Local Hour: ([\d.]+) ===\s*\n' \
                       r't = ([\d.]+) s.*?\n' \
                       r'Qs \(surface flux\) = ([\d.]+) W/m\^2'

    timesteps = []

    # Split by timestep headers
    blocks = re.split(r'(=== TIMESTEP \d+ / \d+ \| Local Hour: [\d.]+ ===)', content)

    for i in range(1, len(blocks), 2):
        header = blocks[i]
        if i+1 < len(blocks):
            data_block = blocks[i+1]
        else:
            continue

        # Parse header
        match = re.search(r'TIMESTEP (\d+) / (\d+) \| Local Hour: ([\d.]+)', header)
        if not match:
            continue

        step = int(match.group(1))
        total = int(match.group(2))
        local_hour = float(match.group(3))

        # Parse Qs
        qs_match = re.search(r'Qs \(surface flux\) = ([\d.]+)', data_block)
        if qs_match:
            Qs = float(qs_match.group(1))
        else:
            Qs = 0.0

        # Parse profile data
        # Look for lines with numeric data after the header line
        lines = data_block.strip().split('\n')

        depths = []
        temps = []
        k_vals = []
        kc_vals = []
        rho_vals = []
        cp_vals = []
        kappa_vals = []

        for line in lines:
            # Match data lines: idx depth T k kc rho cp kappa
            # Format: "   0   0.000000 386.659775 3.433881e-03 ..."
            parts = line.split()
            if len(parts) >= 8:
                try:
                    idx = int(parts[0])
                    depth = float(parts[1])
                    T = float(parts[2])
                    k = float(parts[3])
                    kc = float(parts[4])
                    rho = float(parts[5])
                    cp = float(parts[6])
                    kappa = float(parts[7])

                    depths.append(depth)
                    temps.append(T)
                    k_vals.append(k)
                    kc_vals.append(kc)
                    rho_vals.append(rho)
                    cp_vals.append(cp)
                    kappa_vals.append(kappa)
                except (ValueError, IndexError):
                    continue

        if len(depths) > 0:
            timesteps.append({
                'step': step,
                'local_hour': local_hour,
                'Qs': Qs,
                'depth': np.array(depths),
                'T': np.array(temps),
                'k': np.array(k_vals),
                'kc': np.array(kc_vals),
                'rho': np.array(rho_vals),
                'cp': np.array(cp_vals),
                'kappa': np.array(kappa_vals)
            })

    return props, timesteps


def plot_profiles(props, timesteps, output_prefix='/tmp/thermal_profile'):
    """Create plots of thermal profiles."""

    # Sort by local hour
    timesteps = sorted(timesteps, key=lambda x: x['local_hour'])

    # Create colormap based on local hour
    hours = [ts['local_hour'] for ts in timesteps]
    cmap = plt.cm.twilight_shifted
    norm = Normalize(vmin=0, vmax=24)

    # Figure 1: Temperature vs depth at different local times
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    for ts in timesteps:
        color = cmap(norm(ts['local_hour']))
        ax1.plot(ts['T'], ts['depth'] * 100, color=color, linewidth=1.5,
                label=f"Hour {ts['local_hour']:.1f}")

    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Depth (cm)', fontsize=12)
    ax1.set_title(f"Temperature Profiles - {props['planet']} (Lat: {props['latitude']:.1f}°)", fontsize=14)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 420)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Local Hour', fontsize=12)

    plt.tight_layout()
    fig1.savefig(f'{output_prefix}_temperature.png', dpi=150)
    print(f"Saved: {output_prefix}_temperature.png")

    # Figure 2: Heat capacity vs depth
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    for ts in timesteps:
        color = cmap(norm(ts['local_hour']))
        ax2.plot(ts['cp'], ts['depth'] * 100, color=color, linewidth=1.5)

    ax2.set_xlabel('Heat Capacity cp (J/kg/K)', fontsize=12)
    ax2.set_ylabel('Depth (cm)', fontsize=12)
    ax2.set_title(f"Heat Capacity Profiles - {props['planet']}", fontsize=14)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Local Hour', fontsize=12)

    plt.tight_layout()
    fig2.savefig(f'{output_prefix}_heat_capacity.png', dpi=150)
    print(f"Saved: {output_prefix}_heat_capacity.png")

    # Figure 3: Thermal conductivity vs depth
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    for ts in timesteps:
        color = cmap(norm(ts['local_hour']))
        ax3.plot(ts['k'] * 1000, ts['depth'] * 100, color=color, linewidth=1.5)

    ax3.set_xlabel('Thermal Conductivity k (mW/m/K)', fontsize=12)
    ax3.set_ylabel('Depth (cm)', fontsize=12)
    ax3.set_title(f"Thermal Conductivity Profiles - {props['planet']}", fontsize=14)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Local Hour', fontsize=12)

    plt.tight_layout()
    fig3.savefig(f'{output_prefix}_conductivity.png', dpi=150)
    print(f"Saved: {output_prefix}_conductivity.png")

    # Figure 4: Surface temperature and flux vs local hour
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    hours_sorted = [ts['local_hour'] for ts in timesteps]
    surf_T = [ts['T'][0] for ts in timesteps]
    Qs_vals = [ts['Qs'] for ts in timesteps]

    ax4a.plot(hours_sorted, surf_T, 'b-o', markersize=4, linewidth=2)
    ax4a.set_ylabel('Surface Temperature (K)', fontsize=12)
    ax4a.set_title(f"Diurnal Temperature Curve - {props['planet']} (Lat: {props['latitude']:.1f}°)", fontsize=14)
    ax4a.grid(True, alpha=0.3)
    ax4a.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100 K reference')
    ax4a.legend()

    ax4b.plot(hours_sorted, Qs_vals, 'r-o', markersize=4, linewidth=2)
    ax4b.set_xlabel('Local Hour', fontsize=12)
    ax4b.set_ylabel('Surface Flux Qs (W/m²)', fontsize=12)
    ax4b.grid(True, alpha=0.3)
    ax4b.set_xlim(0, 24)

    plt.tight_layout()
    fig4.savefig(f'{output_prefix}_diurnal.png', dpi=150)
    print(f"Saved: {output_prefix}_diurnal.png")

    # Figure 5: Combined 2x2 panel
    fig5, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Temperature profiles
    ax = axes[0, 0]
    for ts in timesteps:
        color = cmap(norm(ts['local_hour']))
        ax.plot(ts['T'], ts['depth'] * 100, color=color, linewidth=1.2)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Depth (cm)')
    ax.set_title('(a) Temperature Profiles')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 420)

    # Panel B: Heat capacity
    ax = axes[0, 1]
    for ts in timesteps:
        color = cmap(norm(ts['local_hour']))
        ax.plot(ts['cp'], ts['depth'] * 100, color=color, linewidth=1.2)
    ax.set_xlabel('Heat Capacity cp (J/kg/K)')
    ax.set_ylabel('Depth (cm)')
    ax.set_title('(b) Heat Capacity Profiles')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Panel C: Conductivity
    ax = axes[1, 0]
    for ts in timesteps:
        color = cmap(norm(ts['local_hour']))
        ax.plot(ts['k'] * 1000, ts['depth'] * 100, color=color, linewidth=1.2)
    ax.set_xlabel('Thermal Conductivity k (mW/m/K)')
    ax.set_ylabel('Depth (cm)')
    ax.set_title('(c) Thermal Conductivity Profiles')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Panel D: Diurnal curve
    ax = axes[1, 1]
    ax.plot(hours_sorted, surf_T, 'b-o', markersize=5, linewidth=2)
    ax.set_xlabel('Local Hour')
    ax.set_ylabel('Surface Temperature (K)')
    ax.set_title('(d) Diurnal Temperature Curve')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100 K (expected nightside min)')
    ax.legend()

    # Add colorbar to the figure
    fig5.subplots_adjust(right=0.9)
    cbar_ax = fig5.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig5.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Local Hour', fontsize=12)

    fig5.suptitle(f"{props['planet']} Thermal Model - Equatorial Face (Lat: {props['latitude']:.1f}°)",
                  fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig5.savefig(f'{output_prefix}_combined.png', dpi=150)
    print(f"Saved: {output_prefix}_combined.png")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Planet: {props['planet']}")
    print(f"Latitude: {props['latitude']:.2f}°")
    print(f"Number of depth layers: {props['N_z']}")
    print(f"Number of timesteps plotted: {len(timesteps)}")
    print(f"Surface temperature range: {min(surf_T):.1f} K to {max(surf_T):.1f} K")
    print(f"Mean surface temperature: {np.mean(surf_T):.1f} K")

    # Find midnight values
    midnight_ts = min(timesteps, key=lambda x: min(abs(x['local_hour']), abs(x['local_hour']-24)))
    print(f"\nMidnight (Local Hour {midnight_ts['local_hour']:.1f}):")
    print(f"  Surface T: {midnight_ts['T'][0]:.1f} K")
    print(f"  Surface cp: {midnight_ts['cp'][0]:.1f} J/kg/K")
    print(f"  Surface k: {midnight_ts['k'][0]*1000:.3f} mW/m/K")

    # Find noon values
    noon_ts = min(timesteps, key=lambda x: abs(x['local_hour']-12))
    print(f"\nNoon (Local Hour {noon_ts['local_hour']:.1f}):")
    print(f"  Surface T: {noon_ts['T'][0]:.1f} K")
    print(f"  Surface cp: {noon_ts['cp'][0]:.1f} J/kg/K")
    print(f"  Surface k: {noon_ts['k'][0]*1000:.3f} mW/m/K")

    # plt.show()  # Disabled for non-interactive mode
    plt.close('all')


if __name__ == '__main__':
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else '/tmp/thermal_debug.txt'
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else '/tmp/thermal_profile'

    print(f"Parsing {filename}...")
    props, timesteps = parse_debug_file(filename)

    print(f"Found {len(timesteps)} timesteps")

    if len(timesteps) > 0:
        plot_profiles(props, timesteps, output_prefix)
    else:
        print("No timestep data found!")
