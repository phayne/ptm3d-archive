#!/usr/bin/env python3
"""
Depth Profile Plotter for PTM3D

Generates visualization plots from PTM3D depth profile CSV output files.
Can produce:
  1. Temperature vs Time for each depth (time series)
  2. Temperature vs Depth at selected times (depth profiles)
  3. Combined 2D heatmap (depth x time)

Usage:
    python plot_depth_profiles.py <csv_file> [options]
    python plot_depth_profiles.py --directory <dir> [options]

Options:
    --output-dir DIR     Output directory for plots (default: same as input)
    --format FMT         Output format: png, pdf, svg (default: png)
    --dpi DPI            Resolution for raster formats (default: 150)
    --style STYLE        Plot style: paper, presentation, minimal (default: paper)
    --no-heatmap         Skip heatmap plot
    --no-timeseries      Skip time series plot
    --no-profiles        Skip depth profile snapshots
    --time-indices IDX   Comma-separated timestep indices for profile snapshots
    --max-depth DEPTH    Maximum depth to plot in meters (default: auto)
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
import re

import numpy as np

# Try to import matplotlib with Agg backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.dates as mdates


def parse_depth_csv(filepath):
    """
    Parse a PTM3D depth profile CSV file.

    Returns:
        dict with keys:
            'location': str - location name
            'times': list of datetime or float
            'depths': list of float (meters)
            'temperatures': 2D array [n_times x n_depths]
    """
    times = []
    depths = []
    temperatures = []
    location = "unnamed"

    with open(filepath, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            # Check for location comment
            if row[0].startswith('# Location:'):
                location = row[0].replace('# Location:', '').strip()
                continue

            # Skip other comments
            if row[0].startswith('#'):
                continue

            # Header row - extract depths
            if row[0].lower() == 'time':
                for col in row[1:]:
                    # Parse "T(z=0.050 m)" format
                    match = re.search(r'z=([0-9.]+)', col)
                    if match:
                        depths.append(float(match.group(1)))
                continue

            # Data row
            time_str = row[0]
            try:
                # Try ISO format first
                t = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                times.append(t)
            except ValueError:
                # Fall back to numeric (seconds or hours)
                try:
                    times.append(float(time_str))
                except ValueError:
                    continue

            # Temperature values
            temp_row = []
            for val in row[1:]:
                try:
                    temp_row.append(float(val))
                except ValueError:
                    temp_row.append(np.nan)
            temperatures.append(temp_row)

    return {
        'location': location,
        'times': times,
        'depths': np.array(depths),
        'temperatures': np.array(temperatures)
    }


def setup_style(style='paper'):
    """Configure matplotlib style for publication-quality plots."""
    if style == 'paper':
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.figsize': (8, 6),
            'figure.dpi': 100,
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
        })
    elif style == 'presentation':
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (12, 8),
            'figure.dpi': 100,
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
    elif style == 'minimal':
        plt.rcParams.update({
            'font.size': 10,
            'figure.figsize': (6, 4),
            'axes.grid': False,
        })


def plot_heatmap(data, output_path, max_depth=None, dpi=150):
    """
    Create a 2D heatmap of temperature vs depth and time.

    Args:
        data: dict from parse_depth_csv
        output_path: path for output image
        max_depth: maximum depth to show (None = auto)
        dpi: output resolution
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    temps = data['temperatures']
    depths = data['depths']
    times = data['times']
    n_times = len(times)

    # Filter by max_depth if specified
    if max_depth is not None:
        depth_mask = depths <= max_depth
        depths = depths[depth_mask]
        temps = temps[:, depth_mask]

    # Create time axis (hours from start if datetime, else raw values)
    if isinstance(times[0], datetime):
        t0 = times[0]
        time_hours = np.array([(t - t0).total_seconds() / 3600 for t in times])
        time_label = 'Time (hours from start)'
    else:
        time_hours = np.array(times)
        # Assume seconds, convert to hours if large
        if time_hours.max() > 1000:
            time_hours = time_hours / 3600
            time_label = 'Time (hours)'
        else:
            time_label = 'Time'

    # Create meshgrid for pcolormesh
    T_mesh, Z_mesh = np.meshgrid(time_hours, depths)

    # Plot heatmap (transpose temps so depth is Y axis)
    im = ax.pcolormesh(T_mesh, Z_mesh, temps.T, shading='auto', cmap='inferno')

    # Invert Y axis (depth increases downward)
    ax.invert_yaxis()

    # Labels
    ax.set_xlabel(time_label)
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Temperature Profile: {data["location"]}')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label='Temperature (K)')

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved heatmap: {output_path}")


def plot_timeseries(data, output_path, max_depth=None, dpi=150):
    """
    Create time series plot showing temperature vs time for each depth.

    Args:
        data: dict from parse_depth_csv
        output_path: path for output image
        max_depth: maximum depth to show (None = auto)
        dpi: output resolution
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    temps = data['temperatures']
    depths = data['depths']
    times = data['times']

    # Filter by max_depth if specified
    if max_depth is not None:
        depth_mask = depths <= max_depth
        depths = depths[depth_mask]
        temps = temps[:, depth_mask]

    # Create time axis
    if isinstance(times[0], datetime):
        time_axis = times
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.xticks(rotation=45, ha='right')
    else:
        time_axis = np.array(times)
        if time_axis.max() > 1000:
            time_axis = time_axis / 3600
            ax.set_xlabel('Time (hours)')
        else:
            ax.set_xlabel('Time')

    # Color map for depths
    cmap = plt.cm.viridis
    norm = Normalize(vmin=depths.min(), vmax=depths.max())

    # Plot each depth as a line
    for i, z in enumerate(depths):
        color = cmap(norm(z))
        label = f'{z:.3f} m' if i < 10 or i == len(depths)-1 else None
        ax.plot(time_axis, temps[:, i], color=color, linewidth=0.8, label=label)

    ax.set_ylabel('Temperature (K)')
    ax.set_title(f'Temperature Time Series: {data["location"]}')

    # Add colorbar for depth
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Depth (m)')

    # Legend for a few key depths
    if len(depths) <= 10:
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved time series: {output_path}")


def plot_depth_profiles(data, output_path, time_indices=None, max_depth=None, dpi=150):
    """
    Create depth profile snapshots at selected times.

    Args:
        data: dict from parse_depth_csv
        output_path: path for output image
        time_indices: list of timestep indices to plot (None = auto-select)
        max_depth: maximum depth to show (None = auto)
        dpi: output resolution
    """
    fig, ax = plt.subplots(figsize=(6, 8))

    temps = data['temperatures']
    depths = data['depths']
    times = data['times']
    n_times = len(times)

    # Filter by max_depth if specified
    if max_depth is not None:
        depth_mask = depths <= max_depth
        depths = depths[depth_mask]
        temps = temps[:, depth_mask]

    # Auto-select time indices if not provided
    if time_indices is None:
        # Select ~6 evenly spaced times plus first and last
        n_profiles = min(8, n_times)
        time_indices = np.linspace(0, n_times - 1, n_profiles, dtype=int)

    # Color map for times
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=0, vmax=len(time_indices)-1)

    # Plot each time snapshot
    for i, tidx in enumerate(time_indices):
        if tidx >= n_times:
            continue
        color = cmap(norm(i))

        # Create time label
        if isinstance(times[tidx], datetime):
            label = times[tidx].strftime('%Y-%m-%d %H:%M')
        else:
            t_val = times[tidx]
            if t_val > 1000:
                label = f't = {t_val/3600:.1f} hr'
            else:
                label = f't = {t_val:.0f}'

        ax.plot(temps[tidx, :], depths, color=color, linewidth=1.5, label=label)

    # Invert Y axis (depth increases downward)
    ax.invert_yaxis()

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Depth Profiles: {data["location"]}')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved depth profiles: {output_path}")


def plot_combined(data, output_path, max_depth=None, dpi=150):
    """
    Create a combined figure with heatmap, time series, and profiles.

    Args:
        data: dict from parse_depth_csv
        output_path: path for output image
        max_depth: maximum depth to show (None = auto)
        dpi: output resolution
    """
    fig = plt.figure(figsize=(14, 10))

    temps = data['temperatures']
    depths = data['depths']
    times = data['times']
    n_times = len(times)

    # Filter by max_depth if specified
    if max_depth is not None:
        depth_mask = depths <= max_depth
        depths_plot = depths[depth_mask]
        temps_plot = temps[:, depth_mask]
    else:
        depths_plot = depths
        temps_plot = temps

    # Create time axis
    if isinstance(times[0], datetime):
        t0 = times[0]
        time_hours = np.array([(t - t0).total_seconds() / 3600 for t in times])
        time_label = 'Time (hours)'
    else:
        time_hours = np.array(times)
        if time_hours.max() > 1000:
            time_hours = time_hours / 3600
        time_label = 'Time (hours)'

    # Layout: 2x2 grid
    # Top-left: Heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    T_mesh, Z_mesh = np.meshgrid(time_hours, depths_plot)
    im = ax1.pcolormesh(T_mesh, Z_mesh, temps_plot.T, shading='auto', cmap='inferno')
    ax1.invert_yaxis()
    ax1.set_xlabel(time_label)
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Temperature Heatmap')
    fig.colorbar(im, ax=ax1, label='T (K)')

    # Top-right: Depth profiles at selected times
    ax2 = fig.add_subplot(2, 2, 2)
    n_profiles = min(6, n_times)
    time_indices = np.linspace(0, n_times - 1, n_profiles, dtype=int)
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=0, vmax=len(time_indices)-1)

    for i, tidx in enumerate(time_indices):
        color = cmap(norm(i))
        label = f't = {time_hours[tidx]:.1f} hr'
        ax2.plot(temps_plot[tidx, :], depths_plot, color=color, linewidth=1.5, label=label)

    ax2.invert_yaxis()
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('Depth Profiles')
    ax2.legend(loc='lower right', fontsize=8)

    # Bottom-left: Surface temperature time series
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time_hours, temps_plot[:, 0], 'b-', linewidth=1.0, label='Surface')
    if len(depths_plot) > 1:
        # Also plot a deeper layer
        deep_idx = min(len(depths_plot) - 1, 5)
        ax3.plot(time_hours, temps_plot[:, deep_idx], 'r--', linewidth=1.0,
                 label=f'z = {depths_plot[deep_idx]:.3f} m')
    ax3.set_xlabel(time_label)
    ax3.set_ylabel('Temperature (K)')
    ax3.set_title('Temperature Time Series')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Temperature range vs depth
    ax4 = fig.add_subplot(2, 2, 4)
    T_min = temps_plot.min(axis=0)
    T_max = temps_plot.max(axis=0)
    T_mean = temps_plot.mean(axis=0)

    ax4.fill_betweenx(depths_plot, T_min, T_max, alpha=0.3, color='blue', label='Range')
    ax4.plot(T_mean, depths_plot, 'b-', linewidth=2, label='Mean')
    ax4.plot(T_min, depths_plot, 'b--', linewidth=0.8)
    ax4.plot(T_max, depths_plot, 'b--', linewidth=0.8)
    ax4.invert_yaxis()
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Depth (m)')
    ax4.set_title('Temperature Statistics vs Depth')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Thermal Profile Analysis: {data["location"]}', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved combined plot: {output_path}")


def process_file(csv_path, output_dir=None, fmt='png', dpi=150, style='paper',
                 no_heatmap=False, no_timeseries=False, no_profiles=False,
                 time_indices=None, max_depth=None):
    """
    Process a single CSV file and generate all requested plots.

    Args:
        csv_path: path to input CSV file
        output_dir: output directory (None = same as input)
        fmt: output format (png, pdf, svg)
        dpi: resolution
        style: plot style
        no_heatmap: skip heatmap
        no_timeseries: skip time series
        no_profiles: skip depth profiles
        time_indices: specific timesteps for profiles
        max_depth: maximum depth to plot
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return False

    print(f"Processing: {csv_path.name}")

    # Parse data
    try:
        data = parse_depth_csv(csv_path)
    except Exception as e:
        print(f"  Error parsing file: {e}")
        return False

    if len(data['temperatures']) == 0:
        print(f"  Warning: No data found in file")
        return False

    print(f"  Location: {data['location']}")
    print(f"  Time steps: {len(data['times'])}")
    print(f"  Depths: {len(data['depths'])} ({data['depths'].min():.4f} to {data['depths'].max():.4f} m)")

    # Setup output directory
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Base name for output files
    base_name = csv_path.stem.replace('_depths', '')

    # Setup style
    setup_style(style)

    # Generate plots
    if not no_heatmap:
        plot_heatmap(data, output_dir / f'{base_name}_heatmap.{fmt}', max_depth, dpi)

    if not no_timeseries:
        plot_timeseries(data, output_dir / f'{base_name}_timeseries.{fmt}', max_depth, dpi)

    if not no_profiles:
        plot_depth_profiles(data, output_dir / f'{base_name}_profiles.{fmt}',
                           time_indices, max_depth, dpi)

    # Always generate combined plot
    plot_combined(data, output_dir / f'{base_name}_combined.{fmt}', max_depth, dpi)

    return True


def process_directory(directory, **kwargs):
    """Process all *_depths.csv files in a directory."""
    directory = Path(directory)
    csv_files = list(directory.glob('*_depths.csv'))

    if not csv_files:
        print(f"No *_depths.csv files found in {directory}")
        return

    print(f"Found {len(csv_files)} depth profile files")

    for csv_file in sorted(csv_files):
        process_file(csv_file, **kwargs)
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate plots from PTM3D depth profile CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('input', nargs='?', help='Input CSV file or directory')
    parser.add_argument('--directory', '-d', help='Process all *_depths.csv in directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format (default: png)')
    parser.add_argument('--dpi', type=int, default=150, help='Resolution (default: 150)')
    parser.add_argument('--style', default='paper',
                       choices=['paper', 'presentation', 'minimal'],
                       help='Plot style (default: paper)')
    parser.add_argument('--no-heatmap', action='store_true', help='Skip heatmap plot')
    parser.add_argument('--no-timeseries', action='store_true', help='Skip time series plot')
    parser.add_argument('--no-profiles', action='store_true', help='Skip depth profile snapshots')
    parser.add_argument('--time-indices', help='Comma-separated timestep indices for profiles')
    parser.add_argument('--max-depth', type=float, help='Maximum depth to plot (meters)')

    args = parser.parse_args()

    # Parse time indices if provided
    time_indices = None
    if args.time_indices:
        time_indices = [int(x.strip()) for x in args.time_indices.split(',')]

    kwargs = {
        'output_dir': args.output_dir,
        'fmt': args.format,
        'dpi': args.dpi,
        'style': args.style,
        'no_heatmap': args.no_heatmap,
        'no_timeseries': args.no_timeseries,
        'no_profiles': args.no_profiles,
        'time_indices': time_indices,
        'max_depth': args.max_depth,
    }

    if args.directory:
        process_directory(args.directory, **kwargs)
    elif args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            process_directory(input_path, **kwargs)
        else:
            process_file(input_path, **kwargs)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
