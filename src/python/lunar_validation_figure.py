#!/usr/bin/env python3
"""
Lunar Validation Figure Generator

Compares PTM3D simulation output with Hayne et al. (2017) JGR Planets
"Global Regolith Thermophysical Properties of the Moon From the
 Diviner Lunar Radiometer Experiment"

Creates a 3-panel validation figure:
1. Surface temperature vs local time (diurnal cycle)
2. Temperature vs depth at different local times
3. Model parameters comparison table

Author: Paul Hayne, University of Colorado Boulder
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import sys

# Figure style settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'


def read_binary_data(filepath):
    """Read PTM3D binary output file (int32 count header + float32 data)."""
    with open(filepath, 'rb') as f:
        # Read 4-byte header (count)
        count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        # Read float32 data
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data


def read_mesh_ply(filepath):
    """Read mesh PLY file to get face latitudes (handles binary format)."""
    vertices = []
    faces = []
    face_lats = []

    with open(filepath, 'rb') as f:
        # Read header (ASCII)
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header
        is_binary = False
        vertex_count = 0
        face_count = 0
        vertex_props = []
        face_props = []

        for line in header_lines:
            if 'binary' in line:
                is_binary = True
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
            elif line.startswith('property') and vertex_count > 0 and face_count == 0:
                vertex_props.append(line.split()[1:])
            elif line.startswith('property') and face_count > 0:
                face_props.append(line.split()[1:])

        if is_binary:
            # Binary PLY reading
            # Vertex format: x y z nx ny nz (6 doubles = 48 bytes per vertex)
            vertex_bytes = vertex_count * 6 * 8  # 6 doubles
            vertex_data = np.frombuffer(f.read(vertex_bytes), dtype='<f8')
            vertex_data = vertex_data.reshape(vertex_count, 6)
            vertices = vertex_data[:, :3]

            # Face format: uchar count, int*3 indices, float lat, float lon
            # = 1 + 12 + 8 = 21 bytes per face
            for i in range(face_count):
                count = struct.unpack('<B', f.read(1))[0]  # uchar
                indices = struct.unpack('<' + 'i' * count, f.read(4 * count))
                lat, lon = struct.unpack('<ff', f.read(8))  # 2 floats
                faces.append(indices)
                face_lats.append(lat)
        else:
            # ASCII PLY reading
            data = f.read().decode('ascii').strip().split('\n')
            idx = 0
            for _ in range(vertex_count):
                parts = data[idx].split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                idx += 1
            vertices = np.array(vertices)

            for _ in range(face_count):
                parts = data[idx].split()
                n = int(parts[0])
                indices = [int(parts[j+1]) for j in range(n)]
                faces.append(indices)
                if len(parts) > n + 1:
                    face_lats.append(float(parts[n+1]))
                idx += 1

    return np.array(vertices), faces, np.array(face_lats)


def load_ptm3d_results(output_dir):
    """Load PTM3D simulation results from binary output."""
    output_dir = Path(output_dir)

    # Read metadata
    with open(output_dir / 'metadata.txt', 'r') as f:
        metadata = {}
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                metadata[key.strip()] = val.strip()

    n_faces = int(metadata.get('faces', 5120))

    # Read time steps
    with open(output_dir / 'time_steps.txt', 'r') as f:
        times = []
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    times.append(float(parts[0]))  # ET seconds
                except ValueError:
                    continue

    n_times = len(times)
    times = np.array(times)
    print(f"Loaded {n_times} time steps")

    # Read temperature data for all timesteps
    temp_dir = output_dir / 'temperature'
    temperatures = np.zeros((n_times, n_faces))

    print(f"Loading {n_times} temperature files...")
    for t in range(n_times):
        # Try both naming conventions
        temp_file = temp_dir / f'temperature_{t:04d}.bin'
        if not temp_file.exists():
            temp_file = temp_dir / f'{t:06d}.bin'
        if temp_file.exists():
            temperatures[t, :] = read_binary_data(temp_file)
        if (t + 1) % 50 == 0:
            print(f"  Loaded {t + 1}/{n_times} timesteps")
    print(f"  Temperature data loaded.")

    # Read mesh to get latitudes
    mesh_file = output_dir / 'mesh.ply'
    vertices, faces, face_lats = read_mesh_ply(mesh_file)

    # If face latitudes not in PLY, compute from vertex positions
    if len(face_lats) == 0:
        face_lats = np.zeros(n_faces)
        vertices = np.array(vertices)
        for i, face in enumerate(faces):
            center = vertices[face].mean(axis=0)
            r = np.linalg.norm(center)
            face_lats[i] = np.degrees(np.arcsin(center[2] / r))

    # Compute face longitudes
    face_lons = compute_face_longitudes(vertices, faces)

    return {
        'times': times,
        'temperatures': temperatures,
        'latitudes': face_lats,
        'longitudes': face_lons,
        'n_faces': n_faces,
        'n_times': n_times
    }


def get_equatorial_face_indices(latitudes, max_lat=5.0):
    """Get indices of faces near the equator."""
    return np.where(np.abs(latitudes) < max_lat)[0]


def get_latitude_band_indices(latitudes, center_lat, half_width=10.0):
    """Get indices of faces within a latitude band centered at center_lat."""
    return np.where(np.abs(latitudes - center_lat) < half_width)[0]


def compute_face_longitudes(vertices, faces):
    """Compute longitude of each face center."""
    longitudes = np.zeros(len(faces))
    vertices = np.array(vertices)
    for i, face in enumerate(faces):
        face_vertices = vertices[list(face)]  # Convert tuple/list to indices
        center = face_vertices.mean(axis=0)
        longitudes[i] = np.degrees(np.arctan2(center[1], center[0]))
    return longitudes


def compute_local_solar_time_from_temps(times, temps, lunar_day_seconds=29.530588 * 24 * 3600):
    """
    Compute local solar time empirically by finding when temperature peaks (noon).

    For a single face, noon occurs when temperature is maximum.
    We find the peak and use that as the reference for local time.

    Parameters:
    -----------
    times : array
        ET times in seconds
    temps : array
        Temperature values for this face at each time
    lunar_day_seconds : float
        Length of lunar day in seconds

    Returns:
    --------
    local_hours : array
        Local solar time in hours (0-24) for each timestep
    """
    # Find when temperature peaks - this is approximately local noon
    # (Actually slightly after noon due to thermal lag, but close enough)
    peak_idx = np.argmax(temps)
    peak_time = times[peak_idx]

    # Time elapsed since peak (noon)
    time_since_noon = times - peak_time

    # Convert to local hours (noon = 12)
    local_hours = 12.0 + (time_since_noon / lunar_day_seconds) * 24.0

    # Wrap to 0-24
    local_hours = local_hours % 24.0

    return local_hours


def create_validation_figure(output_dir, figure_path=None):
    """
    Create validation figure comparing PTM3D to Hayne et al. (2017).

    Parameters:
    -----------
    output_dir : str
        Path to PTM3D output directory
    figure_path : str, optional
        Path to save figure (default: docs/figures/lunar_validation.png)
    """

    # Load PTM3D results
    print("Loading PTM3D simulation results...")
    data = load_ptm3d_results(output_dir)

    # Get faces for different latitude bands
    eq_indices = get_equatorial_face_indices(data['latitudes'], max_lat=10.0)
    lat30_indices = get_latitude_band_indices(data['latitudes'], 30.0, half_width=10.0)
    lat60_indices = get_latitude_band_indices(data['latitudes'], 60.0, half_width=10.0)
    print(f"Found {len(eq_indices)} equatorial faces (|lat| < 10°)")
    print(f"Found {len(lat30_indices)} faces near lat=30°")
    print(f"Found {len(lat60_indices)} faces near lat=60°")

    # Lunar day parameters
    lunar_day = 29.530588 * 24 * 3600  # seconds

    # Helper function to compute binned temperatures for a set of face indices
    def compute_binned_temps(face_indices, times, temperatures):
        """Compute binned mean/min/max temperatures for given faces."""
        all_local_hours = []
        all_temps = []

        for face_idx in face_indices:
            face_temps = temperatures[:, face_idx]
            local_hours_face = compute_local_solar_time_from_temps(
                times, face_temps, lunar_day
            )
            all_local_hours.extend(local_hours_face)
            all_temps.extend(face_temps)

        all_local_hours = np.array(all_local_hours)
        all_temps = np.array(all_temps)

        # Bin by local hour (0.5 hour bins)
        hour_bins = np.arange(0, 24.5, 0.5)
        hour_centers = (hour_bins[:-1] + hour_bins[1:]) / 2
        binned_mean = np.zeros(len(hour_centers))
        binned_min = np.zeros(len(hour_centers))
        binned_max = np.zeros(len(hour_centers))

        for i in range(len(hour_centers)):
            mask = (all_local_hours >= hour_bins[i]) & (all_local_hours < hour_bins[i+1])
            if mask.sum() > 0:
                binned_mean[i] = all_temps[mask].mean()
                binned_min[i] = all_temps[mask].min()
                binned_max[i] = all_temps[mask].max()

        return hour_centers, binned_mean, binned_min, binned_max

    # Build diurnal curves for all latitude bands
    print("Computing local solar times for all latitude bands...")

    # Equatorial band
    local_hours_sorted, mean_temps_sorted, min_temps, max_temps = compute_binned_temps(
        eq_indices, data['times'], data['temperatures']
    )

    # Lat=30° band
    hour_centers_30, mean_temps_30, _, _ = compute_binned_temps(
        lat30_indices, data['times'], data['temperatures']
    )

    # Lat=60° band
    hour_centers_60, mean_temps_60, _, _ = compute_binned_temps(
        lat60_indices, data['times'], data['temperatures']
    )

    # Also track a single representative face for comparison
    # Pick the face closest to 0° longitude
    eq_lons = data['longitudes'][eq_indices]
    single_face_idx = eq_indices[np.argmin(np.abs(eq_lons))]
    single_face_temps = data['temperatures'][:, single_face_idx]
    single_face_local_hours = compute_local_solar_time_from_temps(
        data['times'], single_face_temps, lunar_day
    )

    # Hayne et al. (2017) reference values (from paper)
    # Peak noontime temperature: ~385 K
    # Midnight temperature: ~101 K
    # Minimum nighttime temperature: ~95 K
    hayne_noon_temp = 385.0
    hayne_midnight_temp = 101.0
    hayne_minimum_temp = 95.0

    # Create figure with 3 panels
    fig = plt.figure(figsize=(12, 10))

    # ===== Panel 1: Diurnal Temperature Curve =====
    ax1 = fig.add_subplot(2, 2, 1)

    # Plot PTM3D results - binned data
    ax1.plot(local_hours_sorted, mean_temps_sorted, 'b-', linewidth=2,
             label='PTM3D (equatorial mean)')
    ax1.fill_between(local_hours_sorted, min_temps, max_temps,
                     alpha=0.2, color='blue', label='Range across faces')

    # Also plot single face trajectory
    sort_single = np.argsort(single_face_local_hours)
    ax1.plot(single_face_local_hours[sort_single], single_face_temps[sort_single],
             'g-', linewidth=1, alpha=0.7, label='Single face (lon=0°)')

    # Add Hayne et al. reference points
    ax1.axhline(y=hayne_noon_temp, color='r', linestyle='--', alpha=0.7,
                label=f'Hayne et al. noon: {hayne_noon_temp} K')
    ax1.axhline(y=hayne_midnight_temp, color='g', linestyle='--', alpha=0.7,
                label=f'Hayne et al. midnight: {hayne_midnight_temp} K')
    ax1.axhline(y=hayne_minimum_temp, color='orange', linestyle='--', alpha=0.7,
                label=f'Hayne et al. minimum: {hayne_minimum_temp} K')

    ax1.set_xlabel('Local Solar Time (hours)')
    ax1.set_ylabel('Surface Temperature (K)')
    ax1.set_title('(a) Lunar Equatorial Surface Temperature Cycle')
    ax1.set_xlim(0, 24)
    ax1.set_ylim(50, 420)
    # Place legend in lower left where nighttime temperatures are low
    ax1.legend(loc='lower left', fontsize=8, ncol=1)
    ax1.grid(True, alpha=0.3)

    # Add vertical lines for noon and midnight
    ax1.axvline(x=12, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=24, color='gray', linestyle=':', alpha=0.5)

    # ===== Panel 2: Temperature vs Local Time (detailed nighttime) =====
    ax2 = fig.add_subplot(2, 2, 2)

    # Focus on nighttime (18:00 to 06:00)
    # Helper to extract nighttime data
    def get_nighttime_data(hour_centers, temps):
        night_mask_pm = hour_centers >= 18
        night_mask_am = hour_centers <= 6
        hours_night = hour_centers.copy()
        hours_night[night_mask_am] += 24
        night_mask = night_mask_pm | night_mask_am
        night_hours = hours_night[night_mask]
        night_temps = temps[night_mask]
        night_sort = np.argsort(night_hours)
        return night_hours[night_sort], night_temps[night_sort]

    # Get nighttime data for all three latitude bands
    night_hours_eq, night_temps_eq = get_nighttime_data(local_hours_sorted, mean_temps_sorted)
    night_hours_30, night_temps_30 = get_nighttime_data(hour_centers_30, mean_temps_30)
    night_hours_60, night_temps_60 = get_nighttime_data(hour_centers_60, mean_temps_60)

    # Plot PTM3D data for all three latitudes
    ax2.plot(night_hours_eq, night_temps_eq, 'b-', linewidth=2, label='PTM3D (lat=0°)')
    ax2.plot(night_hours_30, night_temps_30, 'c-', linewidth=2, label='PTM3D (lat=30°)')
    ax2.plot(night_hours_60, night_temps_60, 'purple', linewidth=2, label='PTM3D (lat=60°)')

    # Hayne et al. (2017) Figure A2 data points for nighttime cooling
    # Local times: 20:00, 22:00, 00:00, 02:00, 04:00
    hayne_local_times = np.array([20, 22, 24, 26, 28])  # Shifted for continuous plot
    hayne_temps_lat0 = np.array([109, 104, 100, 97, 94.5])   # Latitude 0°
    hayne_temps_lat30 = np.array([106.5, 101, 97, 94, 91])   # Latitude 30°
    hayne_temps_lat60 = np.array([95, 90, 87, 84, 82])       # Latitude 60°

    # Plot Hayne et al. data for each latitude (markers only)
    ax2.plot(hayne_local_times, hayne_temps_lat0, 'bo', markersize=8,
             label='Hayne (lat=0°)')
    ax2.plot(hayne_local_times, hayne_temps_lat30, 'cs', markersize=8,
             label='Hayne (lat=30°)')
    ax2.plot(hayne_local_times, hayne_temps_lat60, 'mp', markersize=8,
             label='Hayne (lat=60°)')

    ax2.set_xlabel('Local Solar Time (hours)')
    ax2.set_ylabel('Surface Temperature (K)')
    ax2.set_title('(b) Nighttime Temperature Cooling')
    ax2.set_xlim(18, 30)
    ax2.set_ylim(70, 130)
    ax2.set_xticks([18, 20, 22, 24, 26, 28, 30])
    ax2.set_xticklabels(['18:00', '20:00', '22:00', '00:00', '02:00', '04:00', '06:00'])
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ===== Panel 3: Statistics Comparison =====
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')

    # Calculate PTM3D statistics from binned data
    ptm_max_temp = max_temps.max()
    ptm_min_temp = min_temps.min()
    ptm_mean_temp = mean_temps_sorted.mean()

    # Find specific local times from binned data
    noon_idx = np.argmin(np.abs(local_hours_sorted - 12))
    midnight_idx = np.argmin(np.abs(local_hours_sorted))
    ptm_noon_temp = mean_temps_sorted[noon_idx]
    ptm_midnight_temp = mean_temps_sorted[midnight_idx]


    # Hayne et al. (2017) Table A2 - equatorial average temperature
    hayne_diurnal_mean = 206.0  # K, from Table A2 for latitude 0°

    # Create comparison table - use shorter column names to fit
    table_data = [
        ['Parameter', 'PTM3D', 'Hayne (2017)', 'Diff'],
        ['Peak noon temp', f'{ptm_noon_temp:.1f} K', f'{hayne_noon_temp:.1f} K',
         f'{ptm_noon_temp - hayne_noon_temp:+.1f} K'],
        ['Midnight temp', f'{ptm_midnight_temp:.1f} K', f'{hayne_midnight_temp:.1f} K',
         f'{ptm_midnight_temp - hayne_midnight_temp:+.1f} K'],
        ['Minimum temp', f'{ptm_min_temp:.1f} K', f'{hayne_minimum_temp:.1f} K',
         f'{ptm_min_temp - hayne_minimum_temp:+.1f} K'],
        ['Diurnal mean', f'{ptm_mean_temp:.1f} K', f'{hayne_diurnal_mean:.1f} K',
         f'{ptm_mean_temp - hayne_diurnal_mean:+.1f} K'],
    ]

    # Draw table with adjusted column widths
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center',
                      colWidths=[0.28, 0.18, 0.22, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.8)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax3.set_title('(c) Model Validation Summary\n', fontsize=13, fontweight='bold')

    # ===== Panel 4: Model Parameters =====
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    params_text = """
    Hayne et al. (2017) Lunar Regolith Model Parameters:

    Surface Properties:
      • Bond albedo (A₀): 0.12
      • Emissivity (ε): 0.95
      • Solar constant: 1361 W/m²

    Thermal Properties:
      • Surface density (ρs): 1100 kg/m³
      • Deep density (ρd): 1800 kg/m³
      • Surface conductivity (Ks): 7.4×10⁻⁴ W/(m·K)
      • Deep conductivity (Kd): 3.4×10⁻³ W/(m·K)
      • H-parameter: 0.06 m
      • Radiative parameter (χ): 2.7
      • Geothermal flux (Q): 0.018 W/m²

    Heat Capacity: cp(T) = c₀ + c₁T + c₂T² + c₃T³ + c₄T⁴
      (Polynomial fit from Hemingway et al. 1981)

    Reference: Hayne, P.O. et al. (2017) JGR Planets, 122, 2371-2400
    """

    ax4.text(0.05, 0.95, params_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('(d) Model Parameters\n', fontsize=13, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if figure_path is None:
        figure_path = Path(__file__).parent.parent.parent / 'docs' / 'figures' / 'lunar_validation.png'

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"\nValidation figure saved to: {figure_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"PTM3D peak noon temperature:  {ptm_noon_temp:.1f} K (Hayne: {hayne_noon_temp} K)")
    print(f"PTM3D midnight temperature:   {ptm_midnight_temp:.1f} K (Hayne: {hayne_midnight_temp} K)")
    print(f"PTM3D minimum temperature:    {ptm_min_temp:.1f} K (Hayne: {hayne_minimum_temp} K)")
    print(f"PTM3D diurnal mean:           {ptm_mean_temp:.1f} K")
    print("="*60)

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate lunar validation figure')
    parser.add_argument('--output-dir', '-o', default='output/lunar_validation',
                        help='PTM3D output directory')
    parser.add_argument('--figure', '-f', default=None,
                        help='Output figure path')

    args = parser.parse_args()

    fig = create_validation_figure(args.output_dir, args.figure)
    plt.show()
