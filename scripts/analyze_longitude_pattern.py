#!/usr/bin/env python3
"""
Analyze longitude-dependent patterns in PTM thermal simulation output.

Determines whether observed sublimation patterns are real physics (eclipse,
Jupiter IR) or mesh artifacts (icosahedral pentagon vertices).

Usage:
    python3 scripts/analyze_longitude_pattern.py <output_dir> [--save]
    python3 scripts/analyze_longitude_pattern.py <output_dir> --exosphere-dir <exo_dir> [--save]
"""

import numpy as np
import struct
import sys
import os
from pathlib import Path


def _frame_file(directory, name, t):
    """Frame ``t`` of ``name`` under either naming (BUGS.md PTM-664).

    Current engines pad the index to >= 6 digits; runs written before used 4.
    """
    for w in range(6, 13):
        p = os.path.join(str(directory), "%s_%0*d.bin" % (name, w, t))
        if os.path.exists(p):
            return p
    return os.path.join(str(directory), "%s_%04d.bin" % (name, t))


# Pentagon vertex unit vectors from createIcosphereMesh()
_t = (1.0 + np.sqrt(5.0)) / 2.0
_norm = np.sqrt(1.0 + _t**2)
PENTAGON_VERTICES = np.array([
    [-1.0/_norm,  _t/_norm,  0],         # v0:  lon=121.7, lat=0
    [ 1.0/_norm,  _t/_norm,  0],         # v1:  lon=58.3,  lat=0
    [-1.0/_norm, -_t/_norm,  0],         # v2:  lon=238.3, lat=0
    [ 1.0/_norm, -_t/_norm,  0],         # v3:  lon=301.7, lat=0
    [ 0,         -1.0/_norm, _t/_norm],  # v4:  lon=270,   lat=58.3
    [ 0,          1.0/_norm, _t/_norm],  # v5:  lon=90,    lat=58.3
    [ 0,         -1.0/_norm,-_t/_norm],  # v6:  lon=270,   lat=-58.3
    [ 0,          1.0/_norm,-_t/_norm],  # v7:  lon=90,    lat=-58.3
    [ _t/_norm,   0,        -1.0/_norm], # v8:  lon=0,     lat=-31.7
    [ _t/_norm,   0,         1.0/_norm], # v9:  lon=0,     lat=31.7
    [-_t/_norm,   0,        -1.0/_norm], # v10: lon=180,   lat=-31.7
    [-_t/_norm,   0,         1.0/_norm], # v11: lon=180,   lat=31.7
])


def load_mesh_ply(ply_path):
    """Load mesh from PLY and compute face centers with lat/lon.

    Handles PLY files with double-precision vertices (x,y,z,nx,ny,nz)
    and faces with extra properties (latitude, longitude).
    """
    with open(ply_path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        n_vertices = n_faces = 0
        is_binary = False
        vertex_props = []
        face_props = []
        current_element = None

        for line in header_lines:
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
                current_element = 'vertex'
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
                current_element = 'face'
            elif 'binary' in line:
                is_binary = True
            elif line.startswith('property') and current_element:
                parts = line.split()
                if current_element == 'vertex':
                    vertex_props.append(parts)
                elif current_element == 'face':
                    face_props.append(parts)

        # Compute vertex byte size
        type_sizes = {'float': 4, 'double': 8, 'int': 4, 'uint': 4, 'uchar': 1}
        vertex_byte_size = 0
        for prop in vertex_props:
            # property <type> <name>
            vertex_byte_size += type_sizes.get(prop[1], 4)

        # Compute face extra properties (after the vertex_indices list)
        face_extra_bytes = 0
        face_extra_types = []
        for prop in face_props:
            if prop[1] == 'list':
                continue  # vertex_indices list handled separately
            face_extra_bytes += type_sizes.get(prop[1], 4)
            face_extra_types.append(prop[1])

        print(f"  Vertices: {n_vertices} ({vertex_byte_size} bytes each), "
              f"Faces: {n_faces} (+ {face_extra_bytes} extra bytes)")

        vertices = []
        faces = []
        face_lats = []
        face_lons = []

        if is_binary:
            # Read vertices
            for _ in range(n_vertices):
                vdata = f.read(vertex_byte_size)
                # First 3 properties are x, y, z
                if vertex_props[0][1] == 'double':
                    x, y, z = struct.unpack('<ddd', vdata[:24])
                else:
                    x, y, z = struct.unpack('<fff', vdata[:12])
                vertices.append((x, y, z))

            # Read faces
            for _ in range(n_faces):
                count = struct.unpack('<B', f.read(1))[0]
                indices = struct.unpack(f'<{count}i', f.read(4 * count))
                faces.append(indices)
                # Read extra face properties (lat, lon)
                if face_extra_bytes > 0:
                    extra = f.read(face_extra_bytes)
                    if len(face_extra_types) >= 2:
                        flat, flon = struct.unpack('<ff', extra[:8])
                        face_lats.append(flat)
                        face_lons.append(flon)
        else:
            for _ in range(n_vertices):
                parts = f.readline().decode('ascii').split()
                vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
            for _ in range(n_faces):
                parts = f.readline().decode('ascii').split()
                count = int(parts[0])
                indices = tuple(int(parts[i+1]) for i in range(count))
                faces.append(indices)

    vertices = np.array(vertices)

    # Use embedded lat/lon if available, otherwise compute from face centers
    if face_lats and len(face_lats) == n_faces:
        lat = np.array(face_lats)
        lon = np.array(face_lons)
        # Still compute face centers for pentagon proximity
        face_centers = np.array([np.mean(vertices[list(f)], axis=0) for f in faces])
        print(f"  Using embedded lat/lon from PLY")
    else:
        face_centers = np.array([np.mean(vertices[list(f)], axis=0) for f in faces])
        r = np.linalg.norm(face_centers, axis=1)
        lat = np.degrees(np.arcsin(face_centers[:, 2] / r))
        lon = np.degrees(np.arctan2(face_centers[:, 1], face_centers[:, 0]))
        print(f"  Computing lat/lon from face centers")

    return face_centers, lat, lon, n_faces


def load_statistics(stats_dir, component):
    """Load per-face statistics (float32, first uint32 is face count)."""
    path = Path(stats_dir) / f'{component}.bin'
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.float32)
    # First value is face count stored as uint32 reinterpreted as float32
    return arr[1:]  # Skip header


def load_timestep(data_dir, component, timestep):
    """Load per-face data for a single timestep."""
    path = Path(_frame_file(Path(data_dir) / component, component, timestep))
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.float32)
    return arr[1:]  # Skip header (face count)


def compute_pentagon_proximity(face_centers):
    """Compute angular distance from each face to nearest pentagon vertex."""
    # Normalize face centers to unit vectors
    r = np.linalg.norm(face_centers, axis=1, keepdims=True)
    unit_centers = face_centers / r

    # Compute min angular distance to any pentagon vertex
    min_dist = np.full(len(face_centers), np.pi)
    nearest_vertex = np.zeros(len(face_centers), dtype=int)

    for i, v in enumerate(PENTAGON_VERTICES):
        cos_dist = np.clip(unit_centers @ v, -1, 1)
        dist = np.arccos(cos_dist)
        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        nearest_vertex[closer] = i

    return min_dist, nearest_vertex


def marti_mauersberger_flux(T):
    """Compute sublimation flux from temperature using Marti & Mauersberger (1993).

    Returns flux in kg/m^2/s.
    """
    T = np.asarray(T, dtype=np.float64)
    # Vapor pressure (Pa)
    log10_P = -2663.5 / T + 12.537
    P = 10.0**log10_P
    # Hertz-Knudsen flux: Phi = P / sqrt(2 * pi * m * k_B * T)
    m_H2O = 18.015e-3 / 6.022e23  # kg per molecule
    k_B = 1.381e-23  # J/K
    flux = P / np.sqrt(2 * np.pi * m_H2O * k_B * T)
    return flux


def load_exosphere_epoch(exo_dir, field, epoch):
    """Load per-face exosphere data for a single epoch.

    Binary format: [uint32 count][float32 data[count]]
    Files named: <field>_NNNN.bin where NNNN is zero-padded epoch.
    """
    path = Path(exo_dir) / field / f'{field}_{epoch:04d}.bin'
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.float32)
    # First value is face count stored as uint32 (reinterpreted as float32)
    return data[1:]


def find_exosphere_epochs(exo_dir, field='emission_flux_kg_m2_s'):
    """Find available epoch numbers for an exosphere field."""
    field_dir = Path(exo_dir) / field
    if not field_dir.exists():
        return []
    epochs = []
    for f in sorted(field_dir.glob(f'{field}_*.bin')):
        try:
            epoch = int(f.stem.split('_')[-1])
            epochs.append(epoch)
        except ValueError:
            pass
    return epochs


def bin_by_longitude(lon, values, lat, lat_band=(-30, 30), bin_width=2.0):
    """Bin values by longitude within a latitude band."""
    mask = (lat >= lat_band[0]) & (lat <= lat_band[1])
    lon_sel = lon[mask]
    val_sel = values[mask]

    lon_bins = np.arange(-180, 180 + bin_width, bin_width)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

    binned_mean = np.full(len(lon_centers), np.nan)
    binned_std = np.full(len(lon_centers), np.nan)
    binned_count = np.zeros(len(lon_centers), dtype=int)

    for i in range(len(lon_bins) - 1):
        bin_mask = (lon_sel >= lon_bins[i]) & (lon_sel < lon_bins[i+1])
        n = np.sum(bin_mask)
        if n > 0:
            binned_mean[i] = np.mean(val_sel[bin_mask])
            binned_std[i] = np.std(val_sel[bin_mask])
            binned_count[i] = n

    return lon_centers, binned_mean, binned_std, binned_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/analyze_longitude_pattern.py <output_dir> [--exosphere-dir <exo_dir>] [--save]")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    save_plots = '--save' in sys.argv

    # Parse --exosphere-dir
    exo_dir = None
    if '--exosphere-dir' in sys.argv:
        idx = sys.argv.index('--exosphere-dir')
        if idx + 1 < len(sys.argv):
            exo_dir = Path(sys.argv[idx + 1])

    print(f"\n{'='*72}")
    print("LONGITUDE PATTERN ANALYSIS")
    print(f"{'='*72}")
    print(f"Data directory: {output_dir}")

    # Load mesh
    mesh_path = output_dir / 'mesh.ply'
    print(f"\nLoading mesh from {mesh_path}...")
    face_centers, lat, lon, n_faces = load_mesh_ply(mesh_path)
    print(f"  {n_faces} faces, lat range [{lat.min():.1f}, {lat.max():.1f}], lon range [{lon.min():.1f}, {lon.max():.1f}]")

    # Pentagon proximity
    print(f"\nComputing pentagon vertex proximity...")
    pent_dist, pent_nearest = compute_pentagon_proximity(face_centers)
    pent_dist_deg = np.degrees(pent_dist)
    pentagon_threshold = 5.0  # degrees
    n_pent = np.sum(pent_dist_deg < pentagon_threshold)
    print(f"  {n_pent} faces within {pentagon_threshold}° of a pentagon vertex ({100*n_pent/n_faces:.1f}%)")

    # Load statistics
    stats_dir = output_dir / 'statistics'
    print(f"\nLoading statistics from {stats_dir}...")

    T_avg = load_statistics(stats_dir, 'temperature_avg')
    T_max = load_statistics(stats_dir, 'temperature_max')
    T_min = load_statistics(stats_dir, 'temperature_min')
    solar_avg = load_statistics(stats_dir, 'direct_solar_avg')
    solar_max = load_statistics(stats_dir, 'direct_solar_max')
    ir_avg = load_statistics(stats_dir, 'primary_ir_avg')
    ir_max = load_statistics(stats_dir, 'primary_ir_max')
    refl_avg = load_statistics(stats_dir, 'primary_reflected_avg')
    illum_avg = load_statistics(stats_dir, 'illumination_avg')

    # Truncate to n_faces if needed
    for name, arr in [('T_avg', T_avg), ('T_max', T_max), ('solar_avg', solar_avg),
                      ('ir_avg', ir_avg), ('refl_avg', refl_avg), ('illum_avg', illum_avg)]:
        if arr is not None:
            actual = min(len(arr), n_faces)
            print(f"  {name}: {actual} values, range [{arr[:actual].min():.4f}, {arr[:actual].max():.4f}]")

    # Trim all arrays to n_faces
    def trim(arr):
        return arr[:n_faces] if arr is not None and len(arr) >= n_faces else arr
    T_avg = trim(T_avg)
    T_max = trim(T_max)
    T_min = trim(T_min)
    solar_avg = trim(solar_avg)
    solar_max = trim(solar_max)
    ir_avg = trim(ir_avg)
    ir_max = trim(ir_max)
    refl_avg = trim(refl_avg)
    illum_avg = trim(illum_avg)

    # Compute sublimation from peak temperature
    print(f"\nComputing sublimation from T_max...")
    if T_max is not None:
        # Avoid very low temperatures where formula overflows
        T_max_safe = np.clip(T_max, 80, 200)
        sublimation = marti_mauersberger_flux(T_max_safe)
        log_sublimation = np.log10(np.maximum(sublimation, 1e-30))
        print(f"  Sublimation range: [{sublimation.min():.2e}, {sublimation.max():.2e}] kg/m^2/s")
    else:
        sublimation = None
        log_sublimation = None

    # Analysis for multiple latitude bands
    lat_bands = [(-90, 90), (-30, 30), (-60, -30), (30, 60)]
    band_labels = ['Global', 'Equatorial (±30°)', 'Southern mid-lat', 'Northern mid-lat']

    print(f"\n{'='*72}")
    print("LONGITUDE PROFILES")
    print(f"{'='*72}")

    # Main analysis: equatorial band
    for band, label in zip(lat_bands[:2], band_labels[:2]):
        print(f"\n--- {label} ---")

        if solar_avg is not None:
            lc, sm, ss, sc = bin_by_longitude(lon, solar_avg, lat, lat_band=band)
            valid = ~np.isnan(sm)
            if np.any(valid):
                # Find notable longitudes
                for target_lon in [0, 90, 180, -90]:
                    idx = np.argmin(np.abs(lc - target_lon))
                    if not np.isnan(sm[idx]):
                        zonal_mean = np.nanmean(sm)
                        dev = sm[idx] - zonal_mean
                        display_lon = target_lon if target_lon >= 0 else target_lon + 360
                        print(f"  Direct solar avg at lon={display_lon:3d}°: {sm[idx]:.4f} W/m² (deviation: {dev:+.4f})")
                print(f"  Direct solar zonal mean: {np.nanmean(sm):.4f} W/m²")
                print(f"  Direct solar zonal std: {np.nanstd(sm):.4f} W/m²")

        if T_avg is not None:
            lc, tm, ts, tc = bin_by_longitude(lon, T_avg, lat, lat_band=band)
            valid = ~np.isnan(tm)
            if np.any(valid):
                for target_lon in [0, 90, 180, -90]:
                    idx = np.argmin(np.abs(lc - target_lon))
                    if not np.isnan(tm[idx]):
                        zonal_mean = np.nanmean(tm)
                        dev = tm[idx] - zonal_mean
                        display_lon = target_lon if target_lon >= 0 else target_lon + 360
                        print(f"  T_avg at lon={display_lon:3d}°: {tm[idx]:.4f} K (deviation: {dev:+.4f} K)")
                print(f"  T_avg zonal mean: {np.nanmean(tm):.4f} K, std: {np.nanstd(tm):.4f} K")

        if T_max is not None:
            lc, tmx, ts, tc = bin_by_longitude(lon, T_max, lat, lat_band=band)
            valid = ~np.isnan(tmx)
            if np.any(valid):
                for target_lon in [0, 90, 180, -90]:
                    idx = np.argmin(np.abs(lc - target_lon))
                    if not np.isnan(tmx[idx]):
                        zonal_mean = np.nanmean(tmx)
                        dev = tmx[idx] - zonal_mean
                        display_lon = target_lon if target_lon >= 0 else target_lon + 360
                        print(f"  T_max at lon={display_lon:3d}°: {tmx[idx]:.4f} K (deviation: {dev:+.4f} K)")
                print(f"  T_max zonal mean: {np.nanmean(tmx):.4f} K, std: {np.nanstd(tmx):.4f} K")

        if ir_avg is not None and np.any(ir_avg > 0):
            lc, im, _, _ = bin_by_longitude(lon, ir_avg, lat, lat_band=band)
            print(f"  Jupiter IR avg max: {np.nanmax(im):.6f} W/m²")
        else:
            print(f"  Jupiter IR: ALL ZERO (not computed in this run)")

        if refl_avg is not None and np.any(refl_avg > 0):
            lc, rm, _, _ = bin_by_longitude(lon, refl_avg, lat, lat_band=band)
            print(f"  Jupiter reflected avg max: {np.nanmax(rm):.6f} W/m²")
        else:
            print(f"  Jupiter reflected: ALL ZERO (not computed in this run)")

        if sublimation is not None:
            lc, slm, sls, slc = bin_by_longitude(lon, log_sublimation, lat, lat_band=band)
            valid = ~np.isnan(slm)
            if np.any(valid):
                for target_lon in [0, 90, 180, -90]:
                    idx = np.argmin(np.abs(lc - target_lon))
                    if not np.isnan(slm[idx]):
                        zonal_mean = np.nanmean(slm)
                        dev = slm[idx] - zonal_mean
                        display_lon = target_lon if target_lon >= 0 else target_lon + 360
                        print(f"  log10(sublimation) at lon={display_lon:3d}°: {slm[idx]:.4f} (deviation: {dev:+.4f})")
                print(f"  log10(sublimation) zonal mean: {np.nanmean(slm):.4f}, std: {np.nanstd(slm):.4f}")

    # Pentagon vertex analysis
    print(f"\n{'='*72}")
    print("PENTAGON VERTEX ANALYSIS")
    print(f"{'='*72}")

    # Compare pentagon-adjacent vs non-pentagon faces
    is_pent = pent_dist_deg < pentagon_threshold
    is_regular = ~is_pent
    equatorial = (np.abs(lat) < 30)

    if T_avg is not None:
        for label, mask in [('All faces', np.ones(n_faces, dtype=bool)),
                            ('Equatorial', equatorial)]:
            pent_T = T_avg[mask & is_pent]
            reg_T = T_avg[mask & is_regular]
            if len(pent_T) > 0 and len(reg_T) > 0:
                print(f"\n  {label} T_avg: pentagon={pent_T.mean():.4f}±{pent_T.std():.4f} K, "
                      f"regular={reg_T.mean():.4f}±{reg_T.std():.4f} K, "
                      f"diff={pent_T.mean()-reg_T.mean():.4f} K")

    if T_max is not None:
        for label, mask in [('All faces', np.ones(n_faces, dtype=bool)),
                            ('Equatorial', equatorial)]:
            pent_T = T_max[mask & is_pent]
            reg_T = T_max[mask & is_regular]
            if len(pent_T) > 0 and len(reg_T) > 0:
                print(f"  {label} T_max: pentagon={pent_T.mean():.4f}±{pent_T.std():.4f} K, "
                      f"regular={reg_T.mean():.4f}±{reg_T.std():.4f} K, "
                      f"diff={pent_T.mean()-reg_T.mean():.4f} K")

    if solar_avg is not None:
        for label, mask in [('All faces', np.ones(n_faces, dtype=bool)),
                            ('Equatorial', equatorial)]:
            pent_S = solar_avg[mask & is_pent]
            reg_S = solar_avg[mask & is_regular]
            if len(pent_S) > 0 and len(reg_S) > 0:
                print(f"  {label} solar_avg: pentagon={pent_S.mean():.4f}±{pent_S.std():.4f} W/m², "
                      f"regular={reg_S.mean():.4f}±{reg_S.std():.4f} W/m², "
                      f"diff={pent_S.mean()-reg_S.mean():.4f} W/m²")

    # Per-vertex analysis
    print(f"\n  Per-vertex statistics (T_avg, equatorial faces within {pentagon_threshold}° of each vertex):")
    vertex_lons = np.degrees(np.arctan2(PENTAGON_VERTICES[:, 1], PENTAGON_VERTICES[:, 0]))
    vertex_lats = np.degrees(np.arcsin(PENTAGON_VERTICES[:, 2]))

    for i in range(12):
        vmask = (pent_nearest == i) & (pent_dist_deg < pentagon_threshold)
        if T_avg is not None and np.sum(vmask) > 0:
            vlon = vertex_lons[i]
            if vlon < 0: vlon += 360
            print(f"    v{i:2d} (lon={vlon:6.1f}°, lat={vertex_lats[i]:+6.1f}°): "
                  f"{np.sum(vmask):4d} faces, T_avg={T_avg[vmask].mean():.3f}±{T_avg[vmask].std():.3f} K")

    # Eclipse detection via timestep analysis
    print(f"\n{'='*72}")
    print("ECLIPSE DETECTION (timestep analysis)")
    print(f"{'='*72}")

    # Determine number of timesteps from metadata or directory listing
    metadata_path = output_dir / 'metadata.txt'
    n_timesteps = 0
    if metadata_path.exists():
        with open(metadata_path) as mf:
            for line in mf:
                if 'time_steps=' in line:
                    n_timesteps = int(line.strip().split('=')[1])
                    break
    if n_timesteps == 0:
        # Count timestep files
        ts_dir = output_dir / 'direct_solar'
        if ts_dir.exists():
            n_timesteps = len(list(ts_dir.glob('direct_solar_*.bin')))

    # Load individual timesteps to detect eclipse
    sub_jovian_mask = (np.abs(lat) < 15) & (np.abs(lon) < 15)  # lon ~ 0°
    anti_jovian_mask = (np.abs(lat) < 15) & ((np.abs(lon - 180) < 15) | (np.abs(lon + 180) < 15))
    leading_mask = (np.abs(lat) < 15) & (np.abs(lon - 90) < 15)
    trailing_mask = (np.abs(lat) < 15) & (np.abs(lon + 90) < 15)

    eclipse_count = np.zeros(n_faces, dtype=int)
    solar_at_subjovian = []
    solar_at_antijovian = []

    print(f"\n  Scanning {n_timesteps} timesteps for eclipse events...")
    for t in range(n_timesteps):
        solar_t = load_timestep(output_dir, 'direct_solar', t)
        if solar_t is None:
            continue
        solar_t = solar_t[:n_faces]

        # Eclipse: faces that should be sunlit (based on other faces being sunlit) but have zero flux
        # Simple proxy: sub-Jovian faces with zero solar when other equatorial faces have solar
        if np.sum(sub_jovian_mask) > 0:
            solar_at_subjovian.append(np.mean(solar_t[sub_jovian_mask]))
        if np.sum(anti_jovian_mask) > 0:
            solar_at_antijovian.append(np.mean(solar_t[anti_jovian_mask]))

    if solar_at_subjovian:
        sj = np.array(solar_at_subjovian)
        aj = np.array(solar_at_antijovian)
        print(f"  Sub-Jovian (lon~0°, |lat|<15°):")
        print(f"    Mean solar across timesteps: {sj.mean():.4f} W/m²")
        print(f"    Timesteps with zero solar: {np.sum(sj < 0.01)}/{len(sj)}")
        print(f"  Anti-Jovian (lon~180°, |lat|<15°):")
        print(f"    Mean solar across timesteps: {aj.mean():.4f} W/m²")
        print(f"    Timesteps with zero solar: {np.sum(aj < 0.01)}/{len(aj)}")
        print(f"  Difference (anti-Jovian - sub-Jovian): {aj.mean() - sj.mean():.4f} W/m²")

    # Try to generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f'Longitude Pattern Analysis: {output_dir.name}', fontsize=14, fontweight='bold')

        # Panel 1: Radiation components vs longitude (equatorial)
        ax = axes[0, 0]
        band = (-30, 30)
        if solar_avg is not None:
            lc, sm, ss, sc = bin_by_longitude(lon, solar_avg, lat, lat_band=band)
            zonal = np.nanmean(sm)
            ax.plot(lc, sm - zonal, 'b-', linewidth=1, label=f'Direct solar (mean={zonal:.2f})')
        if illum_avg is not None:
            lc, im, _, _ = bin_by_longitude(lon, illum_avg, lat, lat_band=band)
            zonal_i = np.nanmean(im)
            ax.plot(lc, im - zonal_i, 'g-', linewidth=1, label=f'Total illumination (mean={zonal_i:.2f})')
        if ir_avg is not None and np.any(ir_avg > 0):
            lc, irm, _, _ = bin_by_longitude(lon, ir_avg, lat, lat_band=band)
            ax.plot(lc, irm, 'r-', linewidth=1, label='Jupiter IR')
        if refl_avg is not None and np.any(refl_avg > 0):
            lc, rfm, _, _ = bin_by_longitude(lon, refl_avg, lat, lat_band=band)
            ax.plot(lc, rfm, 'm-', linewidth=1, label='Jupiter reflected')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Deviation from zonal mean (W/m²)')
        ax.set_title('Radiation Components (equatorial ±30°)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        for x in [0, 90, 180, 270]:
            ax.axvline(x, color='gray', linestyle=':', alpha=0.4)
            ax.axvline(x-360, color='gray', linestyle=':', alpha=0.4)

        # Panel 2: Temperature vs longitude
        ax = axes[0, 1]
        if T_avg is not None:
            lc, tm, ts, _ = bin_by_longitude(lon, T_avg, lat, lat_band=band)
            zonal = np.nanmean(tm)
            ax.plot(lc, tm - zonal, 'b-', linewidth=1, label=f'T_avg (mean={zonal:.2f} K)')
        if T_max is not None:
            lc, tmx, ts, _ = bin_by_longitude(lon, T_max, lat, lat_band=band)
            zonal = np.nanmean(tmx)
            ax.plot(lc, tmx - zonal, 'r-', linewidth=1, label=f'T_max (mean={zonal:.2f} K)')
        if T_min is not None:
            T_min = trim(T_min)
            if T_min is not None:
                lc, tmn, ts, _ = bin_by_longitude(lon, T_min, lat, lat_band=band)
                zonal = np.nanmean(tmn)
                ax.plot(lc, tmn - zonal, 'g-', linewidth=1, label=f'T_min (mean={zonal:.2f} K)')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Deviation from zonal mean (K)')
        ax.set_title('Temperature (equatorial ±30°)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        for x in [0, 90, 180, 270]:
            ax.axvline(x, color='gray', linestyle=':', alpha=0.4)
            ax.axvline(x-360, color='gray', linestyle=':', alpha=0.4)

        # Panel 3: Sublimation from T_max vs longitude
        ax = axes[1, 0]
        if log_sublimation is not None:
            lc, slm, sls, _ = bin_by_longitude(lon, log_sublimation, lat, lat_band=band)
            zonal = np.nanmean(slm)
            ax.plot(lc, slm - zonal, 'k-', linewidth=1)
            ax.fill_between(lc, slm - zonal - sls, slm - zonal + sls, alpha=0.2)
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel(f'Deviation from zonal mean\n(zonal mean = {zonal:.2f})')
            ax.set_title('log₁₀(sublimation flux) from T_max (equatorial ±30°)')
            ax.grid(True, alpha=0.3)
            for x in [0, 90, 180, 270]:
                ax.axvline(x, color='gray', linestyle=':', alpha=0.4)
                ax.axvline(x-360, color='gray', linestyle=':', alpha=0.4)

        # Panel 4: Pentagon vertex locations + face density
        ax = axes[1, 1]
        # Face count per longitude bin
        lon_bins = np.arange(-180, 182, 2)
        lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
        counts = np.histogram(lon, bins=lon_bins)[0]
        ax.bar(lon_c, counts, width=2, alpha=0.5, color='steelblue', label='Face count per 2° bin')
        # Mark pentagon vertices
        for i in range(12):
            vlon = vertex_lons[i]
            vlat = vertex_lats[i]
            ax.axvline(vlon, color='red', linestyle='--', alpha=0.6, linewidth=1)
        ax.axvline(vlon, color='red', linestyle='--', alpha=0.6, linewidth=1, label='Pentagon vertex')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Number of faces')
        ax.set_title('Face density vs longitude (all latitudes)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 5: Temperature anomaly vs pentagon proximity
        ax = axes[2, 0]
        if T_max is not None:
            eq_mask = np.abs(lat) < 60
            # Compute local zonal mean for comparison
            T_max_eq = T_max[eq_mask]
            lon_eq = lon[eq_mask]
            pent_dist_eq = pent_dist_deg[eq_mask]

            # Simple scatter
            ax.scatter(pent_dist_eq, T_max_eq, s=0.5, alpha=0.2, c='steelblue')
            # Bin by pentagon distance
            dist_bins = np.arange(0, 35, 1)
            dist_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
            T_by_dist = []
            for i in range(len(dist_bins)-1):
                dmask = (pent_dist_eq >= dist_bins[i]) & (pent_dist_eq < dist_bins[i+1])
                if np.sum(dmask) > 5:
                    T_by_dist.append(np.mean(T_max_eq[dmask]))
                else:
                    T_by_dist.append(np.nan)
            ax.plot(dist_centers, T_by_dist, 'r-', linewidth=2, label='Binned mean')
            ax.set_xlabel('Angular distance to nearest pentagon vertex (°)')
            ax.set_ylabel('T_max (K)')
            ax.set_title('T_max vs pentagon proximity (|lat| < 60°)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Panel 6: Different latitude bands
        ax = axes[2, 1]
        colors = ['blue', 'red', 'green', 'orange']
        for (band, label), color in zip(zip(lat_bands, band_labels), colors):
            if T_max is not None:
                lc, tmx, _, _ = bin_by_longitude(lon, T_max, lat, lat_band=band)
                zonal = np.nanmean(tmx)
                ax.plot(lc, tmx - zonal, '-', color=color, linewidth=1, label=f'{label}')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('T_max deviation from zonal mean (K)')
        ax.set_title('T_max longitude profile by latitude band')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        for x in [0, 90, 180, 270]:
            ax.axvline(x, color='gray', linestyle=':', alpha=0.4)
            ax.axvline(x-360, color='gray', linestyle=':', alpha=0.4)

        plt.tight_layout()
        fig_path = output_dir / 'longitude_pattern_analysis.png'
        plt.savefig(str(fig_path), dpi=150)
        print(f"\n  Figure saved to: {fig_path}")

    except ImportError:
        print("\n  matplotlib not available — skipping plots")

    # =====================================================================
    # EXOSPHERE ANALYSIS (if --exosphere-dir provided)
    # =====================================================================
    if exo_dir is not None:
        print(f"\n{'='*72}")
        print("EXOSPHERE OUTPUT ANALYSIS")
        print(f"{'='*72}")
        print(f"Exosphere directory: {exo_dir}")

        # Find available epochs
        epochs = find_exosphere_epochs(exo_dir, 'emission_flux_kg_m2_s')
        if not epochs:
            epochs = find_exosphere_epochs(exo_dir, 'net_flux_kg_m2_s')
        print(f"  Available epochs: {epochs}")

        if epochs:
            # Load exosphere mesh if different from PTM mesh
            exo_mesh_path = exo_dir / 'mesh.ply'
            if exo_mesh_path.exists():
                print(f"\nLoading exosphere mesh from {exo_mesh_path}...")
                exo_centers, exo_lat, exo_lon, exo_n_faces = load_mesh_ply(exo_mesh_path)
                print(f"  {exo_n_faces} faces")
                exo_pent_dist, exo_pent_nearest = compute_pentagon_proximity(exo_centers)
                exo_pent_dist_deg = np.degrees(exo_pent_dist)
            else:
                exo_lat, exo_lon, exo_n_faces = lat, lon, n_faces
                exo_pent_dist_deg = pent_dist_deg
                exo_pent_nearest = pent_nearest

            # Use final epoch for analysis
            final_epoch = epochs[-1]
            print(f"\n  Analyzing final epoch: {final_epoch}")

            # Load exosphere fields
            exo_fields = {}
            for field in ['emission_flux_kg_m2_s', 'deposition_flux_kg_m2_s',
                          'net_flux_kg_m2_s', 'nonice_concentration',
                          'ice_mass_kg_m2', 'accumulated_mass_kg_m2']:
                data = load_exosphere_epoch(exo_dir, field, final_epoch)
                if data is not None:
                    data = data[:exo_n_faces]
                    exo_fields[field] = data
                    print(f"  {field}: range [{data.min():.6e}, {data.max():.6e}]")

            # Also load initial epoch for comparison
            if epochs[0] != final_epoch:
                initial_epoch = epochs[0]
                print(f"\n  Loading initial epoch {initial_epoch} for comparison...")
                for field in ['emission_flux_kg_m2_s', 'nonice_concentration']:
                    data = load_exosphere_epoch(exo_dir, field, initial_epoch)
                    if data is not None:
                        exo_fields[f'{field}_initial'] = data[:exo_n_faces]

            # Print longitude profiles for exosphere fields
            print(f"\n--- Exosphere longitude profiles (equatorial ±30°) ---")
            band = (-30, 30)
            for field_name, label in [
                ('emission_flux_kg_m2_s', 'Sublimation flux'),
                ('deposition_flux_kg_m2_s', 'Deposition flux'),
                ('net_flux_kg_m2_s', 'Net flux'),
                ('nonice_concentration', 'Non-ice concentration'),
                ('accumulated_mass_kg_m2', 'Accumulated sublimated mass'),
            ]:
                if field_name in exo_fields:
                    data = exo_fields[field_name]
                    lc, bm, bs, bc = bin_by_longitude(exo_lon, data, exo_lat, lat_band=band)
                    zonal = np.nanmean(bm)
                    print(f"\n  {label} (epoch {final_epoch}):")
                    print(f"    Zonal mean: {zonal:.6e}")
                    for target_lon in [0, 90, 180, -90]:
                        idx = np.argmin(np.abs(lc - target_lon))
                        if not np.isnan(bm[idx]):
                            dev = bm[idx] - zonal
                            dev_pct = 100 * dev / zonal if zonal != 0 else 0
                            display_lon = target_lon if target_lon >= 0 else target_lon + 360
                            print(f"    lon={display_lon:3d}°: {bm[idx]:.6e} (deviation: {dev:+.4e}, {dev_pct:+.2f}%)")

            # Pentagon proximity analysis for exosphere fields
            print(f"\n--- Pentagon proximity vs exosphere fields ---")
            is_pent = exo_pent_dist_deg < pentagon_threshold
            is_regular = ~is_pent
            eq_mask = np.abs(exo_lat) < 30

            for field_name, label in [
                ('emission_flux_kg_m2_s', 'Sublimation flux'),
                ('net_flux_kg_m2_s', 'Net flux'),
                ('nonice_concentration', 'Non-ice concentration'),
            ]:
                if field_name in exo_fields:
                    data = exo_fields[field_name]
                    pent_vals = data[eq_mask & is_pent]
                    reg_vals = data[eq_mask & is_regular]
                    if len(pent_vals) > 0 and len(reg_vals) > 0:
                        ratio = pent_vals.mean() / reg_vals.mean() if reg_vals.mean() != 0 else float('inf')
                        print(f"  {label} (equatorial):")
                        print(f"    Pentagon: {pent_vals.mean():.6e} ± {pent_vals.std():.4e}")
                        print(f"    Regular:  {reg_vals.mean():.6e} ± {reg_vals.std():.4e}")
                        print(f"    Ratio (pent/reg): {ratio:.4f}")

            # Generate exosphere figure
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                n_exo_panels = 5
                fig2, axes2 = plt.subplots(3, 2, figsize=(16, 14))
                fig2.suptitle(f'Exosphere Longitude Analysis: {exo_dir.name} (epoch {final_epoch})',
                              fontsize=14, fontweight='bold')

                # Panel 1: Sublimation flux vs longitude
                ax = axes2[0, 0]
                if 'emission_flux_kg_m2_s' in exo_fields:
                    for band, label, color in [
                        ((-30, 30), 'Equatorial ±30°', 'blue'),
                        ((-60, -30), 'Southern mid-lat', 'green'),
                        ((30, 60), 'Northern mid-lat', 'orange'),
                    ]:
                        lc, bm, bs, _ = bin_by_longitude(exo_lon, exo_fields['emission_flux_kg_m2_s'],
                                                          exo_lat, lat_band=band)
                        zonal = np.nanmean(bm)
                        if zonal > 0:
                            ax.plot(lc, 100*(bm - zonal)/zonal, '-', color=color, linewidth=1, label=label)
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Deviation from zonal mean (%)')
                ax.set_title('Sublimation flux')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                for x in [0, 90, 180, -90]:
                    ax.axvline(x, color='gray', linestyle=':', alpha=0.4)

                # Panel 2: Deposition flux vs longitude
                ax = axes2[0, 1]
                if 'deposition_flux_kg_m2_s' in exo_fields:
                    for band, label, color in [
                        ((-30, 30), 'Equatorial ±30°', 'blue'),
                        ((-60, -30), 'Southern mid-lat', 'green'),
                        ((30, 60), 'Northern mid-lat', 'orange'),
                    ]:
                        lc, bm, bs, _ = bin_by_longitude(exo_lon, exo_fields['deposition_flux_kg_m2_s'],
                                                          exo_lat, lat_band=band)
                        zonal = np.nanmean(bm)
                        if zonal > 0:
                            ax.plot(lc, 100*(bm - zonal)/zonal, '-', color=color, linewidth=1, label=label)
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Deviation from zonal mean (%)')
                ax.set_title('Deposition flux')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                for x in [0, 90, 180, -90]:
                    ax.axvline(x, color='gray', linestyle=':', alpha=0.4)

                # Panel 3: Net flux vs longitude
                ax = axes2[1, 0]
                if 'net_flux_kg_m2_s' in exo_fields:
                    lc, bm, bs, _ = bin_by_longitude(exo_lon, exo_fields['net_flux_kg_m2_s'],
                                                      exo_lat, lat_band=(-30, 30))
                    ax.plot(lc, bm, 'k-', linewidth=1)
                    ax.fill_between(lc, bm - bs, bm + bs, alpha=0.2)
                    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Net flux (kg/m²/s)')
                ax.set_title('Net mass flux (equatorial ±30°)')
                ax.grid(True, alpha=0.3)
                for x in [0, 90, 180, -90]:
                    ax.axvline(x, color='gray', linestyle=':', alpha=0.4)

                # Panel 4: Non-ice concentration vs longitude
                ax = axes2[1, 1]
                if 'nonice_concentration' in exo_fields:
                    for band, label, color in [
                        ((-30, 30), 'Equatorial ±30°', 'blue'),
                        ((-60, -30), 'Southern mid-lat', 'green'),
                        ((30, 60), 'Northern mid-lat', 'orange'),
                        ((-90, 90), 'Global', 'black'),
                    ]:
                        lc, bm, bs, _ = bin_by_longitude(exo_lon, exo_fields['nonice_concentration'],
                                                          exo_lat, lat_band=band)
                        ax.plot(lc, 100*bm, '-', color=color, linewidth=1, label=label)
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Non-ice concentration (%)')
                ax.set_title('Non-ice lag concentration')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                for x in [0, 90, 180, -90]:
                    ax.axvline(x, color='gray', linestyle=':', alpha=0.4)

                # Panel 5: Net flux anomaly vs pentagon proximity
                ax = axes2[2, 0]
                if 'net_flux_kg_m2_s' in exo_fields:
                    eq_mask = np.abs(exo_lat) < 60
                    nf_eq = exo_fields['net_flux_kg_m2_s'][eq_mask]
                    pd_eq = exo_pent_dist_deg[eq_mask]
                    ax.scatter(pd_eq, nf_eq, s=0.5, alpha=0.2, c='steelblue')
                    # Bin
                    dist_bins = np.arange(0, 35, 1)
                    dist_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
                    nf_by_dist = []
                    for i in range(len(dist_bins)-1):
                        dmask = (pd_eq >= dist_bins[i]) & (pd_eq < dist_bins[i+1])
                        if np.sum(dmask) > 5:
                            nf_by_dist.append(np.mean(nf_eq[dmask]))
                        else:
                            nf_by_dist.append(np.nan)
                    ax.plot(dist_centers, nf_by_dist, 'r-', linewidth=2, label='Binned mean')
                    ax.set_xlabel('Angular distance to nearest pentagon vertex (°)')
                    ax.set_ylabel('Net flux (kg/m²/s)')
                    ax.set_title('Net flux vs pentagon proximity (|lat| < 60°)')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

                # Panel 6: Accumulated sublimated mass vs longitude
                ax = axes2[2, 1]
                if 'accumulated_mass_kg_m2' in exo_fields:
                    for band, label, color in [
                        ((-30, 30), 'Equatorial ±30°', 'blue'),
                        ((-60, -30), 'Southern mid-lat', 'green'),
                        ((30, 60), 'Northern mid-lat', 'orange'),
                    ]:
                        lc, bm, bs, _ = bin_by_longitude(exo_lon, exo_fields['accumulated_mass_kg_m2'],
                                                          exo_lat, lat_band=band)
                        zonal = np.nanmean(bm)
                        if zonal > 0:
                            ax.plot(lc, 100*(bm - zonal)/zonal, '-', color=color, linewidth=1, label=label)
                    ax.set_xlabel('Longitude (°)')
                    ax.set_ylabel('Deviation from zonal mean (%)')
                    ax.set_title('Accumulated sublimated mass')
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                    for x in [0, 90, 180, -90]:
                        ax.axvline(x, color='gray', linestyle=':', alpha=0.4)

                plt.tight_layout()
                fig2_path = exo_dir / 'exosphere_longitude_analysis.png'
                plt.savefig(str(fig2_path), dpi=150)
                print(f"\n  Exosphere figure saved to: {fig2_path}")

            except ImportError:
                print("\n  matplotlib not available — skipping exosphere plots")

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    if T_max is not None and solar_avg is not None:
        # Compare cardinal vs intermediate longitudes
        cardinal_lons = [0, 90, 180, -90]
        intermediate_lons = [45, 135, -135, -45]

        eq_mask = np.abs(lat) < 30
        for target_lons, label in [(cardinal_lons, 'Cardinal (0,90,180,270)'),
                                    (intermediate_lons, 'Intermediate (45,135,225,315)')]:
            lon_mask = np.zeros(n_faces, dtype=bool)
            for tl in target_lons:
                lon_mask |= (np.abs(lon - tl) < 10)
            combined = eq_mask & lon_mask
            if np.sum(combined) > 0:
                print(f"  {label} (eq ±30°, ±10°):")
                print(f"    T_max = {T_max[combined].mean():.3f} K, "
                      f"T_avg = {T_avg[combined].mean():.3f} K, "
                      f"solar_avg = {solar_avg[combined].mean():.4f} W/m²")

        print(f"\n  Pentagon vertex count at cardinal vs intermediate longitudes:")
        print(f"    Cardinal (0,90,180,270): 2 vertices each = 8 total")
        print(f"    Intermediate (58,122,238,302): 1 vertex each = 4 total")

    if exo_dir is not None and 'net_flux_kg_m2_s' in (exo_fields if 'exo_fields' in dir() else {}):
        print(f"\n  Exosphere cardinal vs intermediate (equatorial ±30°):")
        eq_mask_exo = np.abs(exo_lat) < 30
        for target_lons, label in [(cardinal_lons, 'Cardinal'),
                                    (intermediate_lons, 'Intermediate')]:
            lon_mask = np.zeros(exo_n_faces, dtype=bool)
            for tl in target_lons:
                lon_mask |= (np.abs(exo_lon - tl) < 10)
            combined = eq_mask_exo & lon_mask
            if np.sum(combined) > 0:
                nf = exo_fields['net_flux_kg_m2_s'][combined]
                ef = exo_fields.get('emission_flux_kg_m2_s', np.zeros(exo_n_faces))[combined]
                nc = exo_fields.get('nonice_concentration', np.zeros(exo_n_faces))[combined]
                print(f"    {label}: net_flux={nf.mean():.4e}, sublimation={ef.mean():.4e}, "
                      f"nonice={100*nc.mean():.3f}%")

    print(f"\n{'='*72}\n")


if __name__ == '__main__':
    main()
