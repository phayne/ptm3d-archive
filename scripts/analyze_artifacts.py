#!/usr/bin/env python3
"""
Analyze point-like artifacts in sputtering exosphere simulation output.
Investigates anomalous per-face values at equatorial faces in the icosphere mesh.
"""

import numpy as np
import struct
import os
import sys

BASE = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm"
SPUT_DIR = os.path.join(BASE, "output/europa_ico6_uniform-ice/exosphere/sputtering")
SUBL_DIR = os.path.join(BASE, "output/europa_ico6_uniform-ice/exosphere/coupled_5pct_v2")
EPOCH = 99999

def read_bin(path):
    """Read binary file: uint32 count + float32 array."""
    with open(path, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(count * 4), dtype='<f4')
    return data

def read_ply_faces(ply_path):
    """Read binary PLY mesh, extract face lat/lon and vertex indices."""
    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header
        n_vertices = 0
        n_faces = 0
        for line in header_lines:
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])

        print(f"PLY: {n_vertices} vertices, {n_faces} faces")

        # Read vertices: 3 doubles (x, y, z) per vertex
        vertices = np.zeros((n_vertices, 3))
        for i in range(n_vertices):
            xyz = struct.unpack('<ddd', f.read(24))
            vertices[i] = xyz

        # Read faces: uchar count + count*int indices + float lat + float lon
        face_verts = np.zeros((n_faces, 3), dtype=np.int32)
        face_lat = np.zeros(n_faces, dtype=np.float32)
        face_lon = np.zeros(n_faces, dtype=np.float32)

        for i in range(n_faces):
            nv = struct.unpack('<B', f.read(1))[0]
            assert nv == 3, f"Face {i} has {nv} vertices, expected 3"
            indices = struct.unpack(f'<{nv}i', f.read(nv * 4))
            face_verts[i] = indices
            lat, lon = struct.unpack('<ff', f.read(8))
            face_lat[i] = lat
            face_lon[i] = lon

    return vertices, face_verts, face_lat, face_lon

def compute_face_areas(vertices, face_verts):
    """Compute triangle areas from vertices."""
    v0 = vertices[face_verts[:, 0]]
    v1 = vertices[face_verts[:, 1]]
    v2 = vertices[face_verts[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas

def find_anomalous_faces(data, name, lat, lon, sigma_threshold=5.0, lat_band=15.0):
    """Find faces with values far from local median, focusing near equator."""
    equatorial = np.abs(lat) < lat_band
    eq_data = data[equatorial]
    eq_indices = np.where(equatorial)[0]

    if len(eq_data) == 0:
        print(f"  No equatorial faces found for {name}")
        return np.array([])

    median = np.median(eq_data)
    mad = np.median(np.abs(eq_data - median))
    if mad == 0:
        mad = np.std(eq_data)

    # Use modified z-score
    if mad > 0:
        modified_z = 0.6745 * (eq_data - median) / mad
    else:
        modified_z = np.zeros_like(eq_data)

    anomalous_mask = np.abs(modified_z) > sigma_threshold
    anomalous_eq_idx = np.where(anomalous_mask)[0]
    anomalous_global_idx = eq_indices[anomalous_mask]

    print(f"\n{'='*70}")
    print(f"FIELD: {name}")
    print(f"{'='*70}")
    print(f"  Global stats: min={data.min():.6e}, max={data.max():.6e}, "
          f"mean={data.mean():.6e}, std={data.std():.6e}")
    print(f"  Equatorial (|lat|<{lat_band}°) stats: n={len(eq_data)}, "
          f"min={eq_data.min():.6e}, max={eq_data.max():.6e}, "
          f"mean={eq_data.mean():.6e}, median={median:.6e}, MAD={mad:.6e}")
    print(f"  Anomalous faces (|modified_z| > {sigma_threshold}): {len(anomalous_global_idx)}")

    if len(anomalous_global_idx) > 0:
        print(f"\n  Top anomalous faces (sorted by |deviation|):")
        sort_idx = np.argsort(-np.abs(modified_z[anomalous_eq_idx]))
        for rank, si in enumerate(sort_idx[:30]):
            gi = anomalous_global_idx[si]
            val = data[gi]
            z = modified_z[anomalous_eq_idx[si]]
            print(f"    #{rank+1}: face {gi:6d}, lat={lat[gi]:7.2f}°, lon={lon[gi]:8.2f}°, "
                  f"value={val:.6e}, z-score={z:.1f}")

    return anomalous_global_idx

def analyze_face_neighborhoods(anomalous_faces, face_verts, data, name):
    """Check if anomalous faces share vertices (mesh topology analysis)."""
    if len(anomalous_faces) == 0:
        return

    # Build vertex-to-face adjacency
    max_vert = face_verts.max() + 1
    vert_to_faces = [[] for _ in range(max_vert)]
    for fi in range(len(face_verts)):
        for vi in face_verts[fi]:
            vert_to_faces[vi].append(fi)

    print(f"\n  Neighborhood analysis for {name}:")
    for fi in anomalous_faces[:10]:
        neighbors = set()
        for vi in face_verts[fi]:
            for nf in vert_to_faces[vi]:
                if nf != fi:
                    neighbors.add(nf)
        neighbor_vals = [data[n] for n in neighbors]
        val = data[fi]
        if neighbor_vals:
            mean_neighbor = np.mean(neighbor_vals)
            ratio = val / mean_neighbor if mean_neighbor != 0 else float('inf')
            print(f"    Face {fi}: value={val:.6e}, neighbor_mean={mean_neighbor:.6e}, "
                  f"ratio={ratio:.2f}, n_neighbors={len(neighbors)}")

def check_icosphere_seam(face_verts, vertices, lat, lon, anomalous_all):
    """Check if anomalous faces correspond to icosphere construction seams."""
    print(f"\n{'='*70}")
    print("ICOSPHERE SEAM ANALYSIS")
    print(f"{'='*70}")

    # Icosahedron has 12 original vertices. In an ico6 sphere, the original
    # 12 vertices are at specific locations. 5 around equator at lon=0,72,144,-144,-72
    # and poles.

    # Check vertex distribution near equator
    eq_verts_mask = np.abs(vertices[:, 2]) < 0.1 * np.max(np.abs(vertices[:, 2]))
    eq_verts = vertices[eq_verts_mask]
    eq_vert_indices = np.where(eq_verts_mask)[0]
    print(f"  Vertices near equator (|z| < 10% of max): {len(eq_verts)}")

    # Check for icosphere pentagon vertices (valence 5 instead of 6)
    vert_valence = np.zeros(len(vertices), dtype=int)
    for fi in range(len(face_verts)):
        for vi in face_verts[fi]:
            vert_valence[vi] += 1

    pentagon_verts = np.where(vert_valence == 5)[0]
    print(f"  Pentagon vertices (valence 5): {len(pentagon_verts)}")
    if len(pentagon_verts) > 0:
        for pv in pentagon_verts:
            v = vertices[pv]
            r = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            plat = np.degrees(np.arcsin(v[2] / r)) if r > 0 else 0
            plon = np.degrees(np.arctan2(v[1], v[0]))
            print(f"    Vertex {pv}: lat={plat:.2f}°, lon={plon:.2f}°, valence={vert_valence[pv]}")

    # Check if anomalous faces are near pentagon vertices
    if len(anomalous_all) > 0 and len(pentagon_verts) > 0:
        print(f"\n  Proximity of anomalous faces to pentagon vertices:")
        # Get face centroids
        for fi in anomalous_all[:20]:
            fv = face_verts[fi]
            centroid = vertices[fv].mean(axis=0)
            min_dist = float('inf')
            nearest_pv = -1
            for pv in pentagon_verts:
                dist = np.linalg.norm(centroid - vertices[pv])
                if dist < min_dist:
                    min_dist = dist
                    nearest_pv = pv
            pv_v = vertices[nearest_pv]
            r = np.sqrt(pv_v[0]**2 + pv_v[1]**2 + pv_v[2]**2)
            pv_lat = np.degrees(np.arcsin(pv_v[2] / r)) if r > 0 else 0
            pv_lon = np.degrees(np.arctan2(pv_v[1], pv_v[0]))
            # Convert dist to approximate angular separation
            angular_dist = np.degrees(2 * np.arcsin(min_dist / (2 * r))) if r > 0 else 0
            touches = nearest_pv in fv
            print(f"    Face {fi} (lat={lat[fi]:.2f}°, lon={lon[fi]:.2f}°): "
                  f"nearest pentagon vertex {nearest_pv} (lat={pv_lat:.2f}°, lon={pv_lon:.2f}°), "
                  f"angular_dist={angular_dist:.2f}°, touches={touches}")

def compare_with_sublimation(lat, lon):
    """Compare sputtering anomalies with sublimation-only run."""
    print(f"\n{'='*70}")
    print("COMPARISON WITH SUBLIMATION-ONLY RUN")
    print(f"{'='*70}")

    subl_nonice_path = os.path.join(SUBL_DIR, f"nonice_concentration/nonice_concentration_{EPOCH}.bin")
    subl_net_path = os.path.join(SUBL_DIR, f"net_flux_kg_m2_s/net_flux_kg_m2_s_{EPOCH}.bin")

    if not os.path.exists(subl_nonice_path):
        print(f"  Sublimation nonice file not found: {subl_nonice_path}")
        return

    subl_nonice = read_bin(subl_nonice_path)
    print(f"  Sublimation nonice_concentration: n={len(subl_nonice)}, "
          f"min={subl_nonice.min():.6e}, max={subl_nonice.max():.6e}, "
          f"mean={subl_nonice.mean():.6e}")

    # Find anomalous in sublimation run too
    equatorial = np.abs(lat) < 15.0
    eq_data = subl_nonice[equatorial]
    eq_indices = np.where(equatorial)[0]
    median = np.median(eq_data)
    mad = np.median(np.abs(eq_data - median))
    if mad > 0:
        modified_z = 0.6745 * (eq_data - median) / mad
        subl_anomalous = eq_indices[np.abs(modified_z) > 5.0]
        print(f"  Sublimation anomalous equatorial faces: {len(subl_anomalous)}")
        if len(subl_anomalous) > 0:
            for fi in subl_anomalous[:10]:
                print(f"    Face {fi}: lat={lat[fi]:.2f}°, lon={lon[fi]:.2f}°, "
                      f"nonice={subl_nonice[fi]:.6e}")
    else:
        print(f"  MAD=0 for sublimation equatorial nonice (all same value?)")
        # Check value distribution
        unique, counts = np.unique(eq_data, return_counts=True)
        print(f"  Unique values: {len(unique)}")
        if len(unique) <= 20:
            for u, c in zip(unique, counts):
                print(f"    value={u:.6e}: {c} faces")

    if os.path.exists(subl_net_path):
        subl_net = read_bin(subl_net_path)
        print(f"\n  Sublimation net_flux: min={subl_net.min():.6e}, max={subl_net.max():.6e}, "
              f"mean={subl_net.mean():.6e}")
        eq_net = subl_net[equatorial]
        median_n = np.median(eq_net)
        mad_n = np.median(np.abs(eq_net - median_n))
        if mad_n > 0:
            mz = 0.6745 * (eq_net - median_n) / mad_n
            subl_anom_net = eq_indices[np.abs(mz) > 5.0]
            print(f"  Sublimation anomalous net_flux equatorial faces: {len(subl_anom_net)}")
            if len(subl_anom_net) > 0:
                for fi in subl_anom_net[:10]:
                    print(f"    Face {fi}: lat={lat[fi]:.2f}°, lon={lon[fi]:.2f}°, "
                          f"net_flux={subl_net[fi]:.6e}")

def analyze_sputtering_weight(lat, lon):
    """Check the sputtering spatial weight field for anomalies."""
    weight_path = os.path.join(SPUT_DIR, f"sputtering_spatial_weight/sputtering_spatial_weight_{EPOCH}.bin")
    if not os.path.exists(weight_path):
        # Try other epochs or check directory
        weight_dir = os.path.join(SPUT_DIR, "sputtering_spatial_weight")
        if os.path.isdir(weight_dir):
            files = sorted(os.listdir(weight_dir))
            if files:
                weight_path = os.path.join(weight_dir, files[-1])
                print(f"\n  Using sputtering weight file: {files[-1]}")
            else:
                print(f"\n  No sputtering weight files found")
                return
        else:
            print(f"\n  Sputtering weight directory not found")
            return

    weight = read_bin(weight_path)
    equatorial = np.abs(lat) < 15.0
    eq_w = weight[equatorial]
    eq_idx = np.where(equatorial)[0]

    print(f"\n{'='*70}")
    print("SPUTTERING SPATIAL WEIGHT")
    print(f"{'='*70}")
    print(f"  Global: min={weight.min():.6e}, max={weight.max():.6e}, "
          f"mean={weight.mean():.6e}, std={weight.std():.6e}")
    print(f"  Equatorial: min={eq_w.min():.6e}, max={eq_w.max():.6e}, "
          f"mean={eq_w.mean():.6e}")

    # Check for unusual weights
    median = np.median(eq_w)
    mad = np.median(np.abs(eq_w - median))
    if mad > 0:
        mz = 0.6745 * (eq_w - median) / mad
        anom = eq_idx[np.abs(mz) > 5.0]
        print(f"  Anomalous weight faces: {len(anom)}")
        for fi in anom[:10]:
            print(f"    Face {fi}: lat={lat[fi]:.2f}°, lon={lon[fi]:.2f}°, weight={weight[fi]:.6e}")

def analyze_face_areas(areas, lat, lon, anomalous_all):
    """Check if anomalous faces have unusual areas."""
    print(f"\n{'='*70}")
    print("FACE AREA ANALYSIS")
    print(f"{'='*70}")

    equatorial = np.abs(lat) < 15.0
    eq_areas = areas[equatorial]

    print(f"  Global areas: min={areas.min():.6e}, max={areas.max():.6e}, "
          f"mean={areas.mean():.6e}, std={areas.std():.6e}, "
          f"ratio(max/min)={areas.max()/areas.min():.3f}")
    print(f"  Equatorial areas: min={eq_areas.min():.6e}, max={eq_areas.max():.6e}, "
          f"mean={eq_areas.mean():.6e}")

    if len(anomalous_all) > 0:
        anom_areas = areas[anomalous_all]
        print(f"  Anomalous face areas: min={anom_areas.min():.6e}, max={anom_areas.max():.6e}, "
              f"mean={anom_areas.mean():.6e}")
        print(f"  Anomalous / global mean area ratio: {anom_areas.mean()/areas.mean():.4f}")

def check_temporal_evolution(anomalous_faces, field_name, field_dir, lat, lon):
    """Check how anomalous faces evolve over time."""
    print(f"\n{'='*70}")
    print(f"TEMPORAL EVOLUTION: {field_name}")
    print(f"{'='*70}")

    if len(anomalous_faces) == 0:
        print("  No anomalous faces to track.")
        return

    files = sorted(os.listdir(field_dir))
    # Sample a few epochs
    sample_files = files[::max(1, len(files)//10)]  # ~10 samples
    if files[-1] not in sample_files:
        sample_files.append(files[-1])

    track_faces = anomalous_faces[:5]
    print(f"  Tracking faces: {list(track_faces)}")
    print(f"  {'Epoch':>8s}", end="")
    for fi in track_faces:
        print(f"  face{fi:6d}", end="")
    print(f"  {'eq_median':>12s}  {'eq_std':>12s}")

    equatorial = np.abs(lat) < 15.0

    for fname in sample_files:
        epoch_str = fname.split('_')[-1].replace('.bin', '')
        path = os.path.join(field_dir, fname)
        try:
            data = read_bin(path)
            eq_data = data[equatorial]
            print(f"  {epoch_str:>8s}", end="")
            for fi in track_faces:
                print(f"  {data[fi]:12.4e}", end="")
            print(f"  {np.median(eq_data):12.4e}  {np.std(eq_data):12.4e}")
        except Exception as e:
            print(f"  {epoch_str:>8s}  ERROR: {e}")

def check_sputtering_yield(lat, lon):
    """Check sputtering yield for anomalies."""
    yield_dir = os.path.join(SPUT_DIR, "sputtering_yield")
    if not os.path.isdir(yield_dir):
        print("\n  No sputtering_yield directory")
        return np.array([])

    files = sorted(os.listdir(yield_dir))
    if not files:
        return np.array([])

    yield_path = os.path.join(yield_dir, files[-1])
    syield = read_bin(yield_path)

    print(f"\n{'='*70}")
    print(f"SPUTTERING YIELD (from {files[-1]})")
    print(f"{'='*70}")
    equatorial = np.abs(lat) < 15.0
    eq_y = syield[equatorial]
    print(f"  Global: min={syield.min():.6e}, max={syield.max():.6e}, mean={syield.mean():.6e}")
    print(f"  Equatorial: min={eq_y.min():.6e}, max={eq_y.max():.6e}, mean={eq_y.mean():.6e}")

    return syield


def main():
    print("="*70)
    print("SPUTTERING ARTIFACT ANALYSIS")
    print("="*70)

    # 1. Read mesh
    print("\n--- Reading mesh ---")
    vertices, face_verts, lat, lon = read_ply_faces(os.path.join(SPUT_DIR, "mesh.ply"))
    areas = compute_face_areas(vertices, face_verts)

    # 2. Read binary data for final epoch
    print("\n--- Reading epoch 99999 data ---")
    nonice = read_bin(os.path.join(SPUT_DIR, f"nonice_concentration/nonice_concentration_{EPOCH}.bin"))
    sput_flux = read_bin(os.path.join(SPUT_DIR, f"sputtering_flux_kg_m2_s/sputtering_flux_kg_m2_s_{EPOCH}.bin"))
    net_flux = read_bin(os.path.join(SPUT_DIR, f"net_flux_kg_m2_s/net_flux_kg_m2_s_{EPOCH}.bin"))

    print(f"  nonice_concentration: {len(nonice)} values")
    print(f"  sputtering_flux: {len(sput_flux)} values")
    print(f"  net_flux: {len(net_flux)} values")

    # 3. Find anomalous faces
    anom_nonice = find_anomalous_faces(nonice, "nonice_concentration", lat, lon)
    anom_sput = find_anomalous_faces(sput_flux, "sputtering_flux_kg_m2_s", lat, lon)
    anom_net = find_anomalous_faces(net_flux, "net_flux_kg_m2_s", lat, lon)

    # Union of all anomalous
    all_anomalous = np.unique(np.concatenate([anom_nonice, anom_sput, anom_net]))
    print(f"\n{'='*70}")
    print(f"TOTAL UNIQUE ANOMALOUS FACES: {len(all_anomalous)}")
    print(f"{'='*70}")

    # 4. Neighborhood analysis
    if len(all_anomalous) > 0:
        analyze_face_neighborhoods(all_anomalous, face_verts, nonice, "nonice_concentration")
        analyze_face_neighborhoods(all_anomalous, face_verts, sput_flux, "sputtering_flux")

    # 5. Face area analysis
    analyze_face_areas(areas, lat, lon, all_anomalous)

    # 6. Icosphere seam analysis
    check_icosphere_seam(face_verts, vertices, lat, lon, all_anomalous)

    # 7. Sputtering weight and yield
    analyze_sputtering_weight(lat, lon)
    check_sputtering_yield(lat, lon)

    # 8. Temporal evolution
    if len(anom_nonice) > 0:
        check_temporal_evolution(anom_nonice, "nonice_concentration",
                                os.path.join(SPUT_DIR, "nonice_concentration"), lat, lon)
    if len(anom_sput) > 0:
        check_temporal_evolution(anom_sput, "sputtering_flux",
                                os.path.join(SPUT_DIR, "sputtering_flux_kg_m2_s"), lat, lon)

    # 9. Compare with sublimation-only
    compare_with_sublimation(lat, lon)

    # 10. Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if len(all_anomalous) > 0:
        # Check lat/lon clustering
        anom_lats = lat[all_anomalous]
        anom_lons = lon[all_anomalous]
        print(f"  Anomalous faces: {len(all_anomalous)}")
        print(f"  Latitude range: {anom_lats.min():.2f}° to {anom_lats.max():.2f}°")
        print(f"  Longitude range: {anom_lons.min():.2f}° to {anom_lons.max():.2f}°")

        # Cluster by longitude
        print(f"\n  Longitude distribution of anomalous faces:")
        lon_bins = np.arange(-180, 181, 30)
        for i in range(len(lon_bins)-1):
            in_bin = np.sum((anom_lons >= lon_bins[i]) & (anom_lons < lon_bins[i+1]))
            if in_bin > 0:
                print(f"    [{lon_bins[i]:4d}°, {lon_bins[i+1]:4d}°): {in_bin} faces")

        # Check overlap between fields
        set_nonice = set(anom_nonice)
        set_sput = set(anom_sput)
        set_net = set(anom_net)
        print(f"\n  Overlap:")
        print(f"    nonice & sput: {len(set_nonice & set_sput)}")
        print(f"    nonice & net: {len(set_nonice & set_net)}")
        print(f"    sput & net: {len(set_sput & set_net)}")
        print(f"    all three: {len(set_nonice & set_sput & set_net)}")
    else:
        print("  No anomalous faces found with current thresholds.")

    # Also look at the extreme faces globally, regardless of z-score
    print(f"\n{'='*70}")
    print("EXTREME VALUE CHECK (top/bottom 10 globally)")
    print(f"{'='*70}")
    for name, data in [("nonice_concentration", nonice),
                       ("sputtering_flux", sput_flux),
                       ("net_flux", net_flux)]:
        print(f"\n  {name}:")
        sorted_idx = np.argsort(data)
        print(f"  Bottom 10:")
        for i in sorted_idx[:10]:
            print(f"    face {i:6d}: lat={lat[i]:7.2f}°, lon={lon[i]:8.2f}°, val={data[i]:.6e}")
        print(f"  Top 10:")
        for i in sorted_idx[-10:][::-1]:
            print(f"    face {i:6d}: lat={lat[i]:7.2f}°, lon={lon[i]:8.2f}°, val={data[i]:.6e}")


if __name__ == '__main__':
    main()
