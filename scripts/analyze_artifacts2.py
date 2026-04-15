#!/usr/bin/env python3
"""
Deep-dive into artifact pattern: check the sub-solar longitude connection,
sputtering yield bimodality, and whether the anomalous faces correspond to
the sub-Jovian / anti-Jovian hemisphere pattern expected from sputtering.
"""

import numpy as np
import struct
import os

BASE = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm"
SPUT_DIR = os.path.join(BASE, "output/europa_ico6_uniform-ice/exosphere/sputtering")
SUBL_DIR = os.path.join(BASE, "output/europa_ico6_uniform-ice/exosphere/coupled_5pct_v2")
EPOCH = 99999

def read_bin(path):
    with open(path, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(count * 4), dtype='<f4')
    return data

def read_ply_faces(ply_path):
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        n_vertices = 0
        n_faces = 0
        for line in header_lines:
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
        vertices = np.zeros((n_vertices, 3))
        for i in range(n_vertices):
            xyz = struct.unpack('<ddd', f.read(24))
            vertices[i] = xyz
        face_verts = np.zeros((n_faces, 3), dtype=np.int32)
        face_lat = np.zeros(n_faces, dtype=np.float32)
        face_lon = np.zeros(n_faces, dtype=np.float32)
        for i in range(n_faces):
            nv = struct.unpack('<B', f.read(1))[0]
            indices = struct.unpack(f'<{nv}i', f.read(nv * 4))
            face_verts[i] = indices
            lat, lon = struct.unpack('<ff', f.read(8))
            face_lat[i] = lat
            face_lon[i] = lon
    return vertices, face_verts, face_lat, face_lon

def main():
    vertices, face_verts, lat, lon = read_ply_faces(os.path.join(SPUT_DIR, "mesh.ply"))

    nonice = read_bin(os.path.join(SPUT_DIR, f"nonice_concentration/nonice_concentration_{EPOCH}.bin"))
    sput_flux = read_bin(os.path.join(SPUT_DIR, f"sputtering_flux_kg_m2_s/sputtering_flux_kg_m2_s_{EPOCH}.bin"))
    net_flux = read_bin(os.path.join(SPUT_DIR, f"net_flux_kg_m2_s/net_flux_kg_m2_s_{EPOCH}.bin"))
    syield = read_bin(os.path.join(SPUT_DIR, f"sputtering_yield/sputtering_yield_{EPOCH}.bin"))

    # Also read sputtering mass loss rate and temperatures
    sput_mass_loss = read_bin(os.path.join(SPUT_DIR, f"sputtering_mass_loss_kg_m2_s/sputtering_mass_loss_kg_m2_s_{EPOCH}.bin"))
    temp_mean = read_bin(os.path.join(SPUT_DIR, f"temperature_mean_K/temperature_mean_K_{EPOCH}.bin"))
    temp_max = read_bin(os.path.join(SPUT_DIR, f"temperature_max_K/temperature_max_K_{EPOCH}.bin"))
    ice_mass = read_bin(os.path.join(SPUT_DIR, f"ice_mass_kg_m2/ice_mass_kg_m2_{EPOCH}.bin"))
    bond_albedo = read_bin(os.path.join(SPUT_DIR, f"bond_albedo/bond_albedo_{EPOCH}.bin"))

    # Read sublimation/emission flux and deposition
    emission = read_bin(os.path.join(SPUT_DIR, f"emission_flux_kg_m2_s/emission_flux_kg_m2_s_{EPOCH}.bin"))
    deposition = read_bin(os.path.join(SPUT_DIR, f"deposition_flux_kg_m2_s/deposition_flux_kg_m2_s_{EPOCH}.bin"))
    cumul_sput = read_bin(os.path.join(SPUT_DIR, f"cumulative_sputtered_kg_m2/cumulative_sputtered_kg_m2_{EPOCH}.bin"))
    cumul_subl = read_bin(os.path.join(SPUT_DIR, f"cumulative_sublimated_kg_m2/cumulative_sublimated_kg_m2_{EPOCH}.bin"))
    cumul_dep = read_bin(os.path.join(SPUT_DIR, f"cumulative_deposited_kg_m2/cumulative_deposited_kg_m2_{EPOCH}.bin"))

    # Focus on the anomalous band: lat 0-10, which contains most artifacts
    equatorial = np.abs(lat) < 10.0
    eq_idx = np.where(equatorial)[0]

    # Identify anomalous faces by nonice > 0.5 in equatorial band
    eq_high_nonice = eq_idx[nonice[eq_idx] > 0.5]
    eq_low_nonice = eq_idx[nonice[eq_idx] < 0.01]  # "normal" equatorial faces

    print("="*80)
    print("ANOMALOUS vs NORMAL EQUATORIAL FACES (|lat|<10)")
    print("="*80)
    print(f"Total equatorial faces: {len(eq_idx)}")
    print(f"Anomalous (nonice > 0.5): {len(eq_high_nonice)}")
    print(f"Normal (nonice < 0.01): {len(eq_low_nonice)}")

    print(f"\n{'Field':<30s} {'Anomalous (mean)':<18s} {'Normal (mean)':<18s} {'Ratio':>8s}")
    print("-"*80)

    for name, data in [
        ("nonice_concentration", nonice),
        ("sputtering_flux", sput_flux),
        ("sputtering_mass_loss", sput_mass_loss),
        ("sputtering_yield", syield),
        ("net_flux", net_flux),
        ("emission_flux", emission),
        ("deposition_flux", deposition),
        ("temperature_mean_K", temp_mean),
        ("temperature_max_K", temp_max),
        ("ice_mass_kg_m2", ice_mass),
        ("bond_albedo", bond_albedo),
        ("cumulative_sputtered", cumul_sput),
        ("cumulative_sublimated", cumul_subl),
        ("cumulative_deposited", cumul_dep),
    ]:
        anom_mean = data[eq_high_nonice].mean() if len(eq_high_nonice) > 0 else 0
        norm_mean = data[eq_low_nonice].mean() if len(eq_low_nonice) > 0 else 0
        ratio = anom_mean / norm_mean if norm_mean != 0 else float('inf')
        print(f"{name:<30s} {anom_mean:<18.6e} {norm_mean:<18.6e} {ratio:>8.2f}")

    # Check the bimodality of sputtering yield in the equatorial band
    print(f"\n{'='*80}")
    print("SPUTTERING YIELD DISTRIBUTION AT EQUATOR")
    print(f"{'='*80}")
    eq_yield = syield[eq_idx]
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        print(f"  {p:3d}th percentile: {np.percentile(eq_yield, p):.4f}")

    # Histogram of equatorial yield
    print(f"\n  Yield histogram (equatorial faces):")
    bins = np.linspace(eq_yield.min(), eq_yield.max(), 20)
    hist, edges = np.histogram(eq_yield, bins=bins)
    for i in range(len(hist)):
        bar = '#' * min(60, int(60 * hist[i] / hist.max()))
        print(f"  [{edges[i]:5.2f}, {edges[i+1]:5.2f}): {hist[i]:5d} {bar}")

    # Check which specific faces near sub-Jovian point (lon=0) are anomalous
    print(f"\n{'='*80}")
    print("FACE-BY-FACE AT SUB-JOVIAN POINT (|lat|<5, |lon|<10)")
    print(f"{'='*80}")
    near_sj = eq_idx[(np.abs(lat[eq_idx]) < 5) & (np.abs(lon[eq_idx]) < 10)]
    print(f"Faces near sub-Jovian: {len(near_sj)}")
    print(f"{'face':>6s} {'lat':>7s} {'lon':>8s} {'nonice':>12s} {'sput_flux':>12s} {'yield':>8s} {'Tmean':>7s} {'Tmax':>7s} {'albedo':>7s} {'ice_mass':>12s}")
    sort_idx = np.argsort(nonice[near_sj])[::-1]
    for si in sort_idx[:30]:
        fi = near_sj[si]
        print(f"{fi:6d} {lat[fi]:7.2f} {lon[fi]:8.2f} {nonice[fi]:12.6e} {sput_flux[fi]:12.4e} {syield[fi]:8.4f} {temp_mean[fi]:7.1f} {temp_max[fi]:7.1f} {bond_albedo[fi]:7.4f} {ice_mass[fi]:12.4e}")

    # Check trailing hemisphere apex (lon=90) where sputtering should peak
    print(f"\n{'='*80}")
    print("FACE-BY-FACE AT TRAILING APEX (|lat|<5, |lon-90|<10)")
    print(f"{'='*80}")
    near_trail = eq_idx[(np.abs(lat[eq_idx]) < 5) & (np.abs(lon[eq_idx] - 90) < 10)]
    print(f"Faces near trailing apex: {len(near_trail)}")
    print(f"{'face':>6s} {'lat':>7s} {'lon':>8s} {'nonice':>12s} {'sput_flux':>12s} {'yield':>8s} {'Tmean':>7s} {'Tmax':>7s} {'albedo':>7s} {'ice_mass':>12s}")
    sort_idx = np.argsort(nonice[near_trail])[::-1]
    for si in sort_idx[:30]:
        fi = near_trail[si]
        print(f"{fi:6d} {lat[fi]:7.2f} {lon[fi]:8.2f} {nonice[fi]:12.6e} {sput_flux[fi]:12.4e} {syield[fi]:8.4f} {temp_mean[fi]:7.1f} {temp_max[fi]:7.1f} {bond_albedo[fi]:7.4f} {ice_mass[fi]:12.4e}")

    # KEY QUESTION: Is the issue that once ice depletes, nonice jumps to 1 and
    # sputtering yield goes to max (pure non-ice), creating a runaway?
    # Check ice_mass at anomalous faces vs normal
    print(f"\n{'='*80}")
    print("ICE MASS ANALYSIS")
    print(f"{'='*80}")
    print(f"  Anomalous faces - ice_mass: min={ice_mass[eq_high_nonice].min():.6e}, "
          f"max={ice_mass[eq_high_nonice].max():.6e}, mean={ice_mass[eq_high_nonice].mean():.6e}")
    print(f"  Normal faces - ice_mass: min={ice_mass[eq_low_nonice].min():.6e}, "
          f"max={ice_mass[eq_low_nonice].max():.6e}, mean={ice_mass[eq_low_nonice].mean():.6e}")

    # Faces that have essentially no ice left
    depleted = np.where(ice_mass < 1e-10)[0]
    print(f"  Faces with ice_mass < 1e-10: {len(depleted)}")
    if len(depleted) > 0:
        print(f"    lat range: {lat[depleted].min():.2f} to {lat[depleted].max():.2f}")
        print(f"    lon range: {lon[depleted].min():.2f} to {lon[depleted].max():.2f}")

    # Check: does anomalous nonice correlate with low ice_mass?
    print(f"\n  Correlation: nonice_concentration vs ice_mass at equator:")
    eq_corr = np.corrcoef(nonice[eq_idx], ice_mass[eq_idx])[0, 1]
    print(f"    Pearson r = {eq_corr:.4f}")

    # Check: does anomalous nonice correlate with high sputtering yield?
    print(f"  Correlation: nonice_concentration vs sputtering_yield at equator:")
    eq_corr2 = np.corrcoef(nonice[eq_idx], syield[eq_idx])[0, 1]
    print(f"    Pearson r = {eq_corr2:.4f}")

    # Check: does anomalous nonice correlate with temperature?
    print(f"  Correlation: nonice_concentration vs temp_mean at equator:")
    eq_corr3 = np.corrcoef(nonice[eq_idx], temp_mean[eq_idx])[0, 1]
    print(f"    Pearson r = {eq_corr3:.4f}")

    # Check the sputtering weight
    sput_weight = read_bin(os.path.join(SPUT_DIR, "sputtering_spatial_weight/sputtering_spatial_weight_0000.bin"))
    print(f"\n{'='*80}")
    print("SPUTTERING WEIGHT AT ANOMALOUS vs NORMAL FACES")
    print(f"{'='*80}")
    print(f"  Anomalous: mean={sput_weight[eq_high_nonice].mean():.4f}, "
          f"min={sput_weight[eq_high_nonice].min():.4f}, max={sput_weight[eq_high_nonice].max():.4f}")
    print(f"  Normal:    mean={sput_weight[eq_low_nonice].mean():.4f}, "
          f"min={sput_weight[eq_low_nonice].min():.4f}, max={sput_weight[eq_low_nonice].max():.4f}")

    # Now check if there is something special about the TOPOLOGY of anomalous faces
    # Build adjacency and check if anomalous faces form isolated points or connected patches
    print(f"\n{'='*80}")
    print("TOPOLOGY: Are anomalous faces isolated or in patches?")
    print(f"{'='*80}")

    max_vert = face_verts.max() + 1
    vert_to_faces = [[] for _ in range(max_vert)]
    for fi in range(len(face_verts)):
        for vi in face_verts[fi]:
            vert_to_faces[vi].append(fi)

    anom_set = set(eq_high_nonice)
    isolated = 0
    has_anom_neighbor = 0
    for fi in eq_high_nonice:
        neighbors = set()
        for vi in face_verts[fi]:
            for nf in vert_to_faces[vi]:
                if nf != fi:
                    neighbors.add(nf)
        anom_neighbors = neighbors & anom_set
        if len(anom_neighbors) == 0:
            isolated += 1
        else:
            has_anom_neighbor += 1

    print(f"  Anomalous faces with NO anomalous neighbors: {isolated} (truly isolated)")
    print(f"  Anomalous faces with >=1 anomalous neighbor: {has_anom_neighbor} (in patches)")

    # Connected component analysis
    visited = set()
    components = []
    for fi in eq_high_nonice:
        if fi in visited:
            continue
        # BFS
        comp = []
        queue = [fi]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            comp.append(current)
            neighbors = set()
            for vi in face_verts[current]:
                for nf in vert_to_faces[vi]:
                    if nf != current and nf in anom_set and nf not in visited:
                        neighbors.add(nf)
            queue.extend(neighbors)
        components.append(comp)

    print(f"  Connected components of anomalous faces: {len(components)}")
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"  Component sizes: {comp_sizes[:20]}{'...' if len(comp_sizes) > 20 else ''}")

    # For each component, print its centroid location
    print(f"\n  Component locations (centroid lat/lon):")
    for i, comp in enumerate(sorted(components, key=len, reverse=True)[:20]):
        clat = np.mean([lat[fi] for fi in comp])
        clon = np.mean([lon[fi] for fi in comp])
        print(f"    Component {i}: size={len(comp):3d}, centroid=({clat:.1f}°, {clon:.1f}°)")

    # Compare nonice between sputtering and sublimation runs
    print(f"\n{'='*80}")
    print("SUBLIMATION-ONLY COMPARISON: Same faces?")
    print(f"{'='*80}")
    subl_nonice = read_bin(os.path.join(SUBL_DIR, f"nonice_concentration/nonice_concentration_{EPOCH}.bin"))
    subl_eq_high = eq_idx[subl_nonice[eq_idx] > 0.5]
    print(f"  Sublimation anomalous (nonice>0.5, |lat|<10): {len(subl_eq_high)}")

    # Overlap
    sput_set = set(eq_high_nonice)
    subl_set = set(subl_eq_high)
    overlap = sput_set & subl_set
    sput_only = sput_set - subl_set
    subl_only = subl_set - sput_set
    print(f"  Both runs: {len(overlap)}")
    print(f"  Sputtering only: {len(sput_only)}")
    print(f"  Sublimation only: {len(subl_only)}")

    if len(overlap) > 0:
        ol_list = list(overlap)[:10]
        print(f"\n  Shared anomalous faces (first 10):")
        for fi in ol_list:
            print(f"    Face {fi}: lat={lat[fi]:.2f}, lon={lon[fi]:.2f}, "
                  f"sput_nonice={nonice[fi]:.6f}, subl_nonice={subl_nonice[fi]:.6f}")

    if len(sput_only) > 0:
        so_list = list(sput_only)[:10]
        print(f"\n  Sputtering-only anomalous faces (first 10):")
        for fi in so_list:
            print(f"    Face {fi}: lat={lat[fi]:.2f}, lon={lon[fi]:.2f}, "
                  f"sput_nonice={nonice[fi]:.6f}, subl_nonice={subl_nonice[fi]:.6f}")

    # CRITICAL CHECK: Look at the very first epoch to see initial conditions
    print(f"\n{'='*80}")
    print("INITIAL STATE (epoch 0)")
    print(f"{'='*80}")
    nonice0 = read_bin(os.path.join(SPUT_DIR, f"nonice_concentration/nonice_concentration_0000.bin"))
    yield0 = read_bin(os.path.join(SPUT_DIR, f"sputtering_yield/sputtering_yield_0000.bin"))
    sput0 = read_bin(os.path.join(SPUT_DIR, f"sputtering_flux_kg_m2_s/sputtering_flux_kg_m2_s_0000.bin"))
    temp0 = read_bin(os.path.join(SPUT_DIR, f"temperature_mean_K/temperature_mean_K_0000.bin"))

    print(f"  At future-anomalous faces (nonice>{0.5} at epoch {EPOCH}):")
    print(f"    nonice_0: min={nonice0[eq_high_nonice].min():.6e}, max={nonice0[eq_high_nonice].max():.6e}, mean={nonice0[eq_high_nonice].mean():.6e}")
    print(f"    yield_0:  min={yield0[eq_high_nonice].min():.6e}, max={yield0[eq_high_nonice].max():.6e}, mean={yield0[eq_high_nonice].mean():.6e}")
    print(f"    sput_0:   min={sput0[eq_high_nonice].min():.6e}, max={sput0[eq_high_nonice].max():.6e}, mean={sput0[eq_high_nonice].mean():.6e}")
    print(f"    temp_0:   min={temp0[eq_high_nonice].min():.6e}, max={temp0[eq_high_nonice].max():.6e}, mean={temp0[eq_high_nonice].mean():.6e}")

    print(f"\n  At normal equatorial faces (nonice<0.01 at epoch {EPOCH}):")
    print(f"    nonice_0: min={nonice0[eq_low_nonice].min():.6e}, max={nonice0[eq_low_nonice].max():.6e}, mean={nonice0[eq_low_nonice].mean():.6e}")
    print(f"    yield_0:  min={yield0[eq_low_nonice].min():.6e}, max={yield0[eq_low_nonice].max():.6e}, mean={yield0[eq_low_nonice].mean():.6e}")
    print(f"    sput_0:   min={sput0[eq_low_nonice].min():.6e}, max={sput0[eq_low_nonice].max():.6e}, mean={sput0[eq_low_nonice].mean():.6e}")
    print(f"    temp_0:   min={temp0[eq_low_nonice].min():.6e}, max={temp0[eq_low_nonice].max():.6e}, mean={temp0[eq_low_nonice].mean():.6e}")

    # Check the early divergence: what epoch do these faces start to separate?
    print(f"\n{'='*80}")
    print("DIVERGENCE TIMELINE: When do anomalous faces separate?")
    print(f"{'='*80}")

    nonice_dir = os.path.join(SPUT_DIR, "nonice_concentration")
    files = sorted(os.listdir(nonice_dir))

    # Track the max nonice among eq_high_nonice and the median of eq_low_nonice
    print(f"{'epoch':>8s} {'anom_min':>12s} {'anom_max':>12s} {'anom_mean':>12s} {'norm_mean':>12s} {'max_diff':>12s}")
    for fname in files[::max(1, len(files)//20)]:
        epoch_str = fname.split('_')[-1].replace('.bin', '')
        data = read_bin(os.path.join(nonice_dir, fname))
        anom_vals = data[eq_high_nonice]
        norm_vals = data[eq_low_nonice]
        print(f"{epoch_str:>8s} {anom_vals.min():12.6e} {anom_vals.max():12.6e} "
              f"{anom_vals.mean():12.6e} {norm_vals.mean():12.6e} {(anom_vals.mean()-norm_vals.mean()):12.6e}")


if __name__ == '__main__':
    main()
