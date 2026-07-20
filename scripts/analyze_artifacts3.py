#!/usr/bin/env python3
"""
Final diagnostic: check whether the anomalous faces are a physical tipping point
or a numerical artifact. Key test: do faces just ABOVE and BELOW the transition
show smooth gradients, or are there sharp discontinuities?
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
    temp_mean = read_bin(os.path.join(SPUT_DIR, f"temperature_mean_K/temperature_mean_K_{EPOCH}.bin"))
    temp_max = read_bin(os.path.join(SPUT_DIR, f"temperature_max_K/temperature_max_K_{EPOCH}.bin"))
    bond_albedo = read_bin(os.path.join(SPUT_DIR, f"bond_albedo/bond_albedo_{EPOCH}.bin"))
    sput_flux = read_bin(os.path.join(SPUT_DIR, f"sputtering_flux_kg_m2_s/sputtering_flux_kg_m2_s_{EPOCH}.bin"))
    ice_mass = read_bin(os.path.join(SPUT_DIR, f"ice_mass_kg_m2/ice_mass_kg_m2_{EPOCH}.bin"))
    emission = read_bin(os.path.join(SPUT_DIR, f"emission_flux_kg_m2_s/emission_flux_kg_m2_s_{EPOCH}.bin"))

    # Build adjacency
    max_vert = face_verts.max() + 1
    vert_to_faces = [[] for _ in range(max_vert)]
    for fi in range(len(face_verts)):
        for vi in face_verts[fi]:
            vert_to_faces[vi].append(fi)

    # Find transition zone: anomalous faces with normal neighbors
    anomalous = nonice > 0.5
    equatorial = np.abs(lat) < 10.0

    print("="*80)
    print("TRANSITION ZONE ANALYSIS")
    print("="*80)

    transition_pairs = []  # (anomalous_face, normal_neighbor)

    for fi in np.where(anomalous & equatorial)[0]:
        neighbors = set()
        for vi in face_verts[fi]:
            for nf in vert_to_faces[vi]:
                if nf != fi:
                    neighbors.add(nf)
        for nf in neighbors:
            if not anomalous[nf]:
                transition_pairs.append((fi, nf))

    print(f"Transition pairs (anomalous face touching normal face): {len(transition_pairs)}")

    # Look at a few of these pairs
    print(f"\n{'anom_face':>10s} {'norm_face':>10s} "
          f"{'a_lat':>7s} {'n_lat':>7s} {'a_lon':>8s} {'n_lon':>8s} "
          f"{'a_nonice':>10s} {'n_nonice':>10s} "
          f"{'a_albedo':>8s} {'n_albedo':>8s} "
          f"{'a_Tmean':>8s} {'n_Tmean':>8s} "
          f"{'a_Tmax':>7s} {'n_Tmax':>7s} "
          f"{'a_sput':>12s} {'n_sput':>12s}")

    # Show unique transition pairs, sorted by the contrast
    seen = set()
    unique_pairs = []
    for af, nf in transition_pairs:
        key = (min(af, nf), max(af, nf))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((af, nf))

    # Sort by temperature difference
    unique_pairs.sort(key=lambda p: abs(temp_mean[p[0]] - temp_mean[p[1]]), reverse=True)

    for af, nf in unique_pairs[:30]:
        print(f"{af:10d} {nf:10d} "
              f"{lat[af]:7.2f} {lat[nf]:7.2f} {lon[af]:8.2f} {lon[nf]:8.2f} "
              f"{nonice[af]:10.6f} {nonice[nf]:10.6f} "
              f"{bond_albedo[af]:8.4f} {bond_albedo[nf]:8.4f} "
              f"{temp_mean[af]:8.1f} {temp_mean[nf]:8.1f} "
              f"{temp_max[af]:7.1f} {temp_max[nf]:7.1f} "
              f"{sput_flux[af]:12.4e} {sput_flux[nf]:12.4e}")

    # KEY QUESTION: Is the initial temperature difference the trigger?
    # Read epoch 0 temperatures for these faces
    temp0 = read_bin(os.path.join(SPUT_DIR, f"temperature_mean_K/temperature_mean_K_0000.bin"))

    print(f"\n{'='*80}")
    print("INITIAL STATE OF TRANSITION PAIRS (epoch 0)")
    print(f"{'='*80}")
    print(f"{'anom_face':>10s} {'norm_face':>10s} "
          f"{'a_T0':>8s} {'n_T0':>8s} {'dT0':>8s} "
          f"{'a_Tfinal':>8s} {'n_Tfinal':>8s} {'dTfinal':>8s}")

    for af, nf in unique_pairs[:30]:
        dT0 = temp0[af] - temp0[nf]
        dTf = temp_mean[af] - temp_mean[nf]
        print(f"{af:10d} {nf:10d} "
              f"{temp0[af]:8.2f} {temp0[nf]:8.2f} {dT0:8.4f} "
              f"{temp_mean[af]:8.2f} {temp_mean[nf]:8.2f} {dTf:8.2f}")

    # What is the temperature range at the tipping point?
    # Check if there's a critical temperature that separates depleted from non-depleted
    eq_mask = np.abs(lat) < 10.0
    eq_temp0 = temp0[eq_mask]
    eq_nonice = nonice[eq_mask]

    # Find the boundary: max T of normal faces, min T of anomalous faces
    normal_max_T0 = temp0[eq_mask & ~anomalous].max() if np.any(eq_mask & ~anomalous) else 0
    anom_min_T0 = temp0[eq_mask & anomalous].min() if np.any(eq_mask & anomalous) else 999

    print(f"\n{'='*80}")
    print("CRITICAL TEMPERATURE ANALYSIS (equatorial faces)")
    print(f"{'='*80}")
    print(f"  Max initial T of normal faces: {normal_max_T0:.4f} K")
    print(f"  Min initial T of anomalous faces: {anom_min_T0:.4f} K")
    print(f"  Gap: {anom_min_T0 - normal_max_T0:.4f} K")
    print(f"  Overlap: {'YES' if anom_min_T0 < normal_max_T0 else 'NO'}")

    # Check distribution of initial temperatures
    print(f"\n  Initial temperature distribution (equatorial):")
    bins = np.linspace(105, 112, 30)
    hist_normal, edges = np.histogram(temp0[eq_mask & ~anomalous], bins=bins)
    hist_anom, _ = np.histogram(temp0[eq_mask & anomalous], bins=bins)
    print(f"  {'T range':>12s} {'normal':>8s} {'anomalous':>10s}")
    for i in range(len(hist_normal)):
        if hist_normal[i] > 0 or hist_anom[i] > 0:
            print(f"  [{edges[i]:6.2f}, {edges[i+1]:6.2f}) {hist_normal[i]:8d} {hist_anom[i]:10d}")

    # Check the sublimation vs sputtering comparison
    print(f"\n{'='*80}")
    print("SUBLIMATION vs SPUTTERING COMPARISON AT EQUATOR")
    print(f"{'='*80}")
    # At the anomalous faces, what dominates the ice loss?
    eq_anom = eq_mask & anomalous
    eq_norm = eq_mask & ~anomalous

    print(f"  At anomalous equatorial faces (final epoch):")
    print(f"    Emission (sublim) flux: {emission[eq_anom].mean():.4e} kg/m2/s")
    print(f"    Sputtering flux:        {sput_flux[eq_anom].mean():.4e} kg/m2/s")
    print(f"    Ratio (sublim/sput):    {emission[eq_anom].mean() / sput_flux[eq_anom].mean():.1f}x")

    print(f"\n  At normal equatorial faces (final epoch):")
    print(f"    Emission (sublim) flux: {emission[eq_norm].mean():.4e} kg/m2/s")
    print(f"    Sputtering flux:        {sput_flux[eq_norm].mean():.4e} kg/m2/s")
    print(f"    Ratio (sublim/sput):    {emission[eq_norm].mean() / max(sput_flux[eq_norm].mean(), 1e-30):.1f}x")

    # Also read base sublimation flux from epoch 0 emission
    emission0 = read_bin(os.path.join(SPUT_DIR, f"emission_flux_kg_m2_s/emission_flux_kg_m2_s_0000.bin"))
    sput0 = read_bin(os.path.join(SPUT_DIR, f"sputtering_flux_kg_m2_s/sputtering_flux_kg_m2_s_0000.bin"))

    print(f"\n  At future-anomalous equatorial faces (epoch 0):")
    print(f"    Emission (sublim) flux: {emission0[eq_anom].mean():.4e} kg/m2/s")
    print(f"    Sputtering flux:        {sput0[eq_anom].mean():.4e} kg/m2/s")
    print(f"    Ratio (sublim/sput):    {emission0[eq_anom].mean() / max(sput0[eq_anom].mean(), 1e-30):.1f}x")

    print(f"\n  At normal equatorial faces (epoch 0):")
    print(f"    Emission (sublim) flux: {emission0[eq_norm].mean():.4e} kg/m2/s")
    print(f"    Sputtering flux:        {sput0[eq_norm].mean():.4e} kg/m2/s")
    print(f"    Ratio (sublim/sput):    {emission0[eq_norm].mean() / max(sput0[eq_norm].mean(), 1e-30):.1f}x")

    # Now check the sublimation-only run to compare with sputtering run
    print(f"\n{'='*80}")
    print("SUBLIMATION-ONLY RUN: SAME ANALYSIS")
    print(f"{'='*80}")
    subl_nonice = read_bin(os.path.join(SUBL_DIR, f"nonice_concentration/nonice_concentration_{EPOCH}.bin"))
    subl_temp = read_bin(os.path.join(SUBL_DIR, f"temperature_mean_K/temperature_mean_K_{EPOCH}.bin"))
    subl_temp0 = read_bin(os.path.join(SUBL_DIR, f"temperature_mean_K/temperature_mean_K_0000.bin"))
    subl_emission = read_bin(os.path.join(SUBL_DIR, f"emission_flux_kg_m2_s/emission_flux_kg_m2_s_{EPOCH}.bin"))

    subl_anomalous = subl_nonice > 0.5
    subl_eq_anom = eq_mask & subl_anomalous
    subl_eq_norm = eq_mask & ~subl_anomalous

    print(f"  Anomalous equatorial faces in sublim-only run: {subl_eq_anom.sum()}")
    print(f"  Anomalous equatorial faces in sputtering run: {eq_anom.sum()}")

    # Compare the initial temperatures of sublim-only anomalous faces
    if subl_eq_anom.sum() > 0:
        print(f"\n  Sublim-only anomalous faces:")
        print(f"    Initial T range: {subl_temp0[subl_eq_anom].min():.4f} - {subl_temp0[subl_eq_anom].max():.4f} K")
        print(f"    Final T range:   {subl_temp[subl_eq_anom].min():.4f} - {subl_temp[subl_eq_anom].max():.4f} K")

    # What fraction of sputtering-only anomalous faces have LOWER initial T than sublim-only anomalous?
    sput_only_anom = eq_mask & anomalous & ~subl_anomalous
    print(f"\n  Faces anomalous ONLY with sputtering (not sublimation): {sput_only_anom.sum()}")
    if sput_only_anom.sum() > 0:
        print(f"    Initial T range: {temp0[sput_only_anom].min():.4f} - {temp0[sput_only_anom].max():.4f} K")
        print(f"    Initial T mean:  {temp0[sput_only_anom].mean():.4f} K")
    if subl_eq_anom.sum() > 0:
        print(f"    (vs sublim anomalous initial T mean: {subl_temp0[subl_eq_anom].mean():.4f} K)")

    # THE KEY: Is sputtering the "straw that breaks the camel's back"?
    # Faces that sublim-only run keeps stable, but sputtering tips them over?
    print(f"\n{'='*80}")
    print("TIPPING POINT MECHANISM SUMMARY")
    print(f"{'='*80}")

    # Compare the critical temperature
    if subl_eq_anom.sum() > 0:
        subl_T_thresh = subl_temp0[subl_eq_anom].min()
        print(f"  Sublimation-only tipping T:  {subl_T_thresh:.4f} K (min initial T of depleted faces)")

    sput_T_thresh = temp0[eq_mask & anomalous].min()
    print(f"  Sputtering run tipping T:    {sput_T_thresh:.4f} K (min initial T of depleted faces)")
    print(f"  Normal face max initial T:   {normal_max_T0:.4f} K (max initial T of surviving faces)")

    # How many faces fall in between the two thresholds?
    if subl_eq_anom.sum() > 0:
        in_gap = eq_mask & (temp0 >= sput_T_thresh) & (temp0 < subl_T_thresh) & ~anomalous
        print(f"\n  Faces in the tipping gap (would deplete with sputtering but not without): ~{in_gap.sum()}")

    # Net mass balance at the tipping point
    # At the threshold faces, is sputtering a significant fraction of total loss?
    border_faces = eq_mask & (nonice > 0.4) & (nonice < 0.6)
    if border_faces.sum() > 0:
        print(f"\n  Faces at transition (0.4 < nonice < 0.6): {border_faces.sum()}")
        print(f"    Emission flux: {emission[border_faces].mean():.4e}")
        print(f"    Sputtering flux: {sput_flux[border_faces].mean():.4e}")
    else:
        # No faces at transition - it's a sharp binary switch
        print(f"\n  NO faces in transition zone (0.4 < nonice < 0.6)")
        print(f"  -> The transition is SHARP/BINARY, not gradual")
        print(f"  -> This is consistent with a positive feedback runaway")


if __name__ == '__main__':
    main()
