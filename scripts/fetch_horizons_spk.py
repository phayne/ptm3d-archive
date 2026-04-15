#!/usr/bin/env python3
"""
Fetch spacecraft trajectory from JPL HORIZONS and write a Moon-centered SPICE SPK.

This script:
1. Queries HORIZONS for Earth-centered state vectors of a spacecraft
2. Queries HORIZONS for Earth-centered state vectors of the Moon
3. Subtracts Moon state to get Moon-centered spacecraft state
4. Writes a SPICE SPK (Type 9, Lagrange interpolation) in the J2000 frame

Usage:
    python fetch_horizons_spk.py [options]

Example (Artemis II lunar flyby):
    python fetch_horizons_spk.py \\
        --spacecraft -1024 \\
        --start "2026-04-04" --stop "2026-04-08" \\
        --step "1 min" \\
        --output data/spice/artemis2_moon.bsp

Requirements:
    pip install spiceypy requests numpy
"""

import argparse
import sys
import os
import re
import numpy as np

try:
    import requests
except ImportError:
    sys.exit("Error: 'requests' package required. Install with: pip install requests")

try:
    import spiceypy as spice
except ImportError:
    sys.exit("Error: 'spiceypy' package required. Install with: pip install spiceypy")


HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Default SPICE kernels path
DEFAULT_KERNELS = "/Users/paha3326/software/cspice/data"


def fetch_horizons_vectors(command, center, start_time, stop_time, step_size):
    """
    Fetch state vectors from HORIZONS API.

    Returns list of dicts with keys: jd_tdb, utc, x, y, z, vx, vy, vz
    All positions in km, velocities in km/s, ICRF frame.
    """
    params = {
        'format': 'text',
        'COMMAND': command,
        'OBJ_DATA': 'NO',
        'MAKE_EPHEM': 'YES',
        'EPHEM_TYPE': 'VECTORS',
        'CENTER': center,
        'START_TIME': f"'{start_time}'",
        'STOP_TIME': f"'{stop_time}'",
        'STEP_SIZE': f"'{step_size}'",
        'VEC_TABLE': '2',        # Position + velocity
        'OUT_UNITS': 'KM-S',
        'CSV_FORMAT': 'NO',
        'REF_PLANE': 'FRAME',    # ICRF/J2000 reference plane
        'VEC_LABELS': 'YES',
    }

    print(f"  Querying HORIZONS: COMMAND={command}, CENTER={center}")
    print(f"  Time range: {start_time} to {stop_time}, step={step_size}")

    r = requests.get(HORIZONS_API, params=params, timeout=120)
    if r.status_code != 200:
        print(f"  HTTP error {r.status_code}: {r.text[:200]}")
        return None

    text = r.text

    # Check for API errors
    if 'API ERROR' in text or 'INPUT ERROR' in text:
        print(f"  HORIZONS error:\n{text[:500]}")
        return None

    # Parse the response
    lines = text.split('\n')
    records = []
    in_data = False
    current_record = {}

    for line in lines:
        if '$$SOE' in line:
            in_data = True
            continue
        if '$$EOE' in line:
            break
        if not in_data:
            continue

        line = line.strip()
        if not line:
            continue

        # First line of record: JDTDB = A.D. date TDB
        jd_match = re.match(r'(\d+\.\d+)\s*=\s*A\.D\.\s+(\S+\s+\S+)\s+TDB', line)
        if jd_match:
            if current_record:
                records.append(current_record)
            current_record = {
                'jd_tdb': float(jd_match.group(1)),
                'utc': jd_match.group(2),
            }
            continue

        # Position line: X = ... Y = ... Z = ...
        pos_match = re.match(r'\s*X\s*=\s*([^\s]+)\s+Y\s*=\s*([^\s]+)\s+Z\s*=\s*([^\s]+)', line)
        if pos_match:
            current_record['x'] = float(pos_match.group(1))
            current_record['y'] = float(pos_match.group(2))
            current_record['z'] = float(pos_match.group(3))
            continue

        # Velocity line: VX= ... VY= ... VZ= ...
        vel_match = re.match(r'\s*VX\s*=\s*([^\s]+)\s+VY\s*=\s*([^\s]+)\s+VZ\s*=\s*([^\s]+)', line)
        if vel_match:
            current_record['vx'] = float(vel_match.group(1))
            current_record['vy'] = float(vel_match.group(2))
            current_record['vz'] = float(vel_match.group(3))
            continue

    # Don't forget the last record
    if current_record and 'x' in current_record:
        records.append(current_record)

    print(f"  Retrieved {len(records)} state vectors")
    return records


def jd_tdb_to_et(jd_tdb):
    """Convert Julian Date TDB to SPICE ET (seconds past J2000)."""
    # J2000 epoch = JD 2451545.0 TDB
    return (jd_tdb - 2451545.0) * 86400.0


def write_spk(output_path, naif_id, center_id, records, frame='J2000', degree=7):
    """
    Write a SPICE SPK Type 9 (Lagrange interpolation) kernel.

    Args:
        output_path: Path to write the SPK file
        naif_id: NAIF ID of the spacecraft (e.g., -1024)
        center_id: NAIF ID of the center body (e.g., 301 for Moon)
        records: List of dicts with jd_tdb, x, y, z, vx, vy, vz
        frame: Reference frame (default: J2000)
        degree: Lagrange interpolation degree (must be odd, default: 7)
    """
    n = len(records)
    if n < degree + 1:
        print(f"  Warning: only {n} records, reducing degree to {n-1}")
        degree = n - 1
        if degree % 2 == 0:
            degree -= 1
        if degree < 1:
            print("  Error: not enough data points for SPK")
            return False

    # Build arrays
    epochs = np.zeros(n)
    states = np.zeros((n, 6))

    for i, rec in enumerate(records):
        epochs[i] = jd_tdb_to_et(rec['jd_tdb'])
        states[i, 0] = rec['x']
        states[i, 1] = rec['y']
        states[i, 2] = rec['z']
        states[i, 3] = rec['vx']
        states[i, 4] = rec['vy']
        states[i, 5] = rec['vz']

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Remove existing file if present
    if os.path.exists(output_path):
        os.remove(output_path)

    # Open SPK for writing
    handle = spice.spkopn(output_path, f'Spacecraft {naif_id} wrt {center_id}', 1000)

    # Write Type 9 segment (Lagrange interpolation)
    # spkw09(handle, body, center, frame, first, last, segid, degree, n, states, epochs)
    first_et = epochs[0]
    last_et = epochs[-1]
    segid = f'SC{naif_id}_CENTER{center_id}'

    # spkw09 expects states as N x 6 array
    states_list = states.tolist()
    epoch_list = epochs.tolist()

    spice.spkw09(
        handle,
        naif_id,
        center_id,
        frame,
        first_et,
        last_et,
        segid,
        degree,
        n,
        states_list,
        epoch_list
    )

    spice.spkcls(handle)

    file_size = os.path.getsize(output_path)
    print(f"  Wrote SPK: {output_path} ({file_size/1024:.1f} KB)")
    print(f"  Body: {naif_id}, Center: {center_id}, Frame: {frame}")
    print(f"  ET range: [{first_et:.3f}, {last_et:.3f}]")
    print(f"  Records: {n}, Degree: {degree}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Fetch spacecraft trajectory from HORIZONS and write Moon-centered SPK')
    parser.add_argument('--spacecraft', default='-1024',
                        help='NAIF spacecraft ID (default: -1024 for Artemis II)')
    parser.add_argument('--naif-id', type=int, default=None,
                        help='NAIF ID for SPK (default: same as --spacecraft)')
    parser.add_argument('--start', default='2026-04-04',
                        help='Start time UTC (default: 2026-04-04)')
    parser.add_argument('--stop', default='2026-04-08',
                        help='Stop time UTC (default: 2026-04-08)')
    parser.add_argument('--step', default='1 min',
                        help='Time step (default: "1 min")')
    parser.add_argument('--center-body', default='301',
                        help='NAIF ID of center body for SPK (default: 301 = Moon)')
    parser.add_argument('--output', default='data/spice/artemis2_moon.bsp',
                        help='Output SPK file path')
    parser.add_argument('--kernels-path', default=DEFAULT_KERNELS,
                        help='Path to SPICE kernels directory')
    parser.add_argument('--degree', type=int, default=7,
                        help='Lagrange interpolation degree (default: 7)')
    parser.add_argument('--also-write-earth', action='store_true',
                        help='Also write an Earth-centered SPK')
    parser.add_argument('--text-output', default=None,
                        help='Also write vectors to a text file (for trajectory_reader)')
    args = parser.parse_args()

    naif_id = args.naif_id if args.naif_id is not None else int(args.spacecraft)
    center_id = int(args.center_body)

    # Load SPICE kernels for time conversion
    lsk = os.path.join(args.kernels_path, 'lsk', 'naif0012.tls')
    if os.path.exists(lsk):
        spice.furnsh(lsk)
        print(f"Loaded leapseconds: {lsk}")
    else:
        print(f"Warning: leapseconds kernel not found at {lsk}")

    # =========================================================================
    # Step 1: Fetch spacecraft vectors (Earth-centered)
    # =========================================================================
    print(f"\n--- Fetching spacecraft {args.spacecraft} (Earth-centered) ---")
    sc_records = fetch_horizons_vectors(
        command=args.spacecraft,
        center='500@399',  # Earth geocenter
        start_time=args.start,
        stop_time=args.stop,
        step_size=args.step
    )
    if not sc_records:
        sys.exit("Failed to fetch spacecraft data from HORIZONS")

    # =========================================================================
    # Step 2: Fetch Moon vectors (Earth-centered) at same times
    # =========================================================================
    print(f"\n--- Fetching Moon (Earth-centered) ---")
    moon_records = fetch_horizons_vectors(
        command='301',     # Moon
        center='500@399',  # Earth geocenter
        start_time=args.start,
        stop_time=args.stop,
        step_size=args.step
    )
    if not moon_records:
        sys.exit("Failed to fetch Moon data from HORIZONS")

    # Verify same number of records
    if len(sc_records) != len(moon_records):
        print(f"Warning: spacecraft ({len(sc_records)}) and Moon ({len(moon_records)}) "
              f"record counts differ. Using minimum.")

    n = min(len(sc_records), len(moon_records))

    # =========================================================================
    # Step 3: Compute Moon-centered spacecraft state
    # =========================================================================
    print(f"\n--- Computing Moon-centered state vectors ---")
    moon_centered = []
    for i in range(n):
        sc = sc_records[i]
        moon = moon_records[i]

        # Verify times match
        dt = abs(sc['jd_tdb'] - moon['jd_tdb'])
        if dt > 1e-6:  # > 0.08 seconds
            print(f"  Warning: time mismatch at record {i}: "
                  f"SC={sc['jd_tdb']:.6f}, Moon={moon['jd_tdb']:.6f}")

        rec = {
            'jd_tdb': sc['jd_tdb'],
            'utc': sc['utc'],
            'x': sc['x'] - moon['x'],
            'y': sc['y'] - moon['y'],
            'z': sc['z'] - moon['z'],
            'vx': sc['vx'] - moon['vx'],
            'vy': sc['vy'] - moon['vy'],
            'vz': sc['vz'] - moon['vz'],
        }
        moon_centered.append(rec)

    # Print some diagnostics
    distances = [np.sqrt(r['x']**2 + r['y']**2 + r['z']**2) for r in moon_centered]
    print(f"  Records: {n}")
    print(f"  Distance range: {min(distances):.1f} - {max(distances):.1f} km")
    print(f"  Closest approach: {min(distances):.1f} km "
          f"(alt ~{min(distances) - 1737.4:.1f} km above surface)")

    # Find closest approach time
    min_idx = np.argmin(distances)
    print(f"  Closest approach time: {moon_centered[min_idx]['utc']} TDB")

    # =========================================================================
    # Step 4: Write Moon-centered SPK
    # =========================================================================
    print(f"\n--- Writing Moon-centered SPK ---")
    success = write_spk(
        output_path=args.output,
        naif_id=naif_id,
        center_id=center_id,
        records=moon_centered,
        frame='J2000',
        degree=args.degree
    )
    if not success:
        sys.exit("Failed to write SPK")

    # =========================================================================
    # Optional: Write Earth-centered SPK
    # =========================================================================
    if args.also_write_earth:
        earth_spk = args.output.replace('.bsp', '_earth.bsp')
        print(f"\n--- Writing Earth-centered SPK ---")
        write_spk(
            output_path=earth_spk,
            naif_id=naif_id,
            center_id=399,  # Earth
            records=sc_records[:n],
            frame='J2000',
            degree=args.degree
        )

    # =========================================================================
    # Optional: Write text file for trajectory_reader
    # =========================================================================
    if args.text_output:
        print(f"\n--- Writing text vectors ---")
        with open(args.text_output, 'w') as f:
            f.write("# Moon-centered state vectors for spacecraft %s\n" % args.spacecraft)
            f.write("# Frame: J2000 (ICRF)\n")
            f.write("# Units: km, km/s\n")
            f.write("# Columns: ET  X  Y  Z  VX  VY  VZ\n")
            f.write("$$SOE\n")
            for rec in moon_centered:
                et = jd_tdb_to_et(rec['jd_tdb'])
                f.write(f"{et:.6f}  {rec['x']:.6f}  {rec['y']:.6f}  {rec['z']:.6f}  "
                        f"{rec['vx']:.9f}  {rec['vy']:.9f}  {rec['vz']:.9f}\n")
            f.write("$$EOE\n")
        print(f"  Wrote: {args.text_output}")

    # =========================================================================
    # Verification: read back and check
    # =========================================================================
    print(f"\n--- Verification ---")
    try:
        spice.furnsh(args.output)
        mid_et = jd_tdb_to_et(moon_centered[n//2]['jd_tdb'])
        state, lt = spice.spkez(naif_id, mid_et, 'J2000', 'NONE', center_id)
        expected = moon_centered[n//2]
        dx = state[0] - expected['x']
        dy = state[1] - expected['y']
        dz = state[2] - expected['z']
        err = np.sqrt(dx**2 + dy**2 + dz**2)
        print(f"  Mid-point position error: {err:.6f} km ({err*1000:.3f} m)")
        print(f"  SPK readback: X={state[0]:.3f}, Y={state[1]:.3f}, Z={state[2]:.3f} km")
        if err < 1.0:
            print("  PASS: SPK readback matches input")
        else:
            print("  WARNING: Large position error in SPK readback")
        spice.unload(args.output)
    except Exception as e:
        print(f"  Verification failed: {e}")

    print(f"\n=== Done ===")
    print(f"SPK file: {args.output}")
    print(f"\nTo use with ptm_camera_sim:")
    print(f"  --trajectory {args.output} --traj-format spk --traj-spacecraft {naif_id}")


if __name__ == '__main__':
    main()
