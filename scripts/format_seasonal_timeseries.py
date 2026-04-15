#!/usr/bin/env python3
"""
Phase 2 Post-Processing: Combine per-lunation time series into a single table.

Reads the 25 per-lunation location_timeseries.csv files and concatenates them
chronologically into the final output table with UTC times, lat/lon, and
per-location illumination + temperature columns.

Usage:
    python3 format_seasonal_timeseries.py \
        --thermal-dir output/thermal \
        --locations configs/amundsen_target_locations.csv \
        --num-lunations 25 \
        --output output/thermal/amundsen_seasonal_timeseries.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def read_locations(filepath):
    """Read locations CSV (name,lat,lon), skipping comment lines."""
    locations = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                name = parts[0].strip()
                lat = float(parts[1])
                lon = float(parts[2])
                locations.append({'name': name, 'lat': lat, 'lon': lon})
    return locations


def read_timeseries_csv(filepath):
    """Read a location_timeseries.csv file. Returns header list and data rows."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def main():
    parser = argparse.ArgumentParser(
        description='Combine per-lunation time series into final output table')
    parser.add_argument('--thermal-dir', required=True,
                        help='Base directory of Phase 2 thermal output')
    parser.add_argument('--locations', required=True,
                        help='Locations CSV file (from Phase 1)')
    parser.add_argument('--num-lunations', type=int, default=25,
                        help='Number of lunation directories')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: thermal-dir/amundsen_seasonal_timeseries.csv)')
    args = parser.parse_args()

    thermal_dir = Path(args.thermal_dir)
    if args.output is None:
        args.output = str(thermal_dir / 'amundsen_seasonal_timeseries.csv')

    # Read locations for metadata header
    locations = read_locations(args.locations)
    print(f"Locations: {len(locations)}")

    # Collect all lunation CSV files
    all_rows = []
    header = None
    skipped = 0

    for i in range(args.num_lunations):
        csv_path = thermal_dir / f'lunation_{i:02d}' / 'location_timeseries.csv'
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping lunation {i}")
            skipped += 1
            continue

        h, rows = read_timeseries_csv(csv_path)
        if header is None:
            header = h
        elif h != header:
            print(f"  WARNING: Lunation {i} header differs from lunation 0")

        all_rows.extend(rows)
        print(f"  Lunation {i:02d}: {len(rows)} timesteps")

    if not all_rows:
        print("ERROR: No data found in any lunation directory")
        return

    print(f"\nTotal timesteps: {len(all_rows)} ({skipped} lunations skipped)")

    # Write combined output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Metadata header
        f.write(f"# Amundsen Crater Seasonal Shadow Time Series\n")
        f.write(f"# Period: April 2026 - April 2028 ({args.num_lunations} lunations)\n")
        f.write(f"# {len(locations)} locations, {len(all_rows)} total timesteps\n")
        f.write(f"#\n")
        for i, loc in enumerate(locations):
            f.write(f"# Location {i+1}: {loc['name']}, "
                    f"lat={loc['lat']:.6f}, lon={loc['lon']:.6f}\n")
        f.write(f"#\n")

        # CSV header
        writer = csv.writer(f)
        writer.writerow(header)

        # Data rows (already chronologically ordered from sequential lunations)
        for row in all_rows:
            writer.writerow(row)

    print(f"\nOutput written to: {output_path}")
    print(f"  Header columns: {', '.join(header[:5])}{'...' if len(header) > 5 else ''}")

    # Quick summary statistics
    if header and len(all_rows) > 0:
        # Find temperature columns
        temp_cols = [i for i, h in enumerate(header) if h.endswith('_Ts')]
        if temp_cols:
            print(f"\nTemperature summary (from {len(temp_cols)} locations):")
            for ci in temp_cols[:5]:  # Show first 5
                col_name = header[ci]
                vals = []
                for row in all_rows:
                    try:
                        v = float(row[ci])
                        if v > 0:
                            vals.append(v)
                    except (ValueError, IndexError):
                        pass
                if vals:
                    print(f"  {col_name}: min={min(vals):.1f} K, "
                          f"max={max(vals):.1f} K, mean={np.mean(vals):.1f} K")

        # Find indirect flux columns
        indirect_cols = [i for i, h in enumerate(header) if h.endswith('_IndirectFlux')]
        if indirect_cols:
            print(f"\nIndirect flux summary:")
            for ci in indirect_cols[:5]:
                col_name = header[ci]
                vals = []
                for row in all_rows:
                    try:
                        vals.append(float(row[ci]))
                    except (ValueError, IndexError):
                        pass
                if vals:
                    nonzero = [v for v in vals if v > 0]
                    print(f"  {col_name}: max={max(vals):.4f} W/m2, "
                          f"mean={np.mean(vals):.4f} W/m2, "
                          f"nonzero={len(nonzero)}/{len(vals)} timesteps")


if __name__ == '__main__':
    main()
