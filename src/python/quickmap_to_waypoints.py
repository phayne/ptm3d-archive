#!/usr/bin/env python3
"""Convert LROC Quickmap coordinate lists to Viewshed waypoint CSV."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-",
                        help="Input file (default: stdin)")
    parser.add_argument("-o", "--output", default="-",
                        help="Output file (default: stdout)")
    parser.add_argument("--height-lander", type=float, default=2.0,
                        help="Lander height in metres (default: 2.0)")
    parser.add_argument("--height-rover", type=float, default=1.0,
                        help="Rover height in metres (default: 1.0)")
    args = parser.parse_args()

    # Read input
    src = sys.stdin if args.input == "-" else open(args.input)
    coords = []
    for line in src:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            lat, lon = float(parts[0]), float(parts[1])
            coords.append((lat, lon))
    if src is not sys.stdin:
        src.close()

    if not coords:
        print("Error: no coordinates found.", file=sys.stderr)
        sys.exit(1)

    # Write output
    dst = sys.stdout if args.output == "-" else open(args.output, "w")
    dst.write("# Waypoint locations for PTM3D viewshed analysis\n")
    dst.write("# Format: label,latitude,longitude,height_meters\n")

    lat, lon = coords[0]
    dst.write(f"Lander,{lat:.5f},{lon:.5f},{args.height_lander:.1f}\n")

    for i, (lat, lon) in enumerate(coords[1:], start=1):
        dst.write(f"Rover{i:02d},{lat:.5f},{lon:.5f},{args.height_rover:.1f}\n")

    if dst is not sys.stdout:
        dst.close()


if __name__ == "__main__":
    main()
