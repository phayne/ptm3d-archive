#!/usr/bin/env python3
"""
Query and download ALMA Europa observations from the Science Archive.

Datasets:
  - Trumbo et al. (2018): Project 2015.1.01302.S, Band 6 (233 GHz, 1.3 mm)
  - Thelen et al. (2024): Project 2016.1.00691.S, Bands 3/6/7

Usage:
  python download_alma_europa.py --query          # List available data
  python download_alma_europa.py --download       # Download FITS images
  python download_alma_europa.py --download-ms    # Download calibrated measurement sets
"""

import argparse
import os
import sys
from pathlib import Path

# Project root (two levels up from scripts/alma/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "alma"

PROJECTS = {
    "trumbo2018": {
        "code": "2015.1.01302.S",
        "reference": "Trumbo et al. (2018) AJ 156, 161",
        "bands": ["Band 6 (233 GHz, 1.3 mm)"],
        "n_obs": 4,
        "data_dir": DATA_DIR / "trumbo2018",
    },
    "thelen2024": {
        "code": "2016.1.00691.S",
        "reference": "Thelen et al. (2024) PSJ 5, 42",
        "bands": ["Band 3 (97.5 GHz, 3.05 mm)",
                  "Band 6 (233 GHz, 1.25 mm)",
                  "Band 7 (343.5 GHz, 0.88 mm)"],
        "n_obs": 8,
        "data_dir": DATA_DIR / "thelen2024",
    },
}


def query_archive(project_code, target_filter="Europa"):
    """Query the ALMA Science Archive for a given project code.

    Parameters
    ----------
    project_code : str
        ALMA project code (e.g., '2015.1.01302.S').
    target_filter : str or None
        If set, filter results to this target name only.
        Thelen's project includes Callisto and Ganymede calibrators.
    """
    from astroquery.alma import Alma

    print(f"\n{'='*70}")
    print(f"Querying ALMA archive for project: {project_code}")
    print(f"{'='*70}")

    alma = Alma()
    # Use the North American mirror (NRAO)
    alma.archive_url = "https://almascience.nrao.edu"

    result = alma.query(payload={"project_code": project_code})

    if result is None or len(result) == 0:
        print(f"  No results found for {project_code}")
        return None

    # Filter by target name if requested
    if target_filter and "target_name" in result.colnames:
        n_total = len(result)
        mask = [target_filter.lower() in str(row["target_name"]).lower()
                for row in result]
        result = result[mask]
        n_filtered = len(result)
        if n_filtered < n_total:
            print(f"  Filtered to target '{target_filter}': "
                  f"{n_filtered}/{n_total} observations")

    print(f"\n  Found {len(result)} observation(s):")
    print(f"  {'─'*60}")

    # Print key columns
    cols_to_show = [
        "target_name", "s_ra", "s_dec", "frequency",
        "bandwidth", "t_min", "t_max", "member_ous_uid",
    ]
    available_cols = [c for c in cols_to_show if c in result.colnames]

    for i, row in enumerate(result):
        print(f"\n  Observation {i+1}:")
        for col in available_cols:
            val = row[col]
            print(f"    {col:20s}: {val}")

    return result


def list_data_products(result):
    """List downloadable data products for query results."""
    from astroquery.alma import Alma

    alma = Alma()
    alma.archive_url = "https://almascience.nrao.edu"

    # Convert to plain str to avoid np.str_ issues with astroquery internals
    uids = sorted(set(str(u) for u in result["member_ous_uid"]))
    print(f"\n  Listing data products for {len(uids)} member OUS UID(s)...")

    all_products = []
    for uid in uids:
        print(f"\n  UID: {uid}")
        try:
            data_info = alma.get_data_info(uid, expand_tarfiles=True)
            if data_info is not None and len(data_info) > 0:
                # Print available columns for debugging
                print(f"    Columns: {data_info.colnames}")

                # Find the URL column — name varies by astroquery version
                url_col = None
                for candidate in ["access_url", "URL", "url", "link"]:
                    if candidate in data_info.colnames:
                        url_col = candidate
                        break
                if url_col is None:
                    print(f"    WARNING: No URL column found in data_info")
                    print(f"    First row: {data_info[0]}")
                    continue

                # Find the size column (may not exist)
                size_col = None
                for candidate in ["content_length", "size", "filesize",
                                  "content_size"]:
                    if candidate in data_info.colnames:
                        size_col = candidate
                        break

                # Categorize files by type
                # We distinguish continuum images (what we need) from
                # spectral cubes, masks, and other pipeline products
                fits_continuum = []   # Continuum pbcor images (we want these)
                fits_pb = []          # Primary beam images
                fits_other = []       # Cubes, masks, other FITS
                ms_files = []
                other_files = []
                for row in data_info:
                    url = str(row[url_col])
                    size = str(row[size_col]) if size_col else "?"
                    url_lower = url.lower()
                    fname = url.split("/")[-1]

                    if ".fits" not in url_lower:
                        if ".ms" in url_lower or "calibrated" in url_lower:
                            ms_files.append((url, size))
                        else:
                            other_files.append((url, size))
                        continue

                    # Classify FITS files
                    fname_lower = fname.lower()
                    # Skip calibrator images (bandpass, phase cal, check src)
                    is_calibrator = any(x in fname_lower for x in
                                        ["_bp.", "_ph.", "_chk."])
                    if ("cont" in fname_lower and "pbcor" in fname_lower
                            and not is_calibrator):
                        # Continuum, primary-beam corrected — what we want
                        fits_continuum.append((url, size))
                    elif ".pb." in fname_lower or "flux" in fname_lower:
                        fits_pb.append((url, size))
                    else:
                        fits_other.append((url, size))

                # Report
                print(f"    Continuum pbcor images: {len(fits_continuum)}")
                for url, size in fits_continuum:
                    fname = url.split("/")[-1]
                    sz_mb = int(size) / 1e6 if size != "?" else "?"
                    print(f"      {fname} ({sz_mb:.1f} MB)")
                print(f"    Primary beam / flux: {len(fits_pb)}")
                print(f"    Other FITS (cubes, masks): {len(fits_other)}")
                print(f"    Measurement sets: {len(ms_files)}")
                print(f"    Other files: {len(other_files)}")

                all_products.append({
                    "uid": uid,
                    "fits_continuum": fits_continuum,
                    "fits_pb": fits_pb,
                    "fits_other": fits_other,
                    "fits": fits_continuum + fits_pb + fits_other,  # all FITS
                    "ms": ms_files,
                    "other": other_files,
                    "data_info": data_info,
                })
        except Exception as e:
            import traceback
            print(f"    Error: {e}")
            traceback.print_exc()

    return all_products


def download_fits(products, dest_dir, continuum_only=True):
    """Download FITS images from the archive.

    Parameters
    ----------
    products : list
        Product list from list_data_products().
    dest_dir : str or Path
        Destination directory.
    continuum_only : bool
        If True (default), download only continuum primary-beam-corrected
        images. If False, download all FITS files.
    """
    from astroquery.alma import Alma

    alma = Alma()
    alma.archive_url = "https://almascience.nrao.edu"

    fits_dir = Path(dest_dir) / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)

    all_fits_urls = []
    for prod in products:
        if continuum_only:
            urls = [url for url, _ in prod.get("fits_continuum", prod["fits"])]
        else:
            urls = [url for url, _ in prod["fits"]]
        all_fits_urls.extend(urls)

    if not all_fits_urls:
        print("  No FITS files found to download.")
        print("  Pipeline products may not be available for this project.")
        print("  Try --download-ms to get calibrated measurement sets instead.")
        return []

    kind = "continuum pbcor" if continuum_only else "all"
    print(f"\n  Downloading {len(all_fits_urls)} {kind} FITS file(s) to {fits_dir}/")

    downloaded = []
    for url in all_fits_urls:
        fname = url.split("/")[-1]
        dest = fits_dir / fname
        if dest.exists():
            print(f"    {fname}: already exists, skipping")
            downloaded.append(str(dest))
            continue

        print(f"    Downloading {fname}...")
        try:
            files = alma.download_files([url], savedir=str(fits_dir))
            downloaded.extend(files)
        except Exception as e:
            print(f"    Error downloading {fname}: {e}")

    return downloaded


def download_tarballs(result, dest_dir):
    """Download raw data tarballs (including calibrated MS) from the archive."""
    from astroquery.alma import Alma

    alma = Alma()
    alma.archive_url = "https://almascience.nrao.edu"

    raw_dir = Path(dest_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    uids = sorted(set(str(u) for u in result["member_ous_uid"]))
    print(f"\n  Downloading data for {len(uids)} UID(s) to {raw_dir}/")
    print("  WARNING: This may download very large files (10-50 GB each)!")

    for uid in uids:
        print(f"\n  Downloading UID: {uid}")
        try:
            files = alma.retrieve_data_from_uid(uid, cache=True)
            for f in files:
                print(f"    Downloaded: {f}")
        except Exception as e:
            print(f"    Error: {e}")


def inspect_fits(fits_dir):
    """Inspect downloaded FITS files and report key metadata."""
    from astropy.io import fits as pyfits

    fits_dir = Path(fits_dir)
    fits_files = sorted(fits_dir.glob("*.fits"))

    if not fits_files:
        print(f"  No FITS files found in {fits_dir}")
        return

    print(f"\n  {'='*70}")
    print(f"  Inspecting {len(fits_files)} FITS file(s) in {fits_dir}")
    print(f"  {'='*70}")

    for fpath in fits_files:
        print(f"\n  File: {fpath.name}")
        print(f"  {'─'*60}")
        try:
            with pyfits.open(str(fpath)) as hdul:
                hdr = hdul[0].header

                # Key metadata
                info = {
                    "NAXIS": f"{hdr.get('NAXIS1', '?')} x {hdr.get('NAXIS2', '?')}",
                    "BUNIT": hdr.get("BUNIT", "unknown"),
                    "BTYPE": hdr.get("BTYPE", "unknown"),
                    "OBJECT": hdr.get("OBJECT", "unknown"),
                    "DATE-OBS": hdr.get("DATE-OBS", "unknown"),
                    "RESTFRQ": f"{hdr.get('RESTFRQ', 0)/1e9:.3f} GHz",
                    "BMAJ": f"{hdr.get('BMAJ', 0)*3600*1000:.1f} mas",
                    "BMIN": f"{hdr.get('BMIN', 0)*3600*1000:.1f} mas",
                    "BPA": f"{hdr.get('BPA', 0):.1f} deg",
                    "CDELT1": f"{abs(hdr.get('CDELT1', 0))*3600*1000:.2f} mas/pix",
                    "TELESCOP": hdr.get("TELESCOP", "unknown"),
                }

                for key, val in info.items():
                    print(f"    {key:12s}: {val}")

        except Exception as e:
            print(f"    Error reading: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Query and download ALMA Europa observations"
    )
    parser.add_argument("--query", action="store_true",
                        help="Query archive and list available data")
    parser.add_argument("--download", action="store_true",
                        help="Download continuum FITS images (pbcor only)")
    parser.add_argument("--download-all-fits", action="store_true",
                        help="Download ALL FITS images (cubes, masks, etc.)")
    parser.add_argument("--download-ms", action="store_true",
                        help="Download calibrated measurement sets (large!)")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect downloaded FITS files")
    parser.add_argument("--project", choices=["trumbo2018", "thelen2024", "both"],
                        default="both",
                        help="Which project to process (default: both)")
    args = parser.parse_args()

    want_download = args.download or args.download_all_fits
    if not any([args.query, want_download, args.download_ms, args.inspect]):
        parser.print_help()
        return

    projects = (
        [PROJECTS[args.project]] if args.project != "both"
        else list(PROJECTS.values())
    )

    for proj in projects:
        print(f"\n{'#'*70}")
        print(f"# {proj['reference']}")
        print(f"# Project: {proj['code']}")
        print(f"# Bands: {', '.join(proj['bands'])}")
        print(f"{'#'*70}")

        if args.query or want_download or args.download_ms:
            result = query_archive(proj["code"])
            if result is None:
                continue

            if args.query or want_download:
                products = list_data_products(result)

                if want_download:
                    continuum_only = not args.download_all_fits
                    download_fits(products, proj["data_dir"],
                                  continuum_only=continuum_only)

            if args.download_ms:
                download_tarballs(result, proj["data_dir"])

        if args.inspect:
            fits_dir = proj["data_dir"] / "fits"
            inspect_fits(fits_dir)


if __name__ == "__main__":
    main()
