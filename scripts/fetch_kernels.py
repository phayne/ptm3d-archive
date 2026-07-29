#!/usr/bin/env python3
"""Fetch the SPICE kernels PTM3D needs into data/kernels/.

Standard-library only (urllib); no pip installs required.

PTM3D loads SPICE data from the directory compiled in as SPICE_KERNELS_DIR
(CMake cache variable; default: <repo>/data/kernels). Inside that directory it
expects:

    lsk/naif0012.tls          leapseconds (always required, incl. Keplerian mode)
    <body>.tm                 per-body metakernel (moon.tm, europa.tm, mercury.tm)
    pck/, spk/, fk/           kernels referenced by the metakernels

The metakernel templates shipped in data/kernels/ use PATH_VALUES relative to
the repository root ('data/kernels'), so run ptm_thermal_sim FROM THE REPO
ROOT (SPICE resolves relative metakernel paths against the current working
directory). See INSTALL.md, section "SPICE kernel data".

Default download set (Moon + Mercury capable, ~45 MB):
    naif0012.tls, pck00011.tpc, de440s.bsp, moon_pa_de440_200625.bpc,
    moon_de440_250416.tf
Add --bodies europa to also fetch jup365.bsp (Jupiter/Galilean ephemeris,
~130 MB) needed by europa.tm.

Downloads are resumable: an interrupted transfer leaves a .part file that is
continued with an HTTP Range request on the next run; completed files whose
size matches the server's are skipped. SHA-256 of every completed file is
printed (and checked against a pinned value where one is recorded below).

Usage:
    python3 scripts/fetch_kernels.py                 # Moon/Mercury set
    python3 scripts/fetch_kernels.py --bodies europa # + Galilean ephemeris
    python3 scripts/fetch_kernels.py --bodies all
    python3 scripts/fetch_kernels.py --dest /path/to/kernels
"""

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

# (subdir, filename, url, pinned sha256 or None)
# Pinned hashes were recorded from naif.jpl.nasa.gov on 2026-07-29. NAIF
# versioned filenames are immutable, so a mismatch means a corrupt download.
KERNELS_CORE = [
    ("lsk", "naif0012.tls", f"{NAIF}/lsk/naif0012.tls",
     "678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b"),
    ("pck", "pck00011.tpc", f"{NAIF}/pck/pck00011.tpc",
     "3dff7b1dbeceaa01f25467767d3fa25816051c85d162d1edf04acb310ee28bb1"),
    ("spk", "de440s.bsp", f"{NAIF}/spk/planets/de440s.bsp",
     "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2"),
    # Lunar principal-axes orientation (high-accuracy MOON_PA / MOON_ME frames).
    ("pck", "moon_pa_de440_200625.bpc", f"{NAIF}/pck/moon_pa_de440_200625.bpc",
     "60cd55aa401ea2ea97360636f567554bfe4e37bb829f901b4460a455dfaf783f"),
    ("fk", "moon_de440_250416.tf", f"{NAIF}/fk/satellites/moon_de440_250416.tf",
     "a47c71e9c9f33796bdafb2c9d69a7ee447b6016ecad80f71cd6f3e479f9cf768"),
]

KERNELS_EUROPA = [
    # Galilean satellite ephemeris (Europa position for europa.tm). Large.
    ("spk", "jup365.bsp", f"{NAIF}/spk/satellites/jup365.bsp", None),
]


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024.0 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024.0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl is not None else None


def download(url, dest, expected_sha=None):
    """Download url -> dest with .part resume. Returns True on success."""
    part = dest + ".part"
    try:
        total = remote_size(url)
    except (urllib.error.URLError, OSError) as e:
        print(f"  ERROR: cannot reach {url}: {e}")
        return False

    if os.path.exists(dest):
        local = os.path.getsize(dest)
        if total is None or local == total:
            print(f"  already present ({human(local)}) — skipping download")
            return verify(dest, expected_sha)
        print(f"  size mismatch (local {human(local)} vs remote {human(total)}) — re-fetching")
        os.remove(dest)

    offset = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {}
    mode = "wb"
    if offset and total and offset < total:
        headers["Range"] = f"bytes={offset}-"
        mode = "ab"
        print(f"  resuming at {human(offset)} of {human(total)}")
    elif offset:
        offset = 0  # stale/complete .part with unknown remote size: restart

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(part, mode) as out:
            if offset and resp.status != 206:
                # Server ignored the Range request; start over.
                out.seek(0)
                out.truncate()
                offset = 0
            done = offset
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {human(done)} / {human(total)} ({pct:5.1f}%)",
                          end="", flush=True)
            print()
    except (urllib.error.URLError, OSError) as e:
        print(f"\n  INTERRUPTED: {e}")
        print(f"  partial file kept at {part}; re-run to resume.")
        return False

    if total is not None and os.path.getsize(part) != total:
        print(f"  ERROR: got {os.path.getsize(part)} bytes, expected {total}. "
              f"Partial kept; re-run to resume.")
        return False
    os.replace(part, dest)
    return verify(dest, expected_sha)


def verify(dest, expected_sha):
    digest = sha256_of(dest)
    size = os.path.getsize(dest)
    print(f"  {os.path.basename(dest)}: {human(size)}  sha256={digest}")
    if expected_sha and digest != expected_sha:
        print(f"  ERROR: sha256 mismatch (expected {expected_sha}).")
        print("  The file is corrupt or NAIF changed a supposedly immutable file.")
        print(f"  Delete {dest} and re-run; if it persists, report an issue.")
        return False
    return True


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=os.path.join(repo_root, "data", "kernels"),
                    help="kernel directory (default: <repo>/data/kernels)")
    ap.add_argument("--bodies", default="moon", choices=["moon", "europa", "all"],
                    help="moon = Moon/Mercury set (default); europa/all adds jup365.bsp")
    args = ap.parse_args()

    kernels = list(KERNELS_CORE)
    if args.bodies in ("europa", "all"):
        kernels += KERNELS_EUROPA

    print(f"Kernel directory: {args.dest}")
    ok = True
    for subdir, name, url, sha in kernels:
        outdir = os.path.join(args.dest, subdir)
        os.makedirs(outdir, exist_ok=True)
        dest = os.path.join(outdir, name)
        print(f"[{subdir}/{name}] <- {url}")
        ok = download(url, dest, sha) and ok

    tm_present = [f for f in ("moon.tm", "europa.tm", "mercury.tm")
                  if os.path.exists(os.path.join(args.dest, f))]
    print()
    print(f"Metakernels present in {args.dest}: {', '.join(tm_present) or 'NONE'}")
    if not tm_present:
        print("WARNING: no metakernel templates found — expected them to ship "
              "with the repo in data/kernels/.")
    print("NOTE: the shipped metakernels use repo-root-relative paths "
          "(PATH_VALUES = 'data/kernels'); run ptm_thermal_sim from the "
          "repository root, or edit PATH_VALUES to an absolute path.")
    if not ok:
        print("\nOne or more kernels FAILED to download/verify — see above.")
        return 1
    print("\nAll kernels present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
