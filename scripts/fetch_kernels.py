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

Core download set (Moon + Mercury capable, ~45 MB):
    naif0012.tls, pck00011.tpc, de440s.bsp, moon_pa_de440_200625.bpc,
    moon_de440_250416.tf
The Galilean ephemeris jup365.bsp (Jupiter/Europa position, ~130 MB) is the
extra that europa.tm requires.

By default (--bodies auto) the script fetches the core set and, when europa.tm
is present in the kernel directory, ALSO fetches jup365.bsp — so a bare
`python3 scripts/fetch_kernels.py` pulls exactly what the installed metakernels
need. This matters because the engine's "missing kernel" errors tell the user
to run that bare command, and a Europa config ships europa.tm whose jup365.bsp
dependency the Moon-only default would otherwise omit (leaving the run to fail
on the same missing kernel again). Pass --bodies moon/europa/all to force a set
regardless of which .tm files are present.

Downloads are resumable: an interrupted transfer leaves a .part file that is
continued with an HTTP Range request on the next run; completed files whose
size matches the server's are skipped. SHA-256 of every completed file is
printed (and checked against a pinned value where one is recorded below).

Usage:
    python3 scripts/fetch_kernels.py                 # auto: core + Galilean if europa.tm present
    python3 scripts/fetch_kernels.py --bodies moon   # force Moon/Mercury core only
    python3 scripts/fetch_kernels.py --bodies europa # force core + Galilean ephemeris
    python3 scripts/fetch_kernels.py --bodies all
    python3 scripts/fetch_kernels.py --dest /path/to/kernels
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request

# Bounded retry for transient NAIF failures (connect timeout, reset, 5xx).
# Each retry resumes from the .part left by the previous attempt, and the
# pinned-sha256 check below is unchanged, so this adds resilience without
# weakening integrity. A genuine hash mismatch is not transient: it is still
# reported and still fails after the attempts are exhausted.
FETCH_RETRIES = 5
FETCH_RETRY_DELAY = 10  # seconds between attempts

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


def download(url, dest, expected_sha=None,
             retries=FETCH_RETRIES, delay=FETCH_RETRY_DELAY):
    """Download url -> dest with bounded retry on transient network failure.

    Wraps _download_once, which already resumes from a .part file, so each
    retry continues where the last left off rather than starting over. The
    pinned-sha256 verification inside _download_once is untouched; a corrupt
    or changed file still fails, just after the retries are exhausted.
    """
    for attempt in range(1, retries + 1):
        if _download_once(url, dest, expected_sha):
            return True
        if attempt < retries:
            print(f"  retry {attempt}/{retries - 1} in {delay}s "
                  f"(transient failure fetching {url})")
            time.sleep(delay)
    print(f"  giving up on {url} after {retries} attempts.")
    return False


def _download_once(url, dest, expected_sha=None):
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


def select_kernels(bodies, dest):
    """Return the kernels to fetch for the requested --bodies mode.

    bodies == "auto" (the default): the core set, plus the Galilean ephemeris
    (jup365.bsp) when europa.tm is present in `dest` -- i.e. fetch exactly what
    the installed metakernels require. This lets a bare invocation satisfy a
    Europa config (which ships europa.tm) without the user having to know to add
    --bodies europa; otherwise the run fails again on the same missing kernel.
    The auto-addition is announced, never silent.

    An explicit bodies value (moon/europa/all) forces that set regardless of
    which .tm files are present.
    """
    kernels = list(KERNELS_CORE)
    if bodies in ("europa", "all"):
        kernels += KERNELS_EUROPA
    elif bodies == "auto" and os.path.exists(os.path.join(dest, "europa.tm")):
        print("europa.tm present in kernel directory -> also fetching the "
              "Galilean ephemeris jup365.bsp (europa.tm requires it).")
        kernels += KERNELS_EUROPA
    return kernels


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=os.path.join(repo_root, "data", "kernels"),
                    help="kernel directory (default: <repo>/data/kernels)")
    ap.add_argument("--bodies", default="auto",
                    choices=["auto", "moon", "europa", "all"],
                    help="auto (default) = core set + Galilean ephemeris when "
                         "europa.tm is present; moon = core only; europa/all "
                         "add jup365.bsp unconditionally")
    args = ap.parse_args()

    kernels = select_kernels(args.bodies, args.dest)

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
    # europa.tm cannot resolve without the Galilean ephemeris; a Europa run that
    # loses this race fails on the same missing kernel again. Do not let that
    # gap pass silently even when the user forced a set that omits it.
    if "europa.tm" in tm_present and not os.path.exists(
            os.path.join(args.dest, "spk", "jup365.bsp")):
        print("WARNING: europa.tm is present but spk/jup365.bsp is missing — a "
              "Europa simulation will fail with an unresolved-kernel error.\n"
              "         Fetch it with:  python3 scripts/fetch_kernels.py "
              "--bodies europa")
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
