#!/usr/bin/env python3
"""Fetch PTM3D's large science datasets into data/.

Standard-library only. Companion to scripts/fetch_kernels.py (SPICE kernels);
this script reuses its download machinery (resumable transfers, SHA-256
verification, clear errors) for the science data catalog in
data/data_manifest.json: lunar topography grids, thermophysical property maps,
albedo maps, validation crater meshes, and the Moon3D global mesh.

Usage (from the repository root):

    python3 scripts/fetch_data.py --list             # show the catalog
    python3 scripts/fetch_data.py --core             # every tier="core" dataset (~1.7 GB)
    python3 scripts/fetch_data.py --dataset moon-topo --dataset albedo-maps
    python3 scripts/fetch_data.py --verify --core    # re-hash archives AND unpacked files

Destination: --dest, else $PTM_DATA_ROOT, else <repo>/data. Downloaded
archives are kept under <dest>/_downloads/ (so re-runs are cheap) and unpacked
into the directories the configs expect (e.g. data/topo/moon/).

Until the public deposit exists, the manifest's base URL is a placeholder;
point --base-url at a staging copy (https://... or file:///...) to test.
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_kernels import download, human, sha256_of  # noqa: E402  (shared machinery)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER = "ZENODO_PLACEHOLDER"


def data_root(cli_dest):
    if cli_dest:
        return os.path.abspath(cli_dest)
    env = os.environ.get("PTM_DATA_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.join(REPO_ROOT, "data")


def load_manifest(path):
    with open(path) as f:
        m = json.load(f)
    if m.get("manifest_version") != 1:
        sys.exit(f"ERROR: unsupported manifest_version in {path}")
    return m


def dataset_target(root, rel_unpack_to):
    """Map the manifest's 'data/...' target under the chosen data root."""
    rel = rel_unpack_to
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return os.path.join(root, rel)


def safe_members(tf, dest):
    """Refuse absolute paths / parent-directory escapes in archives."""
    for m in tf.getmembers():
        name = m.name
        if name.startswith("/") or ".." in name.split("/"):
            sys.exit(f"ERROR: unsafe path in archive: {name!r} — refusing to unpack.")
        yield m


def fetch_dataset(name, ds, root, base_url, verify_full, unverifiable=None):
    print(f"\n=== {name} ({ds.get('tier', 'core')}): {ds.get('description', '')}")
    if unverifiable is None:
        unverifiable = []
    ok = True
    for fe in ds["files"]:
        fname = fe["name"]
        url = fe.get("url") or (base_url.rstrip("/") + "/" + fname)
        if PLACEHOLDER in url:
            print(f"  {fname}: the public data deposit is not published yet "
                  f"(manifest URL is a placeholder).\n"
                  f"  Use --base-url to point at a staging copy, or check for "
                  f"an updated repository revision.")
            return False
        target_dir = dataset_target(root, fe["unpack_to"])
        if fe.get("archive"):
            dl_dir = os.path.join(root, "_downloads")
            os.makedirs(dl_dir, exist_ok=True)
            tarball = os.path.join(dl_dir, fname)
            marker = tarball + ".unpacked"
            need_unpack = True
            if os.path.exists(marker) and open(marker).read().strip() == fe["sha256"]:
                if os.path.isdir(target_dir) and os.listdir(target_dir):
                    print(f"  {fname}: already unpacked into {target_dir} — skipping")
                    if verify_full:
                        archive_ok = True
                        if os.path.exists(tarball):
                            archive_ok = verify_download(tarball, fe)
                            if not archive_ok:
                                ok = False
                        # A matching marker only means an unpack happened once. Verify
                        # what is actually on disk now, or a stale tree passes silently.
                        status = report_unpacked(name, fname, tarball, target_dir,
                                                 archive_ok)
                        if status == "stale":
                            ok = False
                        elif status == "unknown":
                            unverifiable.append(f"{name}/{fname}")
                    continue
            print(f"  [{fname}] <- {url}")
            if not download(url, tarball, fe["sha256"]):
                ok = False
                continue
            os.makedirs(target_dir, exist_ok=True)
            with tarfile.open(tarball, "r:gz") as tf:
                tf.extractall(target_dir, members=safe_members(tf, target_dir))
            with open(marker, "w") as mf:
                mf.write(fe["sha256"] + "\n")
            print(f"  unpacked -> {target_dir}")
        else:
            os.makedirs(target_dir, exist_ok=True)
            dest = os.path.join(target_dir, fname)
            if os.path.exists(dest) and os.path.getsize(dest) == fe["size"]:
                print(f"  {fname}: present ({human(fe['size'])}) — "
                      + ("verifying" if verify_full else "skipping (size matches)"))
                if verify_full and not verify_download(dest, fe):
                    ok = False
                continue
            print(f"  [{fname}] <- {url}")
            if not download(url, dest, fe["sha256"]):
                ok = False
    return ok


def verify_download(path, fe):
    digest = sha256_of(path)
    good = digest == fe["sha256"]
    print(f"  {os.path.basename(path)}: sha256 "
          + ("OK" if good else f"MISMATCH (expected {fe['sha256']}, got {digest})"))
    return good


def _stream_sha256(fh):
    """sha256 of an open file object (sha256_of's counterpart for a stream)."""
    h = hashlib.sha256()
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def verify_unpacked(tarball, target_dir):
    """Verify the files ON DISK still match the archive they came from.

    The `.unpacked` marker records only that an unpack once HAPPENED with a given
    archive checksum. It says nothing about the files afterwards, so a tree whose
    marker matches the manifest can still hold stale or hand-edited data. That is
    not hypothetical: it is exactly how the PTM-34 lunar albedo misregistration
    survived a corrected archive for months -- the archive was right, the loader
    was right, and every check that existed passed while the bins on disk were the
    old 180-degree-rolled ones. Checking the archive alone cannot see that.

    Returns (ok, n_checked, mismatched, missing). Raises OSError/tarfile.TarError
    if the archive itself cannot be read; the caller decides how to report that
    (it means "cannot tell", not "clean").
    """
    mismatched, missing, n = [], [], 0
    with tarfile.open(tarball, "r:gz") as tf:
        # Iterate SEQUENTIALLY (`for m in tf`, not getmembers() + random access):
        # seeking backwards in a gzip stream re-decompresses from the start, so
        # random member access costs one full pass per member. Forward-only access
        # keeps the whole verification to a single pass over the archive.
        for m in tf:
            if not m.isfile():
                continue
            n += 1
            on_disk = os.path.join(target_dir, m.name)
            if not os.path.exists(on_disk):
                missing.append(m.name)
                continue
            if os.path.getsize(on_disk) != m.size:
                mismatched.append(m.name)
                continue
            src = tf.extractfile(m)
            if src is None:
                continue
            if _stream_sha256(src) != sha256_of(on_disk):
                mismatched.append(m.name)
    return (not mismatched and not missing), n, mismatched, missing


def report_unpacked(dataset, fname, tarball, target_dir, archive_ok):
    """Verify unpacked content and say so loudly in every case.

    Returns "ok" | "stale" | "unknown". "unknown" means the check could not be
    made at all -- never conflate that with a pass, but do not report it as a
    corrupt data tree either, because the user's files may be perfectly fine.
    """
    def remedy(*paths):
        return (f"\n      rm -f {' '.join(repr(p) for p in paths)} && "
                f"python3 scripts/fetch_data.py --dataset {dataset}")

    def unknown(reason, *paths):
        print(f"  {fname}: unpacked content NOT VERIFIED — {reason}{remedy(*paths)}")
        return "unknown"

    if not os.path.exists(tarball):
        return unknown(
            f"the archive is no longer in _downloads/ (cache reclaimed?), so "
            f"{target_dir} cannot be checked against it. To audit it, re-fetch "
            f"the archive:", f"{tarball}.unpacked")
    if not archive_ok:
        # The archive already failed its own sha256. Comparing files against it
        # would name the user's (possibly good) data as the culprit.
        return unknown(
            "the archive in _downloads/ is itself corrupt, so it is not a valid "
            "reference. Re-download it:", tarball, f"{tarball}.unpacked")
    try:
        ok, n, mismatched, missing = verify_unpacked(tarball, target_dir)
    except (OSError, tarfile.TarError) as e:
        return unknown(
            f"cannot read the archive ({e.__class__.__name__}: {e}). Re-download it:",
            tarball, f"{tarball}.unpacked")
    if ok:
        print(f"  {fname}: unpacked content OK ({n} files match the archive)")
        return "ok"
    bad = mismatched + missing
    print(f"  {fname}: unpacked content MISMATCH in {target_dir} — "
          f"{len(mismatched)} differ, {len(missing)} missing, of {n} files.")
    for nm in bad[:12]:
        print(f"      {nm}")
    if len(bad) > 12:
        print(f"      ... and {len(bad) - 12} more")
    print(f"  These files are STALE or locally modified. Re-fetch with:"
          f"{remedy(f'{tarball}.unpacked')}")
    return "stale"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data", "data_manifest.json"))
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    ap.add_argument("--dataset", action="append", default=[], help="dataset name (repeatable)")
    ap.add_argument("--core", action="store_true", help='fetch every tier="core" dataset')
    ap.add_argument("--dest", default=None, help="data root (default: $PTM_DATA_ROOT or <repo>/data)")
    ap.add_argument("--base-url", default=None,
                    help="override the manifest base URL (staging: https://... or file:///...)")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash the downloaded archives AND the files they were "
                         "unpacked to, catching a stale or locally modified data tree")
    args = ap.parse_args()

    m = load_manifest(args.manifest)
    datasets = m["datasets"]

    if args.list:
        print(f"{'dataset':24} {'tier':9} {'size':>9}  description")
        for name, ds in datasets.items():
            total = sum(f["size"] for f in ds["files"])
            print(f"{name:24} {ds.get('tier', 'core'):9} {human(total):>9}  {ds.get('description', '')}")
        return 0

    selected = list(args.dataset)
    if args.core:
        selected += [n for n, d in datasets.items() if d.get("tier") == "core" and n not in selected]
    if not selected:
        print("Nothing selected. Use --list, --core, or --dataset NAME.")
        return 1
    unknown = [n for n in selected if n not in datasets]
    if unknown:
        print(f"ERROR: unknown dataset(s): {', '.join(unknown)} (see --list)")
        return 1

    root = data_root(args.dest)
    base_url = args.base_url or m.get("base_url", PLACEHOLDER)
    print(f"Data root: {root}")

    all_ok = True
    unverifiable = []
    for name in selected:
        all_ok = fetch_dataset(name, datasets[name], root, base_url, args.verify,
                               unverifiable) and all_ok

    if not all_ok:
        print("\nOne or more datasets FAILED — see above.")
        return 1
    if unverifiable:
        # Not a failure, but not a clean bill of health either -- say which.
        print(f"\nAll selected datasets present. NOTE: the unpacked content of "
              f"{len(unverifiable)} could not be checked "
              f"({', '.join(unverifiable)}) — see above.")
        return 0
    print("\nAll selected datasets present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
