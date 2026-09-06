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

Core download set (Moon + Mercury capable, ~44 MB):
    naif0012.tls, pck00011.tpc, de440s.bsp, moon_pa_de440_200625.bpc,
    moon_de440_250416.tf
Two body-specific extras exist: the Galilean ephemeris jup365.bsp
(Jupiter/Europa position, 1.1 GB) that europa.tm requires, and Bennu's
sb-101955-118_long.bsp (80.7 MB) that bennu.tm requires. Both metakernels ship
with the repository, so the DEFAULT (--bodies auto) download is ~1.23 GB, not
~44 MB; pass --bodies moon for the Moon/Mercury core alone.

By default (--bodies auto) the script fetches the core set and, for each
metakernel present in the kernel directory, the body kernel that metakernel
names — so a bare `python3 scripts/fetch_kernels.py` pulls exactly what the
installed metakernels need. This matters because the engine's "missing kernel"
errors tell the user to run that bare command, and the Europa and Bennu configs
ship metakernels whose body-kernel dependencies the Moon-only default would
otherwise omit (leaving the run to fail on the same missing kernel again). Pass
--bodies moon/europa/bennu/all to force a set regardless of which .tm files are
present.

Downloads are resumable: an interrupted transfer leaves a .part file that is
continued with an HTTP Range request on the next run; completed files whose
size matches the server's are skipped. SHA-256 of every completed file is
printed (and checked against a pinned value where one is recorded below).

A git WORKTREE is the case this script did not cover until 2026-09-05.
`data/kernels/{lsk,pck,spk,fk}` is gitignored (1.2 GB), so every fresh worktree
starts with the metakernels and none of the kernels they name, and every
simulation in it dies with `FATAL: SPICE initialization failed`. Re-downloading
1.2 GB per worktree is the wrong answer when a verified copy is already on the
same disk, so `--link-from <repo>` symlinks them from an existing checkout --
after checking each pinned sha256, so a link is only ever made to bytes this
script would have accepted from NAIF.

Usage:
    python3 scripts/fetch_kernels.py                 # auto: core + whatever the .tm files name
    python3 scripts/fetch_kernels.py --bodies moon   # force Moon/Mercury core only
    python3 scripts/fetch_kernels.py --bodies europa # force core + Galilean ephemeris
    python3 scripts/fetch_kernels.py --bodies bennu  # force core + Bennu ephemeris
    python3 scripts/fetch_kernels.py --bodies all
    python3 scripts/fetch_kernels.py --dest /path/to/kernels
    python3 scripts/fetch_kernels.py --link-from ~/Projects/ptm   # worktree: link, do not download
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

#: Sent on every request this module makes. NOT cosmetic, and not politeness.
#: `urllib` otherwise identifies itself as `Python-urllib/3.x`, and the PDS
#: Small Bodies Node at PSI rejects exactly that string with a flat HTTP 403 --
#: measured 2026-08-28: curl carrying `Python-urllib/3.14` gets 403 while curl's
#: own default gets 206, and urllib carrying any other agent gets 200. The
#: failure is worse than a refusal, because `download` classes 403 as transient
#: and retries it five times over 40 seconds while printing "transient failure",
#: so a permanently blocked agent looks like a flaky network.
#:
#: An honest agent string with a contact URL is also what an archive asks for;
#: NAIF and PSI both serve automated clients on that basis.
USER_AGENT = ("PTM3D/1.0 (planetary thermophysical model; "
              "https://github.com/phayne/ptm)")

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

# (101955) Bennu's heliocentric ephemeris, for bennu.tm.
#
# JPL small-body solution 118 (Farnocchia et al. 2021, the final OSIRIS-REx
# reconstruction), long-span variant: body 2101955 w.r.t. the Sun over
# 1900-01-01 to 2135-10-01. 80.7 MB.
#
# This is the URL JPL Horizons itself prints when it declines to generate a
# Bennu SPK on demand: Bennu's Horizons record is *SPECIAL*, sourced from
# ORX_merged, and answers "SPK creation is not available for pre-computed
# objects in the major body index". NAIF's own copy of solution 118
# (pub/naif/ORX/kernels/spk/sb-101955-118.bsp) covers only 2015-2023 and so
# cannot serve a present-day run; measured against this file over that
# overlap the two agree to 0.4 mm.
#
# UNLIKE the NAIF entries above, /ftp/xfr/ is a transfer directory, not a
# versioned archive path, so this filename is not documented as immutable. The
# pinned hash is therefore load-bearing in a second way: a mismatch may mean
# JPL published a different file under the same name.
SSD_XFR = "https://ssd.jpl.nasa.gov/ftp/xfr"

KERNELS_BENNU = [
    ("spk", "sb-101955-118_long.bsp", f"{SSD_XFR}/sb-101955-118_long.bsp",
     "2e25094e62542f388071b43e6f7b794024ca3fa4e399306976ac62a1a5c3b48f"),
]

#: The body-specific kernel sets, keyed by the ``--bodies`` selector that
#: fetches them. THE SAME LIST OBJECTS as above, not copies: `select_kernels`
#: reads this table and so does the GUI's curated-source registry
#: (`gui/gui_core/curated_sources.py`), so the URL and the pinned sha256 for a
#: body exist in exactly one place in this repository. A second copy of a
#: pinned hash is a hash that goes stale silently, which is the whole failure
#: mode the pin exists to prevent.
#:
#: Keyed by the selector rather than by body name on purpose: the key is what
#: a user types after `--bodies`, so the message a card prints when a kernel
#: is missing ("python3 scripts/fetch_kernels.py --bodies bennu") is built
#: from the registry rather than written out again beside it.
BODY_KERNELS = {
    "europa": KERNELS_EUROPA,
    "bennu": KERNELS_BENNU,
}


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
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl is not None else None


class ChecksumMismatch(Exception):
    """A complete file whose sha256 is not the pinned one.

    Distinct from every other failure here because it is NOT transient: the
    bytes are already on disk, and _download_once's "already present —
    skipping download" short-circuit means a retry re-hashes the same bytes
    and fails identically. Retrying it burns the delay four times over an
    80 MB file and reports "transient failure", which is the opposite of true.
    """


def download(url, dest, expected_sha=None,
             retries=FETCH_RETRIES, delay=FETCH_RETRY_DELAY):
    """Download url -> dest with bounded retry on transient network failure.

    Wraps _download_once, which already resumes from a .part file, so each
    retry continues where the last left off rather than starting over.

    A pinned-sha256 mismatch is terminal, not retried: see ChecksumMismatch.
    The bad file is moved aside rather than deleted (an 80 MB re-download is
    not free, and the bytes are the evidence) and rather than left in place
    (where the next run would skip the download and fail the same way).
    """
    for attempt in range(1, retries + 1):
        try:
            if _download_once(url, dest, expected_sha):
                return True
        except ChecksumMismatch:
            kept = dest + ".sha256-mismatch"
            try:
                os.replace(dest, kept)
                print(f"  the file that failed is kept at {kept}")
            except OSError as e:
                print(f"  WARNING: could not move the failed file aside: {e}")
            print(f"  NOT RETRYING {url}: what failed is the checksum, not "
                  f"the transfer.")
            return False
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
    headers = {"User-Agent": USER_AGENT}
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
        print("  The file is corrupt, or the provider replaced it: NAIF's "
              "versioned filenames are")
        print("  documented as immutable, but JPL's /ftp/xfr/ transfer area "
              "(Bennu) is not.")
        print(f"  Re-run to fetch it again; if the same digest comes back "
              f"twice, the file changed at the source.")
        raise ChecksumMismatch(dest)
    return True


def link_from(src_root, dest, kernels):
    """Symlink already-downloaded kernels from another checkout into `dest`.

    For the gitignored-1.2-GB-per-worktree problem in the module docstring.
    Three properties, each of which was a way to get this wrong:

    1. **Every pinned kernel is hashed before it is linked.** A link is a claim
       that these are the right bytes; making that claim without checking is
       how a corrupt or superseded file propagates into every worktree that
       borrows it. A mismatch refuses the link for that file and fails the run.
    2. **Files the pinned table does not know about are linked too, and said
       out loud.** `pck/pck00010.tpc` and `pck/gm_de431.tpc` are on this
       machine, are named by shipped metakernels, and are in no table here --
       linking only the pinned five would leave a worktree that still cannot
       run and would give no hint why. They are linked with their sha256
       printed and an explicit UNPINNED label, because "verified" and
       "copied" must not look alike.
    3. **Nothing is clobbered.** An existing regular file or directory at the
       destination stops the run; an existing symlink already pointing at the
       same target is left alone and reported. Silently replacing a real file
       with a link to somebody else's copy is not a repair.

    Returns True when every requested kernel ended up resolvable in `dest`.
    """
    src = os.path.join(src_root, "data", "kernels")
    if not os.path.isdir(src):
        print(f"ERROR: --link-from {src_root} has no data/kernels directory.")
        return False
    if os.path.realpath(src) == os.path.realpath(dest):
        print(f"ERROR: --link-from source and destination are the same "
              f"directory ({os.path.realpath(dest)}). Nothing to do.")
        return False

    wanted = {(sub, name): sha for sub, name, _url, sha in kernels}
    ok = True

    # Pass 1: the kernels the caller asked for, pinned-hash checked.
    for (sub, name), sha in sorted(wanted.items()):
        s = os.path.join(src, sub, name)
        d = os.path.join(dest, sub, name)
        if not os.path.exists(s):
            print(f"MISSING in source: {sub}/{name}")
            print(f"  {src_root} does not have it either. Fetch it there first:")
            print(f"    python3 scripts/fetch_kernels.py   # in {src_root}")
            ok = False
            continue
        digest = sha256_of(s)
        if sha and digest != sha:
            print(f"REFUSING to link {sub}/{name}: sha256 {digest}")
            print(f"  does not match the pinned {sha}.")
            print(f"  The source copy is corrupt or was replaced. Do not "
                  f"borrow it; fetch a fresh one.")
            ok = False
            continue
        label = "verified" if sha else "UNPINNED (no hash in this script's table)"
        ok = _link_one(s, d, f"{sub}/{name}", digest, label) and ok

    # Pass 2: everything else the source has under the kernel subdirectories.
    for sub in ("lsk", "pck", "spk", "fk"):
        sdir = os.path.join(src, sub)
        if not os.path.isdir(sdir):
            continue
        for name in sorted(os.listdir(sdir)):
            if (sub, name) in wanted or name.startswith("."):
                continue
            s = os.path.join(sdir, name)
            if not os.path.isfile(s):
                continue
            ok = _link_one(s, os.path.join(dest, sub, name), f"{sub}/{name}",
                           sha256_of(s),
                           "UNPINNED (not in this script's table; linked "
                           "because a shipped metakernel may name it)") and ok
    return ok


def _link_one(src_file, dest_file, label, digest, note):
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    if os.path.islink(dest_file):
        if os.path.realpath(dest_file) == os.path.realpath(src_file):
            print(f"  {label}: already linked to the same file — left alone")
            return True
        print(f"ERROR: {dest_file} is a symlink to something else "
              f"({os.path.realpath(dest_file)}). Remove it deliberately; this "
              f"script will not repoint an existing link.")
        return False
    if os.path.exists(dest_file):
        print(f"ERROR: {dest_file} already exists as a real file. Not "
              f"replacing it with a link — if you meant to, delete it first.")
        return False
    os.symlink(os.path.realpath(src_file), dest_file)
    print(f"  linked {label}  sha256={digest}  [{note}]")
    return True


def select_kernels(bodies, dest):
    """Return the kernels to fetch for the requested --bodies mode.

    bodies == "auto" (the default): the core set, plus the Galilean ephemeris
    (jup365.bsp) when europa.tm is present in `dest`, plus Bennu's ephemeris
    when bennu.tm is present -- i.e. fetch exactly what the installed
    metakernels require. This lets a bare invocation satisfy a Europa or a
    Bennu config (both metakernels ship) without the user having to know to add
    --bodies europa/bennu; otherwise the run fails again on the same missing
    kernel. Each auto-addition is announced, never silent.

    An explicit bodies value (moon/europa/bennu/all) forces that set regardless
    of which .tm files are present.
    """
    kernels = list(KERNELS_CORE)
    if bodies in ("europa", "all"):
        kernels += BODY_KERNELS["europa"]
    elif bodies == "auto" and os.path.exists(os.path.join(dest, "europa.tm")):
        print("europa.tm present in kernel directory -> also fetching the "
              "Galilean ephemeris jup365.bsp (europa.tm requires it).")
        kernels += BODY_KERNELS["europa"]
    if bodies in ("bennu", "all"):
        kernels += BODY_KERNELS["bennu"]
    elif bodies == "auto" and os.path.exists(os.path.join(dest, "bennu.tm")):
        print("bennu.tm present in kernel directory -> also fetching Bennu's "
              "ephemeris sb-101955-118_long.bsp (bennu.tm requires it).")
        kernels += BODY_KERNELS["bennu"]
    return kernels


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=os.path.join(repo_root, "data", "kernels"),
                    help="kernel directory (default: <repo>/data/kernels)")
    ap.add_argument("--bodies", default="auto",
                    choices=["auto", "moon", "europa", "bennu", "all"],
                    help="auto (default) = core set, plus the Galilean "
                         "ephemeris when europa.tm is present and Bennu's when "
                         "bennu.tm is present; moon = core only; europa adds "
                         "jup365.bsp, bennu adds sb-101955-118_long.bsp, all "
                         "adds both -- unconditionally")
    ap.add_argument("--link-from", metavar="REPO", default=None,
                    help="symlink the kernels from another PTM3D checkout's "
                         "data/kernels instead of downloading them, after "
                         "checking each pinned sha256. For git worktrees, "
                         "where data/kernels is gitignored and re-fetching "
                         "1.2 GB per worktree is absurd.")
    args = ap.parse_args()

    kernels = select_kernels(args.bodies, args.dest)

    print(f"Kernel directory: {args.dest}")
    if args.link_from:
        print(f"Linking from: {args.link_from}  (no downloads)")
        ok = link_from(args.link_from, args.dest, kernels)
    else:
        ok = True
        for subdir, name, url, sha in kernels:
            outdir = os.path.join(args.dest, subdir)
            os.makedirs(outdir, exist_ok=True)
            dest = os.path.join(outdir, name)
            print(f"[{subdir}/{name}] <- {url}")
            ok = download(url, dest, sha) and ok

    tm_present = [f for f in ("moon.tm", "europa.tm", "mercury.tm", "bennu.tm")
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
    # Same gap, same reasoning, for Bennu: bennu.tm names the body SPK, and a
    # Bennu run without it stops on SPICE(NOSUCHFILE) rather than substituting
    # anything. Say so here instead of letting the run be the messenger.
    if "bennu.tm" in tm_present and not os.path.exists(
            os.path.join(args.dest, "spk", "sb-101955-118_long.bsp")):
        print("WARNING: bennu.tm is present but spk/sb-101955-118_long.bsp is "
              "missing — a Bennu simulation will fail with an "
              "unresolved-kernel error.\n"
              "         Fetch it with:  python3 scripts/fetch_kernels.py "
              "--bodies bennu")
    print("NOTE: the shipped metakernels use repo-root-relative paths "
          "(PATH_VALUES = 'data/kernels'); run ptm_thermal_sim from the "
          "repository root, or edit PATH_VALUES to an absolute path.")
    if not ok:
        verb = "link/verify" if args.link_from else "download/verify"
        print(f"\nOne or more kernels FAILED to {verb} — see above.")
        return 1
    print("\nAll kernels present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
