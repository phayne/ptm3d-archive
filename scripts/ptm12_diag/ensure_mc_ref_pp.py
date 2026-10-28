#!/usr/bin/env python3
"""Single source of truth for locating a CURRENT mc_ref_pp binary.

scripts/ptm12_diag/mc_ref_pp is an UNTRACKED compiled artifact -- there is no
build step or CI job that keeps it in sync with mc_ref_pp.cpp. A copy compiled
before a merged source fix (e.g. the back-face cull, 44210d8/ce3ca8c) silently
produces defective view factors, and nothing catches it if a script just runs
./mc_ref_pp by its fixed path. That happened: a binary compiled 2026-07-23 sat
at that path and was used verbatim by scripts/ptm12_diag/run_pp_sweep.sh.

Every caller in this repository now resolves the referee through this module
instead of a bare path, so there is exactly one place that knows how to build
mc_ref_pp and exactly one place that decides whether an existing binary may be
trusted. Two entry points, for two different callers:

  resolve_current(cache_root=None)
      The guard production scripts use (run_pp_sweep.sh, via the CLI below).
      ALWAYS compiles mc_ref_pp.cpp: there is no persistent "trust this old
      binary" state, so there is no staleness left to detect by construction.
      Compiles are cached under a directory keyed by a hash of the source
      bytes plus the exact compile command, so re-running with an unchanged
      tree is a hash check and a stat rather than a fresh compile (a real
      source or toolchain change always misses the cache).

  find_or_build_mc_ref_pp(explicit, work)
      The EXPLICIT-override resolution the regression gates use
      (test_vf_assembler_gate.py, test_vf_cross_agreement.py, both --mc-ref-pp
      and $PTM_MC_REF_PP). An operator naming a specific binary is pinning it
      on purpose -- e.g. to gate a scratch build before promoting it -- so a
      missing/non-executable explicit path is FATAL rather than silently
      replaced. Unchanged from the pre-2026-09-03 copy of this function that
      used to live in test_vf_assembler_gate.py.

CLI (called by run_pp_sweep.sh so the shell script never hardcodes the compile
recipe or the staleness check):
  ensure_mc_ref_pp.py               resolve a fresh binary, verify the legacy
                                     fixed path (scripts/ptm12_diag/mc_ref_pp)
                                     against it if that path exists, REFUSE
                                     (nonzero exit, message naming the stale
                                     path and the rebuild command) on a
                                     mismatch, and print ONLY the resolved
                                     path to stdout (all other output goes to
                                     stderr, so `BIN=$(...)` is safe).
  ensure_mc_ref_pp.py --replace     also overwrite the legacy fixed path with
                                     the fresh binary (on-disk hygiene).
  ensure_mc_ref_pp.py --check-only  exit 0/1 only; no path on stdout.
  ensure_mc_ref_pp.py --self-test   dyld-load regression check (two uncached
                                     builds, closed-form smoke run, equal
                                     LC_UUID-masked digests).

Every binary this module returns has just been EXECUTED on a closed-form mesh
(smoke_test); a binary that links but cannot load is fatal.
Embree: $PTM_EMBREE_ROOT (or $EMBREE_ROOT) pins an install prefix; the R1
benchmark uses ~/opt/embree-4.3.3-avx512, matching the revision engine.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MC_SRC = HERE / "mc_ref_pp.cpp"
# Local headers mc_ref_pp.cpp #includes: part of the source, so part of the cache key.
MC_HEADERS = [HERE.parent.parent / "src" / "cpp" / "mesh" / "face_orientation.h"]
LEGACY_BIN = HERE / "mc_ref_pp"
CACHE_ROOT = HERE / ".mc_ref_pp_cache"


def _log(*args):
    print(*args, file=sys.stderr)


def _compiler():
    cxx = os.environ.get("CXX")
    if cxx:
        return cxx
    # The /usr/bin/c++ shim on this machine is a stub; prefer the Xcode
    # CommandLineTools clang++ where it exists. CI runs this on Linux too, so
    # fall back to a plain "c++" rather than assuming that path exists.
    for cand in ("/Library/Developer/CommandLineTools/usr/bin/clang++", "c++"):
        if cand == "c++" or Path(cand).is_file():
            return cand
    return "c++"


def _embree_root():
    """$PTM_EMBREE_ROOT (or $EMBREE_ROOT, the name CMake already honors) selects
    an explicit Embree install prefix. Unset -> the historical pkg-config /
    /usr/local search, unchanged. A set-but-unusable root is FATAL: silently
    falling back to a different Embree would build a referee whose ISA the
    caller did not ask for."""
    raw = os.environ.get("PTM_EMBREE_ROOT") or os.environ.get("EMBREE_ROOT")
    if not raw:
        return None
    root = Path(raw).expanduser()
    if not (root / "include" / "embree4" / "rtcore.h").is_file():
        raise SystemExit(f"FATAL: Embree root {root} has no include/embree4/rtcore.h "
                         "(from $PTM_EMBREE_ROOT/$EMBREE_ROOT). Refusing to fall back.")
    return root


def _build_command(out_path, cxx):
    """Portable build. mc_ref_pp is one translation unit against Embree 4, so
    it needs no CMake. Include/library roots: pkg-config first, then the usual
    prefixes."""
    cmd = [cxx, "-O3", "-std=c++17"]
    if sys.platform == "darwin":
        sdk = subprocess.run(["/usr/bin/xcrun", "--show-sdk-path"],
                              capture_output=True, text=True).stdout.strip()
        if sdk:
            cmd += ["-isysroot", sdk]
        # NO `-Wl,-no_uuid`. That flag (added 2026-09-03 to make two
        # compiles md5-identical) produces a binary with no LC_UUID load
        # command, and the macOS 26 dyld refuses to load it ("dyld: missing
        # LC_UUID load command"): the referee built but could never run. The
        # UUID is ld's only non-content byte range, so the staleness guard now
        # compares a digest with LC_UUID masked (content_digest below) instead
        # of asking the linker to omit the command.
    embree_root = _embree_root()
    if embree_root is not None:
        # Explicit Embree install (e.g. the canonical multi-ISA build at
        # ~/opt/embree-4.3.3-avx512). Its dylib install name is @rpath-relative,
        # so the rpath is required for the binary to load at all.
        cmd += ["-I" + str(embree_root / "include"), str(MC_SRC),
                "-L" + str(embree_root / "lib"), "-Wl,-rpath," + str(embree_root / "lib"),
                "-lembree4", "-o", str(out_path)]
        return cmd
    pkg_c, pkg_l = [], []
    for probe, sink in ((["--cflags"], pkg_c), (["--libs"], pkg_l)):
        r = subprocess.run(["pkg-config"] + probe + ["embree4"], capture_output=True, text=True)
        if r.returncode == 0:
            sink.extend(r.stdout.split())
    if pkg_c:
        cmd += pkg_c
    else:
        for inc in ("/usr/local/include", "/opt/homebrew/include", "/usr/include"):
            if (Path(inc) / "embree4" / "rtcore.h").is_file():
                cmd += ["-I" + inc]
                break
    cmd += [str(MC_SRC)]
    if pkg_l:
        cmd += pkg_l
    else:
        for lib in ("/usr/local/lib", "/opt/homebrew/lib"):
            if Path(lib).is_dir():
                cmd += ["-L" + lib]
        cmd += ["-lembree4"]
    cmd += ["-o", str(out_path)]
    return cmd


def build_mc_ref_pp(out_path):
    """Compile mc_ref_pp.cpp to out_path, then prove it loads and runs
    (smoke_test). FATAL (SystemExit) on either failure, with the compiler's own
    stdout/stderr surfaced first."""
    cmd = _build_command(out_path, _compiler())
    _log("  mc_ref_pp: building --", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(out_path).is_file():
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("FATAL: could not build mc_ref_pp. Install Embree 4 (headers in "
                          "<prefix>/include/embree4) or pass --mc-ref-pp /path/to/mc_ref_pp.")
    smoke_test(out_path)
    return str(out_path)


# ---- dyld-load regression check: the binary must LOAD and RUN, not just link ----------------
# A successful link proves nothing about loadability: the -no_uuid binary linked cleanly for
# five weeks and dyld refused it every time. So every binary this module hands out is first
# executed on a tiny mesh with a closed-form answer.
#
# Geometry: a small launch triangle at z=0 (normal +z, centroid at the origin) under a 2x2
# square at height h=1 (two triangles, normal -z). mc_ref_pp launches cosine-weighted rays from
# the launch-face centroid, so its row sum estimates the point-to-rectangle view factor
#   F = 4 * (1/2pi) [ A/sqrt(1+A^2) atan(B/sqrt(1+A^2)) + B/sqrt(1+B^2) atan(A/sqrt(1+B^2)) ]
# with A = B = 1 (each quadrant is a 1x1 rectangle with a corner above the point) = 0.55415.
SMOKE_NRAYS = 20000
SMOKE_TOL = 2e-3      # absolute on SumF; measured error at 2e4 rays: 3.7e-4 (2026-10-12, both Embree ISAs)


def _smoke_expected():
    import math
    A = B = 1.0
    q = (A / math.sqrt(1 + A * A) * math.atan(B / math.sqrt(1 + A * A))
         + B / math.sqrt(1 + B * B) * math.atan(A / math.sqrt(1 + B * B))) / (2 * math.pi)
    return 4 * q


def _write_smoke_inputs(work):
    import struct
    V = [(-0.01, -0.01, 0.0), (0.02, -0.01, 0.0), (-0.01, 0.02, 0.0),
         (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)]
    F = [(0, 1, 2), (3, 5, 4), (3, 6, 5)]
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(V)}\nproperty double x\nproperty double y\nproperty double z\n"
           f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    ply = Path(work) / "smoke.ply"
    with open(ply, "wb") as fh:
        fh.write(hdr.encode())
        for v in V:
            fh.write(struct.pack("<3d", *v))
        for f in F:
            fh.write(struct.pack("<B3i", 3, *f))
    tmax = Path(work) / "smoke_tmax.bin"
    tmax.write_bytes(struct.pack("<i", len(F)) + struct.pack(f"<{len(F)}f", *([100.0] * len(F))))
    fl = Path(work) / "smoke_faces.txt"
    fl.write_text("0\n")
    return ply, tmax, fl


def smoke_test(binary):
    """Run `binary` on the closed-form mesh above. FATAL on: failure to load/exit 0, a row sum off the closed form, an output record that does not
    hold exactly the two receiver faces, or any back-face hit. Returns the
    measured SumF."""
    import struct
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mc_ref_pp_smoke_") as work:
        ply, tmax, fl = _write_smoke_inputs(work)
        outb = Path(work) / "smoke_out.bin"
        # the closed-form geometry is two facing plates in a LOCAL frame (receiver
        # wound to face the source, not the outside of a body), so no radial orientation rule
        # applies to it. The smoke test checks the integrator and the dyld load, not the
        # orientation (test_mc_ref_pp_orientation.py does that), so it traces the raw winding.
        env = dict(os.environ, PTM_PP_ORIENT="raw")
        r = subprocess.run([str(binary), str(ply), str(tmax), str(fl), str(SMOKE_NRAYS),
                            "1e-4", "0.2", "1", str(outb)],
                           capture_output=True, text=True, timeout=120, env=env)
        why = None
        if r.returncode != 0:
            why = f"exit status {r.returncode} (a dyld load failure exits before main)"
        rows = [ln.split() for ln in r.stdout.splitlines() if ln and not ln.startswith("#")]
        if why is None and (len(rows) != 1 or rows[0][0] != "0"):
            why = f"expected one stdout row for face 0, got {rows!r}"
        if why is None:
            sumf, fsky = float(rows[0][1]), float(rows[0][2])
            want = _smoke_expected()
            if abs(sumf - want) > SMOKE_TOL:
                why = f"SumF {sumf:.5f} vs closed form {want:.5f} (tol {SMOKE_TOL})"
            elif abs(sumf + fsky - 1.0) > 2e-4:
                why = f"SumF + F_sky = {sumf + fsky:.5f}, expected 1 (no occluders here)"
            elif "BACKFACE summary: cull=ON  0 back-side hits" not in r.stderr:
                why = "back-face hits reported on a correctly wound mesh"
        if why is None:
            raw = outb.read_bytes() if outb.is_file() else b""
            if len(raw) != 8 + 2 * 8:
                why = f"output record is {len(raw)} B, expected 24 (src, nnz=2, 2 partners)"
            else:
                src, nnz = struct.unpack("<2i", raw[:8])
                parts = [struct.unpack("<if", raw[8 + 8 * k:16 + 8 * k]) for k in range(2)]
                ids = sorted(p[0] for p in parts)
                fsum = sum(p[1] for p in parts)
                if src != 0 or nnz != 2 or ids != [1, 2] or abs(fsum - sumf) > 2e-4:
                    why = (f"output record src={src} nnz={nnz} partners={ids} "
                           f"sum={fsum:.5f} vs SumF {sumf:.5f}")
        if why is not None:
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
            raise SystemExit(f"FATAL: mc_ref_pp smoke test failed for {binary}: {why}. "
                             "This binary must not be used (dyld-load regression check).")
    _log(f"  mc_ref_pp: smoke test PASS ({binary}): SumF {sumf:.5f} vs closed form "
         f"{want:.5f}, |err| {abs(sumf - want):.2e}")
    return sumf


def find_or_build_mc_ref_pp(explicit, work):
    """Explicit-override resolution used by the regression gates. See module
    docstring: an EXPLICITLY named binary that cannot be used is fatal -- the
    point of naming one is to gate that exact build, and silently substituting
    a fresh compile would return a PASS that describes a different binary.
    With no override, an already-built binary at the legacy fixed path is
    reused as-is; gate callers accept that trust because a caller who wants a
    guaranteed-current binary regardless of what sits on disk should call
    resolve_current() instead (this is what run_pp_sweep.sh does)."""
    if explicit and not (Path(explicit).is_file() and os.access(str(explicit), os.X_OK)):
        raise SystemExit(f"FATAL: --mc-ref-pp {explicit} is missing or not executable. "
                          "Refusing to substitute a different binary.")
    for cand in (explicit, os.environ.get("PTM_MC_REF_PP"), LEGACY_BIN):
        if cand and Path(cand).is_file() and os.access(str(cand), os.X_OK):
            _log(f"  mc_ref_pp: using existing {cand}")
            smoke_test(cand)   # an existing file can still be unloadable
            return str(cand)
    return build_mc_ref_pp(Path(work) / "mc_ref_pp")


def _source_key():
    """sha256 of mc_ref_pp.cpp + its local headers + the exact compile command, truncated to 16 hex
    chars for a directory name. Covers a source edit AND a toolchain/flag
    change (e.g. a different Embree found by pkg-config) -- either can change
    what the binary does, so either must bust the cache."""
    h = hashlib.sha256()
    h.update(MC_SRC.read_bytes())
    for hdr in MC_HEADERS:          # a header edit must miss the cache exactly like a .cpp edit
        if not hdr.is_file():
            raise SystemExit(f"FATAL: mc_ref_pp header {hdr} is missing")
        h.update(hdr.read_bytes())
    h.update(" ".join(_build_command("<out>", _compiler())).encode())
    return h.hexdigest()[:16]


def resolve_current(cache_root=None):
    """Return a binary GUARANTEED to be a fresh compile of the mc_ref_pp.cpp
    currently on disk. Nothing is cached by BINARY identity -- only by
    source+toolchain hash -- so a source change always misses the cache and
    always recompiles; there is no stale-binary state this function can be in.

    Builds to a PID-suffixed temp file and os.replace()s it into place, so two
    callers racing on the same cache entry (e.g. a sweep and a regression gate
    launched together) never hand either one a half-written binary: the loser
    of the race simply overwrites an already-complete file with a byte-identical
    one, and a caller that lands between mkdir and the rename falls through to
    building its own temp file rather than reading a partial out_path."""
    cache_root = Path(cache_root) if cache_root else CACHE_ROOT
    out_dir = cache_root / _source_key()
    out_path = out_dir / "mc_ref_pp"
    if out_path.is_file() and os.access(str(out_path), os.X_OK):
        # Re-prove loadability on every resolve (~0.1 s): a cached binary can stop
        # loading when a dylib it links (Embree, libc++) moves or is upgraded.
        smoke_test(out_path)
        return str(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f".mc_ref_pp.tmp-{os.getpid()}"
    build_mc_ref_pp(tmp_path)
    os.replace(tmp_path, out_path)   # atomic on the same filesystem
    return str(out_path)


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


LC_UUID = 0x1b
LC_CODE_SIGNATURE = 0x1d


def content_digest(path):
    """md5 of the binary with its link-time identity bytes neutralized.

    ld64 writes a 16-byte LC_UUID whose value depends on more than the code:
    measured 2026-10-12 (ld-1267), the same source and command linked to two
    different output paths differ in exactly those 16 bytes. Removing the
    command (-no_uuid) is what broke loading, so the comparison masks
    it instead. On a signed Mach-O the signature blob covers the UUID, so it is
    excluded too. Anything that is not a thin 64-bit little-endian Mach-O (e.g.
    an ELF, whose build-id is already a content hash) is hashed as-is."""
    import struct
    data = bytearray(Path(path).read_bytes())
    if len(data) >= 32 and struct.unpack_from("<I", data, 0)[0] == 0xFEEDFACF:
        ncmds = struct.unpack_from("<I", data, 16)[0]
        off = 32
        sig_off = None
        for _ in range(ncmds):
            cmd, size = struct.unpack_from("<II", data, off)
            if cmd == LC_UUID:
                data[off + 8:off + 24] = b"\0" * 16
            elif cmd == LC_CODE_SIGNATURE:
                sig_off = struct.unpack_from("<I", data, off + 8)[0]
                data[off + 8:off + 16] = b"\0" * 8
            off += size
        if sig_off:
            del data[sig_off:]
    return hashlib.md5(bytes(data)).hexdigest()


def verify_or_refuse(candidate=None, fresh=None):
    """Compare `candidate` (default: the legacy fixed-name binary
    scripts/ptm12_diag/mc_ref_pp) against a fresh compile of current source.
    Returns quietly if candidate does not exist (nothing to be stale) or its
    content matches. REFUSES -- SystemExit, nonzero exit -- naming `candidate`
    and the exact rebuild command if it exists and differs.

    This is the check that would have caught the incident this module exists
    for: mc_ref_pp compiled 2026-07-23, predating the back-face-cull fix
    (44210d8, ce3ca8c), sitting at this fixed path and used verbatim."""
    candidate = Path(candidate) if candidate else LEGACY_BIN
    if not candidate.is_file():
        return
    fresh = fresh or resolve_current()
    if content_digest(candidate) == content_digest(fresh):
        return
    raise SystemExit(
        f"FATAL: {candidate} does not match a fresh compile of {MC_SRC.name}.\n"
        f"  stale on-disk binary : content md5 {content_digest(candidate)}  ({candidate})\n"
        f"  fresh compile        : content md5 {content_digest(fresh)}  ({fresh})\n"
        "This binary predates a merged source fix and MUST NOT be used for a "
        "production sweep. Rebuild it with:\n"
        f"  python3 {Path(__file__).resolve()} --replace\n"
        "or delete it -- every caller in this repository now resolves a "
        "guaranteed-fresh binary through this module and does not need the "
        "fixed path to exist."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replace", action="store_true",
                     help="overwrite the legacy fixed path (scripts/ptm12_diag/mc_ref_pp) "
                          "with the fresh binary, after building/verifying it")
    ap.add_argument("--check-only", action="store_true",
                     help="exit 0/1 only; do not print the resolved path")
    ap.add_argument("--self-test", action="store_true",
                     help="dyld-load regression check: build into two fresh directories, run "
                          "the closed-form smoke test on both, and require equal content "
                          "digests (LC_UUID masked). Uses no cache; exit 0 on PASS.")
    a = ap.parse_args()

    if a.self_test:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="mc_ref_pp_selftest_") as work:
            bins = []
            for k in (1, 2):
                d = Path(work) / f"build{k}"
                d.mkdir()
                bins.append(build_mc_ref_pp(d / f"mc_ref_pp_{k}"))   # smoke-tests each
            raw = [_md5(b) for b in bins]
            cd = [content_digest(b) for b in bins]
            _log(f"  self-test: raw md5 {raw[0]} / {raw[1]} "
                 f"({'equal' if raw[0] == raw[1] else 'DIFFER (LC_UUID only, expected)'})")
            _log(f"  self-test: content md5 {cd[0]} / {cd[1]}")
            if cd[0] != cd[1]:
                raise SystemExit("FATAL: two compiles of the same source differ beyond "
                                 "LC_UUID; the staleness guard would cry wolf.")
            otool = subprocess.run(["otool", "-L", bins[0]], capture_output=True, text=True) \
                if sys.platform == "darwin" else None
            if otool is not None:
                emb = [ln.strip() for ln in otool.stdout.splitlines() if "embree" in ln]
                _log(f"  self-test: links {emb}")
        _log("  self-test: PASS")
        return

    fresh = resolve_current()
    if a.replace:
        old_md5 = _md5(LEGACY_BIN) if LEGACY_BIN.is_file() else None
        shutil.copy2(fresh, LEGACY_BIN)
        os.chmod(LEGACY_BIN, 0o755)
        _log(f"  mc_ref_pp: replaced {LEGACY_BIN} "
             f"(was {old_md5 or 'absent'}, now {_md5(LEGACY_BIN)})")
    else:
        verify_or_refuse(fresh=fresh)

    if not a.check_only:
        print(fresh)


if __name__ == "__main__":
    main()
