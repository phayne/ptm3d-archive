#!/usr/bin/env python3
"""Write-check regression test: mc_ref_pp must exit nonzero when its output cannot be written.

Three arms on the 3-face closed-form mesh from ensure_mc_ref_pp (24-byte .bin):
  [W0] control: unrestricted write -> rc 0, 24-byte record.
  [W1] SHORT WRITE: RLIMIT_FSIZE = 16 bytes with SIGXFSZ ignored, so the kernel
       returns EFBIG instead of killing the process -- the same failure a full disk
       (ENOSPC) produces at the same call. Must exit nonzero with a FATAL line.
       (Before the write checks the binary exited 0 here and left a truncated 16-byte file.)
  [W2] UNOPENABLE OUTPUT: the .bin path is inside a non-existent directory. Must exit
       nonzero with a FATAL line (before: fwrite on a NULL FILE*).

  test_mc_ref_pp_write_errors.py [--mc-ref-pp BIN]
"""
import argparse
import resource
import signal
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensure_mc_ref_pp as E  # noqa: E402


def run(binary, work, outb, fsize=None):
    ply, tmax, fl = E._write_smoke_inputs(work)

    def limit():
        if fsize is not None:
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)   # inherited across exec
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    # The smoke inputs are a closed-form local-frame geometry, not a body-centered
    # mesh, so the shared surface-side rule orientation rule would refuse them (exit 5) before any
    # write happens. Trace the raw winding, as ensure_mc_ref_pp's own smoke test does.
    env = dict(os.environ, PTM_PP_ORIENT="raw")
    return subprocess.run([str(binary), str(ply), str(tmax), str(fl), "2000", "1e-4", "0.2",
                           "1", str(outb)], capture_output=True, text=True,
                          preexec_fn=limit, timeout=120, env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-ref-pp", default=None, help="binary under test (default: fresh build)")
    a = ap.parse_args()
    binary = a.mc_ref_pp or E.resolve_current()
    fails = 0
    with tempfile.TemporaryDirectory(prefix="mc_ref_pp_wr_") as w:
        w = Path(w)
        r = run(binary, w, w / "ok.bin")
        size = (w / "ok.bin").stat().st_size if (w / "ok.bin").exists() else -1
        ok = r.returncode == 0 and size == 24
        print(f"  [W0] control            rc={r.returncode} bin={size} B   {'PASS' if ok else 'FAIL'}")
        fails += not ok
        r = run(binary, w, w / "short.bin", fsize=16)
        size = (w / "short.bin").stat().st_size if (w / "short.bin").exists() else -1
        ok = r.returncode == 4 and " failed for " in r.stderr
        print(f"  [W1] short write (16 B) rc={r.returncode} bin={size} B   {'PASS' if ok else 'FAIL'}"
              + ("" if ok else "  <- a truncated output exited 0"))
        if r.stderr.strip():
            print("       " + [ln for ln in r.stderr.splitlines() if ln.strip()][-1])
        fails += not ok
        r = run(binary, w, w / "no_such_dir" / "x.bin")
        ok = r.returncode == 4 and "fopen failed for" in r.stderr
        print(f"  [W2] unopenable output  rc={r.returncode}   {'PASS' if ok else 'FAIL'}")
        if r.stderr.strip():
            print("       " + [ln for ln in r.stderr.splitlines() if ln.strip()][-1])
        fails += not ok
    print(f"[test_mc_ref_pp_write_errors] {'PASS' if not fails else 'FAIL'} ({binary})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
