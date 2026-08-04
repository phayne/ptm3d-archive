#!/usr/bin/env python3
"""Run ptm_thermal_sim with the ``engine_env:`` block of its OWN config applied.

PTM3D configs may carry an ``engine_env:`` mapping recording the environment
variables a run needs (``PTM_ILLUM_TRUE_NORMALS``, ``PTM_ILLUM_RISE_HABS``, ...).
**The engine does not read that block** -- it is documentation only -- so a
regression pair launched by hand can silently end up with two different
illumination configurations while looking identical on paper.  That is exactly
how a do-no-harm comparison turns into a false positive.

This runner reads the block and exports it, so a config and the environment it
was validated under cannot drift apart.  It prints the environment it applied
and refuses to run if a variable is already set to a *different* value in the
caller's environment.

Usage
-----
    python scripts/ptm_run_with_engine_env.py CONFIG.yaml [--bin PATH] [-- ARGS...]
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_engine_env(path):
    """Return the engine_env mapping. Uses PyYAML when available, else a small
    indentation-aware fallback so the runner works without extra deps."""
    try:
        import yaml
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        block = doc.get("engine_env") or {}
        return {str(k): str(v) for k, v in block.items()}
    except ImportError:
        pass

    env, in_block = {}, False
    with open(path) as fh:
        for line in fh:
            if line.startswith("engine_env:"):
                in_block = True
                continue
            if in_block:
                if line.strip() and not line[0].isspace():
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    env[k.strip()] = v.strip().strip("'\"")
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--bin", default=os.path.join(REPO, "build", "bin", "ptm_thermal_sim"))
    ap.add_argument("--log", default=None, help="write stdout+stderr here")
    # parse_known_args (not REMAINDER): the runner's own flags must work on
    # either side of the config path, and anything it does not recognise is
    # forwarded verbatim to ptm_thermal_sim.
    args, extra = ap.parse_known_args(argv)
    args.extra = extra

    if not os.path.exists(args.bin):
        raise SystemExit("no such binary: %s" % args.bin)

    engine_env = read_engine_env(args.config)
    env = dict(os.environ)
    for k, v in sorted(engine_env.items()):
        prior = env.get(k)
        if prior is not None and prior != v:
            raise SystemExit(
                "REFUSING TO RUN: %s is %r in the environment but %r in %s.\n"
                "Unset it or fix the config -- a silent mismatch here invalidates "
                "any regression comparison." % (k, prior, v, args.config))
        env[k] = v

    print("engine_env from %s: %s" % (
        args.config,
        ", ".join("%s=%s" % kv for kv in sorted(engine_env.items())) or "(none)"))

    cmd = [args.bin, "--config", args.config] + [a for a in args.extra if a != "--"]
    print("+ %s" % " ".join(cmd))
    if args.log:
        with open(args.log, "wb") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    else:
        rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print("ptm_thermal_sim exited %d" % rc, file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
