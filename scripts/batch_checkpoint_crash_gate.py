#!/usr/bin/env python3
"""Crash-resume gate for thermal.batch_checkpoint (plan phase 1D').

WHAT THIS PROVES, AND WHY THE OTHER GATES CANNOT
------------------------------------------------
The acceptance test the design documents for batch-completion checkpointing is
"on with an empty partial set is byte-identical to off".  That test would pass
while banking un-solved faces, because an uninterrupted run never exercises the
restore path at all.  The test that can actually falsify the feature is this
one: kill the engine ABRUPTLY, several times, at different points; let it
resume from whatever partials survived; and require the finished outputs to be
BYTE-IDENTICAL to an uninterrupted control.

SIGKILL, not SIGTERM.  A graceful stop would let the engine flush, which is
precisely the behavior under test; the failure mode being hunted is a partial
that was half-written, or a manifest that outran the files it names.

WEDGE SAFETY -- read before pointing this at another config
-----------------------------------------------------------
This gate SIGKILLs the engine process and nothing else.  It never touches
driver or device state, and it runs ONE engine at a time.

It is written for a CPU, radiation-off configuration on purpose.  A SIGKILL
with OpenCL command queues in flight is the documented compute-wedge class on
this workstation (a hard wedge needs a reboot, and has powered the machine
off).  The shipped default config here never creates an OpenCL context at all
-- verified with lsof: the only GPU-ish handle is the mapped libptm_gpu.dylib
image, no framework handle and no device node -- so the wedge class is
structurally excluded rather than merely avoided.

⛔ If you re-point --config at a GPU or radiation-coupled run, this gate is NOT
safe and the engine will refuse the combination anyway.  Batch checkpointing is
implemented on the CPU SIMD-batch solver only in this engine version.

USAGE
    python3 scripts/batch_checkpoint_crash_gate.py \
        --binary build/bin/ptm_thermal_sim \
        --control-config <uninterrupted config, checkpointing OFF> \
        --resume-config  <same run, checkpointing ON> \
        --kill-at 22 26 30 33

Exit 0 only if every compared output file is byte-identical AND at least one
kill actually landed inside the thermal batch loop (a gate whose kills all
missed would pass vacuously, so that is checked and reported).
"""

import argparse
import filecmp
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_output_dir(cfg_path):
    """The output directory the config names. Deliberately a literal scan and
    not a YAML parse: this gate must not depend on a YAML library being present
    in whatever interpreter runs it."""
    out, in_output = None, False
    for line in open(cfg_path):
        stripped = line.strip()
        if stripped.startswith("output:"):
            in_output = True
            continue
        if in_output:
            if line and not line[0].isspace():
                in_output = False
                continue
            if stripped.startswith("directory:"):
                out = stripped.split(":", 1)[1].strip()
                break
    return out


def read_partial_dir(cfg_path):
    for line in open(cfg_path):
        s = line.strip()
        if s.startswith("batch_checkpoint_dir:"):
            return s.split(":", 1)[1].strip()
    return None


def count_partials(d):
    if not d or not os.path.isdir(d):
        return 0
    return len([n for n in os.listdir(d) if n.endswith(".ptmb")])


def run_once(binary, cfg, log_path, kill_after=None):
    """Launch the engine; SIGKILL it after `kill_after` seconds if it is still
    alive.  Returns (killed, returncode, elapsed)."""
    with open(log_path, "wb") as log:
        t0 = time.time()
        proc = subprocess.Popen([binary, "--config", cfg],
                                stdout=log, stderr=subprocess.STDOUT)
        if kill_after is None:
            proc.wait()
            return False, proc.returncode, time.time() - t0
        deadline = t0 + kill_after
        while time.time() < deadline:
            if proc.poll() is not None:
                return False, proc.returncode, time.time() - t0
            time.sleep(0.05)
        if proc.poll() is None:
            # The ENGINE process only. Never a process group, never a device.
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()
            return True, proc.returncode, time.time() - t0
        return False, proc.returncode, time.time() - t0


def compare_trees(a, b, suffixes=(".bin", ".json", ".csv", ".ply", ".vtk")):
    """Every output file under `a` must exist under `b` with identical bytes.
    Files whose content is EXPECTED to differ between two invocations (the
    provenance stamp and the echoed config) are named, not skipped silently."""
    expected_to_differ = {"provenance.txt", "config_used.yaml"}
    same, differ, missing, skipped, extra = [], [], [], [], []
    # ⛔ BOTH DIRECTIONS. Walking only the control tree would miss the case this
    # gate exists for: a SIGKILLed attempt leaving a half-written product behind
    # that the completing run never rewrites. That file is absent from the
    # control, so a one-way walk reports byte identity over a directory
    # containing garbage from a killed process.
    for root, _dirs, files in os.walk(b):
        for name in sorted(files):
            if name in {"provenance.txt", "config_used.yaml"}:
                continue
            if os.path.splitext(name)[1] not in suffixes:
                continue
            rel = os.path.relpath(os.path.join(root, name), b)
            if not os.path.exists(os.path.join(a, rel)):
                extra.append(rel)
    for root, _dirs, files in os.walk(a):
        for name in sorted(files):
            pa = os.path.join(root, name)
            rel = os.path.relpath(pa, a)
            pb = os.path.join(b, rel)
            if name in expected_to_differ:
                skipped.append(rel)
                continue
            if not os.path.splitext(name)[1] in suffixes:
                skipped.append(rel)
                continue
            if not os.path.exists(pb):
                missing.append(rel)
                continue
            if filecmp.cmp(pa, pb, shallow=False):
                same.append(rel)
            else:
                differ.append(rel)
    return same, differ, missing, skipped, extra


def rewrite(cfg, out_path, subs, section_subs=()):
    """Copy a config with substitutions. Deliberately textual: this gate must
    run without a YAML library.

    `subs` are whole-file replacements, used only for values that occur once
    (paths). `section_subs` are `(section, key, value)` and rewrite the key ONLY
    inside that top-level block.

    ⛔ The distinction is load-bearing. A plain `str.replace("enabled: false",
    "enabled: true")` turns on radiation, the depth map, ice stability and the
    camera at once, so the radiation refusal probe would be scoring a
    configuration nobody meant to run -- and would report FAIL against correct
    engine code as soon as some other subsystem failed first."""
    text = open(cfg).read()
    for a, b in subs:
        text = text.replace(a, b)
    for section, key, value in section_subs:
        out, cur = [], None
        for line in text.split("\n"):
            if line[:1].strip() and line.rstrip().endswith(":"):
                cur = line.strip()[:-1]
            if cur == section and line.strip().split(":")[0].strip() == key:
                indent = line[:len(line) - len(line.lstrip())]
                out.append("%s%s: %s" % (indent, key, value))
            else:
                out.append(line)
        text = "\n".join(out)
    open(out_path, "w").write(text)
    return out_path


def expect_refusal(binary, cfg, log_path, needles, label, extra_args=()):
    """The engine must exit non-zero AND say why. A refusal that fires for the
    wrong reason is not the refusal under test, so the message is checked."""
    with open(log_path, "wb") as log:
        rc = subprocess.call([binary, "--config", cfg] + list(extra_args),
                             stdout=log, stderr=subprocess.STDOUT)
    text = open(log_path, errors="replace").read()
    if rc == 0:
        print("  FAIL  %s -- the run COMPLETED (rc=0)" % label)
        return False
    for n in needles:
        if n not in text:
            print("  FAIL  %s -- refused (rc=%d) but the message does not "
                  "name %r" % (label, rc, n))
            return False
    print("  ok    %s" % label)
    return True


def refusal_probes(args):
    """Every engine-level refusal, fired against a REAL partial set.

    A refusal that has never been fired is a refusal that does not exist. These
    arms exist because the failure they guard against -- a partial loading into
    a run it does not belong to -- produces a plausible-looking wrong answer
    that nothing downstream can detect."""
    work = os.path.join(args.log_dir, "refusals")
    os.makedirs(work, exist_ok=True)
    part_dir = read_partial_dir(args.resume_config)
    out_dir = read_output_dir(args.resume_config)
    if not part_dir or not os.path.isdir(part_dir) or count_partials(part_dir) == 0:
        print("FATAL: refusal probes need a populated partial set; run the "
              "crash gate first so %s exists" % part_dir)
        return 2
    ok = True
    print("=== engine-level refusals, against %d real partial(s) ==="
          % count_partials(part_dir))

    def cfg_for(name, new_part, new_out, more=(), section_subs=()):
        return rewrite(args.resume_config, os.path.join(work, name + ".yaml"),
                       [(part_dir, new_part),
                        ("directory: " + out_dir,
                         "directory: " + new_out)] + list(more),
                       section_subs)

    # 1. A doctored manifest: the index claims a partial nothing can verify.
    p5 = os.path.join(work, "p_manifest")
    if os.path.isdir(p5):
        shutil.rmtree(p5)
    shutil.copytree(part_dir, p5)
    mpath = os.path.join(p5, "manifest.json")
    s = open(mpath).read()
    i = s.index("[", s.index('"files"'))
    open(mpath, "w").write(s[:i + 1] + '"part_g9999_s999999.ptmb", ' + s[i + 1:])
    ok &= expect_refusal(
        args.binary, cfg_for("manifest", p5, os.path.join(work, "o_manifest")),
        os.path.join(work, "manifest.log"),
        ["REFUSING to resume", "manifest", "part_g9999_s999999.ptmb"],
        "a manifest naming an unverifiable partial refuses the resume")

    # 2. A corrupt partial: right length, right magic, wrong bytes. This is the
    #    cloud-storage placeholder class, and only the CRC catches it.
    p6 = os.path.join(work, "p_corrupt")
    if os.path.isdir(p6):
        shutil.rmtree(p6)
    shutil.copytree(part_dir, p6)
    victim = os.path.join(p6, sorted(n for n in os.listdir(p6)
                                     if n.endswith(".ptmb"))[0])
    b = bytearray(open(victim, "rb").read())
    b[len(b) // 2] ^= 0xFF
    open(victim, "wb").write(bytes(b))
    ok &= expect_refusal(
        args.binary, cfg_for("corrupt", p6, os.path.join(work, "o_corrupt")),
        os.path.join(work, "corrupt.log"),
        ["REFUSING to resume", "CRC32"],
        "a corrupt partial refuses the resume")

    # 3. A different mesh. The geometry hash is the clause that must fire.
    ok &= expect_refusal(
        args.binary,
        cfg_for("mesh", part_dir, os.path.join(work, "o_mesh"),
                section_subs=[("mesh", "scale_factor", "1.0001")]),
        os.path.join(work, "mesh.log"),
        ["REFUSING to resume", "DIFFERENT computation", "mesh_geomhash"],
        "a partial from a different mesh refuses, naming mesh_geomhash")

    # 4. ⭐ The digest arm, and the one no identity hash could cover. Turning
    #    primary reflection off changes the ABSORBED FLUX while leaving
    #    config_identity_hash, the mesh, the grid and the epoch all identical --
    #    exactly the class state_identity.h documents as outside its scope.
    ok &= expect_refusal(
        args.binary,
        cfg_for("forcing", part_dir, os.path.join(work, "o_forcing"),
                section_subs=[("illumination",
                               "include_primary_reflection", "false")]),
        os.path.join(work, "forcing.log"),
        ["input digest", "none of which the configuration hash covers"],
        "a partial whose FORCING moved refuses, though the config hash agrees")

    # 5. A partial offered as a whole-mesh state. Different format, so it fails
    #    at the magic -- a stronger guarantee than a flag bit.
    ok &= expect_refusal(
        args.binary,
        rewrite(args.resume_config, os.path.join(work, "asstate.yaml"),
                [("directory: " + out_dir,
                  "directory: " + os.path.join(work, "o_asstate")),
                 ("  batch_checkpoint: true",
                  "  initial_state_file: %s\n  batch_checkpoint: false"
                  % os.path.join(part_dir, sorted(
                      n for n in os.listdir(part_dir)
                      if n.endswith(".ptmb"))[0]))]),
        os.path.join(work, "asstate.log"),
        ["bad magic", "PTMT"],
        "a PARTIAL offered as thermal.initial_state_file refuses at the magic")

    # 6-8. The capability gate: every unsupported solver path refuses rather
    #      than running with the switch inert.
    ok &= expect_refusal(
        args.binary,
        cfg_for("gpu", os.path.join(work, "p_unused_gpu"),
                os.path.join(work, "o_gpu"),
                section_subs=[("thermal", "use_gpu", "true")]),
        os.path.join(work, "gpu.log"),
        ["thermal.batch_checkpoint is enabled", "GPU persistent solver"],
        "the GPU path refuses batch checkpointing")
    ok &= expect_refusal(
        args.binary,
        cfg_for("scalar", os.path.join(work, "p_unused_sc"),
                os.path.join(work, "o_scalar")),
        os.path.join(work, "scalar.log"),
        ["thermal.batch_checkpoint is enabled", "CPU scalar solver"],
        "the CPU scalar path refuses batch checkpointing",
        extra_args=("--no-simd-batch",))
    ok &= expect_refusal(
        args.binary,
        cfg_for("rad", os.path.join(work, "p_unused_rad"),
                os.path.join(work, "o_rad"),
                section_subs=[("radiation", "enabled", "true")]),
        os.path.join(work, "rad.log"),
        ["thermal.batch_checkpoint is enabled together with radiation"],
        "a radiation-coupled run refuses batch checkpointing")

    # 9. The size ceiling, refused BEFORE the run rather than mid-window.
    ok &= expect_refusal(
        args.binary,
        cfg_for("ceiling", os.path.join(work, "p_unused_ceil"),
                os.path.join(work, "o_ceiling"),
                [("  batch_checkpoint: true",
                  "  batch_checkpoint: true\n"
                  "  batch_checkpoint_max_gib: 0.001")]),
        os.path.join(work, "ceiling.log"),
        ["ceiling", "GiB"],
        "a projected partial set over the ceiling refuses before the run")

    print("\n  %s" % ("PASS" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--control-config", required=False)
    ap.add_argument("--resume-config", required=True)
    ap.add_argument("--kill-at", type=float, nargs="+",
                    help="seconds after launch to SIGKILL, one per attempt")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--max-attempts", type=int, default=12)
    ap.add_argument("--refusal-probes", action="store_true",
                    help="run the engine-level refusal arms against the "
                         "partial set the crash gate left behind, instead of "
                         "the crash gate itself")
    args = ap.parse_args()
    if args.refusal_probes:
        os.makedirs(args.log_dir, exist_ok=True)
        return refusal_probes(args)
    if not args.control_config or not args.kill_at:
        ap.error("--control-config and --kill-at are required for the crash gate")

    os.makedirs(args.log_dir, exist_ok=True)
    ctl_out = read_output_dir(args.control_config)
    res_out = read_output_dir(args.resume_config)
    part_dir = read_partial_dir(args.resume_config)
    if not ctl_out or not res_out:
        print("FATAL: could not read output.directory from both configs")
        return 2
    if not part_dir:
        print("FATAL: the resume config names no thermal.batch_checkpoint_dir; "
              "this gate refuses to guess where the partials go")
        return 2

    print("control  output : %s" % ctl_out)
    print("resume   output : %s" % res_out)
    print("partials        : %s" % part_dir)

    for d in (ctl_out, res_out, part_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)

    print("\n=== 1. uninterrupted control (checkpointing OFF) ===")
    killed, rc, el = run_once(args.binary, args.control_config,
                              os.path.join(args.log_dir, "control.log"))
    print("    rc=%d  %.1f s" % (rc, el))
    if rc != 0:
        print("FATAL: the control run failed; there is nothing to compare against")
        return 2

    print("\n=== 2. the same run, SIGKILLed and resumed ===")
    attempts, kills_in_thermal = 0, 0
    partials_before = 0
    for i in range(args.max_attempts):
        kill_at = args.kill_at[i] if i < len(args.kill_at) else None
        log = os.path.join(args.log_dir, "resume_%02d.log" % i)
        killed, rc, el = run_once(args.binary, args.resume_config, log, kill_at)
        attempts += 1
        n_now = count_partials(part_dir)
        if killed:
            gained = n_now - partials_before
            in_thermal = gained > 0
            kills_in_thermal += 1 if in_thermal else 0
            print("    attempt %d: SIGKILL at %.1f s -- %d partial(s) on disk "
                  "(+%d this attempt)%s"
                  % (i, kill_at, n_now, gained,
                     "" if in_thermal else "   [kill landed BEFORE any batch "
                                           "completed -- still a valid arm]"))
            partials_before = n_now
            continue
        print("    attempt %d: completed rc=%d in %.1f s (%d partial file(s))"
              % (i, rc, el, n_now))
        if rc != 0:
            print("FATAL: the resumed run exited %d" % rc)
            return 2
        break
    else:
        print("FATAL: never completed within %d attempts" % args.max_attempts)
        return 2

    print("\n=== 3. byte comparison, resumed vs uninterrupted control ===")
    same, differ, missing, skipped, extra = compare_trees(ctl_out, res_out)
    print("    identical : %d" % len(same))
    print("    DIFFERENT : %d" % len(differ))
    print("    MISSING   : %d" % len(missing))
    print("    EXTRA (only in the resumed tree -- a killed process's leftovers)"
          ": %d" % len(extra))
    print("    not compared (per-invocation stamps / non-product): %d"
          % len(skipped))
    for rel in differ[:20]:
        print("      DIFF  %s\n            control %s\n            resumed %s"
              % (rel, sha256(os.path.join(ctl_out, rel)),
                 sha256(os.path.join(res_out, rel))))
    for rel in missing[:20]:
        print("      MISSING  %s" % rel)
    for rel in extra[:20]:
        print("      EXTRA    %s" % rel)

    print("\n=== verdict ===")
    ok = True
    if not same:
        print("  FAIL  nothing was compared -- a gate that compares nothing "
              "passes vacuously")
        ok = False
    if differ or missing or extra:
        print("  FAIL  %d file(s) differ, %d missing, %d extra"
              % (len(differ), len(missing), len(extra)))
        ok = False
    if attempts < 2:
        print("  FAIL  the run was never interrupted, so the restore path was "
              "never exercised")
        ok = False
    if kills_in_thermal == 0:
        print("  FAIL  no kill landed after a batch had completed, so no "
              "partial was ever RESTORED. Adjust --kill-at.")
        ok = False
    print("  kills: %d, of which %d after at least one completed batch"
          % (attempts - 1, kills_in_thermal))
    print("  PASS" if ok else "  FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
