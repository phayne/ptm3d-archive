#!/usr/bin/env python3
"""Self-test for scripts/ptm_validate_baseline.py. No pytest, no deps, no engine.

WHAT THIS FILE IS FOR
---------------------
On 2026-09-04 the validation suite's own `verify` subcommand was found to be
comparing the baseline with itself. With no `--arm`, it resolved the runs to
compare as `{k: v["run_dir"] for k, v in manifest["arms"].items()}` -- the
manifest's own recorded directories -- so it re-digested the banked runs and
reported `330/330 files byte-identical across 13 arm(s)`, `VERDICT:
BYTE-IDENTICAL`, from a tree nine merges past the bank. That was the command
`docs/dev/validation-baselines.md` printed and the `ptm-validate` skill called the
fast path. It could not fail, and it was read as a pass.

A fix to a gate is worth nothing without an arm that fails when the gate does,
so the load-bearing test here is `doctored_bank_copy_MISMATCHES`: take a
scratch copy of a real bank, change ONE BYTE inside one output file, and demand
a MISMATCH verdict and a non-zero exit. `vacuity_control` is its complement --
it re-derives what the old code did, and shows it reports zero differences no
matter what, which is what makes the doctored arm meaningful rather than
decorative.

Everything runs on a synthetic bank built in a temporary directory: no engine,
no GPU, no science data.

Usage:  python3 scripts/test_ptm_validate_baseline.py
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "ptm_validate_baseline.py")
KERNELS = os.path.join(HERE, "fetch_kernels.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vb = _load("vb", TARGET)
fk = _load("fk", KERNELS)

_fail = 0


def check(cond, what):
    global _fail
    if cond:
        print(f"  PASS  {what}")
    else:
        _fail += 1
        print(f"  FAIL  {what}")


def run_tool(*argv):
    """Invoke the tool as a subprocess: the exit code is part of the contract."""
    proc = subprocess.run([sys.executable, TARGET] + list(argv),
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# A synthetic bank: two arms, each with a statistics/ tree including _raw/.
# ---------------------------------------------------------------------------
def make_run(root, seed):
    os.makedirs(os.path.join(root, "statistics", "_raw"), exist_ok=True)
    for name in ("temperature_max.bin", "illumination_avg.bin"):
        with open(os.path.join(root, "statistics", name), "wb") as fh:
            fh.write(b"\x20\x00\x00\x00" + (seed + name).encode() * 64)
    with open(os.path.join(root, "statistics", "_raw",
                           "temperature_sum.bin"), "wb") as fh:
        fh.write(b"\x20\x00\x00\x00" + (seed + "raw").encode() * 64)
    # The two files digest_run must IGNORE, in the two places they appear.
    with open(os.path.join(root, "provenance.txt"), "w") as fh:
        fh.write("engine_git_sha=deadbeef01\nomp_max_threads=48\n"
                 f"wall_clock={seed}\n")
    with open(os.path.join(root, "statistics", "manifest.json"), "w") as fh:
        fh.write('{"generated": "%s"}' % seed)
    return root


tmp = tempfile.mkdtemp(prefix="ptm_vb_selftest_")
src_root = os.path.join(tmp, "source_runs")
bank = os.path.join(tmp, "bank")
repo = os.path.join(tmp, "repo")
os.makedirs(repo)
fake_binary = os.path.join(repo, "ptm_thermal_sim")
with open(fake_binary, "wb") as fh:
    fh.write(b"not an engine")

ARMS = ("arm_alpha", "arm_beta")
for a in ARMS:
    make_run(os.path.join(src_root, a), a)

manifest = os.path.join(bank, "baseline_manifest_selftest.json")
rc, out = run_tool("bank", "--repo", repo, "--tag", "selftest",
                   "--binary", fake_binary, "--bank-dir", bank,
                   "--suite", "", "--out", os.path.basename(manifest),
                   *[f"--arm={a}={os.path.join(src_root, a)}" for a in ARMS])
print("\n=== bank ===")
check(rc == 0, f"bank exits 0 (got {rc})")
check(os.path.exists(manifest), "bank wrote the manifest inside --bank-dir")
check(os.path.exists(os.path.join(bank, vb.DO_NOT_DELETE)),
      "bank wrote DO-NOT-DELETE.md, so a cleanup has something to read")
check(vb.bank_root_of(os.path.join(bank, "runs", "arm_alpha")) ==
      os.path.realpath(bank),
      "a path inside the bank is recognized as banked")
check(vb.bank_root_of(tmp) is None, "a path outside a bank is not")

banked = json.load(open(manifest))
check(banked["schema"] == vb.MANIFEST_SCHEMA,
      f"manifest is schema {vb.MANIFEST_SCHEMA}")
check(os.path.isfile(os.path.join(bank, "runs", "arm_alpha", "statistics",
                                  "temperature_max.bin")),
      "the bank holds the OUTPUT BYTES, not just a path to them")
check(len(banked["arms"]["arm_alpha"]["files"]) == 3,
      "3 comparable files per arm (provenance.txt and manifest.json skipped)")

# ---------------------------------------------------------------------------
print("\n=== the bug: a bare verify must refuse, not self-compare ===")
rc, out = run_tool("verify", "--manifest", manifest)
check(rc == 2, f"bare `verify --manifest` exits 2 (got {rc})")
check("VERDICT:" not in out,
      "bare `verify --manifest` prints NO verdict at all -- the string the "
      "operator reads as a pass never appears")
check("compared them with" in out or "will not fall back" in out,
      "and it says why, naming the old behavior")

print("\n=== and the self-comparison cannot be requested by hand either ===")
banked_dir = banked["arms"]["arm_alpha"]["run_dir"]
rc, out = run_tool("verify", "--manifest", manifest,
                   f"--arm=arm_alpha={banked_dir}")
check(rc == 2, f"--arm pointed at the banked run dir exits 2 (got {rc})")
check("IS the banked run directory" in out,
      "and says the comparison would pass unconditionally")

rc, out = run_tool("verify", "--manifest", manifest,
                   f"--arm=arm_alpha={os.path.join(bank, 'runs')}")
check(rc == 2, f"--arm pointed anywhere inside the bank exits 2 (got {rc})")

# ---------------------------------------------------------------------------
print("\n=== a pristine fresh copy verifies IDENTICAL ===")
fresh = os.path.join(tmp, "fresh")
shutil.copytree(src_root, fresh)
rc, out = run_tool("verify", "--manifest", manifest,
                   *[f"--arm={a}={os.path.join(fresh, a)}" for a in ARMS])
check(rc == 0, f"pristine copy exits 0 (got {rc})")
check("VERDICT: BYTE-IDENTICAL" in out, "pristine copy reports BYTE-IDENTICAL")
check("6/6 files byte-identical" in out, "and counts all 6 files")

# ---------------------------------------------------------------------------
print("\n=== THE DISCRIMINATING ARM: one doctored byte must MISMATCH ===")
doctored = os.path.join(tmp, "doctored")
shutil.copytree(src_root, doctored)
victim = os.path.join(doctored, "arm_alpha", "statistics", "temperature_max.bin")
data = bytearray(open(victim, "rb").read())
data[40] ^= 0x01                      # one bit, in the payload, same length
open(victim, "wb").write(bytes(data))
check(os.path.getsize(victim) ==
      banked["arms"]["arm_alpha"]["files"]["statistics/temperature_max.bin"]["bytes"],
      "the doctored file is the SAME SIZE as the banked one (a length check "
      "alone would miss it)")
rc, out = run_tool("verify", "--manifest", manifest,
                   *[f"--arm={a}={os.path.join(doctored, a)}" for a in ARMS])
check(rc == 1, f"one flipped bit exits 1 (got {rc})")
check("VERDICT: MISMATCH" in out, "one flipped bit reports MISMATCH")
check("DIFF    statistics/temperature_max.bin" in out,
      "and names the file that moved")
check("IDENTICAL arm_beta" in out,
      "while the untouched arm still reports IDENTICAL (not a blanket failure)")

print("\n=== the same, inside statistics/_raw/ (the recursive walk) ===")
raw = os.path.join(tmp, "doctored_raw")
shutil.copytree(src_root, raw)
victim = os.path.join(raw, "arm_beta", "statistics", "_raw", "temperature_sum.bin")
data = bytearray(open(victim, "rb").read())
data[-1] ^= 0xFF
open(victim, "wb").write(bytes(data))
rc, out = run_tool("verify", "--manifest", manifest,
                   *[f"--arm={a}={os.path.join(raw, a)}" for a in ARMS])
check(rc == 1, f"a byte inside statistics/_raw/ exits 1 (got {rc})")
check("DIFF    statistics/_raw/temperature_sum.bin" in out,
      "and names the _raw file")

print("\n=== a deleted output file is a MISMATCH, not a silent pass ===")
gone = os.path.join(tmp, "missing")
shutil.copytree(src_root, gone)
os.remove(os.path.join(gone, "arm_alpha", "statistics", "illumination_avg.bin"))
rc, out = run_tool("verify", "--manifest", manifest,
                   *[f"--arm={a}={os.path.join(gone, a)}" for a in ARMS])
check(rc == 1, f"a missing file exits 1 (got {rc})")
check("MISSING statistics/illumination_avg.bin" in out, "and names it MISSING")

print("\n=== a mistyped arm name is a failure, never a silent skip ===")
rc, out = run_tool("verify", "--manifest", manifest,
                   f"--arm=arm_alfa={os.path.join(fresh, 'arm_alpha')}")
check(rc == 1, f"an unknown arm name exits non-zero (got {rc})")
check("no such arm in the manifest" in out, "and says so")

# ---------------------------------------------------------------------------
print("\n=== vacuity control: what the OLD code compared ===")
# Re-derive the old resolution rule and show it can never disagree: this is the
# reason the doctored arm above is the real test and a green bare `verify`
# never was.
old_style = {k: v["run_dir"] for k, v in banked["arms"].items()}
diffs = 0
for name, run_dir in old_style.items():
    now = vb.digest_run(run_dir)
    diffs += sum(1 for f, rec in banked["arms"][name]["files"].items()
                 if now.get(f, {}).get("md5") != rec["md5"])
check(diffs == 0,
      "re-digesting the manifest's own run_dirs reports 0 differences -- by "
      "construction, for any engine, at any commit. That is what the old bare "
      "`verify` printed as VERDICT: BYTE-IDENTICAL.")

# ---------------------------------------------------------------------------
print("\n=== --rerun on a manifest with no recipes refuses to guess ===")
noreipe = json.loads(json.dumps(banked))
for rec in noreipe["arms"].values():
    rec.pop("recipe", None)
noreipe_path = os.path.join(tmp, "manifest_no_recipe.json")
json.dump(noreipe, open(noreipe_path, "w"))
rc, out = run_tool("verify", "--manifest", noreipe_path, "--rerun",
                   "--binary", fake_binary, "--repo", repo)
check(rc == 2, f"--rerun without recipes exits 2 (got {rc})")
check("carries no recipe" in out, "and says the manifest predates the recipes")

print("\n=== --arm and --rerun together are refused ===")
rc, out = run_tool("verify", "--manifest", manifest, "--rerun",
                   "--binary", fake_binary, "--repo", repo,
                   f"--arm=arm_alpha={os.path.join(fresh, 'arm_alpha')}")
check(rc == 2, f"--arm with --rerun exits 2 (got {rc})")

# ---------------------------------------------------------------------------
print("\n=== the bank is write-protected, not merely documented ===")
rc, out = run_tool("run", "--repo", repo, "--binary", fake_binary,
                   "--run-root", os.path.join(bank, "scratch"),
                   "--arm", "anything")
check(rc == 2, f"`run` into a bank exits 2 (got {rc})")
check("READ-ONLY input" in out, "and explains that a bank is read-only")

# ---------------------------------------------------------------------------
print("\n=== equivalence still works, and still sees one byte ===")
rc, out = run_tool("equivalence", os.path.join(fresh, "arm_alpha"),
                   os.path.join(src_root, "arm_alpha"))
check(rc == 0 and "VERDICT: BYTE-IDENTICAL" in out,
      "equivalence of two identical runs is BYTE-IDENTICAL")
rc, out = run_tool("equivalence", os.path.join(doctored, "arm_alpha"),
                   os.path.join(src_root, "arm_alpha"))
check(rc == 1 and "NOT EQUIVALENT" in out,
      "equivalence across the flipped bit is NOT EQUIVALENT")

# ---------------------------------------------------------------------------
print("\n=== a filter that matches nothing must not report a pass ===")
# Found in review, and it is the SAME failure class as the bug above: with a
# --only that matches no banked arm, the byte loop ran over zero arms and
# printed `0/0 files byte-identical ... VERDICT: BYTE-IDENTICAL`, exit 0.
rc, out = run_tool("verify", "--manifest", manifest, "--rerun",
                   "--binary", fake_binary, "--repo", repo,
                   "--only", "arm_alfa")
check(rc == 2, f"--only with a name the manifest lacks exits 2 (got {rc})")
check("VERDICT: BYTE-IDENTICAL" not in out,
      "and never reports a pass over nothing")

print("\n=== the run that PRODUCED the bank is a self-comparison too ===")
# `bank` records source_run_dir, and the documented workflow (`run --keep`,
# then `bank`) leaves that directory on disk. Comparing it with the manifest
# is byte-identical by construction.
source_dir = banked["arms"]["arm_alpha"]["source_run_dir"]
check(os.path.realpath(source_dir) == os.path.realpath(
          os.path.join(src_root, "arm_alpha")),
      "the manifest records the run the bank was copied FROM")
rc, out = run_tool("verify", "--manifest", manifest,
                   f"--arm=arm_alpha={source_dir}")
check(rc == 2, f"--arm pointed at the bank's SOURCE run exits 2 (got {rc})")

print("\n=== re-banking must not eat the bank before it checks ===")
# copy_run_into_bank stages and swaps. A re-bank of an arm with FEWER files
# used to merge, leaving the old files in place and only then failing.
shrunk = os.path.join(tmp, "shrunk", "arm_alpha")
shutil.copytree(os.path.join(src_root, "arm_alpha"), shrunk)
os.remove(os.path.join(shrunk, "statistics", "illumination_avg.bin"))
before = vb.digest_run(os.path.join(bank, "runs", "arm_alpha"))
rc, out = run_tool("bank", "--repo", repo, "--tag", "selftest",
                   "--binary", fake_binary, "--bank-dir", bank, "--suite", "",
                   "--out", "baseline_manifest_reshape.json",
                   f"--arm=arm_alpha={shrunk}")
after = vb.digest_run(os.path.join(bank, "runs", "arm_alpha"))
check(len(after) == 2 and rc == 0,
      "a re-bank REPLACES the arm rather than merging into it "
      f"(now {len(after)} files, was {len(before)})")
check(not os.path.exists(os.path.join(bank, "runs", "arm_alpha.incoming")) and
      not os.path.exists(os.path.join(bank, "runs", "arm_alpha.superseded")),
      "and leaves no staging directories behind")
# Put the bank back the way the rest of the file expects it.
rc, _ = run_tool("bank", "--repo", repo, "--tag", "selftest",
                 "--binary", fake_binary, "--bank-dir", bank, "--suite", "",
                 "--out", os.path.basename(manifest),
                 *[f"--arm={a}={os.path.join(src_root, a)}" for a in ARMS])
check(rc == 0 and len(vb.digest_run(os.path.join(bank, "runs", "arm_alpha"))) == 3,
      "and the bank restores cleanly on a re-bank of the full arm")
os.remove(os.path.join(bank, "baseline_manifest_reshape.json"))

print("\n=== a missing --inputs-dir is caught BEFORE anything is built ===")
# `run` used to resolve {input:NAME} to a bare relative filename, BUILD the
# view-factor matrix into the repository root (minutes to hours), and only then
# raise on the argument that had been missing from the start.
suite_path = os.path.join(tmp, "suite.json")
json.dump({"schema": vb.SUITE_SCHEMA,
           "inputs": {"thing.vf": {"config": "c.yaml", "args": ["-o", "{self}"]}},
           "arms": {"arm_alpha": {"group": "cpu", "config": "c.yaml",
                                  "args": ["--vf", "{input:thing.vf}"]}}},
          open(suite_path, "w"))
rc, out = run_tool("run", "--suite", suite_path, "--repo", repo,
                   "--binary", fake_binary,
                   "--run-root", os.path.join(tmp, "runroot"))
check(rc == 2, f"`run` with no --inputs-dir exits 2 (got {rc})")
check("no --inputs-dir was given" in out, "and names the arms that need one")
check(not os.path.exists(os.path.join(repo, "thing.vf")),
      "and builds NOTHING into the repository root")

print("\n=== --rerun will not build a missing input INTO the bank ===")
withrecipe = json.loads(json.dumps(banked))
withrecipe["bank_dir"] = bank
withrecipe["suite_inputs"] = {
    "thing.vf": {"config": "c.yaml", "args": ["-o", "{self}"]}}
for name, rec in withrecipe["arms"].items():
    rec["recipe"] = {"group": "cpu", "config": "c.yaml",
                     "args": ["--vf", "{input:thing.vf}"]}
wr_path = os.path.join(tmp, "manifest_with_recipe.json")
json.dump(withrecipe, open(wr_path, "w"))
rc, out = run_tool("verify", "--manifest", wr_path, "--rerun",
                   "--binary", fake_binary, "--repo", repo,
                   "--run-root", os.path.join(tmp, "rerunroot"))
check(rc == 1, f"--rerun with a missing bank input exits 1 (got {rc})")
check("READ-ONLY input" in out,
      "refusing to build a replacement matrix into the bank")
check(not os.path.exists(os.path.join(bank, "inputs", "thing.vf")),
      "and nothing lands in the bank's inputs/")
check("48 (the count these arms were banked at)" in out,
      "and --rerun defaults to the thread count the MANIFEST banked, not 48 "
      "by coincidence")

print("\n=== a bank input whose bytes moved is exit 1, not a usage error ===")
withrecipe["inputs"] = {"thing.vf": {"path": "x", "md5": "0" * 32, "bytes": 3}}
json.dump(withrecipe, open(wr_path, "w"))
os.makedirs(os.path.join(tmp, "myinputs"), exist_ok=True)
open(os.path.join(tmp, "myinputs", "thing.vf"), "wb").write(b"abc")
rc, out = run_tool("verify", "--manifest", wr_path, "--rerun",
                   "--binary", fake_binary, "--repo", repo,
                   "--inputs-dir", os.path.join(tmp, "myinputs"),
                   "--run-root", os.path.join(tmp, "rerunroot2"))
check(rc == 1, f"a corrupted bank input exits 1 (the reference is wrong), "
               f"not 2 (you asked wrong) — got {rc}")
check("the manifest banked" in out, "and names both digests")

print("\n=== the code digest must never manufacture a hash ===")
# md5("") is a perfectly good-looking hex string, and two libraries this
# failed to read would have compared EQUAL on it -- a provenance check that
# passes precisely when it did not run.
check(vb.text_digest(os.path.join(tmp, "no-such-library.dylib")) == "",
      "text_digest of an unreadable file is empty, not md5('')")
check(vb.text_digest(fake_binary) == "",
      "text_digest of something that is not a Mach-O is empty too")
check(vb.rpath_dirs(fake_binary) == [],
      "rpath_dirs of a non-binary is empty rather than an exception")

print("\n=== fetch_kernels --link-from verifies bytes before it links ===")
ksrc = os.path.join(tmp, "ksrc")
kdst = os.path.join(tmp, "kdst", "data", "kernels")
os.makedirs(os.path.join(ksrc, "data", "kernels", "lsk"))
os.makedirs(kdst)
payload = b"a leapseconds kernel, notionally"
good = os.path.join(ksrc, "data", "kernels", "lsk", "naif0012.tls")
open(good, "wb").write(payload)
good_sha = hashlib.sha256(payload).hexdigest()

ok = fk.link_from(ksrc, kdst, [("lsk", "naif0012.tls", "url", good_sha)])
check(ok, "a matching sha256 links")
check(os.path.islink(os.path.join(kdst, "lsk", "naif0012.tls")),
      "and the link exists")

kdst2 = os.path.join(tmp, "kdst2")
os.makedirs(kdst2)
ok = fk.link_from(ksrc, kdst2, [("lsk", "naif0012.tls", "url", "0" * 64)])
check(not ok, "a MISMATCHED sha256 refuses to link")
check(not os.path.exists(os.path.join(kdst2, "lsk", "naif0012.tls")),
      "and leaves nothing behind")

kdst3 = os.path.join(tmp, "kdst3", "lsk")
os.makedirs(kdst3)
open(os.path.join(kdst3, "naif0012.tls"), "wb").write(b"a real local file")
ok = fk.link_from(ksrc, os.path.dirname(kdst3),
                  [("lsk", "naif0012.tls", "url", good_sha)])
check(not ok, "an existing real file is NOT clobbered by a link")
check(open(os.path.join(kdst3, "naif0012.tls"), "rb").read() ==
      b"a real local file", "and its bytes are untouched")

ok = fk.link_from(ksrc, os.path.join(ksrc, "data", "kernels"),
                  [("lsk", "naif0012.tls", "url", good_sha)])
check(not ok, "linking a directory to itself is refused")

missing_src = os.path.join(tmp, "empty_src")
os.makedirs(os.path.join(missing_src, "data", "kernels"))
ok = fk.link_from(missing_src, os.path.join(tmp, "kdst4"),
                  [("spk", "de440s.bsp", "url", None)])
check(not ok, "a kernel the source does not have is a loud failure")

shutil.rmtree(tmp, ignore_errors=True)

if _fail == 0:
    print("\nALL ptm_validate_baseline SELF-TESTS PASSED")
else:
    print(f"\n{_fail} ptm_validate_baseline SELF-TEST(S) FAILED")


def test_ptm_validate_baseline():
    """The verdict, for a collector rather than a shell."""
    assert _fail == 0, f"{_fail} ptm_validate_baseline self-test(s) failed"


if __name__ == "__main__":
    sys.exit(1 if _fail else 0)
