# GPU-wedge guardrails

On macOS with AMD hardware a long PTM3D GPU run can wedge the GPU hard enough
to take the whole machine down: the compute side stops, the OS keeps running
for a while, and the box then power-cycles with an SMC shutdown cause and no
usable log. This page documents the guardrails the engine and the chunked
runner apply, what each one costs, and which platform each one is active on.

**Everything here is platform-gated.** The failure was measured on macOS with
AMD (Navi21) devices. Guardrails that cost wall time or change a solver rung
*enforce* only there; on every other host they are advisory — they print what
they noticed and get out of the way. An explicit setting (a config key written
down, or the environment variable set) enforces everywhere, because an operator
who asks for a policy should get the policy.

---

## 1. The failure, in one paragraph

Two power-offs in 24 h during the A3GT gen-2 campaign (2026-08-20 SMC `-128`,
2026-08-21 SMC `-108`). Both died **inside a GPU thermal dispatch while a
multi-GB radiosity view-factor replica was resident on all four dies** — not
during the upload, which the surviving log proves completed. One of them had
10.4 GiB of VRAM margin, so margin alone is not the story; the one condition
common to both events is the *co-residency* of the radiosity replica and the
thermal solver's working set on the same devices. A third contributing factor
was a mitigation that had been inert for months: the run printed "Capping
re-thermal concurrency at 1 device(s)" and then ran the re-thermal on four,
rebuilding four OpenCL contexts every coupling iteration — the exact churn the
mitigation existed to remove. (The full forensic write-up is an internal
session note and is not shipped with the repository; everything the guardrails
rest on is summarized here.)

### Wedge signature — how to recognize one

- The job's log stops mid-phase and **never resumes**; the process is still
  alive and consuming little or no CPU.
- `ps` shows `ptm_thermal_sim` present; the output directory stops growing.
- The OS may keep logging for tens of minutes after the compute freezes. In the
  2026-08-20 event the last system event of any kind, 36 minutes later, was
  `Display is turned off`.
- After the reboot, `pmset -g log | grep -i "shutdown cause"` reports a negative
  SMC code. `-108`, `-128` and `-112` are community-attributed to memory-related
  causes; Apple publishes no table, and `powerd` prints the code with an empty
  description. Treat the decode as unverified.
- `log show` needs `sudo` on a stock macOS install. **`pmset -g log` does not**,
  and it keeps about a week of history including `ShutdownCause` and every
  display on/off transition. It is the only unprivileged power forensics
  available; use it first.

### Recovery

1. Power-cycle the machine (a wedged GPU usually cannot be recovered in place).
2. `pmset -g log | grep -i "shutdown cause"` — record the code and time.
3. Delete the partially written window/chunk directory. A chunk without a
   `statistics/_raw/` sum+count pair is incomplete; the runner's `--resume`
   skips only complete chunks, so an incomplete one is re-run automatically
   once its directory is gone.
4. Re-launch with `--resume`. The per-attempt logs of the failed attempt are in
   `<output-dir>/logs.prev-<UTC>/` — do not delete them; they are the only
   record of what the run was doing.
5. Apply the power-settings item in the INSTALL checklist if it is not applied
   already.

---

## 2. The guardrails

### G1 — the re-thermal pool is hoisted by default

*Engine, all platforms.* The radiation-coupling re-thermal builds its GPU
thermal pool **once** per run instead of once per coupling iteration. Kernel
compiles on the 12,800-face crater arm fall from 20 to 9.

Cost: none. Concurrency is unchanged (the configured device count), because
that is what production was already running — the old "cap to 1" message was
inert. `PTM_RAD_RERUN_MAX_GPUS=N` lowers the width; `=1` genuinely serializes
the re-thermal on one device, which is the May-2026 mitigation, at an
unmeasured wall cost.

### G2 — VRAM-margin floor (`gpu.min_vram_margin_gib`, default 2.0)

*Engine.* Enforced on macOS+AMD or when set explicitly; advisory elsewhere.

A radiosity CSR replica that fits with less than this much headroom per device
is refused, and radiosity runs on the CPU. The reserve the fit check leaves is
one lump shared with every other consumer of the same die — above all the GPU
thermal pool, which allocates ~0.77 GiB/device at A3GT production settings
(`gpu_batch_size=100000`, 23 layers, 961 steps) on the very dies holding the
replica, and which the fit check does not count.

Cost: a site in the thin band loses GPU radiosity, and on a large matrix that
is expensive. **Measured on A3GT A03N1016** (1,798,135 faces, 3.5 G nnz), eight
standard windows per generation:

| rung | windows | mean wall | range |
|---|---|---|---|
| CPU radiosity (gen-1, replica refused) | 8 | **108.9 min** | 103.4-115.5 |
| multi-GPU pool (gen-2, replica accepted) | 8 | **31.4 min** | 28.5-37.2 |

**About +78 min per window, a factor of 3.5** — and about +71 min after
correcting for the 9.2% larger matrix gen-1 carried, if the CPU sweep scales
with nnz. Do not confuse this with the ~3 min the *GPU* sweeps themselves take:
that is what radiosity costs when it is on the GPU, not what the CPU
alternative costs. The GPU sweep is ~10% of the window; the CPU sweep is not.

That is why the ordering below exists. Set `gpu.min_vram_margin_gib: 0` for the
pre-2026-08-22 behavior (fit check only).

The engine also records what this host has actually carried
(`~/.ptm3d/vram_profile.json`) and prints it once per run: *"largest replica
committed here 26.55 GiB/device on 4 device(s), leaving 0.64 GiB of margin"*.
That line is what nobody had when a physics fix silently moved a site onto the
largest VRAM commitment the machine had ever made.

#### Two ways to treat a thin margin (`gpu.thin_margin_policy`)

The floor and the placement policy address the **same** risk — the radiosity
replica sharing dies with the GPU thermal pool — by different means, at very
different prices. `gpu.thin_margin_policy` decides which one a thin margin
triggers:

| | `refuse` (default) | `separate` |
|---|---|---|
| below `min_vram_margin_gib` | give up the GPU rung | keep it, **but only when the two pools are on disjoint dies** |
| refuses only below | — | `hard_min_vram_margin_gib` (default 0.25 GiB) |
| escape below the hard floor | — | this host's own VRAM record shows it already committed a replica this large |
| cost for A03N1016 (+0.64 GiB) | **+78 min/window** (measured) | **+7-10 min/window** (derived) |
| removes co-residency? | yes, by removing the GPU rung | yes, by separating the dies |

`separate` never waives the floor on a rung where the separation could not
actually be applied — the single-GPU rung cannot choose its device, and a
one-GPU host cannot be split. In those cases it refuses and says which
mitigation was missing. A floor waived for a mitigation that is not in force
would be a guardrail that reports itself active while doing nothing, which is
the failure this whole page exists to remove.

**The default is `refuse`** until the trade is ruled on: it is the
conservative one, and it is what the rest of this branch was validated with.
Switching an N1016-class site to `separate` is the difference between a
109 min window and a ~39 min window.

### G3 — pool placement (`gpu.placement_policy`, default `auto`)

*Engine.* `auto` engages on macOS+AMD only; `disjoint` is obeyed everywhere.

When the radiosity replica will be resident, the GPU thermal pool would share
those devices, and the post-fit margin is below `gpu.placement_margin_gib`
(default 4.0), the engine hands the two pools **disjoint device sets** —
radiation `[0,1]`, thermal `[2,3]` on a 4-die box. No device then holds the
replica and a thermal working set at the same time, which is the one condition
common to both wedges. The decision is printed on the `[Radiosity] ACTIVE RUNG`
line.

Cost: **derived, not measured** — both pools lose devices, roughly +7-10 min on
a ~30 min A3GT window (radiosity sweeps go from N=4 to N=2, ~+2 min; the
thermal phases likewise, ~+5-8 min).

Two deliberate refusals:

- `auto` will **not** overrule explicitly pinned `gpu_device_ids` on either
  side. It warns and leaves the operator's choice alone. Use `disjoint` to
  override.
- The single-GPU radiosity rung does not choose its own device (OpenCL does),
  so the separation cannot be applied there. The engine says so instead of
  sharing the die quietly. Set `radiation.gpu_devices: 2` to get a rung that
  honors it, or pin non-overlapping ids yourself.

A decision that cannot be honored end to end is **reverted out loud** rather
than half-applied: if the split would leave the radiosity pool with fewer than
two devices it would degrade to the single-GPU rung, and the thermal pool would
have been moved off dies the radiosity solver never left.

### G4 — macOS power-settings advisory

*Engine, macOS only, advisory always.* Before a GPU run the engine reads
`pmset -g custom` (unprivileged, read-only) and warns loudly when `displaysleep`
or `sleep` is nonzero, because a display or system sleep transition issues a GPU
modeset — the documented trigger of the 2026-06-23 `-128` wedge on this
hardware, and the last logged event before the 2026-08-20 one.

Fix, once, with sudo:

```bash
sudo pmset -a disablesleep 1 displaysleep 0     # before a campaign
sudo pmset -a disablesleep 0 displaysleep 30    # after
```

Cost: none. It changes nothing and needs no privileges. When sleep and
displaysleep are already 0 but `disablesleep` is not set, it prints a one-line
note rather than a warning.

### G5 — runner: cool-down, stall abort, rotated logs

*`scripts/run_chunked_simulation.py`.*

| Guardrail | Default | Gate |
|---|---|---|
| Inter-chunk cool-down | 300 s | GPU config **and** macOS; 0 otherwise (`--gpu-cooldown-s`) |
| Stall abort | 45 min of silence | any GPU config (`--stall-min`, 0 = off) |
| Per-attempt rotated logs | on | every host (`--no-log-rotate`) |

- The **cool-down** is prophylaxis against the documented *progressive* wedge
  (roughly 30 clean windows, then degradation). Cool-downs do not clear a
  degraded driver; they slow its accumulation. Not a proven fix. Cost: 5 min
  per chunk boundary.
- The **stall abort** kills the engine's whole process group when it writes
  nothing at all for the timeout, and counts the attempt as a retryable
  failure. Silence, not slowness, is the wedge signature; waiting out
  `--chunk-timeout-min` only holds the GPU longer. Cost: none on a healthy run,
  which writes progress continuously — **but a run with a legitimately silent
  phase longer than the timeout would be killed for no reason**, and that costs
  the whole attempt. The phase to watch is a very large view-factor matrix read
  from slow or cloud-backed storage. Raise `--stall-min` if that is your
  situation; the 45 min default is chosen well above every silent phase
  measured on the A3GT windows (the longest is the ~13 min illumination pass,
  which is not silent at all).
- **Rotated logs** move an existing `<output-dir>/logs/` to
  `logs.prev-<UTC>` at startup and write one log per attempt
  (`chunk_NNN_attempt_M.log`). Attempt numbering restarts on a rerun, so
  without this a rerun overwrites exactly the record you need. A campaign
  driver overwriting its window log destroyed the entire forensic record of the
  2026-08-20 wedge within 16 h. Cost: disk, and only as much as the logs.

### G6 — do NOT change

`gpu_batch_size` and `gpu_inter_batch_pause_ms` are not implicated. Measured
dispatches were healthy (~1.1 s, well under the ~2 s driver ceiling) right up to
the moment of the freeze, and shrinking the batch only multiplies the number of
driver interactions. The per-dispatch timestep ceiling (`--calibrate-gpu`,
`PTM_MAX_TIMESTEPS_PER_DISPATCH`) is a *different* and still-necessary
mitigation for the `gpusKillClientExt` command-buffer watchdog; leave it alone.

---

## 3. Platform-gating table

| Guardrail | macOS + AMD | macOS + other GPU | Linux / other |
|---|---|---|---|
| G1 hoisted re-thermal pool | on | on | on |
| G2 VRAM-margin floor | **enforces** | advisory | advisory |
| G2 when set explicitly | enforces | **enforces** | **enforces** |
| G3 `placement_policy: auto` | separates when thin | note only | note only |
| G3 `placement_policy: disjoint` | separates | separates | separates |
| G4 power advisory | warns | warns | inert (no `pmset`) |
| G5 runner cool-down | 300 s default | 300 s default | 0 s default |
| G5 stall abort | on for GPU runs | on for GPU runs | on for GPU runs |
| G5 log rotation | on | on | on |

The engine detects "AMD" from the OpenCL device vendor string and macOS at
compile time. `PTM_GPU_FORCE_WEDGE_PLATFORM=1|0` forces the gate either way for
testing and announces itself.

---

## 4. What the GUI and a plain CLI run get

The GUI, `ptm run` and the notebook helper all launch **one**
`ptm_thermal_sim`. They therefore get every engine-side guardrail (G1-G4)
automatically, and none of the runner-side ones (G5), which only exist between
chunks. When such a run is long (>= 30 simulated days or >= 2000 timesteps) and
uses the GPU, all three paths print an advisory naming the guardrails they are
not getting and the chunked-runner command that provides them.

They are not rerouted automatically: the chunked workflow produces merged
statistics and no per-step time series, which is a decision for the user, not
for a warning.

---

## 5. Unattended resume after a power cycle (macOS, optional)

`scripts/com.ptm3d.campaign-resume.plist.template` is a launchd template that
runs a resume command once at login after a power-off. It is a **template,
documented and not installed**: nothing in the build or test path touches it,
and nothing depends on it.

```bash
cp scripts/com.ptm3d.campaign-resume.plist.template \
   ~/Library/LaunchAgents/com.ptm3d.campaign-resume.plist
# edit __RESUME_COMMAND__ and __LOG_DIR__, then
launchctl load  ~/Library/LaunchAgents/com.ptm3d.campaign-resume.plist
launchctl list | grep ptm3d
# to remove
launchctl unload ~/Library/LaunchAgents/com.ptm3d.campaign-resume.plist
rm ~/Library/LaunchAgents/com.ptm3d.campaign-resume.plist
```

A worked `__RESUME_COMMAND__` (one line, no line breaks in the plist string):

```
cd /path/to/ptm && python3 scripts/run_chunked_simulation.py --config configs/my_campaign.yaml --total-days 30 --output-dir /path/to/output --resume
```

Think before installing it. A campaign that resumes itself unattended can also
**re-wedge itself unattended**, and every wedge is an unclean power-off of the
whole machine; two or three of those in a night is not obviously better than
one. Apply the power-settings item first, and put a relaunch budget ("stop
after N resumes") in the resume script rather than in launchd — `KeepAlive` is
deliberately false in the template, so launchd will not restart a failing
command, but it will happily run it again at the next login. A chunk killed by
a power-off is incomplete and must be **deleted and re-run**, not resumed into;
have the resume script check for that first.

---

## 6. Related documents

- [configuration.md](configuration.md) — the `gpu:` section reference
- [ENVIRONMENT.md](ENVIRONMENT.md) — `PTM_GPU_MIN_VRAM_MARGIN_GIB`,
  `PTM_RAD_RERUN_MAX_GPUS`, and the two test hooks
- [../INSTALL.md](../INSTALL.md) — the GPU checklist, including the power
  settings
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — what else is known to be
  wrong, with measured magnitudes
