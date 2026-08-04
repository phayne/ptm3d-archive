# PTM3D — Known Limitations and Gotchas

This document lists behaviors that can surprise users of PTM3D. None are correctness bugs in the
core solver; they are modeling limitations, configuration gotchas, and workflow caveats. Reading
the run log resolves several of them — always confirm what the run actually loaded (a mesh-load
failure now aborts rather than silently substituting a default icosphere).

**`PTM-nn` identifiers** appearing in source comments and in the notes below reference the
project's internal issue ledger, which is not distributed. The limitations relevant to released
functionality are summarized in this file; no lookup elsewhere is required to use PTM3D.

## Configuration gotchas

- **Most of the `thermophysical:` block is not read, and the run now says so.** On a *built-in*
  body (Moon/Europa/Mercury) the parser consumes only `use_default_albedo`,
  `constant_thermal_inertia`, `albedo_override`, `emissivity_override`, `ks_override`,
  `kd_override`, `incidence_angle_albedo`, `albedo_model`, `albedo_hapke_w` and
  `conductivity_model`. Plain-named properties (`ks`, `kd`, `H`, `rhos`, `rhod`, `cp0`, `Qb`,
  `chi`, `albedo`, `emissivity`) and every `use_default_*` flag other than `use_default_albedo`
  are **decorative there** — they are honored only under `generic_body.surface:`. Editing them on
  a named body changes nothing. As of v1.0.3 the engine prints one `[config] WARNING:` line naming
  every such key it found, instead of dropping them in silence; **it does not start honoring
  them**, because doing so would change results for the 130+ shipped configs that carry them.
  There is still no config route to a non-default `H` on a named body except `h_parameter_map`.
  (PTM-67)

- **A malformed config file is now fatal.** Previously a YAML parse failure — a bad scalar, a
  malformed document, a rejected enum value — was downgraded to a `Warning:` and the entire
  simulation ran on built-in defaults (default body, mesh, timesteps, properties **and output
  directory**) exiting 0, so no orchestrator could detect it. `--config` with an unloadable file
  now prints `FATAL:` and returns 1. If you relied on the old behavior, omit `--config`
  deliberately rather than passing a broken one. (PTM-66)

- **`constant_thermal_inertia > 0` silently disables the T-dependent conductivity term (`chi`).**
  Selecting a constant thermal inertia forces `chi = 0` (and collapses `kd→ks`, `rhod→rhos`) by
  design, so a `chi` you also set is dropped without warning. To use a real `chi > 0`, set
  `constant_thermal_inertia: 0` and specify `ks/kd/rhos/rhod/cp0` directly.

## Illumination

- **Near-terminator direct illumination can be non-monotonic.** At grazing solar incidence (~90°)
  the finite-solar-disk model can yield small, non-monotonic direct-flux values on near-terminator
  faces (order ≤1 W/m²). Benign for energy balance, visible only in per-face direct flux at the
  terminator.

- **Residual "crater-center flash" on non-watertight meshes.** Grazing sunlight can thread small
  interior mesh holes and briefly light floor faces. The interior-hole seal
  (`PTM_ILLUM_SEAL_HOLES=1`) closes watertight interior hole loops, but **open and non-manifold
  hole components are left unsealed**. Ensure meshes are watertight for micro-relief floor studies;
  the seal is a mitigation, not a guarantee.

## View factors / radiation

- **Deep permanently-shadowed floors remain slightly under-coupled to warm walls.** Even with the
  corrected view-factor operator, radiative coupling into the deepest cold-trap floors is a small
  underestimate (a radiative-equilibrium effect of order ~1–2 K for most floors, less for the
  deepest). Relevant when comparing modeled cold-trap floor temperatures to observations.

- **Do not rebuild the `.vf` from a mesh that changed after the view factors were built.** The
  view-factor builder uses the mesh's raw vertex ordering ("winding"); if the mesh face count or
  winding differs from the one the view factors were built against, radiation exchange is silently
  wrong (or silently does nothing). Reuse the paired mesh + view-factor file together, and confirm
  the load log shows matching dimensions **and Non-zeros > 0** (a nonzero count of stored matrix
  entries).

- **Low-rank (ACA) far-field view factors can carry tiny negative entries.** A physical view factor
  is ≥ 0, but the low-rank far-field approximation does not preserve non-negativity, so near-zero
  true entries reconstruct with small ± sign noise. The net row-sum effect is negligible on smooth
  production meshes (~−3e-5), but can be larger on steep/rough geometry. Clamp reconstructed `F ≥ 0`
  or tighten the ACA tolerance if it matters for your case.

- **The view-factor occlusion test uses meter-scale thresholds — coarse for sub-meter meshes.** The
  visibility test starts rays a fixed ~0.5 m offset and auto-passes pairs closer than ~12.5 cm. On
  sub-meter meshes (face edges ≪ 1 m) most of the near field auto-passes, so genuinely-occluded near
  pairs can still exchange radiation. Affects fine (sub-meter) meshes; the corrected-VF operator
  mitigates the net magnitude.

- **GPU two-band radiosity is limited to meshes with Σ_j F_ij ≤ 1.** The GPU Jacobi kernel does not
  apply the row-sum renormalization, so on meshes whose centroid view-factor row sums exceed 1 the
  solver warns and falls back to the (always-protected) CPU two-band solver. The CPU path is
  unrestricted.

## Output format

- **`statistics/*.bin` files begin with a 4-byte header.** The layout is
  `[int32 n_faces][float32 × n_faces]`. A custom reader must consume the leading `int32` (the
  shipped `scripts/ptm_qa.py` reader does); reading the file as raw `float32` shifts every face by
  one index and mislabels the first value.

## Physical constants

- **Saturn-shine and Mars-shine changed in v1.0.3; this is a correction, not a regression.** The
  four primary-body `(TEMPERATURE, IR_FLUX)` constants are consumed as *emitter-surface* fluxes and
  must therefore satisfy `IR_FLUX = σT⁴`. Three of the four did not. `SATURN_IR_FLUX` was 15.0
  W/m² — almost exactly the *solar* constant at Saturn (1361/9.58² = 14.83), i.e. a solar value
  entered for thermal emission — and is now σ(95 K)⁴ = **4.62 W/m² (÷3.25)**. `MARS_IR_FLUX` was
  70.0 and is now σ(210 K)⁴ = **110.3 W/m² (×1.58)**. Any run with Saturn or Mars as the primary
  will produce different primary-IR heating than a pre-v1.0.3 build. **No published PTM3D result
  is affected**: every campaign to date used Jupiter (flux unchanged) or the Sun alone, and no
  Saturn- or Mars-primary baseline exists to regress against — so this is validated by arithmetic,
  not by comparison. Earth was already consistent (0.1%); Jupiter's *temperature* constant moved
  124.4 → 124.90 K purely to make its pair self-consistent, and is read by nothing. A load-time
  check now warns if any pair drifts more than 5% apart. (PTM-43)

## Workflow / tooling

- **The chunked runner can pick an out-of-date executable.** `run_chunked_simulation.py` may prefer
  a build in `~/dev/heat3d/ptm-build` if that directory exists, even if it is older than your
  in-tree build.
  Pass `--ptm-bin` / `--merge-bin` explicitly to pin the binary you built.

- **The A3GT batch tools depend on an external mesh helper.** `a3gt_v2_orchestrator.py` and
  `run_a3gt_batch.py` expect a `dtm_to_ply_delaunay` binary on a fixed path; provide it (or adjust
  the path) before running an A3GT batch.
