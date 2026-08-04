# PTM3D — Known Limitations and Gotchas

This document lists behaviors that can surprise users of PTM3D. None are correctness bugs in the
core solver; they are modeling limitations, configuration gotchas, and workflow caveats. Reading
the run log resolves several of them — always confirm what the run actually loaded (a mesh-load
failure now aborts rather than silently substituting a default icosphere).

## Configuration gotchas

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

## Workflow / tooling

- **The chunked runner can pick an out-of-date executable.** `run_chunked_simulation.py` may prefer
  a build in `~/dev/heat3d/ptm-build` if that directory exists, even if it is older than your
  in-tree build.
  Pass `--ptm-bin` / `--merge-bin` explicitly to pin the binary you built.

- **The A3GT batch tools depend on an external mesh helper.** `a3gt_v2_orchestrator.py` and
  `run_a3gt_batch.py` expect a `dtm_to_ply_delaunay` binary on a fixed path; provide it (or adjust
  the path) before running an A3GT batch.

## GPU backends

- **The HIP and CUDA backends are UNTESTED. Only the OpenCL backend is exercised.** Every PTM3D
  result to date was produced on macOS with `GPU_BACKEND=OPENCL`, whose kernel is the string in
  `thermal_opencl.cpp::getEmbeddedKernelSource()`. `src/cpp/gpu/thermal_kernels.cpp` is compiled
  **only** by the HIP/CUDA paths, so nothing in it has ever executed here and there is no baseline
  to regress it against. In v1.0.3 its `computeK()` was corrected from `k = kc + χ(T/350)³` to the
  Hayne multiplicative form `k = kc·(1 + χ(T/350)³)` that the CPU, `.cl` and embedded-OpenCL paths
  all use — the additive form was ~700× too large at lunar noon and would have made any HIP/CUDA
  run silently and grossly wrong. **This correction is textual and could not be runtime-tested on
  the development machine (no HIP/CUDA hardware).** Anyone bringing up a HIP or CUDA host should
  validate against the CPU solver before trusting results, and should treat the whole file as
  unverified, not just this function. (PTM-62)

- **`src/cpp/gpu/thermal_kernels.cl` is a generated mirror, not the kernel that runs.** The OpenCL
  solver compiles the embedded string in `thermal_opencl.cpp::getEmbeddedKernelSource()`; editing
  the `.cl` file by hand changes nothing at runtime and will be overwritten. The `.cl` is
  regenerated verbatim from that string by `scripts/sync_cl_from_embedded.py`, and CI enforces
  `--check`, so it does track the live kernel (including `conductivity_model: martinez2021`) — but
  read it as a mirror, and make kernel edits in `thermal_opencl.cpp`.

- **`src/cpp/gpu/radiosity_kernels.cl` has no such mirror check and has drifted. (PTM-70)** The
  radiosity solver likewise compiles an embedded `R"CL_SOURCE(...)"` string
  (`radiosity_opencl.cpp::buildKernels()`), but `scripts/sync_cl_from_embedded.py` and the CI check
  cover only the thermal pair, and the on-disk `radiosity_kernels.cl` is now ~2 kB divergent from
  the string that actually compiles. Unlike the thermal case there is **no** `kernel_path` /
  external-load path in the radiosity solver, so a stale file cannot swap in wrong physics at
  runtime — but the file is copied to the build tree and installed to `share/ptm/kernels`, where a
  reader would reasonably take it for the kernel. Treat it as out of date until the sync tool is
  generalized to both pairs.
