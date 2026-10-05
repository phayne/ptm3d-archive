# Coupled Dynamic Simulation: Darkening + PTM3D + Full-Tensor Polar Wander

> **Status (2026-10-04): the numbers in this document are superseded.** They were
> computed before the 2026-10 corrections to the shell and polar-wander code: the
> thickness law is now O&S Eq. 2.8 with $C_R = 0.5$ (the earlier solver dropped a
> factor of 2 and was tuned to $C_R = 0.85$); every harmonic projection and inertia sum
> now uses a periodic grid (the earlier grids counted the λ = ±180° meridian twice); the
> polar-wander dynamics follow OSPW Eqs. 2.11, 2.18, 3.8–3.10 and 5.39; and the
> pole is seeded with a physical offset $\psi_0$. The inputs are also March-2026 PTM3D
> products. Regenerate before quoting any value.
>
> **2026-10-05 dynamics changes** (Fermi review of the dynamic TPW runs; each has a test in
> `src/python/tests/test_ojakangas_stevenson.py` and an option that restores the old path):
> the pole step includes the tidal torque from Jupiter (`tidal_torque`); the ψ₀ seed also
> fires on a saddle (`seed_rearm`); the shell relaxes on the Stefan time ρLb/q (`relaxation_mode`);
> the sub-Jovian meridian can follow the minimum-moment axis (`meridian_mode`); and
> `albedo_mode='split'` adds the exogenic/intrinsic albedo split (§1.1). With all four
> switches on their old paths the 2026-10-05 `run_tpw.py` runs reproduce bit for bit.

A time-stepping simulation that closes the loop between trailing-hemisphere darkening,
3-D-thermal surface temperature, O&S equilibrium shell thickness, local thermal-diffusion
relaxation, and full-inertia-tensor OSPW reorientation.

*Implemented in `scripts/coupled_shell_simulation.py` (subclasses `scripts/polar_wander_cycle.py`).
Outputs in `plots/coupled_shell/<tag>/`. Run with the `europa_viz` env (needs scipy + imageio).*

---

## 1. The coupled loop

Each step:

1. **Darken** — `f(t+dt) = 1 − (1−f)·e^{−dt/τ_dark}`; albedo `A(θ,λ) = A_bright + (A_dark−A_bright)·f·w`
   with an **apex-centered cosine spot** `w = max(0, cos γ)` (γ = angular distance from the
   trailing-hemisphere apex, in the spin frame). Endmembers **A_bright = 0.5 → A_dark = 0.3814**
   (the Hapke non-ice endmember). `τ_dark` is set with `--tau-dark`; values from 10⁴ to 10⁷ yr
   are supported (the Pwyll constraint of B&H 2021 is ≳ 1 Myr).
2. **Temperature (PTM3D-coupled surrogate)** —
   `T(θ,λ) = [(1−A)·F_A(θ_t)/σ]^{1/4} + ΔT_PTM(θ_t,λ_t)`, where ΔT_PTM is the
   **PTM3D−O&S correction** (eclipse dip + thermal-inertia polar warming), held fixed in the
   tidal frame and rotated with the pole. The PTM3D temperature is the **arithmetic time mean**
   ⟨T⟩ of the snapshots — the shell's top boundary condition — not the radiative-equivalent
   (⟨T⁴⟩)^{1/4}, which is warmer and shifts $t_{20}+2t_{22}$ by ~0.3 km.
3. **Equilibrium thickness** — O&S Eq. 2.8,
   `b = ln(T_M/T_s) / [(2/a₁)D + (H/a₁)²]^{1/2}`, with $C_R = 0.5$
   (`ojakangas_stevenson.shell_thickness`; `--thickness-method exact` integrates the first
   integral of Eqs. 2.4/2.5 instead). D is re-rotated when the pole drifts.
4. **Relax at the local Stefan time** (default, `relaxation_mode='stefan_sensible'`) —
   `b ← b_eq + (b−b_eq)·e^{−dt/τ}`, `τ(θ,λ) = (1 + S/L) ρ L b / q = (1 + S/L) ρ L b² / F`, with L the
   latent heat of fusion, `F = a₁ ln(T_M/T_s) + a₀ (T_M − T_s)` the conductive flux integral and S the
   sensible heat of the steady conductive profile per unit mass, computed exactly per location
   (`sensible_heat_ratio`; S/L = 0.53 / 0.50 / 0.48 at T_s = 90 / 100 / 110 K for k = 651/T,
   c_p = 90 + 7.49 T). τ ≈ 20 Myr for b = 30 km at 100 K, within OSPW's "≲ a few × 10⁷ yr" (p. 247).
   Not included: the change of internal tidal heating with b. Options: `'stefan'` (latent heat only,
   ≈ 13 Myr) and `'diffusive'` (the earlier `τ_diff = b²/κ`, 4–5× too fast).
5. **Reorient (OSPW Eq. 5.39)** — the physical inertia tensor is the Eq. 2.18 bottom-loaded
   (buoyantly compensated, iceberg-effect) tensor of the thickness field plus, optionally
   (`--fossil hydrostatic`), the body-fixed hydrostatic fossil bulge of Eqs. 3.8–3.10. The pole
   moves by the overdamped law `ψ̇ = (t/τ)(n²I₀/2C_f) sin 2ψ`, with `(t/τ)I₀` the difference
   between the moment about the target (maximum) axis and the moment about the spin axis and
   `C_f` from Eq. 5.15. When the spin axis stops being the maximum-moment axis, the pole is
   given a physical offset ψ₀ (`--psi0`; default 0.1° ≈ 1.75 × 10⁻³ rad, Europa's predicted
   forced obliquity, Bills 2005) instead of waiting for round-off. The seed applies whenever the
   pole sits within ψ₀ of any non-maximum axis, a saddle included (`seed_rearm='any_axis'`); it
   re-arms after the pole has moved ≥ max(1°, 5ψ₀), so repeated kicks cannot become a drift.
   The torque includes the **tidal torque** −3n²(I_j − I_k) (OSPW Eqs. 5.19–5.21), so the
   overdamped flow is `ṡ = (n²/C_f)[I s − (s·I s) s + 3 (e·I s) e]` (e = sub-Jovian direction):
   a tilt toward Jupiter is resisted with stiffness 4n²(C − A) (Eqs. 5.26/5.38), a tilt toward
   the trailing apex with n²(C − B). Integrated with RK4 substeps (`overdamped_step_tidal`).
   The sub-Jovian meridian is body x projected into the equatorial plane
   (`meridian_mode='projection'`), or the minimum-moment axis in that plane
   (`'min_moment'`, quasi-static libration under the tidal torque).
6. The darkening apex and tidal heating **rotate with the new pole**; repeat.

`--albedo-mode static_map` replaces the darkening with a **body-fixed** albedo field (the measured
eq36 map, or the uniform control) that moves with the shell as the pole wanders. The surface
temperature is then `T = [(1−A)F_A(θ_t)/σ]^{1/4} + ΔT(θ_t,λ_t) + R(θ,λ)`, with ΔT from the eq36
uniform fields (tidal frame) and R a body-fixed residual that reproduces the measured eq36 field
at the initial pole (`forcing_file`, built by `build_forcing.py`).

### 1.0 Convection bracketing proxy (`conv_s`, `conv_fD`)

The paper model stays the conductive O&S shell. The shell is near the onset of stagnant-lid
convection (Fermi, 2026-10-05), so two knobs bracket its effect on the forced thickness anomaly:
`conv_fD` reweights the tidal-heating (D) part of the equilibrium anomaly,
`b_f = B(T_s, ⟨D⟩) + f_D [B(T_s, D) − B(T_s, ⟨D⟩)]`, leaving the surface-temperature response
(the same per kelvin in a stagnant lid as in O&S) unchanged; `conv_s` damps the whole anomaly by
steady basal flow, `b_eq = ⟨b_f⟩ + s (b_f − ⟨b_f⟩)`, so every degree ≥ 1 harmonic scales by s
exactly. Bracket: s ∈ {1, 0.4, 0.1, 0.03}, f_D ∈ {1, 0.4}. The relaxation time is not changed
(`conv_flow_tau` would also scale it by s; off, not ruled). With a fossil the Eq. 3.12 verdict
flips at s* = threshold / (t₂₀ + 2t₂₂); without one it cannot flip for any s > 0 (only the rate scales).

### 1.1 Split albedo (`albedo_mode='split'`)

`A(x,t) = clamp(A_intr(x_body) + A_exo(x,t), 0.3814, 0.80)`, with
`∂A_exo/∂t = [A_exo^eq(θ_t, λ_t) − A_exo]/τ_exo` (`tau_exo_yr`: 0 = instantaneous tidal frame,
∞ = body-fixed). The exogenic template `A_exo^eq = c₁ cos γ` (γ = angle from the trailing apex)
is the Burnett & Hayne (2021) / Nelson et al. (1986) form; it is north–south symmetric and fitted
to the measured eq36 map at t = 0 (tidal frame = body frame); its constant goes to A_intr, so
`A_intr = A_meas − A_exo^eq(t=0)` and the full split reproduces the measured map at t = 0.
`split_component` selects full, exogenic only (A_intr = uniform A_unif) or intrinsic only.
Options `exo_template='apex_l2'` (adds P₂(cos γ)) and `'bh2021_zonal'` (adds a spin-frame
P₂(cos θ_t) latitude term, not exogenic in origin) are for sensitivity tests.
`exo_template='exo_file'` uses the exosphere-model forcing (`exo_profile_fit`): S(θ) about the spin
pole plus T(γ) = Σ cₙ Pₙ(cos γ) in tidal coordinates (E1); E2 adds the B&H cos γ implantation term
so the total cos γ coefficient equals the map's; E3 also takes the map's own zonal profile as
spin-frame.
See `ojakangas_stevenson/exogenic_albedo.py`. Initial-condition modes for the hypothesis test:
`init_b_mode` `uniform_equilibrium` / `equilibrium` / `blend`, `init_l2_perturb_km`, and
`init_pole_offset_deg`.

**Multi-rate stepping** keeps the 100-Myr-class run tractable: a fine step (τ_dark/20) during the
short darkening ramp, a coarse step afterward, and an **adaptive cap** that limits the pole rotation
to ≤3°/step during active reorientation (the OSPW rotation is fast). A full ICO5 PTM3D re-run to
refresh ΔT_PTM is wired (`use_live_ptm3d`) but **not active in this Phase-1 deliverable** — ΔT_PTM is
static (see §5).

**Initial condition:** uniform shell thickness (= area-mean f=0 equilibrium), **no fossil
bulge** by default, uniform albedo 0.5.

---

## 2. Key results (Maxwell H=0)

| Quantity | τ_dark = 1e4 yr | τ_dark = 1e5 yr |
|----------|:---------------:|:---------------:|
| TPW onset | **~2.4 Myr** | **~2.5 Myr** |
| ⟨b⟩ (initial → final) | 19.80 → 19.04 km | 19.80 → 19.04 km |
| Post-onset behavior | continuous TPW | continuous TPW |

- **Without a fossil bulge, the pole is unstable to darkening.** The shell is stable for the first
  ~2.4 Myr; once the trailing apex has thinned enough, the **max-inertia axis flips from the spin
  pole (ẑ) to the trailing apex (ŷ)** and the pole reorients.
- **Onset is gated by the shell thermal-relaxation timescale τ_diff (~few Myr), NOT by τ_dark.**
  The darkening saturates fast (≈5 τ_dark ≤ 0.5 Myr) relative to τ_diff, so a 10× change in τ_dark
  barely moves the ~2.4–2.5 Myr onset. The bottleneck is how fast the shell can actually thin.
- **The reorientation is continuous true polar wander confined to the trailing–leading meridian**
  (φ = ±90°): because the darkening apex rotates with the pole, the max-MoI axis stays ~90° ahead and
  the pole continuously chases it — the same continuous-rotation regime the published full-tensor
  model found (here with faster onset, since there is no fossil bulge to delay it).
- **⟨b⟩ thins ~0.8 km** (19.8 → 19.0 km) as the darkened equilibrium is approached.

---

## 3. Outputs (per case, in `plots/coupled_shell/<tag>/`)

- **`globe_animation.mp4` / `.gif`** — 3-D globe colored by Bond albedo (dark spot = irradiated
  trailing apex), with the **red spin-axis vector**, the **blue original-pole vector**, and a marker
  at the current trailing apex; viewed broadside to the wander plane so the reorientation is visible.
- **`pole_timeseries.png`** — pole colatitude (angle from the original pole), cumulative TPW rotation,
  darkening `f` and ⟨b⟩, and MoI misalignment vs. time.
- **`pole_trajectory.png`** — polar-stereographic spin-pole track.
- **`snapshots_albedo.png` / `_temperature.png` / `_thickness.png`** — Mollweide maps at a sequence
  of times showing the darkening spot, the surface-temperature response, and the shell-thickness
  evolution as the pole wanders.
- **`run_data.npz`** — full history + map snapshots (regenerate figures without re-running).

---

## 4. Verification

- **f=0 surrogate baseline** reproduces the eclipse analysis: global-mean T = 102.8 K (vs PTM3D
  102.2 K), sub-Jovian eclipse dip and ~+2 K polar warming present, and **thickness `b11s ≈ +0.003`
  km ≈ 0** (no leading/trailing signal without darkening — the albedo separation is clean).
- **MoI sanity:** eigenvalue ordering `I_max ≥ I_mid ≥ I_min` holds every step; a pure OSPW rotation
  leaves the inertia tensor unchanged.
- **Equilibrium bifurcation check:** the fully-darkened `b_eq` directly shows the max-inertia axis
  flipping ẑ → ŷ (90° misalignment) — the precondition for reorientation — independent of how far the
  shell has relaxed.
- **Wall time:** ~100 s per 30-Myr case at 91×181 (model ~30 s + 152-frame animation).

---

## 5. Limitations & next steps

- **Static surrogate (Phase 1).** ΔT_PTM is measured once at A=0.5 and assumed albedo-independent
  between (here, all) timesteps. The analytic `(1−A)^{1/4}` term carries the first-order albedo
  response; the residual eclipse/thermal-inertia coupling to albedo is second-order. **Phase 2**
  (live multi-rate PTM3D refresh: build `ptm_thermal_sim`, write a rotated lat/lon FITS albedo map,
  run ICO5, recompute ΔT_PTM when the pole drifts >10° or the albedo RMS-changes >0.03) is wired
  (`--live-ptm3d`, `_maybe_refresh_ptm3d`, `PTM3DRunner`) but the runner is not yet implemented.
- **Coarse dynamics grid (91×181)** slightly under-resolves the ~0.2 km eclipse `b11c`, which is
  dynamically negligible (the reorientation is degree-2 driven by the ~0.8 km darkening response).
- **Fossil bulge:** none by default. `--fossil hydrostatic` adds the body-fixed hydrostatic
  bulge × (1 − C_w) of OSPW Eqs. 3.8–3.10, which raises the instability threshold to
  $t_{20}+2t_{22} \approx 0.6$ km (Eq. 3.12).
- **OSPW constants** (`I₀`, `C_f`, the Eq. 2.18 factor) are computed once from the initial mean
  shell thickness; for large ⟨b⟩ excursions they could be refreshed.

## 6. Reproduce

```bash
PY=/Users/paha3326/anaconda3/envs/europa_viz/bin/python
$PY scripts/coupled_shell_simulation.py --tau-dark 1e4 --t-max 3e7   # case A
$PY scripts/coupled_shell_simulation.py --tau-dark 1e5 --t-max 3e7   # case B
$PY scripts/coupled_shell_simulation.py --tau-dark 1e7 --t-max 1e8 --parameter-set modern
$PY scripts/coupled_shell_simulation.py --smoke --no-ptm3d           # fast smoke test
```
