# Coupled Dynamic Simulation: Darkening + PTM3D + Full-Tensor Polar Wander

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
   trailing-hemisphere apex, in the spin frame). Endmembers **A_bright=0.5 → A_dark=0.15**.
2. **Temperature (PTM3D-coupled surrogate)** —
   `T(θ,λ) = [(1−A)·F_A(θ_t)/σ]^{1/4} + ΔT_PTM(θ_t,λ_t)`, where ΔT_PTM is the
   **PTM3D−O&S correction** (eclipse dip + thermal-inertia polar warming) measured from the
   existing `europa_ico6_uniform-ice` run, held fixed in the tidal frame and rotated with the pole.
3. **Equilibrium thickness** — O&S `b_eq(θ,λ)` (quadratic-in-b solver) from `T` and the tidal
   dissipation `D(θ,λ)` (re-rotated when the pole drifts).
4. **Relax at the local thermal-diffusion timescale** —
   `b ← b_eq + (b−b_eq)·e^{−dt/τ_diff}`, `τ_diff(θ,λ) = b²/κ`, `κ = k/(ρ c_p)`,
   `k = 488/T + 0.468`. Typically τ_diff ≈ 2–5 Myr (b≈20 km).
5. **Reorient (OSPW)** — full 3×3 inertia tensor → max-MoI eigenvector → overdamped viscous
   torque drives the spin pole toward it (no free TPW timescale).
6. The darkening apex and tidal heating **rotate with the new pole**; repeat.

**Multi-rate stepping** keeps the 100-Myr-class run tractable: a fine step (τ_dark/20) during the
short darkening ramp, a coarse step afterward, and an **adaptive cap** that limits the pole rotation
to ≤3°/step during active reorientation (the OSPW rotation is fast). A full ICO5 PTM3D re-run to
refresh ΔT_PTM is wired (`use_live_ptm3d`) but **not active in this Phase-1 deliverable** — ΔT_PTM is
static (see §5).

**Initial condition:** uniform shell thickness (= area-mean f=0 equilibrium, ≈19.8 km), **no fossil
bulge**, uniform albedo 0.5.

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
- **No fossil bulge** (user-specified) makes the pole unstable early and the TPW continuous/rapid;
  adding a fossil bulge (`use_fossil_bulge=True`, `fossil_bulge_factor`) would delay onset and can
  give episodic rather than continuous wander.
- **OSPW constants** (`I0`, `Cf`, ε-correction) are computed once from a fixed 20-km shell; for large
  ⟨b⟩ excursions they could be refreshed.

## 6. Reproduce

```bash
PY=/Users/paha3326/anaconda3/envs/europa_viz/bin/python
$PY scripts/coupled_shell_simulation.py --tau-dark 1e4 --t-max 3e7   # case A
$PY scripts/coupled_shell_simulation.py --tau-dark 1e5 --t-max 3e7   # case B
$PY scripts/coupled_shell_simulation.py --smoke --no-ptm3d           # fast smoke test
```
