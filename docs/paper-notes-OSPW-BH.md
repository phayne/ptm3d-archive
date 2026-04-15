# Paper Notes: OSPW & B&H

## 1. OSPW — Ojakangas & Stevenson (1989)

**Citation**: Ojakangas, G.W. & Stevenson, D.J. (1989) "Polar Wander of an Ice Shell on Europa", *Icarus* 81, 242-270.

### Core Physics

A Europa ice shell decoupled from the silicate core by a liquid ocean has a thermal-equilibrium thickness that varies with latitude/longitude due to spatial variations in surface temperature and tidal dissipation (Paper I = O&S 1989, Icarus 81, 220-241). The l=2 spherical harmonic components of these thickness variations load the elastic lithosphere from below ("iceberg effect"), driving the inertia tensor toward an unstable configuration where I_yy > I_zz, triggering 90° TPW about the sub-Jovian (x) axis.

### Key Equations

#### Section 2: Bottom-Loading Gravity

The gravity field due to bottom-loaded topography t₂ₘ on an elastic lithosphere (Eq 2.18):

    C₂ₘ = Q₂ₘ · [(ρₘ−ρ_c)/ρ_c] · [(1−C_w) + 4C_w(b_c/a)(ρ_c/Δρ)]

where Q₂ₘ = (3/5)(ρ_c t₂ₘ)/(M̄ a) and the **Willemann compensation parameter** (Eq 2.11):

    C_w = [1 + (ρₘ[1−5(ρₘ/ρ̄)])/(μ̃₀)]⁻¹

with μ̃₀ = (2/3)(μd/ρₘga²) encoding lithosphere rigidity μ, thickness d, density ρₘ, gravity g, radius a.

The **iceberg effect** is the key: for bottom-loaded shells, even perfectly compensated topography contributes to inertia by a factor (ρ_c/Δρ)(b_c/a) larger than for top-loading. For Europa (ρ_c/Δρ ~ 11), this dominates over the elastic support term by ~10×.

#### Section 3: Inertia Tensor & Instability

Inertia tensor components (Eqs 2.12-2.14):

    I_xx = I' + Ma²(C₂₀/3 − 2C₂₂)
    I_yy = I' + Ma²(C₂₀/3 + 2C₂₂)
    I_zz = I' − (2/3)Ma²C₂₀

**Instability criterion** (Eq 3.4): TPW is triggered when I_yy > I_zz, i.e.:

    C₂₀ + 2C₂₂ > 0   ⟺   t₂₀ + 2t₂₂ > 0

This comes from shell thickness approaching thermal equilibrium, where tidal dissipation makes the shell thinner at the trailing hemisphere (lower viscosity) and thicker at the leading hemisphere.

**Fossil bulge** (Eq 3.8-3.10): The elastic lithosphere retains a "memory" of the hydrostatic shape:

    s₂₀)_fossil = −(1/3)hγ(1−C_w)
    s₂₂)_fossil = (1/4)hγ(1−C_w)

The fossil bulge stabilizes the current orientation. Its contribution to the gravity field acts against the driving t₂ₘ terms.

**Combined instability criterion** (Eq 3.12):

    (t₂₀ + 2t₂₂)/b_c > 0.6 · (4×10¹⁰ dyn cm⁻²)/μ

For μ ~ 4×10¹⁰ dyn/cm², this gives (t₂₀ + 2t₂₂)/b_c > ~0.6, or about ~12 km of shell thickness variation for b_c = 20 km. Paper I Table I values: t₂₀+2t₂₂ ranges from −0.25 to +0.97 km depending on rheology.

#### Section 4: Rigid Body Dynamics (Undamped)

The body coordinate x₁ = sub-Jovian, x₂ = trailing, x₃ = polar; Euler angles (ψ, θ, φ) rotate into the synchronous frame. The key equation describing polar wander (Eq 4.17):

    ψ̈ = (1/2)(t/τ) · ε · n² · sin 2ψ

This is an **inverted pendulum** where 2ψ is the angle. When t>0 (I_yy > I_zz), the equilibrium at ψ=0 becomes unstable and the system evolves toward ψ = ±π/2 (90° TPW).

**Undamped timescale** (Eq 4.22):

    T ~ (τ/(nε))^(1/3) · (−ln ψ₀)^(1/3)

For ε ~ 10⁻⁵, τ ~ 10⁷ yr: T ~ few × 10³ yr (extremely fast, but unrealistic without dissipation).

#### Section 5: Realistic Model with Shell Dissipation

**Shell viscous dissipation** (Eq 5.1-5.8): During reorientation, the shell must deform to conform to the changing hydrostatic figure of the ocean. This causes viscous strain in the warm basal layer.

Volumetric dissipation rate: q ~ ηγ²ψ̇², where η is the viscosity with Arrhenius dependence:

    η = η₀ exp[l(T_m/T − 1)]

with η₀ ~ 10¹³-10¹⁵ P, l ~ 15 (activation energy parameter).

**Unfractured shell** — constant-magnitude torque (Eq 5.8, 5.11):

    W ~ C₀γ² ;  F_v = −C₀(ω−n)/|ω−n|

where C₀ ~ 4πa²b_c(T_m/(T_m−T_s))ηₘₐₓγ²/l. This torque is enormous — it prevents TPW entirely for an unfractured shell (Eq 5.30).

**Fractured shell** — viscous torque proportional to velocity (Eq 5.15, 5.17-5.18):

    W_f ~ C_f ψ̇² ;  F_f = −2C_f(ω−n)

where:

    C_f ~ 4πa²b_c(T_m/(T_m−T_s))(T_f/T_m)² η₀ exp[l(T_m/T_f − 1)] γ²/l

T_f = temperature at base of fractured/regolith layer. When fractures accommodate the elastic strain above T_f, only the viscous ice below needs to deform.

**Overdamped pendulum** (Eq 5.39):

    ψ̈ = (1/2)(t/τ)εn² sin 2ψ − (C_f/(4εI₀))ψ̇

System is deeply overdamped (Eq 5.40): (t/τ)(10⁻⁷)/ε ≪ 1. The inertial term ψ̈ is negligible.

**Analytical solution** (Eq 5.41):

    tan ψ = (tan ψ₀) exp(t²/τ_f²)

where the **TPW timescale** is:

    τ_f = [C_f τ / (n² I₀)]^(1/2)

Time for ψ to reach 45° (Eq 5.42):

    t(ψ=45°) = [−ln(tan ψ₀)]^(1/2) · τ_f

Very weakly dependent on ψ₀ — initiation is easy from any finite perturbation.

**Critical result**: τ_f ~ 2×10⁶ yr for T_f ≳ 125 K (Fig 6). Below ~124 K, τ_f > few × 10⁶ yr and TPW cannot occur before thermal equilibrium eliminates the driving mechanism.

### Key Results

1. Shell + ocean reorient as a single entity, decoupled from the core
2. TPW is **impossible** for an unfractured shell (dissipation too large)
3. With fractures/regolith reaching T_f > 125 K: τ_f ~ few × 10⁶ yr
4. Without dissipation: T ~ few × 10³ yr (lower bound)
5. **Episodic**: ~2 Myr TPW events separated by ~10⁷ yr (thermal diffusion time τ_th)
6. Continuous drift possible if τ_f ~ τ_th
7. Maximum misalignment during TPW: θ ~ (t/τ)(n I₀/C_f) sin 2ψ — very small
8. The core is effectively decoupled: gravitational coupling torque from the core actually *pushes* the shell toward its new orientation when t/τ > 0

---

## 2. B&H — Burnett & Hayne (2023)

**Citation**: Burnett, E.R. & Hayne, P.O. (2023) "Spin–orbit coupling of Europa's ice shell and interior", *Icarus* 406, 115731.

### Core Physics

Europa is differentiated into an ice shell and solid interior, separated by a global ocean. The subsurface ocean enables independent rotation of shell and interior. B&H analyze this as a system of **two coupled nonlinear pendula** connected by gravity-gradient torque K_G, finding that the coupling strength qualitatively determines the tidal locking behavior.

### Key Equations

#### Section 2.3: Van Hoolst Coupled Model

**Equations of motion** (Eq 24 / unaveraged Eq 38):

    C_s θ̈_s + (3/2)(B_s−A_s)n² ρ⁻³ sin 2(θ_s−f) = −K_G sin 2(θ_s−θ_i)
    C_i θ̈_i + (3/2)(B_i−A_i)n² ρ⁻³ sin 2(θ_i−f) = +K_G sin 2(θ_s−θ_i)

where ρ = r/a (distance ratio), f = true anomaly.

**Gravity-gradient coupling** (Eq 25):

    K_G = (4πG/5)(ρ_s β_s + (ρ₀−ρ_s)β₀)(1−ρ₀/ρ_i)(B_i−A_i)

where β terms are equatorial flattenings and ρ_s, ρ₀, ρ_i are shell/ocean/interior densities.

**Orbit-averaged form** with libration angles η_j = θ_j − nt (Eq 27):

    C_s η̈_s + (3/2)(B_s−A_s)n² sin 2η_s + K_G sin 2(η_s−η_i) = 0
    C_i η̈_i + (3/2)(B_i−A_i)n² sin 2η_i − K_G sin 2(η_s−η_i) = 0

#### Energy Integral

**Normalized energy** (Eq 32):

    Ê = E/n² = (1/2)C_i η'²_i + (1/2)C_s η'²_s
        − (3/4)(B_i−A_i) cos 2η_i − (3/4)(B_s−A_s) cos 2η_s
        − (1/2)K̂_G cos 2(η_s−η_i)

where K̂_G = K_G/n² and η' = dη/d(nt).

**Critical energies** (Eqs 35-36):

    Ê_crit,1 = −(3/4)(B_i−A_i) + (3/4)(B_s−A_s) + (1/2)K̂_G
    Ê_crit,2 = +(3/4)(B_i−A_i) + (3/4)(B_s−A_s) − (1/2)K̂_G

**Critical coupling** (Eq 37):

    K̂*_G = (3/2)(B_i−A_i)

#### Zero-Velocity Curves

Setting η'_s = η'_i = 0, the **zero-velocity curve** at energy E₀ (Eq 17):

    η*(E₀) = {η : V(η) = E₀}

States inside are accessible (V < E₀ → T > 0), states outside are forbidden (V > E₀ → T < 0 unphysical). As dissipation removes energy, the accessible region shrinks.

#### Tidal Torques

**MacDonald** (Eq 39):

    L_tidal,j = −D̃_j (a/r)⁶ sign(ψ̇_j)

where D̃ = (3/2)(k₂/Q_s)(n⁴/G)R⁵_s (Eq 40).

**Williams-Efroimsky correction** (Eq 41/46):

    L_tidal,j = −D̃_j (a/r)⁶ (1/(nΓ)) ψ̇_j

where Γ is a settling parameter (0 < Γ < 1). Unlike MacDonald (constant-magnitude), this torque decreases as the body approaches synchronous rotation.

**Full dynamics with true anomaly** as independent variable (Eq 38):

    ψ''_s + (3/2)(κ_s)(r/a)/(1−e²) sin 2ψ_s = ... − K_G sin 2(ψ_s−ψ_i)/(C_s f̈²) + L_tidal,s/(C_s f̈²)

where κ_j = (B_j−A_j)/C_j and ψ_j = θ_j − f (libration angle).

#### Europa Parameters

- γ = C_i/C_s ≈ 28 (interior much more massive than shell)
- κ_i, κ_s ~ 0.0015-0.0035 (triaxiality)
- K̂_G range: 0.1(B_i−A_i) to 10(B_i−A_i) — poorly constrained
- e = 0.0094, a = 670,900 km

### Key Results

1. **Weak coupling** (K̂_G < K̂*_G = (3/2)(B_i−A_i)):
   - Core locks first (larger MoI → larger gravity torque)
   - Shell can revolve independently → erratic/chaotic motion
   - Shell eventually locks too as energy decays below E_crit,1
   - Two sequential locking events

2. **Strong coupling** (K̂_G > K̂*_G):
   - Shell and core co-rotate and lock simultaneously
   - Locking occurs when E < E_crit,2
   - Simpler, more predictable dynamics

3. **Chaotic dynamics**: Poincaré maps (Fig 9) show large chaotic regions when shell is free to revolve while core is locked. Regular librations exist near η_s = 0, π, surrounded by chaos.

4. **Modified tidal torque** (Williams-Efroimsky): Dramatically reduces NSR parameter space compared to classical MacDonald model. Many cases that would show NSR with MacDonald now lock tidally.

5. **Shell non-rigidity torque** (from OSPW): Additional energy dissipation shifts boundaries toward tidal locking, but could also impede independent shell-core oscillations.

6. **NSR still possible**: When both κ_i and κ_s are very small, NSR can persist as equilibrium even with Williams-Efroimsky torque.

---

## 3. Comparison: OSPW vs Our Model vs B&H

| Feature | Our Model | OSPW | B&H |
|---------|-----------|------|-----|
| **Geometry** | Full 3×3 MoI from b(θ,λ) grid | l=2 harmonics via Willemann-Turcotte | 1D longitudinal only (η_s, η_i) |
| **Iceberg effect** | Not included | Dominant for bottom-loaded shell | Not considered |
| **Elastic support** | Not included | C_w parameter (Eq 2.11) | Not considered |
| **Reorientation** | Continuous drift toward v_max (τ_TPW) | Overdamped pendulum (τ_f from viscosity) | Gravity torque + tidal locking |
| **Shell dissipation during TPW** | Not included | Critical — can prevent TPW entirely | Mentioned; deferred to future work |
| **Fossil bulge** | 2D field b_fossil = (α−1)×anomaly | Lithospheric memory via (1−C_w) | Not considered |
| **Interior coupling** | None | Core decoupled (justified in §5.3) | Full gravity-gradient coupling K_G |
| **Rotation axis** | Arbitrary 3D pole | About sub-Jovian axis (1D: ψ) | About spin axis (1D: η) |
| **Tidal heating** | Co-rotates with spin pole | Quasi-static (fixed in body frame) | Not considered |
| **Darkening** | Co-rotates with spin pole | Not considered | Not considered |
| **Timescale** | τ_TPW = 10⁵ yr (free parameter) | τ_f ~ 10⁶ yr (from shell rheology) | Orbital timescale (~3.5 days) |
| **Behavior** | Continuous rotation (period ~0.8 Myr) | Episodic 90° events (~2 Myr) or drift | Tidal locking or NSR |

### Why Our Model Shows Continuous Rotation

Our model predicts continuous spin-pole rotation because:
1. τ_TPW = 10⁵ yr is 10× shorter than OSPW's τ_f — our model is effectively **underdamped**
2. We lack shell viscous dissipation — nothing opposes pole motion except the fossil bulge
3. Co-rotating D and darkening fields eliminate any body-frame restoring force
4. Only the fossil bulge (which decays on τ_visc) provides temporary stabilization

OSPW shows that shell viscous dissipation is the dominant impediment and controls the timescale. Our free-parameter τ_TPW should be replaced by OSPW's physically-derived τ_f.

---

## 4. Plan for Model Improvements

### From OSPW

**Priority 1: Shell viscous dissipation during TPW**
- Replace free τ_TPW with τ_f from shell rheology (Eq 5.41)
- Compute C_f from temperature at base of fractured region (Eq 5.15)
- Overdamped pendulum physics gives correct timescale naturally

**Priority 2: Iceberg effect in inertia tensor**
- Apply Eq 2.18 correction when converting shell thickness to gravity coefficients
- Factor ~10× larger contribution than elastic support alone

**Priority 3: Lithospheric fossil bulge**
- Replace ad hoc b_fossil = (α−1)×anomaly with C_w-based formula (Eq 3.10)
- Physically grounded in lithosphere rigidity

### From B&H

**Priority 1: Gravity-gradient coupling K_G**
- Add interior state variables (η_i, η̇_i)
- K_G from interior/shell flattenings (Eq 25)
- Determines weak vs strong coupling regime

**Priority 2: Energy-based analysis**
- Track energy integral Ê (Eq 32) as diagnostic
- Zero-velocity curves bound accessible states
- Critical energies predict locking

**Priority 3: Modified tidal torque**
- Williams-Efroimsky (Eq 46) instead of MacDonald
- Significantly shifts NSR/locking boundary
