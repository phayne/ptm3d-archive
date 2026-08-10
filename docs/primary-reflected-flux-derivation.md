# Primary-body reflected flux ("Jupiter-shine", "Earthshine") — derivation

Derivation supporting the primary-reflected-flux implementation in
`src/cpp/illumination/illumination_model.cpp`.

This note derives the irradiance a PTM3D surface face receives from sunlight
reflected off its primary body, states the assumptions and the limits, and shows
which factors an earlier form of the expression dropped.

---

## 1. Geometry and symbols

| symbol | meaning |
| --- | --- |
| `R` | primary's radius (`primary_radius_`, km) |
| `d` | face-to-primary-center distance (km) |
| `α` | primary's angular radius seen from the face, `α = asin(R/d)` |
| `θ` | angle between the face normal and the direction to the primary |
| `a` | **phase angle** = Sun–primary–face angle (0 = primary "full", π = "new") |
| `F` | solar irradiance at the primary, `S₀/(r_pri/AU)²` |
| `A_B` | primary's **Bond** albedo (`*_BOND_ALBEDO`) |
| `p` | primary's **geometric** albedo |
| `Φ(a)` | phase function, normalized `Φ(0) = 1` |
| `q` | phase integral, `q = 2∫₀^π Φ(a) sin a da`; `A_B = p·q` |

The far-field condition `d ≫ R` holds by 4 orders of magnitude for every primary
PTM3D models (Jupiter/Europa is the tightest: `α = 5.98°`).

## 2. The result

For a sphere whose surface is Lambertian with hemispherical reflectance `A_B`
(a *Lambert sphere*), the irradiance on a face at distance `d` is

```
E = p · F · sin²α · Φ_L(a) · cos θ                                        (1)

    p     = (2/3) A_B                                                     (2)
    Φ_L(a) = [ sin a + (π − a) cos a ] / π                                (3)
```

`sin²α = (R/d)²`, so (1) is the familiar `E = p F (R/d)² Φ(a)` photometric law
projected onto the face by `cos θ`.

### 2.1 Why `p = (2/3) A_B`

`A_B = p·q`, and for a Lambert sphere `q = 3/2` (verified numerically below), so
`p = A_B/(3/2) = (2/3) A_B`. For Jupiter's `A_B = 0.343` that is `p = 0.2287`.

### 2.2 Why the `sin²α` and not `Ω/4π`

`sin²α·cos θ` **is** the view factor from a differential area to a sphere — the
the same factor used on the primary-IR path (`Ω/π = 2(1−cos α)`, which
equals `sin²α` to 0.3 % at `α = 6°`). `Ω/(4π)` is the fraction of the *full
sphere of directions* the primary occupies, which is not an irradiance view
factor at all.

### 2.3 Derivation of (1)

Take the primary's surface element with incidence angle `i` and emission angle
`e`. Lambert: emergent radiance `L = A_B F cos i / π` for `cos i > 0`. The flux
reaching a distant observer is the integral of `L cos e` over the part of the
sphere that is both lit and visible (a lune), divided by `d²`:

```
E_⊥(a) = (1/d²) ∫_{cos i>0, cos e>0} (A_B F cos i / π) cos e dA            (4)
```

With `dA = R² dω` on the unit sphere this gives `E_⊥ = A_B F (R/d)² · (2/3)Φ_L(a)`,
i.e. (1) with (2) and (3). Section 5 evaluates (4) numerically and recovers (1)
to 1 part in 10⁷.

Two independent checks of the normalization:

* **Zero phase.** A Lambert *disk* of radius `R` facing the beam has radiance
  `F/π` (for `A=1`) and delivers `(F/π)(πR²)/d² = F(R/d)²` — that is the
  *definition* of `p`, so `E_⊥(0) = p F (R/d)²`. ✔
* **Energy.** Integrating (1) over a sphere of radius `d`:
  `∮E_⊥ d²dΩ = F p R² π q = F πR² (p q) = A_B F πR²`, exactly the power the
  primary intercepts and reflects. ✔

### 2.4 Limits

| phase | `Φ_L` | meaning |
| --- | --- | --- |
| `a = 0` (full) | 1 | maximum; face sees the fully lit hemisphere |
| `a = π/2` (quadrature) | `1/π = 0.3183` | half the disk lit, limb-darkened |
| `a = π` (new) | 0 | face sees only the primary's night side |

`Φ_L` is monotone decreasing on `[0, π]`, `Φ_L(π−ε) → ε³/(3π)` (cubic
approach to zero, not quadratic).

**For Europa this is not an academic ordering.** Jupiter's phase angle is locked
to local time on the sub-Jovian hemisphere: `a = 0` (full Jupiter) occurs when
Europa is between the Sun and Jupiter, which is **local midnight** on the
sub-Jovian face; `a = π` occurs at local noon (and is the eclipse configuration).
The maximum of `Φ_L` therefore lands exactly where the surface is coldest and
`dT/dF = T/(4εσT⁴)` is largest.

## 3. What the pre-fix code computed

`calculatePrimaryReflectedLight` (pre-fix, `illumination_model.cpp:4571`):

```cpp
reflected = solar_flux_at_primary * primary_albedo * phase_function
          * (solid_angle / (4.0 * M_PI)) * cos_angle;      // Φ_code=(1+cos a)/2
```

Since `Ω/(4π) = (1−cos α)/2 ≈ sin²α/4`, that is

```
E_code = F · (A_B/4) · sin²α · Φ_code(a) · cos θ,      Φ_code = (1+cos a)/2   (5)
```

Comparing (5) with (1)+(2):

| factor | truth | code | ratio truth/code |
| --- | --- | --- | --- |
| albedo | `p = (2/3)A_B` | `A_B` | 2/3 |
| view factor | `sin²α` | `sin²α/4` | 4 |
| phase | `Φ_L(a)` | `(1+cos a)/2` | `Φ_L/Φ_code` |

so

```
E_true / E_code = (8/3) · Φ_L(a) / Φ_code(a)                               (6)
```

**8/3 = 2.667 at full phase** — but (6) is
*not* a constant: it falls through 1.698 at quadrature, crosses 1 at
`a = 128.7°`, and → 0 as `a → π`. Near new phase the old code *over*-estimates.

### 3.1 The conservation-law statement that settles the cancellation argument

An earlier revision declined to apply a blanket ×4 here, on the argument that
the spurious `1/4` in `Ω/(4π)` "is exactly the 1/4 that belongs in the radiosity"
`J = A_B F/4` of a sphere re-radiating isotropically. **That argument is half
right, and the half that is wrong is decisive.**

A `1/4` does belong to an *isotropic* re-radiator — an isotropic sphere has
`Φ ≡ 1` and phase integral `q_iso = 4`, hence `p_iso = A_B/4`. But the code then
multiplies that isotropic normalization by a *non-isotropic* phase function
`Φ_code = (1+cos a)/2`, whose own phase integral is

```
q_code = 2∫₀^π Φ_code sin a da = 2                                          (7)
```

The implied total reflected power is therefore `F πR² · p_iso · q_code
= F πR² · A_B · (2/4) = ½ A_B F πR²`:

> **the pre-fix expression radiates exactly half the sunlight the primary
> reflects.** It is not merely mis-shaped in phase; it violates energy
> conservation by a factor of 2.

So the earlier decision was masking part of the error. A blanket ×4 with
`Φ_code` retained would have given `4 × ½ = 2×` the
correct *total* power and 1.50× the truth at full phase (the note's number is
right). Declining the ×4 avoided that overestimate but preserved a *larger*
underestimate — 2.667× low at full phase, 2× low in total energy. The two
defects had to be repaired together, which is what this fix does. Applying (1)
restores `q = 3/2` and closes the energy budget exactly (§2.3).

## 4. Magnitude sanity — Jupiter-shine at Europa

With the code's own constants (`SOLAR_CONSTANT 1361`, `JUPITER_RADIUS 69911 km`,
`JUPITER_BOND_ALBEDO 0.343`, `JUPITER_IR_FLUX 13.8`), Europa at
`d = 671,100 km`, Jupiter at 5.2038 AU:

```
F              = 1361/5.2038²      = 50.26  W/m²
α              = asin(69911/671100)= 5.978°     sin²α = 1.0853e-2
p (Lambert)    = (2/3)(0.343)      = 0.2287
E_refl(full)   = 50.26·0.2287·sin²α = 0.1247 W/m²      <-- fixed code
E_refl(pre-fix)= 50.26·0.0858·sin²α = 0.0468 W/m²      <-- old code
E_IR           = 13.8·sin²α         = 0.1498 W/m²      (primary IR, unchanged)
```

Cross-check by total power, independent of any view factor:
Jupiter emits `4πR²·13.8 = 8.48e17 W`; spread over `4πd²` at Europa gives
`0.1498 W/m²`. ✔

**Order-of-magnitude check.** A literature expectation of "order 1 W/m²" is
sometimes quoted. The numbers above do not reach it, and geometry forbids it:
the hard upper bound on reflected flux at Europa, obtained by setting
`p = 1` (a perfectly white Jupiter), is

```
E_refl^max = F · 1 · sin²α = 0.545 W/m²
```

and adding the thermal term gives a total Jupiter-shine ceiling of
`0.70 W/m² < 1 W/m²` for *any* physically possible Jupiter. The correct order is
**a few tenths of a W/m²**: `0.15 (IR) + 0.12 (reflected) ≈ 0.28 W/m²` at the
sub-Jovian point at full phase, i.e. **0.55 % of the 50 W/m² solar constant at
Europa**. I could not reproduce a 1 W/m² figure from any combination of Jupiter's
published radius, albedo and effective temperature, and I am reporting that as a
disagreement with the stated expectation rather than tuning to it.

Using Jupiter's *measured* visual geometric albedo `p_V ≈ 0.52` instead of the
Lambert value 0.229 would give 0.297 W/m². That pairing is internally
inconsistent here (`q = A_B/p = 0.343/0.52 = 0.66`, and Jupiter's phase integral
is not 0.66 — the modern Cassini-based `A_B = 0.503 ± 0.012` of Li et al. 2018
implies `q ≈ 0.97`), and `p_V` is a V-band quantity being applied to a
bolometric flux. **The fix therefore uses the Lambert sphere**, which is
energy-consistent with the `*_BOND_ALBEDO` constants already in `constants.cpp`
and cannot double-count. A body-specific measured `(p, Φ)` pair is a separate
ticket; a note in the code says so.

## 5. Numerical verification (`/tmp/jup_derive.py`, reproduced in the gate)

Direct evaluation of the lune integral (4) on a 4000×8000 surface grid vs the
closed form (1)–(3):

```
phase[deg]  Phi_L(closed)  numeric/(2/3)   rel.err
   0.0        1.000000       1.000000     +7.7e-08
  30.0        0.880843       0.880843     +9.0e-08
  45.0        0.755409       0.755409     +9.5e-08
  90.0        0.318310       0.318310     +1.0e-07
 135.0        0.048302       0.048302     +3.6e-07
 170.0        0.000562       0.000562     +7.8e-06
 180.0        0.000000       0.000000        exact
phase integral q (Lambert)          = 1.5000   (expect 3/2)
phase integral q ((1+cos a)/2)      = 2.0000   (the pre-fix code, eq. 7)
phase integral q (isotropic)        = 4.0000
```

## 6. Assumptions and known limitations (carried into the code comment)

1. **Lambert sphere.** Real primaries are not Lambertian; Jupiter is
   backscattering. Bounded by §4: at full phase a measured-`p` Jupiter would be
   up to ~2.3× brighter than the Lambert value used here. This is the largest
   remaining physical uncertainty in the term, and it is *smaller* than the 2.667×
   error being removed.
2. **Uniform primary.** No zonal albedo structure, no Io/Ganymede shadows.
3. **Far field.** `sin²α` is the exact `dA→sphere` view factor; the small-`α`
   step from `Ω/π` to `sin²α` is a 0.3 % effect at Europa, 0.008 % at the Moon.
4. **Gray.** Reflected light is absorbed by the target with `(1 − A_vis)` (see
   `ptm_thermal_sim.cpp:1720`), which is the right spectral bin; no color term.
5. **Occlusion weighting.** When the primary's disk is partly hidden (§7),
   the visible *fraction* is computed treating the disk as uniformly bright.
   At intermediate phase the lit part of the disk is off-center, so a partial
   occultation is weighted slightly wrong. The error is bounded by the disk's
   brightness contrast and applies only to the thin band of faces that are
   partly occulted.

## 7. Fractional visibility of a finite disk

The same code paths treated the primary as a *point*: one ray to the disk center
decided all-or-nothing, and a face whose normal fell 0.1° below the primary's
direction got exactly zero. Both are quantization of a finite source:

* Jupiter from Europa: angular radius **5.98°**;
* Earth from the Moon: **0.95°**.

The generalization of `E = J·(Ω/π)·cos θ` to a partly hidden disk is

```
E = J · (Ω/π) · ⟨cos θ⟩,     ⟨cos θ⟩ ≡ (1/N) Σ_k  vis_k · max(cos θ_k, 0)   (8)
```

where the `N` samples tile the disk and `vis_k` is terrain visibility. `⟨cos θ⟩`
reduces to `cos θ₀` when the disk is small and fully visible, so (8) contains the
old expression exactly.

### 7.1 The horizon term is analytic — no rays needed

For a smooth global mesh (an icosphere: Europa, and every global Moon run) there
is **no terrain at all**, so the entire error is in the `cos θ ≤ 0` step.
That part has a closed form. Writing the disk-center altitude above the face's
horizon plane as `h = asin(cos θ₀)` and linearizing `cos θ ≈ h + x` across the
disk (`|x| ≤ α`):

```
⟨cos θ⟩ = α · g(h/α),        g(t) = [ tπ/2 + t²√(1−t²) + t·asin(t)
                                      + (2/3)(1−t²)^{3/2} ] / π           (9)
```

with `g(t≥1) = t`, `g(t≤−1) = 0`, and `g(0) = 2/(3π) = 0.2122` — i.e. a face
whose normal points exactly at the horizon still receives 21 % of `α·J·Ω/π`,
where the old code gave 0. `g` is C¹ and monotone on `[−1,1]`, so the step is
replaced by a ramp with no kink. Truncation is `O(α²)` (≲1 % of an already small
partial-band flux at Jupiter/Europa, ≲0.03 % at Earth/Moon).

On an icosphere the band `|h| < α` covers a fraction `≈ sin α` of the faces:
**10.4 % of a Europa mesh**, 1.7 % of a global Moon mesh.

### 7.2 Terrain is sampled, with a probe-then-escalate fast path

* **Probe (5 rays):** disk center + 4 rim points, tracing only directions above
  the face's horizon. All clear → `vis ≡ 1`, use (9); all blocked → `E = 0`.
  The probe basis is oriented to the **horizon** — `u = normalize(n̂ − (n̂·d̂)d̂)`,
  so probe `(+1,0)` is the disk's highest point (an arbitrary azimuth
  let all five probes miss a thin above-horizon crescent).
  A *smooth* convex body would always let an above-horizon ray escape, so the
  probe would be unanimously clear. **A faceted icosphere does not**: measured
  on Europa ico5, 1178 of 5.9 M face-steps still reach the sampler and 697 are
  ruled fully occluded by grazing hits on neighboring facets. Deciding those
  with the 512-ray sampler instead finds *more* of them (3857), so they are real
  ray hits, not a probe artifact — but they carry `⟨cos θ⟩ ≤ 2.3e-6`, i.e.
  ≤ 3.5e-7 W/m². Global meshes therefore pay ~5 rays per face instead of 1.
* **Escalate (N = 512 rays):** mixed probe → evaluate (8) directly. Sampling is
  Hammersley + concentric-disk (the same generator the solar-disk path uses).
  The naive `1/N` quantization estimate is **wrong**: a straight edge crosses
  low-discrepancy points in clumps, so the error falls much more slowly.
  Measured (`./build/bin/test_primary_disk` §4, edge swept in steps of 1/500 of
  the disk radius):

  | N | worst \|sampled − analytic\| | 1/N |
  | --- | --- | --- |
  | 16 | 0.0837 | 0.0625 |
  | 32 | 0.0484 | 0.0312 |
  | 64 | 0.0381 | 0.0156 |
  | 128 | 0.0175 | 0.0078 |
  | 256 | 0.0159 | 0.0039 |
  | **512** | **0.0082** | 0.0020 |

  A power-law fit gives `worst ≈ 0.49·N^(−0.65)` — roughly inverse-square-root,
  not `1/N`. (An independent fit on the same data gives `N^(−0.68)`.) Two other
  point sets at N = 512 are no better and are printed by the same binary:
  Fibonacci spiral 0.0092, equal-area polar grid 0.0174. N = 512 is the smallest
  power of two meeting the `< 1 %` criterion.
* The probe is an approximation: an occluder that hides part of the disk's
  interior while missing the center *and* all four rim points is not detected.
  `PTM_PRIMARY_DISK_ALWAYS_SAMPLE=1` disables the fast path for validation.

### 7.3 Fast path criterion (documented, as required)

Sampling is skipped entirely when **both**

1. `h ≥ α` (the whole disk is above the face's horizon plane), and
2. all 5 probe rays miss.

Then `⟨cos θ⟩ = cos θ₀` exactly and the arithmetic is bit-identical to the
pre-fix expression. Faces with `h ≤ −α` return 0 with no rays at all, also
bit-identical. **Everything in between is where the finite-disk treatment
matters.**

## 8. What changes, and by how much

At Europa's sub-Jovian point at full phase (local midnight), Jupiter-shine goes
`0.0468 + 0.1498 = 0.197` → `0.1247 + 0.1498 = 0.275 W/m²`, `ΔF_inc = +0.078 W/m²`
*incident*. What the surface actually gains is the **absorbed** part: reflected
light is solar-band and is absorbed with `(1 − A_vis)` (`ptm_thermal_sim.cpp:1720`),
so with Europa's `A_vis = 0.55`,

```
ΔF_abs = (1 − 0.55) × 0.078 = 0.0351 W/m²
ΔT     = T · ΔF_abs / (4 ε σ T⁴),   ε = 0.95
```

| background T | ΔT from the corrected reflected term (absorbed) |
| --- | --- |
| 130 K (Europa dusk) | +0.07 K |
| 90 K (Europa deep night) | +0.22 K |
| 70 K | +0.47 K |
| 50 K (cold trap) | +1.30 K |

(An earlier revision of this table applied the *incident* ΔF and so overstated
these by 1/(1−A_vis) = 2.2×.) The primary-IR term is thermal-band and absorbed
with `ε`, and is unaffected by this term.

The horizon ramp of §7 is signed: faces just below the old cutoff gain up to
`α·g·J·Ω/π`, and faces just above lose the difference between `cos θ₀` and (9).
