# Illumination Model

## Overview

The PTM3D illumination model computes the radiative energy flux incident on each triangular face of a surface mesh at each simulation time step. Three flux components are evaluated: (1) direct solar irradiance, (2) thermal infrared emission from the primary body, and (3) reflected sunlight from the primary body. Ephemeris data are obtained from the NAIF SPICE toolkit, with an optional Keplerian propagator fallback for bodies lacking high-fidelity SPK kernels.

## Coordinate System and Ephemeris

All celestial body positions are computed in the IAU body-fixed reference frame of the target. The Sun and primary body positions at ephemeris time $t$ are obtained via the SPICE routine `spkpos_c` with light-time and stellar aberration correction (`LT+S`):

```
r_sun(t)     = spkpos_c("SUN",     t, "IAU_<BODY>", "LT+S", "<BODY>")
r_primary(t) = spkpos_c("PRIMARY", t, "IAU_<BODY>", "LT+S", "<BODY>")
```

where both vectors are expressed in kilometers in the target body-fixed frame. For bodies with a user-specified spin state (e.g., DAMIT shape models or Horizons-derived ephemerides for asteroids), positions are rotated from the ecliptic J2000 inertial frame to the body-fixed frame using the pole orientation and prime meridian angle.

## Direct Solar Flux

### Solar Irradiance

The solar irradiance at heliocentric distance $d_\odot$ is

$$E_\odot = \frac{S_0}{(d_\odot / 1\thickspace\text{AU})^2}$$

where $S_0 = 1361\thickspace\text{W m}^{-2}$ is the total solar irradiance at 1 AU.

### Point-Source Approximation

For faces where the Sun is well above the local horizon and unobstructed, a single center ray suffices and the incident direct solar flux reduces to

$$F_{\mathrm{direct},i} = E_\odot \cos\theta_i$$

where $\theta_i$ is the solar incidence angle, i.e., $\cos\theta_i = \hat{\mathbf{n}}_i \cdot \hat{\mathbf{s}}$, with $`\hat{\mathbf{n}}_i`$ the outward face normal and $\hat{\mathbf{s}}$ the unit vector from the face center toward the Sun. This fast path is taken when the solar elevation satisfies $`\sin(\mathrm{elev}) > 2\sin\alpha_\odot`$ and $\cos\theta_i > 0.10$, and the center ray is unobstructed by both terrain and the primary body.

### Finite Solar Disk Model

When the fast-path conditions are not met (for instance near topographic shadow boundaries, during partial eclipses, or at high solar incidence), the Sun is modeled as a finite-extent disk. The apparent angular radius of the solar disk as seen from face $i$ is

$$\alpha_\odot = \arcsin\negthinspace\left(\frac{R_\odot}{d_\odot}\right)$$

where $R_\odot = 695{,}700\thickspace\text{km}$.

#### Disk Sampling

The solar disk is sampled by $N$ points distributed uniformly over the disk area using a low-discrepancy (quasi-random) sequence, which provides more uniform coverage than pseudo-random sampling for the same number of points (e.g., Niederreiter 1992). The sample positions are randomized independently for each face to prevent coherent artifacts across the mesh.

An orthonormal basis $(\hat{\mathbf{u}}, \hat{\mathbf{v}})$ is constructed perpendicular to the Sun direction $`\hat{\mathbf{s}}`$. Each disk sample $k$ with coordinates $(d_{x,k}, d_{y,k})$ satisfying $d_{x,k}^2 + d_{y,k}^2 \leq 1$ generates a ray direction

$$\hat{\mathbf{d}}_k = \frac{\hat{\mathbf{s}} + \sin(\alpha_\odot)\left(d_{x,k}\thinspace\hat{\mathbf{u}} + d_{y,k}\thinspace\hat{\mathbf{v}}\right)}{\left\lVert\hat{\mathbf{s}} + \sin(\alpha_\odot)\left(d_{x,k}\thinspace\hat{\mathbf{u}} + d_{y,k}\thinspace\hat{\mathbf{v}}\right)\right\rVert}$$

which samples directions uniformly across the solid angle subtended by the solar disk. The Sun is treated as a uniform-brightness disk (no limb darkening is applied).

#### Horizon Gate

Before ray tracing, faces for which the entire solar disk lies below the tangent plane are culled:

$$\hat{\mathbf{n}}_i \cdot \hat{\mathbf{s}} \leq -\sin(\alpha_\odot) \implies F_{\mathrm{direct},i} = 0$$

This accounts for the finite angular extent of the disk: a face may receive partial illumination even when the disk center is slightly below the local horizon.

#### Occlusion Testing

Each sample direction $\hat{\mathbf{d}}_k$ undergoes two sequential occlusion tests:

1. **Eclipse test.** For satellites of a larger body (e.g., Europa orbiting Jupiter), a ray--sphere intersection test determines whether the ray from the face toward the sampled point on the solar disk is intercepted by the primary body, modeled as a sphere of radius $R_\mathrm{p}$. Specifically, rays intersecting the primary sphere at a parameter $0 < t < d_\odot$ are classified as eclipsed and excluded from the flux integral. This naturally produces partial eclipse geometries when only a fraction of the disk samples are blocked.

2. **Terrain occlusion test.** Non-eclipsed rays are tested against the triangulated surface mesh using the Intel Embree 4 ray tracing library. SIMD-optimized coherent ray packets are dispatched (16-wide AVX-512 or 8-wide AVX2, auto-detected at runtime). Additionally, a geometric horizon test against the body sphere filters rays that would pass below the planetary limb.

#### Flux Integration

The direct solar flux on face $i$ is computed as a cosine-weighted average over all $N$ valid disk samples:

$$
F_{\mathrm{direct},i} = E_\odot \cdot \frac{1}{N_\mathrm{total}} \sum_{k=1}^{N_\mathrm{total}} \begin{cases}
\max\negthinspace\left(0,\thickspace \hat{\mathbf{n}}_i \cdot \hat{\mathbf{d}}_k\right) & \text{if sample } k \text{ is unoccluded} \\
0 & \text{otherwise}
\end{cases}
$$

where $N_\mathrm{total}$ is the number of samples that survived the unit-disk rejection test (i.e., $d_x^2 + d_y^2 \leq 1$). This formulation naturally reproduces full illumination, penumbra, and umbra as the fraction of unblocked samples varies continuously.

#### Adaptive Sampling

A two-tier adaptive strategy balances computational cost and penumbral accuracy:

- Initial pass: a minimum of $N_0 = \max(8, N_\mathrm{base})$ samples are evaluated, where $N_\mathrm{base}$ is the user-specified sample count (default 9).
- Penumbra upsampling: if the initial pass detects partial terrain occlusion (i.e., at least one sample is blocked and at least one is visible), the face is resampled with $N_\mathrm{heavy} = \max(64, 4 N_\mathrm{base})$ samples to resolve the penumbra boundary more accurately.

### Sub-grid crater shadow flux (optional, off by default)

`thermal.enable_shadow_flux` adds a parameterized estimate, after Ingersoll et al. (1992), of the
scattered and re-emitted flux reaching shadowed floors of craters smaller than a mesh face. It is
off in every shipped configuration, and it should stay off on meshes that resolve the terrain,
where the radiation-exchange module computes this flux directly. Its implementation has known
errors: it omits the emissivity factor that Ingersoll's Eq. 8 carries (the absorbed shadow flux
should be $E_\odot \sin e_0\thinspace(1-A) f/(1-Af)\thinspace[\varepsilon + A(1-f)]$), and it adds the
result to the direct-solar channel.

## Primary Body Thermal Infrared Flux

For satellites orbiting a giant planet or other body with significant thermal emission, the absorbed thermal infrared flux from the primary is computed. The solid angle subtended by the primary at distance $d_\mathrm{p}$ is

$$\Omega_\mathrm{p} = 2\pi\left(1 - \cos\alpha_\mathrm{p}\right), \qquad \alpha_\mathrm{p} = \arcsin\negthinspace\left(\frac{R_\mathrm{p}}{d_\mathrm{p}}\right)$$

The irradiance view factor from a planar face to a sphere of angular radius $\alpha_\mathrm{p}$, for a face that sees the whole disk, is

$$\mathcal{F}_i = \sin^2\alpha_\mathrm{p}\thinspace\left(\hat{\mathbf{n}}_i \cdot \hat{\mathbf{p}}_i\right) \approx \frac{\Omega_\mathrm{p}}{\pi}\left(\hat{\mathbf{n}}_i \cdot \hat{\mathbf{p}}_i\right),$$

where $\hat{\mathbf{p}}_i$ is the unit vector from face $i$ toward the primary center. Faces oriented away from the primary ($\hat{\mathbf{n}}_i \cdot \hat{\mathbf{p}}_i \leq 0$) receive zero flux. The incident thermal infrared flux is

$$F_{\mathrm{IR},i} = M_\mathrm{p}\thinspace\mathcal{F}_i,$$

where $M_\mathrm{p} = \sigma T_\mathrm{p}^4$ is the flux emitted at the primary's surface (13.8 W m⁻² for Jupiter, $T_\mathrm{p} = 124.9$ K). Terrain visibility is checked by ray tracing, and the face absorbs $\varepsilon F_{\mathrm{IR},i}$.

## Primary Body Reflected Sunlight

Sunlight reflected by the primary toward the surface is computed as the irradiance from a
Lambertian sphere of Bond albedo $A_\mathrm{p}$ (e.g., 0.343 for Jupiter) illuminated by the Sun,
integrated over the sunlit part of the disk visible from the face, as a function of the phase
angle at the primary and the face orientation. The derivation and the validation against direct
numerical integration are in [primary-reflected-flux-derivation.md](primary-reflected-flux-derivation.md).
Terrain visibility is checked by ray tracing.

## Total Absorbed Flux

The total radiative flux absorbed by face $i$ at time $t$ is the sum of the three components:

$$F_{\mathrm{abs},i}(t) = (1 - A)\thinspace F_{\mathrm{direct},i}(t) + \varepsilon\thinspace F_{\mathrm{IR},i}(t) + (1 - A)\thinspace F_{\mathrm{refl},i}(t)$$

where $A$ is the surface Bond albedo, which depends on the incidence angle (Hayne et al. 2017 by default, or the Hapke directional-hemispheric albedo), and $\varepsilon$ is the infrared emissivity. In radiation-coupled runs the indirect flux of [radiation-exchange.md](radiation-exchange.md) is added. For heliocentric bodies without a parent planet (e.g., asteroids), the primary IR and reflected terms vanish and only the direct solar term contributes.

## Ray Origin Offset

To avoid self-intersection, each shadow ray starts offset from the face center along the face
normal and along the ray direction, by multiples of

$$\epsilon = \max\negthinspace\left(16\thinspace\mathrm{ulp}(r),\ \max\negthinspace\left(10^{-6}\thinspace r,\ 0.01\ \text{m}\right)\right),$$

where $r$ is the distance of the face center from the scene origin (the scene is recentered on
the mesh before tracing). The origin is $\mathbf{r}_i + 2\epsilon\thinspace\hat{\mathbf{n}}_i + k\epsilon\thinspace\hat{\mathbf{s}}$ with
$k = 1$ on the single-face path and $k = 8$ on the batched path used in production.

The 1 cm minimum is much larger than the relief of millimeter-scale terrain: on such meshes a
shadow ray can step over a small rim and light a face that should be in shadow. The default also
flips face normals radially outward, which turns overhangs into false sunlit faces. The opt-in
settings `PTM_ILLUM_TRUE_NORMALS`, `PTM_ILLUM_RISE_EPS` with `PTM_ILLUM_RISE_HABS=0`, and
`PTM_ILLUM_SEAL_HOLES` correct both; use them together on fine polar meshes
([ENVIRONMENT.md](ENVIRONMENT.md)).

## References

- Acton, C. H. (1996). Ancillary data services of NASA's Navigation and Ancillary Information Facility. *Planetary and Space Science*, 44(1), 65--70.
- Ingersoll, A. P., Svitek, T., & Murray, B. C. (1992). Stability of polar frosts in spherical bowl-shaped craters on the Moon, Mercury, and Mars. *Icarus*, 100(1), 40--47.
- Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.
