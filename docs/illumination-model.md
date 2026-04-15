# Illumination Model

## Overview

The PTM3D illumination model computes the radiative energy flux incident on each triangular face of a surface mesh at each simulation time step. Three flux components are evaluated: (1) direct solar irradiance, (2) thermal infrared emission from the primary body, and (3) reflected sunlight from the primary body. Ephemeris data are obtained from the NAIF SPICE toolkit, with an optional Keplerian propagator fallback for bodies lacking high-fidelity SPK kernels.

## Coordinate System and Ephemeris

All celestial body positions are computed in the IAU body-fixed reference frame of the target. The Sun and primary body positions at ephemeris time $t$ are obtained via the SPICE routine `spkpos_c` with light-time and stellar aberration correction (`LT+S`):

$$\mathbf{r}_\odot(t) = \texttt{spkpos}(\text{SUN},\; t,\; \text{IAU\_BODY},\; \text{LT+S},\; \text{BODY})$$

$$\mathbf{r}_\mathrm{p}(t) = \texttt{spkpos}(\text{PRIMARY},\; t,\; \text{IAU\_BODY},\; \text{LT+S},\; \text{BODY})$$

where both vectors are expressed in kilometers in the target body-fixed frame. For bodies with a user-specified spin state (e.g., DAMIT shape models or Horizons-derived ephemerides for asteroids), positions are rotated from the ecliptic J2000 inertial frame to the body-fixed frame using the pole orientation and prime meridian angle.

## Direct Solar Flux

### Solar Irradiance

The solar irradiance at heliocentric distance $d_\odot$ is

$$E_\odot = \frac{S_0}{(d_\odot / 1\;\text{AU})^2}$$

where $S_0 = 1361\;\text{W m}^{-2}$ is the total solar irradiance at 1 AU.

### Point-Source Approximation

For faces where the Sun is well above the local horizon and unobstructed, a single center ray suffices and the absorbed direct solar flux reduces to

$$F_{\mathrm{direct},i} = E_\odot \cos\theta_i$$

where $\theta_i$ is the solar incidence angle, i.e., $\cos\theta_i = \hat{\mathbf{n}}_i \cdot \hat{\mathbf{s}}$, with $\hat{\mathbf{n}}_i$ the outward face normal and $\hat{\mathbf{s}}$ the unit vector from the face center toward the Sun. This fast path is taken when the solar elevation satisfies $\sin(\mathrm{elev}) > 2\sin\alpha_\odot$ and $\cos\theta_i > 0.10$, and the center ray is unobstructed by both terrain and the primary body.

### Finite Solar Disk Model

When the fast-path conditions are not met --- for instance near topographic shadow boundaries, during partial eclipses, or at high solar incidence --- the Sun is modeled as a finite-extent disk. The apparent angular radius of the solar disk as seen from face $i$ is

$$\alpha_\odot = \arcsin\!\left(\frac{R_\odot}{d_\odot}\right)$$

where $R_\odot = 695{,}700\;\text{km}$.

#### Disk Sampling

The solar disk is sampled by $N$ points distributed uniformly over the disk area using a low-discrepancy (quasi-random) sequence, which provides more uniform coverage than pseudo-random sampling for the same number of points (e.g., Niederreiter 1992). The sample positions are randomized independently for each face to prevent coherent artifacts across the mesh.

An orthonormal basis $(\hat{\mathbf{u}}, \hat{\mathbf{v}})$ is constructed perpendicular to the Sun direction $\hat{\mathbf{s}}$. Each disk sample $k$ with coordinates $(d_{x,k}, d_{y,k})$ satisfying $d_{x,k}^2 + d_{y,k}^2 \leq 1$ generates a ray direction

$$\hat{\mathbf{d}}_k = \frac{\hat{\mathbf{s}} + \sin(\alpha_\odot)\left(d_{x,k}\,\hat{\mathbf{u}} + d_{y,k}\,\hat{\mathbf{v}}\right)}{\left\|\hat{\mathbf{s}} + \sin(\alpha_\odot)\left(d_{x,k}\,\hat{\mathbf{u}} + d_{y,k}\,\hat{\mathbf{v}}\right)\right\|}$$

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

$$F_{\mathrm{direct},i} = E_\odot \cdot \frac{1}{N_\mathrm{total}} \sum_{k=1}^{N_\mathrm{total}} \begin{cases} \max\!\left(0,\; \hat{\mathbf{n}}_i \cdot \hat{\mathbf{d}}_k\right) & \text{if sample } k \text{ is unoccluded} \\ 0 & \text{otherwise} \end{cases}$$

where $N_\mathrm{total}$ is the number of samples that survived the unit-disk rejection test (i.e., $d_x^2 + d_y^2 \leq 1$). This formulation naturally reproduces full illumination, penumbra, and umbra as the fraction of unblocked samples varies continuously.

#### Adaptive Sampling

A two-tier adaptive strategy balances computational cost and penumbral accuracy:

- **Initial pass**: A minimum of $N_0 = \max(8, N_\mathrm{base})$ samples are evaluated, where $N_\mathrm{base}$ is the user-specified sample count (default 9).
- **Penumbra upsampling**: If the initial pass detects partial terrain occlusion --- i.e., at least one sample is blocked and at least one is visible --- the face is automatically resampled with $N_\mathrm{heavy} = \max(64, 4 N_\mathrm{base})$ samples to resolve the penumbra boundary more accurately.

### Shadow Flux (Permanently Shadowed Regions)

For faces in permanent shadow where $F_{\mathrm{direct},i} = 0$, an optional shadow flux model based on Ingersoll et al. (1992) provides a residual heating estimate from multiply-scattered sunlight within craters. The crater geometry is parameterized by $D$, the diameter-to-depth ratio. For shallow craters ($D > 10$):

$$F_\mathrm{shadow} = E_\odot \sin(e_0) \cdot \frac{1 - A}{D^2/4} \left[1 + \frac{A(1-f)}{\varepsilon}\right]$$

where $e_0$ is the solar elevation angle, $A$ is the surface Bond albedo, $\varepsilon$ is the surface emissivity, and $f = 1/(1 + D^2/4)$. For deeper craters ($D \leq 10$), the full multiple-scattering expression is used:

$$F_\mathrm{shadow} = E_\odot \sin(e_0) \cdot \frac{(1 - A) f}{1 - Af} \cdot \left(1 + \frac{Af}{1 - Af}\right)$$

## Primary Body Thermal Infrared Flux

For satellites orbiting a giant planet or other body with significant thermal emission, the absorbed thermal infrared flux from the primary is computed. The solid angle subtended by the primary at distance $d_\mathrm{p}$ is

$$\Omega_\mathrm{p} = 2\pi\left(1 - \cos\alpha_\mathrm{p}\right), \qquad \alpha_\mathrm{p} = \arcsin\!\left(\frac{R_\mathrm{p}}{d_\mathrm{p}}\right)$$

The view factor from a planar face to a spherical primary, assuming Lambertian emission, is

$$\mathcal{F}_i = \frac{\Omega_\mathrm{p}}{4\pi}\left(\hat{\mathbf{n}}_i \cdot \hat{\mathbf{p}}_i\right)$$

where $\hat{\mathbf{p}}_i$ is the unit vector from face $i$ toward the primary center. Faces oriented away from the primary ($\hat{\mathbf{n}}_i \cdot \hat{\mathbf{p}}_i \leq 0$) receive zero flux. The thermal IR flux is then

$$F_{\mathrm{IR},i} = F_\mathrm{IR,p} \cdot \mathcal{F}_i$$

where $F_\mathrm{IR,p}$ is the total thermal infrared flux emitted by the primary (e.g., $F_\mathrm{IR,Jupiter} = 13.8\;\text{W m}^{-2}$ at Europa's orbit). Terrain visibility is verified via ray tracing before assigning the flux.

## Primary Body Reflected Sunlight

Sunlight reflected by the primary body toward the satellite surface is modeled assuming Lambertian scattering with a phase-dependent correction. The phase angle $\varphi$ between the Sun and the target body as seen from the primary is

$$\varphi = \arccos\!\left(-\hat{\mathbf{s}}_\mathrm{p} \cdot \hat{\mathbf{p}}_i\right)$$

where $\hat{\mathbf{s}}_\mathrm{p}$ is the unit vector from the primary to the Sun, and $\hat{\mathbf{p}}_i$ points from the face toward the primary. A Lambert-sphere phase function modulates the reflected intensity:

$$\Phi(\varphi) = \frac{1 + \cos\varphi}{2}$$

The reflected solar flux incident on face $i$ is

$$F_{\mathrm{refl},i} = \frac{E_{\odot,\mathrm{p}} \cdot A_\mathrm{p} \cdot \Phi(\varphi) \cdot \Omega_\mathrm{p}}{4\pi} \left(\hat{\mathbf{n}}_i \cdot \hat{\mathbf{p}}_i\right)$$

where $E_{\odot,\mathrm{p}} = S_0 / (d_{\odot,\mathrm{p}} / 1\;\text{AU})^2$ is the solar irradiance at the primary's heliocentric distance and $A_\mathrm{p}$ is the primary's Bond albedo (e.g., $A_\mathrm{Jupiter} = 0.343$). Terrain visibility is again verified by ray tracing.

## Total Absorbed Flux

The total radiative flux absorbed by face $i$ at time $t$ is the sum of the three components:

$$F_{\mathrm{abs},i}(t) = (1 - A)\,F_{\mathrm{direct},i}(t) + F_{\mathrm{IR},i}(t) + (1 - A)\,F_{\mathrm{refl},i}(t)$$

where $A$ is the surface Bond albedo. For heliocentric bodies without a parent planet (e.g., asteroids), the primary IR and reflected terms vanish and only the direct solar term contributes.

## Ray Origin Offset

To prevent self-intersection artifacts in ray tracing, each ray origin is offset from the face center along both the face normal and the ray direction. The offset magnitude is scale-aware:

$$\epsilon = \max\!\left(16\,\epsilon_\mathrm{mach}\,r,\; \max\!\left(10^{-6}\,r,\; 0.01\;\text{m}\right)\right)$$

where $r = \|\mathbf{r}_i\|$ is the radial distance of the face center from the body origin and $\epsilon_\mathrm{mach}$ is the machine epsilon for double-precision arithmetic. The ray origin for face $i$ with sun direction $\hat{\mathbf{s}}$ is then

$$\mathbf{o}_i = \mathbf{r}_i + 2\epsilon\,\hat{\mathbf{n}}_i + 12\epsilon\,\hat{\mathbf{s}}$$

This ensures robust behavior across scales ranging from meter-scale DEM patches to planetary-radius global meshes.

## References

- Acton, C. H. (1996). Ancillary data services of NASA's Navigation and Ancillary Information Facility. *Planetary and Space Science*, 44(1), 65--70.
- Ingersoll, A. P., Svitek, T., & Murray, B. C. (1992). Stability of polar frosts in spherical bowl-shaped craters on the Moon, Mercury, and Mars. *Icarus*, 100(1), 40--47.
- Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.
