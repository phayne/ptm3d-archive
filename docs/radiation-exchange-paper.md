# Inter-face thermal radiation exchange in PTM3D

*Draft section for the PTM3D framework paper (PSJ / JGR-Planets).*

## 1. Motivation

On airless planetary bodies, surfaces in topographically complex terrain
exchange thermal infrared radiation and reflect incident sunlight onto
their neighbors. In permanently shadowed regions (PSRs) such as polar
craters on the Moon, this inter-face radiative coupling can be the
dominant heat source [@vasavada1999; @paige2010; @hayne2015]: the floor
of a deep crater receives essentially no direct sunlight but is bathed
in infrared emission from the warmer crater walls illuminated above it,
and a small fraction of any direct or scattered solar flux on those
walls is reflected diffusely toward the floor.  Modelling cold-trap
temperatures and the long-term stability of volatiles requires
resolving these multi-bounce exchanges.

The PTM3D framework couples surface energy balance, a 1-D heat
conduction model for the regolith subsurface, ephemeris-driven
illumination, and an inter-face radiation exchange module that we
describe here.  The module computes pairwise view factors between
mutually visible mesh faces, solves the radiosity equation for the
self-consistent outgoing radiance field, and feeds the absorbed
indirect flux back into the thermal solver through a self-consistent
fixed-point iteration.

## 2. Surface energy balance with inter-face exchange

At the surface ($z = 0$) of each mesh face $i$ the thermal model
enforces

$$
Q_{s,i}(t) \;+\; k_0 \frac{T_1 - T_0}{\Delta z_0}
   \;=\; \varepsilon_i \sigma T_0^4,
\tag{1}
$$

where $T_0(t)$ is the surface temperature at face $i$ and time $t$
(the unknown), $T_1(t)$ is the temperature at the first subsurface
node a distance $\Delta z_0$ below the surface, $k_0$ is the thermal
conductivity at the surface, $\varepsilon_i$ the broadband IR
emissivity, and $\sigma=5.6705 \times 10^{-8}\,\mathrm{W\,m^{-2}\,K^{-4}}$
the Stefan–Boltzmann constant.  The absorbed surface flux $Q_{s,i}(t)$
combines four contributions,

$$
Q_{s,i}(t) \;=\;
   \bigl[Q_{\odot,i}(t) + Q_{\mathrm{refl},i}(t)\bigr]\,\bigl[1 - \alpha_i(\theta_i)\bigr]
   \;+\; \varepsilon_i\, Q_{\mathrm{IR},i}(t)
   \;+\; Q_{\mathrm{ind},i}(t),
\tag{2}
$$

namely direct solar flux $Q_{\odot,i}$, the corresponding albedo-weighted
absorbed fraction; primary-body reflected solar flux $Q_{\mathrm{refl},i}$
(e.g.\ earthshine in the lunar case); primary-body thermal IR
$Q_{\mathrm{IR},i}$; and the inter-face indirect flux
$Q_{\mathrm{ind},i}(t)$ computed by the radiation exchange module.
The first three contributions come from the ephemeris-driven
illumination model and are independent of the surface temperature
state; the fourth depends on $\{T_i(t)\}$ across all radiatively
coupled faces and motivates the iterative coupling scheme in §5.

The visible-wavelength albedo $\alpha_i(\theta_i)$ optionally carries
an incidence-angle dependence after Hayne et al.\ (2017),

$$
\alpha(\theta) \;=\; \alpha_0
   \;+\; a\,\Bigl(\tfrac{\theta}{\pi/4}\Bigr)^{3}
   \;+\; b\,\Bigl(\tfrac{\theta}{\pi/2}\Bigr)^{8},
\tag{3}
$$

with body-dependent coefficients $\alpha_0$, $a$, $b$.  The factor
$\varepsilon_i$ on the primary-body IR term encodes Kirchhoff's law,
$\alpha_{\mathrm{IR}} = \varepsilon$, so the IR multiplier in the
radiation-exchange module and in the primary-body IR are consistent
to leading order.

## 3. Radiosity equation

For an enclosure of $N$ diffuse, gray surfaces, the radiosity
$B_i(t)$, defined as the total outgoing radiant flux per unit area
(emission plus reflection), satisfies

$$
B_i \;=\; \varepsilon_i\, \sigma\, T_i^4
       \;+\; \rho_i \sum_{j=1}^{N} F_{ij}\, B_j
       \;+\; \rho_i\, Q_{\odot,i},
\tag{4}
$$

where $\rho_i = 1 - \varepsilon_i$ is the broadband reflectivity and
$F_{ij}$ is the view factor from face $i$ to face $j$.  The three terms
represent thermal self-emission, reflected incoming radiation from
other faces, and reflected direct solar flux.  In matrix form,

$$
(\mathbf{I} - \boldsymbol{\rho} \mathbf{F})\,\mathbf{B}
   \;=\; \mathbf{E} \;+\; \boldsymbol{\rho}\,\mathbf{Q}_{\odot},
\tag{5}
$$

with $\mathbf{E}_i = \varepsilon_i \sigma T_i^4$ and
$\boldsymbol{\rho}=\mathrm{diag}(\rho_i)$.  After solving for
$\mathbf{B}$, the absorbed indirect flux at each face is

$$
Q_{\mathrm{ind},i} \;=\; \varepsilon_i \sum_{j=1}^{N} F_{ij}\, B_j
   \;=\; \varepsilon_i\,(\mathbf{F}\mathbf{B})_i.
\tag{6}
$$

We solve (5) at each output timestep $t \in \{1,\dots,N_t\}$ in the
science window using current surface temperatures $\{T_i(t)\}$.  Solar
fluxes $\mathbf{Q}_{\odot}(t)$ vary on the diurnal timescale and are
sampled at the same cadence as the temperature trajectory.

## 4. View factor computation

The view factor between two mutually visible diffuse facets $i$ and
$j$ in vacuum is

$$
F_{ij}
   \;=\; \frac{1}{A_i} \int_{A_i} \int_{A_j}
   \frac{\cos\theta_i\,\cos\theta_j}{\pi\, r^{2}}\;
   V_{ij}(\mathbf{x},\mathbf{y})\;
   dA_j\, dA_i,
\tag{7}
$$

where $\theta_i,\theta_j$ are the angles between the inter-element ray
$\mathbf{x}\!\to\!\mathbf{y}$ and the local outward normals, $r$ is
the inter-element distance, and $V_{ij}\in\{0,1\}$ is a visibility
indicator (1 if the segment is not occluded by intervening terrain,
0 otherwise).  By construction the matrix $\mathbf{F}$ is row-stochastic
modulo the sky aperture: $\sum_j F_{ij} = 1 - F_{i,\mathrm{sky}}$,
where $F_{i,\mathrm{sky}}$ is the fraction of hemispherical solid
angle at $i$ that is unblocked by mesh faces.

### 4.1 Monte-Carlo estimator

For production runs on meshes with $N \sim 10^{6}$ faces we use a
cosine-weighted Monte-Carlo estimator: from each source face $i$ we
emit $M$ rays sampled with PDF $\cos\theta_i/\pi$ on the upper
hemisphere of the local frame, propagate them to the first intersection
with another face $j$ (via an Embree-built BVH), and accumulate
$F_{ij} \approx M_{ij} / M$, where $M_{ij}$ is the number of rays that
hit $j$.  The cosine weighting matches the integrand of (7) so each
ray contributes equally to the estimator.  Rays that escape without
striking any face contribute to $F_{i,\mathrm{sky}}$.  Statistical
error scales as $\mathcal{O}(M^{-1/2})$; we use $M = 10^{5}$ for
production lunar workloads, giving $\sim 10^{-3}$ relative error on
$F_{ij}$ at the dominant pairs.

After Monte-Carlo sampling we apply an *adaptive top-K compaction*
that retains, per face, the smallest neighbor set whose cumulative
view factor captures a target fraction $\eta$ (default $\eta = 0.95$)
of the non-sky energy.  Faces with very flat local horizons typically
need only $\mathcal{O}(20)$ neighbors; faces inside narrow gullies
saturate to a configurable cap (default 500).  The compaction yields
a sparse $\mathbf{F}$ with $\mathcal{O}(NK)$ entries where $K \sim
100$ on production lunar polar meshes.

### 4.2 Hierarchical alternative

For meshes where ray sampling is too expensive, the framework
optionally constructs a hierarchical matrix (H-matrix) representation
of $\mathbf{F}$ following Hackbusch (1999) and the topographic
application of Potter et al.\ (2023).  An octree-based cluster tree
partitions the face set; admissible block pairs (geometrically
well-separated) are approximated by low-rank factorizations
$\mathbf{F}_{IJ} \approx \mathbf{U}_{IJ}\mathbf{V}_{IJ}^{\top}$ with
rank chosen by either an adaptive cross approximation (ACA) error
tolerance or a truncated randomized SVD; inadmissible blocks at the
leaf level are stored densely or as sparse blocks.  Asymptotically
this yields $\mathcal{O}(N \log N)$ storage and matrix-vector
product complexity.  Practically, on the $1.8\,\mathrm{M}$-face polar
meshes used in this paper, the Monte-Carlo path with adaptive top-K
compaction is faster and simpler and is the default.

### 4.3 Reciprocity check

The view factor matrix satisfies a strict reciprocity relation,
$A_i F_{ij} = A_j F_{ji}$, which we use as an internal consistency
check.  The Monte-Carlo estimator is unbiased so the reciprocity
residual scales as $\mathcal{O}(M^{-1/2})$; we report the maximum
relative violation in the simulation log.

## 5. Coupled thermal–radiation iteration

Because $\{T_i(t)\}$ and $\{Q_{\mathrm{ind},i}(t)\}$ depend on each
other through equations (1), (4), and (6), the coupled system is
solved by a Picard iteration that wraps the thermal model and the
radiosity solver in an outer loop.

### 5.1 Outer fixed-point iteration

We denote by $\mathcal{T}\!\bigl[\,\cdot\,\bigr]$ the thermal-model
operator that, given a time series of total absorbed flux
$\{Q_{s,i}(t)\}$ at each face, returns the surface temperature
trajectory $\{T_i(t)\}$ over one full output cycle of the diurnal
window.  Internally $\mathcal{T}$ also takes the initial subsurface
temperature profile $\mathbf{T}_i(z, t=0)$ as input — a critical
detail revisited in §5.3.  We denote by $\mathcal{R}\!\bigl[\,\cdot\,\bigr]$
the radiosity operator (5)–(6) that maps a temperature trajectory
$\{T_i(t)\}$ to the indirect-flux trajectory
$\{Q_{\mathrm{ind},i}(t)\}$.

The fixed-point iteration is

$$
\begin{aligned}
T^{(0)}_i(t) &= \mathcal{T}\bigl[Q^{(0)}_{s,i}(t)\bigr],
   \qquad Q^{(0)}_{s,i}(t) = (1-\alpha_i)\bigl(Q_{\odot,i}+Q_{\mathrm{refl},i}\bigr) + \varepsilon_i Q_{\mathrm{IR},i}, \\[2pt]
Q^{(k)}_{\mathrm{ind},i}(t) &= \mathcal{R}\bigl[T^{(k-1)}_i(t)\bigr],
   \qquad k \geq 1, \\[2pt]
Q^{(k)}_{s,i}(t) &= Q^{(0)}_{s,i}(t) + Q^{(k)}_{\mathrm{ind},i}(t), \\[2pt]
T^{(k)}_i(t) &= \mathcal{T}\bigl[Q^{(k)}_{s,i}(t)\bigr].
\end{aligned}
\tag{8}
$$

Iteration 0 is the radiation-off solution; each subsequent iteration
refines $\{T_i\}$ using the indirect flux computed from the prior
iteration's temperatures.  The iteration is terminated by the
convergence criterion described in §5.4 or when a maximum count is
reached (default 10).

### 5.2 Two-cadence thermal solver

The thermal operator $\mathcal{T}$ uses an equilibration phase at
coarse cadence followed by an output phase at fine cadence, after
the decoupled equilibration scheme of [@hayne2017]. For lunar polar
workloads we use two equilibration orbits ($\sim 60$ Earth days) at
a body-independent cadence of 120 illumination samples per synodic
day for the equilibration phase, then a single output cycle
($\sim 30$ Earth days = one lunation) at hourly cadence.  The
equilibration phase relaxes the deep subsurface to a quasi-periodic
state consistent with the long-period seasonal forcing; the output
phase resolves the diurnal cycle and is the only portion saved to
disk.

### 5.3 Post-equilibration state hand-off

A subtlety not present in single-shot thermal simulations becomes
important inside the fixed-point loop (8).  Each call to $\mathcal{T}$
needs a starting subsurface profile $\mathbf{T}_i(z, t=0)$.  Two
options at iteration $k$ are:

  (i) restart the equilibration phase from a cold-start guess and
      let it relax against the new flux boundary condition; or

  (ii) inherit the iteration-$0$ post-equilibration state and run
       *only* the output cycle from that profile with the updated
       boundary condition.

Option (i) is conceptually simpler but is sensitive to the
equilibration-phase initial guess: on cold polar terrain where the
diurnal-mean equilibrium temperature is very low (and clamped by a
sanity floor in the cold-start formula), the equilibration phase
converges to a different fixed point than iteration 0 reached, and
the subsequent output cycle inherits the offset.  In our tests on
the lunar polar Mons Mouton site this produced a systematic surface
temperature offset of order $30\,\mathrm{K}$ relative to iteration 0
on the majority of re-equilibrated faces.

Option (ii) eliminates the dependence on the cold-start guess by
construction: iteration $k\!\geq\!1$ uses the subsurface profile that
iteration 0 actually reached at the end of its equilibration phase
(saved off the GPU once at iteration 0 and held in memory).  The only
difference between iteration 0 and iteration $k$ at the same face is
then the additive indirect-flux term in $Q_{s,i}(t)$, and we recover
the physically required property that
$T^{(k)}_i(t) \geq T^{(0)}_i(t)$ pointwise whenever
$Q^{(k)}_{\mathrm{ind},i}(t) \geq 0$ (true for radiosity solutions
with positive emissivities and non-negative view factors).  We adopt
option (ii) as the default coupling scheme.

In practice we observe that on a $1.8\,\mathrm{M}$-face lunar polar
mesh the coupling iteration converges in 2–3 Picard sweeps; the
first sweep absorbs the dominant indirect-flux signal (typically
warming PSR floors by tens of K) and the subsequent sweeps refine
the spatially-correlated wall–floor feedback.  Because $\mathcal{T}$
in option (ii) executes only the output cycle (not the equilibration),
each iteration is roughly $5\times$ faster than the rad-off thermal
solve.

### 5.4 Selective re-thermal and convergence

The thermal operator $\mathcal{T}$ is the most expensive component
of (8) — typically $\mathcal{O}(\mathrm{minutes})$ on
$\mathcal{O}(10^{6})$ face meshes — so we re-solve only those faces
whose temperature is *expected* to change on the next iteration.
The selection criterion uses a linearized radiative response:

$$
\widehat{\Delta T}_i^{\,(k)}
   \;=\; \frac{(1-\alpha_i)\,\Delta F_i^{\,(k)}}
                {4\,\varepsilon_i\,\sigma\,T_{\mathrm{eff},i}^{3}},
   \qquad
   \Delta F_i^{\,(k)} \;=\;
   \max_t Q_{\mathrm{ind},i}^{\,(k)}(t) - \max_t Q_{\mathrm{ind},i}^{\,(k-1)}(t),
\tag{9}
$$

with $T_{\mathrm{eff},i} = \max\bigl(\max_t T^{(k-1)}_i(t),\, 40\,\mathrm{K}\bigr)$.
A face is included in the re-solve set if $\widehat{\Delta T}_i \geq
s\,\tau_T$ or if its observed change from the prior iteration already
exceeds the same threshold.  The linearization naturally gives cold
PSR floors a larger predicted response per unit flux than warm
crater walls — the desired asymmetry, since cold faces are the ones
sensitive to inter-face IR.

The Picard iteration is terminated when *either* the maximum
face-wise temperature change drops below $\tau_T$ or the fraction
of faces still moving exceeds the configured floor $f_{\min}$:

$$
\Delta T^{\,(k)}_{\max} < \tau_T
   \;\;\mathrm{or}\;\;
   f^{(k)}_{\mathrm{moving}} < f_{\min}.
\tag{10}
$$

Default values are $\tau_T = 0.5\,\mathrm{K}$, $f_{\min} = 0.01$,
$s = 0.5$, $k_{\max} = 10$.  The disjunction is necessary because a
handful of pathological faces — typically PSR-floor edges with
extreme radiative feedback — can oscillate above $\tau_T$ for many
iterations after the bulk of the mesh has converged.

## 6. Implementation

The radiation-exchange module is implemented in C++.  The radiosity
solver, view-factor matrix, and indirect-flux accumulation are all
parallelized; the radiosity Gauss–Seidel/Jacobi inner solver and the
matrix-vector product against $\mathbf{F}$ both have OpenCL
implementations that dispatch across multiple GPU devices when
available.  The matrix-vector product against the sparse CSR
representation of $\mathbf{F}$ uses an SpMV kernel; the radiosity
Jacobi update is a single dense vector operation interleaved with
the SpMV.

On the production workload of 11 lunar polar sites
($\sim\!1.8\,\mathrm{M}$ faces each, 8 windows of 720 hourly steps
per site), the coupled solve costs roughly 2 hours of wall-clock per
site on a 4 GPU (AMD W6800X Duo) workstation, divided approximately
into 6 minutes of view-factor construction (cached and reused across
windows), 7 minutes of Monte-Carlo illumination per window, 4
minutes of iteration-0 thermal solve per window, 2 minutes of
radiosity time-marching per coupling iteration, and 1 minute per
selective re-thermal pass.

## 7. Validation

We validate the module against the analytic two-zone bowl-crater
model of Ingersoll, Svitek & Murray (1992).  A spherical-cap crater
with depth-to-diameter ratio $d/D \in \{0.1, 0.2, 0.3\}$ is
discretized into 1404 triangular faces (20 radial × 36 azimuthal
divisions) and embedded in a flat-plain mesh.  We compute the
PTM3D radiosity solution at four latitudes
($0°, 70°, 80°, 85°$) and compare against the analytic
equilibrium-floor temperature $T_f(\beta)$ at each solar elevation
$\beta$.  Mean per-mesh-face sky view factor $F_\mathrm{sky}$
agrees with the analytic spherical-cap value $\sin^{2}\alpha$ to
within $0.5\%$, with a per-face dispersion of $\pm 0.4\%$.  The
GPU radiosity solver agrees with the reference CPU solver to
$10^{-6}$ relative error.

## 8. Discussion and limitations

Three limitations of the present module are worth flagging for
science use.

**Gray, diffuse, isotropic assumption.**  The radiosity formulation
treats all faces as gray (single-band) diffuse Lambertian emitters
and reflectors.  Spectral effects in the visible (which would change
the partition between absorbed solar and reflected solar for
multi-bounce paths) and in the thermal IR (e.g.\ regolith
emissivity wavelength dependence) are folded into the constants
$\alpha_i$ and $\varepsilon_i$, respectively.  For most planetary
thermal applications the gray approximation is acceptable; for
applications sensitive to spectral inter-face transfer (e.g.\
detailed sublimation modelling at sub-percent precision) a band
generalization of (4) is straightforward but not currently
implemented.

**Steady-state radiosity.**  We solve (5) at each output timestep
treating the radiosity field as instantaneous — that is, the speed
of light is taken as infinite.  This is exactly valid for all
planetary applications.

**Topological closure.**  The view-factor matrix is row-stochastic
modulo the sky aperture (§4) but the sparse storage truncates rows
to the top-$K$ neighbors capturing $\eta = 95\%$ of energy by
default.  The remaining $5\%$ is absorbed into the sky-aperture
term, slightly inflating $F_{i,\mathrm{sky}}$ relative to the
unsparsified estimator.  Reducing $\eta$ to $99\%$ tightens this
loss but increases $K$ by roughly $3\times$ and the memory and
matrix-vector cost in proportion.  We have not found any face class
in our lunar polar test cases where $\eta = 95\%$ leads to a
visible artifact in the converged $\{T_i(t)\}$.

## 9. References

- Hackbusch, W.\ (1999), A sparse matrix arithmetic based on
  $\mathcal{H}$-matrices. Part I: Introduction to $\mathcal{H}$-matrices,
  *Computing*, 62, 89–108.
- Hayne, P.O., et al.\ (2015), Evidence for exposed water ice in the
  Moon's south polar regions from Lunar Reconnaissance Orbiter
  ultraviolet albedo and temperature measurements, *Icarus*, 255,
  58–69.
- Hayne, P.O., et al.\ (2017), Global regolith thermophysical
  properties of the Moon from the Diviner Lunar Radiometer
  Experiment, *J. Geophys. Res. Planets*, 122, 2371–2400.
- Ingersoll, A.P., Svitek, T., Murray, B.C.\ (1992), Stability of
  polar frosts in spherical bowl-shaped craters on the Moon,
  Mercury, and Mars, *Icarus*, 100, 40–47.
- Paige, D.A., et al.\ (2010), Diviner Lunar Radiometer observations
  of cold traps in the Moon's south polar region, *Science*, 330,
  479–482.
- Potter, S., et al.\ (2023), Fast hierarchical view factors for
  topographic radiative transfer, *J. Comp. Phys.*, 491, 112381.
- Vasavada, A.R., Paige, D.A., Wood, S.E.\ (1999), Near-surface
  temperatures on Mercury and the Moon and the stability of polar
  ice deposits, *Icarus*, 141, 179–193.
