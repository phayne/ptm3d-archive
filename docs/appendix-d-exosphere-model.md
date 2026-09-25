# Appendix D: Exosphere Model

## D.1 Introduction and Overview

Europa's surface evolves through the interplay of thermal processing, magnetospheric bombardment, and volatile transport. Water ice, the dominant surface constituent, sublimates from warm regions, travels ballistically through Europa's collisionless exosphere, and redeposits on colder surfaces. Concurrently, magnetospheric plasma sputtering ejects molecules from the surface at rates that can exceed sublimation at low temperatures. These processes redistribute mass across the surface and drive the formation of non-ice lag deposits.

Previous Europa exosphere models have addressed individual aspects of this system. Plainaki et al. (2012, 2013) developed a Monte Carlo model of Europa's exosphere driven by sputtering and sublimation, focusing on atmospheric column densities and escape rates. Vorburger & Wurz (2018) extended this approach with improved sputtering yield models and source rate estimates. Teolis et al. (2017) modeled the exosphere structure with emphasis on plume contributions and surface composition effects. Oza et al. (2019) constrained Europa's atmospheric properties from HST observations.

The PTM3D exosphere module advances beyond these models in two key respects. First, it is directly coupled to a 3D thermophysical model that provides spatially and temporally resolved surface temperatures on an unstructured triangular mesh, rather than relying on parameterized or zonally averaged temperature distributions. This captures the full diurnal cycle, eclipse effects, topographic shadowing, and albedo variations that control sublimation rates. Second, the model tracks the long-term evolution of ice layer composition through a multi-timescale operator-splitting scheme, enabling simulation of geological timescales ($\sim 10^3$ years) where lag deposit formation operates.

The exosphere module computes six coupled processes:

1. **Sublimation**: Temperature-dependent vapor pressure and Hertz-Knudsen emission flux (Section D.2)
2. **Ballistic transport**: Redistribution of sublimated molecules via collisionless trajectories (Section D.3)
3. **Jeans escape**: Thermal loss of high-velocity molecules to space (Section D.4)
4. **Magnetospheric sputtering**: Ejection of surface molecules by energetic ion bombardment (Section D.5)
5. **Ice layer evolution**: Tracking of non-ice concentration within the thermally active skin depth (Section D.6)
6. **Multi-timescale extrapolation**: Operator splitting to bridge diurnal and geological timescales (Section D.7)

The coupling between the thermal and exosphere models is bidirectional. The thermal model provides surface temperatures $T(i, t)$ for each mesh face $i$ at each timestep $t$, and the exosphere model computes the resulting mass fluxes and compositional evolution. In return, the exosphere feeds back into the thermal model through an albedo-composition coupling: as the non-ice concentration $c$ evolves, the surface albedo $A(c)$ is updated and the temperature field is rescaled accordingly (Section D.6.4). This closes the thermal segregation feedback loop (Spencer 1987), enabling self-consistent simulation of the albedo $\to$ temperature $\to$ sublimation $\to$ composition $\to$ albedo cycle.

---

## D.2 Sublimation

The equilibrium vapor pressure of hexagonal water ice follows the Marti & Mauersberger (1993) formulation:

$$\log_{10}\negthinspace\bigl(P_{\mathrm{sat}}\bigr) = \frac{-2663.5}{T} + 12.537 \tag{D1}$$

where $P_{\mathrm{sat}}$ is the saturation vapor pressure in Pa and $T$ is the surface temperature in K. This parameterization was established from laboratory measurements over the range $T = 170$--$250$ K. We extrapolate it to the $80$--$150$ K range relevant to Europa's surface; at these temperatures, no direct measurements exist, though Murphy & Koop (2005) obtain consistent values from thermodynamic integration of heat capacity data. The extrapolation uncertainty is unlikely to exceed a factor of a few in vapor pressure, which is negligible compared to the 12 orders of magnitude variation over the relevant temperature range.

The number density of vapor molecules in local thermodynamic equilibrium is given by the ideal gas law:

$$\rho_v = \frac{P_{\mathrm{sat}} \thinspace M}{R \thinspace T} \tag{D2}$$

where $M = 0.01802$ kg/mol is the molar mass of water and $R = 8.314$ J/(mol$\cdot$K) is the universal gas constant. The mean thermal speed of molecules in a Maxwell-Boltzmann distribution is:

$$\bar{v} = \sqrt{\frac{8 \thinspace k_B \thinspace T}{\pi \thinspace m}} \tag{D3}$$

where $k_B = 1.381 \times 10^{-23}$ J/K and $m = 2.992 \times 10^{-26}$ kg is the molecular mass of H$_2$O.

The sublimation mass flux follows the Hertz-Knudsen equation, assuming a sticking coefficient of unity:

$$\Phi_{\mathrm{sub}} = \frac{\rho_v \thinspace \bar{v}}{4} \tag{D4}$$

Table D1 shows the strong temperature sensitivity of the sublimation flux, which varies by more than 12 orders of magnitude across Europa's surface temperature range. This extreme sensitivity means that the spatial distribution of surface temperature is the dominant control on volatile redistribution.

**Table D1.** Vapor pressure and sublimation flux at representative Europa surface temperatures.

| $T$ (K) | $P_{\mathrm{sat}}$ (Pa) | $\bar{v}$ (m/s) | $\Phi_{\mathrm{sub}}$ (kg/m$^2$/s) | $\Phi_{\mathrm{sub}}$ (molec/m$^2$/s) |
|---------|--------------------------|------------------|-------------------------------------|----------------------------------------|
| 80 | $1.8 \times 10^{-21}$ | 307 | $3.6 \times 10^{-24}$ | $1.2 \times 10^{2}$ |
| 90 | $8.8 \times 10^{-18}$ | 325 | $1.7 \times 10^{-20}$ | $5.7 \times 10^{5}$ |
| 100 | $8.0 \times 10^{-15}$ | 343 | $1.5 \times 10^{-17}$ | $5.0 \times 10^{8}$ |
| 110 | $2.1 \times 10^{-12}$ | 360 | $3.7 \times 10^{-15}$ | $1.2 \times 10^{11}$ |
| 120 | $2.2 \times 10^{-10}$ | 376 | $3.7 \times 10^{-13}$ | $1.2 \times 10^{13}$ |
| 130 | $1.1 \times 10^{-8}$ | 391 | $1.8 \times 10^{-11}$ | $6.1 \times 10^{14}$ |

Key assumptions: (1) The sticking coefficient $\alpha = 1$, meaning all impinging molecules adhere to the surface. (2) Local thermodynamic equilibrium exists between the surface ice and the overlying vapor. (3) The Marti & Mauersberger (1993) extrapolation remains valid below 170 K.

---

## D.3 Ballistic Molecular Transport

### D.3.1 Collisionless Exosphere

Europa's exosphere is collisionless. The mean free path for H$_2$O molecules at the expected surface number densities ($\sim 10^4$ cm$^{-3}$) is approximately $10^6$ km, vastly exceeding the barometric scale height:

$$H = \frac{k_B \thinspace T}{m \thinspace g} \approx 35 \text{ km} \quad (T = 100 \text{ K}) \tag{D5}$$

where $g = 1.314$ m/s$^2$ is Europa's surface gravity. The Knudsen number $\text{Kn} = \ell / H \gg 1$ throughout, so molecular trajectories are purely ballistic between surface contacts.

### D.3.2 Derivation of the Transport Kernel

Consider a molecule departing the surface at position $\mathbf{r}_0$ with speed $v$ at zenith angle $\theta$ from the local vertical. We make the flat-surface approximation, which is valid because the characteristic hop distances (45--60 km; Table D2) are much smaller than Europa's radius ($R_E = 1561$ km), so $d/R_E \lesssim 0.04$.

In uniform gravity $g$, the ballistic range (horizontal distance traveled before returning to the surface) is:

$$d = \frac{v^2 \sin 2\theta}{g} \tag{D6}$$

Molecules are emitted from the surface with a Lambertian angular distribution (cosine law) and a Maxwellian speed distribution at the local surface temperature $T$. The joint probability of emission at speed $v$ and zenith angle $\theta$ is:

$$f(v, \theta) \thinspace dv \thinspace d\theta = \frac{m}{k_B T} \thinspace v^3 \exp\negthinspace\left(-\frac{m v^2}{2 k_B T}\right) \cos\theta \thinspace \sin\theta \thickspace dv \thinspace d\theta \tag{D7}$$

where the $v^3$ factor arises from the product of the Maxwell-Boltzmann speed distribution ($\propto v^2 e^{-mv^2/2k_BT}$) and the flux-weighting factor ($v \cos\theta$), and the $\sin\theta$ term is the solid angle element. The $\cos\theta$ factor enforces the Lambertian emission law.

The probability density for a molecule to land at great-circle distance $d$ is obtained by integrating $f(v, \theta)$ over all $(v, \theta)$ pairs that produce range $d$, subject to the constraint of Equation D6. We derive this result below; to our knowledge, this closed-form kernel has not previously appeared in the literature. Prior Monte Carlo models of volatile migration (e.g., Butler 1997; Schorghofer 2015) simulate individual molecular trajectories numerically rather than evaluating the landing probability analytically.

From Equation D6, a molecule landing at distance $d$ must satisfy $v^2 = gd / \sin 2\theta$. For a given $d$, the zenith angle $\theta$ can range from $0$ to $\pi/4$ (since $\sin 2\theta$ must be positive and achieves its maximum at $\theta = \pi/4$, corresponding to the minimum launch speed). We change variables from $(v, \theta)$ to $(d, \theta)$ using $v^2 = gd/\sin 2\theta$, which gives $v \thinspace dv = gd \cos 2\theta \thinspace d\theta / \sin^2 2\theta$. Substituting into Equation D7 and integrating over all zenith angles that produce range $d$:

$$\mathcal{P}(d, T) = \int_0^{\pi/4} f\bigl(v(d,\theta),\thinspace \theta\bigr) \left|\frac{\partial v}{\partial d}\right| d\theta \tag{D8}$$

Introducing the velocity variance $\sigma^2 = k_B T / m$ and the dimensionless distance parameter $\xi = gd/(2\sigma^2)$, the integrand contains the factor $\exp(-\xi/\sin 2\theta)$. After carrying out the algebra, the integral takes the form:

$$\mathcal{P}(d, T) = \frac{g^2 \thinspace d}{4 \thinspace \sigma^4} \int_0^{\pi/4} \frac{\cos\theta \thinspace \cos 2\theta}{\sin^3 2\theta} \exp\negthinspace\left(\frac{-\xi}{\sin 2\theta}\right) d\theta \tag{D9}$$

With the substitution $u = 1/\sin 2\theta$ (so $u$ ranges from $1$ to $\infty$), this integral reduces to the standard integral representation of the modified Bessel function of the second kind:

$$K_0(\xi) = \int_1^{\infty} \frac{e^{-\xi u}}{\sqrt{u^2 - 1}} \thinspace du \tag{D10}$$

yielding the closed-form transport kernel:

$$\mathcal{P}(d, T) = \frac{\xi^2}{d} \thinspace K_0(\xi) \tag{D11}$$

where $\xi = gd/(2\sigma^2)$. The $K_0$ function has a logarithmic singularity at $\xi = 0$ (molecules landing very close to the source) and decays exponentially as $K_0(\xi) \sim e^{-\xi}/\sqrt{\xi}$ for $\xi \gg 1$, providing a natural cutoff at large distances. We truncate the kernel at $\xi = 10$, beyond which $K_0 < 2 \times 10^{-5}$ and the contribution is negligible.

This analytical kernel enables direct construction of the transport matrix without Monte Carlo sampling, providing exact results (within the flat-surface approximation) at a computational cost that scales as $O(Nk)$ rather than $O(N \times N_{\mathrm{particles}})$, where $k$ is the number of mesh neighbors within the $K_0$ cutoff radius.

The $K_0$ function is evaluated using polynomial approximations from Abramowitz & Stegun (1964):

- For $x \le 2$: a series expansion with a logarithmic term
- For $x > 2$: the asymptotic expansion $K_0(x) \approx e^{-x} / \sqrt{x} \thickspace \times \thickspace \text{polynomial}(2/x)$

### D.3.3 Transport Matrix

The transport kernel is discretized on the triangular mesh. The transport matrix element $M_{ji}$ gives the fraction of mass emitted from face $i$ that is deposited on face $j$:

$$M_{ji} = \mathcal{P}(d_{ij}, T_i) \thickspace A_j \thickspace \mathcal{C}_{ij} \tag{D12}$$

where $d_{ij}$ is the great-circle distance between face centers, $A_j$ is the area of the target face, and $\mathcal{C}_{ij}$ is a ring-density correction factor (see below). The deposition flux at face $j$ from all emitting faces is then:

$$\Phi_{\mathrm{dep},j} = \sum_{i} M_{ji} \thickspace \Phi_{\mathrm{sub},i} \thickspace A_i \thickspace / \thickspace A_j \tag{D13}$$

The matrix satisfies the mass conservation constraint:

$$\sum_{j} M_{ji} = 1 - f_{\mathrm{esc}}(T_i) \tag{D14}$$

where $f_{\mathrm{esc}}$ is the Jeans escape fraction (Section D.4). This ensures that all emitted mass is either redeposited on the surface or escapes to space.

**Ring-density correction.** Icosahedral mesh subdivision produces a quasi-uniform triangulation, but residual packing variations create small systematic biases in the transport matrix. We correct for this by computing the ratio of the continuous ring area $2\pi R_E \sin\theta \thickspace \Delta\theta$ to the total discrete face area within each angular shell around the source face, and scaling the matrix entries accordingly. This correction removes mesh-topology artifacts from the mass redistribution pattern.

The transport matrix is stored in compressed sparse column format (Eigen library), exploiting the exponential decay of $K_0$ at large distances. For meshes with $> 10^5$ faces, the matrix is $> 99$% sparse.

**Table D2.** Characteristic hop distances for H$_2$O molecules on Europa.

| $T$ (K) | $\bar{v}$ (m/s) | $d_{\mathrm{char}}$ (km) | $d_{\mathrm{ball}}$ (km) |
|---------|------------------|--------------------------|--------------------------|
| 100 | 343 | 70 | 45 |
| 110 | 360 | 77 | 49 |
| 120 | 376 | 84 | 54 |
| 130 | 391 | 91 | 58 |

The characteristic distance $d_{\mathrm{char}} = 2\sigma^2/g$ corresponds to the peak of the $K_0$ probability kernel ($\xi = 1$). The ballistic distance $d_{\mathrm{ball}} = \bar{v}^2/(2g)$ is the maximum range of a molecule launched vertically.

Key assumptions: (1) Collisionless exosphere (Knudsen number $\gg 1$). (2) Uniform gravity (no topographic deflection of trajectories). (3) Lambertian emission (isotropic half-space velocity distribution). (4) Flat-surface approximation ($d \ll R_E$).

---

## D.4 Jeans Thermal Escape

Molecules in the high-energy tail of the Maxwell-Boltzmann distribution can exceed Europa's escape velocity ($v_{\mathrm{esc}} = 2025$ m/s) and be lost to space. The escape fraction is computed using the Chamberlain (1963) approximation. The Jeans parameter quantifies the ratio of gravitational binding energy to thermal energy:

$$\lambda_J = \frac{v_{\mathrm{esc}}^2 \thinspace m}{2 \thinspace k_B \thinspace T} \tag{D15}$$

The escape fraction is:

$$
f_{\mathrm{esc}} = \begin{cases}
0 & \lambda_J > 50 \\
\dfrac{e^{-\lambda_J}(1 + \lambda_J)}{2\sqrt{\pi\thinspace\lambda_J}} & 2 \le \lambda_J \le 50 \\
0.5 & \lambda_J < 2
\end{cases} \tag{D16}
$$

**Table D3.** Jeans escape parameter and escape fraction for H$_2$O on Europa.

| $T$ (K) | $\lambda_J$ | $f_{\mathrm{esc}}$ |
|---------|-------------|---------------------|
| 80 | 55.5 | 0 |
| 100 | 44.4 | $\sim 10^{-19}$ |
| 120 | 37.0 | $\sim 10^{-16}$ |
| 150 | 29.6 | $\sim 10^{-13}$ |
| 200 | 22.2 | $\sim 10^{-10}$ |

Water escape from Europa is negligible at all relevant surface temperatures. The Jeans parameter $\lambda_J \gg 1$ throughout, meaning the escape velocity far exceeds the thermal velocity. Essentially all sublimated H$_2$O molecules are redeposited on the surface. Jeans escape is relevant only for lighter sputtering products (O$_2$, H$_2$; see Section D.5).

The escape fraction enters the model through the transport matrix normalization (Equation D14): the column sum of the transport matrix equals $1 - f_{\mathrm{esc}}$.

---

## D.5 Magnetospheric Plasma Sputtering

Europa orbits within Jupiter's magnetosphere and is continuously bombarded by energetic ions, primarily S$^+$, O$^+$, and H$^+$ from the Io plasma torus. This sputtering ejects molecules directly from the surface ice, operating independently of temperature (unlike sublimation) and providing the dominant mass-loss mechanism on the cold nightside and polar regions.

### D.5.1 Global Sputtering Rates

We adopt the global sputtering rates from Cassidy et al. (2013), summarized in Table D4.

**Table D4.** Global sputtering rates for Europa (Cassidy et al. 2013).

| Species | Rate (molecules/s) | Mass rate (kg/s) |
|---------|-------------------|-------------------|
| H$_2$O | $2.0 \times 10^{27}$ | 59.8 |
| O$_2$ | $1.0 \times 10^{26}$ | 5.3 |
| H$_2$ | $2.0 \times 10^{26}$ | 0.7 |

The mean H$_2$O sputtering flux averaged over Europa's surface ($A_E = 3.09 \times 10^{13}$ m$^2$) is:

$$\bar{\Phi}_{\mathrm{sput}} = \frac{\dot{N}_{\mathrm{H_2O}} \thinspace m}{A_E} = \frac{2.0 \times 10^{27} \times 2.99 \times 10^{-26}}{3.09 \times 10^{13}} = 1.94 \times 10^{-12} \text{ kg/m}^2\text{/s} \tag{D17}$$

### D.5.2 Spatial Distribution

The sputtering flux varies across the surface due to the geometry of magnetospheric ion precipitation. The spatial distribution has two components:

**Longitude dependence (trailing hemisphere enhancement).** Jupiter's magnetospheric plasma corotates faster than Europa orbits, so ions preferentially impact the trailing hemisphere (centered at $270°$W longitude). We parameterize this as:

$$f_{\mathrm{lon}} = \frac{1}{C_{\mathrm{hem}}} + \left(1 - \frac{1}{C_{\mathrm{hem}}}\right) \frac{\cos(\phi_{\mathrm{lon}} - 270°) + 1}{2} \tag{D18}$$

where $C_{\mathrm{hem}} = 3.0$ is the trailing-to-leading hemisphere flux contrast ratio. This gives $f_{\mathrm{lon}} = 1$ at the trailing apex and $f_{\mathrm{lon}} = 1/C_{\mathrm{hem}} = 0.33$ at the leading apex.

**Latitude dependence (polar enhancement).** Energetic ions are guided along magnetic field lines toward the poles, producing a modest polar enhancement:

$$f_{\mathrm{lat}} = 1 + (E_{\mathrm{pol}} - 1) \frac{|\phi_{\mathrm{lat}}|}{90°} \tag{D19}$$

where $E_{\mathrm{pol}} = 1.2$ is the polar enhancement factor. This linear parameterization is a simplified representation of the full Nordheim et al. (2022) ion flux maps, which show more complex latitude structure particularly for different ion species and energies. Our parameterization captures the dominant spatial trend while remaining computationally tractable for long-duration simulations.

### D.5.3 Temperature-Dependent Yield

The sputtering yield increases with temperature due to thermally-enhanced diffusion of radiolytic products to the surface (Teolis et al. 2017):

$$Y(T) = Y_0 \left[1 + q_0 \thinspace \exp\negthinspace\left(\frac{-Q}{k_B \thinspace T}\right)\right] \tag{D20}$$

where $q_0 = 1000$ is a dimensionless prefactor and $Q = 0.06$ eV is the activation energy for thermally-enhanced sputtering. We normalize the yield relative to a reference temperature $T_{\mathrm{ref}} = 100$ K:

$$\hat{Y}(T) = \frac{1 + q_0 \thinspace e^{-Q/(k_B T)}}{1 + q_0 \thinspace e^{-Q/(k_B T_{\mathrm{ref}})}} \tag{D21}$$

**Table D5.** Temperature-dependent sputtering yield enhancement.

| $T$ (K) | $\hat{Y}(T)$ |
|---------|---------------|
| 80 | 0.60 |
| 100 | 1.00 |
| 110 | 1.43 |
| 120 | 2.07 |
| 130 | 2.94 |
| 150 | 5.47 |

### D.5.4 Sputtering Flux Per Face

The sputtering mass flux at mesh face $i$ is:

$$\Phi_{\mathrm{sput},i} = \bar{\Phi}_{\mathrm{sput}} \thickspace f_{\mathrm{lon},i} \thickspace f_{\mathrm{lat},i} \thickspace \hat{Y}(T_i) \thickspace S \tag{D22}$$

where $S$ is a user-configurable rate scale factor (default $S = 1$).

### D.5.5 Sputtering versus Sublimation

Sublimation exceeds the mean sputtering flux at $T \gtrsim 125$ K. At Europa's subsolar point ($\sim 130$ K), sublimation dominates by approximately an order of magnitude. At nightside and polar temperatures ($80$--$100$ K), sputtering dominates by many orders of magnitude. The sputtering-sublimation crossover temperature is a critical boundary: above it, ice loss is controlled by thermodynamics and is highly sensitive to surface temperature; below it, ice loss is controlled by magnetospheric conditions and is relatively temperature-insensitive.

The gross sputtered mass is ice removed from the face. H$_2$O constitutes a fraction $f_w = 0.85$ of it (`--sputter-h2o-fraction`); the remainder is radiolytic O$_2$ and H$_2$ in the stoichiometric mass ratio of $2\thinspace\mathrm{H_2O} \rightarrow 2\thinspace\mathrm{H_2} + \mathrm{O_2}$ ($m_{\mathrm{O_2}} : 2 m_{\mathrm{H_2}} = 7.94 : 1$), i.e. 13.3% O$_2$ and 1.7% H$_2$. The escaping shares are 10% (H$_2$O), 90% (O$_2$) and 95% (H$_2$), prescribed inputs (`--sputter-escape-h2o/-o2/-h2`). The non-escaping H$_2$O (76.5% of gross) is launched through the same ballistic kernel as sublimated molecules and redeposits; the non-escaping O$_2$ and H$_2$ (1.4%) are non-condensable at Europa surface temperatures and are booked as an atmospheric reservoir that never returns as ice. The surface therefore loses $1 - f_w(1 - e_{\mathrm{H_2O}}) = 23.5$% of the gross sputtered mass, and the run log carries a global audit, gross = redeposited + escaped + reservoir, closed to $10^{-6}$ (PTM-642). Before PTM-642 only the escaping 22.25% was removed and the rest was neither removed nor transported (reproducible with `--sputter-legacy-escape-only`).

Key assumptions: (1) Steady-state magnetospheric conditions (no time variation from Jovian magnetotail passages). (2) Smooth cosine variation of the ion flux, simplified from full particle-tracing simulations. (3) The non-escaping sputtered H$_2$O is launched with the thermal kernel at an equivalent temperature of 683 K (`--sputter-launch-temperature`): the mean energy, 0.118 eV, of a Sigmund-Thompson distribution ($U = 0.055$ eV) truncated at the H$_2$O escape energy (0.383 eV), equated to $2kT$. Its hop scale $2kT/(mg) \approx 480$ km is about $0.31 R_E$, where the flat, uniform-gravity kernel is approximate, so the redistribution pattern (not the global mass accounting) is approximate.

---

## D.6 Ice Layer Model

The ice layer model tracks the composition of the thermally active surface layer---the uppermost region where diurnal temperature oscillations are significant. This layer exchanges mass with the exosphere through sublimation and deposition, and with the magnetosphere through sputtering.

### D.6.1 State Variable

The primary state variable is the non-ice concentration $c$ (dimensionless, $0 \le c \le 1$), representing the volume fraction of refractory (non-volatile) material within the thermal skin depth layer of thickness $\delta_{\mathrm{th}}$. When $c = 0$, the layer is pure ice; when $c = 1$, all ice has been removed, leaving only refractory residue.

The ice mass per unit area in the layer is:

$$m_{\mathrm{ice}} = (1 - c) \thickspace \delta_{\mathrm{th}} \thickspace \rho_{\mathrm{ice}} \tag{D23}$$

where $\delta_{\mathrm{th}} = 0.05$ m is the thermal skin depth (default value, appropriate for Europa's ice regolith with thermal inertia $\sim 60$ SI units) and $\rho_{\mathrm{ice}} = 917$ kg/m$^3$ is the water ice density.

### D.6.2 Mass Balance

**Sublimation and sputtering** remove ice from the layer, increasing $c$:

$$\Delta c = \frac{\Delta m_{\mathrm{loss}}}{\delta_{\mathrm{th}} \thickspace \rho_{\mathrm{ice}}} \tag{D24}$$

where $\Delta m_{\mathrm{loss}}$ is the ice mass removed per unit area over a timestep (from either sublimation or sputtering). Sublimation and sputtering are only active when $c < 1$ (ice is present).

**Deposition** fills void space with fresh ice, decreasing $c$:

$$\Delta c = -\frac{\min\negthinspace\bigl(\Delta m_{\mathrm{dep}},\thickspace c \thinspace \delta_{\mathrm{th}} \thinspace \rho_{\mathrm{ice}}\bigr)}{\delta_{\mathrm{th}} \thickspace \rho_{\mathrm{ice}}} \tag{D25}$$

Deposition cannot reduce $c$ below zero (the layer cannot hold more ice than its volume allows).

### D.6.3 Lag Deposit Formation

Europa's ice contains a small fraction of refractory contaminants (silicates, salts, organics), parameterized as an initial dust fraction $f_{\mathrm{dust}} = 0.01$ (1% by volume, default). As ice sublimates, this non-volatile material accumulates at the surface, forming a lag deposit. The lag deposit does not sublimate, so the non-ice concentration $c$ increases monotonically in the absence of deposition.

This produces a self-limiting feedback: as the lag deposit thickens, it armors the underlying ice against further sublimation (in reality, by reducing the ice surface area exposed to vacuum). In the legacy well-mixed model (`--no-lag-barrier`), sublimation ceases entirely when $c = 1$, and nothing else throttles it. The multi-timescale evolution instead uses a lag-layer diffusion barrier by default (Section D.6.5).

Key assumptions: (1) The layer is well-mixed (uniform concentration throughout the skin depth). (2) Vertical mixing is instantaneous. (3) The lag deposit does not alter the thermal conductivity (but does alter the albedo; see Section D.6.4). (4) No grain-size dependence of sublimation rate.

### D.6.4 Albedo-Composition Coupling

As ice is removed and the non-ice concentration $c$ increases, the surface darkens, increasing the absorbed solar flux and raising the temperature. This positive feedback---thermal segregation of volatiles (Spencer 1987)---is captured through a composition-dependent albedo and a radiative equilibrium temperature correction applied at each geological epoch.

The Bond albedo is computed from the non-ice concentration via linear mixing between pure ice and non-ice endmembers:

$$A(c) = A_{\mathrm{ice}} \thinspace (1 - c) + A_{\mathrm{nonice}} \thinspace c \tag{D26}$$

where $A_{\mathrm{ice}} = 0.65$ is the Bond albedo of pure water ice and $A_{\mathrm{nonice}} = 0.10$ is the Bond albedo of the non-ice lag deposit (silicates, salts, organics). These endmember values span the range observed on Europa's surface.

When the albedo changes from $A_{\mathrm{old}}$ to $A_{\mathrm{new}}$, the surface temperature field is rescaled using the radiative equilibrium relation $T \propto (1 - A)^{1/4}$:

$$T_{\mathrm{new}}(i, t) = T_{\mathrm{old}}(i, t) \left(\frac{1 - A_{\mathrm{new},i}}{1 - A_{\mathrm{old},i}}\right)^{1/4} \tag{D27}$$

This scaling is applied to all diurnal timesteps $t$ for each face $i$ where the albedo change exceeds a threshold $\Delta A_{\mathrm{min}} = 0.01$. The updated temperatures then produce modified sublimation and sputtering rates in the next epoch, closing the feedback loop.

The thermal segregation feedback is self-reinforcing: darker surfaces absorb more sunlight, sublimate faster, concentrate more non-ice material, and darken further. This runaway process is ultimately limited by either complete ice depletion ($c = 1$) or by deposition of fresh ice from colder regions. The competition between these effects determines the steady-state spatial pattern of ice and lag deposits.

### D.6.5 Lag-Layer Diffusion Barrier

The well-mixed layer of Sections D.6.1--D.6.3 books every kilogram of lost ice as non-ice volume ($\Delta c = \Delta m / \delta_{\mathrm{th}} \rho_{\mathrm{ice}}$), so the void left by sublimation is counted as dark material, and the exposed ice area is the only throttle on sublimation. In the `hapke-intimate` mode that feedback is linearly unstable on warm faces (PTM-658). By default the multi-timescale evolution replaces it with a lag layer on top of an unchanged mixture (`--no-lag-barrier` restores the legacy model, for comparison).

**Lag growth.** The mixture keeps its $t = 0$ non-ice volume fraction $x_n$ (frozen from the initialization). Removing a net ice mass $\Delta M$ per unit area frees $x_n \Delta M / [(1 - x_n)\rho_{\mathrm{ice}}]$ of non-ice volume, which stays behind as a lag of porosity $\phi_{\mathrm{lag}}$:

$$\Delta d = \beta \thinspace \Delta M, \qquad \beta = \frac{x_n}{(1 - x_n) \thinspace \rho_{\mathrm{ice}} \thinspace (1 - \phi_{\mathrm{lag}})} \tag{D27a}$$

The substrate ice, the lag thickness $d$, and the non-ice mass of the lag, $\rho_{\mathrm{dust}} (1 - \phi_{\mathrm{lag}}) d$, are tracked per face. A pure-ice face ($x_n = 0$) forms no lag.

**Barrier.** Vapor from the ice table crosses the lag by Knudsen diffusion, $D_K = [\pi/(8+\pi)] [\phi/(1-\phi)] \bar{v} r / \tau$ (Schorghofer 2008, eq. 3), in series with the free-surface flux $J_0 = \rho_{\mathrm{sat}} \bar{v}/4$:

$$J_{\mathrm{sub}} = \frac{J_0 \thinspace f(x_n)}{1 + d/\ell}, \qquad \ell = \frac{4 D_K}{\bar{v}} = \frac{4\pi}{8+\pi} \thinspace \frac{\phi_{\mathrm{lag}}}{1 - \phi_{\mathrm{lag}}} \thinspace \frac{r}{\tau} \tag{D27b}$$

where $f(x_n)$ is the mixture's exposed ice area fraction and $4\pi/(8+\pi) = 1.128$. With the defaults $r = 5$ µm, $\phi_{\mathrm{lag}} = 0.6$, $\tau = 2$, $\ell = 4.23$ µm (`--lag-barrier-length` sets $\ell$ directly). Sputtering is **not** barriered: energetic ions reach at least centimeter depths, so the sputter flux stays $J_{\mathrm{sput}} f(x_n)$ (Paul Hayne, 2026-09-24).

**Deposition and the frost veneer.** On a face whose net loss $E_S = J_0 f/(1+d/\ell) + J_{\mathrm{sput}} f$ exceeds the deposition $B$, deposition offsets the loss and the lag grows at $\beta (E_S - B)$. Such a face approaches $d^* = \ell\thinspace [J_0 f/(B - J_{\mathrm{sput}} f) - 1]$, where it stops being a net water source. Deposition beyond the free-frost loss $J_0 + J_{\mathrm{sput}}$ builds a frost veneer $m_f$ that sublimates and sputters unbarriered over the whole face; the barrier applies again only once $m_f = 0$. Between the two ($E_S < B < J_0 + J_{\mathrm{sput}}$) the face recycles: frost that lands sublimates again as fast as it arrives, so the emission equals $B$ and the state does not change. Deposition never removes lag. The veneer replaces the legacy off-book "layer overflow", and the global inventory (substrate plus veneer) closes against the booked fluxes to rounding.

**Integration.** With $u = 1 + d/\ell$ and $B$ held over a step (Section D.7), the substrate regime separates exactly,

$$t = \frac{\ell}{\beta} \int_{u_0}^{u} \frac{u' \thinspace du'}{J_0 f + (J_{\mathrm{sput}} f - B)\thinspace u'} \tag{D27c}$$

which for $B = J_{\mathrm{sput}} = 0$ gives $d + d^2/2\ell = \beta J_0 f t$. The step is solved in closed form (no hold-$d$ approximation), including the veneer phase and substrate exhaustion. Super-steps end when faces carrying 10% of the current emission have changed their barrier factor by 10%, and the transport is refreshed when the emission pattern has moved by more than 2% (total variation; `--lag-source-refresh-tol`).

**Albedo.** The lag darkens the face toward the non-ice endmember,

$$A = A_{\mathrm{lag}} + (A_{\mathrm{mix}} - A_{\mathrm{lag}}) \thinspace e^{-d/\delta_{\mathrm{opt}}} \tag{D27d}$$

with $A_{\mathrm{mix}}$ the albedo of the mixture (Equation D26, or the Hapke intimate mixture), $A_{\mathrm{lag}}$ the non-ice endmember, and $\delta_{\mathrm{opt}} = 10$ µm. At $t = 0$, $d = 0$ on every face, so the availabilities, the albedo and the initialization gates are unchanged bit for bit; the run checks this at startup. The frost veneer is optically neutral, and the lag does not change the thermal inertia (the run warns if any lag exceeds 5 mm).

### D.6.6 Impact Gardening and the Sputter Split

Without mixing, the lag of Section D.6.5 darkens every net-source face within $\sim 10^4$ yr, far faster than Europa's 10--200 Myr surface age. Two refinements (Paul Hayne's rulings, 2026-09-24; Fermi's formulation) are on by default (`--no-gardening --no-sputter-shield` restores D.6.5 exactly).

**Sputter split.** The gross sputter flux is split per face into H$_2$O ejection $p_w$ and radiolytic O$_2$ + H$_2$ $p_r$ (the ice consumed by radiolysis is 36/32 of the O$_2$ mass). Ions deposit their energy within $\sim 0.1$ to a few µm (Plainaki et al. 2010, p. 389), and H$_2$O ejected below the top grains sticks to the regolith (Johnson et al. 2004, pp. 494, 502; Cassidy & Johnson 2005), so the ejection is **shielded** by the lag like sublimation. O$_2$ does not stick and diffuses out (Johnson et al. 2004; Johnson et al. 2019), so the radiolytic loss stays **unshielded**. Caveat: the engine's O$_2$ source uses the Teolis thin-layer ($\sim 28$ Å) yields, which read literally would be shielded too; the unshielded ruling rests on the trapped-and-diffusing picture.

**Lag with gardening.** Micrometeoroid gardening at $v_m$ mixes the lag back into the box:

$$\frac{dd}{dt} = \beta_c \left[\frac{a + p_w}{1 + d/\ell} + p_r - B\right] - v_m \thinspace \mathbb{1}(d > 0), \qquad \beta_c = \frac{c}{(1 - c)\thinspace \rho_{\mathrm{ice}} \thinspace (1 - \phi)} \tag{D27e}$$

with $c$ the CURRENT box non-ice fraction and $v_m = 0.3$ µm/yr (arm 1.2; Johnson et al. 2004, Table 20.2 p. 490 and Section 20.4.1 p. 491, citing Cooper et al. 2001). This is Equation D27c with $a \to a + p_w$ and $p \to p_r - v_m/\beta_c$, so the closed form carries over. $d = 0$ is absorbing where $\beta_c (a + p_w + p_r - B) \le v_m$ (the Spencer criterion): such a face holds no lag and its box loses ice, so its $c$ rises. Elsewhere the lag tends to $d^*$ with $\beta_c [(a + p_w)/u^* + p_r - B] = v_m$. While $d > 0$ the non-ice freed by ice loss goes to the lag ($c$ fixed) and the gardened lag returns $(1 - \phi)\thinspace v_m$ of non-ice volume to the box. A frost veneer is gardened into the box ice at $\rho_{\mathrm{ice}}(1 - \phi) v_m$.

**Box depth.** The mixed box deepens as $h_g(t) = \sqrt{h_{\mathrm{mm}}^2 + K_{99} t}$, $h_{\mathrm{mm}} = 1$ cm, with $K_{99} = u \pi c^2 h^2 / (2 \times 4.605)$ from Costello et al. (2021) Eq. 1 for secondary craters ($v = -4$) at the 99% level: $3.84 \times 10^{-9}$ m$^2$/yr (Case B, default) or $1.10 \times 10^{-9}$ (Case A). Column material at the $t = 0$ composition is entrained at $dh_g/dt$, and the global audit books it: $\Delta(\mathrm{inventory}) = \mathrm{entrained} - (\mathrm{sublimated} + \mathrm{sputtered} - \mathrm{deposited})$. The values $h = 0.04$ and $c = 1$ are not in Costello et al. (2021) and are unverified.

---

## D.7 Multi-Timescale Evolution

### D.7.1 Motivation

The diurnal cycle of sublimation and deposition has a period $P_{\mathrm{diurnal}} = 3.5512$ days ($306{,}824$ s), while the processes of ice layer evolution, lag deposit formation, and thermal segregation operate over geological timescales of $10^2$--$10^3$ years. Direct simulation of every diurnal timestep over these durations is computationally prohibitive ($> 10^7$ diurnal cycles for 1000 years). We employ operator splitting to bridge these timescales.

### D.7.2 Algorithm

The evolution proceeds in four steps:

**Step 0: Albedo update (optional).** If albedo-composition coupling is enabled (Section D.6.4), compute the current albedo $A(c_i)$ for each face from Equation D26. If the maximum albedo change across all faces exceeds the threshold $\Delta A_{\mathrm{min}}$, rescale the temperature field using Equation D27 and force a recomputation of the diurnal cycle (Step 1).

**Step 1: Diurnal averaging.** Run one complete diurnal cycle ($N_t$ timesteps) with the current surface state. Compute time-averaged fluxes at each face:

$$\bar{\Phi}_{\mathrm{sub},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{sub},i}(t), \quad \bar{\Phi}_{\mathrm{dep},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{dep},i}(t), \quad \bar{\Phi}_{\mathrm{sput},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{sput},i}(t) \tag{D28}$$

**Step 2: Epoch extrapolation.** Apply the time-averaged fluxes over one epoch of $N_c$ diurnal cycles:

$$\Delta t_{\mathrm{epoch}} = N_c \thickspace P_{\mathrm{diurnal}} \tag{D29}$$

$$\Delta c_i = \frac{\bar{\Phi}_{\mathrm{sub},i} + \bar{\Phi}_{\mathrm{sput},i} - \bar{\Phi}_{\mathrm{dep},i}}{\delta_{\mathrm{th}} \thickspace \rho_{\mathrm{ice}}} \thickspace \Delta t_{\mathrm{epoch}} \tag{D30}$$

**Step 3: Iterate.** Repeat Steps 0--2 for $N_e$ epochs.

### D.7.3 Flux Scaling for Depleted Regions

As ice is progressively removed from warm faces ($c \to 1$), the sublimation flux from those faces decreases. To maintain global mass conservation, the deposition flux is rescaled:

$$\bar{\Phi}_{\mathrm{dep}}^{(\mathrm{scaled})} = \bar{\Phi}_{\mathrm{dep}}^{(\mathrm{base})} \thickspace \frac{\sum_i \Phi_{\mathrm{em},i}^{(\mathrm{current})}}{\sum_i \Phi_{\mathrm{em},i}^{(\mathrm{base})}} \tag{D31}$$

where $\Phi_{\mathrm{em},i}^{(\mathrm{base})}$ is the emission flux computed with the initial (undepleted) state and $\Phi_{\mathrm{em},i}^{(\mathrm{current})}$ reflects the current ice availability (zero emission from ice-depleted faces).

### D.7.4 Analytical Acceleration

For faces where the net mass-loss rate implies a timescale much longer than $\Delta t_{\mathrm{epoch}}$, the model uses the analytical solution of the concentration evolution ODE directly, avoiding unnecessary numerical time-stepping. This is particularly effective for cold polar regions where sublimation is negligible and evolution is dominated by the slow accumulation of sputtered debris.

### D.7.5 Default Parameters

**Table D6.** Default multi-timescale evolution parameters.

| Parameter | Default | Description |
|-----------|---------|-------------|
| $N_e$ | 1000 | Number of geological epochs |
| $N_c$ | 100 | Diurnal cycles per epoch ($\approx 1$ Earth year) |
| Export interval | 100 | Export data every $N$ epochs |
| Analytical acceleration | on | Use closed-form solutions for slow regions |
| Cache refresh | adaptive | Recompute diurnal cycle when conditions change significantly |

With default parameters: $N_c \times N_e = 10^5$ orbits $= 10^5 \times 3.5512$ d $/$ $365.25$ d/yr $= 972$ years of geological evolution.

### D.7.6 Approximation Validity

The operator-splitting approach assumes that diurnal-averaged fluxes are representative of the full diurnal cycle, and that surface temperatures do not change significantly within an epoch. The first assumption is well justified because the transport matrix, which depends on temperature, is slowly varying compared to the diurnal cycle. The second assumption requires that changes in non-ice concentration (which could alter albedo and hence temperature) be small over one epoch; this is enforced by the adaptive cache refresh, which triggers a recomputation of the diurnal cycle when the maximum concentration change exceeds a threshold. *Correction pending: the current code implements no such adaptive refresh; it recomputes the diurnal cycle only at the first epoch and when albedo coupling updates temperatures, and it rejects the `--cache-refresh` and `--no-analytical` options.*

When albedo-composition coupling is enabled (Section D.6.4), the model self-consistently captures the thermal segregation feedback (Spencer 1987): as ice is removed and the lag deposit thickens, the surface albedo decreases, increasing temperatures and accelerating further ice loss. The temperature rescaling (Equation D27) uses a radiative equilibrium approximation, which is accurate for the slowly varying geological timescales but does not account for second-order effects such as changes in thermal inertia or emissivity with composition. These refinements are left for future work.

---

## D.8 Model Parameters

### D.8.1 Europa Physical Properties

**Table D7.** Adopted Europa parameters.

| Symbol | Value | Description | Source |
|--------|-------|-------------|--------|
| $R_E$ | $1.5608 \times 10^6$ m | Mean radius | IAU |
| $g$ | 1.314 m/s$^2$ | Surface gravity | $GM/R_E^2$ |
| $v_{\mathrm{esc}}$ | 2025 m/s | Escape velocity | $\sqrt{2gR_E}$ |
| $A_E$ | $3.09 \times 10^{13}$ m$^2$ | Surface area | $4\pi R_E^2$ |
| $P_{\mathrm{diurnal}}$ | 3.5512 d (306,824 s) | Rotation/orbital period | Synchronous |

### D.8.2 Water Ice Properties

**Table D8.** Water ice parameters.

| Symbol | Value | Description |
|--------|-------|-------------|
| $m$ | $2.992 \times 10^{-26}$ kg | H$_2$O molecular mass |
| $M$ | 0.01802 kg/mol | H$_2$O molar mass |
| $\rho_{\mathrm{ice}}$ | 917 kg/m$^3$ | Ice density |

### D.8.3 Model-Specific Parameters

**Table D9.** Adopted model parameters with uncertainties and sensitivities.

| Parameter | Value | Reference | Sensitivity |
|-----------|-------|-----------|-------------|
| Vapor pressure coefficients | $-2663.5/T + 12.537$ | Marti & Mauersberger (1993) | Moderate: extrapolation below 170 K introduces factor $\sim 2$ uncertainty |
| Sticking coefficient $\alpha$ | 1.0 | Standard assumption | Low: reducing to 0.5 would halve sublimation flux |
| H$_2$O sputtering rate | $2.0 \times 10^{27}$ s$^{-1}$ | Cassidy et al. (2013) | Moderate: uncertain by factor $\sim 2$--$3$ |
| Hemisphere contrast $C_{\mathrm{hem}}$ | 3.0 | Cassidy et al. (2013) | Low: affects spatial distribution but not total rate |
| Yield activation energy $Q$ | 0.06 eV | Teolis et al. (2017) | Moderate: controls sputtering-sublimation crossover |
| Ice albedo $A_{\mathrm{ice}}$ | 0.65 | Europa observations | Moderate: sets initial sublimation rate |
| Non-ice albedo $A_{\mathrm{nonice}}$ | 0.10 | Estimated | High: controls thermal segregation strength |
| Albedo update threshold $\Delta A_{\mathrm{min}}$ | 0.01 | Numerical | Low: convergence parameter |
| Thermal skin depth $\delta_{\mathrm{th}}$ | 0.05 m | Thermal model | Moderate: sets ice reservoir capacity |
| Dust fraction $f_{\mathrm{dust}}$ | 0.01 | Estimated | Low: sets initial lag deposit seed |

The most sensitive parameters are the global H$_2$O sputtering rate (Cassidy et al. 2013), uncertain by a factor of $\sim 2$--$3$, and the non-ice endmember albedo $A_{\mathrm{nonice}}$, which controls the strength of the thermal segregation feedback. The sputtering rate uncertainty directly controls the sputtering-sublimation crossover temperature and therefore the latitude at which magnetospheric erosion gives way to thermally-driven ice loss. The non-ice albedo sets the contrast between ice-covered and ice-depleted regions in Equation D26, and thus the magnitude of the temperature increase as lag deposits form. However, the qualitative spatial patterns---equatorial ice depletion and polar accumulation---are robust to these uncertainties.

---

## References

Abramowitz, M. & Stegun, I.A. (1964). *Handbook of Mathematical Functions*. National Bureau of Standards.

Butler, B.J. (1997). The migration of volatiles on the surfaces of Mercury and the Moon. *J. Geophys. Res.*, 102(E8), 19283--19291.

Cassidy, T.A. et al. (2013). Magnetospheric ion sputtering and water ice grain size at Europa. *Planet. Space Sci.*, 77, 64--73. doi:10.1016/j.pss.2012.07.008

Chamberlain, J.W. (1963). Planetary coronae and atmospheric evaporation. *Planet. Space Sci.*, 11(8), 901--960. doi:10.1016/0032-0633(63)90122-3

Marti, J. & Mauersberger, K. (1993). A survey and new measurements of ice vapor pressure at temperatures between 170 and 250 K. *Geophys. Res. Lett.*, 20(5), 363--366. doi:10.1029/93GL00105

Murphy, D.M. & Koop, T. (2005). Review of the vapour pressures of ice and supercooled water for atmospheric applications. *Q. J. R. Meteorol. Soc.*, 131(608), 1539--1565. doi:10.1256/qj.04.94

Nordheim, T.A. et al. (2022). Magnetospheric Ion Bombardment of Europa's Surface. *Planet. Sci. J.*, 3, 5. doi:10.3847/PSJ/ac382a

Oza, A.V. et al. (2019). Sodium and Potassium Atmospheres of the Icy Moons. *Astrophys. J.*, 885(2), 168.

Plainaki, C. et al. (2012). The role of sputtering and radiolysis in the generation of the Europa exosphere. *Icarus*, 218(2), 956--966.

Plainaki, C. et al. (2013). Exospheric O$_2$ densities at Europa during different orbital phases. *Planet. Space Sci.*, 88, 42--52.

Cassidy, T.A. & Johnson, R.E. (2005). Monte Carlo model of sputtering and other ejection processes within a regolith. *Icarus*, 176, 499--507.

Cooper, J.F. et al. (2001). Energetic ion and electron irradiation of the icy Galilean satellites. *Icarus*, 149, 133--159.

Costello, E.S., Phillips, C.B., Lucey, P.G. & Ghent, R.R. (2021). Impact gardening on Europa and repercussions for possible biosignatures. *Nature Astronomy*, 5, 951--956. doi:10.1038/s41550-021-01393-1

Johnson, R.E. et al. (2004). Radiation effects on the surfaces of the Galilean satellites. In *Jupiter: The Planet, Satellites and Magnetosphere*, Ch. 20, 485--512. Cambridge University Press.

Johnson, R.E. et al. (2019). Sputtering of the Galilean satellites (and the Jovian magnetosphere). *Space Sci. Rev.*

Plainaki, C. et al. (2010). Neutral particle release from Europa's surface. *Icarus*, 210, 385--395.

Schorghofer, N. (2008). The lifetime of ice on main belt asteroids. *Astrophys. J.*, 682, 697--705.

Schorghofer, N. (2015). Two-dimensional description of surface-bounded exospheres with application to the migration of water molecules on the Moon. *Phys. Rev. E*, 91, 052154. doi:10.1103/PhysRevE.91.052154

Spencer, J.R. (1987). Thermal segregation of water ice on the Galilean satellites. *Icarus*, 69(2), 297--313. doi:10.1016/0019-1035(87)90107-2

Teolis, B.D. et al. (2017). Plume and surface feature structure and compositional effects on Europa's global exosphere: Preliminary Europa flyby science. *Icarus*, 284, 18--29. doi:10.1016/j.icarus.2016.10.027

Vorburger, A. & Wurz, P. (2018). Europa's ice-related atmosphere: The sputter contribution. *Icarus*, 311, 135--145. doi:10.1016/j.icarus.2018.03.022
