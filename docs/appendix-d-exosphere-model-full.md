# Appendix D: Exosphere Model

## D.1 Introduction and Overview

Europa's surface evolves through the interplay of thermal processing, magnetospheric bombardment, and volatile transport. Water ice, the dominant surface constituent, sublimates from warm regions, travels ballistically through Europa's collisionless exosphere, and redeposits on colder surfaces. Concurrently, magnetospheric plasma sputtering ejects molecules from the surface at rates that can exceed sublimation at low temperatures. These processes redistribute mass across the surface, drive the formation of non-ice lag deposits, and control the crystalline state of the ice.

Previous Europa exosphere models have addressed individual aspects of this system. Plainaki et al. (2012, 2013) developed a Monte Carlo model of Europa's exosphere driven by sputtering and sublimation, focusing on atmospheric column densities and escape rates. Vorburger & Wurz (2018) extended this approach with improved sputtering yield models and source rate estimates. Teolis et al. (2017) modeled the exosphere structure with emphasis on plume contributions and surface composition effects. Oza et al. (2019) constrained Europa's atmospheric properties from HST observations.

The PTM3D exosphere module advances beyond these models in two key respects. First, it is directly coupled to a 3D thermophysical model that provides spatially and temporally resolved surface temperatures on an unstructured triangular mesh, rather than relying on parameterized or zonally averaged temperature distributions. This captures the full diurnal cycle, eclipse effects, topographic shadowing, and albedo variations that control sublimation rates. Second, the model tracks the long-term evolution of ice layer composition through a multi-timescale operator-splitting scheme, enabling simulation of geological timescales ($\sim 10^3$ years) where lag deposit formation and thermal segregation operate.

The exosphere module computes seven coupled processes:

1. **Sublimation**: Temperature-dependent vapor pressure and Hertz-Knudsen emission flux (Section D.2)
2. **Ballistic transport**: Redistribution of sublimated molecules via collisionless trajectories (Section D.3)
3. **Jeans escape**: Thermal loss of high-velocity molecules to space (Section D.4)
4. **Magnetospheric sputtering**: Ejection of surface molecules by energetic ion bombardment (Section D.5)
5. **Ice layer evolution**: Tracking of non-ice concentration within the thermally active skin depth (Section D.6)
6. **Crystallization/amorphization**: Competition between thermal recrystallization and radiation-induced amorphization (Section D.7)
7. **Multi-timescale extrapolation**: Operator splitting to bridge diurnal and geological timescales (Section D.8)

The coupling is currently one-directional: the thermal model provides surface temperatures $T(i, t)$ for each mesh face $i$ at each timestep $t$, and the exosphere model computes the resulting mass fluxes and compositional evolution. The exosphere does not feed back into the thermal model (i.e., albedo changes from ice redistribution do not yet modify the thermal solution). This feedback is planned for future work.

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

Sputtered H$_2$O molecules are assumed to follow the same ballistic transport as sublimated molecules. Of the sputtered products, H$_2$O constitutes approximately 85% of the mass flux; the remainder is O$_2$ ($\sim 10$%) and H$_2$ ($\sim 5$%), with escape fractions of approximately 90% and 95%, respectively (these lighter species have much lower Jeans parameters and escape efficiently).

Key assumptions: (1) Steady-state magnetospheric conditions (no time variation from Jovian magnetotail passages). (2) Smooth cosine variation of the ion flux, simplified from full particle-tracing simulations. (3) Sputtered H$_2$O molecules have the same velocity distribution as sublimated molecules.

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

This produces a self-limiting feedback: as the lag deposit thickens, it armors the underlying ice against further sublimation (in reality, by reducing the ice surface area exposed to vacuum). In our simplified model, sublimation ceases entirely when $c = 1$. More sophisticated treatments incorporating grain-size effects on sublimation rate through a porous lag are left for future work.

Key assumptions: (1) The layer is well-mixed (uniform concentration throughout the skin depth). (2) Vertical mixing is instantaneous. (3) The lag deposit does not alter the thermal conductivity or albedo (one-way coupling). (4) No grain-size dependence of sublimation rate.

---

## D.7 Crystallization and Amorphization

Water ice on Europa exists in a competition between thermally driven crystallization and radiation-induced amorphization. The model tracks the crystalline fraction $\phi_C$ (dimensionless, $0 \le \phi_C \le 1$) at each mesh face, evolving under these two competing processes.

### D.7.1 Thermal Crystallization

The crystallization rate follows Avrami kinetics as parameterized by Mitchell et al. (2017) for water ice:

$$k_{\mathrm{cryst}}^{1/n} = A_0^{1/n} \thinspace \exp\negthinspace\left(\frac{-\Delta H_a}{R \thinspace T}\right) \tag{D26}$$

$$k_{\mathrm{eff}} = n \thickspace k_{\mathrm{cryst}}^{1/n} \tag{D27}$$

where $A_0^{1/n} = 3.9 \times 10^{19}$ s$^{-1}$ is the pre-exponential factor (Mitchell et al. 2017, Table 2, Stage 2), $\Delta H_a = 62$ kJ/mol is the activation energy, $R = 8.314$ J/(mol$\cdot$K) is the gas constant, and $n$ is the Avrami exponent.

The Avrami exponent depends on ice porosity, reflecting the dimensionality of crystal growth (Mitchell et al. 2017):

$$n(\varphi) = n_{\mathrm{low}} + (n_{\mathrm{high}} - n_{\mathrm{low}}) \thinspace \frac{\varphi}{\varphi_{\max}} \tag{D28}$$

where $n_{\mathrm{low}} = 4.0$ (compact ice: bulk nucleation with 3D growth), $n_{\mathrm{high}} = 1.5$ (porous ice: predominantly 1D growth along pore walls), and $\varphi_{\max} = 0.7$. We adopt latitude-dependent porosities of $\varphi_{\mathrm{eq}} = 0.50$ at the equator and $\varphi_{\mathrm{pol}} = 0.20$ at the poles, with linear interpolation, broadly consistent with porosity estimates derived from millimeter-wavelength observations (Thelen et al. 2024).

The Arrhenius temperature dependence produces an extremely steep variation in crystallization rate (Table D6).

**Table D6.** Crystallization timescale at representative temperatures (compact ice, $n = 4$).

| $T$ (K) | $k_{\mathrm{eff}}$ (s$^{-1}$) | Timescale |
|---------|-------------------------------|-----------|
| 80 | $5.1 \times 10^{-21}$ | $> 10^{12}$ yr |
| 100 | $6.4 \times 10^{-13}$ | $\sim 50{,}000$ yr |
| 110 | $5.6 \times 10^{-10}$ | $\sim 56$ yr |
| 120 | $1.6 \times 10^{-7}$ | $\sim 72$ days |
| 130 | $1.9 \times 10^{-5}$ | $\sim 15$ hr |
| 140 | $1.1 \times 10^{-3}$ | $\sim 15$ min |

### D.7.2 Radiation-Induced Amorphization

Magnetospheric ion bombardment deposits energy into the ice lattice, disrupting crystalline order. The amorphization rate is:

$$k_{\mathrm{amorph}} = \frac{\dot{D}}{E_{\mathrm{amorph}}} \tag{D29}$$

where $\dot{D}$ is the local radiation energy deposition rate [eV/(molecule$\cdot$s)] and $E_{\mathrm{amorph}} = 3.0$ eV/molecule is the energy required to amorphize one ice molecule (Famá et al. 2010; Leto & Baratta 2003).

The radiation dose rate varies spatially following the energetic ion flux patterns from Nordheim et al. (2022):

$$\dot{D}(\phi_{\mathrm{lat}}, \phi_{\mathrm{lon}}) = \dot{D}_0 \thickspace f_{\mathrm{lon}}^{(\mathrm{dose})} \thickspace f_{\mathrm{lat}}^{(\mathrm{dose})} \tag{D30}$$

**Longitude factor** (S$^+$ ion flux):

$$f_{\mathrm{lon}}^{(\mathrm{dose})} = \frac{1}{C_D} + \left(1 - \frac{1}{C_D}\right) \max\negthinspace\left(0,\thickspace \frac{\cos(\phi_{\mathrm{lon}} - 270°) + 1}{2}\right) \tag{D31}$$

**Latitude factor** (O$^+$ and S$^+$ at $\sim 1$ keV):

$$f_{\mathrm{lat}}^{(\mathrm{dose})} = 1 + (E_D - 1) \left(\frac{|\phi_{\mathrm{lat}}|}{90°}\right)^{1.5} \tag{D32}$$

where $C_D = 5.0$ is the trailing/leading hemisphere dose contrast and $E_D = 2.5$ is the polar dose enhancement factor. The base dose rate $\dot{D}_0 = 2.3 \times 10^{-6}$ eV/(molecule$\cdot$s) is calibrated to produce an amorphization timescale of $\sim 15$ days at the trailing equator, consistent with JWST observations (Cartwright et al. 2025).

We note that the radiation dose distribution uses different spatial parameters than the sputtering model (Section D.5), despite both being driven by magnetospheric ion bombardment. Sputtering is dominated by thermal plasma ($\sim 100$ eV S$^+$ and O$^+$) with moderate spatial contrast ($C_{\mathrm{hem}} = 3.0$). Amorphization is driven by higher-energy ions ($\sim 1$ keV) that experience stronger magnetic focusing, producing a steeper trailing/leading contrast ($C_D = 5.0$), stronger polar enhancement ($E_D = 2.5$), and a super-linear latitude dependence (exponent 1.5 versus linear). These differences follow Nordheim et al. (2022), who showed that the energetic ion flux map differs substantially from the thermal plasma distribution.

### D.7.3 Equilibrium Crystallinity and the Transition Temperature

The crystalline fraction evolves according to the competition ODE:

$$\frac{d\phi_C}{dt} = k_{\mathrm{cryst}}(T) \thinspace (1 - \phi_C) - k_{\mathrm{amorph}} \thinspace \phi_C \tag{D33}$$

This linear ODE has the analytical solution:

$$\phi_C(t) = \phi_{\mathrm{eq}} + \bigl[\phi_C(0) - \phi_{\mathrm{eq}}\bigr] \thinspace \exp\negthinspace\bigl[-(k_{\mathrm{cryst}} + k_{\mathrm{amorph}}) \thinspace t\bigr] \tag{D34}$$

where the equilibrium crystalline fraction is:

$$\phi_{\mathrm{eq}} = \frac{k_{\mathrm{cryst}}}{k_{\mathrm{cryst}} + k_{\mathrm{amorph}}} \tag{D35}$$

The temperature dependence of the equilibrium crystallinity has a characteristic sigmoid shape, arising from the balance between the Arrhenius-activated crystallization rate (exponential in $1/T$) and the temperature-independent radiation amorphization rate. We can derive the transition temperature $T^*$ where $\phi_{\mathrm{eq}} = 0.5$, which requires $k_{\mathrm{cryst}}(T^*) = k_{\mathrm{amorph}}$. From Equations D27 and D29:

$$n \thinspace A_0^{1/n} \thinspace \exp\negthinspace\left(\frac{-\Delta H_a}{R \thinspace T^*}\right) = \frac{\dot{D}}{E_{\mathrm{amorph}}} \tag{D36}$$

Solving for $T^*$:

$$T^* = \frac{\Delta H_a}{R \thinspace \ln\negthinspace\left(\frac{n \thinspace A_0^{1/n} \thinspace E_{\mathrm{amorph}}}{\dot{D}}\right)} \tag{D37}$$

For compact ice ($n = 4$) at the trailing equator ($\dot{D} = \dot{D}_0 = 2.3 \times 10^{-6}$ eV/(molecule$\cdot$s)):

$$T^* = \frac{62{,}000}{8.314 \times \ln\negthinspace\left(\frac{4 \times 3.9 \times 10^{19} \times 3.0}{2.3 \times 10^{-6}}\right)} = \frac{62{,}000}{8.314 \times 59.6} \approx 125 \text{ K} \tag{D38}$$

This transition temperature falls squarely within the range of Europa's diurnal surface temperature variations ($86$--$132$ K), explaining why both crystalline and amorphous ice are observed on the surface. The transition is sharp because the crystallization rate changes by orders of magnitude over a few kelvin (Table D6), while the amorphization rate is essentially constant. At the leading hemisphere, the dose rate is $\sim 5\times$ lower, shifting $T^*$ down by $\sim 5$ K, which is consistent with JWST observations showing crystalline ice at warm, leading-hemisphere locations (Cartwright et al. 2025).

**Table D7.** Equilibrium crystallinity at the trailing equator ($k_{\mathrm{amorph}} = 7.67 \times 10^{-7}$ s$^{-1}$).

| $T$ (K) | $k_{\mathrm{cryst}}$ (s$^{-1}$) | $\phi_{\mathrm{eq}}$ |
|---------|--------------------------------|----------------------|
| 80 | $5.1 \times 10^{-21}$ | $\sim 0$ |
| 100 | $6.4 \times 10^{-13}$ | $\sim 10^{-6}$ |
| 110 | $5.6 \times 10^{-10}$ | 0.0007 |
| 120 | $1.6 \times 10^{-7}$ | 0.17 |
| 130 | $1.9 \times 10^{-5}$ | 0.96 |

### D.7.4 Depositional Amorphization

Freshly deposited ice at Europa's surface temperatures ($< 150$ K) is amorphous. When new ice of mass $\Delta m_{\mathrm{dep}}$ is deposited on a face with existing ice mass $m_{\mathrm{old}}$, the crystalline fraction is diluted by mass-weighted mixing:

$$\phi_{C,\mathrm{new}} = \phi_{C,\mathrm{old}} \thickspace \frac{m_{\mathrm{old}}}{m_{\mathrm{old}} + \Delta m_{\mathrm{dep}}} \tag{D39}$$

This provides a third amorphization pathway beyond radiation, particularly important in polar regions where deposition rates are high.

Key assumptions: (1) Spatially uniform porosity at a given latitude. (2) A single crystallization mechanism (activation energy and pre-exponential are constant). (3) Radiation dose rate is steady-state (no magnetospheric variability). (4) Dose is deposited uniformly within the skin depth layer.

---

## D.8 Multi-Timescale Evolution

### D.8.1 Motivation

The diurnal cycle of sublimation and deposition has a period $P_{\mathrm{diurnal}} = 3.5512$ days ($306{,}824$ s), while the processes of ice layer evolution, lag deposit formation, and thermal segregation operate over geological timescales of $10^2$--$10^3$ years. Direct simulation of every diurnal timestep over these durations is computationally prohibitive ($> 10^7$ diurnal cycles for 1000 years). We employ operator splitting to bridge these timescales.

### D.8.2 Algorithm

The evolution proceeds in three steps:

**Step 1: Diurnal averaging.** Run one complete diurnal cycle ($N_t$ timesteps) with the current surface state. Compute time-averaged fluxes at each face:

$$\bar{\Phi}_{\mathrm{sub},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{sub},i}(t), \quad \bar{\Phi}_{\mathrm{dep},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{dep},i}(t), \quad \bar{\Phi}_{\mathrm{sput},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{sput},i}(t) \tag{D40}$$

**Step 2: Epoch extrapolation.** Apply the time-averaged fluxes over one epoch of $N_c$ diurnal cycles:

$$\Delta t_{\mathrm{epoch}} = N_c \thickspace P_{\mathrm{diurnal}} \tag{D41}$$

$$\Delta c_i = \frac{\bar{\Phi}_{\mathrm{sub},i} + \bar{\Phi}_{\mathrm{sput},i} - \bar{\Phi}_{\mathrm{dep},i}}{\delta_{\mathrm{th}} \thickspace \rho_{\mathrm{ice}}} \thickspace \Delta t_{\mathrm{epoch}} \tag{D42}$$

The crystalline fraction is also evolved analytically over the epoch using Equation D34.

**Step 3: Iterate.** Repeat Steps 1--2 for $N_e$ epochs.

### D.8.3 Flux Scaling for Depleted Regions

As ice is progressively removed from warm faces ($c \to 1$), the sublimation flux from those faces decreases. To maintain global mass conservation, the deposition flux is rescaled:

$$\bar{\Phi}_{\mathrm{dep}}^{(\mathrm{scaled})} = \bar{\Phi}_{\mathrm{dep}}^{(\mathrm{base})} \thickspace \frac{\sum_i \Phi_{\mathrm{em},i}^{(\mathrm{current})}}{\sum_i \Phi_{\mathrm{em},i}^{(\mathrm{base})}} \tag{D43}$$

where $\Phi_{\mathrm{em},i}^{(\mathrm{base})}$ is the emission flux computed with the initial (undepleted) state and $\Phi_{\mathrm{em},i}^{(\mathrm{current})}$ reflects the current ice availability (zero emission from ice-depleted faces).

### D.8.4 Analytical Acceleration

For faces where the net mass-loss rate implies a timescale much longer than $\Delta t_{\mathrm{epoch}}$, the model uses the analytical solution of the concentration evolution ODE directly, avoiding unnecessary numerical time-stepping. This is particularly effective for cold polar regions where sublimation is negligible and evolution is dominated by the slow accumulation of sputtered debris.

### D.8.5 Default Parameters

**Table D8.** Default multi-timescale evolution parameters.

| Parameter | Default | Description |
|-----------|---------|-------------|
| $N_e$ | 1000 | Number of geological epochs |
| $N_c$ | 100 | Diurnal cycles per epoch ($\approx 1$ Earth year) |
| Export interval | 100 | Export data every $N$ epochs |
| Analytical acceleration | on | Use closed-form solutions for slow regions |
| Cache refresh | adaptive | Recompute diurnal cycle when conditions change significantly |

With default parameters: $N_c \times N_e = 10^5$ orbits $= 10^5 \times 3.5512$ d $/$ $365.25$ d/yr $= 972$ years of geological evolution.

### D.8.6 Approximation Validity

The operator-splitting approach assumes that diurnal-averaged fluxes are representative of the full diurnal cycle, and that surface temperatures do not change significantly within an epoch. The first assumption is well justified because the transport matrix, which depends on temperature, is slowly varying compared to the diurnal cycle. The second assumption requires that changes in non-ice concentration (which could alter albedo and hence temperature) be small over one epoch; this is enforced by the adaptive cache refresh, which triggers a recomputation of the diurnal cycle when the maximum concentration change exceeds a threshold.

The one-way coupling (thermal model $\to$ exosphere, but not the reverse) is the primary limitation: as ice is removed and the lag deposit thickens, the surface albedo decreases, increasing temperatures and accelerating further ice loss (thermal segregation feedback; Spencer 1987). The current model captures the mass-loss side of this feedback but does not self-consistently update the thermal solution.

---

## D.9 Model Parameters

### D.9.1 Europa Physical Properties

**Table D9.** Adopted Europa parameters.

| Symbol | Value | Description | Source |
|--------|-------|-------------|--------|
| $R_E$ | $1.5608 \times 10^6$ m | Mean radius | IAU |
| $g$ | 1.314 m/s$^2$ | Surface gravity | $GM/R_E^2$ |
| $v_{\mathrm{esc}}$ | 2025 m/s | Escape velocity | $\sqrt{2gR_E}$ |
| $A_E$ | $3.09 \times 10^{13}$ m$^2$ | Surface area | $4\pi R_E^2$ |
| $P_{\mathrm{diurnal}}$ | 3.5512 d (306,824 s) | Rotation/orbital period | Synchronous |

### D.9.2 Water Ice Properties

**Table D10.** Water ice parameters.

| Symbol | Value | Description |
|--------|-------|-------------|
| $m$ | $2.992 \times 10^{-26}$ kg | H$_2$O molecular mass |
| $M$ | 0.01802 kg/mol | H$_2$O molar mass |
| $\rho_{\mathrm{ice}}$ | 917 kg/m$^3$ | Ice density |

### D.9.3 Model-Specific Parameters

**Table D11.** Adopted model parameters with uncertainties and sensitivities.

| Parameter | Value | Reference | Sensitivity |
|-----------|-------|-----------|-------------|
| Vapor pressure coefficients | $-2663.5/T + 12.537$ | Marti & Mauersberger (1993) | Moderate: extrapolation below 170 K introduces factor $\sim 2$ uncertainty |
| Sticking coefficient $\alpha$ | 1.0 | Standard assumption | Low: reducing to 0.5 would halve sublimation flux |
| H$_2$O sputtering rate | $2.0 \times 10^{27}$ s$^{-1}$ | Cassidy et al. (2013) | Moderate: uncertain by factor $\sim 2$--$3$ |
| Hemisphere contrast $C_{\mathrm{hem}}$ | 3.0 | Cassidy et al. (2013) | Low: affects spatial distribution but not total rate |
| Yield activation energy $Q$ | 0.06 eV | Teolis et al. (2017) | Moderate: controls sputtering-sublimation crossover |
| Activation energy $\Delta H_a$ | 62 kJ/mol | Mitchell et al. (2017) | High: shifts crystallization transition temperature |
| Pre-exponential $A_0^{1/n}$ | $3.9 \times 10^{19}$ s$^{-1}$ | Mitchell et al. (2017) | High: scales crystallization rate |
| Amorphization dose $E_{\mathrm{amorph}}$ | 3.0 eV/molec | Famá et al. (2010) | High: sets amorphization timescale |
| Base dose rate $\dot{D}_0$ | $2.3 \times 10^{-6}$ eV/(molec$\cdot$s) | Calibrated to JWST | High: sets amorphization timescale ($\sim 15$ d) |
| Thermal skin depth $\delta_{\mathrm{th}}$ | 0.05 m | Thermal model | Moderate: sets ice reservoir capacity |
| Dust fraction $f_{\mathrm{dust}}$ | 0.01 | Estimated | Low: sets initial lag deposit seed |

The most sensitive parameters are those controlling the crystallization-amorphization equilibrium ($\Delta H_a$, $A_0^{1/n}$, $\dot{D}_0$, $E_{\mathrm{amorph}}$), as small changes in these shift the transition temperature $T^*$ (Equation D37). The sputtering rate is uncertain by a factor of $\sim 2$--$3$, which affects the relative importance of sputtering versus sublimation but does not qualitatively change the spatial patterns.

---

## References

Abramowitz, M. & Stegun, I.A. (1964). *Handbook of Mathematical Functions*. National Bureau of Standards.

Butler, B.J. (1997). The migration of volatiles on the surfaces of Mercury and the Moon. *J. Geophys. Res.*, 102(E8), 19283--19291.

Cartwright, R.J. et al. (2025). Crystalline Water Ice on Europa's Surface. *Planet. Sci. J.*, 6, 125. doi:10.3847/PSJ/adba5b

Cassidy, T.A. et al. (2013). Magnetospheric ion sputtering and water ice grain size at Europa. *Planet. Space Sci.*, 77, 64--73. doi:10.1016/j.pss.2012.07.008

Chamberlain, J.W. (1963). Planetary coronae and atmospheric evaporation. *Planet. Space Sci.*, 11(8), 901--960. doi:10.1016/0032-0633(63)90122-3

Famá, M. et al. (2008). Sputtering of ice by low-energy ions. *Surf. Sci.*, 602(1), 156--161. doi:10.1016/j.susc.2007.10.002

Famá, M. et al. (2010). Radiation-induced amorphization of crystalline ice. *Surf. Sci.*, 604, 1602--1607.

Leto, G. & Baratta, G.A. (2003). Ly-alpha photon induced amorphization of Ic water ice at 16 K. *Astron. Astrophys.*, 397(1), 7--13.

Marti, J. & Mauersberger, K. (1993). A survey and new measurements of ice vapor pressure at temperatures between 170 and 250 K. *Geophys. Res. Lett.*, 20(5), 363--366. doi:10.1029/93GL00105

Mitchell, E.H. et al. (2017). Porosity effects on crystallization kinetics of amorphous water ice: Implications for cold icy bodies in the outer Solar System. *Icarus*, 285, 291--299. doi:10.1016/j.icarus.2016.11.004

Murphy, D.M. & Koop, T. (2005). Review of the vapour pressures of ice and supercooled water for atmospheric applications. *Q. J. R. Meteorol. Soc.*, 131(608), 1539--1565. doi:10.1256/qj.04.94

Nordheim, T.A. et al. (2022). Magnetospheric Ion Bombardment of Europa's Surface. *Planet. Sci. J.*, 3, 5. doi:10.3847/PSJ/ac382a

Oza, A.V. et al. (2019). Sodium and Potassium Atmospheres of the Icy Moons. *Astrophys. J.*, 885(2), 168.

Plainaki, C. et al. (2012). The role of sputtering and radiolysis in the generation of the Europa exosphere. *Icarus*, 218(2), 956--966.

Plainaki, C. et al. (2013). Exospheric O$_2$ densities at Europa during different orbital phases. *Planet. Space Sci.*, 88, 42--52.

Schorghofer, N. (2015). Two-dimensional description of surface-bounded exospheres with application to the migration of water molecules on the Moon. *Phys. Rev. E*, 91, 052154. doi:10.1103/PhysRevE.91.052154

Spencer, J.R. (1987). Thermal segregation of water ice on the Galilean satellites. *Icarus*, 69(2), 297--313. doi:10.1016/0019-1035(87)90107-2

Teolis, B.D. et al. (2017). Plume and surface feature structure and compositional effects on Europa's global exosphere: Preliminary Europa flyby science. *Icarus*, 284, 18--29. doi:10.1016/j.icarus.2016.10.027

Thelen, A.E. et al. (2024). Subsurface Thermophysical Properties of Europa's Leading and Trailing Hemispheres as Revealed by ALMA. *Planet. Sci. J.*, 5, 42. doi:10.3847/PSJ/ad251c

Vorburger, A. & Wurz, P. (2018). Europa's ice-related atmosphere: The sputter contribution. *Icarus*, 311, 135--145. doi:10.1016/j.icarus.2018.03.022
