# Lateral Ice Transport from Vapor Pressure Gradients

Europa's sub-jovian hemisphere is periodically shadowed by Jupiter, reducing the time-averaged solar flux by up to 9.8% at the sub-jovian point.  The resulting temperature asymmetry—only 2.3 K at the equator—is amplified by the exponential dependence of H$_2$O vapor pressure on temperature into a factor $\sim 3.5$ variation in sublimation rate.  Superimposed on this longitudinal contrast, the equator-to-pole insolation gradient produces a 47 K temperature decline that collapses vapor pressures by over four orders of magnitude.  Together, these gradients drive lateral ballistic transport of water ice that preferentially removes ice from the equatorial flanks and low latitudes.

This document develops an analytical diffusive framework for this transport, calibrated to orbit-averaged temperatures from the PTM3D thermal model (327,680-face icosahedral mesh), and derives the resulting ice redistribution rates and equilibrium lag deposit concentrations.

## Temperature and Vapor Pressure Fields

The equilibrium H$_2$O vapor pressure over ice (Marti & Mauersberger 1993),

$$\log_{10} P_v\thickspace[\mathrm{Pa}] = -2663.5/T + 12.537$$

amplifies temperature contrasts exponentially.  From the PTM3D orbit-averaged temperatures, the equatorial vapor pressure profile as a function of longitude is:

| $\lambda$ | $T$ (K) | $P_v / \bar{P}$ |
|-----------|---------|-----------------|
| $0°$ (sub-jovian) | 105.1 | 0.39 |
| $\pm 45°$ | 105.6 | 0.51 |
| $\pm 90°$ (flanks) | 107.2 | 1.26 |
| $\pm 135°$ | 107.4 | 1.36 |
| $180°$ (anti-jovian) | 107.4 | 1.37 |

where $\bar{P} = 3.96 \times 10^{-13}$ Pa is the equatorial mean.  The profile is strongly asymmetric: $P_v$ is nearly constant across the anti-jovian hemisphere ($|\lambda| > 90°$) and drops steeply through a transition zone centered at $|\lambda| \approx 90°$.

The latitudinal profile is far steeper.  Averaging over longitude, the temperature drops from $T_\mathrm{eq} = 106.7$ K at the equator to $\sim 60$ K at the poles, collapsing vapor pressures by over four orders of magnitude:

| Latitude $\varphi$ | $T$ (K) | $P_v / P_\mathrm{eq}$ | $\log_{10}(P_v/P_\mathrm{eq})$ |
|---------------------|---------|------------------------|-------------------------------|
| $0°$ (equator) | 106.7 | $1.0$ | $0.0$ |
| $\pm 10°$ | 106.2 | $0.80$ | $-0.1$ |
| $\pm 20°$ | 105.1 | $0.43$ | $-0.4$ |
| $\pm 30°$ | 103.1 | $0.14$ | $-0.9$ |
| $\pm 45°$ | 98.4 | $8.2 \times 10^{-3}$ | $-2.1$ |
| $\pm 60°$ | 90.7 | $4.5 \times 10^{-5}$ | $-4.3$ |

Both profiles serve as inputs to the diffusive transport model developed below.

## Diffusive Limit of Ballistic Transport

Jeans escape is negligible at Europa ($\lambda_J \approx 42$; $f_\mathrm{esc} \sim 10^{-18}$), so all sublimated H$_2$O returns to the surface.  Let $K(\Delta\lambda)$ be the ballistic transport kernel—the probability that a molecule sublimated at $\lambda'$ lands at $\lambda = \lambda' + \Delta\lambda$.  The deposition flux is the convolution

$$\Phi_\mathrm{dep}(\lambda) = \int K(\lambda - \lambda')\thinspace \Phi_\mathrm{sub}(\lambda')\thinspace d\lambda'$$

Taylor-expanding $\Phi_\mathrm{sub}(\lambda')$ about $\lambda$, the zeroth-order term recovers $\Phi_\mathrm{sub}$ by normalization of $K$, the first-order term vanishes by symmetry, and the leading correction is

$$\Phi_\mathrm{net} = \Phi_\mathrm{sub} - \Phi_\mathrm{dep} \approx -\frac{\ell^2}{2}\thinspace\frac{d^2\Phi_\mathrm{sub}}{d\lambda^2}$$

where $\ell^2 \equiv \int (\Delta\lambda)^2\thinspace K(\Delta\lambda)\thinspace d(\Delta\lambda)$ is the mean-square angular hop distance.  Since the fractional temperature variation is small ($\Delta T / \bar{T} \approx 2$%), the mass sublimation flux $\Phi_\mathrm{sub} = P_v \sqrt{m/(2\pi k_B T)}$ is proportional to $P_v$ to excellent approximation, giving

$$\boxed{\Phi_\mathrm{net}(\lambda) \thickspace\approx\thickspace -D_b\thinspace\frac{d^2 P_v}{d\lambda^2} \thickspace\times\thickspace \sqrt{\frac{m}{2\pi k_B \bar{T}}}}$$

where $D_b = \ell^2/2$ is the effective ballistic diffusivity.  For the $K_0$ transport kernel, the analytical second moment is $\langle\xi^2\rangle = 4$ (with $\xi = g d / 2\sigma^2$ the dimensionless hop parameter, $g = 1.314$ m s$^{-2}$, $\sigma^2 = k_BT/m$), giving a characteristic hop distance $\ell \approx 5.5°$ at 107 K and $D_b \approx 4.6 \times 10^{-3}$ rad$^2$.  Since $\ell$ is much shorter than the $\sim 60°$ longitudinal transition zone, the diffusive approximation is well justified for equatorial transport ($\ell/H_\lambda \approx 0.08$).

## Equatorial (Longitudinal) Transport

Decomposing $P_v(\lambda)$ into a cosine series $P_v = \sum a_n \cos n\lambda$ converts the Laplacian to the algebraic weighting $d^2P_v/d\lambda^2 = -\sum n^2 a_n \cos n\lambda$.  The leading Fourier coefficients from the PTM3D temperature field are:

| $n$ | $a_n / \bar{P}$ | $n^2 \times a_n / \bar{P}$ |
|-----|-----------------|--------------------------|
| 1 | $-0.553$ | $-0.553$ |
| 2 | $-0.160$ | $-0.640$ |
| 4 | $+0.065$ | $+1.048$ |
| 6 | $-0.032$ | $-1.168$ |

The $n^2$ weighting amplifies the higher harmonics encoding the steep transition at $\pm 90°$.  At the flanks, only even harmonics contribute ($\cos n\pi/2 = 0$ for odd $n$), and their signs combine constructively to produce a large negative curvature.  Evaluating the net flux at key longitudes, with the mean equatorial sublimation flux $\bar{\Phi} = \bar{P}\sqrt{m/(2\pi k_B \bar{T})} = 7.1 \times 10^{-16}$ kg m$^{-2}$ s$^{-1}$:

| $\lambda$ | $\Phi_\mathrm{net} / \bar{\Phi}$ | Ice rate ($\mu$m Myr$^{-1}$) | Interpretation |
|-----------|-------------------------------|-------------------------------|----------------|
| $0°$ (sub-jovian) | $-0.5$% | $-0.13$ (accretion) | Cold trap |
| $\pm 90°$ (flanks) | $\mathbf{+1.3}$**%** | $\mathbf{+0.32}$ (removal) | **Net ice loss** |
| $135°$ | $-0.3$% | $-0.07$ (accretion) | Slight deposition |
| $180°$ (anti-jovian) | $-0.2$% | $-0.05$ (accretion) | Near equilibrium |

The flanks at $\pm 90°$ are the only equatorial longitudes with net ice removal.  Despite having nearly the same $P_v$ as the anti-jovian hemisphere, the flanks sit at the inflection point of the vapor pressure profile: molecules hopping anti-jovianward are replaced by equivalent return flux, while those hopping sub-jovianward land on colder terrain that returns far less.

## Latitudinal Transport

The equatorial analysis above treats longitudinal transport in isolation.  In the latitudinal direction, the much steeper vapor pressure gradient produces correspondingly larger transport.  The vapor pressure curvature at the equator is $d^2P_v/d\varphi^2\big|_{\varphi=0} = -5.5 \times 10^{-12}$ Pa rad$^{-2}$—a factor 4.9 larger in magnitude than the longitudinal curvature at the flanks:

| Direction | Location | $|d^2P_v/d\mathrm{coord}^2|$ (Pa rad$^{-2}$) | Ice rate ($\mu$m Myr$^{-1}$) |
|-----------|----------|-----------------------------------------------|-------------------------------|
| Latitudinal | Equator | $5.5 \times 10^{-12}$ | $+1.56$ (removal) |
| Longitudinal | $\pm 90°$ flanks | $1.13 \times 10^{-12}$ | $+0.32$ (removal) |

These rates describe transport along a **single axis only**: the latitudinal rate measures net poleward flux at the equator (a longitude-averaged quantity), while the longitudinal rate measures net zonal flux at the flanks.  The latitudinal rate appears larger because the equator-to-pole temperature gradient (47 K) is far steeper than the eclipse-driven longitudinal variation (2.3 K), and the two components should not be read as comparable estimates of the same quantity.

The two directions play complementary roles.  Latitudinal transport is **zonally symmetric**—it depletes equatorial ice at all longitudes equally, driving poleward migration.  Longitudinal transport **breaks this symmetry**, preferentially removing ice from the $\pm 90°$ flanks.  Both operate simultaneously, but the latitudinal component dominates by a factor $\sim 5$.

The diffusion approximation is less well justified in the latitudinal direction: the vapor pressure scale height $H_\varphi \approx 16°$ at $\varphi = 15°$ gives $\ell/H_\varphi \approx 0.34$, versus $\ell/H_\lambda \approx 0.08$ longitudinally.  Higher-order Taylor corrections contribute $\sim 10$%.  The spherical harmonic treatment below provides the proper two-dimensional generalization.

## Spherical Harmonic Generalization

On the full sphere, the transport kernel becomes a convolution over solid angle, and the same Taylor expansion yields the angular Laplacian $\nabla^2_\Omega$ in place of the 1D second derivative:

$$\boxed{\Phi_\mathrm{net}(\theta,\phi)  =  -D_b\thinspace\nabla^2_\Omega P_v \thickspace\times\thickspace \sqrt{\frac{m}{2\pi k_B \bar{T}}}}$$

Expanding $P_v$ in spherical harmonics, $P_v = \sum c_{\ell m}\thinspace Y_\ell^m$, the eigenvalue relation $\nabla^2_\Omega Y_\ell^m = -\ell(\ell+1)\thinspaceY_\ell^m$ gives

$$\Phi_\mathrm{net} = D_b \sum_{\ell,m} \ell(\ell+1)\thinspace c_{\ell m}\thinspace Y_\ell^m \thickspace\times\thickspace \sqrt{\frac{m}{2\pi k_B \bar{T}}}$$

This is the natural generalization: the 1D Fourier weighting $n^2$ becomes the spherical harmonic eigenvalue $\ell(\ell+1)$.  The equatorial Fourier modes $a_n$ correspond to sectoral harmonics ($\ell = n$, $m = n$), while the latitudinal structure maps onto zonal harmonics ($m = 0$).

### Spherical harmonic coefficients

Projecting the PTM3D vapor pressure field onto spherical harmonics up to $\ell_\mathrm{max} = 20$, the leading coefficients (normalized to $c_{00}$) are:

| $\ell$ | $m$ | $c_{\ell m}/c_{00}$ | $\ell(\ell+1) \times c_{\ell m}/c_{00}$ | Interpretation |
|--------|-----|---------------------|----------------------------------------|----------------|
| 2 | 0 | $-0.914$ | $-5.49$ | Equator-to-pole gradient |
| 4 | 0 | $+0.570$ | $+11.40$ | Latitudinal structure |
| 6 | 0 | $-0.261$ | $-10.94$ | Latitudinal structure |
| 8 | 0 | $+0.081$ | $+5.85$ | Latitudinal structure |
| 1 | 1 | $+0.328$ | $+0.66$ | Eclipse dipole |
| 3 | 1 | $-0.216$ | $-2.59$ | Eclipse higher harmonic |
| 2 | 2 | $-0.113$ | $-0.68$ | Eclipse sectoral mode |

The zonal ($m = 0$) modes dominate the spectrum, encoding the latitudinal gradient; the $m \neq 0$ modes carry the eclipse-driven longitudinal asymmetry at comparatively small amplitude.

### Transport power spectrum

The transport power per degree, $S_\ell = \ell(\ell+1)\sum_m |c_{\ell m}|^2$, measures the contribution of each angular scale to net ice redistribution:

| $\ell$ | $S_\ell / S_\mathrm{total}$ | Cumulative |
|--------|----------------------------|------------|
| 2 | 27% | 27% |
| 4 | **35%** | 62% |
| 6 | 16% | 78% |
| 3 | 6% | 84% |
| 5 | 5% | 89% |

The $\ell = 4$ mode dominates because the $\ell(\ell+1)$ weighting amplifies intermediate-scale structure—the steep equator-to-midlatitude $P_v$ transition projects most strongly onto this degree.  Zonal ($m = 0$) modes account for $\sim 85$% of total transport power, confirming that poleward transport dominates.

### Combined 2D ice rates

Evaluating the full 2D $\Phi_\mathrm{net}$ at the equator with $\ell_\mathrm{max} = 5$:

| Equatorial longitude | Ice rate ($\mu$m Myr$^{-1}$) | Relative to flanks |
|----------------------|-------------------------------|---------------------|
| $0°$ (sub-jovian) | $+0.11$ | $0.11\times$ |
| $\pm 90°$ (flanks) | $\mathbf{+0.96}$ | $1.00\times$ |
| $180°$ (anti-jovian) | $+1.05$ | $1.09\times$ |

In contrast to the 1D longitudinal analysis, the 2D model predicts net ice removal at *all* equatorial longitudes—poleward transport dominates everywhere.  The eclipse asymmetry modulates the rate by nearly an order of magnitude between the sub-jovian point and the flanks, but even the sub-jovian cold trap cannot fully compensate for the latitudinal outflow.

## Ice Evolution and Steady State

Since $\Phi_\mathrm{net}$ depends on temperature alone (not on the ice column), ice thickness at any location evolves linearly: $\sigma(t) = \sigma_0 - \Phi_\mathrm{net}\thinspace t$.  Ice is depleted first at the equatorial flanks, with a depletion timescale

$$t^*(\pm 90°) = \frac{\rho_\mathrm{ice}\thinspace h_0}{\Phi_\mathrm{net}(\pm 90°)} \approx 3.1\thickspace\mathrm{Myr} \times \left(\frac{h_0}{1\thickspace\mu\mathrm{m}}\right)$$

However, as ice sublimes, refractory material (silicates, salts) accumulates as a lag deposit.  The growing non-ice concentration $c$ reduces the exposed ice fraction to $(1-c)$, throttling further sublimation.  For a well-mixed surface layer of gardening depth $\delta$, the concentration evolves as

$$\rho_\mathrm{ice}\thinspace\delta\thinspace\frac{dc}{dt} = (1-c)\thinspace\Phi_\mathrm{sub}(\lambda) - \Phi_\mathrm{dep}(\lambda)$$

In the diffusive model $\Phi_\mathrm{dep} \approx \Phi_\mathrm{sub} - \Phi_\mathrm{net}$, yielding the analytical solution

$$\boxed{c(\lambda, t) = c_\mathrm{eq}(\lambda)\left(1 - e^{-t/\tau(\lambda)}\right)}$$

where $c_\mathrm{eq} = \Phi_\mathrm{net}/\Phi_\mathrm{sub}$ is the equilibrium concentration at which the lag deposit balances net transport, and $\tau = \rho_\mathrm{ice}\thinspace\delta/\Phi_\mathrm{sub}$ is the sublimation turnover time.  Evaluating at $\pm 90°$ ($\Phi_\mathrm{sub} \approx 9.0 \times 10^{-16}$ kg m$^{-2}$ s$^{-1}$):

| Gardening depth $\delta$ | Turnover time $\tau$ | $c_\mathrm{eq}$ |
|--------------------------|---------------------|------------------|
| 1 $\mu$m | 32 kyr | 1.0% |
| 100 $\mu$m (thermal skin) | 3.2 Myr | 1.0% |
| 1 cm | 320 Myr | 1.0% |

The equilibrium concentration $c_\mathrm{eq} \approx 1$% is independent of gardening depth—set purely by the ratio of net transport to local sublimation.  The gardening depth controls only how quickly equilibrium is reached: $\tau \propto \delta$.

This $\sim 1$% enrichment results from lateral diffusive transport alone, without albedo feedback.  In the full numerical model, the positive feedback loop (higher $c$ → lower albedo → higher $T$ → exponentially faster sublimation) amplifies $c_\mathrm{eq}$ well beyond the diffusive prediction, with the amplification depending sensitively on the albedo endmembers and gardening depth.
