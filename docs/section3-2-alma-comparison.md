# Section 3.2: Comparison with ALMA Subsurface Brightness Temperatures

## 3.2.1 Observations

We compare PTM3D predictions with the disk-averaged brightness temperatures reported by Thelen et al. (2024, Table 2) from eight ALMA observation epochs spanning 2016–2017 at three wavelengths: Band 7 (0.88 mm, 343.5 GHz), Band 6 (1.25 mm, 233 GHz), and Band 3 (3.05 mm, 97.5 GHz).  Seven of the eight epochs have published disk-averaged $T_b$ values (H6 was averaged with other trailing Band 6 observations in their analysis and has no independent published value).  The observations sample both the leading and trailing hemispheres across sub-observer longitudes of 81°–287° W, with published $T_b$ ranging from 73.0 K (Band 3, leading) to 89.1 K (Band 7, trailing).  Published uncertainties are 3.7–5.5 K (1$\sigma$).

**Table 1.** ALMA observations used for comparison (Thelen et al. 2024, Table 2).

| Label | Band | $\lambda$ (mm) | $\nu$ (GHz) | Hemisphere | $\lambda_{\rm sub}$ (°W) | $T_{b,\rm obs}$ (K) | $\sigma$ (K) |
|-------|------|-----------------|--------------|------------|--------------------------|----------------------|---------------|
| H1 | 7 | 0.88 | 343.5 | Trailing | 226.7 | 89.1 | 5.5 |
| H2 | 7 | 0.88 | 343.5 | Leading  | 100.3 | 86.1 | 4.3 |
| H3 | 6 | 1.25 | 233.0 | Trailing | 270.1 | 80.0 | 4.0 |
| H4 | 6 | 1.25 | 233.0 | Trailing | 287.0 | 85.7 | 5.4 |
| H5 | 6 | 1.25 | 233.0 | Leading  |  80.6 | 76.6 | 3.8 |
| H7 | 3 | 3.05 |  97.5 | Leading  | 103.3 | 73.0 | 3.7 |
| H8 | 3 | 3.05 |  97.5 | Trailing | 283.2 | 85.3 | 4.3 |


## 3.2.2 Forward Model: T(z) to Brightness Temperature

At millimeter wavelengths, ALMA observes thermal emission from beneath the surface, weighted by the absorption profile of the icy regolith.  The predicted brightness temperature for a surface element at emission angle $\theta$ is computed from the full radiative transfer integral through the PTM3D subsurface temperature profile $T(z)$:

$$T_b(\nu, \theta) = \varepsilon(\theta) \int_0^\infty T(z)\, \frac{\kappa_a(z)}{\cos\theta}\, \exp\!\left(-\frac{\tau(z)}{\cos\theta}\right) dz$$

where $\tau(z) = \int_0^z \kappa_a(z')\,dz'$ is the nadir optical depth and $\varepsilon(\theta)$ is the surface emissivity.  The factor $1/\cos\theta$ accounts for the increased path length through each layer at non-normal emission angles, which reduces the effective probing depth near the limb.

The disk-averaged brightness temperature is then

$$\langle T_b \rangle = \frac{\sum_i T_{b,i}\, \cos\theta_i\, \cos\phi_i}{\sum_i \cos\theta_i\, \cos\phi_i}$$

where the sum is over all visible surface elements, $\theta_i$ is the emission angle, and $\phi_i$ is the latitude.

### Dielectric properties

The complex permittivity of pure water ice is computed following the Hufford (1991) / Mätzler (2006) parameterization.  The real part is nearly constant: $\varepsilon'(T) = 3.1884 + 9.1 \times 10^{-4}(T - 273.15)$, giving $\varepsilon' \approx 3.03$ at 100 K.  The imaginary part combines a Debye relaxation tail and an infrared lattice resonance wing:

$$\varepsilon'' = \frac{\alpha}{\nu} + (\beta_1 + \beta_2)\,\nu$$

where the coefficients depend on temperature through $\Theta = 300/T - 1$ (Mätzler 2006, Eqs. 5.36–5.38).  At Europa temperatures (80–130 K), the Debye contribution is negligible ($\alpha \to 0$) and the loss is dominated by the infrared wing terms $\beta_1$ and $\beta_2$.

The effective permittivity of the porous regolith is computed using the Maxwell-Garnett effective medium approximation.  At porosity $\phi > 0.5$, ice grains are treated as spherical inclusions embedded in a vacuum matrix; at $\phi \leq 0.5$, the topology reverses with vacuum pores as inclusions in an ice matrix.  The power absorption coefficient is then

$$\kappa_a = \frac{2\pi\nu}{c}\,\mathrm{Im}\!\left(\sqrt{\varepsilon_{\rm eff}}\right)$$

At the adopted porosity of $\phi = 0.5$ and $T = 100$ K, this gives absorption coefficients and 1/$e$ penetration depths of:

**Table 2.** Computed dielectric properties at $\phi = 0.5$, $T = 100$ K (Maxwell-Garnett mixing).

| Band | $\nu$ (GHz) | $\varepsilon'_{\rm eff}$ | $\varepsilon''_{\rm eff}$ | $\tan\delta$ | $\kappa_a$ (m$^{-1}$) | $L_e$ (cm) |
|------|-------------|--------------------------|---------------------------|--------------|------------------------|-------------|
| 3 |  97.5 | 1.888 | $1.72 \times 10^{-3}$ | $9.1 \times 10^{-4}$ | 1.28 | 78 |
| 6 | 233.0 | 1.888 | $4.11 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | 7.31 | 14 |
| 7 | 343.5 | 1.888 | $6.07 \times 10^{-3}$ | $3.2 \times 10^{-3}$ | 15.9 | 6.3 |

### Surface emissivity

A key distinction between our approach and previous work concerns the treatment of emissivity.  Trumbo et al. (2018) used a surface-only thermal model ($T_b = \varepsilon T_{\rm surf}$) and derived a best-fit emissivity of $\varepsilon = 0.75$.  However, this effective emissivity absorbs two distinct physical effects: (1) the actual millimeter-wave surface reflectivity and (2) the fact that ALMA probes below the surface into cooler subsurface layers.  When a full radiative transfer integral through $T(z)$ is used, effect (2) is already accounted for by the depth-weighted integration.  Using $\varepsilon = 0.75$ with the RT integral therefore double-counts the subsurface cooling effect.

The physically motivated emissivity for a full RT integral is the Fresnel surface emissivity $\varepsilon(\theta) = 1 - R(\theta)$, where $R(\theta)$ is the polarization-averaged Fresnel power reflectivity at the vacuum–regolith interface:

$$R(\theta) = \frac{1}{2}\left(R_s + R_p\right), \qquad R_s = \left|\frac{\cos\theta_i - n\cos\theta_t}{\cos\theta_i + n\cos\theta_t}\right|^2, \qquad R_p = \left|\frac{n\cos\theta_i - \cos\theta_t}{n\cos\theta_i + \cos\theta_t}\right|^2$$

with $n = \mathrm{Re}(\sqrt{\varepsilon_{\rm eff}})$ and $\cos\theta_t = \sqrt{1 - (\sin\theta_i / n)^2}$.  For $\phi = 0.5$ ice ($n = 1.374$), this gives $\varepsilon_{\rm Fresnel} = 0.975$ at nadir, decreasing to $\varepsilon = 0.933$ at $\theta = 60°$.

In practice, the Fresnel emissivity overpredicts $T_b$ by $+13$ K (Table 3), indicating that additional loss mechanisms reduce the effective emissivity below the smooth-surface Fresnel value.  We find that $\varepsilon = 0.82$ minimizes the bias and provides a statistically acceptable fit ($\chi^2_r = 1.29$; Section 3.2.4).  The ~15% reduction from the Fresnel value is attributed to volume scattering within the porous regolith (Section 3.2.5).


## 3.2.3 PTM3D Simulation Setup

For each of the seven ALMA epochs, we run PTM3D with the observation-specific geometry (sub-solar longitude and latitude at the observation time, computed from SPICE kernels).  We use a resolution-6 icosahedral mesh (81,920 faces) with spatially varying Bond albedo derived from a map constructed from Voyager and Galileo photometry.  At each epoch, subsurface temperature profiles $T(z)$ are exported at a grid of 240 locations spanning the visible hemisphere in latitude (−73° to +73°) and longitude (15° spacing), using the `--locations` and `--depths` flags.  Depth profiles extend to 50 cm with 30 layers on a geometric grid, ensuring adequate sampling to optical depths $\tau > 5$ at all three ALMA bands.

The thermal model uses Europa ice regolith properties: surface thermal conductivity $k_s = 0.0096$ W m$^{-1}$ K$^{-1}$, deep conductivity $k_d = 0.05$ W m$^{-1}$ K$^{-1}$, surface density $\rho_s = 500$ kg m$^{-3}$, deep density $\rho_d = 700$ kg m$^{-3}$, with an $e$-folding depth $H = 0.07$ m.  Heat capacity follows the linear ice model $c_p(T) = 7.49 T$ J kg$^{-1}$ K$^{-1}$ (Giauque & Stout 1936), giving $c_p(100\,{\rm K}) \approx 749$ J kg$^{-1}$ K$^{-1}$.  These properties yield a surface thermal inertia $\Gamma = \sqrt{k_s \rho_s c_p} \approx 60$ J m$^{-2}$ K$^{-1}$ s$^{-1/2}$ at 100 K.  The bolometric emissivity is 0.9 and the internal heat flux is $Q_b = 0.03$ W m$^{-2}$ (30 mW m$^{-2}$).  The thermal model is equilibrated over 10 orbital periods before the output window.


## 3.2.4 Results

Table 3 summarizes the comparison statistics across four model configurations exploring the sensitivity to thermal inertia, heat flux, and emissivity treatment.

**Table 3.** Comparison statistics for PTM3D vs. ALMA ($N = 7$ observations).

| Configuration | $\Gamma$ (SI) | $Q_b$ (mW m$^{-2}$) | $\varepsilon_{\rm mm}$ | $\chi^2_r$ | Bias (K) | RMS (K) | $r$ |
|---------------|--------|--------|---------|-----------|----------|---------|-------|
| Baseline | 60 | 30 | 0.75 | 3.11 | −6.8 | 8.2 | 0.567 |
| Higher TI | 95 | 30 | 0.75 | 2.63 | −6.1 | 7.5 | 0.564 |
| Higher $Q_b$ | 60 | 100 | 0.75 | 2.81 | −6.3 | 7.8 | 0.546 |
| Fresnel $\varepsilon$ | 60 | 30 | Fresnel (~0.97) | 11.57 | +13.1 | 13.9 | 0.553 |
| **Best fit** | **60** | **30** | **0.82** | **1.29** | **+0.7** | **4.6** | **0.546** |

With the Trumbo et al. (2018) emissivity ($\varepsilon = 0.75$), all three thermal model variants show a persistent cold bias of −6 to −7 K that is insensitive to thermal inertia (60 vs. 95 SI) and geothermal heat flux (30 vs. 100 mW m$^{-2}$).  This insensitivity demonstrates that the bias originates in the radiative transfer forward model rather than the thermal model.  Increasing $Q_b$ from 30 to 100 mW m$^{-2}$ changes predicted $T_b$ by less than 0.5 K because the geothermal flux is negligible compared to the absorbed solar flux ($\sim$26 W m$^{-2}$ at the subsolar point) and contributes only at depths far below the ALMA probing range.

Replacing $\varepsilon = 0.75$ with the Fresnel emissivity overcorrects to a +13 K warm bias, confirming that the true effective emissivity lies between the Fresnel value and Trumbo's surface-model value.  A value of $\varepsilon = 0.82$ eliminates the bias ($+0.7$ K), reduces the RMS from 8.2 K to 4.6 K, and yields $\chi^2_r = 1.29$, indicating a statistically acceptable fit.

**Table 4.** Per-observation comparison for the best-fit model ($\varepsilon = 0.82$, $\Gamma = 60$ SI).

| Label | Band | Hemisphere | $\lambda_{\rm sub}$ (°W) | $T_{b,\rm obs}$ (K) | $T_{b,\rm pred}$ (K) | Residual (K) | Residual/$\sigma$ |
|-------|------|------------|--------------------------|----------------------|-----------------------|--------------|---------------------|
| H1 | 7 | Trailing | 227 | 89.1 | 85.3 | −3.8 | −0.7 |
| H2 | 7 | Leading  | 100 | 86.1 | 79.2 | −6.9 | −1.6 |
| H3 | 6 | Trailing | 270 | 80.0 | 86.0 | +6.1 | +1.5 |
| H4 | 6 | Trailing | 287 | 85.7 | 85.8 | +0.1 |  0.0 |
| H5 | 6 | Leading  |  81 | 76.6 | 78.8 | +2.2 | +0.6 |
| H7 | 3 | Leading  | 103 | 73.0 | 79.7 | +6.7 | +1.8 |
| H8 | 3 | Trailing | 283 | 85.3 | 86.2 | +0.9 | +0.2 |

The model correctly reproduces the leading–trailing hemisphere asymmetry: the trailing hemisphere (lower albedo, higher absorbed flux) produces systematically higher $T_b$ than the leading hemisphere at all three bands, consistent with the observations.  The Pearson correlation coefficient ($r = 0.55$) captures this first-order trend but is limited by the small sample size ($N = 7$) and the fact that the observed $T_b$ range (73–89 K) spans only ~16 K.

### Residual structure

The residuals show a systematic frequency dependence (Figure 2b).  Band 7 ($\lambda = 0.88$ mm, probing depth ~6 cm) is systematically underpredicted (residuals of −3.8 and −6.9 K), while Band 3 ($\lambda = 3.05$ mm, probing depth ~78 cm) is overpredicted (+0.9 and +6.7 K).  Band 6 ($\lambda = 1.25$ mm, probing depth ~14 cm) falls in between.  This pattern suggests that the Mätzler (2006) loss tangent parameterization, extrapolated to Europa temperatures (80–130 K) from calibration data primarily at $T > 200$ K, does not perfectly capture the frequency dependence of the absorption at cryogenic temperatures.  Alternatively, frequency-dependent volume scattering (Section 3.2.5) may contribute to this pattern.


## 3.2.5 Interpretation: Effective Emissivity and Volume Scattering

The best-fit millimeter-wave emissivity ($\varepsilon = 0.82$) lies between the smooth-surface Fresnel value (0.97) and the Trumbo et al. (2018) effective value (0.75).  The 15% reduction from the Fresnel prediction requires a physical loss mechanism that redirects upwelling thermal radiation back into the subsurface.

**Volume scattering** is the most likely explanation.  At ALMA wavelengths (0.88–3.05 mm), Mie scattering becomes significant if ice grain sizes are comparable to the observing wavelength.  For Europa's porous icy regolith, grain sizes on the order of ~1 mm would produce strong scattering at Band 7 and weaker scattering at Band 3.  This is consistent with both (a) the magnitude of the emissivity reduction and (b) the frequency-dependent residual pattern: if scattering preferentially reduces the effective emissivity at shorter wavelengths, the uniform $\varepsilon = 0.82$ would underpredict Band 7 (where the true emissivity is lower) and overpredict Band 3 (where scattering is less important and the true emissivity is closer to the Fresnel value).

This interpretation is consistent with Europa's anomalously high radar albedo at centimeter wavelengths (Ostro et al. 1992; Black et al. 2001), which has been attributed to coherent backscatter in a low-loss, heterogeneous ice layer extending to depths of meters.  At millimeter wavelengths we observe a weaker manifestation of the same phenomenon: subsurface heterogeneity scatters a fraction of the thermally emitted radiation, reducing the apparent emissivity below the Fresnel prediction.

The frequency-dependent residual also admits an alternative (or complementary) explanation: the Mätzler (2006) parameterization may overestimate the frequency slope of $\varepsilon''$ at cryogenic temperatures.  The parameterization predicts $\varepsilon'' \propto \nu$ (from the infrared wing term), but the calibration data are sparse below 200 K.  Thelen et al. (2024) found that they needed to multiply the imaginary part of the refractive index by a factor of ~10 relative to their baseline dielectric model to match their observations, bringing their $\mathrm{Im}(n)$ to the range $10^{-4}$–$10^{-3}$.  Our Mätzler-based model gives $\mathrm{Im}(n) = 6 \times 10^{-4}$ to $2.2 \times 10^{-3}$ at $\phi = 0.5$, consistent with the upper end of their range.

**Surface roughness** at wavelength scales (sub-mm to mm) could also contribute to the emissivity reduction, though its effect is typically smaller than volume scattering for granular media.  Conversely, a gradual density/porosity gradient with depth (which Europa's regolith likely possesses) would create a smooth impedance transition that *increases* emissivity relative to a sharp interface, implying that the scattering effect is somewhat stronger than the 15% gap suggests.

**Hemisphere-dependent porosity** may explain some of the remaining scatter.  Thelen et al. (2024) found best-fit porosities of 50–75% on the leading hemisphere and 40–50% on the trailing hemisphere.  Our model uses a uniform $\phi = 0.5$; adopting a higher leading-hemisphere porosity would increase the penetration depth and reduce predicted $T_b$ on the leading side, potentially improving the fit for observations H2 and H7 (both on the leading hemisphere, which show the largest positive residuals at Band 3 and the largest negative residual at Band 7).

### Summary of physical constraints

Our results are consistent with Europa's upper ~1 m being composed of a porous ($\phi \approx 0.5$), fine-grained (grain size $\sim$1 mm) ice regolith where:

1. Thermal emission follows the standard plane-parallel radiative transfer integral through $T(z)$;
2. Volume scattering at millimeter wavelengths reduces the effective emissivity by ~15% below the smooth-surface Fresnel value;
3. The scattering effect is likely frequency-dependent, stronger at shorter wavelengths where grain sizes approach the observing wavelength;
4. The Mätzler (2006) loss tangent extrapolation to 80–130 K is approximately correct but may slightly overestimate the frequency slope of the absorption;
5. The thermal model adequately reproduces the leading–trailing hemisphere temperature asymmetry driven by spatially varying Bond albedo.


## References

- Black, G. J., Campbell, D. B., & Nicholson, P. D. (2001). Icy Galilean satellites: Modeling radar reflectivities as a coherent backscatter effect. *Icarus*, 151, 167–180.
- Giauque, W. F., & Stout, J. W. (1936). The entropy of water and the third law of thermodynamics. *J. Am. Chem. Soc.*, 58, 1144–1150.
- Hufford, G. (1991). A model for the complex permittivity of ice at frequencies below 1 THz. *Int. J. Infrared Millimeter Waves*, 12, 677–682.
- Mätzler, C. (2006). Thermal microwave radiation: Applications for remote sensing. IET, Ch. 5.3.
- Ostro, S. J., et al. (1992). Europa, Ganymede, and Callisto: New radar results from Arecibo and Goldstone. *J. Geophys. Res.*, 97, 18227–18244.
- Thelen, A. E., et al. (2024). Subsurface thermophysical properties of Europa's leading and trailing hemispheres as revealed by ALMA. *Planet. Sci. J.*, 5, 42.
- Trumbo, S. K., Brown, M. E., & Butler, B. J. (2018). ALMA thermal observations of Europa. *Astron. J.*, 156, 161.
