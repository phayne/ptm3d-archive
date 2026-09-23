# PTM Exosphere: Europa Surface-Atmosphere Interaction Model

A coupled sublimation-ballistic transport-deposition model for Europa's tenuous water ice exosphere, operating on unstructured triangular meshes and driven by PTM3D thermal model output. The model computes vapor pressure, sublimation and deposition fluxes, ballistic molecular transport across the surface, Jeans escape, magnetospheric plasma sputtering, ice layer evolution with lag deposit tracking, amorphous-crystalline ice phase competition, and multi-timescale geological evolution via operator splitting.

## Key Capabilities

- **Sublimation & deposition** driven by spatially-resolved surface temperatures from PTM3D
- **Ballistic molecular transport** using modified Bessel function $K_0$ probability kernel on a sphere
- **Jeans thermal escape** with Chamberlain (1963) approximation
- **Magnetospheric sputtering** with Cassidy et al. (2013) global rates, trailing/leading hemisphere contrast, and temperature-dependent yields (Teolis et al. 2017)
- **Ice layer model** tracking non-ice concentration within a thermally active skin depth layer
- **Ice crystallization** via Avrami kinetics (Mitchell et al. 2017) competing with radiation-induced amorphization (Fama et al. 2010; Nordheim et al. 2022)
- **Multi-timescale evolution** from diurnal cycles to geological timescales (~$10^3$ years) via operator splitting
- **Unstructured mesh** support (icosahedral or arbitrary triangulations) with latitude/longitude metadata

---

## Quick Start

### Build

```bash
cd /path/to/ptm
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target ptm_exosphere_main -j4
```

### Run

**Minimal (sublimation only):**
```bash
./bin/ptm_exosphere_main \
    --ptm-output /path/to/ptm3d_output \
    --output exosphere_results \
    --timesteps 144
```

**Full physics (sputtering + ice layer + crystallization):**
```bash
./bin/ptm_exosphere_main \
    --ptm-output /path/to/ptm3d_output \
    --output exosphere_results \
    --timesteps 144 \
    --sputtering \
    --ice-layer \
    --crystallization
```

**Multi-timescale geological evolution (~1000 years):**
```bash
./bin/ptm_exosphere_main \
    --ptm-output /path/to/ptm3d_output \
    --output exosphere_results \
    --timesteps 144 \
    --sputtering \
    --crystallization \
    --multiscale \
    --epochs 1000 \
    --cycles-per-epoch 100 \
    --export-interval 100
```

See `examples/` for YAML configuration files.

---

## Interface to PTM3D

The exosphere model reads thermal simulation output produced by `ptm_thermal_sim`. The PTM3D output directory must contain:

### Required Input Files

| File | Description |
|------|-------------|
| `mesh.ply` | Binary PLY mesh with face latitude/longitude |
| `temperature/temperature_XXXX.bin` | Surface temperature [K] at each timestep |

The illumination data (`illumination/solar_flux_XXXX.bin`) is not required by the exosphere model; it uses only temperature fields.

### Binary Data Format

Each `.bin` file uses the same format as PTM3D output:

```
Offset  Type        Description
0       uint32      Number of faces (N)
4       float32[N]  Field values (one per mesh face)
```

All values are little-endian. File size = $4 + 4N$ bytes.

### PLY Mesh Format

Binary little-endian PLY with face-level geographic coordinates:

```
ply
format binary_little_endian 1.0
element vertex <V>
  property double x
  property double y
  property double z
element face <F>
  property list uchar int vertex_indices
  property float latitude
  property float longitude
end_header
<binary vertex data: V x 3 x double>
<binary face data: F x (1 byte + 3 x int + 2 x float)>
```

Vertices are in meters (body-fixed Cartesian). Latitude ranges from $-90°$ to $+90°$, longitude from $0°$ to $360°$ (east-positive).

### Data Flow

```
PTM3D thermal simulation
    |
    v
temperature/temperature_XXXX.bin  (N_faces x N_timesteps)
mesh.ply                          (triangulated sphere)
    |
    v
ptm_exosphere_main
    |-- Reads temperature per face per timestep
    |-- Computes vapor pressure, sublimation flux
    |-- Builds ballistic transport matrix
    |-- Iterates: emission -> transport -> deposition -> escape
    |-- (Optional) Sputtering, ice layer, crystallization
    |-- (Optional) Multi-timescale extrapolation
    |
    v
exosphere_output/
    |-- field_name/field_name_XXXX.bin
    |-- atmosphere/atmospheric_mass.csv
    |-- mesh.ply (copy)
```

---

## Physics

### Vapor Pressure and Sublimation

The equilibrium vapor pressure of water ice follows the Marti & Mauersberger (1993) formulation for hexagonal ice at low temperatures:

$$\log_{10}\!\bigl(P_{\mathrm{sat}}\bigr) = \frac{-2663.5}{T} + 12.537$$

where $P_{\mathrm{sat}}$ is the saturation vapor pressure in Pa and $T$ is the surface temperature in K. This parameterization is valid for $T = 170$-$250$ K in laboratory measurements, but is commonly extrapolated to the $80$-$150$ K range relevant to Europa's surface. At these extremely low vapor pressures, the formula provides an adequate approximation since no direct measurements exist below ~$170$ K.

The mass density of water vapor in local thermodynamic equilibrium is given by the ideal gas law:

$$\rho_v = \frac{P_{\mathrm{sat}} \, M}{R \, T}$$

where $M = 0.01802$ kg/mol is the molar mass of water and $R = 8.314$ J/(mol K) is the universal gas constant.

The mean thermal velocity of molecules in a Maxwell-Boltzmann distribution is:

$$\bar{v} = \sqrt{\frac{8 \, k_B \, T}{\pi \, m}}$$

where $k_B = 1.381 \times 10^{-23}$ J/K is Boltzmann's constant and $m = 2.992 \times 10^{-26}$ kg is the molecular mass of H$_2$O.

The sublimation mass flux (Hertz-Knudsen equation, assuming a sticking coefficient of unity) is:

$$\Phi_{\mathrm{sub}} = \frac{\rho_v \, \bar{v}}{4}$$

**Vapor pressure and sublimation flux at representative temperatures:**

| $T$ (K) | $P_{\mathrm{sat}}$ (Pa) | $\bar{v}$ (m/s) | $\Phi_{\mathrm{sub}}$ (kg/m$^2$/s) | $\Phi_{\mathrm{sub}}$ (molec/m$^2$/s) |
|---------|--------------------------|------------------|-------------------------------------|----------------------------------------|
| 80 | $1.8 \times 10^{-21}$ | 307 | $3.6 \times 10^{-24}$ | $1.2 \times 10^{2}$ |
| 90 | $8.8 \times 10^{-18}$ | 325 | $1.7 \times 10^{-20}$ | $5.7 \times 10^{5}$ |
| 100 | $8.0 \times 10^{-15}$ | 343 | $1.5 \times 10^{-17}$ | $5.0 \times 10^{8}$ |
| 110 | $2.1 \times 10^{-12}$ | 360 | $3.7 \times 10^{-15}$ | $1.2 \times 10^{11}$ |
| 120 | $2.2 \times 10^{-10}$ | 376 | $3.7 \times 10^{-13}$ | $1.2 \times 10^{13}$ |
| 130 | $1.1 \times 10^{-8}$ | 391 | $1.8 \times 10^{-11}$ | $6.1 \times 10^{14}$ |

The sublimation flux varies by more than 12 orders of magnitude over the $80$-$130$ K temperature range typical of Europa's surface, making the thermal distribution the dominant control on mass redistribution.

**Assumptions:**
- Sticking coefficient $\alpha = 1$ (all impinging molecules stick)
- Local thermodynamic equilibrium between surface ice and vapor
- Marti & Mauersberger (1993) extrapolation valid below 170 K

**Reference:** Marti, J. & Mauersberger, K. (1993). A survey and new measurements of ice vapor pressure at temperatures between 170 and 250 K. *Geophys. Res. Lett.*, 20(5), 363-366.

### Ballistic Molecular Transport

Sublimated molecules follow ballistic (collisionless) trajectories in Europa's gravitational field. The exosphere is collisionless because the mean free path (~$10^6$ km at surface number densities of ~$10^4$ cm$^{-3}$) vastly exceeds the atmospheric scale height (~$35$ km).

The probability density for a molecule launched from one surface location to land at great-circle distance $d$ on the sphere, given a Maxwellian velocity distribution at temperature $T$, is:

$$\mathcal{P}(d, T) = \frac{g^2 \, d}{4 \, \sigma^4} \, K_0(\xi)$$

where:
- $g = 1.314$ m/s$^2$ is Europa's surface gravity
- $\sigma^2 = k_B T / m$ is the one-dimensional velocity variance, with $k_B = 1.381 \times 10^{-23}$ J/K and $m = 2.992 \times 10^{-26}$ kg
- $\xi = g \, d \, / \, (2\sigma^2)$ is the dimensionless distance parameter
- $K_0(\xi)$ is the modified Bessel function of the second kind, order zero
- $d$ is the great-circle distance between source and target face centers [m]

The $K_0$ function is evaluated using polynomial approximations (Abramowitz & Stegun):
- For $x \le 2$: logarithmic series expansion
- For $x > 2$: asymptotic expansion $K_0(x) \sim e^{-x} / \sqrt{x}$

The probability is exponentially suppressed for $\xi > 10$, providing a natural cutoff for the transport matrix. The deposition flux at face $j$ from all emitting faces is:

$$F_{\mathrm{dep},j} = \sum_{i} \Phi_{\mathrm{sub},i} \; A_i \; \mathcal{P}(d_{ij}, T_i) \; \frac{\Omega_j}{4\pi}$$

where $\Phi_{\mathrm{sub},i}$ is the sublimation flux at face $i$ [kg/m$^2$/s], $A_i$ is the area of face $i$ [m$^2$], $d_{ij}$ is the great-circle distance between face centers $i$ and $j$ [m], and $\Omega_j = A_j / d_{ij}^2$ is the solid angle subtended by the target face [sr].

The transport matrix is stored as a sparse Eigen matrix to exploit the sparsity enforced by the $K_0$ cutoff.

**Characteristic hop distances:**

| $T$ (K) | $\bar{v}$ (m/s) | $d_{\mathrm{char}}$ (km) | $d_{\mathrm{ball}}$ (km) |
|---------|------------------|--------------------------|--------------------------|
| 100 | 343 | 70 | 45 |
| 110 | 360 | 77 | 49 |
| 120 | 376 | 84 | 54 |
| 130 | 391 | 91 | 58 |

The characteristic distance $d_{\mathrm{char}} = 2\sigma^2 / g$ corresponds to $\xi = 1$ (peak of the $K_0$ probability kernel). The ballistic distance $d_{\mathrm{ball}} = \bar{v}^2 / (2g)$ is the maximum range of a molecule launched vertically.

**Assumptions:**
- Collisionless exosphere (Knudsen number $\gg 1$)
- Uniform gravity (no topographic effects on trajectories)
- Lambertian emission from surface (isotropic half-space velocity distribution)
- Flat surface approximation for solid angle computation

### Jeans Escape

Molecules in the high-energy tail of the Maxwell-Boltzmann distribution can exceed the escape velocity and be lost to space. The escape fraction follows the Chamberlain (1963) approximation. The Jeans parameter is:

$$\lambda_J = \frac{v_{\mathrm{esc}}^2 \, m}{2 \, k_B \, T}$$

where $v_{\mathrm{esc}} = 2025$ m/s is Europa's escape velocity, $m = 2.992 \times 10^{-26}$ kg is the H$_2$O molecular mass, $k_B = 1.381 \times 10^{-23}$ J/K is Boltzmann's constant, and $T$ is the surface temperature [K].

The escape fraction is:

$$f_{\mathrm{esc}} = \begin{cases} 0 & \lambda_J > 50 \\[4pt] \dfrac{e^{-\lambda_J}(1 + \lambda_J)}{2\sqrt{\pi\,\lambda_J}} & 2 \le \lambda_J \le 50 \\[4pt] 0.5 & \lambda_J < 2 \end{cases}$$

**Jeans parameter and escape fraction for H$_2$O:**

| $T$ (K) | $\lambda_J$ | $f_{\mathrm{esc}}$ |
|---------|-------------|---------------------|
| 80 | 55.5 | 0 |
| 100 | 44.4 | $\sim 10^{-19}$ |
| 120 | 37.0 | $\sim 10^{-16}$ |
| 150 | 29.6 | $\sim 10^{-13}$ |
| 200 | 22.2 | $\sim 10^{-10}$ |
| 300 | 14.8 | $\sim 10^{-7}$ |

Water escape from Europa is negligible at all relevant surface temperatures. The Jeans parameter $\lambda_J \gg 1$ throughout, meaning the escape velocity is many times the thermal velocity. Escape is relevant only for lighter species (H$_2$, O$_2$) or at much higher temperatures.

**Assumption:** Exobase at the surface (collisionless above the surface).

**Reference:** Chamberlain, J.W. (1963). Planetary coronae and atmospheric evaporation. *Planet. Space Sci.*, 11(8), 901-960.

### Magnetospheric Plasma Sputtering

Europa is embedded in Jupiter's magnetosphere and subject to bombardment by energetic ions (primarily S$^+$, O$^+$, and H$^+$). Sputtering ejects molecules directly from the surface ice.

**Global sputtering rates** from Cassidy et al. (2013):

| Species | Rate (molecules/s) | Mass rate (kg/s) |
|---------|-------------------|-------------------|
| H$_2$O | $2.0 \times 10^{27}$ | 59.8 |
| O$_2$ | $1.0 \times 10^{26}$ | 5.3 |
| H$_2$ | $2.0 \times 10^{26}$ | 0.7 |

The spatial distribution of ion flux across the surface has two components:

**Longitude dependence** (trailing hemisphere enhancement):

$$f_{\mathrm{lon}} = f_{\min} + (1 - f_{\min}) \, \frac{\cos(\phi_{\mathrm{lon}} - 270°) + 1}{2}$$

where $\phi_{\mathrm{lon}}$ is the longitude [degrees], $f_{\min} = 1/C_{\mathrm{hem}}$ is the minimum flux factor, and $C_{\mathrm{hem}} = 3.0$ is the trailing/leading hemisphere contrast ratio (default; configurable via `--sputtering-contrast`). The trailing hemisphere (centered at $270°$W) receives the highest ion flux because Jupiter's corotating plasma overtakes Europa's orbital motion.

**Latitude dependence** (polar enhancement):

$$f_{\mathrm{lat}} = 1 + (E_{\mathrm{pol}} - 1) \, \frac{|\phi_{\mathrm{lat}}|}{90°}$$

where $\phi_{\mathrm{lat}}$ is the latitude [degrees] and $E_{\mathrm{pol}} = 1.2$ is the polar enhancement factor (default). This is a simplified version of the full Nordheim et al. (2022) ion flux maps.

**Temperature-dependent sputtering yield** (Teolis et al. 2017):

$$Y(T) = Y_0 \left[1 + q_0 \, \exp\!\left(\frac{-Q}{k_B \, T}\right)\right]$$

where $q_0 = 1000$ is a dimensionless prefactor, $Q = 0.06$ eV is the activation energy for thermally-enhanced sputtering, $k_B = 8.617 \times 10^{-5}$ eV/K is Boltzmann's constant (in eV units), and $Y_0$ is the baseline yield at $T = 0$. The yield factor relative to the reference temperature $T_{\mathrm{ref}} = 100$ K is:

$$\hat{Y}(T) = \frac{Y(T)}{Y(T_{\mathrm{ref}})} = \frac{1 + q_0 \, e^{-Q/(k_B T)}}{1 + q_0 \, e^{-Q/(k_B T_{\mathrm{ref}})}}$$

Below $T = 80$ K the yield is constant (temperature-independent regime).

**Current parameter set (PTM-649, default).** The gross (ice removed, H$_2$O-equivalent mass) at face $i$ is

$$\Phi_{\mathrm{gross},i} = m \thinspace S \thinspace \phi_i \left[ f + (1 - f) \thinspace \tau(T_i) \right], \qquad \tau(T) = \frac{1 + q_0 \exp(-E_a / k_B T)}{1 + q_0 \exp(-E_a / k_B T_{\mathrm{ref}})}$$

where $\phi_i$ is the H$_2$O-equivalent molecule flux at $T_{\mathrm{ref}}$. By default $\phi_i = \dot{N}_{\mathrm{tot}} w_i / A$, with the ion-access weight $w_i$ renormalized to an area-weighted mean of 1 over the mesh area $A$, so a pure-ice surface at $T_{\mathrm{ref}}$ delivers exactly $\dot{N}_{\mathrm{tot}} = 2 \times 10^{27}$ s$^{-1}$. The geometric weight is renormalized, not the realized gross: ice availability and $\tau(T)$ then reduce or raise the realized rate (the magnetosphere sets the ion flux; the surface sets the yield). The legacy code normalized by face count over a fixed $3.09 \times 10^{13}$ m$^2$. The H$_2$O part ($f$) is temperature-independent; only the radiolytic O$_2$ + H$_2$ part follows $\tau(T)$, Cassidy et al.'s O$_2$ yield law ($q_0 = 1000$, $E_a = 0.06$ eV), which is continuous and monotonic (no cutoff; $\tau \to 0.514$ as $T \to 0$). Radiolytic O$_2$ : H$_2$ is stoichiometric. The default ion pattern (BASE) is Cassidy's trailing/leading apex ratio of 3 as a cosine in longitude, with no polar term.

The escaping share of sputtered H$_2$O and the launch temperature of the redepositing share both come from the Sigmund distribution $f(E) \propto E/(E+U)^3$: $e_{\mathrm{H_2O}} = 1 - [E_{\mathrm{esc}}/(E_{\mathrm{esc}}+U)]^2$ and $T_{\mathrm{launch}} = \langle E \rangle_{E<E_{\mathrm{esc}}} / 2k_B$, with $E_{\mathrm{esc}} = m v_{\mathrm{esc}}^2/2 = 0.3828$ eV. $U = 0.054$ eV gives 0.2320 and 677 K; `--sputter-binding-energy` changes both together. H$_2$ escapes. O$_2$ is gravitationally bound but non-condensable and is booked as atmospheric loss (ionization, dissociation, pickup), not escape. Shares of the gross at $T_{\mathrm{ref}}$: 69.12% redeposited, 22.00% escaped, 8.88% O$_2$ atmospheric loss. Per face they follow $\tau(T)$, and the multiscale path uses the diurnal-mean per-face shares.

Ion maps: `--sputter-ion-map FILE` replaces $w_i$ (per-face weights, renormalized as above); `--sputter-ion-map-absolute FILE` gives $\phi_i$ directly in H$_2$O-equivalent molecules cm$^{-2}$ s$^{-1}$ at $T_{\mathrm{ref}}$ and sets the total. `--sputter-o2-map-absolute FILE` (O$_2$ molecules cm$^{-2}$ s$^{-1}$) sets the radiolytic part per face: H$_2$ is stoichiometric to it, H$_2$O is the gross minus O$_2$ and H$_2$, and $\tau(T)$ is not applied, because the Addison et al. 2022 O$_2$ map already carries the same temperature law at a diurnal-mean temperature profile. All map files are a uint64 count followed by float64 per face. HIGH bracket case: both Addison 2022 absolute maps.

**Citations** (checked 2026-09-23 against the PDFs; VERIFIED = value found at the locator, DIFFERS = the source gives another value, NOT-FOUND = the source does not contain it):

| Parameter | Value used | Source and locator | Status |
|---|---|---|---|
| $\dot{N}_{\mathrm{tot}}$ | $2 \times 10^{27}$ H$_2$O-equivalent s$^{-1}$ | Cassidy et al. 2013, PSS 77, 64: abstract (p. 64); Table 2 (p. 70); yield counts O$_2$/H$_2$ as H$_2$O mass equivalent (p. 66) | VERIFIED |
| $f$ | 0.90 | Cassidy p. 71: O$_2$ + H$_2$ "~10% by mass"; O$_2$ $10^{26}$ s$^{-1}$ (Table 2), H$_2$ twice that (p. 71) | VERIFIED |
| Trailing/leading ratio | 3 | Cassidy abstract (p. 64): leading apex "about 1/3rd of the peak"; p. 69: "about a factor of 3" | VERIFIED |
| Cosine in longitude | | Cassidy gives a computed map (Fig. 7, p. 69), not a formula; $\cos\theta$ is the cold-ion flux only (p. 70). Fig. 7 also rises with latitude on the leading hemisphere, which the cosine omits | DIFFERS (approximation) |
| Linear polar factor | 1 (legacy 1.2) | not in Cassidy; not in Nordheim et al. 2022, PSJ 3:5 (ion access only: polar and high latitudes get the largest flux, abstract p. 1 and p. 14) | NOT-FOUND |
| $q_0$, $E_a$ of $\tau(T)$ | 1000, 0.06 eV | Cassidy p. 71 (O$_2$ yield, after Teolis et al. 2010); Addison et al. 2022 Eqs. 3 and 5 (p. 14). Total-yield law $q_0 = 220$: Fama et al. 2008 Sec. 3.1 (p. 158); Cassidy Eq. 2 (p. 66) | VERIFIED |
| $T_{\mathrm{ref}}$ | 100 K | Addison et al. 2022 yield curves at 100 K (p. 14); GENTOo source | VERIFIED |
| $U$ | 0.054 eV | Plainaki et al. 2010, Icarus 210, 385: Table 1 and Sec. 2.2.1.3 (p. 388), after Boring et al. 1984 and Haring et al. 1984. The earlier 0.055 eV is not in Johnson and Sittler 1990 (GRL 17, 1629) | VERIFIED (0.055 NOT-FOUND) |
| $e_{\mathrm{H_2O}}$ | 0.2320 (derived) | Plainaki abstract and p. 388: ~22% at 0.054 eV (with the $T_m$ cutoff, 100 keV S$^+$); Cassidy p. 69: ~1/3 | DIFFERS slightly (Plainaki); DIFFERS (Cassidy) |
| $f(E) \propto E/(E+U)^3$ | | Plainaki Eq. 2 (p. 388), after Sigmund 1969 ($T_m \to \infty$ limit) | VERIFIED |
| O$_2$ bound, H$_2$ escapes | | Cassidy p. 71 | VERIFIED |
| O$_2$ loss channels | | Teolis et al. 2017, Icarus 284, 18, Sec. 2 (pp. 21-22) | VERIFIED |
| JADE O$_2$ production | $12 \pm 6$ kg/s | Szalay et al. 2024, Nat. Astron.: abstract; Table 1 | VERIFIED |
| Addison map units | cm$^{-2}$ s$^{-1}$ | Addison et al. 2022, JGR Space Phys. 127, e2021JA030136, pp. 21, 30 | VERIFIED |
| Addison H$_2$O total | map 1.98 $\times 10^{28}$ s$^{-1}$ | paper p. 25: $\approx 5.9 \times 10^{27}$ s$^{-1}$ (draped fields), after the ×0.3 regolith sticking factor (p. 12) that the shipped GENTOo source does not apply | DIFFERS (×3.3) |
| Addison O$_2$ total | map 2.65 $\times 10^{26}$ s$^{-1}$ = 14.1 kg/s | paper p. 32: 1.5–4.9 $\times 10^{26}$ s$^{-1}$ (thin layer, five temperature profiles) | VERIFIED (in range) |
| Addison energy quadrature | | discrete energies (p. 10); the integration rule is not stated | NOT-FOUND |

`--sputter-legacy-params` restores the earlier set below (implied by `--sputter-legacy-escape-only`).

**Legacy parameter set** (`--sputter-legacy-params`). Temperature-dependent yield enhancement on the whole yield, 0 below 50 K and 1 from 50 to 80 K:

| $T$ (K) | $\hat{Y}(T)$ |
|---------|---------------|
| 80 | 0.60 |
| 100 | 1.00 |
| 110 | 1.43 |
| 120 | 2.07 |
| 130 | 2.94 |
| 150 | 5.47 |

The sputtering flux at each mesh face $i$ is:

$$\Phi_{\mathrm{sput},i} = \frac{\dot{N}_{\mathrm{global}} \, m}{A_{\mathrm{E}}} \; f_{\mathrm{lon},i} \; f_{\mathrm{lat},i} \; \hat{Y}(T_i) \; S$$

where $\dot{N}_{\mathrm{global}} = 2.0 \times 10^{27}$ molecules/s is the global H$_2$O sputtering rate (Cassidy et al. 2013), $m = 2.992 \times 10^{-26}$ kg is the H$_2$O molecular mass, $A_{\mathrm{E}} = 3.09 \times 10^{13}$ m$^2$ is Europa's surface area, and $S$ is a user-configurable rate scale factor (default $S = 1.0$; set via `--sputtering-rate`).

**Sputtering vs. sublimation crossover:** The mean sputtering flux is $\bar{\Phi}_{\mathrm{sput}} = \dot{N}_{\mathrm{global}} \, m / A_{\mathrm{E}} = 1.94 \times 10^{-12}$ kg/m$^2$/s. Sublimation exceeds this at $T \gtrsim 125$ K. At the subsolar point (~$130$ K), sublimation dominates by approximately an order of magnitude. At the nightside and polar regions (~$80$-$100$ K), sputtering dominates by many orders of magnitude.

**Assumptions:**
- Steady-state magnetospheric conditions (no Jovian magnetotail passages)
- Smooth cosine variation of ion flux (simplified from full particle tracing)
- Sputtered molecules follow the same ballistic transport as sublimated molecules

**References:**
- Cassidy, T.A. et al. (2013). Magnetospheric ion sputtering and water ice grain size at Europa. *Planet. Space Sci.*, 77, 64-73.
- Teolis, B.D. et al. (2017). Plume and surface feature structure and compositional effects on Europa's global exosphere. *Icarus*, 284, 18-29.
- Fama, M. et al. (2008). Sputtering of ice by low-energy ions. *Surf. Sci.*, 602(1), 156-161.

### Ice Layer Model

The ice layer model tracks the composition of a thermally active surface layer of thickness equal to the diurnal thermal skin depth. This layer exchanges mass with the exosphere through sublimation and deposition.

The key state variable is the non-ice concentration $c$ (dimensionless, $0 \le c \le 1$), representing the volume fraction of refractory (non-volatile) material within the skin depth layer. When $c = 0$, the layer is pure ice; when $c = 1$, the layer is fully depleted of ice.

**Ice mass per unit area:**

$$m_{\mathrm{ice}} = (1 - c) \; \delta_{\mathrm{th}} \; \rho_{\mathrm{ice}}$$

where $\delta_{\mathrm{th}} = 0.05$ m is the thermal skin depth (default; configurable) and $\rho_{\mathrm{ice}} = 917$ kg/m$^3$ is the water ice density.

**Sublimation and sputtering** remove ice and increase $c$:

$$\Delta c = \frac{\Delta m_{\mathrm{loss}}}{\delta_{\mathrm{th}} \; \rho_{\mathrm{ice}}}$$

where $\Delta m_{\mathrm{loss}}$ is the ice mass removed per unit area [kg/m$^2$] over the timestep.

**Deposition** fills void space with fresh ice, decreasing $c$:

$$\Delta c = -\frac{\min\!\bigl(\Delta m_{\mathrm{dep}},\; c \, \delta_{\mathrm{th}} \, \rho_{\mathrm{ice}}\bigr)}{\delta_{\mathrm{th}} \; \rho_{\mathrm{ice}}}$$

where $\Delta m_{\mathrm{dep}}$ is the deposited ice mass per unit area [kg/m$^2$]. Deposition cannot reduce $c$ below zero (cannot overfill the layer).

A dust fraction parameter $f_{\mathrm{dust}}$ (default $0.05 = 5\%$ by volume; set via `--dust-fraction`) sets the initial non-ice concentration, representing refractory contaminants embedded in Europa's ice. As ice sublimates, this dust accumulates, forming a lag deposit that can eventually armor the surface against further sublimation.

**Physical constants used in this module:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $\rho_{\mathrm{ice}}$ | 917 kg/m$^3$ | Water ice density |
| $\rho_{\mathrm{dust}}$ | 1500 kg/m$^3$ | Refractory dust density |
| $\delta_{\mathrm{th}}$ | 0.05 m | Thermal skin depth (default) |
| $c_0$ | 0.01 | Initial non-ice concentration (1%) |

**Ice fraction from albedo (optional).** `--ice-fraction-from-albedo=MODE` sets how the
initial composition relates to the albedo map that drove the thermal run:

- `uniform` (default): $c = f_{\mathrm{dust}}$ on every face; all legacy behavior unchanged.
- `percentile`: the legacy `--init-from-albedo-map` rank mapping between the 5th and 95th
  albedo percentiles. It is not a physical inversion.
- `hapke-intimate`: invert the per-face **Bond** albedo map (`--ice-fraction-albedo-map`,
  the thermal run's own `<Body>_Albedo_res<N>.bin`) with an intimate Hapke mixture of an
  ice and a non-ice endmember. Scattering follows the engine's closed form (isotropic
  scatterers, $H(\mu)=(1+2\mu)/(1+2\mu\gamma)$, $\gamma=\sqrt{1-w^*}$), and the Bond
  albedo is $A_B(w^*) = 2\int_0^1 \mu\,[1-\gamma H(\mu)]\,d\mu$ (64-point Gauss-Legendre),
  with $w^* = w(1-g)/(1-wg)$. Each endmember's $w$ is solved from its Bond albedo. The
  mixture weights $w$ by extinction cross section and $g$ by scattering cross section
  ($\sigma_{\mathrm{sca}} = w\,\sigma_{\mathrm{ext}}$). Per face, $f$ (the ice share of
  extinction, which equals the exposed ice area in the geometric-optics limit) is found by
  bisection and clamped at the endmembers.
  With grain-size ratio $r = D_{\mathrm{ice}}/D_{\mathrm{nonice}}$ the ice volume fraction is
  $\phi = f r / (f r + 1 - f)$ (inverse: $f = \phi / (\phi + r(1-\phi))$), and the layer
  is seeded with $c = 1 - \phi$. **Sublimation and sputtering scale with the exposed area
  $f$, not with $1-c$**. In multi-timescale mode, $d\phi/dt = B - A f(\phi)$ is solved in
  closed form (`solveExposedAreaStep`); it reduces to the legacy exponential when $r = 1$.
  The albedo-coupling forward model uses the same mixture, so the albedo computed from
  the initial composition reproduces the map (the startup gate fails the run if the
  mismatch exceeds $10^{-6}$ on inverted faces). Faces whose baked map value mixed in FITS
  fill pixels (the map's $-1$ fill is not its `MISSING` value) are flagged against
  `--albedo-fill-fits`. Their map value is fill-contaminated, so it is not a measurement,
  but it is the albedo the thermal run used there: by default (`--fill-ice-fraction=map-value`)
  a fill face is inverted from that value like any other face, so its composition and its
  temperature agree (PTM-637). Fill faces without a usable value, or every fill face with
  `--fill-ice-fraction=uniform` (the earlier behavior, which warns), get
  $f = 1 - f_{\mathrm{dust}}$. The run reports the fill faces' share of the final-state
  sublimation rate separately, with their count and area. Defaults: ice $A_B = 0.80$ (Fermi's base
  case); non-ice $A_B$ = the map's area-weighted 10th percentile over non-fill faces (Paul,
  2026-09-22; `--nonice-albedo-percentile`, or `--nonice-albedo` to set it explicitly);
  $g = 0$ for both; $r = 10$.
  Requires `--multiscale`.

**Multi-timescale availability (all modes).** Each full diurnal cycle accumulates the
pure-ice *potential* emission of every face (the ice-mass gate is lifted for the
accumulation), scales it by the face's ice availability ($1-c$, or $f$ in `hapke-intimate`)
before transport, and caches the unscaled potential as the base flux. So the initial
composition sets the first-epoch sublimation, and a face that starts with no ice sublimates
again once deposition gives it ice.

**Assumptions:**
- Well-mixed layer (uniform concentration within skin depth)
- Instantaneous vertical mixing
- No grain size effects on sublimation rate
- Lag deposit does not affect thermal conductivity (future work)

### Ice Crystallization

Water ice on Europa exists in a competition between thermally-driven crystallization and radiation-induced amorphization. The model tracks the crystalline fraction $\phi_C$ (dimensionless, $0 \le \phi_C \le 1$) at each mesh face.

**Crystallization rate** follows Avrami kinetics (Mitchell et al. 2017):

$$k_{\mathrm{cryst}}^{1/n} = A_0^{1/n} \, \exp\!\left(\frac{-\Delta H_a}{R \, T}\right)$$

$$k_{\mathrm{eff}} = n \; k_{\mathrm{cryst}}^{1/n}$$

where:
- $A_0^{1/n} = 3.9 \times 10^{19}$ s$^{-1}$ is the pre-exponential factor (Mitchell et al. 2017, Stage 2; not configurable via CLI)
- $\Delta H_a = 62{,}000$ J/mol $= 62$ kJ/mol is the activation energy (default; set via `--activation-energy`)
- $R = 8.314$ J/(mol K) is the universal gas constant
- $T$ is the surface temperature [K]
- $n$ is the Avrami exponent (porosity-dependent; see below)
- $k_{\mathrm{eff}}$ is the effective first-order crystallization rate constant [s$^{-1}$]

The Avrami exponent varies with ice porosity $\varphi$ (Mitchell et al. 2017, Table 2):

$$n(\varphi) = n_{\mathrm{low}} + (n_{\mathrm{high}} - n_{\mathrm{low}}) \, \frac{\varphi}{\varphi_{\max}}$$

where $n_{\mathrm{low}} = 4.0$ (compact ice, bulk nucleation + 3D growth), $n_{\mathrm{high}} = 1.5$ (porous ice, predominantly 1D growth along pore walls), and $\varphi_{\max} = 0.7$. Default porosities: equatorial $\varphi_{\mathrm{eq}} = 0.5$ (`--equatorial-porosity`), polar $\varphi_{\mathrm{pol}} = 0.2$ (`--polar-porosity`).

**Crystallization timescale at representative temperatures** (compact ice, $n = 4$):

| $T$ (K) | $k_{\mathrm{eff}}$ (s$^{-1}$) | Timescale |
|---------|-------------------------------|-----------|
| 80 | $5.1 \times 10^{-21}$ | $> 10^{12}$ yr (frozen) |
| 100 | $6.4 \times 10^{-13}$ | ~50,000 yr |
| 110 | $5.6 \times 10^{-10}$ | ~56 yr |
| 120 | $1.6 \times 10^{-7}$ | ~72 days |
| 130 | $1.9 \times 10^{-5}$ | ~15 hr |
| 140 | $1.1 \times 10^{-3}$ | ~15 min |

**Radiation amorphization rate** (Fama et al. 2010; Leto & Baratta 2003):

$$k_{\mathrm{amorph}} = \frac{\dot{D}}{E_{\mathrm{amorph}}}$$

where $\dot{D}$ is the local radiation energy deposition rate [eV/(molecule s)] and $E_{\mathrm{amorph}} = 3.0$ eV/molecule is the energy required to amorphize one ice molecule (default; set via `--amorphization-dose`).

The radiation dose rate $\dot{D}$ varies spatially following Nordheim et al. (2022):

**Longitude factor** (S$^+$ ion flux, trailing hemisphere enhanced):

$$f_{\mathrm{lon}}^{(\mathrm{dose})} = \frac{1}{C_D} + \left(1 - \frac{1}{C_D}\right) \max\!\left(0,\; \frac{\cos(\phi_{\mathrm{lon}} - 270°) + 1}{2}\right)$$

where $C_D = 5.0$ is the trailing/leading hemisphere dose contrast (default; set via `--dose-hemisphere-contrast`) and $\phi_{\mathrm{lon}}$ is the longitude [degrees].

**Latitude factor** (polar enhancement from O$^+$ and S$^+$ at ~1 keV):

$$f_{\mathrm{lat}}^{(\mathrm{dose})} = 1 + (E_D - 1) \left(\frac{|\phi_{\mathrm{lat}}|}{90°}\right)^{1.5}$$

where $E_D = 2.5$ is the polar dose enhancement factor (default; set via `--dose-polar-enhancement`) and $\phi_{\mathrm{lat}}$ is the latitude [degrees].

**Total dose rate:**

$$\dot{D}(\phi_{\mathrm{lat}}, \phi_{\mathrm{lon}}) = \dot{D}_0 \; f_{\mathrm{lon}}^{(\mathrm{dose})} \; f_{\mathrm{lat}}^{(\mathrm{dose})}$$

where $\dot{D}_0 = 2.3 \times 10^{-6}$ eV/(molecule s) is the base dose rate at the trailing apex (default; set via `--base-dose-rate`). This is calibrated to give an amorphization timescale of ~15 days at the trailing equator, consistent with JWST observations (Cartwright et al. 2025).

At the trailing equator: $k_{\mathrm{amorph}} = \dot{D}_0 / E_{\mathrm{amorph}} = 2.3 \times 10^{-6} / 3.0 = 7.67 \times 10^{-7}$ s$^{-1}$, giving $\tau_{\mathrm{amorph}} = 1/k_{\mathrm{amorph}} = 1.30 \times 10^6$ s $= 15.1$ days.

> **Note — Relationship to sputtering spatial model.** The radiation dose distribution uses different parameters than the [sputtering spatial model](#magnetospheric-plasma-sputtering), despite both being driven by magnetospheric ion bombardment. Sputtering is dominated by thermal plasma ($\sim 100$ eV S$^+$ and O$^+$) with moderate spatial contrast ($C_{\mathrm{hem}} = 3.0$, $E_{\mathrm{pol}} = 1.2$, linear latitude dependence). Amorphization is driven by higher-energy ions ($\sim 1$ keV) that experience stronger magnetic focusing, producing a steeper trailing/leading contrast ($C_D = 5.0$), stronger polar enhancement ($E_D = 2.5$), and a super-linear latitude dependence (exponent $1.5$). Both models share the same cosine longitude functional form centered at $270°$W. The parameter differences follow Nordheim et al. (2022), who showed that the energetic ion flux map differs substantially from the thermal plasma distribution.

**Crystallinity evolution ODE:**

$$\frac{d\phi_C}{dt} = k_{\mathrm{cryst}} \, (1 - \phi_C) - k_{\mathrm{amorph}} \, \phi_C$$

This has the analytical solution:

$$\phi_C(t) = \phi_{\mathrm{eq}} + \bigl[\phi_C(0) - \phi_{\mathrm{eq}}\bigr] \, \exp\!\bigl[-(k_{\mathrm{cryst}} + k_{\mathrm{amorph}}) \, t\bigr]$$

where the equilibrium crystalline fraction is:

$$\phi_{\mathrm{eq}} = \frac{k_{\mathrm{cryst}}}{k_{\mathrm{cryst}} + k_{\mathrm{amorph}}}$$

**Equilibrium crystallinity at the trailing equator** ($k_{\mathrm{amorph}} = 7.67 \times 10^{-7}$ s$^{-1}$):

| $T$ (K) | $k_{\mathrm{cryst}}$ (s$^{-1}$) | $k_{\mathrm{amorph}}$ (s$^{-1}$) | $\phi_{\mathrm{eq}}$ |
|---------|--------------------------------|----------------------------------|----------------------|
| 80 | $5.1 \times 10^{-21}$ | $7.67 \times 10^{-7}$ | $\sim 0$ |
| 100 | $6.4 \times 10^{-13}$ | $7.67 \times 10^{-7}$ | $\sim 10^{-6}$ |
| 110 | $5.6 \times 10^{-10}$ | $7.67 \times 10^{-7}$ | 0.0007 |
| 120 | $1.6 \times 10^{-7}$ | $7.67 \times 10^{-7}$ | 0.17 |
| 130 | $1.9 \times 10^{-5}$ | $7.67 \times 10^{-7}$ | 0.96 |

This predicts a sharp transition from amorphous to crystalline ice between 120-130 K, consistent with the JWST detection of crystalline ice predominantly at the sub-solar point and warm latitudes (Cartwright et al. 2025).

**Depositional amorphization:** Freshly deposited ice at Europa's surface temperatures is amorphous. When new ice of mass $\Delta m_{\mathrm{dep}}$ is deposited on a face with existing ice mass $m_{\mathrm{old}}$, the crystalline fraction is diluted:

$$\phi_{C,\mathrm{new}} = \phi_{C,\mathrm{old}} \; \frac{m_{\mathrm{old}}}{m_{\mathrm{old}} + \Delta m_{\mathrm{dep}}}$$

**Assumptions:**
- Spatially uniform porosity at a given latitude (no grain-scale heterogeneity)
- Activation energy and pre-exponential are constant (single crystallization mechanism)
- Radiation dose rate is steady-state (no magnetospheric variability)
- Dose is deposited uniformly within the skin depth layer

**References:**
- Mitchell, E.H. et al. (2017). Porosity effects on crystallization kinetics of amorphous water ice. *Icarus*, 285, 291-299.
- Fama, M. et al. (2010). Radiation-induced amorphization of crystalline ice. *Surf. Sci.*, 604, 1602-1607.
- Leto, G. & Baratta, G.A. (2003). Ly-alpha photon induced amorphization of Ic water ice at 16 K. *Astron. Astrophys.*, 397(1), 7-13.
- Nordheim, T.A. et al. (2022). Magnetospheric Ion Bombardment of Europa's Surface. *Planet. Sci. J.*, 3, 5.
- Cartwright, R.J. et al. (2025). Crystalline Water Ice on Europa's Surface. *Planet. Sci. J.*, 6, 125.

### Multi-Timescale Evolution

The diurnal cycle of sublimation and deposition ($P_{\mathrm{diurnal}} = 3.5512$ days $= 306{,}824$ s) is much shorter than the geological timescales over which ice layer composition evolves (~$10^2$-$10^3$ years). Direct simulation of every diurnal timestep over geological time is computationally prohibitive. The model uses operator splitting to bridge these timescales.

**Algorithm:**

1. **Diurnal cycle:** Run one complete diurnal cycle ($N_t$ timesteps) with the current surface state. Compute time-averaged fluxes at each face $i$:

$$\bar{\Phi}_{\mathrm{sub},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{sub},i}(t) \;, \quad \bar{\Phi}_{\mathrm{dep},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{dep},i}(t) \;, \quad \bar{\Phi}_{\mathrm{sput},i} = \frac{1}{N_t} \sum_{t=1}^{N_t} \Phi_{\mathrm{sput},i}(t)$$

   where $N_t$ is the number of timesteps per diurnal cycle (set via `--timesteps`).

2. **Epoch extrapolation:** Apply the time-averaged fluxes over one epoch of $N_c$ diurnal cycles:

$$\Delta t_{\mathrm{epoch}} = N_c \; P_{\mathrm{diurnal}}$$

$$\Delta c_i = \frac{\bar{\Phi}_{\mathrm{sub},i} + \bar{\Phi}_{\mathrm{sput},i} - \bar{\Phi}_{\mathrm{dep},i}}{\delta_{\mathrm{th}} \; \rho_{\mathrm{ice}}} \; \Delta t_{\mathrm{epoch}}$$

   where $N_c = 100$ is the number of diurnal cycles per epoch (default; set via `--cycles-per-epoch`), $\delta_{\mathrm{th}} = 0.05$ m is the thermal skin depth, and $\rho_{\mathrm{ice}} = 917$ kg/m$^3$ is the ice density. Crystallinity is also evolved analytically over the epoch using the equilibrium solution from the previous section.

3. **Repeat** for $N_e$ epochs (default $N_e = 1000$; set via `--epochs`).

**Flux scaling for depleted regions:** As ice is removed from a face ($c \to 1$), the sublimation flux from that face decreases proportionally. The deposition flux is rescaled to conserve global mass:

$$\bar{\Phi}_{\mathrm{dep}}^{(\mathrm{scaled})} = \bar{\Phi}_{\mathrm{dep}}^{(\mathrm{base})} \; \frac{\sum_i \Phi_{\mathrm{em},i}^{(\mathrm{current})}}{\sum_i \Phi_{\mathrm{em},i}^{(\mathrm{base})}}$$

where $\Phi_{\mathrm{em},i}^{(\mathrm{base})}$ is the emission flux computed with the initial (undepleted) state and $\Phi_{\mathrm{em},i}^{(\mathrm{current})}$ reflects the current ice availability.

**Analytical acceleration:** For faces where the evolution is slow (time constants $\gg \Delta t_{\mathrm{epoch}}$), the model uses the analytical ODE solution directly rather than time-stepping. Always on; it cannot be disabled (`--no-analytical` is refused).

**Diurnal-cycle cache:** The diurnal cycle is computed at the first epoch and recomputed only when `--albedo-coupling` refreshes the temperatures. There is no concentration-based or periodic refresh; `--cache-refresh N` is refused.

**Albedo-coupling temperature refresh:** the exosphere does not re-solve the thermal model. It rescales the thermal-run temperatures, $T_{\mathrm{new}} = T_{\mathrm{ref}}\,[(1-A_{\mathrm{new}})/(1-A_{\mathrm{ref}})]^{1/4}$, where $A_{\mathrm{ref}}$ is the per-face albedo the current temperature array corresponds to: the thermal run's own albedo (`--thermal-albedo` / `--thermal-albedo-map`; `hapke-intimate` defaults to its `--ice-fraction-albedo-map`) until the first refresh, and the albedo of the last refresh after that. A refresh fires when $\max_f |A_f - A_{\mathrm{ref},f}|$ exceeds `--albedo-update-threshold`, so slow drift accumulates until it fires; every face is then rescaled from its own $A_{\mathrm{ref}}$, so no sub-threshold drift is lost. If the $t=0$ composition albedo differs from the thermal albedo by more than $10^{-3}$ on any face, the run prints a WARNING and the first pass rescales those temperatures; that difference is a configuration mismatch, not feedback. Exception: `hapke-intimate` faces the forward model cannot reproduce (fill, or clamped at $f=0$ or $f=1$) are anchored at their $t=0$ composition albedo, so their thermal temperatures are kept at $t=0$ (they are reported separately at startup). With the default `--fill-ice-fraction=map-value`, an in-range fill face's $t=0$ composition albedo equals its thermal albedo, so its anchor is the thermal albedo; before PTM-637 fill faces were seeded at $f = 1 - f_{\mathrm{dust}}$ (bright) while their temperatures came from a dark map value, and the first refresh after they lost ice heated them by up to 35 K (a one-snapshot spike to 1,313 kg/s in the Europa albedo-map product).

**Default parameters:**

| Symbol | Default | Description |
|--------|---------|-------------|
| $N_e$ | 1000 | Number of geological epochs |
| $N_c$ | 100 | Diurnal cycles per epoch (~1 Earth year) |
| export interval | 100 | Export data every $N$ epochs |
| analytical acceleration | on | Use analytical solutions for slow regions |
| cache refresh | 0 (adaptive) | Epochs between forced diurnal recomputation |

With defaults: $N_c \times N_e = 100 \times 1000 = 100{,}000$ orbits $= 100{,}000 \times 3.5512$ d $/ 365.25$ d/yr $= 972$ years of geological evolution.

**Assumptions:**
- Diurnal-averaged fluxes are representative of the full diurnal cycle
- Surface temperature distribution is quasi-steady (no secular trend within an epoch)
- Ice layer changes do not significantly alter thermal properties (one-way coupling)

---

## Physical Constants

All constants are defined in the source headers and used consistently throughout.

**Europa:**

| Symbol | Value | Source |
|--------|-------|--------|
| $R_{\mathrm{E}}$ | $1.5608 \times 10^6$ m | IAU |
| $g$ | 1.314 m/s$^2$ | $GM/R_{\mathrm{E}}^2$ |
| $v_{\mathrm{esc}}$ | 2025 m/s | $\sqrt{2gR_{\mathrm{E}}}$ |
| $A_{\mathrm{E}}$ | $3.09 \times 10^{13}$ m$^2$ | $4\pi R_{\mathrm{E}}^2$ |
| $P_{\mathrm{diurnal}}$ | 3.5512 days ($306{,}824$ s) | Synchronous rotation |

**Water ice:**

| Symbol | Value |
|--------|-------|
| $m$ | $2.992 \times 10^{-26}$ kg (molecular mass) |
| $M$ | 0.01802 kg/mol (molar mass) |
| $\rho_{\mathrm{ice}}$ | 917 kg/m$^3$ |

**Fundamental:**

| Symbol | Value |
|--------|-------|
| $k_B$ | $1.381 \times 10^{-23}$ J/K |
| $R$ | 8.314 J/(mol K) |
| $N_A$ | $6.022 \times 10^{23}$ mol$^{-1}$ |

---

## Quantitative Estimates and Validation

The following estimates are computed analytically from the model equations using the constants defined in the code. They serve as sanity checks for simulation output.

### Vapor Pressure

The Marti & Mauersberger (1993) formula $\log_{10}(P) = -2663.5/T + 12.537$ gives:

| $T$ (K) | $\log_{10}(P)$ | $P_{\mathrm{sat}}$ (Pa) |
|---------|----------------|--------------------------|
| 80 | $-20.76$ | $1.8 \times 10^{-21}$ |
| 90 | $-17.06$ | $8.8 \times 10^{-18}$ |
| 100 | $-14.10$ | $8.0 \times 10^{-15}$ |
| 110 | $-11.68$ | $2.1 \times 10^{-12}$ |
| 120 | $-9.66$ | $2.2 \times 10^{-10}$ |
| 130 | $-7.95$ | $1.1 \times 10^{-8}$ |
| 140 | $-6.49$ | $3.3 \times 10^{-7}$ |
| 150 | $-5.22$ | $6.0 \times 10^{-6}$ |

Europa's global mean surface temperature is approximately 100-110 K, giving a characteristic vapor pressure of $10^{-15}$ to $10^{-12}$ Pa. The subsolar point can reach ~130 K, producing locally much higher sublimation rates.

### Sputtering vs. Sublimation Balance

The mean sputtering flux (global rate divided by surface area) is:

$$\bar{\Phi}_{\mathrm{sput}} = \frac{\dot{N}_{\mathrm{global}} \, m}{A_{\mathrm{E}}} = \frac{2.0 \times 10^{27} \times 2.99 \times 10^{-26}}{3.09 \times 10^{13}} = 1.94 \times 10^{-12} \;\text{kg/m}^2\text{/s}$$

Sublimation exceeds this mean sputtering rate at $T \gtrsim 125$ K. At the subsolar point (~130 K), sublimation dominates by approximately an order of magnitude. At the nightside and polar regions (~80-100 K), sputtering dominates by many orders of magnitude.

### Atmospheric Scale Height

The barometric scale height for H$_2$O is:

$$H = \frac{k_B \, T}{m \, g}$$

where $k_B = 1.381 \times 10^{-23}$ J/K, $m = 2.992 \times 10^{-26}$ kg, and $g = 1.314$ m/s$^2$. This gives $H = 35$ km at 100 K, $39$ km at 110 K, and $42$ km at 120 K.

### Mass Conservation

In steady state, global mass balance requires:

$$\sum_i \Phi_{\mathrm{em},i} \, A_i = \sum_j \Phi_{\mathrm{dep},j} \, A_j + \sum_i \Phi_{\mathrm{esc},i} \, A_i$$

where $\Phi_{\mathrm{em}}$ includes both sublimation and sputtering, $\Phi_{\mathrm{dep}}$ is deposition, $\Phi_{\mathrm{esc}}$ is Jeans escape, and $A_i$ is the area of face $i$. Since H$_2$O escape is negligible ($\lambda_J \gg 1$), essentially all sublimated and sputtered molecules are redeposited on the surface. The simulation tracks this balance via `atmosphere/atmospheric_mass.csv`.

### Crystallization-Amorphization Equilibrium

At the trailing equator (maximum radiation dose, $k_{\mathrm{amorph}} = 7.67 \times 10^{-7}$ s$^{-1}$), the equilibrium crystalline fraction $\phi_{\mathrm{eq}} = k_{\mathrm{cryst}} / (k_{\mathrm{cryst}} + k_{\mathrm{amorph}})$ transitions sharply:
- $T < 110$ K: $\phi_{\mathrm{eq}} < 0.001$ (radiation wins, ice is amorphous)
- $T \sim 120$ K: $\phi_{\mathrm{eq}} \sim 0.17$ (competition)
- $T > 130$ K: $\phi_{\mathrm{eq}} > 0.96$ (crystallization wins)

On the leading hemisphere (dose rate ~$5\times$ lower), the transition shifts to lower temperatures. This is consistent with JWST observations showing crystalline ice at warm, leading-hemisphere locations (Cartwright et al. 2025).

The amorphization timescale at the trailing equator is $\tau_{\mathrm{amorph}} = 1/k_{\mathrm{amorph}} = 15.1$ days, matching the Cartwright et al. (2025) constraint.

---

## Output Format

### Directory Structure

```
exosphere_output/
    mesh.ply                                    # Copy of input mesh
    temperature_K/temperature_K_0000.bin        # (diurnal mode)
    temperature_mean_K/temperature_mean_K_0000.bin  # (multiscale mode)
    temperature_min_K/temperature_min_K_0000.bin    # (multiscale mode)
    temperature_max_K/temperature_max_K_0000.bin    # (multiscale mode)
    vapor_pressure_Pa/vapor_pressure_Pa_0000.bin
    emission_flux_kg_m2_s/emission_flux_kg_m2_s_0000.bin
    deposition_flux_kg_m2_s/deposition_flux_kg_m2_s_0000.bin
    net_flux_kg_m2_s/net_flux_kg_m2_s_0000.bin
    accumulated_mass_kg_m2/accumulated_mass_kg_m2_0000.bin
    ice_thickness_m/ice_thickness_m_0000.bin
    escape_flux_kg_m2_s/escape_flux_kg_m2_s_0000.bin
    atmosphere/atmospheric_mass.csv
```

### Complete Field List

| Field Name | Units | Description | Condition |
|------------|-------|-------------|-----------|
| `temperature_K` | K | Instantaneous surface temperature | Diurnal mode |
| `temperature_mean_K` | K | Diurnal-mean surface temperature | Multiscale mode |
| `temperature_min_K` | K | Diurnal minimum temperature | Multiscale mode |
| `temperature_max_K` | K | Diurnal maximum temperature | Multiscale mode |
| `vapor_pressure_Pa` | Pa | Equilibrium vapor pressure | Always |
| `emission_flux_kg_m2_s` | kg/m$^2$/s | Sublimation mass flux | Always |
| `deposition_flux_kg_m2_s` | kg/m$^2$/s | Deposition mass flux | Always |
| `net_flux_kg_m2_s` | kg/m$^2$/s | Net mass flux (emission $-$ deposition) | Always |
| `accumulated_mass_kg_m2` | kg/m$^2$ | Cumulative net mass change | Always |
| `ice_thickness_m` | m | Equivalent ice thickness change | Always |
| `escape_flux_kg_m2_s` | kg/m$^2$/s | Jeans escape flux | Always |
| `sputtering_flux_kg_m2_s` | kg/m$^2$/s | H$_2$O sputtering flux | `--sputtering` |
| `sputtering_yield` | dimensionless | Temperature-dependent yield $\hat{Y}(T)$ | `--sputtering` |
| `sputtering_mass_loss_kg_m2_s` | kg/m$^2$/s | Total sputtering mass loss | `--sputtering` |
| `sputtering_spatial_weight` | dimensionless | Ion flux spatial distribution $f_{\mathrm{lon}} f_{\mathrm{lat}}$ | `--sputtering` (step 0) |
| `nonice_concentration` | dimensionless | Non-ice volume fraction $c$ | `--ice-layer` |
| `ice_mass_kg_m2` | kg/m$^2$ | Ice mass per unit area $m_{\mathrm{ice}}$ | `--ice-layer` |
| `cumulative_sublimated_kg_m2` | kg/m$^2$ | Total ice sublimated | `--ice-layer` |
| `cumulative_sputtered_kg_m2` | kg/m$^2$ | Total ice sputtered | `--ice-layer` |
| `cumulative_deposited_kg_m2` | kg/m$^2$ | Total ice deposited | `--ice-layer` |
| `crystalline_fraction` | dimensionless | Crystalline ice fraction $\phi_C$ | `--crystallization` |
| `porosity` | dimensionless | Ice porosity $\varphi$ | `--crystallization` |
| `radiation_dose_rate_eV_mol_s` | eV/(molecule s) | Radiation dose rate $\dot{D}$ | `--crystallization` |
| `cumulative_dose_eV_mol` | eV/molecule | Cumulative radiation dose | `--crystallization` |
| `crystallization_timescale_yr` | years | $1/k_{\mathrm{eff}}$ in years | `--crystallization` |

### Atmospheric Mass CSV

The file `atmosphere/atmospheric_mass.csv` contains one row per exported timestep:

| Column | Units | Description |
|--------|-------|-------------|
| `timestep` | index | Timestep number |
| `time_s` | seconds | Elapsed time |
| `time_years` | years | Elapsed time |
| `h2o_column_mass_kg_m2` | kg/m$^2$ | Global mean H$_2$O column mass |
| `h2o_vapor_pressure_Pa` | Pa | Global mean vapor pressure |
| `o2_vapor_pressure_Pa` | Pa | O$_2$ partial pressure (if tracked) |
| `h2_vapor_pressure_Pa` | Pa | H$_2$ partial pressure (if tracked) |

---

## Command-Line Reference

### Required Arguments

| Flag | Description |
|------|-------------|
| `--ptm-output DIR` | Directory containing PTM3D temperature output |
| `--output DIR` | Output directory for exosphere results (default: `./exosphere_output`) |
| `--timesteps N` | Number of diurnal timesteps to simulate |

### Optional Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--mesh FILE` | `<ptm-output>/mesh.ply` | Mesh file path (if different from PTM3D output) |
| `--equatorial-ice MM` | 0.0 | Initial equatorial ice thickness [mm] |
| `--polar-ice MM` | 0.0 | Initial polar ice thickness [mm] |

### Sputtering Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sputtering` | off | Enable magnetospheric plasma sputtering |
| `--no-sputtering` | | Disable sputtering (sublimation only) |
| `--sputtering-rate` $S$ | 1.0 | Sputtering rate scale factor ($S = 1.0$ = Cassidy et al. 2013) |
| `--sputtering-contrast` $C_{\mathrm{hem}}$ | 3.0 | Trailing/leading hemisphere flux ratio |

### Ice Layer Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ice-layer` | off | Enable ice layer model with lag deposit tracking |
| `--no-ice-layer` | | Disable ice layer model |
| `--dust-fraction` $f_{\mathrm{dust}}$ | 0.05 | Dust volume fraction in ice |
| `--ice-fraction-from-albedo=MODE` | `uniform` | `uniform`, `percentile` (= `--init-from-albedo-map`), or `hapke-intimate` |
| `--ice-fraction-albedo-map` $F$ | (required for `hapke-intimate`) | Per-face Bond albedo `.bin` used by the thermal run |
| `--albedo-fill-fits` $F$ | `Europa-0.5deg-BondAlbedo.fits` beside the `.bin` | FITS map for fill flagging; `none` skips it, with a warning |
| `--fill-ice-fraction` $M$ | `map-value` | Fill-face initial composition (`hapke-intimate`): `map-value` inverts the face's own map value (the thermal run's albedo); `uniform` gives $f = 1 - f_{\mathrm{dust}}$ and warns |
| `--ice-albedo` $A$ | 0.80 | Ice endmember Bond albedo (`hapke-intimate`) |
| `--nonice-albedo` $A$ | map P10 | Pin the non-ice endmember Bond albedo (`hapke-intimate`); needed for a uniform map, whose P10 is the map value itself |
| `--nonice-albedo-percentile` $P$ | 10 | Area-weighted map percentile (fill excluded) used as the non-ice endmember |
| `--g-ice`, `--g-nonice` $g$ | 0, 0 | Endmember asymmetry parameters (`hapke-intimate`) |
| `--grain-ratio` $r$ | 10 | $D_{\mathrm{ice}}/D_{\mathrm{nonice}}$ (`hapke-intimate`) |

### Albedo Coupling Options

| Flag | Default | Description |
|------|---------|-------------|
| `--albedo-coupling` | off | Composition-dependent albedo with temperature refresh (enables `--ice-layer`) |
| `--thermal-albedo` $A$ | (required) | Uniform Bond albedo the thermal run used; the temperature reference (`--initial-albedo` is an alias) |
| `--thermal-albedo-map` $F$ | (required) | Per-face Bond albedo `.bin` (uint64 count + double[count]) the thermal run used |
| `--albedo-update-threshold` $T$ | 0.01 | Refresh when any face has moved by more than $T$ since the last refresh |
| `--albedo-ice`, `--albedo-nonice` $A$ | 0.65, 0.10 | Linear-law endmembers (not `hapke-intimate`) |

Albedo coupling (`--albedo-coupling`, `percentile`, `hapke-intimate`) refuses to run unless the thermal-run albedo is declared; `hapke-intimate` takes its `--ice-fraction-albedo-map` unless one of the two flags overrides it. Declaring a uniform value when the thermal run's `config_used.yaml` says `use_albedo_map: true`, or a map when it says `false`, prints a WARNING. The flags are refused without albedo coupling. For the Europa uniform-ice thermal configs, pass `--thermal-albedo 0.5`.

### Multi-Timescale Options

| Flag | Default | Description |
|------|---------|-------------|
| `--multiscale` | off | Enable multi-timescale geological evolution |
| `--epochs` $N_e$ | 1000 | Number of geological epochs |
| `--cycles-per-epoch` $N_c$ | 100 | Diurnal cycles per epoch (~100 = 1 Earth year) |
| `--export-interval` $N$ | 100 | Export every $N$ epochs, plus the final state (0 = the final state only; earlier builds wrote nothing for 0). Snapshot $L$ is the state after exactly $L$ epochs ($t = L\,\Delta t_{\mathrm{epoch}}$); the last is labeled `--epochs`. `time_steps.txt` lists one time per snapshot (PTM-636; earlier builds wrote state fields after the epoch's super-step, up to one interval past the label, and omitted the final snapshot from `time_steps.txt`) |
| `--max-superstep-epochs` $N$ | none | Cap one super-step at $N$ epochs (step-size refinement checks; the physical bound is the emission-weighted availability change) |
| `--no-albedo-step-limit` | off | Pre-PTM-640 step: super-steps are not ended when a face's albedo would pass `--albedo-update-threshold`, so the temperature refresh can lag by a whole step and the answer depends on the step size. Implied by `--sputter-legacy-escape-only`. By default, with `--albedo-coupling`, each super-step ends at the first epoch count whose predicted albedo deviation (from the albedo the temperatures correspond to) exceeds the threshold (bisection on the closed-form step), so every refresh fires within one epoch of being due |
| `--superstep-legacy-deposition` | off | Pre-PTM-644 step: deposition held at its start-of-step value while the sources follow the ice availability, and the trapezoid mean availability in the linear mode. Not mass-conserving; reproduces earlier products only. Implied by `--sputter-legacy-escape-only` |
| `--no-analytical` | (refused) | Not implemented; the CLI exits with an error |
| `--cache-refresh` $N$ | (refused) | Not implemented; the CLI exits with an error |

**Deposition over a step (PTM-644).** Within a step, each face's emission follows its own ice availability, while deposition is the transport kernel's cached pattern times a global scale per source group (sublimation; sputtered H$_2$O). The scale is held at the value whose deposition equals what the kernel retains from the emission the step actually produced:
$s_k S_{k,\mathrm{base}} = \sum_i a_i E_{k,i} \langle f_i \rangle (B(s))$, solved by Newton on 1-2 scalars, with $\langle f_i \rangle$ the exact step-mean availability for that constant deposition. Each step then satisfies deposited $= \rho_k \times$ emitted per group ($\rho_{\mathrm{sub}} = 1 -$ Jeans/unscattered fraction; $\rho_{\mathrm{sput}}$ = the non-escaping H$_2$O share), and the run throws if a step departs from it by more than $10^{-6}$ relative. `evolution_summary.txt` reports the cumulative residual, the non-redeposited mass, deposition that overflows a pure-ice layer, and the ice inventory against the booked fluxes. Earlier builds held deposition at its start-of-step value, so a step in which availability fell deposited more than it emitted, by an amount that depended on the step size.

### Ice Crystallization Options

| Flag | Default | Description |
|------|---------|-------------|
| `--crystallization` | off | Enable crystallization tracking (auto-enables `--ice-layer`) |
| `--initial-crystallinity` $\phi_{C,0}$ | 0.0 | Initial crystalline fraction $[0,1]$ |
| `--activation-energy` $\Delta H_a$ | 62000 | Activation energy [J/mol] |
| `--amorphization-dose` $E_{\mathrm{amorph}}$ | 3.0 | Amorphization threshold [eV/molecule] |
| `--equatorial-porosity` $\varphi_{\mathrm{eq}}$ | 0.5 | Equatorial ice porosity $[0,1]$ |
| `--polar-porosity` $\varphi_{\mathrm{pol}}$ | 0.2 | Polar ice porosity $[0,1]$ |
| `--dose-hemisphere-contrast` $C_D$ | 5.0 | Trailing/leading radiation dose ratio |
| `--dose-polar-enhancement` $E_D$ | 2.5 | Pole/equator radiation dose ratio |
| `--base-dose-rate` $\dot{D}_0$ | $2.3 \times 10^{-6}$ | Base dose rate [eV/(molecule s)] |

### Dependency Rules

- `--crystallization` automatically enables `--ice-layer`
- `--multiscale` automatically enables `--ice-layer`
- `--sputtering-rate` automatically enables `--sputtering`
- `--dust-fraction` automatically enables `--ice-layer`
- `--initial-crystallinity` automatically enables `--crystallization` and `--ice-layer`
- `--epochs` automatically enables `--multiscale` and `--ice-layer`
- `--ice-fraction-from-albedo=percentile` or `=hapke-intimate` enables `--ice-layer` and albedo coupling; `hapke-intimate` requires `--multiscale` and refuses the legacy linear-model flags (`--albedo-ice`, `--albedo-nonice`, `--albedo-nonice-map`, `--albedo-maps-dir`, `--albedo-percentile-*`); the `hapke-intimate` parameters are refused in the other modes

---

## Code Architecture

### Source Files

| File | Lines | Description |
|------|-------|-------------|
| `ptm_exosphere_main.cpp` | ~1900 | Main driver: CLI parsing, simulation loop, diagnostics |
| `ptm_exosphere_integration.h/cpp` | ~900 | PTM3D interface: mesh/temperature I/O, binary export, PLY writer |
| `exosphere_physics.h/cpp` | ~2200 | Core physics: vapor pressure, ballistic transport matrix, $K_0$ kernel |
| `sputtering_model.h/cpp` | ~700 | Sputtering: Cassidy rates, spatial distribution, $T$-dependent yield |
| `ice_layer_model.h` | ~500 | Ice layer: concentration tracking, sublimation/deposition bookkeeping |
| `ice_crystallization_model.h` | ~600 | Crystallization: Avrami kinetics, radiation amorphization, dose maps |
| `multitimescale_evolution.h` | ~650 | Operator splitting: epoch extrapolation, flux caching, analytical accel. |
| `hapke_mixing.h` | ~220 | Hapke Bond albedo, endmember inversion, intimate mixture (`hapke-intimate`) |
| `composition_albedo.h` | ~240 | Composition-to-albedo forward model (linear, or Hapke in `hapke-intimate`) |
| `albedo_refresh.h` | ~170 | Albedo-coupling temperature refresh baseline and $t=0$ mismatch check |
| `distance_cache.h` | ~300 | Great-circle distance cache with spatial indexing |
| `spherical_index.h` | ~400 | Spatial hash grid on sphere for neighbor queries |

### Performance Features

- **Distance cache:** Pre-computed great-circle distances between face pairs, avoiding repeated trigonometric evaluations during transport matrix construction.
- **Spherical spatial index:** Hash grid on the unit sphere for $O(1)$ neighbor lookup, reducing transport matrix construction from $O(N^2)$ to $O(Nk)$ where $k$ is the number of neighbors within the $K_0$ cutoff radius.
- **Sparse transport matrix:** Eigen sparse matrix exploiting the exponential decay of $K_0$ at large distances. Typical sparsity: >99% zeros for meshes with $> 10^5$ faces.
- **Analytical acceleration:** For multi-timescale mode, faces with slow evolution (long time constants) use the closed-form ODE solution instead of explicit time-stepping.

### Test Programs

| File | Description |
|------|-------------|
| `test_transport_physics.cpp` | Unit tests for $K_0$ kernel, transport matrix normalization |
| `test_smooth_kernel_real_data.cpp` | Transport kernel validation with real mesh data |
| `test_pentagon_bias.cpp` | Icosahedral mesh bias detection in transport |
| `test_cutoff_benchmark.cpp` | Performance benchmarks for $K_0$ cutoff distance |
| `test_equilibrium_bias.cpp` | Equilibrium state bias analysis |
| `test_bessel_k0_and_areas.cpp` | EXO-01/EXO-02 regression: $K_0$ values, m$^2$ areas, smoothing gate |
| `test_albedo_refresh.cpp` | Albedo-coupling refresh: cumulative-drift trigger, no drift lost across refreshes, $t=0$ mismatch warning, thermal-albedo refusal, pinned `hapke-intimate` endmember; with `<mesh.ply> <map.bin>`, fill faces follow the thermal albedo (PTM-637) |
| `test_multiscale_snapshot_time.cpp` | PTM-636: snapshot $L$ is the state after $L$ epochs (== a run stopped at $L$), same-state fields, complete `time_steps.txt` |
| `test_ice_fraction_from_albedo.cpp` | `hapke-intimate`: Hapke $A_B$ and inverse, mixture monotonicity, forward/initialization match, exposed-area ODE |
| `analyze_diagnostics.py` | Python diagnostic analysis of simulation output |

---

## References

Cartwright, R.J. et al. (2025). Crystalline Water Ice on Europa's Surface. *Planet. Sci. J.*, 6, 125. doi:10.3847/PSJ/adba5b

Cassidy, T.A. et al. (2013). Magnetospheric ion sputtering and water ice grain size at Europa. *Planet. Space Sci.*, 77, 64-73. doi:10.1016/j.pss.2012.07.008

Chamberlain, J.W. (1963). Planetary coronae and atmospheric evaporation. *Planet. Space Sci.*, 11(8), 901-960. doi:10.1016/0032-0633(63)90122-3

Fama, M. et al. (2008). Sputtering of ice by low-energy ions. *Surf. Sci.*, 602(1), 156-161. doi:10.1016/j.susc.2007.10.002

Fama, M. et al. (2010). Radiation-induced amorphization of crystalline ice. *Surf. Sci.*, 604, 1602-1607.

Leto, G. & Baratta, G.A. (2003). Ly-alpha photon induced amorphization of Ic water ice at 16 K. *Astron. Astrophys.*, 397(1), 7-13.

Marti, J. & Mauersberger, K. (1993). A survey and new measurements of ice vapor pressure at temperatures between 170 and 250 K. *Geophys. Res. Lett.*, 20(5), 363-366. doi:10.1029/93GL00105

Mitchell, E.H. et al. (2017). Porosity effects on crystallization kinetics of amorphous water ice: Implications for cold icy bodies in the outer Solar System. *Icarus*, 285, 291-299. doi:10.1016/j.icarus.2016.11.004

Nordheim, T.A. et al. (2022). Magnetospheric Ion Bombardment of Europa's Surface. *Planet. Sci. J.*, 3, 5. doi:10.3847/PSJ/ac382a

Teolis, B.D. et al. (2017). Plume and surface feature structure and compositional effects on Europa's global exosphere: Preliminary Europa flyby science. *Icarus*, 284, 18-29. doi:10.1016/j.icarus.2016.10.027
