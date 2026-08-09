# Crystallization–Amorphization Module

*Standalone module description for the PTM3D exosphere paper. Mirrors and extends `appendix-d-exosphere-model-full.md` §D.7, adds the Jensen's-inequality-corrected diurnal averaging used in production, and points to the implementation files.*

## 1. Scope

Each mesh face $i$ carries a crystalline fraction $\phi_{C,i}\in[0,1]$ that evolves under three concurrent processes:

1. **Thermal crystallization** of amorphous H₂O ice (Mitchell et al. 2017 Avrami kinetics).
2. **Radiation-induced amorphization** by magnetospheric ion bombardment (Famá et al. 2010; spatial map from Nordheim et al. 2022).
3. **Depositional amorphization** — freshly transported H₂O lands amorphous at Europan surface temperatures and dilutes the existing $\phi_C$ by mass-weighted mixing.

Because $k_{\mathrm{cryst}}$ varies by ~20 orders of magnitude across the diurnal cycle while $k_{\mathrm{amorph}}$ is essentially constant, the integration uses **operator splitting** between the diurnal cycle (where $k_{\mathrm{cryst}}$ is accumulated as a time average) and the geological epoch (where the analytical D34 solution is applied).

## 2. Thermal Crystallization

The Avrami parameterization (Mitchell et al. 2017, Stage 2):

$$k_{\mathrm{cryst}}^{1/n}(T) = A_0^{1/n} \exp\negthinspace\left(\frac{-\Delta H_a}{R\thinspaceT}\right) \tag{1}$$

$$k_{\mathrm{cryst}}(T) \equiv k_{\mathrm{eff}}(T) = n\thinspace k_{\mathrm{cryst}}^{1/n}(T) \tag{2}$$

with $A_0^{1/n}=3.9\times10^{19}\thinspace\mathrm{s^{-1}}$, $\Delta H_a=62\thinspace\mathrm{kJ\thinspacemol^{-1}}$, $R=8.314\thinspace\mathrm{J\thinspacemol^{-1}\thinspaceK^{-1}}$.

The Avrami exponent $n$ is a function of porosity $\varphi$ — compact ice nucleates volumetrically (3-D growth, $n_{\mathrm{low}}=4.0$); porous ice is dominated by 1-D growth along pore walls ($n_{\mathrm{high}}=1.5$):

$$n(\varphi) = n_{\mathrm{low}} + (n_{\mathrm{high}} - n_{\mathrm{low}})\thinspace\frac{\varphi}{\varphi_{\max}}, \qquad \varphi_{\max}=0.7 \tag{3}$$

We adopt $\varphi_{\mathrm{eq}}=0.50$ at the equator and $\varphi_{\mathrm{pol}}=0.20$ at the poles, with linear interpolation in $|\phi_{\mathrm{lat}}|$ — consistent with the porosity range derived from ALMA disk-resolved millimeter brightness temperatures (Thelen et al. 2024).

**Order-of-magnitude calibration** (compact ice, $n=4$):

| $T$ (K) | $k_{\mathrm{eff}}$ (s⁻¹) | $\tau_{\mathrm{cryst}}$ |
|---:|---:|---|
| 80  | $5.1\times10^{-21}$ | $> 10^{12}\thinspace\mathrm{yr}$ |
| 100 | $6.4\times10^{-13}$ | $5\times10^{4}\thinspace\mathrm{yr}$ |
| 110 | $5.6\times10^{-10}$ | $56\thinspace\mathrm{yr}$ |
| 120 | $1.6\times10^{-7}$ | $72\thinspace\mathrm{d}$ |
| 130 | $1.9\times10^{-5}$ | $15\thinspace\mathrm{hr}$ |
| 140 | $1.1\times10^{-3}$ | $15\thinspace\mathrm{min}$ |

The variation of $k_{\mathrm{cryst}}$ across the diurnal cycle is therefore enormous, which motivates the Jensen-correct averaging in §5.

## 3. Radiation-Induced Amorphization

Magnetospheric ion bombardment deposits energy that disrupts crystalline order:

$$k_{\mathrm{amorph}} = \frac{\dot{D}}{E_{\mathrm{amorph}}} \tag{4}$$

with $E_{\mathrm{amorph}}=3.0\thinspace\mathrm{eV\thinspacemolec^{-1}}$ (Famá et al. 2010; Leto & Baratta 2003). The local energy deposition rate $\dot{D}$ [eV molec⁻¹ s⁻¹] uses the Nordheim et al. (2022) energetic-ion flux topology:

$$\dot{D}(\phi_{\mathrm{lat}},\phi_{\mathrm{lon}}) = \dot{D}_0\thinspace f^{(D)}_{\mathrm{lon}}\thinspace f^{(D)}_{\mathrm{lat}} \tag{5}$$

$$f^{(D)}_{\mathrm{lon}} = \frac{1}{C_D} + \left(1-\frac{1}{C_D}\right)\max\negthinspace\left(0,\ \frac{\cos(\phi_{\mathrm{lon}} - 270°)+1}{2}\right) \tag{6}$$

$$f^{(D)}_{\mathrm{lat}} = 1 + (E_D-1)\left(\frac{|\phi_{\mathrm{lat}}|}{90°}\right)^{1.5} \tag{7}$$

with $C_D=5.0$ (trailing/leading dose contrast — peak at $\phi_{\mathrm{lon}}=270°$ on the trailing apex), $E_D=2.5$ (polar enhancement). The base rate $\dot{D}_0 = 2.3\times10^{-6}\thinspace\mathrm{eV\thinspacemolec^{-1}\thinspaces^{-1}}$ is set so that $\tau_{\mathrm{amorph}}=E_{\mathrm{amorph}}/\dot{D}_0\approx15\thinspace\mathrm{d}$ at the trailing equator, matching the ~weeks-scale amorphization required for the JWST/NIRSpec observation of mostly-amorphous trailing-equator ice (Cartwright et al. 2025).

**Note** — these spatial parameters differ from those used by the sputtering module (§D.5) despite both deriving from magnetospheric ion bombardment. Sputtering is dominated by ~100 eV thermal plasma with weaker spatial contrast ($C_{\mathrm{hem}}=3$, linear latitude); amorphization is driven by $\sim$keV ions which experience stronger magnetic focusing ($C_D=5$, super-linear latitude). Both follow the parameterizations in Nordheim et al. (2022).

## 4. Equilibrium Crystallinity and the Crystallization Knife-Edge

The analytical solution to $\dot\phi_C = k_{\mathrm{cryst}}(1-\phi_C) - k_{\mathrm{amorph}}\phi_C$ is

$$\phi_C(t) = \phi_{\mathrm{eq}} + \bigl[\phi_C(0) - \phi_{\mathrm{eq}}\bigr]\exp\negthinspace\bigl[-(k_{\mathrm{cryst}}+k_{\mathrm{amorph}})\thinspacet\bigr] \tag{8}$$

$$\phi_{\mathrm{eq}}(T,\dot{D}) = \frac{k_{\mathrm{cryst}}(T)}{k_{\mathrm{cryst}}(T) + k_{\mathrm{amorph}}(\dot{D})} \tag{9}$$

The transition temperature $T^*$ where $\phi_{\mathrm{eq}}=1/2$ (i.e. $k_{\mathrm{cryst}}(T^*)=k_{\mathrm{amorph}}$) follows from (1)–(4):

$$T^* = \frac{\Delta H_a}{R\thinspace\ln\negthinspace\bigl[n\thinspaceA_0^{1/n}\thinspaceE_{\mathrm{amorph}}/\dot{D}\bigr]} \tag{10}$$

For compact ice ($n=4$) at the trailing equator ($\dot{D}=\dot{D}_0$): $T^*\approx 125\thinspace\mathrm{K}$. The leading hemisphere has $\dot{D}\sim\dot{D}_0/5$ which depresses $T^*$ by $\sim 5\thinspace\mathrm{K}$. Both threshold temperatures sit comfortably inside Europa's diurnal $T_{\max}$ range ($\sim$86–132 K), so $\phi_{\mathrm{eq}}$ is a sigmoid in latitude/longitude that pivots over a few kelvin — the "crystallization knife-edge."

| $T$ (K) | $k_{\mathrm{cryst}}$ (s⁻¹) | $\phi_{\mathrm{eq}}$ |
|---:|---:|---:|
| 80  | $5.1\times10^{-21}$ | $\sim 0$ |
| 100 | $6.4\times10^{-13}$ | $\sim 10^{-6}$ |
| 110 | $5.6\times10^{-10}$ | $7\times10^{-4}$ |
| 120 | $1.6\times10^{-7}$ | 0.17 |
| 125 | — | 0.5 |
| 130 | $1.9\times10^{-5}$ | 0.96 |

Trailing equator $k_{\mathrm{amorph}}=7.67\times10^{-7}\thinspace\mathrm{s^{-1}}$.

## 5. Jensen's-Inequality-Corrected Diurnal Averaging

Because $k_{\mathrm{cryst}}\propto\exp(-\Delta H_a/RT)$ is strongly convex in $T$, the diurnal-mean rate is *not* the rate evaluated at the diurnal-mean temperature:

$$\bigl\langle k_{\mathrm{cryst}}(T(t))\bigr\rangle \neq k_{\mathrm{cryst}}\negthinspace\bigl(\langle T(t)\rangle\bigr) \tag{11}$$

By Jensen's inequality the LHS dominates by orders of magnitude near the knife-edge — physically, the few hours per Europan day during which the surface is at $T_{\max}\sim 130\thinspace\mathrm{K}$ contribute almost the entire integrated crystallization advance, even though they are a small fraction of the cycle. Substituting $k_{\mathrm{cryst}}(\langle T\rangle)$ would underestimate the rate by 5–10 orders of magnitude at high latitudes and produce a spuriously amorphous mid-/high-latitude band.

The production code therefore accumulates, during the diurnal-cycle integrator, a per-face running mean

$$\bar{k}_{\mathrm{cryst},i} \equiv \frac{1}{N_t}\sum_{t=1}^{N_t} k_{\mathrm{cryst}}\negthinspace\bigl(T_i(t),\thinspace n(\varphi_i)\bigr) \tag{12}$$

and feeds it to the epoch-level analytical update:

$$\phi_{C,i}(\Delta t_{\mathrm{epoch}}) = \phi_{\mathrm{eq},i} + \bigl[\phi_{C,i}(0)-\phi_{\mathrm{eq},i}\bigr]\thinspace\exp\negthinspace\bigl[-(\bar{k}_{\mathrm{cryst},i}+k_{\mathrm{amorph},i})\thinspace\Delta t_{\mathrm{epoch}}\bigr] \tag{13}$$

with $\phi_{\mathrm{eq},i}=\bar{k}_{\mathrm{cryst},i}/(\bar{k}_{\mathrm{cryst},i}+k_{\mathrm{amorph},i})$. This preserves the analytical D34 form while remaining unbiased against full diurnal time-stepping at the order of $10^{-2}$ in $\phi_C$ — verified against a Python reference integrator (`scripts/jwst/crystallization.py`) at all 45 regional Europa runs.

The `MultitimescaleEvolution::DiurnalAverage` accumulator carries the field `avg_crystallization_rate` for this purpose; the epoch update calls `IceCrystallizationModel::evolveAllFacesWithRate(...)` rather than the simpler temperature-based path.

## 6. Depositional Amorphization

H₂O molecules redepositing from the exosphere arrive with kinetic energies set by the source-face temperature ($\sim k_B T \ll$ glass-transition energy) and freeze in amorphous configuration. When mass $\Delta m_{\mathrm{dep}}$ deposits on a face holding existing ice mass $m_{\mathrm{old}}$ at fraction $\phi_{C,\mathrm{old}}$,

$$\phi_{C,\mathrm{new}} = \phi_{C,\mathrm{old}}\thinspace\frac{m_{\mathrm{old}}}{m_{\mathrm{old}} + \Delta m_{\mathrm{dep}}} \tag{14}$$

This dilution is applied at the same epoch cadence as (13) and is the third arm of the competition that dominates polar regions where deposition flux exceeds both $k_{\mathrm{cryst}}$ and $k_{\mathrm{amorph}}$.

## 7. Coupling with the Multi-Timescale Loop

Within one operator-splitting epoch ($N_c$ diurnal cycles, default $N_c=100$), the workflow is:

1. **Diurnal pass.** Step the surface temperature for $N_t$ sub-steps of one $P_{\mathrm{diurnal}}=3.5512\thinspace\mathrm{d}$ cycle. At each sub-step accumulate $k_{\mathrm{cryst}}(T_i)$ into $\bar{k}_{\mathrm{cryst},i}$ (12), and accumulate $\bar{\Phi}_{\mathrm{sub},i}$, $\bar{\Phi}_{\mathrm{dep},i}$, $\bar{\Phi}_{\mathrm{sput},i}$.
2. **Epoch update.** Apply (13) with $\Delta t_{\mathrm{epoch}}=N_c P_{\mathrm{diurnal}}$. Apply (14) with $\Delta m_{\mathrm{dep},i}=\bar{\Phi}_{\mathrm{dep},i}\thinspace\Delta t_{\mathrm{epoch}}$.
3. **Albedo–composition feedback.** If the non-ice concentration $c_i$ has shifted, recompute the surface Bond albedo and rescale $T(t)$ (Section D.6.4); refresh the diurnal cache on the next epoch.

The analytical form keeps the per-face update O(1) regardless of $\Delta t_{\mathrm{epoch}}$, allowing $\sim 10^3$ yr integrations to converge to the steady state in seconds of wall-clock time on the ico-6 mesh.

## 8. Implementation

| Component | File | Notes |
|---|---|---|
| Avrami $k_{\mathrm{cryst}}$, $k_{\mathrm{amorph}}$, dose map, depositional dilution | `src/cpp/exosphere/ice_crystallization_model.h` | Header-only; SIMD-friendly per-face loops |
| Time-averaged $\bar{k}_{\mathrm{cryst}}$ accumulator | `src/cpp/exosphere/multitimescale_evolution.h` (`DiurnalAverage::avg_crystallization_rate`) | Reset at start of each diurnal cycle |
| Epoch update with $\bar{k}_{\mathrm{cryst}}$ | `IceCrystallizationModel::evolveAllFacesWithRate(...)` | Avoids Jensen bias |
| Diurnal accumulation hook | `ptm_exosphere_integration.cpp` (search for `avg_crystallization_rate`) | Called inside the per-step loop |
| Python reference integrator | `scripts/jwst/crystallization.py` | Used for validating the C++ run; same parameter set |
| Forward-model figures | `scripts/jwst/predict_phi_eq.py`, `plot_sigmoid_theory.py`, `plot_phi_C_map.py`, `plot_timescales.py` | Generate the headline 1-D and 2-D maps |

## 9. Limitations and Open Questions

1. **Single-mechanism kinetics.** $\Delta H_a$ and $A_0^{1/n}$ are global constants. Mitchell et al. (2017) report distinct stages with different parameters; only Stage 2 is used here.
2. **Steady-state radiation environment.** $\dot{D}$ is fixed in time; we ignore magnetospheric variability driven by Io's torus loading or storm activity. The order-of-magnitude calibration against JWST is robust to this; finer comparisons are not.
3. **Skin-depth assumption.** Both crystallization and amorphization are assumed uniform in the thermally active skin layer. The 3.1 µm Fresnel-peak probe depth is $\sim 10\thinspace\mathrm{\mu m}$, finer than the typical 5 cm thermal skin — a depth-resolved $\phi_C(z)$ post-processor (radiation-amorphized "skin" over thermally-controlled interior) explains why exposed crystalline ice survives despite ~15-day surface amorphization. This is treated as a forward-model overlay rather than evolved in the C++ solver.
4. **Single porosity per latitude.** Real porosity varies face-to-face with terrain age and roughness; we treat it as a smooth function of latitude only.
5. **No salt or hydrate chemistry.** Cartwright et al. (2025) report hydrated salts at Tara/Powys, which the present module does not represent.

## 10. References

- Cartwright, R. J. et al. (2025) "Crystalline Water Ice on Europa's Surface." *Planet. Sci. J.* **6**, 125. doi:10.3847/PSJ/adba5b
- Famá, M., Loeffler, M. J., Raut, U., Baragiola, R. A. (2010). "Radiation-induced amorphization of crystalline ice." *Icarus* **207**, 314–319.
- Leto, G., Baratta, G. A. (2003). "Ly-α photon induced amorphization of Ic water ice at 16 K." *A&A* **397**, 7–13.
- Mitchell, E. H., Raut, U., Teolis, B. D., Baragiola, R. A. (2017). "Porosity effects on crystallization kinetics of amorphous solid water: implications for cold icy bodies." *Icarus* **285**, 291–299.
- Nordheim, T. A. et al. (2022). "Magnetospheric ion bombardment of Europa's surface." *Planet. Sci. J.* **3**, 5. doi:10.3847/PSJ/ac382a
- Thelen, A. E., Cordiner, M. A., Sykes, M. V., Rivkin, A. S. (2024). ALMA-derived disk-resolved porosity / thermal-inertia maps of Europa.
