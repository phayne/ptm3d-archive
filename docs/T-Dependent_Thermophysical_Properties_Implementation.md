This update modifies the implementation plan to account for **Temperature-Dependent Thermophysical Properties** (specifically thermal conductivity $k(T)$ and specific heat $c\_p(T)$).

This is critical for lunar regolith, where conductivity can change by a factor of 2–3 between lunar noon (380 K) and midnight (100 K).

### **Architectural Change: The "Effective Property" Iteration**

The Spectral Method relies on linear superposition, which strictly assumes constant material properties. To reconcile this with $k(T)$ and $c\_p(T)$, we introduce an **Effective Property Loop**.

Instead of using a single static conductivity, we calculate an "Effective Conductivity" ($k\_{eff}$) and "Effective Heat Capacity" ($c\_{p, eff}$) that represent the weighted average behavior of the regolith over the diurnal cycle. We then iterate the spectral solution until these effective properties converge with the resulting temperature profile.

### ---

**Phase 1: Data Structures & Definition (Updated)**

**1.1. The Pixel/Facet Object (Modifications)**

* **Remove:** Static Thermal Inertia, Conductivity, Specific Heat.  
* **Add:** Property Coefficients (Polynomial or lookup table parameters).  
  * Example for Lunar Regolith (Hickson/Hemingway models):  
    * $k(T) \= k\_0 \+ k\_1 T^3$ (Radiative transfer term).  
    * $c\_p(T) \= c\_0 \+ c\_1 T \+ \\dots$  
* **Add:** "Effective" State Variables (The iteration targets):  
  * k\_eff: Scalar (Float).  
  * rho\_cp\_eff: Scalar (Float) \- Volumetric heat capacity.

### ---

**Phase 4: The "Inner Loop" (Spectral Solver with T-Dependence)**

This phase is significantly modified. We now have two nested loops:

1. **Property Loop:** Updates $k\_{eff}$ and $(\\rho c\_p)\_{eff}$ based on the temperature profile.  
2. **Linearization Loop:** Updates $T\_{mean}$ and $h\_{rad}$ (as before).

4.1. The Property Update Algorithm  
Before running the spectral solver, or after a convergence pass, update the effective properties using the current temperature profile estimate $T(t)$.

* Weighted Averaging Strategy:  
  Simply averaging $k(T)$ over time is often insufficient because heat flow is highest when gradients are steepest. A common weighting scheme is an energy-weighted average, but for computational efficiency, a Time-Averaged approach usually suffices for the spectral method if iterated:

  $$k\_{eff} \= \\frac{1}{P} \\int\_{0}^{P} k(T(t)) dt$$  
  $$(\\rho c\_p)\_{eff} \= \\frac{1}{P} \\int\_{0}^{P} \\rho(z) c\_p(T(t)) dt$$

**4.2. Modified Solver Steps**

1. **Initialize:**  
   * Set $k\_{eff}$ and $(\\rho c\_p)\_{eff}$ using the initial guess $T\_{mean}$.  
2. **Calculate Impedance (Dynamic):**  
   * Recalculate the Regolith Impedance using the new effective properties:

     $$Z\_{regolith}(\\omega) \= \\sqrt{k\_{eff} \\cdot (\\rho c\_p)\_{eff}} \\cdot \\sqrt{i \\omega}$$  
3. **Run Linearization Loop (Standard Spectral Solve):**  
   * Solve for $\\hat{T}\_n$ and reconstruct $T(t)$ using the current $Z\_{regolith}$.  
   * (This is the same "Step 2 & 3" from the previous plan).  
4. **Update Effective Properties:**  
   * Use the new profile $T(t)$ to calculate the instantaneous properties $k(t)$ and $c\_p(t)$.  
   * Compute new means $k\_{eff, new}$ and $(\\rho c\_p)\_{eff, new}$.  
5. **Check Property Convergence:**  
   * If $|k\_{eff, new} \- k\_{eff, old}| \> \\text{Tolerance}$, update the properties and go back to **Step 2**.

### **Detailed Step-by-Step Logic for Engineers**

**Function:** Solve\_Pixel\_T\_Dependent(Flux\_In, Params)

1. **Initial Guesses:**  
   * $T\_{profile}(t) \\leftarrow$ Constant array of value 250 K.  
   * $k\_{eff} \\leftarrow k(250)$, $C\_{eff} \\leftarrow \\rho c\_p(250)$.  
2. **Start Property Loop (Max 10 iters):**  
   a. Compute Spectral Admittance:  
   \* $\\Gamma\_{eff} \= \\sqrt{k\_{eff} \\cdot C\_{eff}}$  
   \* For each frequency $n$: $Z\_n \= \\Gamma\_{eff} \\sqrt{i \\omega\_n}$  
   b. Start Linearization Loop (Max 5 iters):  
   \* $T\_{mean} \= \\text{mean}(T\_{profile})$  
   \* $h\_{rad} \= 4 \\epsilon \\sigma T\_{mean}^3$  
   \* FFT: $\\hat{F} \= \\text{FFT}(Flux\\\_In)$  
   \* Solve: $\\hat{T}\_n \= \\hat{F}\_n / (Z\_n \+ h\_{rad})$  
   \* Reconstruct: $T\_{pert}(t) \= \\text{iFFT}(\\hat{T}\_n)$  
   \* DC Correction: Solve energy balance for exact $T\_{mean}$.  
   \* $T\_{profile} \= T\_{mean} \+ T\_{pert}(t)$  
   c. Update Properties:  
   \* Compute arrays: $K\_{inst} \= k(T\_{profile})$, $C\_{inst} \= \\rho c\_p(T\_{profile})$.  
   \* $k\_{new} \= \\text{mean}(K\_{inst})$  
   \* $C\_{new} \= \\text{mean}(C\_{inst})$  
   d. Check Convergence:  
   \* If $|k\_{new} \- k\_{eff}| / k\_{eff} \< 0.01$: BREAK  
   \* Else: $k\_{eff} \= k\_{new}$, $C\_{eff} \= C\_{new}$.  
3. **Return:** $T\_{profile}$

### **Engineer's Note: Handling "Conductivity Drift"**

There is a subtle physical issue: **The Mean Temperature at depth is NOT the Mean Temperature at the surface** when $k$ is T-dependent.

* As $k$ increases with T, heat flows *in* easier during the day than it flows *out* at night.  
* This pumps up the deep temperature ("solid-state greenhouse effect").  
* **Implementation Fix:**  
  * In "Phase 4, Step 4 (Energy Balance)," the term $J\_{geo}$ represents the deep flux.  
  * You must ensure the *effective* conductivity used for the background gradient calculation matches the *harmonic* conductivity.  
  * For the DC component specifically, it is safer to define the deep gradient using the harmonic mean of conductivity:

    $$k\_{DC} \= \\left( \\frac{1}{P} \\int \\frac{1}{k(T)} dt \\right)^{-1}$$  
  * Use $k\_{DC}$ when calculating the deep temp offset, but use the arithmetic mean $k\_{eff}$ for the AC skin-depth damping.