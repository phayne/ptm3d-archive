This implementation plan outlines the architecture for a high-performance **Iterative Spectral Thermal Model (ISTM)** with coupled terrain radiation. This document is designed for systems architects and software engineers to build the simulation engine.

### **Phase 1: Data Structures & Definition**

Before logic implementation, define the core data structures to ensure memory locality and efficient vectorization.

1.1. The Pixel/Facet Object  
Each surface element (pixel) requires a state object containing:

* **Static Properties:**  
  * Albedo ($\\alpha$), Emissivity ($\\epsilon$).  
  * Thermal Inertia ($\\Gamma$), Density ($\\rho$), Specific Heat ($c\_p$), Conductivity ($k$).  
  * Geothermal Flux ($J\_{geo}$).  
* **Geometric Properties:**  
  * SkyViewFactor ($V\_{sky}$): Portion of sky visible (0.0 to 1.0).  
* **Dynamic State (The Unknowns):**  
  * Temperature\_Profile: Array of length $N\_{time}$ (Floats).  
  * Mean\_Temperature: Scalar (Float).  
  * Net\_Flux\_Profile: Array of length $N\_{time}$ (Floats).

**1.2. The Coupling Matrix (Topology)**

* **View Factor Matrix ($F\_{ij}$):** A sparse matrix or adjacency list defining visibility between pixels.  
  * Row $i$, Column $j$ contains the view factor from Pixel $i$ to Pixel $j$.  
  * *Note:* Ensure $\\sum F\_{ij} \+ V\_{sky} \\approx 1.0$ for energy conservation checks.

**1.3. Time & Frequency Domain Configuration**

* N\_time: Number of time steps per diurnal cycle (e.g., 128, 256). Must be a power of 2 for FFT efficiency.  
* Frequencies: Pre-computed array of angular frequencies $\\omega\_n$ corresponding to the FFT bins.  
* dt: Time step size (Seconds).

### ---

**Phase 2: Pre-Processing Module**

This module runs once before the simulation loops.

**2.1. Solar Flux Generation**

* **Input:** Ephemeris data, surface normals, horizon masks.  
* **Action:** Ray-trace or calculate the direct solar flux $S\_{direct}(t)$ for every pixel for one full diurnal cycle.  
* **Output:** Solar\_Flux\_Array of size $\[N\_{pixels}, N\_{time}\]$.  
  * *Constraint:* Must account for shadows (flux \= 0\) accurately.

**2.2. Spectral Admittance Calculation**

* Since material properties are constant (in this version), pre-calculate the "Regolith Impedance" for each frequency.  
* **Formula:** $Z\_{regolith}(\\omega) \= \\Gamma \\sqrt{i \\omega}$  
  * *Implementation:* Store as a complex array of size $\[N\_{freqs}\]$.  
  * *Optimization:* If properties vary per pixel, store as $\[N\_{pixels}, N\_{freqs}\]$.

### ---

**Phase 3: The "Outer Loop" (Radiation Coupling)**

This loop handles the communication between pixels. It solves the non-linear "Terrain IR" term.

**3.1. Initialization**

* Initialize all pixel temperatures $T(t)$ to a reasonable guess (e.g., the local radiative equilibrium temperature based on mean solar flux).

**3.2. Iteration Logic (Repeat until global convergence)**

1. **Calculate Terrain Irradiance (Vectorized over time):**  
   * For every pixel $i$, calculate the incoming IR flux from neighbors at every time step $t$.  
   * **Simple Exchange:** $Q\_{IR, i}(t) \= \\epsilon \\sigma \\sum\_{j} F\_{i \\to j} (T\_j(t))^4$  
   * **Advanced (Multi-bounce Radiosity):** If high fidelity is required, solve the linear system $(I \- \\rho F) \\cdot J \= \\epsilon \\sigma T^4$ at each time step.  
2. **Construct Total Effective Flux:**  
   * $Q\_{total, i}(t) \= (1 \- \\text{Albedo}) \\cdot S\_{direct, i}(t) \+ \\epsilon \\cdot Q\_{IR, i}(t)$  
3. **Launch Spectral Solver (The Inner Loop):**  
   * Pass $Q\_{total, i}(t)$ and the current $T\_{mean, i}$ to the 1D Spectral Solver (Phase 4).  
   * *Note:* This step is perfectly parallel. Each pixel can be solved by a different CPU thread/GPU kernel.  
4. **Convergence Check:**  
   * Compare the maximum change in temperature ($\\Delta T\_{max}$) across all pixels against a tolerance (e.g., 0.1 K).  
   * If $\\Delta T\_{max} \> \\text{Tolerance}$, update $T(t)$ and repeat Phase 3.2.

### ---

**Phase 4: The "Inner Loop" (1D Spectral Solver)**

This is the physics engine that equilibrates the subsurface instantly. It runs independently for each pixel.

**4.1. Fast Fourier Transform (FFT)**

* Perform a Real-input FFT (rFFT) on the Total Effective Flux ($Q\_{total}$).  
* **Result:** Complex array $\\hat{Q}\_{flux}$ representing flux harmonics.

4.2. Linearization Iteration  
Since surface emission ($T^4$) is non-linear, iterate locally to find the correct "Radiative Conductivity."

1. **Calculate Linearization Parameter ($h\_{rad}$):**  
   * $h\_{rad} \= 4 \\epsilon \\sigma (\\bar{T}\_{guess})^3$  
   * This represents the gradient of the Planck curve at the current mean temperature.  
2. **Apply Spectral Transfer Function:**  
   * For each frequency harmonic $n$:

     $$\\hat{T}\_n \= \\frac{\\hat{Q}\_{flux, n}}{Z\_{regolith}(\\omega\_n) \+ h\_{rad}}$$  
   * *Edge Case:* For the DC component ($n=0$, $\\omega=0$), handle separately or ensure $Z(0)=0$. Usually, DC is handled by the energy balance step below.  
3. **Inverse FFT (Reconstruction):**  
   * Perform Inverse rFFT on $\\hat{T}\_n$ to get the time-domain perturbation $\\Delta T(t)$.  
   * Construct full profile: $T\_{temp}(t) \= \\bar{T}\_{guess} \+ \\Delta T(t)$.  
4. **Enforce Energy Conservation (DC Component correction):**  
   * Calculate the true non-linear mean emission: $\\bar{F}\_{out} \= \\text{Mean}(\\epsilon \\sigma (T\_{temp}(t))^4)$.  
   * Calculate the true mean input: $\\bar{F}\_{in} \= \\text{Mean}(Q\_{total}) \+ J\_{geo}$.  
   * Update Guess: Adjust $\\bar{T}\_{guess}$ using Newton's method:

     $$\\bar{T}\_{new} \= \\bar{T}\_{old} \+ \\frac{\\bar{F}\_{in} \- \\bar{F}\_{out}}{4 \\epsilon \\sigma \\bar{T}\_{old}^3}$$  
5. **Local Convergence:**  
   * Repeat steps 1-4 until $\\bar{T}\_{guess}$ stabilizes (typically 3-5 iterations).

### ---

**Phase 5: Output & Subsurface Calculation**

Once the Outer Loop converges:

**5.1. Subsurface Propagation (Optional/On-Demand)**

* If the user requests temperature at depth $z$:  
  1. Take the converged surface harmonics $\\hat{T}\_n$.  
  2. Apply depth decay/phase shift: $\\hat{T}\_n(z) \= \\hat{T}\_n \\cdot e^{-z \\sqrt{i\\omega\_n / \\alpha}}$.  
  3. Inverse FFT to get $T(z, t)$.  
  4. Add the geothermal gradient term: $T\_{final}(z, t) \= T(z, t) \+ z \\cdot (J\_{geo} / k)$.

**5.2. Data Serialization**

* Store the results. To minimize storage, save only:  
  * Mean\_Temperature  
  * Harmonic\_Coefficients (Real/Imag arrays)  
  * (This allows reconstruction of the full curve later without storing dense time arrays).

### **Engineer's Cheat Sheet: Common Pitfalls**

1. **FFT Normalization:** Ensure the FFT library (e.g., FFTW, cuFFT, NumPy) normalization matches the math. Standard physics definition usually requires normalizing by $1/N$ during the Inverse Transform.  
2. **The "Square Wave" Ringing:** If the solar flux has sharp shadows (step functions), the FFT will produce "Gibbs Phenomenon" (ringing artifacts).  
   * *Fix:* Apply a "Lanczos" or "Sigma" smoothing factor to high-frequency harmonics before Inverse FFT.  
3. **DC Component ($n=0$):** The spectral division $\\hat{T} \= \\hat{Q}/Z$ fails at $\\omega=0$. Always handle the mean temperature ($\\bar{T}$) separately using the scalar energy balance equation (Step 4.4).  
4. **Emissivity Scaling:** Ensure Solar Flux is absorbed by $(1-\\alpha)$, but Terrain IR Flux is absorbed by $\\epsilon$. Do not mix these scaling factors in the total flux summation.