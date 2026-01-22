/**
 * @file thermal_opencl.cpp
 * @brief OpenCL thermal solver implementation
 */

#ifdef PTM_GPU_OPENCL

// Define PTM_OPENCL_NO_THROW BEFORE including headers to use non-throwing
// CL_CHECK macro. This avoids SIGILL on macOS when exceptions cross shared
// library boundaries with aggressive optimization.
#define PTM_OPENCL_NO_THROW

#include "thermal_opencl.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>

namespace gpu {
namespace opencl {

// Embedded kernel source (generated from thermal_kernels.cl)
// This avoids needing to distribute a separate .cl file
std::string getEmbeddedKernelSource() {
    return R"CL_SOURCE(
// Embedded OpenCL kernel source for PTM thermal solver
#define MAX_LAYERS 64

#ifndef SIGMA
#define SIGMA 5.67051196e-8f
#endif

inline float compute_k(float T, float kc, float R350) {
    float T3 = T * T * T;
    return kc * (1.0f + R350 * T3);
}

inline float compute_cp(float T, float cp0, __constant float* cp_coeffs, int n_coeffs) {
    // Polynomial: cp = cp0 + T * (c0 + T * (c1 + T * (c2 + ...)))
    // Horner form for coefficients: result = c[n-1]; then result = result*T + c[i] for i=n-2..0
    // Final: cp = cp0 + T * result
    if (n_coeffs == 0) return cp0;
    float result = cp_coeffs[n_coeffs - 1];
    for (int i = n_coeffs - 2; i >= 0; --i) {
        result = result * T + cp_coeffs[i];
    }
    // IMPORTANT: cp = cp0 + T * result, NOT cp0 * result!
    return cp0 + T * result;
}

__kernel void materialUpdateKernel(
    __global const float* T,
    __global float* k,
    __global float* cp,
    __global float* kappa,
    __global const float* rho,
    __global const float* kc,
    float R350,
    __constant float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    int n_faces,
    int n_layers)
{
    int idx = get_global_id(0);
    int total = n_faces * n_layers;
    if (idx >= total) return;

    float T_val = T[idx];
    float kc_val = kc[idx];
    float rho_val = rho[idx];

    float k_val = compute_k(T_val, kc_val, R350);
    float cp_val = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);

    k[idx] = k_val;
    cp[idx] = cp_val;
    kappa[idx] = k_val / (rho_val * cp_val);
}

__kernel void combinedThermalKernel(
    __global float* T_global,
    __global float* k_global,
    __global float* cp_global,
    __global float* kappa_global,
    __global const float* rho,
    __global const float* kc,
    __global const float* flux,
    __global const float* emissivity,
    __global const float* z,
    __global const float* T_deep,
    __global const float* Qb,
    float R350,
    __constant float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    float sigma,
    float dt,
    int n_faces,
    int n_layers)
{
    int face = get_global_id(0);
    if (face >= n_faces) return;
    if (n_layers > MAX_LAYERS) return;

    float T_local[MAX_LAYERS];
    float k_local[MAX_LAYERS];
    float cp_local[MAX_LAYERS];
    float kappa_local[MAX_LAYERS];
    float a[MAX_LAYERS];
    float b[MAX_LAYERS];
    float c[MAX_LAYERS];
    float d[MAX_LAYERS];

    // Load and compute material properties
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        float T_val = T_global[idx];
        float kc_val = kc[idx];
        float rho_val = rho[idx];

        T_local[i] = T_val;
        k_local[i] = compute_k(T_val, kc_val, R350);
        cp_local[i] = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);
        kappa_local[i] = k_local[i] / (rho_val * cp_local[i]);
    }

    float eps = emissivity[face];
    float Q_surf = flux[face];
    float T_bottom = T_deep[face];
    float Q_bottom = Qb[face];

    // Surface layer
    {
        float dz = z[1] - z[0];
        float dz2 = dz * dz;
        float Fo = kappa_local[0] * dt / dz2;
        float T3 = T_local[0] * T_local[0] * T_local[0];
        float rad = 4.0f * eps * sigma * T3 * dz / k_local[0];

        a[0] = 0.0f;
        b[0] = 1.0f + Fo + 0.5f * rad;
        c[0] = -Fo;

        float Q_rad_old = eps * sigma * T_local[0] * T3;
        float rho_0 = rho[face];
        d[0] = (1.0f - Fo) * T_local[0] + Fo * T_local[1]
             + dt * (Q_surf - Q_rad_old) / (rho_0 * cp_local[0] * dz);
    }

    // Interior layers - flux-based formulation matching CPU code
    for (int i = 1; i < n_layers - 1; ++i) {
        float dz_minus = z[i] - z[i - 1];
        float dz_plus = z[i + 1] - z[i];
        float dz_avg = 0.5f * (dz_minus + dz_plus);

        // Grid coefficients (matching CPU g1, g2 formulation)
        float g1 = 1.0f / (dz_minus * dz_avg);  // coupling to i-1
        float g2 = 1.0f / (dz_plus * dz_avg);   // coupling to i+1

        // Flux-based: use k[i-1] for lower interface, k[i] for upper
        float alpha = g1 * k_local[i - 1];
        float beta = g2 * k_local[i];
        float inv_rhocp = dt / (rho[i * n_faces + face] * cp_local[i]);

        a[i] = -0.5f * inv_rhocp * alpha;
        b[i] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
        c[i] = -0.5f * inv_rhocp * beta;

        // RHS: Crank-Nicolson explicit part
        d[i] = T_local[i] + 0.5f * inv_rhocp * (
            alpha * T_local[i - 1]
            - (alpha + beta) * T_local[i]
            + beta * T_local[i + 1]);
    }

    // Bottom layer
    {
        int i = n_layers - 1;
        int idx = i * n_faces + face;
        float dz = z[i] - z[i - 1];
        float dz2 = dz * dz;
        float Fo = kappa_local[i] * dt / dz2;

        a[i] = -0.5f * Fo;
        b[i] = 1.0f + 0.5f * Fo;
        c[i] = 0.0f;

        d[i] = 0.5f * Fo * T_local[i - 1]
             + (1.0f - 0.5f * Fo) * T_local[i]
             + 0.5f * Fo * T_bottom;

        if (Q_bottom != 0.0f) {
            float rho_n = rho[idx];
            d[i] += dt * Q_bottom / (rho_n * cp_local[i] * dz);
        }
    }

    // Thomas algorithm
    for (int i = 1; i < n_layers; ++i) {
        float m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    T_local[n_layers - 1] = d[n_layers - 1] / b[n_layers - 1];
    for (int i = n_layers - 2; i >= 0; --i) {
        T_local[i] = (d[i] - c[i] * T_local[i + 1]) / b[i];
    }

    // Write back
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_global[idx] = T_local[i];
        k_global[idx] = k_local[i];
        cp_global[idx] = cp_local[i];
        kappa_global[idx] = kappa_local[i];
    }
}

// Persistent thermal kernel - processes ALL timesteps on GPU
// Uses internal sub-stepping to maintain numerical stability when output dt is large
__kernel void persistentThermalKernel(
    __global float* T_global,
    __global float* k_global,
    __global float* cp_global,
    __global float* kappa_global,
    __global const float* rho,
    __global const float* kc,
    __global const float* flux_all,
    __global float* output_temps,
    __global const float* emissivity,
    __global const float* z,
    __global const float* T_deep,
    __global const float* Qb,
    float R350,
    __constant float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    float sigma,
    float dt_output,  // OUTPUT timestep (may be large, e.g., hours)
    int n_faces,
    int n_layers,
    int n_timesteps)
{
    int face = get_global_id(0);
    if (face >= n_faces) return;
    if (n_layers > MAX_LAYERS) return;

    float T_local[MAX_LAYERS];
    float k_local[MAX_LAYERS];
    float cp_local[MAX_LAYERS];
    float kappa_local[MAX_LAYERS];
    float a[MAX_LAYERS];
    float b[MAX_LAYERS];
    float c_coef[MAX_LAYERS];
    float d[MAX_LAYERS];
    float kc_local[MAX_LAYERS];
    float rho_local[MAX_LAYERS];

    // Load initial state
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_local[i] = T_global[idx];
        kc_local[i] = kc[idx];
        rho_local[i] = rho[idx];
    }

    float eps = emissivity[face];
    float Q_bottom = Qb[face];  // Interior heat flux (W/m²)

    // Compute stable timestep based on von Neumann criterion (Fo < 0.5)
    // Need to compute kappa first to get stable dt
    // Use initial temperature to estimate kappa
    float max_kappa = 0.0f;
    float min_dz2 = 1e10f;
    for (int i = 0; i < n_layers; ++i) {
        float k_val = compute_k(T_local[i], kc_local[i], R350);
        float cp_val = compute_cp(T_local[i], cp0, cp_coeffs, n_cp_coeffs);
        if (cp_val <= 0.0f) cp_val = 800.0f;
        float kappa_val = k_val / (rho_local[i] * cp_val);
        if (kappa_val > max_kappa) max_kappa = kappa_val;

        if (i > 0) {
            float dz = z[i] - z[i-1];
            float dz2 = dz * dz;
            if (dz2 < min_dz2) min_dz2 = dz2;
        }
    }

    // Stable dt: Fo = kappa * dt / dz^2 < 0.49
    float dt_stable = 0.49f * min_dz2 / max_kappa;

    // Clamp dt_stable to reasonable range
    dt_stable = fmax(1.0f, fmin(dt_stable, dt_output));  // At least 1 second, at most output dt

    // Number of sub-steps per output interval
    int n_substeps = (int)ceil(dt_output / dt_stable);
    float dt = dt_output / (float)n_substeps;  // Actual sub-step size

    // Process ALL output timesteps
    for (int t = 0; t < n_timesteps; ++t) {
        float Q_surf = flux_all[t * n_faces + face];

        // Sub-stepping loop for numerical stability
        for (int substep = 0; substep < n_substeps; ++substep) {
            // Material properties (updated each sub-step)
            for (int i = 0; i < n_layers; ++i) {
                float T_val = T_local[i];
                k_local[i] = compute_k(T_val, kc_local[i], R350);
                cp_local[i] = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);
                // Guard against invalid cp values (e.g., negative from polynomial at extreme T)
                if (cp_local[i] <= 0.0f || !isfinite(cp_local[i])) {
                    cp_local[i] = 800.0f;  // Fallback to reasonable value
                }
                kappa_local[i] = k_local[i] / (rho_local[i] * cp_local[i]);
            }

            // Step 1: Newton-Raphson for surface temperature (matches CPU surfTemp)
            // Energy balance at surface: Qs + k*(T1-T0)/dz = eps*sigma*T0^4
            {
                float dz = z[1] - z[0];
                float k0 = k_local[0];
                float T1 = T_local[1];

                // Initial guess - MUST match CPU surfTemp() exactly!
                // CPU uses: sqrt(sqrt(Qs / (eps * sigma))) when Qs > 0, else T1
                float T_surf;
                if (Q_surf > 0.0f) {
                    // Equilibrium temperature as initial guess (matching CPU)
                    // sqrt(sqrt(x)) = x^0.25 but more accurate
                    float x = Q_surf / (eps * sigma);
                    T_surf = native_sqrt(native_sqrt(x));
                } else {
                    // Night side: use subsurface temperature as initial guess (matching CPU)
                    T_surf = T1;
                }

                // Newton-Raphson iterations with early termination
                // Matches CPU: max_iter=10, tolerance=0.001, relaxation=0.8
                float dT_nr = 1.0f;
                for (int iter = 0; iter < 10 && fabs(dT_nr) > 0.001f; ++iter) {
                    float T0 = T_surf;
                    float T3 = T0 * T0 * T0;
                    float T4 = T0 * T3;

                    // f = Qs + k*(T1 - T0)/dz - eps*sigma*T0^4
                    float f = Q_surf + k0 * (T1 - T0) / dz - eps * sigma * T4;

                    // df/dT0 = -k/dz - 4*eps*sigma*T0^3
                    float df = -k0 / dz - 4.0f * eps * sigma * T3;

                    if (fabs(df) < 1e-10f) df = -1e-10f;

                    // dT = relaxation * f / df, then T_new = T0 - dT
                    dT_nr = 0.8f * f / df;
                    T_surf = T0 - dT_nr;

                    // Clamp to physical range (matching CPU MIN_TEMP=3.0, MAX_TEMP=500.0)
                    T_surf = fmax(3.0f, fmin(500.0f, T_surf));
                }

                T_local[0] = T_surf;

                // CRITICAL: Recompute surface material properties using NEW surface temperature
                // This matches CPU order: surfTemp() → material refresh → Thomas solve
                // Without this, k_local[0] would use OLD T[0] and the coupling between
                // layers 0 and 1 in the tridiagonal system would be wrong.
                k_local[0] = compute_k(T_surf, kc_local[0], R350);
                cp_local[0] = compute_cp(T_surf, cp0, cp_coeffs, n_cp_coeffs);
                if (cp_local[0] <= 0.0f || !isfinite(cp_local[0])) {
                    cp_local[0] = 800.0f;
                }
                kappa_local[0] = k_local[0] / (rho_local[0] * cp_local[0]);
            }

            // Step 2: Apply bottom flux boundary condition (matches CPU botTemp)
            // CPU does: T[n] = T[n-1] + Qb * dz / k
            // This is applied BEFORE the interior solve, using OLD T[n-1]
            {
                int n = n_layers - 1;
                float dz_bottom = z[n] - z[n - 1];
                float k_bottom = k_local[n];
                // Flux BC: ensures heat flux at bottom equals Qb (interior heat flux)
                T_local[n] = T_local[n - 1] + Q_bottom * dz_bottom / k_bottom;
            }

            // Step 3: Interior diffusion using Crank-Nicolson (excluding surface and bottom)
            // Build tridiagonal system for layers 1 to n_layers-2

            // Layer 1 (next to surface) - surface T[0] is known from Newton-Raphson
            // IMPORTANT: CPU code copies T AFTER surfTemp(), so T_old[0] = T_new[0]
            // We must use T_local[0] (new surface temp) for BOTH explicit and implicit parts
            int n_int = n_layers - 2;  // Interior node count (excluding surface and bottom)

            if (n_layers > 2) {
                float dz_minus = z[1] - z[0];
                float dz_plus = z[2] - z[1];
                float dz_avg = 0.5f * (dz_minus + dz_plus);

                float g1 = 1.0f / (dz_minus * dz_avg);
                float g2 = 1.0f / (dz_plus * dz_avg);

                float alpha = g1 * k_local[0];
                float beta = g2 * k_local[1];
                float inv_rhocp = dt / (rho_local[1] * cp_local[1]);

                // a[1] couples to T[0] which is now known - move to RHS
                a[1] = 0.0f;
                b[1] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
                c_coef[1] = -0.5f * inv_rhocp * beta;

                // RHS: Matches CPU exactly - both explicit and implicit use NEW surface temp
                // CPU does: d[0] = T_old[i] + 0.5*inv_rhocp*(alpha*T_old[i-1] - ...)
                // then:     d[0] -= a[0] * Tptr[0]  which adds +0.5*inv_rhocp*alpha*Tptr[0]
                // Since T_old[0] = Tptr[0] = T_new (surfTemp called before copy), total = inv_rhocp*alpha*T0_new
                d[1] = T_local[1] + inv_rhocp * alpha * T_local[0]
                     - 0.5f * inv_rhocp * (alpha + beta) * T_local[1]
                     + 0.5f * inv_rhocp * beta * T_local[2];
            }

            // Interior layers 2 to n_layers-2 (EXCLUDING bottom layer)
            for (int i = 2; i < n_layers - 1; ++i) {
                float dz_minus = z[i] - z[i - 1];
                float dz_plus = z[i + 1] - z[i];
                float dz_avg = 0.5f * (dz_minus + dz_plus);

                float g1 = 1.0f / (dz_minus * dz_avg);
                float g2 = 1.0f / (dz_plus * dz_avg);

                float alpha = g1 * k_local[i - 1];
                float beta = g2 * k_local[i];
                float inv_rhocp = dt / (rho_local[i] * cp_local[i]);

                a[i] = -0.5f * inv_rhocp * alpha;
                b[i] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
                c_coef[i] = -0.5f * inv_rhocp * beta;

                d[i] = T_local[i] + 0.5f * inv_rhocp * (
                    alpha * T_local[i - 1]
                    - (alpha + beta) * T_local[i]
                    + beta * T_local[i + 1]);
            }

            // Fold bottom boundary into last interior node's RHS
            // CPU does: d[n_int-1] -= c[n_int-1] * Tptr[n_tot - 1]
            // Our last interior node is at layer n_layers-2
            // T[n_layers-1] was set by flux BC above
            if (n_int > 0) {
                int last_int = n_layers - 2;  // Last interior layer
                d[last_int] -= c_coef[last_int] * T_local[n_layers - 1];
                c_coef[last_int] = 0.0f;  // No longer couples upward
            }

            // Thomas algorithm for interior only (layers 1 to n_layers-2)
            // Layer 0 is already solved via Newton-Raphson
            // Layer n_layers-1 is set by flux BC (computed from T[n_layers-2])
            for (int i = 2; i < n_layers - 1; ++i) {
                float m = a[i] / b[i - 1];
                b[i] = b[i] - m * c_coef[i - 1];
                d[i] = d[i] - m * d[i - 1];
            }

            // Back substitution for interior only (layers 1 to n_layers-2)
            // Start from last interior node, not the bottom boundary
            if (n_int > 0) {
                int last_int = n_layers - 2;
                T_local[last_int] = d[last_int] / b[last_int];
                for (int i = last_int - 1; i >= 1; --i) {
                    T_local[i] = (d[i] - c_coef[i] * T_local[i + 1]) / b[i];
                }
            }

            // Note: Bottom layer T[n_layers-1] was already set by flux BC before solve
            // and remains unchanged during interior solve (not part of tridiagonal system)

            // Physical clamping for ALL layers (matching CPU solver)
            // This prevents numerical instabilities from causing runaway temperatures
            for (int i = 0; i < n_layers; ++i) {
                if (!isfinite(T_local[i])) {
                    // NaN/Inf guard: reset to reasonable temperature
                    T_local[i] = 200.0f;  // Safe fallback value
                }
                T_local[i] = fmax(3.0f, fmin(1000.0f, T_local[i]));
            }
        }  // End of sub-stepping loop

        // Store surface temperature after all sub-steps are complete
        output_temps[t * n_faces + face] = T_local[0];
    }  // End of output timestep loop

    // Write final state back
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_global[idx] = T_local[i];
        k_global[idx] = k_local[i];
        cp_global[idx] = cp_local[i];
        kappa_global[idx] = kappa_local[i];
    }
}
)CL_SOURCE";
}

bool isOpenCLAvailable() {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    return (err == CL_SUCCESS && num_platforms > 0);
}

void printOpenCLInfo(std::ostream& out) {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);

    if (err != CL_SUCCESS || num_platforms == 0) {
        out << "No OpenCL platforms found.\n";
        return;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    out << "OpenCL Platforms: " << num_platforms << "\n";

    for (cl_uint p = 0; p < num_platforms; ++p) {
        char buffer[256];
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(buffer), buffer, nullptr);
        out << "\nPlatform " << p << ": " << buffer << "\n";

        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, nullptr);
        out << "  Vendor: " << buffer << "\n";

        clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, sizeof(buffer), buffer, nullptr);
        out << "  Version: " << buffer << "\n";

        // Get devices
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

            for (cl_uint d = 0; d < num_devices; ++d) {
                clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
                out << "  Device " << d << ": " << buffer << "\n";

                cl_device_type type;
                clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
                out << "    Type: " << ((type & CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU") << "\n";

                cl_ulong mem;
                clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
                out << "    Memory: " << (mem / (1024*1024)) << " MB\n";

                cl_uint units;
                clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr);
                out << "    Compute Units: " << units << "\n";
            }
        }
    }
}

// ThermalOpenCLSolver implementation

ThermalOpenCLSolver::ThermalOpenCLSolver(size_t n_faces, size_t n_layers,
                                         const ThermalOpenCLConfig& config)
    : n_faces_(n_faces)
    , n_layers_(n_layers)
    , config_(config)
{
}

ThermalOpenCLSolver::~ThermalOpenCLSolver() {
    release();
}

bool ThermalOpenCLSolver::initialize() {
    if (initialized_) return true;

    // Create OpenCL context
    context_ = std::make_unique<OpenCLContext>();
    if (!context_->initialize(true)) {  // Prefer GPU
        if (config_.verbose) {
            std::cerr << "Failed to initialize OpenCL context\n";
        }
        return false;
    }

    if (config_.verbose) {
        printDeviceInfo(context_->deviceInfo());
    }

    // Build kernels and create buffers (non-throwing)
    buildKernels();
    createBuffers();

    // Check if critical resources were created successfully
    if (!program_ || !persistent_kernel_ || !buf_T_ || !buf_rho_) {
        std::cerr << "[GPU Thermal] Failed to initialize OpenCL resources\n";
        release();
        return false;
    }

    initialized_ = true;

    if (config_.verbose) {
        std::cout << "ThermalOpenCLSolver initialized: "
                  << n_faces_ << " faces, " << n_layers_ << " layers\n";
    }

    return true;
}

void ThermalOpenCLSolver::buildKernels() {
    std::string source;

    // Try to load from file first
    if (!config_.kernel_path.empty()) {
        std::ifstream file(config_.kernel_path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            source = buffer.str();
        }
    }

    // Fall back to embedded source
    if (source.empty()) {
        source = getEmbeddedKernelSource();
    }

    // Build program directly (avoid context_->buildProgram() which throws exceptions
    // that cause SIGILL on macOS when crossing shared library boundaries)
    const char* src = source.c_str();
    size_t len = source.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_->context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to create program from source: "
                  << clErrorString(err) << "\n";
        program_ = nullptr;
        return;
    }

    std::string options = "-cl-fast-relaxed-math -cl-mad-enable";
    cl_device_id device = context_->device();
    err = clBuildProgram(program_, 1, &device, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                             0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                 log_size, log.data(), nullptr);
            std::cerr << "[GPU Thermal] OpenCL program build failed:\n" << log.data() << "\n";
        } else {
            std::cerr << "[GPU Thermal] OpenCL program build failed: "
                      << clErrorString(err) << "\n";
        }
        clReleaseProgram(program_);
        program_ = nullptr;
        return;
    }

    // Create kernels
    material_kernel_ = clCreateKernel(program_, "materialUpdateKernel", &err);
    CL_CHECK(err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to create materialUpdateKernel\n";
        return;
    }

    combined_kernel_ = clCreateKernel(program_, "combinedThermalKernel", &err);
    CL_CHECK(err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to create combinedThermalKernel\n";
        return;
    }

    persistent_kernel_ = clCreateKernel(program_, "persistentThermalKernel", &err);
    CL_CHECK(err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to create persistentThermalKernel\n";
        return;
    }
}

void ThermalOpenCLSolver::createBuffers() {
    cl_context ctx = context_->context();
    cl_int err;

    size_t total = n_faces_ * n_layers_;

    // Allocate device buffers
    buf_T_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                           total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_k_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                           total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_cp_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                            total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_kappa_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_rho_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                             total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_kc_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                            total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_flux_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                              n_faces_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_emissivity_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    n_faces_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                           n_layers_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_T_deep_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                n_faces_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_Qb_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                            n_faces_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // Placeholder for cp_coeffs (will resize when setMaterialParams is called)
    buf_cp_coeffs_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   sizeof(float), nullptr, &err);
    CL_CHECK(err);
}

void ThermalOpenCLSolver::uploadMeshData(
    const std::vector<float>& z_grid,
    const std::vector<float>& rho,
    const std::vector<float>& kc,
    const std::vector<float>& emissivity,
    const std::vector<float>& T_deep,
    const std::vector<float>& Qb)
{
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    cl_command_queue queue = context_->queue();

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_z_, CL_FALSE, 0,
                                 z_grid.size() * sizeof(float),
                                 z_grid.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_rho_, CL_FALSE, 0,
                                 rho.size() * sizeof(float),
                                 rho.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_kc_, CL_FALSE, 0,
                                 kc.size() * sizeof(float),
                                 kc.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_emissivity_, CL_FALSE, 0,
                                 emissivity.size() * sizeof(float),
                                 emissivity.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_T_deep_, CL_FALSE, 0,
                                 T_deep.size() * sizeof(float),
                                 T_deep.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_Qb_, CL_TRUE, 0,
                                 Qb.size() * sizeof(float),
                                 Qb.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::setMaterialParams(
    float cp0,
    const std::vector<float>& cp_coeffs,
    float R350,
    float sigma)
{
    cp0_ = cp0;
    R350_ = R350;
    sigma_ = sigma;
    n_cp_coeffs_ = static_cast<int>(cp_coeffs.size());

    if (!cp_coeffs.empty() && initialized_) {
        // Recreate cp_coeffs buffer with correct size
        if (buf_cp_coeffs_) {
            clReleaseMemObject(buf_cp_coeffs_);
        }

        cl_int err;
        buf_cp_coeffs_ = clCreateBuffer(context_->context(), CL_MEM_READ_ONLY,
                                       cp_coeffs.size() * sizeof(float),
                                       nullptr, &err);
        CL_CHECK(err);

        CL_CHECK(clEnqueueWriteBuffer(context_->queue(), buf_cp_coeffs_, CL_TRUE, 0,
                                     cp_coeffs.size() * sizeof(float),
                                     cp_coeffs.data(), 0, nullptr, nullptr));
    }
}

void ThermalOpenCLSolver::uploadInitialTemperature(const std::vector<float>& T_initial) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    CL_CHECK(clEnqueueWriteBuffer(context_->queue(), buf_T_, CL_TRUE, 0,
                                 T_initial.size() * sizeof(float),
                                 T_initial.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::uploadFlux(const std::vector<float>& flux) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    CL_CHECK(clEnqueueWriteBuffer(context_->queue(), buf_flux_, CL_TRUE, 0,
                                 flux.size() * sizeof(float),
                                 flux.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::advanceTimestep(float dt) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    auto start = std::chrono::high_resolution_clock::now();

    cl_command_queue queue = context_->queue();
    int n_faces = static_cast<int>(n_faces_);
    int n_layers = static_cast<int>(n_layers_);

    // Set kernel arguments
    int arg = 0;
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_T_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_k_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_cp_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_kappa_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_rho_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_kc_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_flux_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_z_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_T_deep_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_Qb_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(float), &R350_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_cp_coeffs_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(int), &n_cp_coeffs_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(float), &cp0_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(float), &sigma_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(float), &dt));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(int), &n_faces));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(int), &n_layers));

    // Launch kernel
    size_t global_size = ((n_faces_ + config_.work_group_size - 1) /
                          config_.work_group_size) * config_.work_group_size;
    size_t local_size = config_.work_group_size;

    CL_CHECK(clEnqueueNDRangeKernel(queue, combined_kernel_, 1, nullptr,
                                   &global_size, &local_size, 0, nullptr, nullptr));

    // Wait for completion
    CL_CHECK(clFinish(queue));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    last_kernel_time_ms_ = duration.count() / 1000.0f;
}

void ThermalOpenCLSolver::downloadSurfaceTemps(std::vector<float>& T_surface) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    T_surface.resize(n_faces_);

    // Surface temps are at layer 0, which is at indices [0, n_faces)
    CL_CHECK(clEnqueueReadBuffer(context_->queue(), buf_T_, CL_TRUE, 0,
                                n_faces_ * sizeof(float),
                                T_surface.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::downloadAllTemps(std::vector<float>& T_all) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    T_all.resize(n_faces_ * n_layers_);

    CL_CHECK(clEnqueueReadBuffer(context_->queue(), buf_T_, CL_TRUE, 0,
                                T_all.size() * sizeof(float),
                                T_all.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::synchronize() {
    if (context_) {
        context_->finish();
    }
}

void ThermalOpenCLSolver::solvePersistentAllTimesteps(
    const std::vector<float>& all_flux,
    std::vector<float>& output_temps,
    int n_timesteps,
    float dt)
{
    if (!initialized_) {
        std::cerr << "[GPU Thermal] Solver not initialized\n";
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;

    // Validate input size
    size_t expected_flux_size = static_cast<size_t>(n_timesteps) * n_faces_;
    if (all_flux.size() != expected_flux_size) {
        std::cerr << "[GPU Thermal] all_flux size mismatch: expected "
                  << expected_flux_size << ", got " << all_flux.size() << "\n";
        return;
    }

    // Allocate output
    output_temps.resize(expected_flux_size);

    // Create temporary buffers for all flux data and output temperatures
    cl_mem buf_flux_all = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         all_flux.size() * sizeof(float),
                                         const_cast<float*>(all_flux.data()), &err);
    CL_CHECK(err);

    cl_mem buf_output_temps = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                             output_temps.size() * sizeof(float),
                                             nullptr, &err);
    CL_CHECK(err);

    // Set kernel arguments
    int n_faces = static_cast<int>(n_faces_);
    int n_layers = static_cast<int>(n_layers_);

    // Verify kernel is valid
    if (!persistent_kernel_) {
        std::cerr << "[GPU Thermal] ERROR: persistent_kernel_ is null!\n";
        clReleaseMemObject(buf_flux_all);
        clReleaseMemObject(buf_output_temps);
        return;
    }

    int arg = 0;
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_T_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_k_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_cp_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_kappa_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_rho_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_kc_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_flux_all));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_output_temps));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_z_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_T_deep_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_Qb_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &R350_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_cp_coeffs_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_cp_coeffs_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &cp0_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &sigma_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &dt));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_faces));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_layers));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_timesteps));

    // Launch kernel - one work item per face
    size_t global_size = ((n_faces_ + config_.work_group_size - 1) /
                          config_.work_group_size) * config_.work_group_size;
    size_t local_size = config_.work_group_size;

    err = clEnqueueNDRangeKernel(queue, persistent_kernel_, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Kernel launch failed: " << clErrorString(err) << "\n";
    }

    // Wait for kernel completion and check for errors
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Kernel execution failed: " << clErrorString(err) << "\n";
    }

    // Download results
    err = clEnqueueReadBuffer(queue, buf_output_temps, CL_TRUE, 0,
                              output_temps.size() * sizeof(float),
                              output_temps.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to read output buffer: " << clErrorString(err) << "\n";
    }

    // Debug output (only when verbose mode is enabled)
    if (config_.verbose) {
        float min_val = output_temps[0], max_val = output_temps[0];
        int nan_count = 0, zero_count = 0, valid_count = 0;
        for (size_t i = 0; i < std::min(output_temps.size(), (size_t)(n_timesteps * 100)); ++i) {
            float val = output_temps[i];
            if (std::isnan(val)) nan_count++;
            else if (val == 0.0f) zero_count++;
            else {
                valid_count++;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
        std::cerr << "[GPU Thermal DEBUG] output_temps sample (first 100 faces x " << n_timesteps << " timesteps):\n"
                  << "  NaN count: " << nan_count << ", Zero count: " << zero_count
                  << ", Valid count: " << valid_count << "\n"
                  << "  Valid range: " << min_val << " to " << max_val << " K\n";

        // Show first few values for face 0
        std::cerr << "[GPU Thermal DEBUG] Face 0 temps (first 10 timesteps): ";
        for (int t = 0; t < std::min(10, n_timesteps); ++t) {
            std::cerr << output_temps[t * n_faces_ + 0] << " ";
        }
        std::cerr << "\n";

        // Also show later timesteps to track equilibration
        std::cerr << "[GPU Thermal DEBUG] Face 0 temps at later steps: ";
        for (int t : {1000, 2000, 3000, 4000, 5000, 5500, 5800, 5989}) {
            if (t < n_timesteps) {
                std::cerr << "t=" << t << ":" << output_temps[t * n_faces_ + 0] << " ";
            }
        }
        std::cerr << "\n";

        // Check flux values for face 0
        std::cerr << "[GPU Thermal DEBUG] Flux sample for face 0: ";
        for (int t : {0, 100, 200, 500, 1000, 2000, 3000}) {
            if (t < n_timesteps) {
                std::cerr << "t=" << t << ":" << all_flux[t * n_faces + 0] << " ";
            }
        }
        std::cerr << "\n";

        // Count zero flux timesteps
        int zero_flux_count = 0;
        for (int t = 0; t < n_timesteps; ++t) {
            if (all_flux[t * n_faces + 0] == 0.0f) zero_flux_count++;
        }
        std::cerr << "[GPU Thermal DEBUG] Face 0: zero flux timesteps = " << zero_flux_count
                  << " / " << n_timesteps << "\n";

        // Read back final T profile to check interior temperatures
        std::vector<float> T_final(n_faces_ * n_layers_);
        cl_int read_err = clEnqueueReadBuffer(queue, buf_T_, CL_TRUE, 0,
                                               T_final.size() * sizeof(float),
                                               T_final.data(), 0, nullptr, nullptr);
        if (read_err == CL_SUCCESS) {
            std::cerr << "[GPU Thermal DEBUG] Final T profile for face 0 (all layers): ";
            for (size_t layer = 0; layer < n_layers_; ++layer) {
                std::cerr << T_final[layer * n_faces_ + 0] << " ";
            }
            std::cerr << "\n";
        }

        // Debug R350 and other params
        std::cerr << "[GPU Thermal DEBUG] Material params: R350=" << R350_
                  << ", cp0=" << cp0_ << ", sigma=" << sigma_
                  << ", n_cp_coeffs=" << n_cp_coeffs_ << "\n";
    }

    // Clean up temporary buffers
    clReleaseMemObject(buf_flux_all);
    clReleaseMemObject(buf_output_temps);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    last_kernel_time_ms_ = static_cast<float>(duration.count());

    std::cout << "[GPU Thermal] Persistent kernel processed " << n_timesteps << " timesteps for "
              << n_faces_ << " faces in " << last_kernel_time_ms_ << " ms\n";
}

const OpenCLDeviceInfo& ThermalOpenCLSolver::getDeviceInfo() const {
    static OpenCLDeviceInfo empty;
    return context_ ? context_->deviceInfo() : empty;
}

void ThermalOpenCLSolver::release() {
    // Release kernels
    if (material_kernel_) {
        clReleaseKernel(material_kernel_);
        material_kernel_ = nullptr;
    }
    if (thermal_kernel_) {
        clReleaseKernel(thermal_kernel_);
        thermal_kernel_ = nullptr;
    }
    if (combined_kernel_) {
        clReleaseKernel(combined_kernel_);
        combined_kernel_ = nullptr;
    }
    if (persistent_kernel_) {
        clReleaseKernel(persistent_kernel_);
        persistent_kernel_ = nullptr;
    }

    // Release program
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }

    // Release buffers
    if (buf_T_) { clReleaseMemObject(buf_T_); buf_T_ = nullptr; }
    if (buf_k_) { clReleaseMemObject(buf_k_); buf_k_ = nullptr; }
    if (buf_cp_) { clReleaseMemObject(buf_cp_); buf_cp_ = nullptr; }
    if (buf_kappa_) { clReleaseMemObject(buf_kappa_); buf_kappa_ = nullptr; }
    if (buf_rho_) { clReleaseMemObject(buf_rho_); buf_rho_ = nullptr; }
    if (buf_kc_) { clReleaseMemObject(buf_kc_); buf_kc_ = nullptr; }
    if (buf_flux_) { clReleaseMemObject(buf_flux_); buf_flux_ = nullptr; }
    if (buf_emissivity_) { clReleaseMemObject(buf_emissivity_); buf_emissivity_ = nullptr; }
    if (buf_z_) { clReleaseMemObject(buf_z_); buf_z_ = nullptr; }
    if (buf_T_deep_) { clReleaseMemObject(buf_T_deep_); buf_T_deep_ = nullptr; }
    if (buf_Qb_) { clReleaseMemObject(buf_Qb_); buf_Qb_ = nullptr; }
    if (buf_cp_coeffs_) { clReleaseMemObject(buf_cp_coeffs_); buf_cp_coeffs_ = nullptr; }

    // Release context (via unique_ptr)
    context_.reset();

    initialized_ = false;
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
