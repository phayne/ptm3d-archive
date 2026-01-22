Based on the code provided, I have analyzed your GPU-based radiosity solver. While the implementation correctly follows the mathematical formulation of the radiosity method (Jacobi iteration), there are significant architectural and kernel-level inefficiencies that will bottleneck performance, particularly for the large, sparse matrices typical of topographic radiation exchange.

The following analysis focuses on the OpenCL implementation and its integration into the PTM3D model.

### **Executive Summary**

The current solver uses a "Scalar CSR" (Compressed Sparse Row) approach for matrix-vector multiplication (SpMV). This is the standard textbook implementation but is notoriously inefficient on GPUs due to **warp divergence** and **uncoalesced memory access**. Additionally, the convergence checking mechanism likely introduces excessive PCIe latency synchronization overhead.

I have drafted a technical feedback document for your engineering team below.

# ---

**Technical Feedback: PTM3D GPU Radiosity Solver**

To: Engineering Team (PTM3D)  
From: Code Review  
Subject: Optimization & Critical Analysis of radiosity\_opencl Module

## **1\. Critical Performance Bottlenecks**

### **1.1. Inefficient SpMV Kernel (Scalar CSR)**

The current spmv\_kernel in radiosity\_kernels.cl assigns one thread per row (one face).

C

// Current Implementation  
\_\_kernel void spmv\_kernel(...) {  
    int row \= get\_global\_id(0);  
    // ...  
    for (int j \= row\_start; j \< row\_end; \++j) {  
        sum \+= values\[j\] \* x\[col\_indices\[j\]\];  
    }  
    // ...  
}

**The Problem:**

* **Load Imbalance (Warp Divergence):** In topographic meshes, some faces see many other faces (peaks), while others see few (valleys). In a GPU warp (32 threads), if Thread A has 100 iterations and Thread B has 5, Thread B sits idle for 95 cycles. This "Scalar" approach destroys parallelism.  
* **Uncoalesced Memory Access:** The read x\[col\_indices\[j\]\] is an indirect gathering operation. Since col\_indices are determined by view factors, adjacent threads will access widely separated memory locations in x, leading to poor cache hit rates and memory bandwidth saturation.

**Recommendation:**

* **Switch to CSR-Vector or CSR-Stream:** Assign a "warp" (or partial warp) to process a single row. This allows parallel reduction of the row's dot product and ensures the warp retires together.  
* **Investigate ELLPACK Format:** If the number of view factors per face is relatively uniform (e.g., limited by a horizon cutoff), the ELLPACK (ELL) format pads rows to the same length, allowing perfectly coalesced memory access and removing divergence entirely.

### **1.2. Convergence Checking Latency**

The solver appears to check for convergence (calculating the residual $|B\_{new} \- B\_{old}|$) at *every* iteration by transferring data or partial reduction results from GPU to CPU.

**The Problem:**

* **PCIe Overhead:** The latency of clEnqueueReadBuffer and the CPU/GPU synchronization (implied by the host-side while loop) is often orders of magnitude higher than the kernel execution time itself.  
* Running 100 iterations with 100 synchronizations effectively serializes the GPU dispatch.

**Recommendation:**

* **Asynchronous Check:** Only check convergence every $K$ iterations (e.g., every 10 or 20 steps).  
* **GPU-Side Predicate:** Use a device-side flag or persistent kernel approach where the loop remains on the GPU until convergence is met, returning only the final result.

## **2\. Algorithmic & Precision Issues**

### **2.1. "Warm Starting" the Solver**

The current snippets do not explicitly show the solve() entry point, but standard implementations often initialize the radiosity vector $B$ to thermal emission $E$ or zero.  
Recommendation:

* Ensure the solver uses **Warm Starting**. In a time-series simulation, the radiosity solution for time $t$ is an excellent initial guess for time $t+1$.  
* *Impact:* This typically reduces the required iterations by 50-80%, as the illumination changes gradually.

### **2.2. Precision Mismatch (Float vs. Double)**

* **Observation:** The kernels use float (32-bit), but the host PTM code (ptm\_thermal\_sim.cpp) uses double.  
* **Risk:** If the convergence tolerance is set too tight (e.g., $10^{-7}$), the single-precision solver may never converge due to floating-point noise/quantization, causing it to run for the maximum iteration count every time.  
* **Fix:** Explicitly cap the tolerance at a safe single-precision floor (e.g., $10^{-5}$) or switch to double in kernels (at the cost of 2x performance penalty).

## **3\. Memory & Locality Optimization**

### **3.1. Spatial Memory Locality**

The efficiency of x\[col\_indices\[j\]\] depends heavily on the numbering of the mesh vertices.  
Recommendation:

* Pre-process the mesh using a **Space-Filling Curve (e.g., Hilbert or Morton Order)** to re-index faces. This ensures that faces physically close to each other have IDs close to each other.  
* *Benefit:* This drastically improves L1/L2 cache hit rates during the gather operation, as faces typically exchange radiation with their geometric neighbors.

### **3.2. Sparse Matrix Format**

Large topographic meshes ($\>100k$ faces) result in massive View Factor matrices.  
Recommendation:

* Verify that values (View Factors) are stored in float (4 bytes) rather than double (8 bytes) to halve memory bandwidth usage.  
* If memory is tight, consider **quantization** (storing view factors as 16-bit integers or half-precision floats), as view factors have an inherent uncertainty often larger than FP16 precision.

## **4\. Code Quality Notes**

* **Hardcoded Workgroup Sizes:** RadiosityOpenCLConfig has a default work\_group\_size \= 256\. This should be tunable or queried from the device (CL\_KERNEL\_PREFERRED\_WORK\_GROUP\_SIZE\_MULTIPLE), as AMD and NVIDIA GPUs have different optimal wavefront/warp sizes (64 vs 32).  
* **Error Handling:** Ensure clEnqueueNDRangeKernel error codes are checked in the release build. A silent launch failure (e.g., due to TDR timeout on Windows/Max) will return uninitialized buffers, leading to unphysical thermal results (0K temperatures).

### ---

**Implementation Roadmap**

1. **Immediate:** Implement "Check Every N Iterations" for convergence to reduce PCIe latency.  
2. **Immediate:** Enable "Warm Starting" using the previous timestep's solution.  
3. **Short Term:** Re-index mesh faces using a Hilbert curve before uploading to GPU.  
4. **Long Term:** Replace the spmv\_kernel with a CSR-Vector or ELLPACK implementation.