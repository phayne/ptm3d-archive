#pragma once

/**
 * @file opencl_config.h
 * @brief OpenCL backend configuration and initialization
 *
 * Provides OpenCL platform/device detection and context management.
 * Works on macOS with AMD GPUs, Linux, and Windows.
 */

#ifdef PTM_GPU_OPENCL

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

namespace gpu {
namespace opencl {

/**
 * @brief OpenCL error code to string
 */
inline const char* clErrorString(cl_int error) {
    switch (error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        default: return "Unknown OpenCL error";
    }
}

/**
 * @brief OpenCL error checking macro
 *
 * Note: On macOS, throwing exceptions from shared libraries can cause SIGILL
 * (illegal instruction) when compiled with aggressive optimizations.
 * Define PTM_OPENCL_NO_THROW before including this header to use a non-throwing
 * version that only logs errors.
 */
#ifdef PTM_OPENCL_NO_THROW
#define CL_CHECK(call)                                                      \
    do {                                                                    \
        cl_int _cl_err = (call);                                            \
        if (_cl_err != CL_SUCCESS) {                                        \
            std::cerr << "[OpenCL Error] "                                  \
                      << gpu::opencl::clErrorString(_cl_err)                \
                      << " at " << __FILE__ << ":" << __LINE__              \
                      << " - " << #call << std::endl;                       \
        }                                                                   \
    } while (0)
#else
#define CL_CHECK(call)                                                      \
    do {                                                                    \
        cl_int err = (call);                                                \
        if (err != CL_SUCCESS) {                                            \
            std::ostringstream oss;                                         \
            oss << "OpenCL error at " << __FILE__ << ":" << __LINE__ << "\n"\
                << "  " << #call << "\n"                                    \
                << "  Error: " << gpu::opencl::clErrorString(err);          \
            throw std::runtime_error(oss.str());                            \
        }                                                                   \
    } while (0)
#endif

/**
 * @brief OpenCL device information
 */
struct OpenCLDeviceInfo {
    cl_device_id device_id = nullptr;
    cl_platform_id platform_id = nullptr;
    std::string name;
    std::string vendor;
    std::string version;
    cl_device_type type = 0;
    size_t global_mem_size = 0;
    size_t local_mem_size = 0;
    size_t max_work_group_size = 0;
    cl_uint max_compute_units = 0;
    cl_uint max_work_item_dimensions = 0;
    std::vector<size_t> max_work_item_sizes;
    bool available = false;
};

/**
 * @brief OpenCL context wrapper with automatic resource management
 */
class OpenCLContext {
public:
    OpenCLContext() = default;

    ~OpenCLContext() {
        release();
    }

    // No copy
    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;

    // Move
    OpenCLContext(OpenCLContext&& other) noexcept {
        *this = std::move(other);
    }

    OpenCLContext& operator=(OpenCLContext&& other) noexcept {
        if (this != &other) {
            release();
            context_ = other.context_;
            queue_ = other.queue_;
            device_ = other.device_;
            platform_ = other.platform_;
            device_info_ = std::move(other.device_info_);
            initialized_ = other.initialized_;

            other.context_ = nullptr;
            other.queue_ = nullptr;
            other.device_ = nullptr;
            other.platform_ = nullptr;
            other.initialized_ = false;
        }
        return *this;
    }

    /**
     * @brief Enumerate all GPU devices across every OpenCL platform.
     *
     * Returns a stable, deterministic list (platform-major, device-minor)
     * suitable for indexing via `--gpu-device-ids`. The single-device
     * `initialize(bool)` path below is left unchanged for byte-identical
     * behaviour at N=1.
     *
     * @param only_available If true, drop devices whose CL_DEVICE_AVAILABLE
     *        is false (busy, faulted, disabled).
     */
    static std::vector<OpenCLDeviceInfo> enumerateGPUs(bool only_available = true) {
        std::vector<OpenCLDeviceInfo> result;
        cl_uint num_platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
            return result;
        }
        std::vector<cl_platform_id> platforms(num_platforms);
        if (clGetPlatformIDs(num_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
            return result;
        }
        for (auto platform : platforms) {
            cl_uint num_devices = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS
                || num_devices == 0) {
                continue;
            }
            std::vector<cl_device_id> devs(num_devices);
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devs.data(), nullptr)
                != CL_SUCCESS) {
                continue;
            }
            for (auto dev : devs) {
                cl_bool avail = CL_FALSE;
                clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(avail), &avail, nullptr);
                if (only_available && !avail) continue;

                OpenCLDeviceInfo info;
                info.device_id = dev;
                info.platform_id = platform;
                info.available = (avail == CL_TRUE);
                char buf[256] = {0};
                clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
                info.name = buf;
                buf[0] = 0;
                clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(buf), buf, nullptr);
                info.vendor = buf;
                buf[0] = 0;
                clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(buf), buf, nullptr);
                info.version = buf;
                clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(info.type), &info.type, nullptr);
                clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE,
                                sizeof(info.global_mem_size), &info.global_mem_size, nullptr);
                clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE,
                                sizeof(info.local_mem_size), &info.local_mem_size, nullptr);
                clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                sizeof(info.max_work_group_size), &info.max_work_group_size, nullptr);
                clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
                                sizeof(info.max_compute_units), &info.max_compute_units, nullptr);
                result.push_back(std::move(info));
            }
        }
        return result;
    }

    /**
     * @brief Initialize OpenCL context pinned to a specific enumerated device.
     *
     * Used by the multi-GPU device pool. The single-device default path
     * (`initialize(bool)` below) is byte-identical to legacy behaviour.
     *
     * @return true if initialization succeeded.
     */
    bool initialize(const OpenCLDeviceInfo& handle) {
        if (initialized_) return true;
        if (handle.device_id == nullptr || handle.platform_id == nullptr) return false;

        device_   = handle.device_id;
        platform_ = handle.platform_id;
        queryDeviceInfo();

        cl_int err = CL_SUCCESS;
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform_, 0
        };
        context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;

#ifdef CL_VERSION_2_0
        cl_queue_properties queue_props[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
        };
        queue_ = clCreateCommandQueueWithProperties(context_, device_, queue_props, &err);
#else
        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
        if (err != CL_SUCCESS) {
            clReleaseContext(context_);
            context_ = nullptr;
            return false;
        }

        initialized_ = true;
        return true;
    }

    /**
     * @brief Initialize OpenCL context with best available GPU
     * @param prefer_gpu If true, prefer GPU over CPU devices
     * @return true if initialization succeeded
     */
    bool initialize(bool prefer_gpu = true) {
        if (initialized_) return true;

        // Get platforms
        cl_uint num_platforms = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            return false;
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

        // Find best device (prefer GPU)
        cl_device_id best_device = nullptr;
        cl_platform_id best_platform = nullptr;
        bool found_gpu = false;

        for (auto platform : platforms) {
            cl_uint num_devices = 0;

            // Try GPU first
            if (prefer_gpu && !found_gpu) {
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
                if (err == CL_SUCCESS && num_devices > 0) {
                    std::vector<cl_device_id> devices(num_devices);
                    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                                            num_devices, devices.data(), nullptr));

                    // Pick first available GPU
                    for (auto dev : devices) {
                        cl_bool available;
                        CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE,
                                                sizeof(available), &available, nullptr));
                        if (available) {
                            best_device = dev;
                            best_platform = platform;
                            found_gpu = true;
                            break;
                        }
                    }
                }
            }

            // Fall back to CPU if no GPU
            if (!found_gpu) {
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
                if (err == CL_SUCCESS && num_devices > 0) {
                    std::vector<cl_device_id> devices(num_devices);
                    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU,
                                            num_devices, devices.data(), nullptr));

                    for (auto dev : devices) {
                        cl_bool available;
                        CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE,
                                                sizeof(available), &available, nullptr));
                        if (available && best_device == nullptr) {
                            best_device = dev;
                            best_platform = platform;
                            break;
                        }
                    }
                }
            }
        }

        if (best_device == nullptr) {
            return false;
        }

        device_ = best_device;
        platform_ = best_platform;

        // Query device info
        queryDeviceInfo();

        // Create context
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform_,
            0
        };

        context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            return false;
        }

        // Create command queue with profiling enabled
#ifdef CL_VERSION_2_0
        cl_queue_properties queue_props[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
            0
        };
        queue_ = clCreateCommandQueueWithProperties(context_, device_, queue_props, &err);
#else
        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif

        if (err != CL_SUCCESS) {
            clReleaseContext(context_);
            context_ = nullptr;
            return false;
        }

        initialized_ = true;
        return true;
    }

    /**
     * @brief Build program from source
     */
    cl_program buildProgram(const std::string& source,
                           const std::string& options = "") {
        if (!initialized_) {
            throw std::runtime_error("OpenCL context not initialized");
        }

        const char* src = source.c_str();
        size_t len = source.size();

        cl_int err;
        cl_program program = clCreateProgramWithSource(context_, 1, &src, &len, &err);
        CL_CHECK(err);

        err = clBuildProgram(program, 1, &device_, options.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // Get build log
            size_t log_size;
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG,
                                 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG,
                                 log_size, log.data(), nullptr);

            std::ostringstream oss;
            oss << "OpenCL program build failed:\n" << log.data();
            clReleaseProgram(program);
            throw std::runtime_error(oss.str());
        }

        return program;
    }

    /**
     * @brief Build program from file
     */
    cl_program buildProgramFromFile(const std::string& filename,
                                    const std::string& options = "") {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open kernel file: " + filename);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buildProgram(buffer.str(), options);
    }

    // Accessors
    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const { return device_; }
    const OpenCLDeviceInfo& deviceInfo() const { return device_info_; }
    bool isInitialized() const { return initialized_; }
    bool isGPU() const { return (device_info_.type & CL_DEVICE_TYPE_GPU) != 0; }

    /**
     * @brief Wait for all queued commands to complete
     */
    void finish() {
        if (queue_) {
            CL_CHECK(clFinish(queue_));
        }
    }

private:
    void queryDeviceInfo() {
        if (!device_) return;

        char buffer[256];

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr));
        device_info_.name = buffer;

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(buffer), buffer, nullptr));
        device_info_.vendor = buffer;

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(buffer), buffer, nullptr));
        device_info_.version = buffer;

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_TYPE,
                                sizeof(device_info_.type), &device_info_.type, nullptr));

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,
                                sizeof(device_info_.global_mem_size),
                                &device_info_.global_mem_size, nullptr));

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE,
                                sizeof(device_info_.local_mem_size),
                                &device_info_.local_mem_size, nullptr));

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                sizeof(device_info_.max_work_group_size),
                                &device_info_.max_work_group_size, nullptr));

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                                sizeof(device_info_.max_compute_units),
                                &device_info_.max_compute_units, nullptr));

        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                sizeof(device_info_.max_work_item_dimensions),
                                &device_info_.max_work_item_dimensions, nullptr));

        device_info_.max_work_item_sizes.resize(device_info_.max_work_item_dimensions);
        CL_CHECK(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                sizeof(size_t) * device_info_.max_work_item_dimensions,
                                device_info_.max_work_item_sizes.data(), nullptr));

        device_info_.device_id = device_;
        device_info_.platform_id = platform_;
        device_info_.available = true;
    }

    void release() {
        if (queue_) {
            clReleaseCommandQueue(queue_);
            queue_ = nullptr;
        }
        if (context_) {
            clReleaseContext(context_);
            context_ = nullptr;
        }
        initialized_ = false;
    }

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_platform_id platform_ = nullptr;
    OpenCLDeviceInfo device_info_;
    bool initialized_ = false;
};

/**
 * @brief Print OpenCL device information
 */
inline void printDeviceInfo(const OpenCLDeviceInfo& info, std::ostream& out = std::cout) {
    out << "OpenCL Device Information:\n";
    out << "  Name: " << info.name << "\n";
    out << "  Vendor: " << info.vendor << "\n";
    out << "  Version: " << info.version << "\n";
    out << "  Type: " << (info.type & CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") << "\n";
    out << "  Global Memory: " << (info.global_mem_size / (1024*1024)) << " MB\n";
    out << "  Local Memory: " << (info.local_mem_size / 1024) << " KB\n";
    out << "  Max Work Group Size: " << info.max_work_group_size << "\n";
    out << "  Compute Units: " << info.max_compute_units << "\n";
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
