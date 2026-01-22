// memory_mapped_io.cpp
#include "memory_mapped_io.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <vector>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD includes for optimized conversion
#ifdef __AVX__
#include <immintrin.h>
#endif

// Linux-specific for memory info
#ifdef __linux__
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace thermal {

// ============================================================================
// WindowedMemoryMappedFile Implementation
// ============================================================================

WindowedMemoryMappedFile::WindowedMemoryMappedFile()
    : total_size_(0)
    , current_window_offset_(0)
    , current_window_size_(0)
    , current_window_data_(nullptr)
    , read_only_(false)
#ifdef _WIN32
    , file_handle_(INVALID_HANDLE_VALUE)
    , map_handle_(nullptr)
#else
    , fd_(-1)
    , page_size_(sysconf(_SC_PAGESIZE))
#endif
{}

WindowedMemoryMappedFile::~WindowedMemoryMappedFile() {
    close();
}

bool WindowedMemoryMappedFile::open(const std::string& filename, size_t total_size, bool read_only) {
    close(); // Ensure clean state
    
    total_size_ = total_size;
    read_only_ = read_only;
    
#ifdef _WIN32
    // Windows implementation
    DWORD access = read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
    DWORD share = read_only ? FILE_SHARE_READ : 0;
    DWORD creation = read_only ? OPEN_EXISTING : OPEN_ALWAYS;
    
    file_handle_ = CreateFileA(
        filename.c_str(),
        access,
        share,
        nullptr,
        creation,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );
    
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to open file: " << GetLastError() << std::endl;
        return false;
    }
    
    if (!read_only) {
        // Resize file to required size
        LARGE_INTEGER li;
        li.QuadPart = total_size;
        if (!SetFilePointerEx(file_handle_, li, nullptr, FILE_BEGIN) ||
            !SetEndOfFile(file_handle_)) {
            std::cerr << "Failed to resize file: " << GetLastError() << std::endl;
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
    }
    
    // Create file mapping object (but don't map yet)
    DWORD protect = read_only ? PAGE_READONLY : PAGE_READWRITE;
    LARGE_INTEGER max_size;
    max_size.QuadPart = total_size;
    
    map_handle_ = CreateFileMappingA(
        file_handle_,
        nullptr,
        protect,
        max_size.HighPart,
        max_size.LowPart,
        nullptr
    );
    
    if (!map_handle_) {
        std::cerr << "Failed to create file mapping: " << GetLastError() << std::endl;
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }
#else
    // Unix/Linux implementation
    int flags = read_only ? O_RDONLY : (O_RDWR | O_CREAT);
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP;
    
    fd_ = ::open(filename.c_str(), flags, mode);
    if (fd_ == -1) {
        std::cerr << "Failed to open file: " << strerror(errno) << std::endl;
        return false;
    }
    
    if (!read_only) {
        // Resize file to required size
        if (ftruncate(fd_, total_size) == -1) {
            std::cerr << "Failed to resize file: " << strerror(errno) << std::endl;
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        // Pre-allocate disk space to avoid fragmentation
        #ifdef __linux__
        if (fallocate(fd_, 0, 0, total_size) == -1) {
            // Non-critical, just a performance optimization
        }
        #endif
    }
    
    // Advise kernel about access pattern
    #ifdef POSIX_FADV_SEQUENTIAL
    posix_fadvise(fd_, 0, total_size, POSIX_FADV_SEQUENTIAL);
    #endif
#endif
    
    return true;
}

void* WindowedMemoryMappedFile::mapWindow(size_t offset, size_t window_size) {
    // Unmap current window if exists
    if (current_window_data_) {
        unmapWindow();
    }
    
    if (!isOpen()) {
        return nullptr;
    }
    
    // Ensure window doesn't exceed file size
    window_size = std::min(window_size, total_size_ - offset);
    if (window_size == 0) {
        return nullptr;
    }
    
#ifdef _WIN32
    // Windows implementation
    DWORD access = read_only_ ? FILE_MAP_READ : (FILE_MAP_READ | FILE_MAP_WRITE);
    LARGE_INTEGER file_offset;
    file_offset.QuadPart = offset;
    
    current_window_data_ = MapViewOfFile(
        map_handle_,
        access,
        file_offset.HighPart,
        file_offset.LowPart,
        window_size
    );
    
    if (!current_window_data_) {
        std::cerr << "Failed to map window: " << GetLastError() << std::endl;
        return nullptr;
    }
#else
    // Unix/Linux implementation
    // Align offset to page boundary
    size_t page_offset = offset % page_size_;
    size_t aligned_offset = offset - page_offset;
    size_t aligned_size = window_size + page_offset;
    
    int prot = read_only_ ? PROT_READ : (PROT_READ | PROT_WRITE);
    int flags = MAP_SHARED;
    
    // Try to use huge pages for large windows
    #ifdef MAP_HUGETLB
    if (aligned_size >= (2 * 1024 * 1024)) {
        flags |= MAP_HUGETLB;
    }
    #endif
    
    void* mapped = mmap(nullptr, aligned_size, prot, flags, fd_, aligned_offset);
    
    if (mapped == MAP_FAILED) {
        // Retry without huge pages if that was the issue
        #ifdef MAP_HUGETLB
        flags &= ~MAP_HUGETLB;
        mapped = mmap(nullptr, aligned_size, prot, flags, fd_, aligned_offset);
        if (mapped == MAP_FAILED) {
        #endif
            std::cerr << "Failed to map window at offset " << offset 
                      << ": " << strerror(errno) << std::endl;
            return nullptr;
        #ifdef MAP_HUGETLB
        }
        #endif
    }
    
    // Advise kernel about access pattern
    if (madvise(mapped, aligned_size, MADV_SEQUENTIAL) == -1) {
        // Non-critical, just a performance hint
    }
    
    // Store the actual mapped region (for unmapping)
    current_window_data_ = mapped;
    current_window_size_ = aligned_size;
    
    // But return the user-requested offset
    current_window_data_ = static_cast<char*>(mapped) + page_offset;
#endif
    
    current_window_offset_ = offset;
    current_window_size_ = window_size;
    
    return current_window_data_;
}

void WindowedMemoryMappedFile::unmapWindow() {
    if (!current_window_data_) {
        return;
    }
    
#ifdef _WIN32
    UnmapViewOfFile(current_window_data_);
#else
    // Need to unmap the aligned region
    size_t page_offset = current_window_offset_ % page_size_;
    void* aligned_addr = static_cast<char*>(current_window_data_) - page_offset;
    size_t aligned_size = current_window_size_ + page_offset;
    
    munmap(aligned_addr, aligned_size);
#endif
    
    current_window_data_ = nullptr;
    current_window_offset_ = 0;
    current_window_size_ = 0;
}

bool WindowedMemoryMappedFile::syncWindow() {
    if (!current_window_data_) {
        return false;
    }
    
#ifdef _WIN32
    return FlushViewOfFile(current_window_data_, current_window_size_) != 0;
#else
    size_t page_offset = current_window_offset_ % page_size_;
    void* aligned_addr = static_cast<char*>(current_window_data_) - page_offset;
    size_t aligned_size = current_window_size_ + page_offset;
    
    return msync(aligned_addr, aligned_size, MS_ASYNC) == 0;
#endif
}

void WindowedMemoryMappedFile::close() {
    unmapWindow();
    
#ifdef _WIN32
    if (map_handle_) {
        CloseHandle(map_handle_);
        map_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    
    total_size_ = 0;
    read_only_ = false;
}

// ============================================================================
// Helper Functions
// ============================================================================

// Get available system memory
static size_t getAvailableSystemMemory() {
#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram + info.bufferram;
    }
#elif defined(__APPLE__)
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &host_size) == KERN_SUCCESS) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        size_t free_pages = vm_stat.free_count + vm_stat.inactive_count;
        return free_pages * page_size;
    }
#endif
    return 4ULL * 1024 * 1024 * 1024; // Default 4GB
}

void MemoryMappedConfig::optimizeForSystem() {
    size_t available_memory = getAvailableSystemMemory();
    
    // Use at most 25% of available memory for window
    size_t max_window = available_memory / 4;
    
    // But cap at 2GB for safety
    max_window = std::min(max_window, size_t(2) * 1024 * 1024 * 1024);
    
    // And ensure at least 512MB
    max_window = std::max(max_window, size_t(512) * 1024 * 1024);
    
    window_size_mb = max_window / (1024 * 1024);
    
    // Auto-detect thread count
    if (num_threads == 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }
}

// Fast float conversion with SIMD
inline void convertDoublesToFloats(const double* src, float* dst, size_t count) {
#ifdef __AVX__
    size_t simd_count = count & ~3;  // Process 4 at a time
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d d = _mm256_loadu_pd(&src[i]);
        __m128 f = _mm256_cvtpd_ps(d);
        _mm_storeu_ps(&dst[i], f);
    }
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
#else
    for (size_t i = 0; i < count; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
#endif
}

// ============================================================================
// Windowed Save Function
// ============================================================================

bool saveIlluminationMemoryMappedWindowed(
    const std::string& output_dir,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    size_t num_faces,
    int time_steps,
    const MemoryMappedConfig& config_in) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Copy and optimize config
    MemoryMappedConfig config = config_in;
    if (config.window_size_mb == 0) {
        config.optimizeForSystem();
    }
    
    // Set up OpenMP if available
    int num_threads = config.num_threads;
#ifdef _OPENMP
    if (num_threads == 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
#else
    num_threads = 1;
#endif
    
    // Calculate layout
    const size_t chunk_size = config.chunk_size;
    const size_t num_chunks = (num_faces + chunk_size - 1) / chunk_size;
    
    const size_t header_size = sizeof(MappedIlluminationData::Header);
    const size_t chunk_headers_size = num_chunks * sizeof(MappedIlluminationData::ChunkHeader);
    const size_t data_size = num_faces * time_steps * sizeof(float) * 3;
    const size_t total_size = header_size + chunk_headers_size + data_size;
    
    const size_t window_size = config.window_size_mb * 1024 * 1024;
    const size_t num_windows = (total_size + window_size - 1) / window_size;
    
    std::cout << "\n=== Windowed Memory-Mapped Illumination Save ===" << std::endl;
    std::cout << "  Faces: " << num_faces << std::endl;
    std::cout << "  Time steps: " << time_steps << std::endl;
    std::cout << "  Chunk size: " << chunk_size << " faces" << std::endl;
    std::cout << "  Number of chunks: " << num_chunks << std::endl;
    std::cout << "  Total file size: " << (total_size / (1024.0 * 1024.0 * 1024.0)) 
              << " GB" << std::endl;
    std::cout << "  Window size: " << config.window_size_mb << " MB" << std::endl;
    std::cout << "  Number of windows: " << num_windows << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    // Create output directory
    std::string command = "mkdir -p " + output_dir + "/illumination_cache";
    system(command.c_str());
    
    std::string filename = output_dir + "/illumination_cache/illumination.mmap";
    
    // Open windowed memory-mapped file
    WindowedMemoryMappedFile mmap_file;
    if (!mmap_file.open(filename, total_size, false)) {
        std::cerr << "Failed to open windowed memory-mapped file" << std::endl;
        return false;
    }
    
    // Write header in first window
    {
        void* window = mmap_file.mapWindow(0, std::min(window_size, total_size));
        if (!window) {
            std::cerr << "Failed to map first window" << std::endl;
            return false;
        }
        
        uint8_t* base_ptr = static_cast<uint8_t*>(window);
        
        // Write main header
        MappedIlluminationData::Header* header = 
            reinterpret_cast<MappedIlluminationData::Header*>(base_ptr);
        
        memset(header, 0, sizeof(MappedIlluminationData::Header));
        header->magic = MappedIlluminationData::MAGIC;
        header->version = MappedIlluminationData::VERSION;
        header->num_faces = num_faces;
        header->time_steps = time_steps;
        header->chunk_size = chunk_size;
        header->num_chunks = num_chunks;
        header->data_offset = header_size + chunk_headers_size;
        header->window_size_mb = config.window_size_mb;
        header->checksum = 0; // Will be computed later
        
        // Write chunk headers
        MappedIlluminationData::ChunkHeader* chunk_headers =
            reinterpret_cast<MappedIlluminationData::ChunkHeader*>(
                base_ptr + sizeof(MappedIlluminationData::Header));
        
        size_t data_offset = 0;
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            size_t chunk_start = chunk_idx * chunk_size;
            size_t chunk_faces = std::min(chunk_size, num_faces - chunk_start);
            size_t chunk_data_size = chunk_faces * time_steps * sizeof(float);
            
            MappedIlluminationData::ChunkHeader& ch = chunk_headers[chunk_idx];
            ch.chunk_id = chunk_idx;
            ch.num_faces = chunk_faces;
            ch.start_face = chunk_start;
            ch.compressed = 0;
            ch.data_offsets[0] = data_offset;
            ch.data_offsets[1] = data_offset + chunk_data_size;
            ch.data_offsets[2] = data_offset + 2 * chunk_data_size;
            ch.data_sizes[0] = chunk_data_size;
            ch.data_sizes[1] = chunk_data_size;
            ch.data_sizes[2] = chunk_data_size;
            
            data_offset += 3 * chunk_data_size;
        }
        
        mmap_file.syncWindow();
    }
    
    // Process data in windows
    std::cout << "Converting and writing data using " << num_threads << " threads..." << std::endl;
    
    // Temporary buffer for conversion (per thread)
    std::vector<std::vector<float>> thread_buffers(num_threads);
    for (int i = 0; i < num_threads; i++) {
        thread_buffers[i].resize(chunk_size * time_steps);
    }
    
    // Track progress
    std::atomic<size_t> chunks_processed(0);
    const size_t progress_interval = std::max(size_t(1), num_chunks / 100);
    
    // Process chunks
    size_t current_window_start = header_size + chunk_headers_size;
    size_t current_window_end = current_window_start;
    void* current_window = nullptr;
    
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        size_t chunk_start = chunk_idx * chunk_size;
        size_t chunk_faces = std::min(chunk_size, num_faces - chunk_start);
        size_t chunk_data_size = chunk_faces * time_steps * sizeof(float) * 3;
        
        // Calculate where this chunk's data should go
        size_t chunk_file_offset = header_size + chunk_headers_size;
        for (size_t i = 0; i < chunk_idx; i++) {
            size_t prev_faces = std::min(chunk_size, num_faces - i * chunk_size);
            chunk_file_offset += prev_faces * time_steps * sizeof(float) * 3;
        }
        
        // Check if we need to map a new window
        if (!current_window || chunk_file_offset >= current_window_end ||
            chunk_file_offset + chunk_data_size > current_window_end) {
            
            // Sync and unmap current window
            if (current_window) {
                mmap_file.syncWindow();
            }
            
            // Map new window
            current_window_start = chunk_file_offset;
            current_window_end = std::min(current_window_start + window_size, total_size);
            
            current_window = mmap_file.mapWindow(current_window_start, 
                                                 current_window_end - current_window_start);
            if (!current_window) {
                std::cerr << "Failed to map window at offset " << current_window_start << std::endl;
                return false;
            }
        }
        
        // Calculate pointer within current window
        size_t offset_in_window = chunk_file_offset - current_window_start;
        float* chunk_data_ptr = reinterpret_cast<float*>(
            static_cast<uint8_t*>(current_window) + offset_in_window);
        
        // Convert and copy data for this chunk
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (size_t face = 0; face < chunk_faces; face++) {
            int thread_id = 0;
            #ifdef _OPENMP
            thread_id = omp_get_thread_num();
            #endif
            
            float* local_buffer = thread_buffers[thread_id].data();
            
            size_t global_face = chunk_start + face;
            size_t src_offset = global_face * time_steps;
            
            // Convert this face's data
            convertDoublesToFloats(&direct_solar_values[src_offset], 
                                 local_buffer, time_steps);
            
            // Copy to file (direct solar)
            size_t dst_offset = face * time_steps;
            memcpy(&chunk_data_ptr[dst_offset], local_buffer, 
                   time_steps * sizeof(float));
            
            // Convert and copy primary reflected
            convertDoublesToFloats(&primary_reflected_values[src_offset],
                                 local_buffer, time_steps);
            dst_offset += chunk_faces * time_steps;
            memcpy(&chunk_data_ptr[dst_offset], local_buffer,
                   time_steps * sizeof(float));
            
            // Convert and copy primary IR
            convertDoublesToFloats(&primary_ir_values[src_offset],
                                 local_buffer, time_steps);
            dst_offset += chunk_faces * time_steps;
            memcpy(&chunk_data_ptr[dst_offset], local_buffer,
                   time_steps * sizeof(float));
        }
        
        // Update progress
        size_t processed = chunks_processed.fetch_add(1) + 1;
        if (processed % progress_interval == 0 || processed == num_chunks) {
            float percent = 100.0f * processed / num_chunks;
            std::cout << "\rProgress: [";
            int bar_width = 50;
            int filled = static_cast<int>(bar_width * processed / num_chunks);
            for (int i = 0; i < bar_width; i++) {
                if (i < filled) std::cout << "=";
                else if (i == filled) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::fixed << std::setprecision(1) 
                      << percent << "%" << std::flush;
        }
    }
    
    // Final sync
    if (current_window) {
        mmap_file.syncWindow();
    }
    
    std::cout << std::endl;
    
    // Close file
    mmap_file.close();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "\nIllumination cache saved successfully" << std::endl;
    std::cout << "Total time: " << elapsed << " seconds" << std::endl;
    std::cout << "Write speed: " << (total_size / (1024.0 * 1024.0 * elapsed)) 
              << " MB/s" << std::endl;
    
    return true;
}

// ============================================================================
// Windowed Load Function  
// ============================================================================

bool loadIlluminationMemoryMappedWindowed(
    const std::string& output_dir,
    std::vector<double>& direct_solar_values,
    std::vector<double>& primary_reflected_values,
    std::vector<double>& primary_ir_values,
    size_t& num_faces,
    int& time_steps,
    const MemoryMappedConfig& config_in) {
    
    std::string filename = output_dir + "/illumination_cache/illumination.mmap";
    
    // Check if file exists
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        std::cerr << "Illumination cache file not found: " << filename << std::endl;
        return false;
    }
    
    size_t file_size = st.st_size;
    
    // Copy and optimize config
    MemoryMappedConfig config = config_in;
    if (config.window_size_mb == 0) {
        config.optimizeForSystem();
    }
    
    const size_t window_size = config.window_size_mb * 1024 * 1024;
    
    // Open windowed memory-mapped file
    WindowedMemoryMappedFile mmap_file;
    if (!mmap_file.open(filename, file_size, true)) {
        std::cerr << "Failed to open windowed memory-mapped file for reading" << std::endl;
        return false;
    }
    
    // Read header
    MappedIlluminationData::Header header;
    {
        void* window = mmap_file.mapWindow(0, std::min(window_size, file_size));
        if (!window) {
            std::cerr << "Failed to map first window for reading" << std::endl;
            return false;
        }
        
        memcpy(&header, window, sizeof(MappedIlluminationData::Header));
        
        if (header.magic != MappedIlluminationData::MAGIC) {
            std::cerr << "Invalid file format (bad magic number)" << std::endl;
            return false;
        }
        
        if (header.version != MappedIlluminationData::VERSION) {
            std::cerr << "Incompatible file version" << std::endl;
            return false;
        }
    }
    
    num_faces = header.num_faces;
    time_steps = header.time_steps;
    
    const size_t total_values = num_faces * time_steps;
    direct_solar_values.resize(total_values);
    primary_reflected_values.resize(total_values);
    primary_ir_values.resize(total_values);
    
    std::cout << "\n=== Loading Illumination from Windowed Memory Map ===" << std::endl;
    std::cout << "  Faces: " << num_faces << std::endl;
    std::cout << "  Time steps: " << time_steps << std::endl;
    std::cout << "  Chunks: " << header.num_chunks << std::endl;
    std::cout << "  Window size: " << config.window_size_mb << " MB" << std::endl;
    
    // Read chunk headers
    std::vector<MappedIlluminationData::ChunkHeader> chunk_headers(header.num_chunks);
    {
        size_t chunk_headers_offset = sizeof(MappedIlluminationData::Header);
        size_t chunk_headers_size = header.num_chunks * sizeof(MappedIlluminationData::ChunkHeader);
        
        void* window = mmap_file.mapWindow(0, 
            std::min(window_size, chunk_headers_offset + chunk_headers_size));
        if (!window) {
            std::cerr << "Failed to map chunk headers" << std::endl;
            return false;
        }
        
        memcpy(chunk_headers.data(), 
               static_cast<uint8_t*>(window) + chunk_headers_offset,
               chunk_headers_size);
    }
    
    // Read data in windows
    std::cout << "Loading data..." << std::endl;
    
    size_t current_window_start = 0;
    size_t current_window_end = 0;
    void* current_window = nullptr;
    
    for (size_t chunk_idx = 0; chunk_idx < header.num_chunks; chunk_idx++) {
        const MappedIlluminationData::ChunkHeader& ch = chunk_headers[chunk_idx];
        
        // Calculate file offset for this chunk's data
        size_t chunk_file_offset = header.data_offset + ch.data_offsets[0];
        size_t chunk_data_size = ch.data_sizes[0] + ch.data_sizes[1] + ch.data_sizes[2];
        
        // Map new window if needed
        if (!current_window || chunk_file_offset >= current_window_end ||
            chunk_file_offset + chunk_data_size > current_window_end) {
            
            current_window_start = chunk_file_offset;
            current_window_end = std::min(current_window_start + window_size, file_size);
            
            current_window = mmap_file.mapWindow(current_window_start,
                                                 current_window_end - current_window_start);
            if (!current_window) {
                std::cerr << "Failed to map window for chunk " << chunk_idx << std::endl;
                return false;
            }
        }
        
        // Read and convert data
        size_t offset_in_window = chunk_file_offset - current_window_start;
        const float* chunk_data = reinterpret_cast<const float*>(
            static_cast<uint8_t*>(current_window) + offset_in_window);
        
        const float* direct_src = chunk_data;
        const float* reflected_src = chunk_data + ch.num_faces * time_steps;
        const float* ir_src = chunk_data + 2 * ch.num_faces * time_steps;
        
        // Convert floats back to doubles
        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < ch.num_faces; f++) {
            size_t global_face = ch.start_face + f;
            size_t src_offset = f * time_steps;
            size_t dst_offset = global_face * time_steps;
            
            for (int t = 0; t < time_steps; t++) {
                direct_solar_values[dst_offset + t] = direct_src[src_offset + t];
                primary_reflected_values[dst_offset + t] = reflected_src[src_offset + t];
                primary_ir_values[dst_offset + t] = ir_src[src_offset + t];
            }
        }
        
        // Progress
        if ((chunk_idx + 1) % 100 == 0 || chunk_idx == header.num_chunks - 1) {
            float percent = 100.0f * (chunk_idx + 1) / header.num_chunks;
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                      << percent << "%" << std::flush;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Successfully loaded illumination data" << std::endl;
    
    return true;
}

// ============================================================================
// Legacy MemoryMappedFile implementation
// ============================================================================

MemoryMappedFile::MemoryMappedFile()
    : data_(nullptr), size_(0)
#ifdef _WIN32
    , file_handle_(INVALID_HANDLE_VALUE), map_handle_(nullptr)
#else
    , fd_(-1)
#endif
{}

MemoryMappedFile::~MemoryMappedFile() {
    close();
}

bool MemoryMappedFile::open(const std::string& filename, size_t size, bool read_only) {
    close();  // Ensure any existing mapping is closed
    
#ifdef _WIN32
    // Windows implementation
    DWORD access = read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
    DWORD share = read_only ? FILE_SHARE_READ : 0;
    DWORD create = read_only ? OPEN_EXISTING : OPEN_ALWAYS;
    
    file_handle_ = CreateFileA(filename.c_str(), access, share, nullptr, 
                               create, FILE_ATTRIBUTE_NORMAL, nullptr);
    
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    // Set file size if writing
    if (!read_only) {
        LARGE_INTEGER li;
        li.QuadPart = size;
        if (!SetFilePointerEx(file_handle_, li, nullptr, FILE_BEGIN) ||
            !SetEndOfFile(file_handle_)) {
            std::cerr << "Failed to set file size" << std::endl;
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
    }
    
    DWORD protect = read_only ? PAGE_READONLY : PAGE_READWRITE;
    map_handle_ = CreateFileMappingA(file_handle_, nullptr, protect,
                                     (size >> 32), size & 0xFFFFFFFF, nullptr);
    
    if (map_handle_ == nullptr) {
        std::cerr << "Failed to create file mapping" << std::endl;
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }
    
    DWORD map_access = read_only ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS;
    data_ = MapViewOfFile(map_handle_, map_access, 0, 0, size);
    
    if (data_ == nullptr) {
        std::cerr << "Failed to map view of file" << std::endl;
        CloseHandle(map_handle_);
        CloseHandle(file_handle_);
        map_handle_ = nullptr;
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }
    
#else
    // Unix/Linux implementation
    int flags = read_only ? O_RDONLY : (O_RDWR | O_CREAT);
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP;
    
    fd_ = ::open(filename.c_str(), flags, mode);
    if (fd_ == -1) {
        std::cerr << "Failed to open file: " << filename << " - " << strerror(errno) << std::endl;
        return false;
    }
    
    // Set file size if writing
    if (!read_only) {
        if (ftruncate(fd_, size) == -1) {
            std::cerr << "Failed to set file size: " << strerror(errno) << std::endl;
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        // Pre-allocate disk space to avoid fragmentation
        #ifdef __linux__
        if (posix_fallocate(fd_, 0, size) != 0) {
            std::cerr << "Warning: Could not pre-allocate file space" << std::endl;
        }
        #endif
    }
    
    int prot = read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
    int flags_mmap = MAP_SHARED;
    
    // Use huge pages if available for better performance
    #ifdef MAP_HUGETLB
    if (size >= (2 * 1024 * 1024)) {  // Use huge pages for files >= 2MB
        flags_mmap |= MAP_HUGETLB;
    }
    #endif
    
    data_ = mmap(nullptr, size, prot, flags_mmap, fd_, 0);
    
    if (data_ == MAP_FAILED) {
        // Retry without huge pages if that was the issue
        #ifdef MAP_HUGETLB
        flags_mmap &= ~MAP_HUGETLB;
        data_ = mmap(nullptr, size, prot, flags_mmap, fd_, 0);
        if (data_ == MAP_FAILED) {
        #endif
            std::cerr << "Failed to map file: " << strerror(errno) << std::endl;
            ::close(fd_);
            fd_ = -1;
            data_ = nullptr;
            return false;
        #ifdef MAP_HUGETLB
        }
        #endif
    }
    
    // Advise kernel about access pattern
    if (madvise(data_, size, MADV_SEQUENTIAL) == -1) {
        // Non-critical, just a performance hint
    }
#endif
    
    size_ = size;
    return true;
}

void MemoryMappedFile::close() {
    if (!is_open()) return;
    
#ifdef _WIN32
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (map_handle_) {
        CloseHandle(map_handle_);
        map_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (data_) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    
    size_ = 0;
}

bool MemoryMappedFile::sync() {
    if (!is_open()) return false;
    
#ifdef _WIN32
    return FlushViewOfFile(data_, size_) != 0;
#else
    return msync(data_, size_, MS_SYNC) == 0;
#endif
}

// ============================================================================
// Optimized Save Function Using Memory Mapping
// ============================================================================

bool saveIlluminationMemoryMapped(
    const std::string& output_dir,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    size_t num_faces,
    int time_steps,
    int num_threads) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Determine optimal chunk size based on system memory
    const size_t bytes_per_face = time_steps * sizeof(float) * 3;
    size_t chunk_size = 200000;  // Default 200k faces per chunk
    
    // Adjust based on L3 cache size (typically 8-64MB)
    const size_t l3_cache_size = 32 * 1024 * 1024;  // Assume 32MB L3
    const size_t optimal_chunk_bytes = l3_cache_size / 4;  // Use 1/4 of L3
    chunk_size = std::min(chunk_size, optimal_chunk_bytes / bytes_per_face);
    chunk_size = std::max(chunk_size, size_t(10000));  // Minimum 10k faces
    
    const size_t num_chunks = (num_faces + chunk_size - 1) / chunk_size;
    
    // Calculate total file size
    const size_t header_size = sizeof(MappedIlluminationData::Header);
    const size_t chunk_headers_size = num_chunks * sizeof(MappedIlluminationData::ChunkHeader);
    const size_t data_size = num_faces * time_steps * sizeof(float) * 3;
    const size_t total_size = header_size + chunk_headers_size + data_size;
    
    std::cout << "Memory-mapped illumination save:" << std::endl;
    std::cout << "  Faces: " << num_faces << std::endl;
    std::cout << "  Time steps: " << time_steps << std::endl;
    std::cout << "  Chunk size: " << chunk_size << " faces" << std::endl;
    std::cout << "  Number of chunks: " << num_chunks << std::endl;
    std::cout << "  Total file size: " << (total_size / (1024.0 * 1024.0 * 1024.0)) 
              << " GB" << std::endl;
    
    // Create output directory
    std::string command = "mkdir -p " + output_dir + "/illumination_cache";
    system(command.c_str());
    
    // Open memory-mapped file
    std::string filename = output_dir + "/illumination_cache/illumination.mmap";
    MemoryMappedFile mmap_file;
    
    if (!mmap_file.open(filename, total_size, false)) {
        std::cerr << "Failed to create memory-mapped file" << std::endl;
        return false;
    }
    
    // Get pointer to mapped memory
    uint8_t* base_ptr = static_cast<uint8_t*>(mmap_file.data());
    
    // Write header
    MappedIlluminationData::Header* header = 
        reinterpret_cast<MappedIlluminationData::Header*>(base_ptr);
    
    header->magic = MappedIlluminationData::MAGIC;
    header->version = MappedIlluminationData::VERSION;
    header->num_faces = num_faces;
    header->time_steps = time_steps;
    header->chunk_size = chunk_size;
    header->num_chunks = num_chunks;
    header->data_offset = header_size + chunk_headers_size;
    header->checksum = 0;  // Will be updated later
    
    // Pointer to chunk headers
    MappedIlluminationData::ChunkHeader* chunk_headers = 
        reinterpret_cast<MappedIlluminationData::ChunkHeader*>(base_ptr + header_size);
    
    // Pointer to data area
    float* data_ptr = reinterpret_cast<float*>(base_ptr + header->data_offset);
    
    // Set up OpenMP threads
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    
    std::cout << "Converting and writing data using " << num_threads << " threads..." << std::endl;
    
    // Progress tracking
    std::atomic<size_t> chunks_processed(0);
    std::atomic<uint64_t> checksum_accumulator(0);
    
    // Process chunks in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        const size_t start_face = chunk_idx * chunk_size;
        const size_t end_face = std::min(start_face + chunk_size, num_faces);
        const size_t chunk_faces = end_face - start_face;
        const size_t chunk_values = chunk_faces * time_steps;
        
        // Calculate offsets for this chunk
        size_t chunk_offset = 0;
        for (size_t i = 0; i < chunk_idx; i++) {
            size_t faces_in_chunk = std::min(chunk_size, num_faces - i * chunk_size);
            chunk_offset += faces_in_chunk * time_steps * 3;
        }
        
        float* chunk_data = data_ptr + chunk_offset;
        
        // Fill in chunk header
        MappedIlluminationData::ChunkHeader& ch = chunk_headers[chunk_idx];
        ch.chunk_id = chunk_idx;
        ch.num_faces = chunk_faces;
        ch.start_face = start_face;
        ch.compressed = 0;  // No compression for now
        ch.data_offsets[0] = 0;
        ch.data_offsets[1] = chunk_values * sizeof(float);
        ch.data_offsets[2] = 2 * chunk_values * sizeof(float);
        ch.data_sizes[0] = chunk_values * sizeof(float);
        ch.data_sizes[1] = chunk_values * sizeof(float);
        ch.data_sizes[2] = chunk_values * sizeof(float);
        
        // Convert and copy data - optimized memory access pattern
        float* direct_dst = chunk_data;
        float* reflected_dst = chunk_data + chunk_values;
        float* ir_dst = chunk_data + 2 * chunk_values;
        
        // Process in blocks for better cache utilization
        const size_t block_size = 64;  // Process 64 faces at a time
        
        for (size_t block_start = 0; block_start < chunk_faces; block_start += block_size) {
            const size_t block_end = std::min(block_start + block_size, chunk_faces);
            
            for (size_t f = block_start; f < block_end; f++) {
                const size_t global_face = start_face + f;
                const size_t src_offset = global_face * time_steps;
                const size_t dst_offset = f * time_steps;
                
                // Use SIMD-optimized conversion
                convertDoublesToFloats(
                    &direct_solar_values[src_offset],
                    &direct_dst[dst_offset],
                    time_steps
                );
                
                convertDoublesToFloats(
                    &primary_reflected_values[src_offset],
                    &reflected_dst[dst_offset],
                    time_steps
                );
                
                convertDoublesToFloats(
                    &primary_ir_values[src_offset],
                    &ir_dst[dst_offset],
                    time_steps
                );
            }
        }
        
        // Simple checksum for data integrity
        uint64_t chunk_checksum = 0;
        for (size_t i = 0; i < chunk_values * 3; i += 16) {
            chunk_checksum ^= *reinterpret_cast<uint64_t*>(&chunk_data[i]);
        }
        checksum_accumulator.fetch_add(chunk_checksum);
        
        // Update progress
        size_t completed = chunks_processed.fetch_add(1) + 1;
        if (completed % std::max(size_t(1), num_chunks / 100) == 0 || completed == num_chunks) {
            #pragma omp critical
            {
                float progress = 100.0f * completed / num_chunks;
                std::cout << "\rProgress: [";
                int bar_width = 50;
                int pos = bar_width * progress / 100.0f;
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << std::fixed << std::setprecision(1) 
                          << progress << "%" << std::flush;
            }
        }
    }
    
    std::cout << std::endl;
    
    // Update header checksum
    header->checksum = checksum_accumulator.load();
    
    // Sync to disk
    std::cout << "Syncing to disk..." << std::endl;
    if (!mmap_file.sync()) {
        std::cerr << "Warning: Failed to sync memory-mapped file" << std::endl;
    }
    
    // Close file
    mmap_file.close();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "Memory-mapped save completed in " << duration.count() << " seconds" << std::endl;
    std::cout << "Output file: " << filename << std::endl;
    
    // Also write a metadata file for easier reading
    std::ofstream meta(output_dir + "/illumination_cache/metadata.txt");
    meta << "format: memory_mapped" << std::endl;
    meta << "version: " << MappedIlluminationData::VERSION << std::endl;
    meta << "num_faces: " << num_faces << std::endl;
    meta << "time_steps: " << time_steps << std::endl;
    meta << "chunk_size: " << chunk_size << std::endl;
    meta << "num_chunks: " << num_chunks << std::endl;
    meta << "file_size: " << total_size << std::endl;
    meta << "checksum: " << header->checksum << std::endl;
    meta.close();
    
    return true;
}

// ============================================================================
// Load Function Using Memory Mapping
// ============================================================================

bool loadIlluminationMemoryMapped(
    const std::string& output_dir,
    std::vector<double>& direct_solar_values,
    std::vector<double>& primary_reflected_values,
    std::vector<double>& primary_ir_values,
    size_t& num_faces,
    int& time_steps) {
    
    std::string filename = output_dir + "/illumination_cache/illumination.mmap";
    
    // Check if file exists
    std::ifstream test(filename);
    if (!test.good()) {
        std::cerr << "Memory-mapped file not found: " << filename << std::endl;
        return false;
    }
    test.close();
    
    // Get file size
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        std::cerr << "Failed to stat file: " << filename << std::endl;
        return false;
    }
    size_t file_size = st.st_size;
    
    // Open memory-mapped file for reading
    MemoryMappedFile mmap_file;
    if (!mmap_file.open(filename, file_size, true)) {
        std::cerr << "Failed to open memory-mapped file for reading" << std::endl;
        return false;
    }
    
    // Read header
    const uint8_t* base_ptr = static_cast<const uint8_t*>(mmap_file.data());
    const MappedIlluminationData::Header* header = 
        reinterpret_cast<const MappedIlluminationData::Header*>(base_ptr);
    
    // Validate header
    if (header->magic != MappedIlluminationData::MAGIC) {
        std::cerr << "Invalid magic number in memory-mapped file" << std::endl;
        return false;
    }
    
    if (header->version != MappedIlluminationData::VERSION) {
        std::cerr << "Unsupported version in memory-mapped file" << std::endl;
        return false;
    }
    
    // Extract metadata
    num_faces = header->num_faces;
    time_steps = header->time_steps;
    
    std::cout << "Loading memory-mapped illumination data:" << std::endl;
    std::cout << "  Faces: " << num_faces << std::endl;
    std::cout << "  Time steps: " << time_steps << std::endl;
    std::cout << "  Chunks: " << header->num_chunks << std::endl;
    
    // Resize output vectors
    size_t total_values = num_faces * time_steps;
    direct_solar_values.resize(total_values);
    primary_reflected_values.resize(total_values);
    primary_ir_values.resize(total_values);
    
    // Get pointers
    const MappedIlluminationData::ChunkHeader* chunk_headers = 
        reinterpret_cast<const MappedIlluminationData::ChunkHeader*>(base_ptr + sizeof(MappedIlluminationData::Header));
    const float* data_ptr = reinterpret_cast<const float*>(base_ptr + header->data_offset);
    
    // Load data in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t chunk_idx = 0; chunk_idx < header->num_chunks; chunk_idx++) {
        const MappedIlluminationData::ChunkHeader& ch = chunk_headers[chunk_idx];
        
        // Calculate offset for this chunk's data
        size_t chunk_offset = 0;
        for (size_t i = 0; i < chunk_idx; i++) {
            size_t faces_in_chunk = chunk_headers[i].num_faces;
            chunk_offset += faces_in_chunk * time_steps * 3;
        }
        
        const float* chunk_data = data_ptr + chunk_offset;
        const float* direct_src = chunk_data;
        const float* reflected_src = chunk_data + ch.num_faces * time_steps;
        const float* ir_src = chunk_data + 2 * ch.num_faces * time_steps;
        
        // Convert floats back to doubles
        for (size_t f = 0; f < ch.num_faces; f++) {
            size_t global_face = ch.start_face + f;
            size_t src_offset = f * time_steps;
            size_t dst_offset = global_face * time_steps;
            
            for (int t = 0; t < time_steps; t++) {
                direct_solar_values[dst_offset + t] = direct_src[src_offset + t];
                primary_reflected_values[dst_offset + t] = reflected_src[src_offset + t];
                primary_ir_values[dst_offset + t] = ir_src[src_offset + t];
            }
        }
    }
    
    // Verify checksum
    uint64_t computed_checksum = 0;
    for (size_t i = 0; i < header->num_faces * time_steps * 3; i += 16) {
        computed_checksum ^= *reinterpret_cast<const uint64_t*>(&data_ptr[i]);
    }
    
    if (computed_checksum != header->checksum) {
        std::cerr << "Warning: Checksum mismatch in memory-mapped file" << std::endl;
    }
    
    std::cout << "Successfully loaded illumination data from memory-mapped file" << std::endl;
    
    return true;
}

} // namespace thermal