// memory_mapped_io.h
#ifndef MEMORY_MAPPED_IO_H
#define MEMORY_MAPPED_IO_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <atomic>
#include <mutex>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <errno.h>
#endif

namespace thermal {

// Memory-mapped file wrapper for cross-platform support
class MemoryMappedFile {
public:
    MemoryMappedFile();
    ~MemoryMappedFile();
    
    // Open/create a file for memory mapping
    bool open(const std::string& filename, size_t size, bool read_only = false);
    void close();
    
    // Get pointer to mapped memory
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }
    
    // Sync changes to disk
    bool sync();
    
private:
    void* data_;
    size_t size_;
    
#ifdef _WIN32
    HANDLE file_handle_;
    HANDLE map_handle_;
#else
    int fd_;
#endif
    
    // Disable copying
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
};

// NEW: Windowed memory-mapped file for handling large files
class WindowedMemoryMappedFile {
public:
    WindowedMemoryMappedFile();
    ~WindowedMemoryMappedFile();
    
    // Open file and prepare for windowed mapping
    bool open(const std::string& filename, size_t total_size, bool read_only = false);
    void close();
    
    // Map a window of the file
    void* mapWindow(size_t offset, size_t window_size);
    void unmapWindow();
    bool syncWindow();
    
    // Get current window info
    void* currentWindowData() { return current_window_data_; }
    size_t currentWindowOffset() const { return current_window_offset_; }
    size_t currentWindowSize() const { return current_window_size_; }
    bool hasWindow() const { return current_window_data_ != nullptr; }
    
    size_t totalSize() const { return total_size_; }
    bool isOpen() const { 
    #ifdef _WIN32
        return file_handle_ != INVALID_HANDLE_VALUE;
    #else
        return fd_ != -1; 
    #endif
    }
    
private:
    size_t total_size_;
    size_t current_window_offset_;
    size_t current_window_size_;
    void* current_window_data_;
    bool read_only_;
    
#ifdef _WIN32
    HANDLE file_handle_;
    HANDLE map_handle_;
#else
    int fd_;
    size_t page_size_;
#endif
    
    // Disable copying
    WindowedMemoryMappedFile(const WindowedMemoryMappedFile&) = delete;
    WindowedMemoryMappedFile& operator=(const WindowedMemoryMappedFile&) = delete;
};

// Structure for memory-mapped illumination data
struct MappedIlluminationData {
    struct Header {
        uint64_t magic;           // Magic number for validation
        uint64_t version;         // Format version  
        uint64_t num_faces;       // Total number of faces
        uint32_t time_steps;      // Number of time steps
        uint32_t chunk_size;      // Faces per chunk
        uint32_t num_chunks;      // Total number of chunks
        uint32_t data_offset;     // Offset to first data chunk
        uint64_t checksum;        // Data integrity checksum
        uint32_t window_size_mb;  // Window size used for mapping (for validation)
        char padding[452];        // Pad to 512 bytes for alignment
    };
    
    struct ChunkHeader {
        uint32_t chunk_id;        // Chunk identifier
        uint32_t num_faces;       // Faces in this chunk
        uint32_t start_face;      // Starting face index
        uint32_t compressed;      // Compression flag (0=none)
        uint64_t data_offsets[3]; // Offsets for direct/reflected/IR data
        uint64_t data_sizes[3];   // Sizes of each data array
    };
    
    static constexpr uint64_t MAGIC = 0x4D4150494C4C554D; // "MAPILLUM"
    static constexpr uint64_t VERSION = 2; // Bumped for windowed support
};

// Configuration for memory-mapped operations
struct MemoryMappedConfig {
    size_t window_size_mb = 1024;      // Default 1 GB windows
    size_t chunk_size = 10000;         // Faces per chunk
    bool use_huge_pages = true;        // Try to use huge pages
    bool prefault_pages = false;       // Pre-fault pages into memory
    int num_threads = 0;               // 0 = auto-detect
    
    // Calculate optimal window size based on available memory
    void optimizeForSystem();
};

// Optimized save function using WINDOWED memory mapping
bool saveIlluminationMemoryMappedWindowed(
    const std::string& output_dir,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,  
    const std::vector<double>& primary_ir_values,
    size_t num_faces,
    int time_steps,
    const MemoryMappedConfig& config = MemoryMappedConfig());

// Load function using WINDOWED memory mapping
bool loadIlluminationMemoryMappedWindowed(
    const std::string& output_dir,
    std::vector<double>& direct_solar_values,
    std::vector<double>& primary_reflected_values,
    std::vector<double>& primary_ir_values,
    size_t& num_faces,
    int& time_steps,
    const MemoryMappedConfig& config = MemoryMappedConfig());

} // namespace thermal

#endif // MEMORY_MAPPED_IO_H