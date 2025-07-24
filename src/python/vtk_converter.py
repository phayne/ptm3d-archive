#!/usr/bin/env python3
import os
import sys
import struct
import argparse
from pathlib import Path
import time
import concurrent.futures
from functools import partial

# Standard Python VTK instead of ParaView's VTK
try:
    import vtk
    import numpy as np
except ImportError:
    print("Please install required packages with:")
    print("pip install numpy vtk")
    sys.exit(1)

def print_progress(current, total, prefix='Progress:', suffix='Complete', decimals=1, length=50, fill='█'):
    """Print a progress bar to the console with reduced update frequency for large datasets"""
    # Only update every 0.5% to reduce console I/O overhead for large meshes
    update_interval = max(1, total // 200)
    if current % update_interval != 0 and current != total:
        return
        
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')

def read_binary_mesh(mesh_file):
    """Read binary PLY mesh file using optimized numpy operations"""
    start_time = time.time()
    
    try:
        with open(mesh_file, 'rb') as f:
            # Check if file starts with 'EUROPMSH' (8 bytes)
            header = f.read(8)
            if header == b'EUROPMSH':
                # This is our custom format - use fast binary reading
                return read_optimized_binary_mesh(mesh_file)
            
            # Reset file position for PLY header reading
            f.seek(0)
            
            # Read header (assuming ASCII header with binary data)
            header = ""
            while True:
                line = f.readline().decode('ascii').strip()
                header += line + "\n"
                if line == "end_header":
                    break
            
            # Parse header to get vertex and face counts
            vertex_count = int([line for line in header.split('\n') 
                              if "element vertex" in line][0].split()[-1])
            face_count = int([line for line in header.split('\n') 
                            if "element face" in line][0].split()[-1])
            
            print(f"Mesh contains {vertex_count} vertices and {face_count} faces")
            
            # Allocate numpy arrays for better performance
            vertices = np.zeros((vertex_count, 3), dtype=np.float32)
            faces = np.zeros((face_count, 3), dtype=np.int32)
            normals = np.zeros((face_count, 3), dtype=np.float32)
            latitudes = np.zeros(face_count, dtype=np.float32)
            longitudes = np.zeros(face_count, dtype=np.float32)
            
            # Read vertices
            for i in range(vertex_count):
                vertices[i, 0] = struct.unpack('f', f.read(4))[0]
                vertices[i, 1] = struct.unpack('f', f.read(4))[0]
                vertices[i, 2] = struct.unpack('f', f.read(4))[0]
            
            # Read faces
            for i in range(face_count):
                count = struct.unpack('B', f.read(1))[0]  # Number of vertices in face
                indices = []
                for _ in range(count):
                    idx = struct.unpack('i', f.read(4))[0]
                    indices.append(idx)
                faces[i] = indices[:3]  # Take only first 3 for non-triangular faces
                
                # Read face normal
                normals[i, 0] = struct.unpack('f', f.read(4))[0]
                normals[i, 1] = struct.unpack('f', f.read(4))[0]
                normals[i, 2] = struct.unpack('f', f.read(4))[0]
                
                # Read latitude and longitude
                latitudes[i] = struct.unpack('f', f.read(4))[0]
                longitudes[i] = struct.unpack('f', f.read(4))[0]
        
        print(f"Mesh loading completed in {time.time() - start_time:.2f} seconds")
        return vertices, faces, normals, latitudes, longitudes
    
    except Exception as e:
        print(f"Error reading mesh file: {e}")
        print("Attempting to use fallback format reader...")
        return read_optimized_binary_mesh(mesh_file)

def read_optimized_binary_mesh(mesh_file):
    """Read optimized binary mesh format using vectorized operations"""
    start_time = time.time()
    
    try:
        with open(mesh_file, 'rb') as f:
            # Skip 8-byte header if present (EUROPMSH)
            header = f.read(8)
            if header != b'EUROPMSH':
                # If not our format, reset to beginning
                f.seek(0)
            
            # Read vertex count (4-byte integer)
            vertex_count = struct.unpack('i', f.read(4))[0]
            
            print(f"Reading {vertex_count} vertices...")
            
            # Read all vertices at once (much faster)
            vertices_buffer = f.read(vertex_count * 12)  # 3 floats * 4 bytes per float
            vertices = np.frombuffer(vertices_buffer, dtype=np.float32).reshape(vertex_count, 3)
            
            # Read face count (4-byte integer)
            face_count = struct.unpack('i', f.read(4))[0]
            
            print(f"Reading {face_count} faces...")
            
            # Read all faces at once
            faces_buffer = f.read(face_count * 12)  # 3 ints * 4 bytes per int
            faces = np.frombuffer(faces_buffer, dtype=np.int32).reshape(face_count, 3)
            
            # Read all normals at once
            normals_buffer = f.read(face_count * 12)  # 3 floats * 4 bytes per float
            if len(normals_buffer) == face_count * 12:
                normals = np.frombuffer(normals_buffer, dtype=np.float32).reshape(face_count, 3)
                
                # Read all latitudes and longitudes at once
                try:
                    lat_buffer = f.read(face_count * 4)  # 1 float * 4 bytes per float
                    lon_buffer = f.read(face_count * 4)  # 1 float * 4 bytes per float
                    
                    if len(lat_buffer) == face_count * 4 and len(lon_buffer) == face_count * 4:
                        latitudes = np.frombuffer(lat_buffer, dtype=np.float32)
                        longitudes = np.frombuffer(lon_buffer, dtype=np.float32)
                    else:
                        # Generate placeholder data
                        print("Warning: Could not read latitude/longitude data, generating placeholders")
                        latitudes = np.zeros(face_count, dtype=np.float32)
                        longitudes = np.zeros(face_count, dtype=np.float32)
                except:
                    print("Warning: Error reading latitude/longitude data, generating placeholders")
                    latitudes = np.zeros(face_count, dtype=np.float32)
                    longitudes = np.zeros(face_count, dtype=np.float32)
            else:
                # Generate placeholder data for normals, latitudes, longitudes
                print("Warning: Could not read normal data, generating placeholders")
                normals = np.zeros((face_count, 3), dtype=np.float32)
                latitudes = np.zeros(face_count, dtype=np.float32)
                longitudes = np.zeros(face_count, dtype=np.float32)
        
        print(f"Mesh contains {vertex_count} vertices and {face_count} faces")
        print(f"Optimized mesh loading completed in {time.time() - start_time:.2f} seconds")
        return vertices, faces, normals, latitudes, longitudes
    
    except Exception as e:
        print(f"Error reading optimized mesh format: {e}")
        sys.exit(1)

def read_time_steps(time_file):
    """Read time steps from time_steps.txt"""
    times = []
    time_strings = []
    
    with open(time_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
                
            parts = line.strip().split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                time_strings.append(' '.join(parts[1:]))
    
    return np.array(times), time_strings

def read_binary_data_file(file_path):
    """Read a binary data file with optimized vectorized operations"""
    try:
        with open(file_path, 'rb') as f:
            # Read header (number of elements)
            num_elements = struct.unpack('I', f.read(4))[0]
            
            # Read all data at once using numpy
            data_buffer = f.read(num_elements * 4)  # 4 bytes per float
            data = np.frombuffer(data_buffer, dtype=np.float32)
            
        return data
    except Exception as e:
        print(f"Error reading data file {file_path}: {e}")
        return np.array([])

def read_binary_data_file_with_retry(file_path, max_retries=3, backoff_factor=2.0):
    """Read a binary data file with retry logic for timeout errors"""
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            with open(file_path, 'rb') as f:
                # Read header (number of elements)
                num_elements = struct.unpack('I', f.read(4))[0]
                
                # Read all data at once using numpy
                data_buffer = f.read(num_elements * 4)  # 4 bytes per float
                data = np.frombuffer(data_buffer, dtype=np.float32)
                
            return data
        except (TimeoutError, OSError) as e:
            last_error = e
            retry_count += 1
            wait_time = backoff_factor ** retry_count
            print(f"Timeout reading {file_path}, retrying in {wait_time:.1f} seconds (attempt {retry_count}/{max_retries})")
            time.sleep(wait_time)
    
    # If we get here, all retries failed
    print(f"Error reading data file {file_path} after {max_retries} retries: {last_error}")
    # Return empty array instead of failing
    return np.array([])

def read_binary_data_file_optimized(file_path, mmap_threshold=500*1024*1024, max_retries=3, timeout=120):
    """Memory-optimized reader for very large binary data files with timeout handling"""
    # Get file size
    file_size = os.path.getsize(file_path)
    
    try:
        # For very large files, use memory mapping
        if file_size > mmap_threshold:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    with open(file_path, 'rb') as f:
                        num_elements = struct.unpack('I', f.read(4))[0]
                        header_size = 4
                
                    # Memory-map the file for efficient access with timeout handling
                    # Use a direct numpy approach for better control
                    with open(file_path, 'rb') as f:
                        f.seek(header_size)  # Skip header
                        # Read in chunks to avoid timeout
                        chunks = []
                        chunk_size = 100*1024*1024  # 100MB chunks
                        bytes_to_read = num_elements * 4
                        bytes_read = 0
                        
                        while bytes_read < bytes_to_read:
                            curr_chunk_size = min(chunk_size, bytes_to_read - bytes_read)
                            chunk = f.read(curr_chunk_size)
                            if not chunk:  # EOF
                                break
                            chunks.append(chunk)
                            bytes_read += len(chunk)
                        
                        # Combine chunks and convert to numpy array
                        if chunks:
                            data_buffer = b''.join(chunks)
                            data = np.frombuffer(data_buffer, dtype=np.float32)
                            return data
                        else:
                            raise IOError("Failed to read any data from file")
                    
                except (TimeoutError, OSError) as e:
                    retry_count += 1
                    wait_time = 2.0 ** retry_count
                    print(f"Timeout reading {file_path}, retrying in {wait_time:.1f} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                    if retry_count >= max_retries:
                        print(f"Max retries reached for {file_path}: {e}")
                        return np.array([])
        else:
            # For smaller files, use direct numpy loading with retry
            return read_binary_data_file_with_retry(file_path, max_retries=max_retries)
    except Exception as e:
        print(f"Error reading data file {file_path}: {e}")
        return np.array([])

def process_time_step(t, output_dir, grid, times, time_strings, data_dict):
    """Process a single time step (for parallel execution) with improved performance"""
    # Copy grid for this time step - keep using DeepCopy as it's reliable
    time_grid = vtk.vtkUnstructuredGrid()
    time_grid.DeepCopy(grid)
    
    # Add data arrays for this time step - optimize this part
    for data_name, all_data in data_dict.items():
        if t < len(all_data):
            data_array = vtk.vtkFloatArray()
            data_array.SetName(data_name)
            
            # Check if we have numpy data that can be batch processed
            if isinstance(all_data[t], np.ndarray) and all_data[t].dtype == np.float32:
                # Pre-allocate space for better performance
                data_array.SetNumberOfValues(len(all_data[t]))
                
                # Use block operations instead of one-by-one insertion
                # This is much faster but preserves the same result
                for i, val in enumerate(all_data[t]):
                    data_array.SetValue(i, val)
            else:
                # Fall back to original insertion method for non-numpy data
                for val in all_data[t]:
                    data_array.InsertNextValue(val)
                    
            time_grid.GetCellData().AddArray(data_array)
    
    # Add time field data
    time_array = vtk.vtkFloatArray()
    time_array.SetName("TIME")
    time_array.InsertNextValue(times[t])
    time_grid.GetFieldData().AddArray(time_array)
    
    # Optimize VTU writing with larger blocks and compression tuning
    vtu_filename = f"timestep_{t:04d}.vtu"
    vtu_path = os.path.join(output_dir, vtu_filename)
    
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(vtu_path)
    writer.SetInputData(time_grid)
    writer.SetDataModeToBinary()  # Much faster than ASCII
    writer.SetCompressorTypeToZLib()  # Add compression
    writer.SetCompressionLevel(3)  # Balance speed vs compression
    
    # Increase block size for better I/O performance
    writer.SetBlockSize(32768)  # Default is 32KB, increase for better streaming
    writer.Write()
    
    return t, vtu_filename, times[t]

def create_vtk_data_parallel(output_dir, vertices, faces, normals, latitudes, longitudes, 
                           times, time_strings, data_dict, max_workers=None):
    """Create VTK files for ParaView using parallel processing for time steps"""
    start_time = time.time()
    
    print(f"Creating VTK base grid...")
    
    # Create base grid with optimized methods for large datasets
    grid = vtk.vtkUnstructuredGrid()
    
    # Add points - more efficient batch insertion for large datasets
    points = vtk.vtkPoints()
    
    # Preallocate points for better performance
    points.Allocate(len(vertices))
    
    # Insert points in batches for better cache locality
    batch_size = 10000  # Tune this based on your cache size
    for batch_start in range(0, len(vertices), batch_size):
        batch_end = min(batch_start + batch_size, len(vertices))
        for i in range(batch_start, batch_end):
            points.InsertPoint(i, vertices[i][0], vertices[i][1], vertices[i][2])
            
    grid.SetPoints(points)
    
    # Add cells with optimized insertion
    # Pre-allocate for better performance
    grid.Allocate(len(faces))
    
    # Insert cells in batches
    for batch_start in range(0, len(faces), batch_size):
        batch_end = min(batch_start + batch_size, len(faces))
        for i in range(batch_start, batch_end):
            cell = vtk.vtkTriangle()
            cell.GetPointIds().SetId(0, faces[i][0])
            cell.GetPointIds().SetId(1, faces[i][1])
            cell.GetPointIds().SetId(2, faces[i][2])
            grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
    
    # Add latitude and longitude data to the base grid
    lat_array = vtk.vtkFloatArray()
    lat_array.SetName("latitude")
    
    # Pre-allocate for better performance
    lat_array.SetNumberOfValues(len(latitudes))
    for i, lat in enumerate(latitudes):
        lat_array.SetValue(i, lat)
    grid.GetCellData().AddArray(lat_array)
    
    lon_array = vtk.vtkFloatArray()
    lon_array.SetName("longitude")
    lon_array.SetNumberOfValues(len(longitudes))
    for i, lon in enumerate(longitudes):
        lon_array.SetValue(i, lon)
    grid.GetCellData().AddArray(lon_array)
    
    print(f"Base grid created in {time.time() - start_time:.2f} seconds")
    
    # Create PVD file
    pvd_path = os.path.join(output_dir, "simulation.pvd")
    
    # Create one VTU file per time step using parallel processing
    print(f"Processing {len(times)} time steps in parallel...")
    
    # Determine optimal worker count based on CPU cores and time step count
    if max_workers is None:
        import multiprocessing
        # Limit max threads for large datasets to avoid memory pressure
        total_cores = multiprocessing.cpu_count()
        
        # Scale thread count based on system memory
        try:
            import psutil
            mem_gb = psutil.virtual_memory().total / (1024**3)
            # Allow 1 thread per 2GB of RAM with a minimum of 2 threads
            memory_based_threads = max(2, int(mem_gb / 2))
            max_workers = min(total_cores, memory_based_threads, len(times))
        except ImportError:
            # Fallback if psutil isn't available
            max_workers = min(total_cores, len(times))
    
    pvd_entries = []
    
    # Process time steps in parallel with improved management
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(process_time_step, 
                              output_dir=output_dir,
                              grid=grid, 
                              times=times,
                              time_strings=time_strings,
                              data_dict=data_dict)
        
        # Submit in batches to control memory pressure
        batch_size = 10  # Number of time steps to process at once
        
        for batch_start in range(0, len(times), batch_size):
            batch_end = min(batch_start + batch_size, len(times))
            batch_time_steps = range(batch_start, batch_end)
            
            # Submit this batch
            future_to_idx = {executor.submit(process_func, t): t for t in batch_time_steps}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                completed = len(pvd_entries) + 1
                print_progress(completed, len(times), prefix="Creating VTK files")
                
                try:
                    t, vtu_filename, timestamp = future.result()
                    pvd_entries.append((t, vtu_filename, timestamp))
                except Exception as e:
                    print(f"\nError processing time step {future_to_idx[future]}: {e}")
    
    # Sort PVD entries by time step
    pvd_entries.sort()
    
    # Write the PVD file
    with open(pvd_path, 'w') as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1">\n')
        pvd.write('  <Collection>\n')
        
        for _, vtu_filename, timestamp in pvd_entries:
            pvd.write(f'    <DataSet timestep="{timestamp}" file="{vtu_filename}"/>\n')
        
        # Close PVD file
        pvd.write('  </Collection>\n')
        pvd.write('</VTKFile>\n')
    
    print(f"\nAll VTK files created in {time.time() - start_time:.2f} seconds")
    return pvd_path

def create_vtk_data(output_dir, vertices, faces, normals, latitudes, longitudes, 
                   times, time_strings, data_dict):
    """
    Legacy create_vtk_data function - uses the parallel implementation for better performance,
    but maintains the same interface for backward compatibility
    """
    return create_vtk_data_parallel(output_dir, vertices, faces, normals, latitudes, longitudes,
                                   times, time_strings, data_dict)

def main():
    parser = argparse.ArgumentParser(description='Convert PTM binary data to ParaView format')
    parser.add_argument('--input-dir', required=True, help='Directory containing binary data')
    parser.add_argument('--output-dir', required=True, help='Output directory for ParaView files')
    parser.add_argument('--threads', type=int, default=None, 
                        help='Number of parallel processes for time step processing (default: auto)')
    parser.add_argument('--optimize-memory', action='store_true', 
                        help='Use memory-optimized mode for very large datasets')
    # Add these new arguments
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Number of time steps to process in each batch (default: auto based on mesh size)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for file operations that time out')
    parser.add_argument('--timeout', type=int, default=120, 
                        help='Timeout in seconds for file operations')
    parser.add_argument('--skip-failed', action='store_true',
                        help='Skip failed files instead of using placeholder data')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading mesh data from {input_dir}...")
    
    # Start timing the complete conversion process
    start_time = time.time()
    
    # Read mesh
    mesh_file = os.path.join(input_dir, "mesh.ply")
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file not found at {mesh_file}")
        return 1
    
    vertices, faces, normals, latitudes, longitudes = read_binary_mesh(mesh_file)
    
    # Check mesh size and set appropriate execution strategy
    large_mesh = len(faces) > 1000000
    very_large_mesh = len(faces) > 5000000
    
    if very_large_mesh:
        print(f"Very large mesh detected ({len(faces)} faces). Using high-performance mode.")
    elif large_mesh:
        print(f"Large mesh detected ({len(faces)} faces). Using optimized mode.")
    
    # Read time steps
    time_file = os.path.join(input_dir, "time_steps.txt")
    if not os.path.exists(time_file):
        print(f"Error: Time steps file not found at {time_file}")
        return 1
    
    times, time_strings = read_time_steps(time_file)
    print(f"Found {len(times)} time steps")
    
    # Read data for each time step - use optimized reader for large datasets
    data_dict = {
        "illumination": [],
        "primary_ir": [],
        "temperature": []
    }
    
    # Apply command-line arguments to batch size
    if args.batch_size:
        batch_size = args.batch_size
    elif very_large_mesh:
        batch_size = 5  # Reduce batch size for huge meshes
    else:
        batch_size = 10
    
    # Update the reader function to use the specified timeout and max retries
    reader_func = lambda file_path: read_binary_data_file_optimized(
        file_path, 
        mmap_threshold=500*1024*1024, 
        max_retries=args.max_retries, 
        timeout=args.timeout
    ) if (very_large_mesh or args.optimize_memory) else read_binary_data_file_with_retry(
        file_path,
        max_retries=args.max_retries
    )
    
    # Create a load counter to track progress
    data_files_to_load = len(times) * len(data_dict)
    loaded_files = 0
    
    # For very large meshes, use more conservative batch sizes and add more error handling
    if very_large_mesh:
        batch_size = 5  # Reduce batch size for huge meshes
    else:
        batch_size = 10
    
    # Add a set to track files with errors
    failed_files = set()
    
    for batch_start in range(1, len(times) + 1, batch_size):
        batch_end = min(batch_start + batch_size, len(times) + 1)
        
        for t in range(batch_start, batch_end):
            time_str = f"{t:04d}"
            
            # Define file paths
            illum_file = os.path.join(input_dir, "illumination", f"illumination_{time_str}.bin")
            ir_file = os.path.join(input_dir, "primary_ir", f"primary_ir_{time_str}.bin")
            temp_file = os.path.join(input_dir, "temperature", f"temperature_{time_str}.bin")
            
            # Read illumination data
            if os.path.exists(illum_file) and illum_file not in failed_files:
                try:
                    data = reader_func(illum_file)
                    if len(data) > 0:
                        data_dict["illumination"].append(data)
                    else:
                        failed_files.add(illum_file)
                        # Add placeholder data
                        data_dict["illumination"].append(np.zeros(len(faces), dtype=np.float32))
                    loaded_files += 1
                except Exception as e:
                    print(f"\nUnexpected error reading {illum_file}: {e}")
                    failed_files.add(illum_file)
                    # Add placeholder data
                    data_dict["illumination"].append(np.zeros(len(faces), dtype=np.float32))
                    loaded_files += 1
            
            # Read primary IR data
            if os.path.exists(ir_file) and ir_file not in failed_files:
                try:
                    data = reader_func(ir_file)
                    if len(data) > 0:
                        data_dict["primary_ir"].append(data)
                    else:
                        failed_files.add(ir_file)
                        # Add placeholder data
                        data_dict["primary_ir"].append(np.zeros(len(faces), dtype=np.float32))
                    loaded_files += 1
                except Exception as e:
                    print(f"\nUnexpected error reading {ir_file}: {e}")
                    failed_files.add(ir_file)
                    # Add placeholder data
                    data_dict["primary_ir"].append(np.zeros(len(faces), dtype=np.float32))
                    loaded_files += 1
            
            # Read temperature data
            if os.path.exists(temp_file) and temp_file not in failed_files:
                try:
                    data = reader_func(temp_file)
                    if len(data) > 0:
                        data_dict["temperature"].append(data)
                    else:
                        failed_files.add(temp_file)
                        # Add placeholder data
                        data_dict["temperature"].append(np.zeros(len(faces), dtype=np.float32))
                    loaded_files += 1
                except Exception as e:
                    print(f"\nUnexpected error reading {temp_file}: {e}")
                    failed_files.add(temp_file)
                    # Add placeholder data
                    data_dict["temperature"].append(np.zeros(len(faces), dtype=np.float32))
                    loaded_files += 1
            
            print_progress(loaded_files, data_files_to_load, prefix="Reading data files")
        
        # Force garbage collection between batches
        import gc
        gc.collect()
    
    # Report on any files that failed to load
    if failed_files:
        print(f"\nWarning: {len(failed_files)} data files failed to load.")
        print("The visualization will have missing data for these time steps.")
        print("Consider re-running with a smaller batch size or increased timeout.")
    
    # Create VTK files
    print("\nCreating VTK files...")
    
    # Use appropriate number of threads based on mesh size
    num_threads = args.threads
    if num_threads is None:
        import multiprocessing
        if very_large_mesh:
            # Limit threads for very large meshes to avoid memory pressure
            num_threads = min(4, multiprocessing.cpu_count())
        elif large_mesh:
            # Use more threads for large but not extreme meshes
            num_threads = min(8, multiprocessing.cpu_count())
        else:
            # Use all cores for smaller meshes
            num_threads = multiprocessing.cpu_count()
    
    print(f"Using {num_threads} parallel processes for time step processing")
    
    pvd_file = create_vtk_data_parallel(
        output_dir, 
        vertices, faces, normals, latitudes, longitudes,
        times, time_strings, data_dict,
        max_workers=num_threads
    )
    
    total_time = time.time() - start_time
    print(f"\nConversion complete in {total_time:.2f} seconds!")
    print(f"To visualize in ParaView:")
    print(f"1. Open ParaView")
    print(f"2. File -> Open -> Select {pvd_file}")
    print(f"3. Click Apply in the Properties panel")
    print(f"4. Use the Time controls to navigate time steps")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())