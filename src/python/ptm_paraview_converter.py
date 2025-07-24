#!/usr/bin/env python3
# filepath: ptm_paraview_converter.py
import os
import sys
import struct
import numpy as np
import argparse
from pathlib import Path
import time

# Import ParaView modules - these need to be available in your Python environment
try:
    from paraview.simple import *
    from vtk.util.numpy_support import numpy_to_vtk
except ImportError:
    print("ERROR: ParaView Python modules not found.")
    print("Make sure to run this script with the Python that comes with ParaView.")
    print("Typically: /Applications/ParaView-X.X.X.app/Contents/bin/pvpython")
    sys.exit(1)

def print_progress(current, total, prefix='Progress:', suffix='Complete', decimals=1, length=50, fill='█'):
    """
    Print a progress bar to the console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')

def read_binary_mesh(mesh_file):
    """Read binary PLY mesh file"""
    vertices = []
    faces = []
    normals = []
    latitudes = []
    longitudes = []
    
    with open(mesh_file, 'rb') as f:
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
        
        # Read vertices
        for _ in range(vertex_count):
            x = struct.unpack('f', f.read(4))[0]
            y = struct.unpack('f', f.read(4))[0]
            z = struct.unpack('f', f.read(4))[0]
            vertices.append((x, y, z))
        
        # Read faces
        face_vertex_indices = []
        for _ in range(face_count):
            count = struct.unpack('B', f.read(1))[0]  # Number of vertices in face
            if count != 3:
                print(f"Warning: Face with {count} vertices found (expected 3)")
            indices = []
            for _ in range(count):
                idx = struct.unpack('i', f.read(4))[0]
                indices.append(idx)
            faces.append(indices[:3])  # Take only first 3 for non-triangular faces
    
    # Calculate face centers, normals, and lat/lon
    for face in faces:
        v1 = np.array(vertices[face[0]])
        v2 = np.array(vertices[face[1]])
        v3 = np.array(vertices[face[2]])
        
        # Calculate face center
        center = (v1 + v2 + v3) / 3.0
        
        # Calculate face normal
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)
        normals.append(normal)
        
        # Calculate latitude and longitude
        r = np.linalg.norm(center)
        lat = np.arcsin(center[2] / r) * 180.0 / np.pi
        lon = np.arctan2(center[1], center[0]) * 180.0 / np.pi
        latitudes.append(lat)
        longitudes.append(lon)
    
    return np.array(vertices), np.array(faces), np.array(normals), np.array(latitudes), np.array(longitudes)

def read_time_steps(time_file):
    """Read time steps from time_steps.txt"""
    times = []
    time_strings = []
    
    with open(time_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                times.append(float(parts[0]))
                time_strings.append(parts[1])
    
    return np.array(times), time_strings

def read_data_files(data_dir, pattern, face_count, time_count):
    """Read all binary data files for a given pattern"""
    data = np.zeros((face_count, time_count), dtype=np.float32)
    
    for t in range(1, time_count+1):
        # Format time step with leading zeros
        time_str = f"{t:04d}"
        filename = os.path.join(data_dir, f"{pattern}_{time_str}.bin")
        
        if not os.path.exists(filename):
            print(f"Warning: Missing file {filename}")
            continue
        
        with open(filename, 'rb') as f:
            # Read the entire file as float32 values
            values = np.fromfile(f, dtype=np.float32)
            if len(values) != face_count:
                print(f"Warning: Expected {face_count} values but got {len(values)} in {filename}")
                continue
            
            # Store data for this time step
            data[:, t-1] = values
    
    return data

def create_paraview_data(output_dir, vertices, faces, normals, latitudes, longitudes, times, 
                        time_strings, illumination_data, primary_ir_data, temperature_data):
    """Create ParaView data files directly using ParaView Python API"""
    
    # Create a new UnstructuredGrid
    print("Creating VTK data structures...")
    grid = vtk.vtkUnstructuredGrid()
    
    # Add points
    points = vtk.vtkPoints()
    for x, y, z in vertices:
        points.InsertNextPoint(x, y, z)
    grid.SetPoints(points)
    
    # Add cells (triangles)
    for v1, v2, v3 in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, v1)
        triangle.GetPointIds().SetId(1, v2)
        triangle.GetPointIds().SetId(2, v3)
        grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())
    
    # Add latitude and longitude as cell data
    lat_array = numpy_to_vtk(latitudes.astype(np.float32))
    lat_array.SetName("latitude")
    grid.GetCellData().AddArray(lat_array)
    
    lon_array = numpy_to_vtk(longitudes.astype(np.float32))
    lon_array.SetName("longitude")
    grid.GetCellData().AddArray(lon_array)
    
    # Create a PVD file for the time series
    pvd_file = os.path.join(output_dir, "ptm_data.pvd")
    with open(pvd_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')
    
    # Save each time step as a separate VTU file
    print(f"Creating {len(times)} time step files...")
    for t in range(len(times)):
        # Create a copy of the grid for this time step
        time_grid = vtk.vtkUnstructuredGrid()
        time_grid.DeepCopy(grid)
        
        # Add data arrays for this time step
        illum_array = numpy_to_vtk(illumination_data[:, t].astype(np.float32))
        illum_array.SetName("illumination")
        time_grid.GetCellData().AddArray(illum_array)
        
        ir_array = numpy_to_vtk(primary_ir_data[:, t].astype(np.float32))
        ir_array.SetName("primary_ir")
        time_grid.GetCellData().AddArray(ir_array)
        
        temp_array = numpy_to_vtk(temperature_data[:, t].astype(np.float32))
        temp_array.SetName("temperature")
        time_grid.GetCellData().AddArray(temp_array)
        
        # Add time value as field data
        time_array = vtk.vtkFloatArray()
        time_array.SetName("TIME")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, times[t])
        time_grid.GetFieldData().AddArray(time_array)
        
        # Set active arrays
        time_grid.GetCellData().SetActiveScalars("temperature")
        
        # Write VTU file
        vtu_filename = os.path.join(output_dir, f"timestep_{t:04d}.vtu")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(vtu_filename)
        writer.SetInputData(time_grid)
        writer.SetDataModeToAscii()  # Use ASCII mode for better debugging
        writer.Write()
        
        # Add to PVD collection
        with open(pvd_file, 'a') as f:
            f.write(f'    <DataSet timestep="{times[t]}" file="{os.path.basename(vtu_filename)}"/>\n')
        
        # Show progress
        print_progress(t+1, len(times))
    
    # Close the PVD file
    with open(pvd_file, 'a') as f:
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    
    return pvd_file

def main():
    parser = argparse.ArgumentParser(description='Convert PTM binary data to ParaView format')
    parser.add_argument('--input-dir', required=True, help='Directory containing binary data')
    parser.add_argument('--output-dir', required=True, help='Output directory for ParaView files')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    print(f"PTM Binary to ParaView Converter")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read mesh
    mesh_file = os.path.join(input_dir, 'mesh.ply')
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file not found: {mesh_file}")
        return 1
    
    print(f"Reading mesh from {mesh_file}...")
    start_time = time.time()
    vertices, faces, normals, latitudes, longitudes = read_binary_mesh(mesh_file)
    print(f"Read mesh with {len(vertices)} vertices and {len(faces)} faces in {time.time() - start_time:.2f} seconds")
    
    # Read time steps
    time_file = os.path.join(input_dir, 'time_steps.txt')
    if not os.path.exists(time_file):
        print(f"Error: Time steps file not found: {time_file}")
        return 1
    
    print(f"Reading time steps from {time_file}...")
    times, time_strings = read_time_steps(time_file)
    print(f"Read {len(times)} time steps")
    
    # Read data for each data type
    face_count = len(faces)
    time_count = len(times)
    
    print(f"Reading illumination data...")
    illumination_data = read_data_files(os.path.join(input_dir, 'illumination'), 
                                       'illumination', face_count, time_count)
    
    print(f"Reading primary IR data...")
    primary_ir_data = read_data_files(os.path.join(input_dir, 'primary_ir'), 
                                     'primary_ir', face_count, time_count)
    
    print(f"Reading temperature data...")
    temperature_data = read_data_files(os.path.join(input_dir, 'temperature'), 
                                      'temperature', face_count, time_count)
    
    # Create ParaView data
    print(f"Creating ParaView data files...")
    pvd_file = create_paraview_data(
        output_dir, vertices, faces, normals, latitudes, longitudes, 
        times, time_strings, illumination_data, primary_ir_data, temperature_data
    )
    
    print(f"\nConversion complete!")
    print(f"To visualize the data:")
    print(f"1. Open ParaView")
    print(f"2. File -> Open -> Select {pvd_file}")
    print(f"3. Click 'Apply' in the Properties panel")
    print(f"4. Use the Time control to animate through time steps")
    print(f"5. Select different arrays (temperature, illumination, primary_ir) in the Properties panel")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())