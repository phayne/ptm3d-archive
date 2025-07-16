#!/usr/bin/env python3

# NumPy compatibility fix - add before other imports
import numpy
if not hasattr(numpy, 'bool'):
    numpy.bool = bool

import os
import sys
import math
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import argparse
import imageio
from matplotlib import cm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from matplotlib.ticker import MaxNLocator

class EuropaVisualizer:
    def __init__(self, verbose=False, width=1024, height=768):
        self.mesh = None
        self.thermal_mesh = None
        self.time_data = None
        self.time_steps = 0
        self.jupiter_positions = None
        self.sun_positions = None
        self.illumination_model = None
        self.verbose = verbose
        self.width = width
        self.height = height
        self._plotter_cache = {}
        self._actor_cache = {}

    def load_vtk(self, vtk_file):
        """Load VTK file with Europa illumination data"""
        if self.verbose:
            print(f"Loading {vtk_file}...")
        try:
            self.mesh = pv.read(vtk_file)
            # Store the file path for later direct access if needed
            self.mesh.path = vtk_file
            # Find arrays named illumination_0, illumination_1, etc.
            illumination_arrays = [key for key in self.mesh.array_names if key.startswith('illumination_')]
            self.time_steps = len(illumination_arrays)
            if self.verbose:
                print(f"Loaded VTK with {self.time_steps} time steps")
                print(f"Mesh has {self.mesh.n_points} points and {self.mesh.n_cells} cells")
            
            # Check if mesh needs to be centered
            center = self.mesh.center
            
            # If the mesh is significantly off-center, recenter it
            if np.linalg.norm(center) > 0.1:
                if self.verbose:
                    print(f"Mesh is off-center: {center}, recentering...")
                self.mesh.points -= center
                if self.verbose:
                    print(f"New mesh center: {self.mesh.center}")
            
            return True
        except Exception as e:
            print(f"Error loading VTK file: {e}")
            return False

    def load_thermal_vtk(self, vtk_file):
        """Load VTK file with Europa thermal data"""
        if self.verbose:
            print(f"Loading thermal VTK: {vtk_file}...")
        try:
            self.thermal_mesh = pv.read(vtk_file)
            # Find arrays named temperature_0, temperature_1, etc.
            temperature_arrays = [key for key in self.thermal_mesh.array_names if key.startswith('temperature_')]
            
            if len(temperature_arrays) != self.time_steps:
                print(f"Warning: Found {len(temperature_arrays)} temperature arrays but expected {self.time_steps}")
            
            if self.verbose:
                print(f"Loaded thermal VTK with {len(temperature_arrays)} temperature steps")
                print(f"Thermal mesh has {self.thermal_mesh.n_points} points and {self.thermal_mesh.n_cells} cells")
            
            return True
        except Exception as e:
            print(f"Error loading thermal VTK file: {e}")
            return False
            
    def extract_body_positions_from_vtk(self):
        """Extract Jupiter and Sun positions from VTK data"""
        print("Extracting body positions from VTK data...")

        # Simplify this method to just initialize empty position arrays
        # since we can't dynamically calculate positions without the module
        if not self.sun_positions or not self.jupiter_positions:
            print("No position data found in VTK. Please use --positions parameter.")
            # Initialize empty lists if they're not set at all
            if not self.sun_positions:
                self.sun_positions = [None] * self.time_steps
            if not self.jupiter_positions:
                self.jupiter_positions = [None] * self.time_steps
            return False
        
        return True

    def make_debug_image(self, output_file):
        """Create a single debug image to verify mesh visibility"""
        if self.mesh is None:
            print("No mesh data loaded")
            return False
            
        # Create a debug plotter with basic settings - use instance variables
        plotter = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        plotter.set_background('black')
        
        # Add axes for orientation
        plotter.add_axes(interactive=False, line_width=2)
        
        # Add the mesh with multiple display options to ensure visibility
        mesh = self.mesh.copy()
        if 'illumination_0' in mesh.array_names:
            mesh['illumination'] = mesh['illumination_0']
            mesh.set_active_scalars('illumination')
            
            # Add with color map
            plotter.add_mesh(mesh, 
                            scalars='illumination', 
                            cmap='inferno',
                            show_edges=False,
                            edge_color='White',
                            lighting=True,
                            clim=[0, 50],
                            scalar_bar_args={'title': 'Illumination (W/m²)'})
            
            # Also add wireframe for better visibility
            #plotter.add_mesh(mesh.extract_all_edges(), color='white', line_width=0.01, opacity=0.1)
        else:
            # Add with solid color if no illumination data
            plotter.add_mesh(mesh, color='white', show_edges=True)
        
        # Set a default view that should work
        plotter.view_isometric()
        
        # Save the image
        print(f"Saving debug image to {output_file}")
        plotter.screenshot(output_file)
        return True
    
    def render_single_frame(self, frame_idx, prefix, array_name, cmap, clim, camera_distance,
                      camera_latitude, camera_longitude, emission_angle, azimuth,
                      time_strings, show_jupiter, show_sun, mesh_radius, save_frames, frames_dir):
        """Render a single frame in a separate process"""
        
        # Create a new plotter in this process
        plotter = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        plotter.set_background('black')
        
        # Get the mesh data for this frame
        mesh_copy = self.mesh.copy()
        data_name = f'{prefix}_{frame_idx}'
        if data_name in mesh_copy.array_names:
            # Set up mesh and scalars
            mesh_copy[array_name] = mesh_copy[data_name]
            mesh_copy.set_active_scalars(array_name)
            
            # Add mesh with proper configuration
            plotter.add_mesh(
                mesh_copy,
                scalars=array_name, 
                cmap=cmap,
                show_edges=False,
                lighting=True,
                clim=clim
            )
            
            # Set up camera view
            if emission_angle is not None and azimuth is not None:
                self.setup_camera_emission_azimuth(plotter, emission_angle, azimuth, camera_distance)
            else:
                self.setup_camera_for_mesh(
                    plotter=plotter,
                    camera_distance=camera_distance,
                    camera_latitude=camera_latitude, 
                    camera_longitude=camera_longitude
                )
            
            # Add timestamp
            if frame_idx < len(time_strings) and time_strings[frame_idx]:
                timestamp_str = str(time_strings[frame_idx])
                if isinstance(time_strings[frame_idx], bytes):
                    timestamp_str = time_strings[frame_idx].decode('utf-8')
                timestamp_text = f"UTC: {timestamp_str}"
            else:
                timestamp_text = f"Time Step: {frame_idx}"
                
            plotter.add_text(timestamp_text, position='upper_left', 
                            font_size=12, color='white')
            
            # Add Jupiter and Sun indicators if needed
            if frame_idx < len(self.sun_positions) and frame_idx < len(self.jupiter_positions):
                if show_jupiter and self.jupiter_positions[frame_idx] is not None:
                    # Add Jupiter indicators
                    jup_pos = np.array(self.jupiter_positions[frame_idx])
                    jup_dir = jup_pos / np.linalg.norm(jup_pos)
                    jup_end = jup_dir * mesh_radius * 1.2
                    
                    plotter.add_mesh(
                        pv.Line(np.array([0, 0, 0]), jup_end), 
                        color='orange', 
                        line_width=2
                    )
                    
                    plotter.add_point_labels(
                        [jup_end], 
                        ["Jupiter"], 
                        text_color='orange', 
                        font_size=12,
                        bold=True,
                        shadow=True
                    )
                
                if show_sun and self.sun_positions[frame_idx] is not None:
                    # Add Sun indicators
                    sun_pos = np.array(self.sun_positions[frame_idx])
                    sun_dir = sun_pos / np.linalg.norm(sun_pos)
                    sun_end = sun_dir * mesh_radius * 1.2
                    
                    plotter.add_mesh(
                        pv.Line(np.array([0, 0, 0]), sun_end), 
                        color='yellow', 
                        line_width=2
                    )
                    
                    plotter.add_point_labels(
                        [sun_end], 
                        ["Sun"], 
                        text_color='yellow', 
                        font_size=12, 
                        bold=True,
                        shadow=True
                    )
            
            # Render the frame
            frame = plotter.screenshot(return_img=True)
            
            # Save individual frame if requested
            if save_frames and frames_dir:
                frame_path = os.path.join(frames_dir, f"{prefix}_frame_{frame_idx:04d}.png")
                plotter.screenshot(frame_path)
            
            # Clean up
            plotter.close()
            return frame
        
        return None

    def create_animation(self, output_file, camera_distance=5.0, camera_latitude=30.0, 
                  camera_longitude=0.0, emission_angle=None, azimuth=None, 
                  fps=10, save_frames=False):
        """Create a 3D animation of Europa illumination"""
        return self.create_animation_generic_parallel(
            output_file, 
            data_type='illumination',
            camera_distance=camera_distance, 
            camera_latitude=camera_latitude, 
            camera_longitude=camera_longitude,
            emission_angle=emission_angle,
            azimuth=azimuth, 
            fps=fps,
            save_frames=save_frames
        )

    def create_ir_animation(self, output_file, camera_distance=5.0, camera_latitude=30.0, 
                           camera_longitude=0.0, emission_angle=None, azimuth=None, 
                           fps=10, save_frames=False):
        """Create a 3D animation of Jupiter IR flux on Europa"""
        return self.create_animation_generic(
            output_file, 
            data_type='jupiter_ir',
            camera_distance=camera_distance, 
            camera_latitude=camera_latitude, 
            camera_longitude=camera_longitude,
            emission_angle=emission_angle,  
            azimuth=azimuth,                
            fps=fps,
            save_frames=save_frames
        )

    def create_thermal_animation(self, output_file, camera_distance=5.0, camera_latitude=30.0, 
                                camera_longitude=0.0, emission_angle=None, azimuth=None, 
                                fps=10, save_frames=False):
        """Create a 3D animation of Europa surface temperatures"""
        if self.thermal_mesh is None:
            print("Error: No thermal mesh data available")
            return False
        
        # Store original mesh
        original_mesh = self.mesh
        
        # Temporarily swap to thermal mesh
        self.mesh = self.thermal_mesh
        
        # Check if we need to recenter the mesh for better viewing
        bounds = self.mesh.bounds
        if abs(bounds[0]) > 10000 or abs(bounds[1]) > 10000:
            print("Warning: Large coordinate values detected in thermal mesh.")
            print(f"Mesh bounds: {bounds}")
            print("Consider normalizing your mesh coordinates for better visualization.")
        
        # Use the existing animation function with adapted camera distance
        result = self.create_animation_generic(
            output_file, 
            data_type='temperature',
            camera_distance=camera_distance,
            camera_latitude=camera_latitude, 
            camera_longitude=camera_longitude,
            emission_angle=emission_angle,
            azimuth=azimuth,
            fps=fps,
            save_frames=save_frames 
        )
        
        # Restore original mesh
        self.mesh = original_mesh
        return result

    def create_cylindrical_maps(self, output_dir):
        """Create cylindrical map projections with optimized performance"""
        print(f"Creating cylindrical maps in {output_dir}...")
        if self.mesh is None:
            print("No mesh data loaded")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract positions if not already done
        if self.jupiter_positions is None:
            self.extract_body_positions_from_vtk()
            
        # Extract lat/lon data once
        lat = self.mesh['latitude']
        lon = self.mesh['longitude']
        
        print("Creating grid for faster rendering...")
        # Create a regular grid for pcolormesh
        grid_resolution = 1.0  # degrees
        lon_grid = np.arange(-180, 180.1, grid_resolution)
        lat_grid = np.arange(-90, 90.1, grid_resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # PRE-COMPUTE INTERPOLATION POINTS - this is the key optimization
        points = np.column_stack((lon, lat))
        xi = np.column_stack((lon_mesh.flatten(), lat_mesh.flatten()))
        
        # Create the figure once and reuse it
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Setup colorbar once
        norm = plt.Normalize(0, 50)  # Adjust as needed
        pcm = ax.pcolormesh(lon_grid, lat_grid, np.zeros_like(lon_mesh), 
                           cmap='inferno', norm=norm)
        cbar = fig.colorbar(pcm)
        cbar.set_label('Illumination (W/m²)')
        
        # Setup fixed elements
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        
        # Jupiter and Sun point references
        jup_point, = ax.plot([], [], 'o', color='orange', markersize=8, label='Sub-Jupiter')
        sun_point, = ax.plot([], [], 'o', color='yellow', markersize=8, label='Sub-Solar')
        ax.legend()
        
        # Title object
        title = ax.set_title('')
        
        # Use LinearNDInterpolator instead of griddata for better performance
        from scipy.interpolate import LinearNDInterpolator
        
        # Progress bar
        from tqdm import tqdm
        
        # Process each time step
        for i in tqdm(range(self.time_steps), desc="Creating maps"):
            # Get current illumination array
            illumination_name = f'illumination_{i}'
            if illumination_name in self.mesh.array_names:
                illumination = self.mesh[illumination_name]
                
                # Create interpolator - reuse points from above
                interp = LinearNDInterpolator(points, illumination, fill_value=0)
                
                # Interpolate values - much faster than griddata
                grid_z = interp(xi).reshape(lon_mesh.shape)
                
                # Update the pcolormesh data - don't recreate it
                pcm.set_array(grid_z.ravel())
                
                # Update sub-Jupiter and sub-Solar points
                jup_pos = self.jupiter_positions[i]
                sun_pos = self.sun_positions[i]
                
                # Convert to lat/lon
                jup_lon = np.degrees(np.arctan2(jup_pos[1], jup_pos[0]))
                jup_lat = np.degrees(np.arcsin(jup_pos[2] / np.linalg.norm(jup_pos)))
                
                sun_lon = np.degrees(np.arctan2(sun_pos[1], sun_pos[0]))
                sun_lat = np.degrees(np.arcsin(sun_pos[2] / np.linalg.norm(sun_pos)))
                
                # Update point positions
                jup_point.set_data(jup_lon, jup_lat)
                sun_point.set_data(sun_lon, sun_lat)
                
                # Update title
                title.set_text(f'Europa Illumination - Time Step {i}')
                
                # Save figure - reuse the same figure
                map_file = os.path.join(output_dir, f"map_{i:04d}.png")
                fig.savefig(map_file, dpi=200, bbox_inches='tight')
                
            else:
                print(f"Warning: No illumination data for time step {i}")
                
        # Close the figure at the end
        plt.close(fig)
        return True

    def create_cylindrical_maps_parallel(self, output_dir, max_workers=None):
        """Create cylindrical map projections using multiprocessing for speed"""
        print(f"Creating cylindrical maps in parallel in {output_dir}...")
        if self.mesh is None:
            print("No mesh data loaded")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract positions if not already done
        if self.jupiter_positions is None:
            self.extract_body_positions_from_vtk()
        
        # Extract lat/lon and illumination data
        lat_values = self.mesh['latitude']
        lon_values = self.mesh['longitude']
        
        # Prepare illumination data for each frame
        illuminations = []
        for i in range(self.time_steps):
            illumination_name = f'illumination_{i}'
            if illumination_name in self.mesh.array_names:
                illuminations.append(self.mesh[illumination_name])
            else:
                print(f"Warning: No illumination data for time step {i}")
                illuminations.append(np.zeros_like(lat_values))  # Empty frame
        
        # Use process pool for parallel execution
        import multiprocessing
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        print(f"Starting parallel processing with {max_workers} workers...")
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Process each frame with individual arguments (not using wrapper)
                futures = []
                for i in range(self.time_steps):
                    future = executor.submit(
                        _create_map_for_frame,
                        i,                        # Frame index
                        lon_values,               # Longitude values
                        lat_values,               # Latitude values
                        illuminations[i],         # Illumination data
                        self.jupiter_positions[i], # Jupiter position
                        self.sun_positions[i],    # Sun position
                        output_dir,               # Output directory
                        1.0,                      # Grid resolution
                        (0, 50)                   # Color limits
                    )
                    futures.append(future)
                
                # Wait for all to complete and gather results
                results = []
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Error in worker: {e}")
                
            print(f"Created {len(results)} maps in parallel")
            return True
        except Exception as e:
            print(f"Error during parallel processing: {e}")
            import traceback
            traceback.print_exc()
            return False

    def debug_visualization(self, output_file):
        """Create a simple diagnostic visualization to debug mesh issues"""
        print(f"Creating debug visualization at {output_file}...")
        if self.mesh is None:
            print("No mesh data")
            return False
            
        # Create plotter in off-screen mode to avoid window issues
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background('black')
        
        # Get the mesh to display
        mesh_to_show = self.mesh
            
        # Show normal mesh with illumination
        try:
            if 'illumination_0' in mesh_to_show.array_names:
                mesh_to_show['illumination'] = mesh_to_show['illumination_0']
                mesh_to_show.set_active_scalars('illumination')
            elif 'illumination' in mesh_to_show.array_names:
                mesh_to_show.set_active_scalars('illumination')
            
            plotter.add_mesh(mesh_to_show, show_edges=True, point_size=5)
        except:
            # Fallback to wireframe if there are issues
            plotter.add_mesh(mesh_to_show, style='wireframe', color='white')
            plotter.add_points(mesh_to_show.points, color='red', point_size=5)
        
        # Add helpful visualization elements
        plotter.add_axes()
        plotter.add_bounding_box()
        
        # Add sphere at origin for reference
        origin_sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
        plotter.add_mesh(origin_sphere, style='wireframe', color='green', opacity=0.5)
        
        # Save the image
        plotter.screenshot(output_file)
        return True

    def load_positions(self, positions_file):
        """Load Sun and Jupiter positions from CSV file"""
        print(f"Loading positions from {positions_file}...")
        try:
            df = pd.read_csv(positions_file)
            
            # Extract columns
            self.time_data = df['time'].values
            
            # Extract Sun positions
            self.sun_positions = []
            for i in range(len(df)):
                self.sun_positions.append([
                    df['sun_x'].values[i],
                    df['sun_y'].values[i],
                    df['sun_z'].values[i]
                ])
            
            # Extract Jupiter positions
            self.jupiter_positions = []
            for i in range(len(df)):
                self.jupiter_positions.append([
                    df['jupiter_x'].values[i],
                    df['jupiter_y'].values[i],
                    df['jupiter_z'].values[i]
                ])
            
            print(f"Loaded {len(self.sun_positions)} position entries")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load positions file: {e}")
            print("Cannot proceed without valid position data.")
            return False

    def visualize_ir_flux(self, time_step=0):
        """Create visualization of Jupiter IR flux"""
        if self.mesh is None:
            print("No mesh data loaded")
            return False
            
        # Create a plotter
        plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
        plotter.set_background('black')
        
        # Add the mesh with IR data
        mesh = self.mesh.copy()
        ir_field_name = f'jupiter_ir_{time_step}'
        
        if ir_field_name in mesh.array_names:
            mesh[ir_field_name] = mesh[ir_field_name]
            mesh.set_active_scalars(ir_field_name)
            
            # Add with color map
            plotter.add_mesh(mesh, 
                            scalars=ir_field_name, 
                            cmap='inferno',
                            show_edges=False,
                            edge_color='White',
                            lighting=True,
                            clim=[0, 10],  # Adjust this range as needed
                            scalar_bar_args={'title': 'Jupiter IR Flux (W/m²)'})
            
            # Set a default view
            plotter.view_isometric()
            
            # Display
            plotter.show()
            return True
        else:
            print(f"IR flux data '{ir_field_name}' not found")
            return False

    def create_animation_generic(self, output_file, data_type='illumination', 
                      camera_distance=5.0, camera_latitude=30.0, 
                      camera_longitude=0.0, emission_angle=None, azimuth=None, fps=10, save_frames=False):
        """Create a 3D animation of Europa data with optimized object caching for better performance"""
        # Print just once at the beginning
        print(f"Creating {data_type} animation {output_file}...")
        print(f"Camera settings: lat={camera_latitude}°, lon={camera_longitude}°, distance={camera_distance}")
        
        if self.mesh is None:
            print("No mesh data loaded")
            return False
        
        # Configure based on data type
        if data_type == 'illumination':
            prefix = 'illumination'
            array_name = 'illumination'
            cmap = 'inferno'
            title = 'Illumination (W/m²)'
            show_sun = True
            show_jupiter = True
            frames_dirname = "frames"
            
            # Calculate dynamic range for solar illumination
            max_val = 0
            for i in range(self.time_steps):
                data_name = f'{prefix}_{i}'
                if data_name in self.mesh.array_names:
                    max_val = max(max_val, np.max(self.mesh[data_name]))
            
            # Round up to nearest 5 W/m²
            upper_bound = 5 * math.ceil(max_val / 5)
            if upper_bound < 5:  # Ensure minimum scale
                upper_bound = 5
            
            clim = [0, upper_bound]
            print(f"Dynamic solar illumination range: {clim}")
            
        elif data_type == 'jupiter_ir':
            prefix = 'jupiter_ir'
            array_name = 'ir_flux'
            cmap = 'plasma'
            title = 'Jupiter IR Flux (W/m²)'
            show_sun = False
            show_jupiter = True
            frames_dirname = "ir_frames"
            
            # Calculate dynamic range for IR flux
            max_val = 0
            for i in range(self.time_steps):
                data_name = f'{prefix}_{i}'
                if data_name in self.mesh.array_names:
                    max_val = max(max_val, np.max(self.mesh[data_name]))
            
            # Use the nice scale method
            upper_bound = self.get_nice_scale_ceiling(max_val)
            
            clim = [0, upper_bound]
            print(f"Dynamic Jupiter IR flux range: {clim}")
            
        elif data_type == 'temperature':
            prefix = 'temperature'
            array_name = 'temperature'
            cmap = 'plasma'
            title = 'Temperature (K)'
            show_sun = True
            show_jupiter = True
            frames_dirname = "thermal_frames"
            
            if self.verbose:
                print(f"Setting up temperature visualization")
            
            # Check for temperature arrays
            temp_keys = [key for key in self.mesh.array_names if key.startswith('temperature_')]
            if self.verbose:
                print(f"Found {len(temp_keys)} temperature arrays: {temp_keys[:5]}")
            
            # Calculate temperature range
            min_val = float('inf')
            max_val = float('-inf')
            
            for i in range(self.time_steps):
                data_name = f'{prefix}_{i}'
                if data_name in self.mesh.array_names:
                    temp_data = self.mesh[data_name]
                    valid_temps = temp_data[temp_data > 0]
                    
                    if len(valid_temps) > 0:
                        curr_min = float(np.min(valid_temps))
                        curr_max = float(np.max(valid_temps))
                        min_val = min(min_val, curr_min)
                        max_val = max(max_val, curr_max)
                        if self.verbose:
                            print(f"Frame {i}: Temperature range {curr_min:.1f}K - {curr_max:.1f}K")
            
            # Verify we found valid data
            if min_val == float('inf') or max_val == float('-inf'):
                print("ERROR: No valid temperature data found. Cannot proceed with visualization.")
                return False
            
            # Round to nice values
            min_val = 5 * math.floor(min_val / 5)
            max_val = 5 * math.ceil(max_val / 5)
            
            clim = [min_val, max_val]
            print(f"Temperature range: {clim[0]:.1f}K - {clim[1]:.1f}K")
            
        else:
            print(f"Error: Unknown data type: {data_type}")
            return False
        
        # Extract positions if not already done
        if not self.sun_positions and not self.jupiter_positions:
            if hasattr(self, 'args') and self.args.positions:
                self.load_positions(self.args.positions)
            else:
                self.extract_body_positions_from_vtk()
        
        # Get mesh size for proper scaling
        mesh_bounds = self.mesh.bounds
        mesh_radius = max([mesh_bounds[1]-mesh_bounds[0], 
                         mesh_bounds[3]-mesh_bounds[2], 
                         mesh_bounds[5]-mesh_bounds[4]]) / 2
        
        # Create output directory for frames if needed
        frames_dir = None
        if save_frames:
            frames_dir = Path(output_file).parent / frames_dirname
            frames_dir.mkdir(exist_ok=True)
            print(f"Saving individual frames to {frames_dir}")
        
        # ================= OBJECT CACHING SYSTEM =================
        
        # Create cache key based on data type and resolution
        cache_key = f"{data_type}_plotter_{self.width}x{self.height}"
        
        # Initialize cache containers if they don't exist
        if not hasattr(self, '_plotter_cache'):
            self._plotter_cache = {}
            self._actor_cache = {}
        
        # Create or retrieve cached plotter
        if cache_key not in self._plotter_cache:
            if self.verbose:
                print("Creating new plotter (first render)")
            plotter = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
            plotter.set_background('black')
            
            # Add axes for orientation once
            plotter.add_axes(interactive=False, line_width=2)
            
            # Store in cache
            self._plotter_cache[cache_key] = plotter
            self._actor_cache[cache_key] = {}
        else:
            if self.verbose:
                print("Using cached plotter for better performance")
            plotter = self._plotter_cache[cache_key]
            
            # Clear existing actors from previous runs
            for actor_name in list(self._actor_cache[cache_key].keys()):
                if actor_name not in ['mesh']:  # Keep the base mesh
                    plotter.remove_actor(self._actor_cache[cache_key][actor_name])
                    del self._actor_cache[cache_key][actor_name]
        
        # Add or retrieve mesh template
        if 'mesh_template' not in self._actor_cache[cache_key]:
            # Create template mesh once
            mesh_template = self.mesh.copy()
            
            # Set up the mesh with initial data for first frame
            data_name = f'{prefix}_0'
            if data_name in mesh_template.array_names:
                mesh_template[array_name] = mesh_template[data_name]
                mesh_template.set_active_scalars(array_name)
                
                # Add mesh ONCE with proper configuration
                mesh_actor = plotter.add_mesh(
                    mesh_template,
                    scalars=array_name, 
                    cmap=cmap,
                    show_edges=False,
                    edge_color='white',
                    lighting=True,
                    clim=clim,
                    scalar_bar_args={'title': title}
                )
                
                # Cache both the mesh template and its actor
                self._actor_cache[cache_key]['mesh'] = mesh_actor
                self._actor_cache[cache_key]['mesh_template'] = mesh_template
            else:
                print(f"Error: {data_name} not found in mesh arrays")
                return False
        else:
            # Retrieve the cached mesh template
            mesh_template = self._actor_cache[cache_key]['mesh_template']
            
            # Update the color map and range
            plotter.update_scalar_bar_range(clim, self._actor_cache[cache_key]['mesh'])
        
        # ================= CAMERA SETUP WITH REDUCED OUTPUT =================
        
        # Setup camera view (only once per animation)
        if emission_angle is not None and azimuth is not None:
            if self.verbose:
                print(f"Using emission/azimuth camera mode: angle={emission_angle}°, azimuth={azimuth}°")
            self.setup_camera_emission_azimuth(plotter, emission_angle, azimuth, camera_distance)
        else:
            # IMPORTANT: Pass explicit values for all parameters
            self.setup_camera_for_mesh(
                plotter=plotter,
                camera_distance=camera_distance,
                camera_latitude=camera_latitude, 
                camera_longitude=camera_longitude
            )
        
        # Store frames
        frames = []
        
        # Get time strings
        time_strings = []
        if hasattr(self.mesh, 'field_data'):
            field_data_keys = list(self.mesh.field_data.keys())
            if self.verbose:
                print(f"Available field data: {field_data_keys}")
            
            # Try direct key access first
            if 'TIME_STRINGS' in field_data_keys:
                time_strings = self.mesh.field_data['TIME_STRINGS']
                if self.verbose:
                    print(f"Found {len(time_strings)} time strings with key: TIME_STRINGS")
            
            # Otherwise search for likely time string arrays
            elif len(field_data_keys) > 0:
                if self.verbose:
                    print("Looking for time strings in available field data")
                for key in field_data_keys:
                    data = self.mesh.field_data[key]
                    if isinstance(data, (list, np.ndarray)) and len(data) == self.time_steps:
                        if isinstance(data[0], (str, bytes)):
                            time_strings = [str(item) for item in data]
                            if self.verbose:
                                print(f"Using '{key}' as time strings")
                            break

        # Create fallback time strings if none found
        if not isinstance(time_strings, (list, np.ndarray)) or len(time_strings) == 0:
            if self.verbose:
                print("No time strings found, creating dummy time strings")
            time_strings = [f"Time {i}" for i in range(self.time_steps)]
        
        # Create progressbar
        from tqdm import tqdm
        for i in tqdm(range(self.time_steps), desc=f"Rendering {data_type} frames"):
            # Remove verbose camera position output for first frame
            
            # Clear temporary actors from previous frame
            for actor_name in list(self._actor_cache[cache_key].keys()):
                if actor_name not in ['mesh', 'mesh_template']:
                    plotter.remove_actor(self._actor_cache[cache_key][actor_name])
                    del self._actor_cache[cache_key][actor_name]
            
            # Get current data name
            data_name = f'{prefix}_{i}'
            
            if data_name in self.mesh.array_names:
                # Update the mesh data instead of creating a new one
                mesh_template[array_name] = self.mesh[data_name]
                
                # Update the scalars on the existing mesh
                plotter.update_scalars(mesh_template[array_name], mesh=mesh_template)
                
                # Add timestamp
                if i < len(time_strings) and time_strings[i]:
                    # Handle byte strings
                    timestamp_str = str(time_strings[i])
                    if isinstance(time_strings[i], bytes):
                        timestamp_str = time_strings[i].decode('utf-8')
                    timestamp_text = f"UTC: {timestamp_str}"
                else:
                    timestamp_text = f"Time Step: {i}"
                    
                timestamp_actor = plotter.add_text(timestamp_text, position='upper_left', 
                                                  font_size=12, color='white')
                self._actor_cache[cache_key]['timestamp'] = timestamp_actor
                
                # Update Jupiter and Sun positions
                if i < len(self.sun_positions) and i < len(self.jupiter_positions):
                    # Add Jupiter indicators
                    if show_jupiter and self.jupiter_positions[i] is not None:
                        jup_pos = np.array(self.jupiter_positions[i])
                        jup_dir = jup_pos / np.linalg.norm(jup_pos)
                        jup_end = jup_dir * mesh_radius * 1.2
                        
                        # Add line for Jupiter
                        jupiter_line = plotter.add_mesh(
                            pv.Line(np.array([0, 0, 0]), jup_end), 
                            color='orange', 
                            line_width=2
                        )
                        self._actor_cache[cache_key]['jupiter_line'] = jupiter_line
                        
                        # Add text label for Jupiter
                        jupiter_text = plotter.add_point_labels(
                            [jup_end], 
                            ["Jupiter"], 
                            text_color='orange', 
                            font_size=12,
                            bold=True,
                            shadow=True,
                            name="jupiter_text"
                        )
                        self._actor_cache[cache_key]['jupiter_text'] = jupiter_text
                    
                    # Add Sun indicators if needed
                    if show_sun and self.sun_positions[i] is not None:
                        sun_pos = np.array(self.sun_positions[i])
                        sun_dir = sun_pos / np.linalg.norm(sun_pos)
                        sun_end = sun_dir * mesh_radius * 1.2
                        
                        sun_line = plotter.add_mesh(
                            pv.Line(np.array([0, 0, 0]), sun_end), 
                            color='yellow', 
                            line_width=2
                        )
                        self._actor_cache[cache_key]['sun_line'] = sun_line
                        
                        sun_text = plotter.add_point_labels(
                            [sun_end], 
                            ["Sun"], 
                            text_color='yellow', 
                            font_size=12, 
                            bold=True,
                            shadow=True,
                            name="sun_text"
                        )
                        self._actor_cache[cache_key]['sun_text'] = sun_text
                
                # Save individual frame if requested
                if save_frames and frames_dir:
                    frame_path = frames_dir / f"{prefix}_frame_{i:04d}.png"
                    plotter.screenshot(str(frame_path))
                
                # Get frame for the animation
                frame = plotter.screenshot(return_img=True)
                frames.append(frame)
            else:
                print(f"Warning: No {data_type} data for time step {i}")
        
        # Note: We don't close the plotter here as we're reusing it
        # Memory cleanup will be handled by clear_visualization_cache() or when needed
        
        # Save animation
        if frames:
            print(f"Writing {data_type} animation to {output_file}...")
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                imageio.mimsave(output_file, frames, fps=fps)
                print(f"{data_type.capitalize()} animation saved successfully")
                
                # Release frames from memory now that we're done with them
                frames = None
                import gc
                gc.collect()
                
                return True
            except Exception as e:
                print(f"Error saving animation: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return False

    def create_animation_generic_parallel(self, output_file, data_type='illumination', 
                  camera_distance=5.0, camera_latitude=30.0, 
                  camera_longitude=0.0, emission_angle=None, azimuth=None, fps=10, save_frames=False):
        """Parallelized version of animation creation"""
        print(f"Creating {data_type} animation {output_file} using parallel processing...")
        
        # Setup prefix, array name, and color map based on data type
        if data_type == 'illumination':
            prefix = 'illumination'
            array_name = 'illumination'
            cmap = 'inferno'
            title = 'Illumination (W/m²)'
            show_sun = True
            show_jupiter = True
            frames_dirname = "illumination_frames"
            
            # Calculate dynamic range for solar illumination
            max_val = 0
            for i in range(self.time_steps):
                data_name = f'{prefix}_{i}'
                if data_name in self.mesh.array_names:
                    max_val = max(max_val, np.max(self.mesh[data_name]))
            
            # Round up to nearest 5 W/m²
            import math
            upper_bound = 5 * math.ceil(max_val / 5)
            if upper_bound < 5:  # Ensure minimum scale
                upper_bound = 5
            
            clim = [0, upper_bound]
            print(f"Dynamic solar illumination range: {clim}")
            
        elif data_type == 'jupiter_ir':
            prefix = 'jupiter_ir'
            array_name = 'jupiter_ir'
            cmap = 'inferno'
            title = 'Jupiter IR Flux (W/m²)'
            show_jupiter = True
            show_sun = False
            frames_dirname = "jupiter_ir_frames"
            
            # Calculate dynamic range for Jupiter IR
            max_val = 0
            for i in range(self.time_steps):
                data_name = f'{prefix}_{i}'
                if data_name in self.mesh.array_names:
                    max_val = max(max_val, np.max(self.mesh[data_name]))
            
            # Round up to nearest 0.5 W/m²
            import math
            upper_bound = 0.5 * math.ceil(max_val / 0.5)
            if upper_bound < 0.5:
                upper_bound = 0.5
            
            clim = [0, upper_bound]
            print(f"Dynamic Jupiter IR range: {clim}")
            
        elif data_type == 'thermal':
            prefix = 'surface_temp'
            array_name = 'temperature'
            cmap = 'turbo'
            title = 'Surface Temperature (K)'
            show_jupiter = False
            show_sun = False
            frames_dirname = "thermal_frames"
            
            # Calculate dynamic range for temperature
            min_val = float('inf')
            max_val = 0
            for i in range(self.time_steps):
                data_name = f'{prefix}_{i}'
                if data_name in self.mesh.array_names:
                    vals = self.mesh[data_name]
                    if len(vals) > 0:
                        min_val = min(min_val, np.min(vals))
                        max_val = max(max_val, np.max(vals))
            
            # Round to nearest 5K for nice values
            import math
            lower_bound = 5 * math.floor(min_val / 5)
            upper_bound = 5 * math.ceil(max_val / 5)
            
            clim = [lower_bound, upper_bound]
            print(f"Dynamic temperature range: {clim}")
        
        else:
            print(f"Unknown data type: {data_type}")
            return False
        
        # Get time strings for timestamps
        time_strings = []
        if hasattr(self, 'time_strings'):
            time_strings = self.time_strings
        
        # Calculate mesh radius for proper indicators
        mesh_radius = 0
        if hasattr(self.mesh, 'points'):
            points = np.array(self.mesh.points)
            mesh_radius = np.max(np.linalg.norm(points, axis=1))
        
        # Create frames_dir if needed
        frames_dir = None
        if save_frames:
            frames_dir = os.path.join(os.path.dirname(output_file), frames_dirname)
            os.makedirs(frames_dir, exist_ok=True)
        
        # ================= OBJECT CACHING SYSTEM =================
        
        # Create cache key based on data type and resolution
        cache_key = f"{data_type}_plotter_{self.width}x{self.height}"
        
        # Initialize cache containers if they don't exist
        if not hasattr(self, '_plotter_cache'):
            self._plotter_cache = {}
            self._actor_cache = {}
        
        # Create or retrieve cached plotter
        if cache_key not in self._plotter_cache:
            if self.verbose:
                print("Creating new plotter (first render)")
            plotter = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
            plotter.set_background('black')
            
            # Add axes for orientation once
            plotter.add_axes(interactive=False, line_width=2)
            
            # Store in cache
            self._plotter_cache[cache_key] = plotter
            self._actor_cache[cache_key] = {}
        else:
            if self.verbose:
                print("Using cached plotter for better performance")
            plotter = self._plotter_cache[cache_key]
            
            # Clear existing actors from previous runs
            for actor_name in list(self._actor_cache[cache_key].keys()):
                if actor_name not in ['mesh']:  # Keep the base mesh
                    plotter.remove_actor(self._actor_cache[cache_key][actor_name])
                    del self._actor_cache[cache_key][actor_name]
        
        # Add or retrieve mesh template
        if 'mesh_template' not in self._actor_cache[cache_key]:
            # Create template mesh once
            mesh_template = self.mesh.copy()
            
            # Set up the mesh with initial data for first frame
            data_name = f'{prefix}_0'
            if data_name in mesh_template.array_names:
                mesh_template[array_name] = mesh_template[data_name]
                mesh_template.set_active_scalars(array_name)
                
                # Add mesh ONCE with proper configuration
                mesh_actor = plotter.add_mesh(
                    mesh_template,
                    scalars=array_name, 
                    cmap=cmap,
                    show_edges=False,
                    edge_color='white',
                    lighting=True,
                    clim=clim,
                    scalar_bar_args={'title': title}
                )
                
                # Cache both the mesh template and its actor
                self._actor_cache[cache_key]['mesh'] = mesh_actor
                self._actor_cache[cache_key]['mesh_template'] = mesh_template
            else:
                print(f"Error: {data_name} not found in mesh arrays")
                return False
        else:
            # Retrieve the cached mesh template
            mesh_template = self._actor_cache[cache_key]['mesh_template']
        
        # ================= CAMERA SETUP WITH REDUCED OUTPUT =================
        
        # Setup camera view (only once per animation)
        if emission_angle is not None and azimuth is not None:
            if self.verbose:
                print(f"Using emission/azimuth camera mode: angle={emission_angle}°, azimuth={azimuth}°")
            self.setup_camera_emission_azimuth(plotter, emission_angle, azimuth, camera_distance)
        else:
            # IMPORTANT: Pass explicit values for all parameters
            self.setup_camera_for_mesh(
                plotter=plotter,
                camera_distance=camera_distance,
                camera_latitude=camera_latitude, 
                camera_longitude=camera_longitude
            )
        
        # Store frames
        frames = []
        
        # Get time strings
        time_strings = []
        if hasattr(self.mesh, 'field_data'):
            field_data_keys = list(self.mesh.field_data.keys())
            if self.verbose:
                print(f"Available field data: {field_data_keys}")
            
            # Try direct key access first
            if 'TIME_STRINGS' in field_data_keys:
                time_strings = self.mesh.field_data['TIME_STRINGS']
                if self.verbose:
                    print(f"Found {len(time_strings)} time strings with key: TIME_STRINGS")
            
            # Otherwise search for likely time string arrays
            elif len(field_data_keys) > 0:
                if self.verbose:
                    print("Looking for time strings in available field data")
                for key in field_data_keys:
                    data = self.mesh.field_data[key]
                    if isinstance(data, (list, np.ndarray)) and len(data) == self.time_steps:
                        if isinstance(data[0], (str, bytes)):
                            time_strings = [str(item) for item in data]
                            if self.verbose:
                                print(f"Using '{key}' as time strings")
                            break

        # Create fallback time strings if none found
        if not isinstance(time_strings, (list, np.ndarray)) or len(time_strings) == 0:
            if self.verbose:
                print("No time strings found, creating dummy time strings")
            time_strings = [f"Time {i}" for i in range(self.time_steps)]
        
        # Create progressbar
        from tqdm import tqdm
        for i in tqdm(range(self.time_steps), desc=f"Rendering {data_type} frames"):
            # Remove verbose camera position output for first frame
            
            # Clear temporary actors from previous frame
            for actor_name in list(self._actor_cache[cache_key].keys()):
                if actor_name not in ['mesh', 'mesh_template']:
                    plotter.remove_actor(self._actor_cache[cache_key][actor_name])
                    del self._actor_cache[cache_key][actor_name]
            
            # Get current data name
            data_name = f'{prefix}_{i}'
            
            if data_name in self.mesh.array_names:
                # Update the mesh data instead of creating a new one
                mesh_template[array_name] = self.mesh[data_name]
                
                # Update the scalars on the existing mesh
                plotter.update_scalars(mesh_template[array_name], mesh=mesh_template)
                
                # Add timestamp
                if i < len(time_strings) and time_strings[i]:
                    # Handle byte strings
                    timestamp_str = str(time_strings[i])
                    if isinstance(time_strings[i], bytes):
                        timestamp_str = time_strings[i].decode('utf-8')
                    timestamp_text = f"UTC: {timestamp_str}"
                else:
                    timestamp_text = f"Time Step: {i}"
                    
                timestamp_actor = plotter.add_text(timestamp_text, position='upper_left', 
                                                  font_size=12, color='white')
                self._actor_cache[cache_key]['timestamp'] = timestamp_actor
                
                # Update Jupiter and Sun positions
                if i < len(self.sun_positions) and i < len(self.jupiter_positions):
                    # Add Jupiter indicators
                    if show_jupiter and self.jupiter_positions[i] is not None:
                        jup_pos = np.array(self.jupiter_positions[i])
                        jup_dir = jup_pos / np.linalg.norm(jup_pos)
                        jup_end = jup_dir * mesh_radius * 1.2
                        
                        # Add line for Jupiter
                        jupiter_line = plotter.add_mesh(
                            pv.Line(np.array([0, 0, 0]), jup_end), 
                            color='orange', 
                            line_width=2
                        )
                        self._actor_cache[cache_key]['jupiter_line'] = jupiter_line
                        
                        # Add text label for Jupiter
                        jupiter_text = plotter.add_point_labels(
                            [jup_end], 
                            ["Jupiter"], 
                            text_color='orange', 
                            font_size=12,
                            bold=True,
                            shadow=True,
                            name="jupiter_text"
                        )
                        self._actor_cache[cache_key]['jupiter_text'] = jupiter_text
                    
                    # Add Sun indicators if needed
                    if show_sun and self.sun_positions[i] is not None:
                        sun_pos = np.array(self.sun_positions[i])
                        sun_dir = sun_pos / np.linalg.norm(sun_pos)
                        sun_end = sun_dir * mesh_radius * 1.2
                        
                        sun_line = plotter.add_mesh(
                            pv.Line(np.array([0, 0, 0]), sun_end), 
                            color='yellow', 
                            line_width=2
                        )
                        self._actor_cache[cache_key]['sun_line'] = sun_line
                        
                        sun_text = plotter.add_point_labels(
                            [sun_end], 
                            ["Sun"], 
                            text_color='yellow', 
                            font_size=12, 
                            bold=True,
                            shadow=True,
                            name="sun_text"
                        )
                        self._actor_cache[cache_key]['sun_text'] = sun_text
                
                # Save individual frame if requested
                if save_frames and frames_dir:
                    frame_path = frames_dir / f"{prefix}_frame_{i:04d}.png"
                    plotter.screenshot(str(frame_path))
                
                # Get frame for the animation
                frame = plotter.screenshot(return_img=True)
                frames.append(frame)
            else:
                print(f"Warning: No {data_type} data for time step {i}")
        
        # Note: We don't close the plotter here as we're reusing it
        # Memory cleanup will be handled by clear_visualization_cache() or when needed
        
        # Save animation
        if frames:
            print(f"Writing {data_type} animation to {output_file}...")
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                imageio.mimsave(output_file, frames, fps=fps)
                print(f"{data_type.capitalize()} animation saved successfully")
                return True
            except Exception as e:
                print(f"Error saving animation: {e}")
                return False
        
        return False

    def create_location_time_series(self, csv_file, output_dir):
        """Create time series plots for specific locations"""
        print(f"Creating time series plots for locations in {csv_file}...")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Read the CSV file
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                try:
                    # Try alternative parser settings
                    df = pd.read_csv(csv_file, sep=',', engine='python', skiprows=0)
                    print(f"Successfully read CSV with alternative parser")
                except Exception as e2:
                    print(f"Failed to read CSV file: {e2}")
                    return False
            
            if df.empty:
                print("Error: CSV file is empty or not properly formatted")
                return False
            
            # Debug: Show available columns
            print(f"Available columns in CSV file: {list(df.columns)}")
            
            # Get time column (first column)
            time_col = df.columns[0]
            times = df[time_col].values
            
            # Define exact column suffix patterns
            column_suffixes = ['_Total', '_DirectSolar', '_JupiterReflected', '_JupiterIR', '_Temperature']
            
            # Extract location names by matching exact suffixes
            locations = set()
            for col in df.columns[1:]:  # Skip time column
                for suffix in column_suffixes:
                    if col.endswith(suffix):
                        # Extract location name by removing the suffix
                        loc_name = col[:-len(suffix)]
                        locations.add(loc_name)
                        break
            
            print(f"Found {len(locations)} locations in CSV: {', '.join(locations)}")
            if not locations:
                print("No location data found in CSV file")
                return False
            
            # Import MaxNLocator here for consistent availability
            from matplotlib.ticker import MaxNLocator
            
            # Create plots for each location
            for location in sorted(locations):
                print(f"Creating plots for location: {location}")
                
                # Try to find flux component columns
                solar_col = f"{location}_Total"
                direct_col = f"{location}_DirectSolar"
                reflected_col = f"{location}_JupiterReflected"
                ir_col = f"{location}_JupiterIR"
                temp_col = f"{location}_Temperature"
                
                # Skip if we don't have at least solar data
                if solar_col not in df.columns:
                    print(f"No solar flux data for location {location}, skipping")
                    continue
                
                # Create figure for flux components
                plt.figure(figsize=(10, 6))
                
                # Plot solar flux components
                plt.plot(times, df[solar_col], 'r-', linewidth=2, label='Total Solar')
                
                if direct_col in df.columns:
                    plt.plot(times, df[direct_col], 'b--', linewidth=1.5, label='Direct Solar')
                
                if reflected_col in df.columns:
                    plt.plot(times, df[reflected_col], 'g:', linewidth=1.5, label='Jupiter Reflected')
                
                if ir_col in df.columns:
                    plt.plot(times, df[ir_col], 'c-.', linewidth=1.5, label='Jupiter IR')
                
                plt.title(f"Solar Flux Components at {location}")
                plt.xlabel("Time (UTC)")
                plt.ylabel("Solar Flux (W/m²)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Format time axis better
                plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
                plt.gcf().autofmt_xdate()
                
                # Save solar flux figure
                solar_file = os.path.join(output_dir, f"{location}_solar_flux.png")
                plt.tight_layout()
                plt.savefig(solar_file, dpi=150)
                plt.close()
                print(f"Saved solar flux plot to {solar_file}")
                
                # Create temperature plot if available
                if temp_col in df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(times, df[temp_col], 'r-', linewidth=2)
                    
                    # Add statistics
                    avg_temp = np.mean(df[temp_col])
                    min_temp = np.min(df[temp_col])
                    max_temp = np.max(df[temp_col])
                    
                    plt.text(0.02, 0.95, 
                             f"Mean: {avg_temp:.1f}K\nMin: {min_temp:.1f}K\nMax: {max_temp:.1f}K",
                             transform=plt.gca().transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    
                    plt.title(f"Surface Temperature at {location}")
                    plt.xlabel("Time (UTC)")
                    plt.ylabel("Temperature (K)")
                    plt.grid(True, alpha=0.3)
                    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
                    plt.gcf().autofmt_xdate()
                    
                    # Save temperature figure
                    temp_file = os.path.join(output_dir, f"{location}_temperature.png")
                    plt.tight_layout()
                    plt.savefig(temp_file, dpi=150)
                    plt.close()
                    print(f"Saved temperature plot to {temp_file}")
            
            # Create summary temperature plot
            plt.figure(figsize=(12, 7))
            for location in sorted(locations):
                temp_col = f"{location}_Temperature"
                if temp_col in df.columns:
                    plt.plot(times, df[temp_col], label=location)
            
            plt.title("Surface Temperature at All Locations")
            plt.xlabel("Time (UTC)")
            plt.ylabel("Temperature (K)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.gcf().autofmt_xdate()
            
            # Save summary temperature figure
            summary_temp_file = os.path.join(output_dir, "all_locations_temperatures.png")
            plt.tight_layout()
            plt.savefig(summary_temp_file, dpi=150)
            plt.close()
            print(f"Saved summary temperature plot to {summary_temp_file}")
            
            # Create summary solar flux plot
            plt.figure(figsize=(12, 7))
            for location in sorted(locations):
                solar_col = f"{location}_Total"
                if solar_col in df.columns:
                    plt.plot(times, df[solar_col], label=location)
            
            plt.title("Total Solar Flux at All Locations")
            plt.xlabel("Time (UTC)")
            plt.ylabel("Total Solar Flux (W/m²)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.gcf().autofmt_xdate()
            
            # Save summary solar flux figure
            summary_flux_file = os.path.join(output_dir, "all_locations_solar_flux.png")
            plt.tight_layout()
            plt.savefig(summary_flux_file, dpi=150)
            plt.close()
            print(f"Saved summary solar flux plot to {summary_flux_file}")
            
            return True
            
        except Exception as e:
            print(f"Error creating location time series plots: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_summary_plot(self, df, locations, times, output_dir):
        """Create a summary plot showing all locations together"""
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot each location's solar flux
            for location in sorted(locations):
                # Look for solar flux column with various naming patterns
                for pattern in ['_Solar', '_Total', '_DirectSolar']:
                    col_name = f"{location}{pattern}"
                    if col_name in df.columns:
                        ax.plot(times, df[col_name], '-', linewidth=1.5, label=location)
                        break
            
            # Format the plot
            ax.set_xlabel('Time (UTC)')
            ax.set_ylabel('Total Solar Flux (W/m²)')
            ax.set_title('Total Solar Flux at All Locations')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            
            # Reduce x-tick label count for readability
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            fig.autofmt_xdate()
            
            # Save the summary figure
            output_file = os.path.join(output_dir, "all_locations_summary.png")
            fig.tight_layout()
            plt.savefig(output_file, dpi=150)
            plt.close(fig)
            print(f"Saved summary plot to {output_file}")
            
            return True
        except Exception as e:
            print(f"Error creating summary plot: {e}")
            return False

    def get_nice_scale_ceiling(self, max_value):
        """Create a nice upper limit for a scale based on the most significant digit."""
        if max_value <= 0:
            return 0.001  # Default small value if there's no positive data
            
        # Find the order of magnitude
        order = math.floor(math.log10(max_value))
        
        # Calculate the scale unit (power of 10)
        scale_unit = 10 ** order
        
        # Get the leading digit(s) by normalizing to [1-10)
        leading = max_value / scale_unit
        
        # Round up to the next integer
        ceiling = math.ceil(leading)
        
        # Return the ceiling multiplied by the scale unit
        return ceiling * scale_unit

    def setup_camera_for_mesh(self, plotter, camera_distance=None, camera_latitude=None, camera_longitude=None):
        """Set up camera position optimized for the current mesh geometry"""
        
        # If no mesh is loaded, use default view
        if self.mesh is None:
            plotter.view_isometric()
            return
            
        # Get mesh bounds and calculate size
        bounds = self.mesh.bounds
        mesh_center = np.array([
            (bounds[0] + bounds[1])/2,
            (bounds[2] + bounds[3])/2,
            (bounds[4] + bounds[5])/2
        ])
        
        mesh_size = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ])
        
        # Determine if this is a partial mesh (DEM) or full sphere
        center_distance = np.linalg.norm(mesh_center)
        max_dimension = np.max(mesh_size)
        is_partial_mesh = center_distance > 0.1  # If center is significantly off origin
        
        # If verbose logging is enabled, provide mesh details
        if self.verbose:
            print(f"Mesh type: {'partial (DEM) mesh' if is_partial_mesh else 'full sphere mesh'}")
        
        # If custom camera parameters are provided, use them
        if camera_latitude is not None and camera_longitude is not None:
            if is_partial_mesh:
                # For partial mesh: aim at mesh center from specified direction
                lat_rad = np.radians(camera_latitude)
                lon_rad = np.radians(camera_longitude)
                
                # Calculate direction vector from lat/lon
                direction = np.array([
                    np.cos(lat_rad) * np.cos(lon_rad),
                    np.cos(lat_rad) * np.sin(lon_rad),
                    np.sin(lat_rad)
                ])
                
                # Position camera based on mesh size
                cam_distance = camera_distance if camera_distance is not None else max_dimension * 2
                look_from = mesh_center + direction * cam_distance
                
                # Set up camera to look at mesh center
                plotter.camera.position = look_from
                plotter.camera.focal_point = mesh_center
                plotter.camera.up = [0, 0, 1]  # Default up direction
                
            else:
                # For full sphere: use standard lat/lon positioning with improved up vector
                europa_radius = 1.0  # Normalized radius
                cam_distance = camera_distance if camera_distance is not None else 5.0
                
                # Convert lat/lon to Cartesian coordinates for camera position
                lat_rad = np.radians(camera_latitude)
                lon_rad = np.radians(camera_longitude)
                
                cam_x = cam_distance * np.cos(lat_rad) * np.cos(lon_rad)
                cam_y = cam_distance * np.cos(lat_rad) * np.sin(lon_rad)
                cam_z = cam_distance * np.sin(lat_rad)
                
                # Calculate view direction (towards origin)
                view_dir = -np.array([cam_x, cam_y, cam_z])
                view_dir = view_dir / np.linalg.norm(view_dir)
                
                # Calculate up vector that's perpendicular to view direction
                # For poles, use special handling
                if abs(camera_latitude) > 85.0:
                    # Near poles, use a fixed up vector pointing along meridian
                    meridian_dir = np.array([
                        -np.sin(lon_rad),
                        np.cos(lon_rad),
                        0.0
                    ])
                    up_vec = meridian_dir if camera_latitude > 0 else -meridian_dir
                else:
                    # Otherwise use a proper perpendicular to viewing direction
                    # that approximates "up" relative to planet
                    world_up = np.array([0, 0, 1])  # Z is world up
                    
                    # Make sure up vector is perpendicular to view direction
                    # by using cross product twice
                    right = np.cross(view_dir, world_up)
                    # Handle case when view_dir is parallel to world_up
                    if np.linalg.norm(right) < 1e-6:
                        right = np.array([1, 0, 0])  # Use X axis as fallback
                    else:
                        right = right / np.linalg.norm(right)
                        
                    up_vec = np.cross(right, view_dir)
                    up_vec = up_vec / np.linalg.norm(up_vec)
                
                # Set camera position
                camera_pos = [cam_x, cam_y, cam_z]
                focal_point = [0, 0, 0]  # Origin for full sphere
                
                # Set the camera
                plotter.camera_position = [camera_pos, focal_point, up_vec.tolist()]
                
        else:
            # Automatic camera positioning
            if is_partial_mesh:
                # For partial mesh: position camera for best view of mesh
                # Find the "front" direction (away from the origin through the mesh center)
                if center_distance > 0.01:
                    front_dir = mesh_center / center_distance
                else:
                    # If center is very close to origin, use a default direction
                    front_dir = np.array([1, 0, 0])
                
                # Position camera behind the mesh looking toward origin
                view_distance = max_dimension * 2
                camera_pos = mesh_center + front_dir * view_distance
                
                # Set camera
                plotter.camera.position = camera_pos
                plotter.camera.focal_point = mesh_center
                plotter.camera.up = [0, 0, 1]
                
                if self.verbose:
                    print("Using automatic camera position for partial mesh")
            else:
                # For full sphere: use standard isometric view
                plotter.view_isometric()
                if self.verbose:
                    print("Using isometric view for full sphere")
                    
        # Reset clipping range to ensure the entire mesh is visible
        plotter.reset_camera()

    def setup_camera_emission_azimuth(self, plotter, emission_angle, azimuth, camera_distance=None):
        """Set up camera position using emission and azimuth angles"""
        print(f"Setting camera position: emission={emission_angle}°, azimuth={azimuth}°")
        
        # Get mesh center - important for custom meshes that may not be centered at origin
        center = self.mesh.center
        
        # Get mesh bounds and calculate diagonal length for scaling
        bounds = self.mesh.bounds
        diagonal = np.linalg.norm([
            bounds[1] - bounds[0],  # x extent
            bounds[3] - bounds[2],  # y extent
            bounds[5] - bounds[4]   # z extent
        ])
        
        # Calculate appropriate camera distance if not provided
        if camera_distance is None or camera_distance <= 0:
            # Use a multiple of the mesh diagonal for proper framing
            camera_distance = diagonal * 2.0
        else:
            # If provided as Europa radii, convert to mesh units
            if center[0] > 1000 or center[1] > 1000 or center[2] > 1000:
                # For large coordinate values (likely km), maintain scale
                pass
            else:
                # For normalized coordinates, scale camera_distance by diagonal
                camera_distance *= diagonal
        
        print(f"Using camera distance: {camera_distance:.2f} units (mesh diagonal: {diagonal:.2f})")
        
        # Convert angles from degrees to radians
        emission_rad = np.radians(emission_angle)
        azimuth_rad = np.radians(azimuth)
        
        # Calculate camera position in spherical coordinates relative to mesh center
        # emission_angle: 0=directly above (nadir), 90=at horizon
        # azimuth: 0=North, 90=East, 180=South, 270=West
        x = center[0] + camera_distance * np.sin(emission_rad) * np.sin(azimuth_rad)
        y = center[1] + camera_distance * np.sin(emission_rad) * np.cos(azimuth_rad)
        z = center[2] + camera_distance * np.cos(emission_rad)
        
        # Set camera position and focal point
        camera_position = [x, y, z]
        focal_point = center
        
        # Adjust up vector based on viewing angle to prevent camera flipping
        # For near-polar views, adjust up vector
        if emission_angle < 10:
            # Near north pole, use Y axis as up
            up = [0, 1, 0]
        elif emission_angle > 170:
            # Near south pole, use Y axis as up
            up = [0, 1, 0]
        else:
            # Default up vector
            up = [0, 0, 1]
        
        # Set the camera
        plotter.camera_position = [camera_position, focal_point, up]
        
        # Print debug information about camera setup
        print(f"Camera position: {camera_position}")
        print(f"Focal point: {focal_point}")
        print(f"Up vector: {up}")
        
        # Make sure the mesh is visible by zooming to it
        plotter.reset_camera()
        
        return True

def generate_location_temperature_plots(location_data_file, output_dir):
    """Generate temperature plots for specific locations."""
    from matplotlib.ticker import MaxNLocator
    
    if not os.path.exists(location_data_file):
        print(f"Location data file not found: {location_data_file}")
        return False
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Reading location data from: {location_data_file}")
        
        # Parse the space-separated location file manually
        locations = []
        location_names = []
        
        with open(location_data_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse the line (space-separated: lat lon name)
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        name = '_'.join(parts[2:])  # Join remaining parts as name with underscores
                        locations.append((lat, lon))
                        location_names.append(name)
                        print(f"Parsed location: {name} at ({lat}°, {lon}°)")
                    except ValueError as e:
                        print(f"Error parsing line {line_num}: {line} - {e}")
                        continue
                else:
                    print(f"Skipping malformed line {line_num}: {line}")
                    continue
        
        if not locations:
            print("No valid locations found in file")
            return False
        
        print(f"Found {len(locations)} locations: {location_names}")
        
        # Look for the thermal simulation's location temperature CSV
        # The C++ code should generate this file
        possible_csv_files = [
            "./output/location_temperatures.csv",
            "location_temperatures.csv",
            os.path.join(os.path.dirname(location_data_file), "location_temperatures.csv")
        ]
        
        temperature_csv = None
        for csv_file in possible_csv_files:
            if os.path.exists(csv_file):
                temperature_csv = csv_file
                print(f"Found temperature CSV: {csv_file}")
                break
        
        if not temperature_csv:
            print("No temperature CSV file found.")
            print("Expected files:")
            for f in possible_csv_files:
                print(f"  - {f}")
            print("Make sure the thermal simulation has been run with --location-data parameter.")
            return False
        
        print(f"Using temperature data file: {temperature_csv}")
        
        # Read the temperature CSV file
        try:
            import pandas as pd
            df = pd.read_csv(temperature_csv)
            print(f"CSV columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error reading temperature CSV: {e}")
            return False
        
        # Look for temperature columns (should start with 'temp_')
        temp_cols = [col for col in df.columns if col.startswith('temp_')]
        if not temp_cols:
            print(f"No temperature columns found in CSV file. Available columns: {list(df.columns)}")
            return False
        
        print(f"Found {len(temp_cols)} temperature time steps")
        
        # Extract time information from column names
        times = []
        for col in temp_cols:
            # Extract time index from column name (temp_0, temp_1, etc.)
            try:
                time_idx = int(col.split('_')[1])
                times.append(time_idx)
            except (IndexError, ValueError):
                times.append(0)
        
        # Create plots for each location
        import matplotlib.pyplot as plt
        for idx, row in df.iterrows():
            location_name = str(row['location_name']).replace(' ', '_')
            lat = row['lat']
            lon = row['lon']
            
            # Extract temperature data for this location
            temps = [row[col] for col in temp_cols if col in row.index and not pd.isna(row[col])]
            
            if not temps:
                print(f"No valid temperature data found for {location_name}")
                continue
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(times[:len(temps)], temps, 'b-', linewidth=2, label=f'{location_name}')
            plt.xlabel('Time Step')
            plt.ylabel('Temperature (K)')
            plt.title(f'Surface Temperature at {location_name} ({lat:.1f}°, {lon:.1f}°)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the plot
            safe_name = location_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            output_file = os.path.join(output_dir, f'{safe_name}_temperature.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Created temperature plot: {output_file}")
        
        print(f"Successfully created {len(df)} location temperature plots")
        return True
        
    except Exception as e:
        print(f"Error reading location data: {e}")
        import traceback
        traceback.print_exc()
        return False
def read_vtk_temperature_direct(vtk_file, mesh, time_steps):
    """
    Manually read temperature data from VTK file, bypassing PyVista's parser.
    This should provide a more reliable way to access temperature data.
    """
    import re
    print(f"Directly reading temperature data from {vtk_file}")
    
    # Initialize temp data structure to hold values for all time steps
    temp_data = {}
    for i in range(time_steps):
        temp_data[f'temperature_{i}'] = []
    
    # Read the VTK file directly as text
    with open(vtk_file, 'r') as f:
        lines = f.readlines()
    
    # Process the file line by line
    current_array = None
    reading_data = False
    cell_count = -1
    
    for line in lines:
        line = line.strip()
        
        # Detect the number of cells (important for knowing how many values to read)
        if line.startswith("CELL_DATA"):
            cell_count = int(line.split()[1])
            
        # Detect start of a temperature data array
        elif line.startswith("SCALARS temperature_"):
            match = re.match(r"SCALARS (temperature_\d+)", line)
            if match:
                current_array = match.group(1)
                reading_data = False
                print(f"Found temperature array: {current_array}")
                
        # Start reading values after the LOOKUP_TABLE line
        elif line.startswith("LOOKUP_TABLE") and current_array:
            reading_data = True
            continue
        
        # If we're reading a temperature array, collect the values
        elif reading_data and current_array and current_array in temp_data:
            try:
                value = float(line)
                temp_data[current_array].append(value)
                
                # If we've read all values for this cell data, stop reading this array
                if len(temp_data[current_array]) >= cell_count:
                    reading_data = False
                    current_array = None
            except ValueError:
                # If we can't convert to float, we've likely reached the end of values
                reading_data = False
                current_array = None
    
    # Verify we got expected amounts of data
    valid_arrays = []
    for array_name, values in temp_data.items():
        if len(values) == cell_count:
            valid_arrays.append(array_name)
            # Create proper numpy array for PyVista
            import numpy as np
            temp_array = np.array(values)
            
            # Manually add this array to the mesh data
            mesh.cell_data[array_name] = temp_array
            
            # Basic statistics for debugging
            valid_values = temp_array[temp_array > 0]
            if len(valid_values) > 0:
                print(f"  {array_name}: {len(valid_values)} valid values, " 
                      f"range: {valid_values.min():.2f}K - {valid_values.max():.2f}K")
    
    print(f"Successfully loaded {len(valid_arrays)} temperature arrays directly from the VTK file")
    return len(valid_arrays) > 0

def generate_descriptive_filename(mesh, data_type, camera_lat, camera_lon, camera_distance, base_dir="./output"):
    """Generate a descriptive filename based on mesh and camera parameters"""
    # Extract mesh resolution or name
    mesh_descriptor = ""
    if hasattr(mesh, "n_cells"):
        if mesh.n_cells in [20480, 81920, 327680, 1310720]:
            # These are the standard resolution meshes (res 5-8)
            res_map = {20480: 5, 81920: 6, 327680: 7, 1310720: 8}
            mesh_descriptor = f"res{res_map[mesh.n_cells]}"
        else:
            # For non-standard meshes, use face count
            mesh_descriptor = f"{mesh.n_cells}faces"
    
    # Get VTK file basename if available
    vtk_descriptor = ""
    if hasattr(mesh, "path"):
        vtk_name = os.path.basename(mesh.path)
        # Remove extension and use as descriptor if it's a custom mesh
        if "custom" in vtk_name.lower() or "dem" in vtk_name.lower():
            vtk_descriptor = os.path.splitext(vtk_name)[0]
    
    # Combine mesh descriptors
    if vtk_descriptor and mesh_descriptor:
        descriptor = f"{vtk_descriptor}_{mesh_descriptor}"
    else:
        descriptor = vtk_descriptor or mesh_descriptor
    
    # Format camera positions for filename
    camera_str = f"lat{int(camera_lat)}_lon{int(camera_lon)}"
    
    # Add timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create filename
    if data_type == "illumination":
        prefix = "europa_solar_flux"
    elif data_type == "jupiter_ir":
        prefix = "europa_jupiter_ir"
    elif data_type == "temperature":
        prefix = "europa_surface_temp"
    else:
        prefix = f"europa_{data_type}"
    
    filename = f"{prefix}_{descriptor}_{camera_str}_{timestamp}.mp4"
    
    # Ensure output directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    return os.path.join(base_dir, filename)

def create_label_file(args, mesh, output_files, base_dir="./output"):
    """Create a label file with information about the simulation run"""
    import datetime
    
    # Ensure output directory exists
    os.makedirs(base_dir, exist_ok=True)
    label_file = os.path.join(base_dir, f"simulation_info_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt")
    
    # Use UTF-8 encoding to handle the degree symbol
    with open(label_file, "w", encoding='utf-8') as f:
        f.write("EUROPA VISUALIZATION SIMULATION INFORMATION\n")
        f.write("========================================\n\n")
        f.write(f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"VTK File: {args.vtk}\n")
        if args.thermal_vtk:
            f.write(f"Thermal VTK File: {args.thermal_vtk}\n")
        f.write(f"Positions File: {args.positions}\n\n")
        
        f.write("MESH INFORMATION\n")
        f.write("----------------\n")
        f.write(f"Vertices: {mesh.n_points}\n")
        f.write(f"Faces: {mesh.n_cells}\n")
        f.write(f"Time Steps: {len([name for name in mesh.array_names if name.startswith('illumination_')])}\n\n")
        
        f.write("CAMERA SETTINGS\n")
        f.write("---------------\n")
        f.write(f"Camera Latitude: {args.camera_lat} degrees\n")  # Remove degree symbol
        f.write(f"Camera Longitude: {args.camera_lon} degrees\n")  # Remove degree symbol
        f.write(f"Camera Distance: {args.camera_distance} Europa radii\n")
        if args.emission_angle is not None:
            f.write(f"Emission Angle: {args.emission_angle} degrees\n")  # Remove degree symbol
        if args.azimuth is not None:
            f.write(f"Azimuth: {args.azimuth} degrees\n\n")  # Remove degree symbol
        
        f.write("OUTPUT FILES\n")
        f.write("------------\n")
        for desc, file in output_files.items():
            f.write(f"{desc}: {file}\n")
        
    print(f"Created simulation label file: {label_file}")
    return label_file

def main():
    """Command line interface for the visualizer"""
    parser = argparse.ArgumentParser(description='Europa Illumination Visualizer')
    parser.add_argument('--vtk', required=True, help='Path to illumination VTK file')
    parser.add_argument('--thermal-vtk', help='Path to thermal VTK file')
    parser.add_argument('--output-video', help='Output path for animation video (auto-generated if not specified)')
    parser.add_argument('--output-ir-video', help='Output path for Jupiter IR flux animation video (optional)')
    parser.add_argument('--output-thermal-video', help='Output path for thermal animation video (optional)')
    parser.add_argument('--output-maps', help='Output directory for cylindrical maps')
    parser.add_argument('--output-dir', default='./output', help='Directory for auto-generated output files')
    parser.add_argument('--camera-distance', type=float, default=5.0, help='Camera distance (Europa radii)')
    parser.add_argument('--camera-lat', type=float, default=30.0, help='Camera latitude (degrees)')
    parser.add_argument('--camera-lon', type=float, default=0.0, help='Camera longitude (degrees)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for animation')
    parser.add_argument('--positions', required=True, 
                        help='Path to positions CSV file (required for body indicators)')
    parser.add_argument('--location-data', help='Path to location illumination CSV file')
    parser.add_argument('--output-location-plots', help='Output directory for location time series plots')
    parser.add_argument('--save-frames', action='store_true', help='Save individual animation frames')
    parser.add_argument('--width', type=int, default=1024, help='Video frame width in pixels')
    parser.add_argument('--height', type=int, default=768, help='Video frame height in pixels')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true', help='Suppress all non-essential output')
    parser.add_argument('--skip-maps', action='store_true', 
                        help='Skip creation of cylindrical maps')
    parser.add_argument('--create-label-file', action='store_true', 
                        help='Create a label file with simulation details')
    parser.add_argument('--use-default-filenames', action='store_true', 
                        help='Use simple default filenames instead of descriptive ones')
    
    # Camera control options
    parser.add_argument('--emission-angle', type=float, help='Camera emission angle in degrees (0=nadir, 90=horizon)')
    parser.add_argument('--azimuth', type=float, help='Camera azimuth angle in degrees (0=north, 90=east)')
    
    args = parser.parse_args()
    
    # Print camera settings once at program start
    print(f"Camera settings from command line: lat={args.camera_lat}°, lon={args.camera_lon}°, distance={args.camera_distance}")
    
    # Force descriptive filenames by default - simplify the logic
    use_descriptive_filenames = not args.use_default_filenames
    
    # Check for default filenames passed from C++ and override them
    default_paths = {
        'output_video': os.path.join(args.output_dir, "europa_solar_flux.mp4"),
        'output_ir_video': os.path.join(args.output_dir, "europa_jupiter_ir.mp4"),
        'output_thermal_video': os.path.join(args.output_dir, "europa_surface_temp.mp4")
    }
    
    # Reset paths if they match defaults and descriptive filenames are enabled
    if use_descriptive_filenames:
        if args.output_video == default_paths['output_video']:
            args.output_video = None
            print("Resetting default solar animation path to use descriptive filename")
            
        if args.output_ir_video == default_paths['output_ir_video']:
            args.output_ir_video = None
            print("Resetting default IR animation path to use descriptive filename")
            
        if args.output_thermal_video == default_paths['output_thermal_video']:
            args.output_thermal_video = None
            print("Resetting default thermal animation path to use descriptive filename")
    
    print(f"Using descriptive filenames: {use_descriptive_filenames}")
    
    # Only show debug info if verbose is enabled and quiet is not
    if args.verbose and not args.quiet:
        print(f"\n{'='*60}")
        print("EUROPA VISUALIZATION STARTING")
        print(f"VTK file: {args.vtk}")
        print(f"Output thermal video: {args.output_thermal_video}")
        print(f"{'='*60}\n")
    
    visualizer = EuropaVisualizer(
        verbose=args.verbose and not args.quiet,
        width=args.width,
        height=args.height
    )
    if not visualizer.load_vtk(args.vtk):
        return 1
    
    # Track output files for label file creation
    output_files = {}
    
    # Only show array debug info if verbose is enabled and quiet is not
    if args.verbose and not args.quiet:
        print(f"\n{'='*60}")
        print(f"VTK ARRAYS AVAILABLE IN {args.vtk}:")
        for array_name in visualizer.mesh.array_names:
            print(f"  - {array_name}")
        print(f"{'='*60}\n")

    # Ensure we have position data
    position_loaded = False
    if args.positions:
        position_loaded = visualizer.load_positions(args.positions)
    else:
        # Only try to extract from VTK if we don't have a positions file
        position_loaded = visualizer.extract_body_positions_from_vtk()
    
    # Hard requirement: we need valid positions
    if not position_loaded:
        print("ERROR: No valid position data available. Cannot continue.")
        return 1
    
    # Load thermal VTK if specified
    if args.thermal_vtk:
        if not visualizer.load_thermal_vtk(args.thermal_vtk):
            print("Warning: Failed to load thermal VTK file. Thermal animation will be skipped.")
    
    # Save the args for later use
    visualizer.args = args
    
    # Handle illumination animation
    if args.output_video is not None:
        # User explicitly provided a filename
        output_video = args.output_video
    elif use_descriptive_filenames:
        # Auto-generate a descriptive filename
        output_video = generate_descriptive_filename(
            visualizer.mesh,
            "illumination",
            args.camera_lat,
            args.camera_lon,
            args.camera_distance,
            args.output_dir
        )
        print(f"Using auto-generated filename: {output_video}")
    else:
        # Use simple default filename
        output_video = os.path.join(args.output_dir, "europa_solar_flux.mp4")
    
    print(f"Creating illumination animation: {output_video}")
    if visualizer.create_animation(
        output_video,
        camera_distance=args.camera_distance,
        camera_latitude=args.camera_lat,
        camera_longitude=args.camera_lon,
        emission_angle=args.emission_angle,
        azimuth=args.azimuth,
        fps=args.fps,
        save_frames=args.save_frames
    ):
        output_files["Solar Illumination Animation"] = output_video
    
    # Handle Jupiter IR flux animation
    if args.output_ir_video is not None:
        # User explicitly provided a filename
        output_ir_video = args.output_ir_video
    elif use_descriptive_filenames:
        # Auto-generate a descriptive filename
        output_ir_video = generate_descriptive_filename(
            visualizer.mesh,
            "jupiter_ir",
            args.camera_lat,
            args.camera_lon,
            args.camera_distance,
            args.output_dir
        )
        print(f"Using auto-generated filename: {output_ir_video}")
    else:
        # Use simple default filename
        output_ir_video = os.path.join(args.output_dir, "europa_jupiter_ir.mp4")
    
    print(f"Creating Jupiter IR flux animation: {output_ir_video}")
    if visualizer.create_ir_animation(
        output_ir_video,
        camera_distance=args.camera_distance,
        camera_latitude=args.camera_lat,
        camera_longitude=args.camera_lon,
        emission_angle=args.emission_angle,
        azimuth=args.azimuth,
        fps=args.fps,
        save_frames=args.save_frames
    ):
        output_files["Jupiter IR Flux Animation"] = output_ir_video
    
    # Handle thermal animation
    if hasattr(visualizer, 'thermal_mesh') and visualizer.thermal_mesh is not None:
        if args.output_thermal_video is not None:
            # User explicitly provided a filename
            output_thermal_video = args.output_thermal_video
        elif use_descriptive_filenames:
            # Auto-generate a descriptive filename
            output_thermal_video = generate_descriptive_filename(
                visualizer.thermal_mesh,
                "temperature",
                args.camera_lat,
                args.camera_lon,
                args.camera_distance,
                args.output_dir
            )
            print(f"Using auto-generated filename: {output_thermal_video}")
        else:
            # Use simple default filename
            output_thermal_video = os.path.join(args.output_dir, "europa_surface_temp.mp4")
        
        print(f"Creating thermal animation: {output_thermal_video}")
        if visualizer.create_thermal_animation(
            output_thermal_video,
            camera_distance=args.camera_distance,
            camera_latitude=args.camera_lat,
            camera_longitude=args.camera_lon,
            emission_angle=args.emission_angle,
            azimuth=args.azimuth,
            fps=args.fps,
            save_frames=args.save_frames
        ):
            output_files["Surface Temperature Animation"] = output_thermal_video
    elif args.output_thermal_video:
        print("Skipping thermal animation - no thermal data available")
    
    # Handle cylindrical maps
    maps_dir = args.output_maps if args.output_maps else os.path.join(args.output_dir, "maps")
    if not args.skip_maps:
        print(f"Creating cylindrical maps in {maps_dir}...")
        if visualizer.create_cylindrical_maps(maps_dir):
            output_files["Cylindrical Maps Directory"] = maps_dir
    elif args.output_maps and args.skip_maps:
        print("Skipping cylindrical map creation (--skip-maps flag set)")
    
    # Handle location time series plots
    if args.location_data:
        print(f"Location data file provided: {args.location_data}")
        if os.path.exists(args.location_data):
            print(f"Confirmed location data file exists at: {args.location_data}")
            if args.output_location_plots:
                print(f"Will save location plots to: {args.output_location_plots}")
                os.makedirs(args.output_location_plots, exist_ok=True)
                success = visualizer.create_location_time_series(args.location_data, args.output_location_plots)
                if success:
                    print("Location plots created successfully")
                    output_files["location_plots"] = args.output_location_plots
                else:
                    print("Failed to create location plots")
            else:
                print("Warning: Location data provided but no output directory specified. Use --output-location-plots.")
        else:
            print(f"Warning: Location data file does not exist: {args.location_data}")
    else:
        print("No location data file provided, skipping location plots")
    
    # Always create a label file when using descriptive filenames
    if use_descriptive_filenames or args.create_label_file:
        label_file = create_label_file(args, visualizer.mesh, output_files, args.output_dir)
        print(f"Created simulation information file: {label_file}")
    
    return 0

if __name__ == "__main__":
    print("Europa Visualization - Use --width and --height to control output resolution")
    sys.exit(main())