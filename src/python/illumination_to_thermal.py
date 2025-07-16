import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
from datetime import datetime
import argparse
from tqdm import tqdm

# Add necessary paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src/python'))
sys.path.insert(0, os.path.join(project_root, 'build'))

# Try to import thermal model
print(f"Build directory: {os.path.abspath('build')}")
print(f"Python path: {sys.path}")

# Add the build directory to the path if needed
sys.path.insert(0, 'build')
# Add python_modules directory to path to find thermal_model_cpp
sys.path.insert(0, os.path.join(project_root, 'build/python_modules'))

try:
    import thermal_model_cpp as tm
    print(f"Successfully imported thermal_model_cpp")
    
    # Test creating a basic object
    config = tm.Config()
    planet = tm.Planet("Europa")
    print(f"Created Planet object named: {planet.name}")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the thermal model module is properly compiled.")
    print("You may need to run 'cmake .. && make' in the build directory.")
    sys.exit(1)

def read_vtk_mesh(vtk_file):
    """Read VTK file and return the mesh object"""
    print(f"Reading VTK file: {vtk_file}")
    try:
        mesh = pv.read(vtk_file)
        print(f"Loaded mesh with {mesh.n_points} points and {mesh.n_cells} cells")
        
        # Print available arrays to help with debugging
        print("Available arrays in the mesh:")
        for name in mesh.array_names:
            print(f"  - {name}")
        
        return mesh
    except Exception as e:
        print(f"Error reading VTK file: {str(e)}")
        return None

def extract_times_from_vtk(mesh):
    """Extract time values from the VTK file"""
    if not mesh:
        return []
        
    # Try to find time data in field data
    if 'TIME_STEPS' in mesh.field_data:
        times = mesh.field_data['TIME_STEPS']
        print(f"Found {len(times)} time steps in VTK file")
        return times
    else:
        print("No TIME_STEPS field found, checking for time series arrays...")
        # Count the number of illumination arrays
        illumination_arrays = [key for key in mesh.array_names if key.startswith('illumination_')]
        
        if illumination_arrays:
            print(f"Found {len(illumination_arrays)} illumination time series arrays")
            
            # Extract time step from array names (format: illumination_X where X is the time step)
            times = []
            for arr_name in illumination_arrays:
                try:
                    step = int(arr_name.split('_')[1])
                    times.append(step)
                except (IndexError, ValueError):
                    pass
            
            # Sort times and convert to Europa day fraction
            times = sorted(times)
            
            # Europa day in seconds
            europa_day = 3.55 * 24 * 3600
            
            # If we found time steps, convert them to seconds
            if times:
                normalized_times = [t * (europa_day / len(times)) for t in times]
                return np.array(normalized_times)
        
        # If no time data found, create evenly spaced points across one Europa day
        print("No time data found, creating dummy time values")
        europa_day = 3.55 * 24 * 3600  # Europa day in seconds
        return np.linspace(0, europa_day, 360)  # Default to 360 points

def find_nearest_location_on_mesh(mesh, lat, lon):
    """Find the nearest face on the mesh to a given lat/lon point"""
    # Make sure we have lat/lon data
    if 'latitude' not in mesh.array_names or 'longitude' not in mesh.array_names:
        print("Error: Mesh does not have latitude/longitude data")
        return None
    
    # Get mesh lat/lon values
    mesh_lats = mesh['latitude']
    mesh_lons = mesh['longitude']
    
    # Handle longitude wrapping
    mesh_lons_adj = np.copy(mesh_lons)
    if lon < 0:
        mesh_lons_adj[mesh_lons_adj > 180] -= 360
    
    # Calculate spherical distance
    dlat = mesh_lats - lat
    dlon = mesh_lons_adj - lon
    # Handle longitude wrapping at boundaries
    dlon = np.minimum(np.abs(dlon), 360 - np.abs(dlon))
    
    # Weight latitude more heavily to account for convergence at poles
    dist = np.sqrt(dlat**2 + (dlon * np.cos(np.radians(lat)))**2)
    
    # Find index of closest point
    face_idx = np.argmin(dist)
    
    # Print match info
    print(f"Location (lat={lat:.2f}, lon={lon:.2f}) mapped to mesh face {face_idx}")
    print(f"  Mesh face coordinates: lat={mesh_lats[face_idx]:.2f}, lon={mesh_lons[face_idx]:.2f}")
    
    return face_idx

def extract_flux_data_for_location(mesh, face_idx, times):
    """Extract solar and IR flux time series data for a specific mesh face"""
    if not mesh or face_idx is None:
        return None, None
    
    time_steps = len(times)
    solar_flux = np.zeros(time_steps)
    jupiter_ir = np.zeros(time_steps)
    
    # Get visible albedo for this face
    vis_albedo = 0.67  # Default Europa albedo
    ir_albedo = 0.05   # Default IR albedo
    
    if 'albedo' in mesh.array_names:
        vis_albedo = mesh['albedo'][face_idx]
        print(f"Using mesh visible albedo: {vis_albedo}")
    
    if 'ir_albedo' in mesh.array_names:
        ir_albedo = mesh['ir_albedo'][face_idx]
        print(f"Using mesh IR albedo: {ir_albedo}")
    
    # Extract flux values for each time step
    for t in range(time_steps):
        solar_name = f'illumination_{t}'
        ir_name = f'jupiter_ir_{t}'
        
        if solar_name in mesh.array_names:
            solar_flux[t] = mesh[solar_name][face_idx]
        else:
            # Try alternative naming convention
            alt_solar_name = f'solar_flux_{t}'
            if alt_solar_name in mesh.array_names:
                solar_flux[t] = mesh[alt_solar_name][face_idx]
        
        if ir_name in mesh.array_names:
            jupiter_ir[t] = mesh[ir_name][face_idx]
        else:
            # Try alternative naming convention
            alt_ir_name = f'jupiter_thermal_{t}'
            if alt_ir_name in mesh.array_names:
                jupiter_ir[t] = mesh[alt_ir_name][face_idx]
    
    return solar_flux, jupiter_ir, vis_albedo, ir_albedo

def create_europa_thermal_model():
    """Create a Europa planet object with appropriate parameters"""
    europa = tm.Planet("Europa")
    
    # Set Europa parameters from thermal_model.cpp
    europa.albedo = 0.67       # Europa's average albedo
    europa.emissivity = 0.95   # Typical ice emissivity
    europa.day = 3.55 * 24 * 3600  # Rotation period in seconds (~3.55 Earth days)
    europa.year = europa.day   # Same as day for synchronous rotation)
    europa.rAU = 5.2           # Distance from Sun in AU
    europa.S = 50.0            # Solar constant at Europa's orbit (~50 W/m²)
    europa.Qb = 0.02           # Internal heat flux (20 mW/m²)
    
    # Ice thermal properties
    europa.ks = 0.4            # Surface thermal conductivity
    europa.kd = 5.0            # Deep thermal conductivity of ice
    europa.rhos = 400.0        # Surface density (kg/m³)
    europa.rhod = 920.0        # Deep density (solid ice)
    europa.H = 0.1             # Scale height for density/conductivity change
    europa.cp0 = 900.0         # Reference heat capacity of ice (J/kg/K)
    
    # Temperature-dependent properties
    europa.cpCoeff = [7.0, 0.0]  # Heat capacity coefficients
    europa.albedoCoef = [0.1, 0.05]  # Albedo coefficients
    
    return europa

def run_thermal_model_with_flux(lat, lon, times, solar_flux, jupiter_ir, vis_albedo=0.67, ir_albedo=0.05, ndays=2):
    """
    Run thermal model using flux data from illumination model.
    """
    # Create Europa planet object
    europa = create_europa_thermal_model()
    
    # Override with provided albedo
    europa.albedo = vis_albedo
    
    # Create configuration with safer parameters for numerical stability
    config = tm.Config()
    config.F = 0.05          # Reduced Fourier number for better stability
    config.NYEARSEQ = 5      # Fewer equilibration days to avoid overshooting
    config.NPERDAY = 48      # Keep higher time resolution for output
    
    # Calculate absorbed flux
    absorbed_solar = solar_flux * (1.0 - vis_albedo)
    absorbed_ir = jupiter_ir * (1.0 - ir_albedo)
    absorbed_flux = absorbed_solar + absorbed_ir
    
    # Report flux statistics
    print(f"  Flux statistics:")
    print(f"    Solar flux range: {solar_flux.min():.3f} to {solar_flux.max():.3f} W/m²")
    print(f"    Jupiter IR range: {jupiter_ir.min():.3f} to {jupiter_ir.max():.3f} W/m²")
    print(f"    Total absorbed flux range: {absorbed_flux.min():.3f} to {absorbed_flux.max():.3f} W/m²")
    
    # Create flux array for thermal model
    flux_array = np.column_stack((times, absorbed_flux))
    
    # Convert latitude to radians
    lat_rad = np.radians(lat)
    
    # Create and run the model
    model = tm.Model(europa, lat_rad, ndays, flux_array, config)
    model.run()
    
    return model

def plot_thermal_results(model, times, solar_flux, jupiter_ir, latitude, longitude, 
                        vis_albedo, ir_albedo, output_file=None):
    """Plot thermal model results with flux inputs"""
    plt.figure(figsize=(12, 12))
    
    # Create four subplots
    gs = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 1.5])
    
    # 1. Plot input solar flux
    ax1 = plt.subplot(gs[0])
    hours = np.array(times) / 3600.0
    ax1.plot(hours, solar_flux, 'r-', lw=2, label='Solar Flux')
    ax1.set_ylabel('Solar Flux (W/m²)')
    ax1.set_title(f'Illumination-Thermal Model Results for Lat={latitude}°, Lon={longitude}°')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # 2. Plot input Jupiter IR flux
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(hours, jupiter_ir, 'orange', lw=2, label='Jupiter IR')
    ax2.set_ylabel('Jupiter IR (W/m²)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # 3. Plot surface temperature
    ax3 = plt.subplot(gs[2], sharex=ax1)
    # Convert time to hours
    model_hours = model.lt / 3600.0
    surface_temp = model.T[:, 0]
    ax3.plot(model_hours, surface_temp, 'b-', lw=2)
    ax3.set_ylabel('Surface Temp (K)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Plot temperature depth profile at key times
    ax4 = plt.subplot(gs[3])
    depths = np.array(model.profile.z)
    
    # Select a few representative times
    time_indices = []
    
    # Find indices for key times (midnight, dawn, noon, dusk)
    day_fraction = model_hours / (model.planet.day/3600)
    day_fraction = day_fraction % 1.0  # Get fractional part of the day
    
    # Find closest indices to desired day fractions
    key_fractions = [0.0, 0.25, 0.5, 0.75]  # midnight, dawn, noon, dusk
    
    for frac in key_fractions:
        idx = np.argmin(np.abs(day_fraction - frac))
        time_indices.append(idx)
    
    # Plot depth profiles
    labels = ['Midnight', 'Dawn', 'Noon', 'Dusk']
    colors = ['purple', 'orange', 'red', 'blue']
    
    for i, idx in enumerate(time_indices):
        if idx < len(model.T):  # Make sure index is valid
            ax4.plot(model.T[idx, :], depths, c=colors[i], lw=2, label=f'{labels[i]} ({model_hours[idx]:.1f}h)')
    
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Depth (m)')
    ax4.set_yscale('log')  # Use log scale for better visualization of depths
    ax4.invert_yaxis()  # Depth increases downward
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add model parameters as text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    param_text = (f"Visible Albedo: {vis_albedo:.2f}\n"
                 f"IR Albedo: {ir_albedo:.2f}\n"
                 f"Emissivity: {model.planet.emissivity:.2f}\n"
                 f"Day Length: {model.planet.day/3600:.1f} hrs\n"
                 f"Skin Depth: {depths[1]*8:.2f}m")
    ax4.text(0.05, 0.05, param_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        
    return plt.gcf(), (ax1, ax2, ax3, ax4)

def save_temperature_data(model, times, solar_flux, jupiter_ir, lat, lon, output_file):
    """Save temperature time series data to CSV file"""
    # Create dataframe with times, fluxes, and temperatures
    data = {
        'Time_Seconds': model.lt,
        'Time_Hours': model.lt / 3600.0,
        'Surface_Temp_K': model.T[:, 0],
        'Solar_Flux_Wm2': np.interp(model.lt, times, solar_flux),
        'Jupiter_IR_Wm2': np.interp(model.lt, times, jupiter_ir)
    }
    
    # Add some deeper layer temperatures
    depths = np.array(model.profile.z)
    layer_indices = [0, 1, 2, 5, 10, 20]
    for i in layer_indices:
        if i < len(depths):
            data[f'Temp_at_{depths[i]:.4f}m'] = model.T[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved temperature data to {output_file}")
    
    return df

def process_location(vtk_file, lat, lon, vis_albedo=None, ir_albedo=None, ndays=2, output_dir='output'):
    """Process a single location from the VTK file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Read the VTK file
    mesh = read_vtk_mesh(vtk_file)
    if mesh is None:
        print("Failed to read VTK file")
        return None
    
    # Extract times from VTK file
    times = extract_times_from_vtk(mesh)
    if len(times) == 0:
        print("No time steps found in VTK file")
        return None
    
    # Find the nearest face on the mesh
    face_idx = find_nearest_location_on_mesh(mesh, lat, lon)
    if face_idx is None:
        print("Failed to find matching location on mesh")
        return None
    
    # Extract flux data for the location
    solar_flux, jupiter_ir, mesh_vis_albedo, mesh_ir_albedo = extract_flux_data_for_location(mesh, face_idx, times)
    
    # Use provided albedo values if available, otherwise use mesh values
    final_vis_albedo = vis_albedo if vis_albedo is not None else mesh_vis_albedo
    final_ir_albedo = ir_albedo if ir_albedo is not None else mesh_ir_albedo
    
    # Save flux data to file
    flux_file = os.path.join(output_dir, f"flux_lat{lat:.1f}_lon{lon:.1f}_{timestamp}.csv")
    flux_df = pd.DataFrame({
        'Time_Seconds': times,
        'Time_Hours': times / 3600.0,
        'Solar_Flux_Wm2': solar_flux,
        'Jupiter_IR_Wm2': jupiter_ir,
        'Visible_Albedo': final_vis_albedo,
        'IR_Albedo': final_ir_albedo
    })
    flux_df.to_csv(flux_file, index=False)
    print(f"Saved flux data to {flux_file}")
    
    # Run thermal model
    model = run_thermal_model_with_flux(
        lat, lon, times, solar_flux, jupiter_ir, 
        vis_albedo=final_vis_albedo, 
        ir_albedo=final_ir_albedo,
        ndays=ndays
    )
    
    # Plot and save results
    plot_file = os.path.join(output_dir, f"thermal_lat{lat:.1f}_lon{lon:.1f}_{timestamp}.png")
    plot_thermal_results(
        model, times, solar_flux, jupiter_ir, 
        lat, lon, final_vis_albedo, final_ir_albedo, plot_file
    )
    
    # Save temperature data
    temp_file = os.path.join(output_dir, f"temp_lat{lat:.1f}_lon{lon:.1f}_{timestamp}.csv")
    save_temperature_data(model, times, solar_flux, jupiter_ir, lat, lon, temp_file)
    
    return model

def read_locations_file(filename):
    """Read locations from a file with format: lat lon name"""
    locations = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.strip() == '' or line.strip().startswith('#'):
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    name = ' '.join(parts[2:]) if len(parts) > 2 else f"Location_{lat}_{lon}"
                    locations.append((lat, lon, name))
    except Exception as e:
        print(f"Error reading locations file: {str(e)}")
        
    return locations

def main():
    """Main function to run the illumination to thermal pipeline"""
    parser = argparse.ArgumentParser(description="Convert illumination model output to thermal model temperatures")
    parser.add_argument('--vtk', required=True, help='Path to VTK file with illumination data')
    parser.add_argument('--lat', type=float, help='Single latitude to process')
    parser.add_argument('--lon', type=float, help='Single longitude to process')
    parser.add_argument('--locations', help='File with locations to process')
    parser.add_argument('--vis-albedo', type=float, help='Override visible albedo for solar flux (default: use mesh value)')
    parser.add_argument('--ir-albedo', type=float, help='Override IR albedo for Jupiter flux (default: use mesh value)')
    parser.add_argument('--days', type=int, default=2, help='Number of days to simulate (default: 2)')
    parser.add_argument('--output', default='thermal_output', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process either a single location or multiple locations from file
    if args.locations:
        locations = read_locations_file(args.locations)
        print(f"Processing {len(locations)} locations from file: {args.locations}")
        for lat, lon, name in locations:
            print(f"\nProcessing location: {name} (lat={lat}, lon={lon})...")
            location_dir = os.path.join(args.output, name.replace(' ', '_'))
            process_location(
                args.vtk, lat, lon, 
                vis_albedo=args.vis_albedo, 
                ir_albedo=args.ir_albedo,
                ndays=args.days, 
                output_dir=location_dir
            )
    elif args.lat is not None and args.lon is not None:
        print(f"Processing single location: lat={args.lat}, lon={args.lon}")
        process_location(
            args.vtk, args.lat, args.lon, 
            vis_albedo=args.vis_albedo, 
            ir_albedo=args.ir_albedo,
            ndays=args.days, 
            output_dir=args.output
        )
    else:
        print("Error: Must provide either --lat/--lon or --locations")
        parser.print_help()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())