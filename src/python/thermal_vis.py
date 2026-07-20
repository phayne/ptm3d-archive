#!/usr/bin/env python3
# thermal_vis.py - Europa thermal model visualization tool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse
import os

def read_thermal_data(csv_file):
    """Read thermal model data from CSV file."""
    # Read data using pandas
    df = pd.read_csv(csv_file)
    
    # Extract time and temperature columns
    time_hours = df.iloc[:, 0].values  # First column is time in hours
    temperatures = df.iloc[:, 1:].values  # All other columns are temperatures
    
    # Get depth values from column headers
    depths = []
    for col in df.columns[1:]:
        # Extract numerical part from column name (e.g. "T_0.01m" -> 0.01)
        try:
            depth = float(col.split('_')[1].replace('m', ''))
        except:
            depth = float(col.replace('T_', '').replace('m', ''))
        depths.append(depth)
    
    return time_hours, depths, temperatures

def plot_diurnal_temperatures(time_hours, depths, temperatures, output_file=None):
    """Plot surface temperature over a diurnal cycle."""
    # Europa's day in hours
    day_length = 3.55 * 24
    
    # Calculate how many complete days we have in the data
    n_days = int(time_hours[-1] / day_length)
    
    if n_days < 1:
        print("Warning: Less than one day of data available.")
        
    # Extract just the last complete day of data
    # This keeps the data in the original time sequence
    day_start_idx = None
    day_end_idx = None
    
    for i in range(len(time_hours)):
        # Find start of the last day
        if time_hours[i] >= (n_days - 1) * day_length and day_start_idx is None:
            day_start_idx = i
        # Find end of the last day
        if time_hours[i] >= n_days * day_length:
            day_end_idx = i
            break
    
    # If we didn't find the end, use all remaining data
    if day_end_idx is None:
        day_end_idx = len(time_hours)
    
    # Get time and temperature data for just this day
    if day_start_idx is not None:
        day_times = time_hours[day_start_idx:day_end_idx]
        day_temps = temperatures[day_start_idx:day_end_idx, 0]
        
        # Convert to local time (hours since start of day)
        local_time = day_times - (n_days - 1) * day_length
        
        print(f"Using data from day {n_days} with {len(local_time)} data points")
    else:
        # Fallback if we couldn't identify a full day
        print("Warning: Could not extract a full day. Using all data.")
        local_time = time_hours % day_length
        day_temps = temperatures[:, 0]
    
    plt.figure(figsize=(12, 7))
    
    # Plot the actual data points and connect them
    plt.plot(local_time, day_temps, 'ro-', linewidth=1.5, markersize=4)
    
    plt.xlabel('Local Time (hours since midnight)')
    plt.ylabel('Surface Temperature (K)')
    plt.title('Diurnal Surface Temperature Variation on Europa')
    plt.grid(True, alpha=0.3)
    
    # Add markers for key times of day
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Midnight')
    plt.axvline(x=day_length/4, color='orange', linestyle='--', alpha=0.5, label='Sunrise')
    plt.axvline(x=day_length/2, color='yellow', linestyle='--', alpha=0.5, label='Noon')
    plt.axvline(x=3*day_length/4, color='blue', linestyle='--', alpha=0.5, label='Sunset')
    
    # Format x-axis to show in Earth days
    plt.xticks(np.arange(0, day_length+1, 12), 
              [f"{h/24:.1f}" for h in np.arange(0, day_length+1, 12)])
    plt.xlabel('Local Time (Earth days)')
    
    plt.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Diurnal temperature plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_temperature_profiles(time_hours, depths, temperatures, output_file=None):
    """Plot temperature profiles at different times of day."""
    # Europa's day in hours
    day_length = 3.55 * 24
    
    # Convert time to local time
    local_time = np.array([t % day_length for t in time_hours])
    
    # Select profiles at key times of day
    times_to_show = [0, day_length/4, day_length/2, 3*day_length/4]
    labels = ['Midnight', 'Sunrise', 'Noon', 'Sunset']
    colors = ['navy', 'orange', 'red', 'purple']
    
    plt.figure(figsize=(10, 8))
    
    # Maximum depth to show (in meters)
    max_depth = depths[-1]
    
    for target_time, label, color in zip(times_to_show, labels, colors):
        # Find closest time in the data
        idx = np.argmin(np.abs(local_time - target_time))
        
        # Get temperature profile at this time
        temp_profile = temperatures[idx]
        
        # Plot temperature vs depth
        valid_depths = np.array(depths) <= max_depth
        plt.plot(temp_profile[valid_depths], np.array(depths)[valid_depths], 
                 label=f"{label}", linewidth=2.5, color=color)
    
    plt.ylim(max_depth, 0)  # Invert y-axis (depth increases downward)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Depth (m)')
    plt.title('Europa Subsurface Temperature Profiles')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Temperature profile plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def create_temperature_animation(time_hours, depths, temperatures, output_file=None, fps=10):
    """Create an animation of subsurface temperatures over a diurnal cycle."""
    # Europa's day in hours
    day_length = 3.55 * 24
    
    # Calculate how many complete days we have in the data
    n_days = int(time_hours[-1] / day_length)
    
    # Extract just the last complete day of data
    day_start_idx = None
    day_end_idx = None
    
    for i in range(len(time_hours)):
        # Find start of the last day
        if time_hours[i] >= (n_days - 1) * day_length and day_start_idx is None:
            day_start_idx = i
        # Find end of the last day
        if time_hours[i] >= n_days * day_length:
            day_end_idx = i
            break
    
    # If we didn't find the end, use all remaining data
    if day_end_idx is None:
        day_end_idx = len(time_hours)
    
    # Get time and temperature data for just this day
    if day_start_idx is not None:
        day_times = time_hours[day_start_idx:day_end_idx]
        day_temps = temperatures[day_start_idx:day_end_idx]
        
        # Convert to local time (hours since start of day)
        local_time = day_times - (n_days - 1) * day_length
        
        print(f"Using data from day {n_days} with {len(local_time)} data points for animation")
    else:
        # Fallback if we couldn't identify a full day
        print("Warning: Could not extract a full day. Using all data for animation.")
        local_time = time_hours % day_length
        day_temps = temperatures
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Set up the depth vs temperature plot
    max_depth_to_show = depths[-1]
    depth_indices = [i for i, d in enumerate(depths) if d <= max_depth_to_show]
    depths_to_show = [depths[i] for i in depth_indices]
    
    max_temp = np.max(day_temps)
    min_temp = np.min(day_temps)
    
    # Initial plot for depth vs temperature
    line1, = ax1.plot([], [], 'r-', linewidth=2.5)
    ax1.set_xlim(min_temp-2, max_temp+2)
    ax1.set_ylim(max_depth_to_show, 0)  # Invert y-axis
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Subsurface Temperature Profile')
    ax1.grid(True, alpha=0.3)
    
    # Add temperature gradient colormap
    sm = plt.cm.ScalarMappable(cmap='inferno', 
                              norm=plt.Normalize(min_temp, max_temp))
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Temperature (K)')
    
    # Time series plot for surface temperature
    surface_temp = day_temps[:, 0]
    
    # Plot the entire day's surface temperature curve
    ax2.plot(local_time, surface_temp, 'k-', linewidth=1, alpha=0.7)
    point2, = ax2.plot([], [], 'ro', markersize=10)
    ax2.set_xlim(0, day_length)
    ax2.set_ylim(min(surface_temp)-2, max(surface_temp)+2)
    
    # Format x-axis to show in Earth days
    ax2.set_xticks(np.arange(0, day_length+1, 12))
    ax2.set_xticklabels([f"{h/24:.1f}" for h in np.arange(0, day_length+1, 12)])
    ax2.set_xlabel('Local Time (Earth days)')
    ax2.set_ylabel('Surface Temperature (K)')
    ax2.set_title('Surface Temperature Variation')
    ax2.grid(True, alpha=0.3)
    
    # Add markers for key times
    ax2.axvline(x=0, color='navy', linestyle='--', alpha=0.5, label='Midnight')
    ax2.axvline(x=day_length/4, color='orange', linestyle='--', alpha=0.5, label='Sunrise')
    ax2.axvline(x=day_length/2, color='red', linestyle='--', alpha=0.5, label='Noon')
    ax2.axvline(x=3*day_length/4, color='purple', linestyle='--', alpha=0.5, label='Sunset')
    ax2.legend(loc='lower right')
    
    # Add time indicator text
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12)
    
    plt.suptitle('Europa Thermal Model - Temperature Evolution', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Animation update function
    def update(frame):
        # Get temperature profile for current frame
        temp_profile = day_temps[frame, depth_indices]
        
        # Update depth vs temperature plot
        line1.set_data(temp_profile, depths_to_show)
        
        # Update surface temperature time marker
        current_time = local_time[frame]
        current_temp = surface_temp[frame]
        point2.set_data([current_time], [current_temp])
        
        # Update time text
        days = int(current_time / 24)
        hours = int(current_time) % 24
        time_text.set_text(f'Local Time: {days} days, {hours:02d} hours (Surface: {current_temp:.1f}K)')
        
        return line1, point2, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(local_time),
                                   interval=1000/fps, blit=True)
    
    if output_file:
        # Save as MP4
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Europa Thermal Model'), 
                                      bitrate=1800)
        anim.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize Europa thermal model results')
    parser.add_argument('csv_file', help='CSV file containing thermal model results')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    parser.add_argument('--diurnal', action='store_true', help='Create diurnal temperature plot')
    parser.add_argument('--profiles', action='store_true', help='Create temperature profiles plot')
    parser.add_argument('--animation', action='store_true', help='Create temperature animation')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for animation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read data
    time_hours, depths, temperatures = read_thermal_data(args.csv_file)
    
    base_name = Path(args.csv_file).stem
    
    # If no specific plots requested, create all
    if not (args.diurnal or args.profiles or args.animation):
        args.diurnal = args.profiles = args.animation = True
    
    # Create visualizations
    if args.diurnal:
        output_file = os.path.join(args.output_dir, f'{base_name}_diurnal.png')
        plot_diurnal_temperatures(time_hours, depths, temperatures, output_file)
        
    if args.profiles:
        output_file = os.path.join(args.output_dir, f'{base_name}_profiles.png')
        plot_temperature_profiles(time_hours, depths, temperatures, output_file)
        
    if args.animation:
        output_file = os.path.join(args.output_dir, f'{base_name}_animation.mp4')
        create_temperature_animation(time_hours, depths, temperatures, output_file, args.fps)
        
if __name__ == '__main__':
    main()