"""
Europa Exosphere Analysis and Visualization Tools
==================================================

This module provides Python tools for analyzing and visualizing the 
ballistic exosphere transport model results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.sparse as sp
from scipy.special import kn  # Modified Bessel function of second kind
from scipy.integrate import quad
import os
from typing import Dict, List, Tuple, Optional
import h5py
import struct

# Physical constants for Europa
EUROPA_RADIUS = 1560.8e3  # meters
EUROPA_GRAVITY = 1.314    # m/s^2
BOLTZMANN_K = 1.38064852e-23  # J/K
WATER_MOLAR_MASS = 0.01801528  # kg/mol
AVOGADRO_NA = 6.022140857e23  # mol^-1
WATER_MOLECULAR_MASS = WATER_MOLAR_MASS / AVOGADRO_NA  # kg

class ExosphereAnalysis:
    """Main class for analyzing exosphere simulation results."""
    
    def __init__(self, output_dir: str = "./exosphere_output"):
        self.output_dir = output_dir
        self.mesh_data = None
        self.temperature_data = None
        self.exosphere_states = []
        self.transport_matrix = None
        
    def load_mesh(self, mesh_file: str = None) -> bool:
        """Load mesh data from file or create default icosphere."""
        if mesh_file and os.path.exists(mesh_file):
            # Load custom mesh
            self.mesh_data = self._load_custom_mesh(mesh_file)
        else:
            # Create default icosphere
            self.mesh_data = self._create_icosphere(resolution=4)
        
        print(f"Loaded mesh with {len(self.mesh_data['faces'])} faces")
        return True
    
    def _create_icosphere(self, resolution: int = 4) -> Dict:
        """Create an icosphere mesh."""
        # Golden ratio
        t = (1.0 + np.sqrt(5.0)) / 2.0
        norm = np.sqrt(1.0 + t * t)
        
        # Initial icosahedron vertices
        vertices = np.array([
            [-1/norm, t/norm, 0],
            [1/norm, t/norm, 0],
            [-1/norm, -t/norm, 0],
            [1/norm, -t/norm, 0],
            [0, -1/norm, t/norm],
            [0, 1/norm, t/norm],
            [0, -1/norm, -t/norm],
            [0, 1/norm, -t/norm],
            [t/norm, 0, -1/norm],
            [t/norm, 0, 1/norm],
            [-t/norm, 0, -1/norm],
            [-t/norm, 0, 1/norm]
        ])
        
        # Initial faces
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        
        # Subdivide to increase resolution
        for _ in range(resolution):
            vertices, faces = self._subdivide_mesh(vertices, faces)
        
        # Scale to Europa radius
        vertices *= EUROPA_RADIUS
        
        # Calculate face centers and areas
        face_centers = []
        face_areas = []
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            center = (v0 + v1 + v2) / 3.0
            # Project to sphere
            center = center / np.linalg.norm(center) * EUROPA_RADIUS
            face_centers.append(center)
            
            # Calculate area
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            face_areas.append(area)
        
        return {
            'vertices': vertices,
            'faces': faces,
            'face_centers': np.array(face_centers),
            'face_areas': np.array(face_areas),
            'num_faces': len(faces)
        }
    
    def _subdivide_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple:
        """Subdivide a triangular mesh."""
        edge_cache = {}
        
        def get_midpoint(v1_idx, v2_idx):
            key = tuple(sorted([v1_idx, v2_idx]))
            if key not in edge_cache:
                v1 = vertices[v1_idx]
                v2 = vertices[v2_idx]
                midpoint = (v1 + v2) / 2.0
                # Project to unit sphere
                midpoint = midpoint / np.linalg.norm(midpoint)
                edge_cache[key] = len(vertices) + len(edge_cache)
            return edge_cache[key]
        
        new_faces = []
        for face in faces:
            v0, v1, v2 = face
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)
            
            new_faces.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20]
            ])
        
        # Add new vertices
        new_vertices = list(vertices)
        for key, idx in sorted(edge_cache.items(), key=lambda x: x[1]):
            v1, v2 = vertices[key[0]], vertices[key[1]]
            midpoint = (v1 + v2) / 2.0
            midpoint = midpoint / np.linalg.norm(midpoint)
            new_vertices.append(midpoint)
        
        return np.array(new_vertices), np.array(new_faces)
    
    def load_temperature_data(self, temperature_dir: str, num_timesteps: int) -> bool:
        """Load temperature data from PTM binary output."""
        self.temperature_data = []
        
        for t in range(num_timesteps):
            filename = f"{temperature_dir}/temperature/temperature_{t:03d}.bin"
            
            if not os.path.exists(filename):
                print(f"Warning: Temperature file not found: {filename}")
                continue
            
            with open(filename, 'rb') as f:
                # Read header
                num_faces = struct.unpack('I', f.read(4))[0]
                
                # Read temperature data
                temps = np.frombuffer(f.read(num_faces * 4), dtype=np.float32)
                self.temperature_data.append(temps)
        
        self.temperature_data = np.array(self.temperature_data)
        print(f"Loaded temperature data: {self.temperature_data.shape}")
        return True
    
    def calculate_vapor_pressure(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate vapor pressure of water ice."""
        # Avoid log of zero
        temp = np.maximum(temperature, 1.0)
        
        # Wagner & Pruss formulation for ice
        ln_p = -5723.265 / temp + 3.53068 * np.log(temp) - 0.00728332 * temp + 11.8829
        return np.exp(ln_p)
    
    def calculate_sublimation_flux(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate sublimation flux in kg/(m^2·s)."""
        vapor_pressure = self.calculate_vapor_pressure(temperature)
        vapor_density = vapor_pressure * WATER_MOLAR_MASS / (8.314 * temperature)
        thermal_velocity = np.sqrt(8 * BOLTZMANN_K * temperature / (np.pi * WATER_MOLECULAR_MASS))
        
        # Hertz-Knudsen equation
        return vapor_density * thermal_velocity / 4.0
    
    def build_transport_matrix(self, temperature: np.ndarray, 
                              sparse: bool = True) -> sp.csr_matrix:
        """Build the ballistic transport matrix."""
        n_faces = len(self.mesh_data['face_centers'])
        face_centers = self.mesh_data['face_centers']
        face_areas = self.mesh_data['face_areas']
        
        if sparse:
            rows, cols, data = [], [], []
        else:
            matrix = np.zeros((n_faces, n_faces))
        
        for i in range(n_faces):
            if temperature[i] <= 50:  # Skip frozen faces
                if sparse:
                    rows.append(i)
                    cols.append(i)
                    data.append(1.0)
                else:
                    matrix[i, i] = 1.0
                continue
            
            # Calculate sigma for this temperature
            sigma2 = BOLTZMANN_K * temperature[i] / WATER_MOLECULAR_MASS
            
            # Calculate transport probabilities
            probs = np.zeros(n_faces)
            
            for j in range(n_faces):
                if i == j:
                    continue
                
                # Great circle distance
                cos_angle = np.dot(face_centers[i], face_centers[j]) / (EUROPA_RADIUS ** 2)
                cos_angle = np.clip(cos_angle, -1, 1)
                distance = EUROPA_RADIUS * np.arccos(cos_angle)
                
                # Transport probability
                xi = EUROPA_GRAVITY * distance / (2 * sigma2)
                if xi < 10:  # Avoid underflow
                    # Modified Bessel function K0
                    k0_val = kn(0, xi) if xi > 0 else np.inf
                    prefactor = EUROPA_GRAVITY ** 2 * distance / (4 * sigma2 ** 2)
                    prob = prefactor * k0_val * face_areas[j] / (distance ** 2)
                    probs[j] = prob
            
            # Normalize probabilities
            total_prob = np.sum(probs)
            if total_prob > 0:
                probs = probs / total_prob
                # Account for escape
                escape_fraction = min(0.1, temperature[i] / 200)  # Simple escape model
                probs = probs * (1 - escape_fraction)
                probs[i] = escape_fraction  # Fraction that returns to same face
            else:
                probs[i] = 1.0  # All stays on same face
            
            # Store in matrix
            if sparse:
                for j in range(n_faces):
                    if probs[j] > 1e-6:
                        rows.append(j)
                        cols.append(i)
                        data.append(probs[j])
            else:
                matrix[:, i] = probs
        
        if sparse:
            return sp.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
        else:
            return matrix
    
    def simulate_timestep(self, temperature: np.ndarray, dt: float = 3600.0) -> Dict:
        """Simulate one timestep of exosphere evolution."""
        sublimation_flux = self.calculate_sublimation_flux(temperature)
        transport_matrix = self.build_transport_matrix(temperature)
        
        # Total mass emission from each face
        emission = sublimation_flux * self.mesh_data['face_areas'] * dt
        
        # Transport the mass
        deposited = transport_matrix @ emission
        
        # Net change
        net_change = deposited - emission
        
        return {
            'sublimation_flux': sublimation_flux,
            'deposited_mass': deposited,
            'net_change': net_change,
            'total_sublimation': np.sum(emission),
            'total_deposition': np.sum(deposited[deposited > 0]),
            'escape_rate': np.sum(emission) - np.sum(deposited)
        }
    
    def run_simulation(self, export_interval: int = 10) -> None:
        """Run complete exosphere simulation."""
        num_timesteps = len(self.temperature_data)
        dt = 3600.0  # 1 hour timesteps
        
        accumulated_mass = np.zeros(self.mesh_data['num_faces'])
        
        for t in range(num_timesteps):
            print(f"Processing timestep {t+1}/{num_timesteps}")
            
            result = self.simulate_timestep(self.temperature_data[t], dt)
            accumulated_mass += result['net_change']
            
            # Store state
            self.exosphere_states.append({
                'timestep': t,
                'accumulated_mass': accumulated_mass.copy(),
                **result
            })
            
            if t % export_interval == 0:
                self.export_timestep(t)
    
    def export_timestep(self, timestep: int) -> None:
        """Export simulation state at given timestep."""
        state = self.exosphere_states[timestep]
        filename = f"{self.output_dir}/exosphere_t{timestep:03d}.npz"
        
        np.savez(filename, **state)
        print(f"Exported timestep {timestep} to {filename}")
    
    def plot_surface_distribution(self, timestep: int = -1, 
                                 quantity: str = 'sublimation_flux') -> None:
        """Plot surface distribution on sphere."""
        state = self.exosphere_states[timestep]
        values = state[quantity]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert to lat/lon for plotting
        face_centers = self.mesh_data['face_centers']
        x = face_centers[:, 0]
        y = face_centers[:, 1]
        z = face_centers[:, 2]
        
        # Create scatter plot
        p = ax.scatter(x, y, z, c=values, cmap='viridis', s=1, 
                      vmin=np.percentile(values, 5), 
                      vmax=np.percentile(values, 95))
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{quantity} at timestep {timestep}')
        
        plt.colorbar(p, ax=ax, label=quantity)
        plt.show()
    
    def plot_mass_balance_evolution(self) -> None:
        """Plot evolution of mass balance over time."""
        timesteps = [s['timestep'] for s in self.exosphere_states]
        sublimation = [s['total_sublimation'] for s in self.exosphere_states]
        deposition = [s['total_deposition'] for s in self.exosphere_states]
        escape = [s['escape_rate'] for s in self.exosphere_states]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        axes[0].plot(timesteps, sublimation, label='Sublimation', color='red')
        axes[0].plot(timesteps, deposition, label='Deposition', color='blue')
        axes[0].set_ylabel('Mass Rate (kg)')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(timesteps, escape, label='Escape', color='orange')
        axes[1].set_ylabel('Escape Rate (kg)')
        axes[1].legend()
        axes[1].grid(True)
        
        net_change = np.array(sublimation) - np.array(deposition)
        axes[2].plot(timesteps, np.cumsum(net_change), label='Cumulative Net Loss', color='green')
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Cumulative Mass (kg)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.suptitle('Europa Exosphere Mass Balance Evolution')
        plt.tight_layout()
        plt.show()
    
    def plot_latitude_distribution(self, timestep: int = -1) -> None:
        """Plot distribution vs latitude."""
        state = self.exosphere_states[timestep]
        face_centers = self.mesh_data['face_centers']
        
        # Calculate latitudes
        r = np.linalg.norm(face_centers, axis=1)
        latitudes = np.degrees(np.arcsin(face_centers[:, 2] / r))
        
        # Bin by latitude
        lat_bins = np.linspace(-90, 90, 37)
        lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
        
        quantities = ['sublimation_flux', 'accumulated_mass']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, quantity in zip(axes, quantities):
            values = state[quantity]
            binned_values = []
            
            for i in range(len(lat_bins) - 1):
                mask = (latitudes >= lat_bins[i]) & (latitudes < lat_bins[i+1])
                if np.any(mask):
                    binned_values.append(np.mean(values[mask]))
                else:
                    binned_values.append(0)
            
            ax.plot(lat_centers, binned_values)
            ax.set_xlabel('Latitude (degrees)')
            ax.set_ylabel(quantity)
            ax.set_title(f'{quantity} vs Latitude')
            ax.grid(True)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.suptitle(f'Latitude Distribution at Timestep {timestep}')
        plt.tight_layout()
        plt.show()
    
    def calculate_transport_statistics(self) -> Dict:
        """Calculate transport statistics."""
        stats = {}
        
        # Average hop distance
        face_centers = self.mesh_data['face_centers']
        distances = []
        
        for i in range(min(100, len(face_centers))):  # Sample
            for j in range(i+1, min(100, len(face_centers))):
                cos_angle = np.dot(face_centers[i], face_centers[j]) / (EUROPA_RADIUS ** 2)
                cos_angle = np.clip(cos_angle, -1, 1)
                distance = EUROPA_RADIUS * np.arccos(cos_angle)
                distances.append(distance)
        
        stats['mean_hop_distance'] = np.mean(distances) / 1000  # km
        stats['max_hop_distance'] = np.max(distances) / 1000  # km
        
        # Characteristic hop time
        mean_temp = np.mean(self.temperature_data)
        thermal_velocity = np.sqrt(8 * BOLTZMANN_K * mean_temp / (np.pi * WATER_MOLECULAR_MASS))
        stats['mean_hop_time'] = stats['mean_hop_distance'] * 1000 / thermal_velocity / 60  # minutes
        
        # Escape velocity
        escape_velocity = np.sqrt(2 * EUROPA_GRAVITY * EUROPA_RADIUS)
        stats['escape_velocity'] = escape_velocity
        stats['thermal_to_escape_ratio'] = thermal_velocity / escape_velocity
        
        return stats

def main():
    """Main analysis workflow."""
    print("Europa Exosphere Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ExosphereAnalysis("./exosphere_output")
    
    # Load mesh
    analyzer.load_mesh()
    
    # Load temperature data
    analyzer.load_temperature_data("./output", 72)
    
    # Run simulation
    analyzer.run_simulation(export_interval=10)
    
    # Generate plots
    analyzer.plot_surface_distribution(timestep=-1, quantity='sublimation_flux')
    analyzer.plot_mass_balance_evolution()
    analyzer.plot_latitude_distribution(timestep=-1)
    
    # Calculate statistics
    stats = analyzer.calculate_transport_statistics()
    print("\nTransport Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")

if __name__ == "__main__":
    main()