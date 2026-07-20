#!/usr/bin/env python3
"""
Diagnostic Analysis for Europa Exosphere Transport Model
Run this after the C++ simulation with diagnostics enabled
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, ndimage
from matplotlib import cm
import seaborn as sns
from pathlib import Path
import sys

def analyze_transport_diagnostics(output_dir):
    """Main diagnostic analysis function"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    print("\n" + "="*60)
    print("TRANSPORT PATTERN DIAGNOSTIC ANALYSIS")
    print("="*60)
    
    # Load diagnostic data
    print("\n1. Loading diagnostic data...")
    
    # Load transport diagnostics
    try:
        trans_data = pd.read_csv(output_path / "transport_diagnostics.txt", 
                                 sep=' ', comment='#',
                                 names=['face_id', 'column_sum', 'escape_fraction', 
                                       'temperature', 'deposition_flux', 
                                       'emission_flux', 'net_flux'])
        print(f"   Loaded {len(trans_data)} faces")
    except Exception as e:
        print(f"   Error loading transport diagnostics: {e}")
        return
    
    # Load neighbor information
    try:
        neighbor_data = []
        with open(output_path / "mesh_neighbors.txt", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                face_id = int(parts[0])
                num_neighbors = int(parts[1])
                # Extract is_pentagon flag
                is_pentagon = int(parts[2 + num_neighbors]) if len(parts) > 2 + num_neighbors else 0
                neighbor_data.append({
                    'face_id': face_id,
                    'num_neighbors': num_neighbors,
                    'is_pentagon': is_pentagon
                })
        
        neighbor_df = pd.DataFrame(neighbor_data)
        print(f"   Loaded neighbor data for {len(neighbor_df)} faces")
    except Exception as e:
        print(f"   Warning: Could not load neighbor data: {e}")
        neighbor_df = None
    
    # Merge datasets
    if neighbor_df is not None:
        data = trans_data.merge(neighbor_df, on='face_id', how='left')
    else:
        data = trans_data
        data['is_pentagon'] = 0
    
    # 2. Mass Conservation Analysis
    print("\n2. Mass Conservation Analysis:")
    print(f"   Column sum statistics:")
    print(f"     Min: {data['column_sum'].min():.6f}")
    print(f"     Max: {data['column_sum'].max():.6f}")
    print(f"     Mean: {data['column_sum'].mean():.6f}")
    print(f"     Std: {data['column_sum'].std():.6f}")
    
    violations = len(data[data['column_sum'] > 1.001])
    if violations > 0:
        print(f"   ⚠ WARNING: {violations} faces violate mass conservation!")
        print(f"     ({100*violations/len(data):.1f}% of all faces)")
    
    # 3. Pentagon Effect Analysis
    print("\n3. Pentagon Effect Analysis:")
    if 'is_pentagon' in data.columns and data['is_pentagon'].any():
        pentagon_faces = data[data['is_pentagon'] == 1]
        regular_faces = data[data['is_pentagon'] == 0]
        
        print(f"   Pentagon faces: {len(pentagon_faces)}")
        print(f"   Regular faces: {len(regular_faces)}")
        
        # Compare fluxes
        for flux_type in ['deposition_flux', 'emission_flux', 'net_flux']:
            if flux_type in data.columns:
                p_mean = pentagon_faces[flux_type].mean()
                r_mean = regular_faces[flux_type].mean()
                ratio = p_mean / r_mean if r_mean != 0 else 0
                
                # Statistical test
                t_stat, p_value = stats.ttest_ind(pentagon_faces[flux_type], 
                                                  regular_faces[flux_type])
                
                print(f"\n   {flux_type}:")
                print(f"     Pentagon mean: {p_mean:.3e}")
                print(f"     Regular mean: {r_mean:.3e}")
                print(f"     Ratio: {ratio:.3f}")
                print(f"     T-test p-value: {p_value:.3e}")
                
                if p_value < 0.001 and abs(ratio - 1.0) > 0.2:
                    print(f"     ⚠ SIGNIFICANT PENTAGON ARTIFACT DETECTED!")
    
    # 4. Spatial Pattern Analysis
    print("\n4. Spatial Pattern Analysis:")
    
    # Check for regular patterns using FFT
    if 'deposition_flux' in data.columns:
        flux_values = data['deposition_flux'].values
        
        # Remove DC component and compute FFT
        flux_centered = flux_values - np.mean(flux_values)
        fft = np.fft.fft(flux_centered)
        power = np.abs(fft)**2
        
        # Find peaks in power spectrum
        power_sorted = np.sort(power)[::-1]
        peak_ratio = power_sorted[1] / power_sorted[10] if len(power_sorted) > 10 else 0
        
        print(f"   FFT Peak ratio: {peak_ratio:.2f}")
        if peak_ratio > 10:
            print(f"     ⚠ STRONG PERIODIC PATTERN DETECTED!")
        
        # Check for clustering
        if len(flux_values) > 0:
            # Simple clustering metric: ratio of variance to mean
            cv = np.std(flux_values) / np.mean(flux_values) if np.mean(flux_values) > 0 else 0
            print(f"   Coefficient of variation: {cv:.3f}")
            if cv > 1.0:
                print(f"     ⚠ HIGH SPATIAL VARIABILITY!")
    
    # 5. Create diagnostic plots
    print("\n5. Creating diagnostic plots...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Mass conservation histogram
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(data['column_sum'], bins=50, edgecolor='black')
    ax1.axvline(x=1.0, color='r', linestyle='--', label='Ideal (1.0)')
    ax1.set_xlabel('Column Sum')
    ax1.set_ylabel('Count')
    ax1.set_title('Mass Conservation Check')
    ax1.legend()
    
    # Plot 2: Pentagon vs Regular comparison
    ax2 = plt.subplot(3, 3, 2)
    if 'is_pentagon' in data.columns and 'deposition_flux' in data.columns:
        pentagon_flux = data[data['is_pentagon'] == 1]['deposition_flux']
        regular_flux = data[data['is_pentagon'] == 0]['deposition_flux']
        
        if len(pentagon_flux) > 0 and len(regular_flux) > 0:
            ax2.boxplot([regular_flux, pentagon_flux], labels=['Regular', 'Pentagon'])
            ax2.set_ylabel('Deposition Flux')
            ax2.set_title('Pentagon Effect')
            ax2.set_yscale('log')
    
    # Plot 3: Emission vs Deposition
    ax3 = plt.subplot(3, 3, 3)
    if 'emission_flux' in data.columns and 'deposition_flux' in data.columns:
        colors = data['is_pentagon'] if 'is_pentagon' in data.columns else 'blue'
        scatter = ax3.scatter(data['emission_flux'], data['deposition_flux'], 
                            c=colors, alpha=0.5, s=1, cmap='coolwarm')
        ax3.set_xlabel('Emission Flux')
        ax3.set_ylabel('Deposition Flux')
        ax3.set_title('Emission vs Deposition')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        if 'is_pentagon' in data.columns:
            plt.colorbar(scatter, ax=ax3, label='Is Pentagon')
    
    # Plot 4: Temperature distribution
    ax4 = plt.subplot(3, 3, 4)
    if 'temperature' in data.columns:
        ax4.hist(data['temperature'], bins=50, edgecolor='black')
        ax4.set_xlabel('Temperature (K)')
        ax4.set_ylabel('Count')
        ax4.set_title('Temperature Distribution')
    
    # Plot 5: Flux distributions
    ax5 = plt.subplot(3, 3, 5)
    if 'deposition_flux' in data.columns:
        ax5.hist(data['deposition_flux'][data['deposition_flux'] > 0], 
                bins=50, alpha=0.7, label='Deposition', density=True)
    if 'emission_flux' in data.columns:
        ax5.hist(data['emission_flux'][data['emission_flux'] > 0], 
                bins=50, alpha=0.7, label='Emission', density=True)
    ax5.set_xlabel('Flux (kg/m²/s)')
    ax5.set_ylabel('Probability Density')
    ax5.set_title('Flux Distributions')
    ax5.set_yscale('log')
    ax5.legend()
    
    # Plot 6: Escape fraction vs Temperature
    ax6 = plt.subplot(3, 3, 6)
    if 'temperature' in data.columns and 'escape_fraction' in data.columns:
        ax6.scatter(data['temperature'], data['escape_fraction'], alpha=0.5, s=1)
        ax6.set_xlabel('Temperature (K)')
        ax6.set_ylabel('Escape Fraction')
        ax6.set_title('Temperature vs Escape')
    
    # Plot 7: Net flux distribution
    ax7 = plt.subplot(3, 3, 7)
    if 'net_flux' in data.columns:
        positive_net = data[data['net_flux'] > 0]
        negative_net = data[data['net_flux'] < 0]
        
        ax7.hist(positive_net['net_flux'], bins=30, alpha=0.7, 
                label=f'Gain ({len(positive_net)} faces)', color='green')
        ax7.hist(negative_net['net_flux'], bins=30, alpha=0.7, 
                label=f'Loss ({len(negative_net)} faces)', color='red')
        ax7.set_xlabel('Net Flux (kg/m²/s)')
        ax7.set_ylabel('Count')
        ax7.set_title('Net Flux Distribution')
        ax7.legend()
    
    # Plot 8: Correlation matrix
    ax8 = plt.subplot(3, 3, 8)
    numeric_cols = ['temperature', 'column_sum', 'escape_fraction', 
                   'emission_flux', 'deposition_flux', 'net_flux']
    numeric_cols = [c for c in numeric_cols if c in data.columns]
    
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        im = ax8.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax8.set_xticks(range(len(numeric_cols)))
        ax8.set_yticks(range(len(numeric_cols)))
        ax8.set_xticklabels([c.replace('_', '\n') for c in numeric_cols], rotation=45, ha='right')
        ax8.set_yticklabels(numeric_cols)
        ax8.set_title('Correlation Matrix')
        plt.colorbar(im, ax=ax8)
    
    # Plot 9: Diagnostic summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "DIAGNOSTIC SUMMARY\n\n"
    
    # Check for issues
    issues = []
    
    if violations > len(data) * 0.01:
        issues.append("❌ Mass conservation violated")
    else:
        issues.append("✓ Mass conservation OK")
    
    if 'is_pentagon' in data.columns and data['is_pentagon'].any():
        pentagon_ratio = 0
        if 'deposition_flux' in data.columns:
            p_mean = data[data['is_pentagon'] == 1]['deposition_flux'].mean()
            r_mean = data[data['is_pentagon'] == 0]['deposition_flux'].mean()
            pentagon_ratio = p_mean / r_mean if r_mean != 0 else 0
        
        if abs(pentagon_ratio - 1.0) > 0.2:
            issues.append(f"❌ Pentagon artifact (ratio: {pentagon_ratio:.2f})")
        else:
            issues.append("✓ No pentagon artifact")
    
    if cv > 1.0:
        issues.append(f"❌ High variability (CV: {cv:.2f})")
    else:
        issues.append("✓ Variability acceptable")
    
    if peak_ratio > 10:
        issues.append(f"❌ Periodic pattern (peak ratio: {peak_ratio:.1f})")
    else:
        issues.append("✓ No strong periodicity")
    
    summary_text += "\n".join(issues)
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
            va='top', fontsize=12, family='monospace')
    
    plt.suptitle('Europa Exosphere Transport Diagnostics', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / "diagnostic_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved diagnostic plots to: {output_file}")
    
    plt.show()
    
    # 6. Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if violations > len(data) * 0.01:
        print("\n⚠ Mass Conservation Issues:")
        print("  - Check transport matrix normalization")
        print("  - Verify escape fraction calculation")
        print("  - Ensure proper boundary conditions")
    
    if 'is_pentagon' in data.columns and abs(pentagon_ratio - 1.0) > 0.2:
        print("\n⚠ Mesh Topology Artifacts:")
        print("  - Increase regularization distance")
        print("  - Consider mesh refinement")
        print("  - Implement adaptive smoothing")
    
    if cv > 1.0:
        print("\n⚠ High Spatial Variability:")
        print("  - Check for numerical instabilities")
        print("  - Verify temperature interpolation")
        print("  - Consider timestep reduction")
    
    if peak_ratio > 10:
        print("\n⚠ Periodic Patterns:")
        print("  - Review neighbor identification logic")
        print("  - Check for aliasing in transport calculation")
        print("  - Verify mesh connectivity")
    
    print("\n" + "="*60)
    
    return data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "./exosphere_output"
    
    print(f"Analyzing diagnostics in: {output_dir}")
    data = analyze_transport_diagnostics(output_dir)