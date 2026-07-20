#!/usr/bin/env python3
"""Minimal demonstration of ice_depth_pipeline

Creates synthetic binary temperature files, runs the pipeline, and prints results.
This is a quick way to verify the installation and see the module in action.

Usage:
    python demo_pipeline.py
"""
import os
import sys
import tempfile
import shutil
import numpy as np

# Add parent dir to path if running directly
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(__file__))

from ptm_ice.ice_depth_pipeline import run_pipeline


def main():
    print("=" * 70)
    print("Ice Depth Pipeline Demo")
    print("=" * 70)
    
    # Configuration
    n_faces = 50
    n_timesteps = 10
    
    print(f"\n1. Creating synthetic temperature data...")
    print(f"   Faces: {n_faces}")
    print(f"   Timesteps: {n_timesteps}")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate synthetic binary files
        for t in range(n_timesteps):
            phase = 2 * np.pi * t / n_timesteps
            base_temp = 90.0 + 15.0 * np.sin(phase)
            
            # Spatial gradient + noise
            temps = np.linspace(base_temp + 5, base_temp - 5, n_faces)
            temps += np.random.uniform(-1, 1, n_faces)
            
            filename = os.path.join(temp_dir, f'temperature_{t:04d}.bin')
            temps.astype('float32').tofile(filename)
        
        print(f"   ✓ Created {n_timesteps} binary files in {temp_dir}")
        
        # Run pipeline
        print(f"\n2. Running ice depth pipeline...")
        out_dir = os.path.join(temp_dir, 'output')
        
        csv_path = run_pipeline(
            source_dir=temp_dir,
            fmt='binary',
            timesteps=n_timesteps,
            out_dir=out_dir,
            dtype='float32',
            backend='numpy',
            chunk_size=25,
            J0=1e-5,
        )
        
        print(f"   ✓ Pipeline complete")
        print(f"   ✓ Output: {csv_path}")
        
        # Read and display results
        print(f"\n3. Results preview:")
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Statistics
        depths = np.array([
            float(r['ice_depth']) if r['ice_depth'] != 'nan' else np.nan
            for r in rows
        ])
        T_mean = np.array([float(r['T_mean']) for r in rows])
        T_max = np.array([float(r['T_max']) for r in rows])
        
        valid_mask = ~np.isnan(depths)
        n_valid = np.sum(valid_mask)
        
        print(f"\n   Total faces:      {len(rows)}")
        print(f"   Valid solutions:  {n_valid} ({100*n_valid/len(rows):.1f}%)")
        print(f"   T_mean range:     [{T_mean.min():.2f}, {T_mean.max():.2f}] K")
        print(f"   T_max range:      [{T_max.min():.2f}, {T_max.max():.2f}] K")
        
        if n_valid > 0:
            print(f"   Ice depth range:  [{np.nanmin(depths):.6f}, {np.nanmax(depths):.6f}] m")
            print(f"   Mean depth:       {np.nanmean(depths):.6f} m")
        
        # Show sample rows
        print(f"\n   Sample entries:")
        print(f"   {'Face':<6} {'T_mean (K)':<12} {'T_max (K)':<12} {'Depth (m)':<12}")
        print(f"   {'-'*48}")
        for i in [0, n_faces//2, n_faces-1]:
            row = rows[i]
            depth_str = f"{float(row['ice_depth']):.8f}" if row['ice_depth'] != 'nan' else 'nan'
            print(f"   {row['face']:<6} {float(row['T_mean']):<12.2f} "
                  f"{float(row['T_max']):<12.2f} {depth_str:<12}")
        
        print(f"\n{'=' * 70}")
        print("Demo complete! ✓")
        print("=" * 70)
        print(f"\nFor production use, see:")
        print(f"  - ICE_DEPTH_README.md for full documentation")
        print(f"  - ice_depth_pipeline_demo.ipynb for interactive examples")
        print(f"  - Run 'python -m ptm_ice.ice_depth_pipeline --help' for CLI usage")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
