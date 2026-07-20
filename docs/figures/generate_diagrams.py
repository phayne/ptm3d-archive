#!/usr/bin/env python3
"""
Generate PTM3D documentation diagrams using matplotlib.
Run from the docs/figures/ directory.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrow, Circle, Rectangle
import numpy as np

# Color scheme
COLORS = {
    'input': '#E3F2FD',      # Light blue
    'process': '#FFF3E0',    # Light orange
    'output': '#E8F5E9',     # Light green
    'core': '#BBDEFB',       # Blue
    'optional': '#C8E6C9',   # Green
    'research': '#F8BBD9',   # Pink
    'external': '#E1BEE7',   # Purple
    'gpu': '#B2EBF2',        # Cyan
    'border': '#1565C0',     # Dark blue
}


def draw_rounded_box(ax, x, y, width, height, text, color, fontsize=9,
                     text_color='black', bold=True, alpha=1.0):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor=COLORS['border'],
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color,
            linespacing=1.3)


def draw_arrow(ax, start, end, color='#1565C0', style='->', lw=1.5,
               connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle=connectionstyle))


def generate_workflow_diagram():
    """Generate the PTM3D simulation workflow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 15.5, 'PTM3D Simulation Workflow', ha='center', va='center',
            fontsize=16, fontweight='bold', color=COLORS['border'])

    # === INPUTS SECTION ===
    # Input boxes (top row)
    draw_rounded_box(ax, 0.5, 13.5, 2.5, 1.2, 'Mesh File\n(PLY/OBJ/VTK)', COLORS['input'])
    draw_rounded_box(ax, 3.75, 13.5, 2.5, 1.2, 'SPICE Kernels\n(Ephemeris)', COLORS['input'])
    draw_rounded_box(ax, 7, 13.5, 2.5, 1.2, 'Config File\n(YAML)', COLORS['input'])

    # Arrows from inputs to mesh generation
    draw_arrow(ax, (1.75, 13.5), (4, 12.4))
    draw_arrow(ax, (5, 13.5), (5, 12.4))
    draw_arrow(ax, (8.25, 13.5), (6, 12.4))

    # === PROCESSING SECTION ===
    # Mesh Generation
    draw_rounded_box(ax, 2.5, 11.2, 5, 1.2,
                     'Mesh Generation\n• Icosphere creation  • Normal vectors  • Face areas',
                     COLORS['process'])
    draw_arrow(ax, (5, 11.2), (5, 10.1))

    # Illumination Model
    draw_rounded_box(ax, 2.5, 8.9, 5, 1.2,
                     'Illumination Model\n• Ray tracing (Embree)  • Solar flux  • Primary reflection',
                     COLORS['process'])
    draw_arrow(ax, (5, 8.9), (5, 7.8))

    # Thermal Model
    draw_rounded_box(ax, 2.5, 6.6, 5, 1.2,
                     'Thermal Model\n• FFT equilibration  • Crank-Nicolson  • T-dependent k, cp',
                     COLORS['process'])

    # Branch point - Radiation Exchange decision
    # Diamond for decision
    diamond_x, diamond_y = 5, 5.6
    diamond = plt.Polygon([(diamond_x, diamond_y+0.4), (diamond_x+0.5, diamond_y),
                           (diamond_x, diamond_y-0.4), (diamond_x-0.5, diamond_y)],
                          facecolor='white', edgecolor=COLORS['border'], linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, 'Rad\nExch?', ha='center', va='center', fontsize=7)

    draw_arrow(ax, (5, 6.6), (5, 6.0))

    # Left path - No radiation exchange
    draw_arrow(ax, (4.5, 5.6), (2.5, 5.6), style='->')
    ax.text(3.5, 5.85, 'No', fontsize=8, ha='center')
    draw_arrow(ax, (2.5, 5.6), (2.5, 3.9))

    # Right path - Radiation exchange
    draw_arrow(ax, (5.5, 5.6), (7.5, 5.6), style='->')
    ax.text(6.5, 5.85, 'Yes', fontsize=8, ha='center')

    # Radiation Exchange box (right side)
    draw_rounded_box(ax, 6.5, 4.2, 3, 1.2,
                     'Radiation Exchange\n• View factors\n• H-matrix\n• Radiosity',
                     COLORS['optional'])

    # Coupling loop
    draw_rounded_box(ax, 6.5, 2.7, 3, 1.0,
                     'Coupling Loop\nT→B→Q→T',
                     COLORS['optional'])
    draw_arrow(ax, (8, 4.2), (8, 3.7))

    # Loop back arrow
    ax.annotate('', xy=(7.5, 7.8), xytext=(9.5, 3.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['border'], lw=1.5,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(9.3, 5.5, 'iterate', fontsize=7, rotation=90, va='center')

    # Converged arrow down
    draw_arrow(ax, (6.5, 3.2), (5.5, 3.2))
    ax.text(6, 3.4, 'converged', fontsize=7, ha='center')

    # Ice Stability
    draw_rounded_box(ax, 2.5, 2.2, 5, 1.2,
                     'Ice Stability Depth\n• Vapor diffusion  • Newton-Raphson solver',
                     COLORS['process'])
    draw_arrow(ax, (2.5, 2.8), (2.5, 2.8))  # Connection from left path
    draw_arrow(ax, (5, 2.2), (5, 1.1))

    # === OUTPUTS SECTION ===
    draw_rounded_box(ax, 0.5, 0.0, 2.5, 1.0, 'Binary Files\ntemperature.bin\nillumination.bin', COLORS['output'])
    draw_rounded_box(ax, 3.75, 0.0, 2.5, 1.0, 'VTK/PVD\nParaView\nAnimation', COLORS['output'])
    draw_rounded_box(ax, 7, 0.0, 2.5, 1.0, 'CSV Series\nLocations\nTime series', COLORS['output'])

    # Arrows to outputs
    draw_arrow(ax, (4, 1.1), (1.75, 1.0))
    draw_arrow(ax, (5, 1.1), (5, 1.0))
    draw_arrow(ax, (6, 1.1), (8.25, 1.0))

    # Legend
    legend_y = 15.0
    draw_rounded_box(ax, 0.3, legend_y, 0.4, 0.3, '', COLORS['input'])
    ax.text(0.9, legend_y+0.15, 'Input', fontsize=8, va='center')
    draw_rounded_box(ax, 2.0, legend_y, 0.4, 0.3, '', COLORS['process'])
    ax.text(2.6, legend_y+0.15, 'Processing', fontsize=8, va='center')
    draw_rounded_box(ax, 4.2, legend_y, 0.4, 0.3, '', COLORS['optional'])
    ax.text(4.8, legend_y+0.15, 'Optional', fontsize=8, va='center')
    draw_rounded_box(ax, 6.2, legend_y, 0.4, 0.3, '', COLORS['output'])
    ax.text(6.8, legend_y+0.15, 'Output', fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig('ptm3d_workflow.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('ptm3d_workflow.svg', bbox_inches='tight', facecolor='white')
    print("Saved: ptm3d_workflow.png and ptm3d_workflow.svg")
    plt.close()


def generate_architecture_diagram():
    """Generate the PTM3D module architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'PTM3D Module Architecture', ha='center', va='center',
            fontsize=16, fontweight='bold', color=COLORS['border'])

    # === EXTERNAL DEPENDENCIES (Top) ===
    ax.text(7, 8.8, 'External Dependencies', ha='center', fontsize=10,
            fontstyle='italic', color='gray')

    ext_y = 8.0
    ext_boxes = [
        (0.5, 'Embree 4\nRay Tracing'),
        (2.5, 'CSPICE\nEphemeris'),
        (4.5, 'CFITSIO\nFITS I/O'),
        (6.5, 'Eigen3\nLinear Algebra'),
        (8.5, 'OpenMP\nParallelization'),
        (10.5, 'yaml-cpp\n(Optional)'),
        (12.0, 'VTK\n(Optional)'),
    ]
    for x, text in ext_boxes:
        draw_rounded_box(ax, x, ext_y, 1.8, 0.8, text, COLORS['external'], fontsize=7)

    # === MAIN EXECUTABLE (Center) ===
    draw_rounded_box(ax, 4.5, 5.8, 5, 1.2,
                     'ptm_thermal_sim\nMain Simulation Driver (~7000 lines)',
                     '#90CAF9', fontsize=10)

    # === CORE MODULES (Middle Layer) ===
    ax.text(2.5, 5.0, 'Core Modules', ha='center', fontsize=10,
            fontstyle='italic', color='gray')

    core_y = 3.8
    # Row 1
    draw_rounded_box(ax, 0.3, core_y, 2.2, 1.0,
                     'thermal\n1D Heat Conduction\nCrank-Nicolson', COLORS['core'], fontsize=8)
    draw_rounded_box(ax, 2.7, core_y, 2.2, 1.0,
                     'illumination\nRay-traced Solar\nSPICE Integration', COLORS['core'], fontsize=8)
    draw_rounded_box(ax, 5.1, core_y, 2.2, 1.0,
                     'radiation\nView Factors\nH-matrix Radiosity', COLORS['core'], fontsize=8)
    draw_rounded_box(ax, 7.5, core_y, 2.2, 1.0,
                     'mesh\nMesh I/O\nGeometry Utils', COLORS['core'], fontsize=8)
    draw_rounded_box(ax, 9.9, core_y, 2.0, 1.0,
                     'ice\nIce Stability\nDepth Solver', COLORS['core'], fontsize=8)
    draw_rounded_box(ax, 12.1, core_y, 1.6, 1.0,
                     'maps\nAlbedo\nFITS', COLORS['core'], fontsize=8)

    # Arrows from main to core modules
    for x in [1.4, 3.8, 6.2, 8.6, 10.9, 12.9]:
        draw_arrow(ax, (7, 5.8), (x, 4.8), lw=1)

    # === GPU MODULE (Side) ===
    ax.text(12.5, 5.0, 'GPU (Optional)', ha='center', fontsize=10,
            fontstyle='italic', color='gray')
    draw_rounded_box(ax, 10.5, 5.8, 3.2, 1.2,
                     'gpu\nOpenCL | HIP | CUDA\nThermal & Radiosity Kernels',
                     COLORS['gpu'], fontsize=8)
    draw_arrow(ax, (9.5, 6.4), (10.5, 6.4), style='<->')

    # === RESEARCH MODULES (Bottom) ===
    ax.text(3.5, 2.5, 'Research Modules', ha='center', fontsize=10,
            fontstyle='italic', color='gray')

    research_y = 1.5
    draw_rounded_box(ax, 0.5, research_y, 2.8, 0.9,
                     'exosphere\nWater Vapor | Sputtering\nIce Layer Evolution',
                     COLORS['research'], fontsize=8)
    draw_rounded_box(ax, 3.5, research_y, 2.8, 0.9,
                     'viewshed\nVisibility Analysis\nMission Planning',
                     COLORS['research'], fontsize=8)

    # Dashed lines to research modules (indicating optional/separate)
    ax.plot([1.9, 1.9], [2.4, 3.8], 'k--', lw=1, alpha=0.5)
    ax.plot([4.9, 4.9], [2.4, 3.8], 'k--', lw=1, alpha=0.5)

    # === UTILITIES (Bottom Right) ===
    ax.text(10.5, 2.5, 'Utilities', ha='center', fontsize=10,
            fontstyle='italic', color='gray')

    draw_rounded_box(ax, 7.0, research_y, 2.2, 0.9,
                     'tools\nConfig Parser\nYAML Support',
                     '#FFF9C4', fontsize=8)
    draw_rounded_box(ax, 9.4, research_y, 2.2, 0.9,
                     'fileio\nMemory-mapped\nI/O Utilities',
                     '#FFF9C4', fontsize=8)
    draw_rounded_box(ax, 11.8, research_y, 1.9, 0.9,
                     'constants\nPhysics\nNumerics',
                     '#FFF9C4', fontsize=8)

    # === PYTHON TOOLS (Very Bottom) ===
    ax.text(7, 0.6, 'Python Tools', ha='center', fontsize=10,
            fontstyle='italic', color='gray')

    py_y = 0.0
    draw_rounded_box(ax, 2.5, py_y, 2.5, 0.5, 'vtk_converter.py', '#DCEDC8', fontsize=8)
    draw_rounded_box(ax, 5.2, py_y, 2.5, 0.5, 'netcdf_converter.py', '#DCEDC8', fontsize=8)
    draw_rounded_box(ax, 7.9, py_y, 2.8, 0.5, 'plot_depth_profiles.py', '#DCEDC8', fontsize=8)

    # === LEGEND ===
    legend_x, legend_y = 0.3, 6.8
    ax.text(legend_x, legend_y + 0.5, 'Legend:', fontsize=9, fontweight='bold')

    items = [
        (COLORS['core'], 'Core Module'),
        (COLORS['gpu'], 'GPU Module'),
        (COLORS['research'], 'Research Module'),
        (COLORS['external'], 'External Dependency'),
    ]
    for i, (color, label) in enumerate(items):
        draw_rounded_box(ax, legend_x, legend_y - i*0.45, 0.35, 0.3, '', color)
        ax.text(legend_x + 0.5, legend_y - i*0.45 + 0.15, label, fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig('architecture.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('architecture.svg', bbox_inches='tight', facecolor='white')
    print("Saved: architecture.png and architecture.svg")
    plt.close()


def generate_thermal_profile_diagram():
    """Generate example thermal profile diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Simulated thermal profiles at different local times
    depths = np.logspace(-3, 0, 50)  # 1mm to 1m

    # Surface temperatures at different local times
    T_surface = {'Noon': 390, '6h': 200, 'Midnight': 100, '18h': 200}
    T_deep = 250  # Deep equilibrium temperature

    skin_depth = 0.05  # 5cm thermal skin depth

    for label, T_surf in T_surface.items():
        # Simple exponential decay model
        T = T_deep + (T_surf - T_deep) * np.exp(-depths / skin_depth)
        ax.semilogx(depths * 100, T, lw=2, label=f'Local Time: {label}')

    ax.set_xlabel('Depth (cm)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('Subsurface Temperature Profiles\n(Example: Lunar Equator)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.1, 100])
    ax.set_ylim([80, 420])

    # Add skin depth annotation
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    ax.text(5.5, 350, 'Thermal\nSkin Depth', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig('thermal_profile.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('thermal_profile.svg', bbox_inches='tight', facecolor='white')
    print("Saved: thermal_profile.png and thermal_profile.svg")
    plt.close()


if __name__ == '__main__':
    print("Generating PTM3D documentation diagrams...")
    generate_workflow_diagram()
    generate_architecture_diagram()
    generate_thermal_profile_diagram()
    print("\nDone! Generated files:")
    print("  - ptm3d_workflow.png/svg")
    print("  - architecture.png/svg")
    print("  - thermal_profile.png/svg")
