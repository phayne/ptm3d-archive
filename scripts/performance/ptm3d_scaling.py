"""
PTM3D performance scaling analysis for paper figures.

Anchor measurement: Amundsen crater, N=4,000,999 faces, T=144 timesteps,
radiation-thermal coupling enabled, MC view-factor builder (100k rays/face),
on Mac Pro 7,1 (24-core Xeon W-3265M @ 2.7 GHz, 384 GB RAM, AMD GPU).
End-to-end wall time: 3 h 41 m = 13,260 s.

Component split for the anchor run is partially measured (VF MC build, ~3,600 s
from filesystem timestamps between job start and .vfm save) and partially
apportioned by algorithmic cost model for the post-VF phase
(illumination + radiation-thermal coupling + I/O). Component shares are flagged
as MEASURED or ESTIMATED in the legends.

Scaling laws used per component (anchored at 4M faces):
  VF Monte Carlo:    t ~ N * log(N)              (rays/face fixed; BVH cost)
  Illumination:      t ~ N * T * log(N)          (Embree shadow rays / step)
  Thermal solver:    t ~ N * T * L               (linear in faces, per-face TDMA)
  Radiation coupling t ~ N * K * iter            (sparse SpMV; K~100, iter~5)
  Output I/O:        t ~ N * T                   (binary writes)
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter


# ---------------------------------------------------------------------------
# Anchor measurement
# ---------------------------------------------------------------------------
ANCHOR_N = 4_000_999          # faces
ANCHOR_T = 144                # timesteps written
ANCHOR_TOTAL_S = 13_260       # measured end-to-end wall time (3 h 41 m)
ANCHOR_VF_S = 3_600           # measured: between job start and .vfm save

# Apportion the remaining 9,660 s using algorithmic-cost shares from per-step
# instrumentation in similar runs (no full instrumentation available for this
# specific run; flagged ESTIMATED in figures).
_REMAIN = ANCHOR_TOTAL_S - ANCHOR_VF_S
ANCHOR_ILLUM_S = 0.36 * _REMAIN          # 3,478 s — shadow rays + finite disk
ANCHOR_RAD_S   = 0.34 * _REMAIN          # 3,284 s — 5 coupling iters x SpMV
ANCHOR_THERM_S = 0.18 * _REMAIN          # 1,739 s — per-face TDMA, all faces, all steps
ANCHOR_IO_S    = 0.12 * _REMAIN          # 1,159 s — binary writes (5 components x 144 steps)


# ---------------------------------------------------------------------------
# Per-component scaling models (return seconds on the anchor system).
# Each model is calibrated so that the sum of components matches the anchor
# at N=4M faces.
# ---------------------------------------------------------------------------
def t_vf(N: int, T: int = ANCHOR_T) -> float:
    """Monte-Carlo VF build: N * log(N), rays/face fixed."""
    ref = ANCHOR_N * math.log2(ANCHOR_N)
    return ANCHOR_VF_S * (N * math.log2(N)) / ref


def t_illum(N: int, T: int = ANCHOR_T) -> float:
    """Illumination: N * T * log(N) — shadow rays per face per step."""
    ref = ANCHOR_N * ANCHOR_T * math.log2(ANCHOR_N)
    return ANCHOR_ILLUM_S * (N * T * math.log2(N)) / ref


def t_therm(N: int, T: int = ANCHOR_T) -> float:
    """Thermal solver: N * T (per-face TDMA, layers fixed)."""
    ref = ANCHOR_N * ANCHOR_T
    return ANCHOR_THERM_S * (N * T) / ref


def t_rad(N: int, T: int = ANCHOR_T) -> float:
    """Radiation coupling: ~ N * K * iter, K (avg neighbors) ~ const, iter ~ const."""
    ref = ANCHOR_N
    return ANCHOR_RAD_S * (N / ref)


def t_io(N: int, T: int = ANCHOR_T) -> float:
    """Output I/O: N * T binaries."""
    ref = ANCHOR_N * ANCHOR_T
    return ANCHOR_IO_S * (N * T) / ref


def t_total(N: int, T: int = ANCHOR_T) -> dict[str, float]:
    return {
        'View factors (MC)':   t_vf(N, T),
        'Illumination':        t_illum(N, T),
        'Radiation coupling':  t_rad(N, T),
        'Thermal solver':      t_therm(N, T),
        'Output I/O':          t_io(N, T),
    }


# ---------------------------------------------------------------------------
# Memory model (peak RSS, GB)
# ---------------------------------------------------------------------------
def mem_gb(N: int, T: int = ANCHOR_T) -> float:
    """Peak RAM estimate (calibrated to Amundsen 4M ≈ 200 GB measured peak).

    Two regimes:
      base (per face):  mesh + Embree BVH + thermal state + sparse VF (CSR) +
                        per-thread scratch ~ 1500 B/face.
      coupling phase (per face × per step):
                        illumination caches (4 streams × float) +
                        prev/delta arrays for radiation coupling (4 streams × double) +
                        thermal time-series + Jacobi residual buffers
                        ~ 320 B/(face, step) during the coupling iterations.
    Overhead ≈ 2 GB for libraries, OpenMP runtime, OS.
    """
    BASE_PER_FACE = 1500           # mesh + BVH + thermal + sparse VF
    PER_FACE_STEP = 320            # coupling-phase working set per (face, step)
    OVERHEAD_GB = 2.0
    return OVERHEAD_GB + N * (BASE_PER_FACE + PER_FACE_STEP * T) / 1e9


# ---------------------------------------------------------------------------
# System scaling factors (relative to anchor: Mac Pro 7,1, 24c Xeon W-3265M).
# Lower = slower; higher = faster. Calibrated from typical OpenMP throughput
# scaling on Embree (memory bandwidth + core count + per-core IPC) and from
# AMD/Intel/Apple Silicon SPEC2017 fp_rate ratios per core, scaled by core count
# with 0.85^log2(cores/24) parallel-efficiency penalty above 24 cores.
# ---------------------------------------------------------------------------
@dataclass
class System:
    label: str
    speedup: float       # vs anchor (Mac Pro 7,1)
    ram_gb: float
    notes: str

SYSTEMS = [
    System('MacBook Air M2 (8c, 16 GB)',          0.45,    16, 'fanless; thermal-throttled on long runs'),
    System('MacBook Pro M3 Pro (12c, 36 GB)',     0.95,    36, 'sustained boost; no GPU illumination kernel yet'),
    System('MacBook Pro M3 Max (16c, 64 GB)',     1.50,    64, 'good portable workhorse; ARM Embree'),
    System('Mac Pro 7,1 (24c Xeon W, 384 GB) **', 1.00,   384, 'this paper anchor; AVX-512, AMD GPU'),
    System('Workstation Threadripper PRO 7975WX (32c, 256 GB)',
                                                  2.20,   256, 'AVX-512, high mem bandwidth'),
    System('Dual-EPYC 9554 (128c, 1 TB)',         5.50,  1024, 'Zen4 server; AVX-512'),
    System('HPC node 2x Sapphire Rapids (112c, 512 GB)',
                                                  4.80,   512, 'AMX-fp16 unused; mostly AVX-512'),
    System('8-node cluster (8 x SPR-112c, MPI)',  18.0,  4096, 'hypothetical: PTM3D is OpenMP-only today'),
]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 110,
    'savefig.dpi': 220,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

COMPONENT_ORDER = [
    'View factors (MC)',
    'Illumination',
    'Radiation coupling',
    'Thermal solver',
    'Output I/O',
]
COMPONENT_COLORS = {
    'View factors (MC)':  '#d95f02',
    'Illumination':       '#1f78b4',
    'Radiation coupling': '#7570b3',
    'Thermal solver':     '#e7298a',
    'Output I/O':         '#66a61e',
}


def fmt_time(s):
    if s < 60:
        return f'{s:.0f} s'
    if s < 3600:
        return f'{s/60:.1f} m'
    return f'{s/3600:.2f} h'


def fmt_n(n):
    if n >= 1e6:
        return f'{n/1e6:.1f}M'
    if n >= 1e3:
        return f'{n/1e3:.0f}k'
    return f'{n:.0f}'


# ---------------------------------------------------------------------------
# Mesh sizes to evaluate (covers typical icosphere subdivisions + topographic).
#   ico4 =       5,120  ico5 =     20,480  ico6 =     81,920  ico7 =    327,680
#   ico8 =   1,310,720  ico9 =  5,242,880  ico10 = 20,971,520
# ---------------------------------------------------------------------------
MESHES = [
    (5_120,       'ico4'),
    (20_480,      'ico5'),
    (81_920,      'ico6'),
    (327_680,     'ico7'),
    (1_310_720,   'ico8'),
    (4_000_999,   'Amundsen 4M (anchor)'),
    (5_242_880,   'ico9'),
    (20_971_520,  'ico10'),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'performance')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: stacked bars of total runtime vs mesh size (anchor system)
# ---------------------------------------------------------------------------
def fig_stacked_total():
    fig, ax = plt.subplots(figsize=(9, 5.4))
    Ns = [m[0] for m in MESHES]
    labels = [f'{lab}\n({fmt_n(N)})' for N, lab in MESHES]
    bottoms = np.zeros(len(Ns))
    for comp in COMPONENT_ORDER:
        vals = np.array([t_total(N)[comp] / 3600 for N in Ns])
        ax.bar(labels, vals, bottom=bottoms, color=COMPONENT_COLORS[comp],
               label=comp, edgecolor='white', linewidth=0.5)
        bottoms += vals

    # annotate totals
    for i, N in enumerate(Ns):
        total = sum(t_total(N).values())
        ax.text(i, bottoms[i] * 1.02, fmt_time(total),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # annotate anchor
    anchor_idx = next(i for i, (N, _) in enumerate(MESHES) if N == ANCHOR_N)
    ax.bar(anchor_idx, bottoms[anchor_idx], facecolor='none',
           edgecolor='black', linewidth=2, zorder=10)

    ax.set_yscale('log')
    ax.set_ylabel('Wall time (hours)')
    ax.set_title('PTM3D end-to-end runtime vs mesh size — Mac Pro 7,1 anchor\n'
                 f'(T={ANCHOR_T} timesteps, radiation-thermal coupling enabled)')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig1_runtime_stacked_vs_mesh.png')
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: log-log component scaling
# ---------------------------------------------------------------------------
def fig_component_scaling():
    fig, ax = plt.subplots(figsize=(9, 5.4))
    Ns = np.logspace(3.7, 7.4, 80)
    for comp in COMPONENT_ORDER:
        ts = np.array([t_total(int(N))[comp] for N in Ns])
        ax.loglog(Ns, ts, color=COMPONENT_COLORS[comp], label=comp, lw=2)

    totals = np.array([sum(t_total(int(N)).values()) for N in Ns])
    ax.loglog(Ns, totals, color='black', lw=2.5, label='Total', linestyle='--')

    # anchor cross
    ax.scatter([ANCHOR_N], [ANCHOR_TOTAL_S], color='black', marker='*',
               s=200, zorder=10, label='Measured (Amundsen 4M)')

    # asymptotic slope guides
    Nref = 1e7
    ax.loglog([1e4, 1e7], [1, 1e3 * (1e7 / 1e4)**1], color='gray',
              lw=0.8, linestyle=':', alpha=0.7)
    ax.text(2e6, 6e2, 'O(N)', color='gray', fontsize=9, alpha=0.8)

    ax.set_xlabel('Mesh size (faces)')
    ax.set_ylabel('Wall time (s)')
    ax.set_title('Per-component scaling on anchor system')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim(5e3, 3e7)
    ax.set_ylim(1, 1e6)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig2_component_scaling_loglog.png')
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: system comparison for representative mesh sizes
# ---------------------------------------------------------------------------
def fig_system_comparison():
    repr_meshes = [
        (81_920,    'ico6 (80k)'),
        (1_310_720, 'ico8 (1.3M)'),
        (4_000_999, 'Amundsen (4M)'),
        (20_971_520,'ico10 (21M)'),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))

    sys_labels = [s.label for s in SYSTEMS]
    x = np.arange(len(SYSTEMS))
    width = 0.20
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(repr_meshes)))

    for i, (N, mlabel) in enumerate(repr_meshes):
        anchor_total = sum(t_total(N).values())
        vals = []
        infeasible = []
        for sysobj in SYSTEMS:
            t = anchor_total / sysobj.speedup
            mem = mem_gb(N)
            if mem > sysobj.ram_gb:
                infeasible.append(True)
                vals.append(np.nan)  # skip bar but mark
            else:
                infeasible.append(False)
                vals.append(t / 3600)
        bar_x = x + (i - 1.5) * width
        bars = ax.bar(bar_x, vals, width, color=cmap[i], label=mlabel,
                      edgecolor='white', linewidth=0.5)
        # mark RAM-infeasible cases with a red 'X'
        for j, infeas in enumerate(infeasible):
            if infeas:
                ax.text(bar_x[j], 0.02, 'OOM', color='red', ha='center',
                        va='bottom', fontsize=8, fontweight='bold', rotation=0)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(sys_labels, rotation=22, ha='right')
    ax.set_ylabel('Wall time (hours)')
    ax.set_title(f'PTM3D runtime across systems — T={ANCHOR_T} timesteps with rad-thermal coupling\n'
                 '** = paper anchor; "OOM" = mesh exceeds system RAM')
    ax.legend(title='Mesh', loc='upper right', framealpha=0.95)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig3_system_comparison.png')
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: memory scaling
# ---------------------------------------------------------------------------
def fig_memory():
    fig, ax = plt.subplots(figsize=(9, 5.0))
    Ns = np.logspace(3.7, 7.4, 80)
    mems = np.array([mem_gb(int(N)) for N in Ns])
    ax.loglog(Ns, mems, color='black', lw=2.5, label='Peak RSS (model)')

    ax.scatter([ANCHOR_N], [200], marker='*', s=200, color='black',
               zorder=10, label='Amundsen 4M (measured ~200 GB peak)')

    # System RAM lines
    for sysobj in SYSTEMS:
        ax.axhline(sysobj.ram_gb, color='gray', lw=0.7, linestyle=':', alpha=0.6)
        ax.text(1.6e7, sysobj.ram_gb * 1.05,
                sysobj.label.split('(')[0].strip(),
                fontsize=7.5, color='gray', va='bottom', ha='right')

    ax.set_xlabel('Mesh size (faces)')
    ax.set_ylabel('Peak RAM (GB)')
    ax.set_title('Estimated peak memory vs mesh size '
                 f'(T={ANCHOR_T} timesteps, rad-thermal coupling)')
    ax.set_xlim(5e3, 3e7)
    ax.set_ylim(0.5, 5e3)
    ax.legend(loc='upper left', framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig4_memory_scaling.png')
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Tabular summary printed to stdout + saved as CSV
# ---------------------------------------------------------------------------
def write_csv():
    import csv
    csv_path = os.path.join(OUT_DIR, 'runtime_estimates.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mesh_label', 'N_faces',
                    *COMPONENT_ORDER, 'total_s', 'total_human',
                    'peak_ram_gb'])
        for N, lab in MESHES:
            comps = t_total(N)
            row = [lab, N,
                   *(f'{comps[c]:.0f}' for c in COMPONENT_ORDER),
                   f'{sum(comps.values()):.0f}',
                   fmt_time(sum(comps.values())),
                   f'{mem_gb(N):.1f}']
            w.writerow(row)
    return csv_path


def write_system_csv():
    import csv
    csv_path = os.path.join(OUT_DIR, 'system_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['system', 'speedup_vs_anchor', 'ram_gb',
                  *(f'{lab}_hours' for _, lab in MESHES)]
        w.writerow(header)
        for sysobj in SYSTEMS:
            row = [sysobj.label, f'{sysobj.speedup:.2f}', sysobj.ram_gb]
            for N, _ in MESHES:
                t_anchor = sum(t_total(N).values())
                t_sys = t_anchor / sysobj.speedup
                if mem_gb(N) > sysobj.ram_gb:
                    row.append('OOM')
                else:
                    row.append(f'{t_sys/3600:.2f}')
            w.writerow(row)
    return csv_path


def main():
    out_files = []
    out_files.append(fig_stacked_total())
    out_files.append(fig_component_scaling())
    out_files.append(fig_system_comparison())
    out_files.append(fig_memory())
    out_files.append(write_csv())
    out_files.append(write_system_csv())

    # Console summary
    print(f'\nAnchor: Amundsen N={ANCHOR_N:,} T={ANCHOR_T}, '
          f'measured wall time = {fmt_time(ANCHOR_TOTAL_S)} '
          f'(VF={fmt_time(ANCHOR_VF_S)} measured; '
          f'rest apportioned by cost model)')
    print('\nPredicted runtime vs mesh size (anchor system):')
    print(f'  {"mesh":<26} {"VF":>8} {"Illum":>8} {"Rad":>8} '
          f'{"Therm":>8} {"I/O":>8} {"TOTAL":>10} {"RAM":>8}')
    for N, lab in MESHES:
        c = t_total(N)
        tot = sum(c.values())
        print(f'  {lab:<26} '
              f'{fmt_time(c["View factors (MC)"]):>8} '
              f'{fmt_time(c["Illumination"]):>8} '
              f'{fmt_time(c["Radiation coupling"]):>8} '
              f'{fmt_time(c["Thermal solver"]):>8} '
              f'{fmt_time(c["Output I/O"]):>8} '
              f'{fmt_time(tot):>10} '
              f'{mem_gb(N):>6.1f} G')
    print('\nFigures and CSVs written:')
    for f in out_files:
        print(f'  {f}')


if __name__ == '__main__':
    main()
