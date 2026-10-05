#!/usr/bin/env python
"""Generate docs/ice-shell-presentation.pdf — a one-slide-per-page 16:9 deck
summarizing the Europa ice-shell thickness & polar-wander study.

Usage
-----
    /Users/paha3326/anaconda3/bin/python scripts/make_ice_shell_presentation.py

Requirements
------------
- matplotlib + Pillow  (e.g. the base anaconda env, which has both).
- Optional: `pdftoppm` (poppler) to rasterize the vector figure PDFs at 300 dpi
  for maximum crispness. If absent, the script falls back to the existing PNGs.

Why a Python/matplotlib renderer instead of LaTeX/pandoc?  The local MiKTeX
install is broken (missing format files + a fatal "no admin checked for updates"
nag) and pandoc is not installed, so .tex/markdown -> PDF both fail here. See
the project memory note `pdf_doc_build_tooling.md`.

The companion markdown lives at docs/ice-shell-presentation.md.
"""
import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------- geometry / palette
W, H = 13.333, 7.5
BLUE = "#1E5A96"
DARK = "#222222"
GRAY = "#555555"
LIGHT = "#EBF0F8"
SLATE = "#5B6B7A"

# ---------------------------------------------------------------- paths
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../ptm
PLOTS = os.path.join(REPO, "plots", "ice-shell-dynamics")
FIGDIR = os.path.join(REPO, "docs", "figures", "ice_shell_dynamics")
OUT = os.path.join(REPO, "docs", "ice-shell-presentation.pdf")

# Figure sources, keyed by deck figure index. Paths are WITHOUT extension; the
# script prefers a 300-dpi raster of the .pdf and falls back to .png.
#   1 surface-T  2 thickness  3 SH-conv  4 all-rheo  5 darkening-mollweide
#   6 time-series  7 cycle-period  8 trajectories  9 timescales  10 model-table
FIG_SOURCES = {
    1: os.path.join(PLOTS, "albedo_map", "fig5_surface_temperature"),
    2: os.path.join(PLOTS, "albedo_map", "fig1_thickness_mollweide"),
    3: os.path.join(PLOTS, "smoothed", "fig7_sh_convergence"),
    4: os.path.join(PLOTS, "polar_wander", "fig5_all_rheologies"),
    5: os.path.join(PLOTS, "polar_wander", "fig4_mollweide"),
    6: os.path.join(PLOTS, "polar_wander", "fig2_time_series"),
    7: os.path.join(PLOTS, "polar_wander", "fig3_cycle_period"),
    8: os.path.join(FIGDIR, "comparison", "compare_trajectories"),
    9: os.path.join(FIGDIR, "comparison", "compare_timescales"),
    10: os.path.join(FIGDIR, "comparison", "compare_summary"),
}
RASTER_DPI = 300


def prepare_figures(build):
    """Populate `build` with fig01.png .. fig10.png and return {idx: path}.

    Rasterizes a figure's vector .pdf at RASTER_DPI when `pdftoppm` is present;
    otherwise copies the existing .png.
    """
    have_pdftoppm = shutil.which("pdftoppm") is not None
    if not have_pdftoppm:
        print("note: pdftoppm not found — using existing PNGs (lower resolution).")
    fig = {}
    for i, base in FIG_SOURCES.items():
        dst = os.path.join(build, f"fig{i:02d}.png")
        pdf, png = base + ".pdf", base + ".png"
        if have_pdftoppm and os.path.exists(pdf):
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(RASTER_DPI), "-singlefile", pdf, dst[:-4]],
                check=True,
            )
        elif os.path.exists(png):
            shutil.copyfile(png, dst)
        else:
            raise FileNotFoundError(f"no .pdf or .png source for figure {i}: {base}")
        fig[i] = dst
    return fig


# ---------------------------------------------------------------- primitives

def new_fig():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    return fig


def titlebar(fig, title, number=None):
    fig.patches.append(Rectangle((0, 0.875), 1, 0.125, transform=fig.transFigure,
                                 color=BLUE, zorder=0))
    fig.text(0.045, 0.937, title, color="white", fontsize=21, fontweight="bold",
             va="center", ha="left")
    if number:
        fig.text(0.965, 0.937, number, color="white", fontsize=15, va="center",
                 ha="right", alpha=0.85)


def footer(fig, idx, total, short):
    fig.patches.append(Rectangle((0.045, 0.062), 0.91, 0.0016, transform=fig.transFigure,
                                 color="#CCCCCC", zorder=0))
    fig.text(0.045, 0.032, short, color=GRAY, fontsize=8.5, va="center", ha="left", alpha=0.8)
    fig.text(0.955, 0.032, "%d / %d" % (idx, total), color=GRAY, fontsize=8.5,
             va="center", ha="right", alpha=0.8)


def place_image(fig, path, box):
    x0, y0, bw, bh = box
    img = mpimg.imread(path)
    ih, iw = img.shape[0], img.shape[1]
    img_ar = iw / ih
    box_ar = (bw * W) / (bh * H)
    if img_ar >= box_ar:
        aw = bw
        ah = (bw * W / img_ar) / H
    else:
        ah = bh
        aw = (bh * H * img_ar) / W
    cx = x0 + (bw - aw) / 2.0
    cy = y0 + (bh - ah) / 2.0
    ax = fig.add_axes([cx, cy, aw, ah])
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")


def place_images(fig, paths, box, gap=0.035):
    x0, y0, bw, bh = box
    n = len(paths)
    sub_h = (bh - gap * (n - 1)) / n
    for i, p in enumerate(paths):
        sub_y = y0 + bh - (i + 1) * sub_h - i * gap
        place_image(fig, p, (x0, sub_y, bw, sub_h))


def text_block(fig, items, box, fontsize=15, gap_frac=0.45, color=DARK):
    x0, y0, bw, bh = box
    ax = fig.add_axes(box)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    lh = fontsize * 1.3 / (bh * H * 72.0)
    y = 1.0
    for it in items:
        nl = it.count("\n") + 1
        ax.text(0.0, y, it, va="top", ha="left", fontsize=fontsize, color=color,
                linespacing=1.3, transform=ax.transAxes)
        y -= nl * lh + gap_frac * lh


def footnote(fig, lines, box=(0.045, 0.078, 0.91, 0.05), fontsize=9.5):
    text_block(fig, lines, box, fontsize=fontsize, gap_frac=0.35, color=SLATE)


def draw_table(fig, col_labels, rows, box, fontsize=12, colwidths=None):
    ax = fig.add_axes(box)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                   loc="center", colWidths=colwidths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.7)
        else:
            cell.set_facecolor(LIGHT if r % 2 == 1 else "white")
        if c == 0 and r != 0:
            cell.set_text_props(ha="left")
            cell._text.set_x(0.04)
            cell._text.set_horizontalalignment("left")


# ---------------------------------------------------------------- slide builders

def title_slide(pp):
    fig = new_fig()
    fig.patches.append(Rectangle((0, 0.62), 1, 0.38, transform=fig.transFigure, color=BLUE, zorder=0))
    fig.text(0.5, 0.84, "Europa's Ice Shell", color="white", fontsize=44,
             fontweight="bold", ha="center", va="center")
    fig.text(0.5, 0.73, "Thickness Structure  &  Polar-Wander Dynamics", color="white",
             fontsize=24, ha="center", va="center")
    fig.text(0.5, 0.45, "Coupling a 3-D thermal model (PTM3D) to the", fontsize=17,
             ha="center", va="center", color=DARK)
    fig.text(0.5, 0.395, "Ojakangas & Stevenson (1989) equilibrium ice-shell model",
             fontsize=17, ha="center", va="center", color=DARK)
    fig.text(0.5, 0.235, "Paul O. Hayne", fontsize=20, fontweight="bold",
             ha="center", va="center", color=BLUE)
    fig.text(0.5, 0.185, "CU Boulder / LASP", fontsize=15, ha="center", va="center", color=GRAY)
    pp.savefig(fig); plt.close(fig)


def divider(pp, part, subtitle, accent=BLUE, bg=LIGHT):
    fig = new_fig()
    fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=bg, zorder=0))
    fig.patches.append(Rectangle((0.0, 0.45), 1, 0.12, transform=fig.transFigure, color=accent, zorder=0))
    fig.text(0.5, 0.62, part, fontsize=40, fontweight="bold", ha="center", va="center", color=accent)
    fig.text(0.5, 0.51, subtitle, fontsize=24, ha="center", va="center", color="white")
    pp.savefig(fig); plt.close(fig)


def content(pp, idx, total, title, short, number, bullets=None, image=None,
            images=None, layout="text", table=None, bullets_fs=15, foot=None,
            foot_box=None, foot_fs=9.5):
    fig = new_fig()
    titlebar(fig, title, number)
    footer(fig, idx, total, short)

    if layout == "text":
        if bullets:
            text_block(fig, bullets, (0.05, 0.085, 0.90, 0.775), fontsize=bullets_fs)
    elif layout == "text_foot":
        if bullets:
            text_block(fig, bullets, (0.05, 0.175, 0.90, 0.685), fontsize=bullets_fs)
    elif layout == "right":
        text_block(fig, bullets or [], (0.05, 0.10, 0.43, 0.74), fontsize=bullets_fs)
        place_image(fig, image, (0.50, 0.09, 0.47, 0.74))
    elif layout == "below":
        if bullets:
            text_block(fig, bullets, (0.06, 0.60, 0.88, 0.23), fontsize=bullets_fs)
        place_image(fig, image, (0.06, 0.09, 0.88, 0.49))
    elif layout == "below_tall":   # more text room, smaller figure
        if bullets:
            text_block(fig, bullets, (0.05, 0.50, 0.91, 0.36), fontsize=bullets_fs)
        place_image(fig, image, (0.06, 0.075, 0.88, 0.40))
    elif layout == "two_below":
        if bullets:
            text_block(fig, bullets, (0.06, 0.62, 0.88, 0.21), fontsize=bullets_fs)
        place_images(fig, images, (0.10, 0.09, 0.80, 0.50))
    elif layout == "table_image":   # slide: thickness (big figure)
        if bullets:
            text_block(fig, bullets, (0.04, 0.77, 0.93, 0.09), fontsize=bullets_fs)
        draw_table(fig, table["cols"], table["rows"], table["box"],
                   fontsize=table.get("fs", 11), colwidths=table.get("cw"))
        place_image(fig, image, (0.47, 0.06, 0.51, 0.68))
    elif layout == "bullets_table":
        text_block(fig, bullets, (0.05, 0.56, 0.90, 0.27), fontsize=bullets_fs)
        draw_table(fig, table["cols"], table["rows"], table["box"],
                   fontsize=table.get("fs", 12), colwidths=table.get("cw"))
    if foot:
        if foot_box:
            footnote(fig, foot, box=foot_box, fontsize=foot_fs)
        else:
            footnote(fig, foot, fontsize=foot_fs)
    pp.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- deck
TOTAL = 21


def build_deck(pp, FIG):
    # 1 -----------------------------------------------------------------
    title_slide(pp)

    # 2  Motivation -----------------------------------------------------
    content(pp, 2, TOTAL, "1.  Motivation & Approach", "Motivation", "1",
            layout="text_foot", bullets_fs=13.5, bullets=[
        r"Question:  the 3-D thickness structure of Europa's ice shell — and is the resulting mass",
        r"distribution dynamically stable against (i) true polar wander (TPW) and (ii) non-synchronous",
        r"rotation (NSR, driven by tidal torques on shell-thickness/mass asymmetries)?",
        r"",
        r"Method:  PTM3D thermal model $\rightarrow$ surface-T field $T_s(\theta,\lambda)$  $\rightarrow$  O&S (1989) equilibrium shell",
        r"   model  $\rightarrow$  thickness $b(\theta,\lambda)$  $\rightarrow$  spherical harmonics  $\rightarrow$  moments of inertia / stability",
        r"   (PTM3D: ICO7 mesh, 327,680 faces, 288 timesteps / orbit, N–S symmetrized)",
        r"",
        r"Two new ingredients vs. classic 1-D O&S:",
        r"   •  Longitude-dependent surface temperature — from insolation differences due to",
        r"       eclipses by Jupiter + Jupiter-shine  (sub-Jovian / anti-Jovian asymmetry)",
        r"   •  Spatially variable albedo map  (leading / trailing asymmetry)",
    ], foot_box=(0.045, 0.085, 0.91, 0.135), foot=[
        r"Rheology cases:   $H$ = basal core heat flux (0 vs 10 mW m$^{-2}$)      ·      $n_G$ = Glen's-law creep exponent (3 vs 5.2)",
        r"",
        r"Symbols:   $C_R$ = tidal-strain correction (Ross & Schubert 1987)  ·  $e$ = orbital eccentricity (0.01)  ·  $a$ = Europa radius (1569 km)",
        r"                  $\rho$ = mean density (3040 kg m$^{-3}$)  ·  $g$ = surface gravity (1.31 m s$^{-2}$)",
    ])

    # 3  Part 1 divider -------------------------------------------------
    divider(pp, "Part 1", "Equilibrium Shell Thickness & Structure")

    # 4  Surface temperature (last bullet removed) ----------------------
    content(pp, 4, TOTAL, "2.  Surface temperature: 3-D vs. analytical", "Surface temperature", "2",
            layout="right", image=FIG[1], bullets_fs=14.5, bullets=[
        r"O&S analytical (a):",
        r"   smooth; equator 109 K, pole 52 K.",
        r"",
        r"PTM3D (b):",
        r"   equator 105–110 K, pole ~46 K,",
        r"   with real spatial structure.",
        r"",
        r"Difference (c):",
        r"   reveals hemispheric T asymmetries —",
        r"   sub-/anti-Jovian (~2–4 K, from",
        r"   eclipses + Jupiter-shine) and",
        r"   leading/trailing (from albedo) —",
        r"   that 1-D models cannot capture.",
    ])

    # 5  Equilibrium thickness (BIG sharp figure) -----------------------
    content(pp, 5, TOTAL, "3.  Equilibrium shell thickness — 4 rheologies", "Shell thickness", "3",
            layout="table_image", image=FIG[2], bullets_fs=13,
            bullets=[r"Mean thickness ~14 km (stiff) to ~27 km (soft). Bright equatorial band = tidal thickening ($t_{22}$).",
                     r"Flow-law cases match the paper within 2–4%; Maxwell cases run 14–24% thinner (structural offset)."],
            table={"cols": ["Case", "O&S", "PTM3D\n(alb-map)", "Paper"],
                   "box": (0.035, 0.30, 0.42, 0.28), "fs": 11.5,
                   "cw": [0.36, 0.20, 0.24, 0.20],
                   "rows": [["Maxwell, H=0", "19.7", "20.7", "23"],
                            ["Maxwell, H=10", "16.0", "16.8", "21"],
                            ["Flow law nG=3", "25.4", "26.6", "25"],
                            ["Flow law nG=5.2", "13.5", "14.1", "13"]]})

    # 6  Asymmetries (clarify c/s superscripts) -------------------------
    content(pp, 6, TOTAL, "4.  Hemispheric asymmetries (temperature & albedo)", "Asymmetries", "4",
            layout="bullets_table", bullets_fs=14.5, bullets=[
        r"The 3-D temperature field adds degree-1 (hemispheric) harmonics absent from the 1-D model:",
        r"   •  Longitude-dependent $T_s$  $\rightarrow$  cooler sub-Jovian hemisphere ~0.2–0.3 km thicker ($b_{11}^c$)",
        r"   •  Albedo map  $\rightarrow$  dominant non-tidal signal: brighter, colder leading hemisphere",
        r"       ~1–1.7 km thicker ($b_{11}^s$) — comparable to the tidal $b_{22}$ term",
        r"   •  Small $b_{22}^s$ ($-$0.07 to $-$0.13 km) rotates the tidal pattern a few degrees in longitude",
    ], table={"cols": ["Harmonic", "Physical meaning", "Maxwell H=0 (km)", "Range (km)"],
              "box": (0.10, 0.22, 0.80, 0.30), "fs": 12.5,
              "rows": [["b11^c", "sub-Jovian / anti-Jovian", "+0.36", "+0.25 to +0.46"],
                       ["b11^s", "leading / trailing", "-1.29", "-0.89 to -1.66"],
                       ["b22^c (=t22)", "tidal pattern", "+1.50", "+0.58 to +1.22"]],
              },
            foot=[r"Superscripts:  $b_{\ell m}^{c}$ and $b_{\ell m}^{s}$ are the $\cos(m\lambda)$ and $\sin(m\lambda)$ parts of the spherical harmonic of degree $\ell$, order $m$  (here $\ell$ = 1, 2)."])

    # 7  Moments of inertia & stability ---------------------------------
    content(pp, 7, TOTAL, "5.  Moments of inertia & stability", "Moments of inertia", "5",
            layout="bullets_table", bullets_fs=14.5, bullets=[
        r"$\Delta(B{-}A)/C \approx 10^{-7}(4t_{22})$        $\Delta(B{-}C)/A \approx 10^{-7}(t_{20}+2t_{22})$",
        r"   •  $\Delta(B{-}A)/C$ is robust (<1% albedo sensitivity) — set by the tidal dissipation pattern",
        r"   •  $\Delta(B{-}C)/A$ is albedo-controlled (30–60% change). For Maxwell $H$=0, variable albedo",
        r"       flips it from stable to marginally unstable  $\rightarrow$  motivates polar wander",
    ], table={"cols": ["Case", "Δ(B-A)/C\n(albedo-map)", "Δ(B-C)/A\nconst-albedo", "Δ(B-C)/A\nalbedo-map"],
              "box": (0.08, 0.13, 0.84, 0.32), "fs": 11.5,
              "rows": [["Maxwell, H=0", "5.99e-7", "-2.9e-8  (stable)", "+1.0e-8  (UNSTABLE)"],
                       ["Maxwell, H=10", "3.71e-7", "+2.9e-8", "+6.1e-8"],
                       ["Flow law nG=3", "4.89e-7", "+9.0e-8", "+1.4e-7"],
                       ["Flow law nG=5.2", "2.31e-7", "+6.6e-8", "+9.4e-8"]]})

    # 8  Part 2 divider -------------------------------------------------
    divider(pp, "Part 2", "Polar-Wander Dynamics")

    # 9  Polar Wander (retitled; B&H 2021 added) ------------------------
    content(pp, 9, TOTAL, "6.  Polar Wander", "Polar Wander", "6",
            layout="text", bullets_fs=15, bullets=[
        r"Magnetospheric plasma preferentially darkens the trailing hemisphere on $10^3$–$10^5$ yr",
        r"timescales (Paranicas et al. 2001; Burnett & Hayne 2021). This suggests a self-resetting cycle:",
        r"",
        r"   1.  Darkening  $\rightarrow$  lower albedo  $\rightarrow$  higher $T_s$  $\rightarrow$  thins trailing-hemisphere shell",
        r"   2.  Mass redistribution drives $\Delta(B{-}C)/A>0$  $\rightarrow$  instability",
        r"   3.  Shell reorients (TPW)  $\rightarrow$  exposes fresh bright ice on new trailing side",
        r"   4.  Albedo resets  $\rightarrow$  cycle restarts",
        r"",
        r"A stabilizer is required (3 of 4 rheologies are unstable even without darkening):",
        r"the fossil bulge — a permanent elastic $\ell$=2 deformation of the cold outer shell.",
    ])

    # 10  Instability + fossil bulge ------------------------------------
    content(pp, 10, TOTAL, "7.  Darkening drives instability; fossil sets baseline", "Instability", "7",
            layout="below", image=FIG[4], bullets_fs=13.5, bullets=[
        r"   •  Without fossil bulge (left): 3 of 4 rheologies have $\Delta(B{-}C)/A>0$ even at $f$=0",
        r"       $\rightarrow$ permanent instability (unphysical — Europa is stable today)",
        r"   •  With fossil bulge (right): all rheologies start at $\Delta(B{-}C)/A$=0, grow with darkening $f$",
        r"   •  For Maxwell $H$=0, instability sets in at a critical darkening  $f_{crit}\approx0.12$",
    ])

    # 11  Darkening cycle — WITHOUT fossil bulge ------------------------
    content(pp, 11, TOTAL, "8.  Shell thickness through a darkening cycle", "Darkening cycle", "8",
            layout="right", image=FIG[5], bullets_fs=15, bullets=[
        r"Quasi-static shell-only field",
        r"(NO fossil bulge) — the bare",
        r"destabilizing trend.",
        r"",
        r"Maxwell $H$=0, trailing hemisphere",
        r"darkens ($f$: 0 $\rightarrow$ 1):",
        r"",
        r"   •  Mean thickness drops",
        r"       20.3 km (bright) $\rightarrow$ 19.3 km (dark)",
        r"",
        r"   •  $\Delta(B{-}C)/A$:",
        r"       $-1.3\times10^{-8}$ (stable)  $\rightarrow$",
        r"       $+6.2\times10^{-8}$ (unstable)",
    ])

    # 12  Condensed A — cyclical & continuous TPW -----------------------
    content(pp, 12, TOTAL, "9.  Cyclical & continuous polar wander", "Cyclical TPW", "9",
            layout="right", image=FIG[6], bullets_fs=14.5, bullets=[
        r"Full 3×3 inertia-tensor model with",
        r"the fossil bulge as stabilizer",
        r"($\alpha$=1.5):",
        r"",
        r"   •  Cycle period ~1.5 Myr",
        r"       (Maxwell $H$=0, $\tau_{shell}$=$10^7$ yr)",
        r"",
        r"   •  Period scales linearly with",
        r"       $\tau_{shell}$: ~0.2–15 Myr over",
        r"       $10^6$–$10^8$ yr; insensitive to $\tau_{dark}$",
        r"",
        r"   •  Discrete $90^\circ$ flips become",
        r"       continuous rotation along the",
        r"       trailing–leading meridian once",
        r"       the fossil decays.",
    ])

    # 13  Condensed B — physical timescales + interior coupling ---------
    content(pp, 13, TOTAL, "10.  Physical timescales & interior coupling", "Timescales & coupling", "10",
            layout="below_tall", image=FIG[8], bullets_fs=13.5, bullets=[
        r"   •  Replace the free TPW timescale with the O&S overdamped (OSPW) model (viscous shell",
        r"       dissipation + iceberg/isostatic corrections): $\tau_f\sim0.64$ Myr; the free $\tau_{TPW}$=$10^5$ yr was ~100× too fast.",
        r"   •  Burnett & Hayne (2023) gravity-gradient shell–interior coupling: a near-hydrostatic interior",
        r"       delays TPW onset ~5 $\rightarrow$ ~16 Myr; a rigid interior $\rightarrow$ free rotation.",
        r"   •  Interior triaxiality sets the recurrence interval — constrainable via libration / degree-2 gravity.",
    ])

    # 14  Takeaways -----------------------------------------------------
    content(pp, 14, TOTAL, "11.  Key takeaways", "Takeaways", "11",
            layout="text", bullets_fs=15.5, bullets=[
        r"1.  3-D thermal structure matters.  Spatially variable albedo adds the largest",
        r"     non-tidal, degree-1 thickness signal — a leading/trailing asymmetry",
        r"     $b_{11}^s\sim$1–1.7 km — that 1-D models miss entirely.",
        r"",
        r"2.  Stability splits by moment.  $\Delta(B{-}A)/C$ is tidally robust; $\Delta(B{-}C)/A$ is",
        r"     albedo-controlled and can be destabilizing (flips Maxwell $H$=0 stable $\rightarrow$ unstable).",
        r"",
        r"3.  Recurrent polar wander.  Trailing-hemisphere darkening plausibly drives TPW on",
        r"     ~1–2 Myr (shell-thermal) timescales; interior coupling can delay onset to ~16 Myr.",
        r"",
        r"4.  Robust to smoothing.  All geophysical quantities converge by SH degree $\ell\approx4$",
        r"     (see backup) — insensitive to the unknown lateral-diffusion scale.",
    ])

    # 15  Caveats + updated references ----------------------------------
    content(pp, 15, TOTAL, "12.  Caveats & references", "Caveats", "12",
            layout="text", bullets_fs=13.5, bullets=[
        r"   •  Maxwell-rheology thicknesses run 14–24% below the original paper; the $4t_{22}$ spatial",
        r"       amplitude is ~2× the paper's — both unresolved by $C_R$ tuning (structural).",
        r"   •  The fossil-bulge factor ($\alpha$=1.5) and the instability threshold are partly ad hoc.",
        r"   •  Strong/weak interior-coupling outcome hinges on Europa's poorly constrained interior triaxiality $\varepsilon_i$.",
        r"",
        r"References",
        r"   Ojakangas & Stevenson (1989a). Thermal state of an ice shell on Europa. Icarus 81, 220–241.",
        r"   Ojakangas & Stevenson (1989b). Polar wander of an ice shell on Europa. Icarus 81, 242–270.",
        r"   Ross & Schubert (1987). Tidal heating in an internal ocean model of Europa. Nature 325, 133–134.",
        r"   Paranicas et al. (2001). Electron bombardment of Europa. Geophys. Res. Lett. 28, 673–676.",
        r"   Burnett & Hayne (2021). Europa's hemispheric color dichotomy as a constraint on non-synchronous",
        r"        rotation. Icarus 364, 114438.",
        r"   Burnett & Hayne (2023). Coupled spin dynamics of Europa's ice shell and interior. Planet. Sci. J.",
    ])

    # 16  Backup divider ------------------------------------------------
    divider(pp, "Backup", "Supporting Detail", accent=SLATE, bg="#EEF1F4")

    # 17  Backup: Robustness --------------------------------------------
    content(pp, 17, TOTAL, "Backup — Robustness to thermal smoothing", "Backup: robustness", "B1",
            layout="below", image=FIG[3], bullets_fs=14, bullets=[
        r"Truncate the surface-T spherical-harmonic expansion at degree $\ell_{max}$ (proxy for lateral diffusion):",
        r"   •  All geophysical quantities converge by $\ell_{max}\approx4$  (mostly <1% of full resolution)",
        r"   •  Albedo signal is degree-1, tidal signal degree-2 — both captured at low degree",
        r"   •  Results insensitive to the poorly-known lateral-diffusion timescale",
    ])

    # 18  Backup: Cycle-period scaling ----------------------------------
    content(pp, 18, TOTAL, "Backup — Cycle-period scaling", "Backup: cycle period", "B2",
            layout="right", image=FIG[7], bullets_fs=15, bullets=[
        r"   •  Cycle period scales linearly",
        r"       with the shell thermal",
        r"       timescale $\tau_{shell}$:",
        r"       ~0.2 Myr ($10^6$ yr)  $\rightarrow$",
        r"       ~15 Myr ($10^8$ yr)",
        r"",
        r"   •  Insensitive to the darkening",
        r"       timescale $\tau_{dark}$",
        r"       (over two decades)",
        r"",
        r"   •  Nominal $\tau_{shell}$=$10^7$ yr  $\rightarrow$",
        r"       recurrence ~1–2 Myr",
    ])

    # 19  Backup: TPW timescale comparison ------------------------------
    content(pp, 19, TOTAL, "Backup — TPW timescale comparison", "Backup: timescales", "B3",
            layout="below", image=FIG[9], bullets_fs=14, bullets=[
        r"OSPW overdamped timescale $\tau_f$ (viscous shell dissipation) vs. the free exponential $\tau_{TPW}$=$10^5$ yr.",
        r"   •  $\tau_f\sim0.64$ Myr at $T_f$=130 K — the free $\tau_{TPW}$ is ~100× too fast and unconstrained.",
        r"   •  The OSPW timescale derives from the shell's viscous structure (Arrhenius viscosity profile).",
    ])

    # 20  Backup: Model-framework comparison ----------------------------
    content(pp, 20, TOTAL, "Backup — Model-framework comparison: ours vs OSPW vs B&H", "Backup: model comparison", "B4",
            layout="below", image=FIG[10], bullets_fs=14, bullets=[
        r"Reorientation type, timescale, driving mechanism, shell dissipation, interior coupling,",
        r"geometry, fossil bulge, and instability criterion across the three frameworks.",
    ])

    # 21  Backup: Interior-coupling table -------------------------------
    content(pp, 21, TOTAL, "Backup — Interior coupling regimes", "Backup: interior coupling", "B5",
            layout="bullets_table", bullets_fs=14, bullets=[
        r"Adding Burnett & Hayne (2023) gravity-gradient shell–interior coupling:",
        r"   •  Near-hydrostatic interior delays TPW onset from ~5 to ~16 Myr; rigid interior $\rightarrow$ free rotation",
        r"   •  Interior triaxiality $\varepsilon_i$ sets the TPW recurrence interval (constrainable by libration / gravity)",
    ], table={"cols": ["Mode", "Onset", "Rotation character", "Interior"],
              "box": (0.06, 0.13, 0.88, 0.33), "fs": 12, "cw": [0.34, 0.16, 0.28, 0.22],
              "rows": [["Exponential", "~5 Myr", "continuous, ~450 deg/Myr", "N/A"],
                       ["OSPW (overdamped)", "~5 Myr", "continuous, ~2500 deg/Myr", "N/A"],
                       ["Coupled — rigid (ei~1e-5)", "~5 Myr", "continuous, ~2700 deg/Myr", "fixed"],
                       ["Coupled — hydrostatic (ei~5e-4)", "~16 Myr", "episodic -> continuous", "dragged"]]})


def main():
    build = tempfile.mkdtemp(prefix="iceshell_deck_")
    try:
        FIG = prepare_figures(build)
        with PdfPages(OUT) as pp:
            build_deck(pp, FIG)
        print("wrote", OUT)
    finally:
        shutil.rmtree(build, ignore_errors=True)


if __name__ == "__main__":
    main()
