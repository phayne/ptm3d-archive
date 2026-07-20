#!/usr/bin/env python3
"""
Generate polar stereographic maps and analysis plots for the survey.
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from survey_config import SurveyConfig, load_config
from utils import read_ply_mesh, read_stats_binary, read_classification_binary


def make_stereo_mesh(verts, faces, nf, R=1737400.0):
    """Transform mesh vertices to south polar stereographic coordinates."""
    v_r = np.sqrt(np.sum(verts**2, axis=1))
    v_lat = np.arcsin(verts[:, 2] / v_r)
    v_lon = np.arctan2(verts[:, 1], verts[:, 0])
    v_k = 2.0 * R * np.tan(np.pi/4.0 + v_lat/2.0)
    sp_verts = np.column_stack([
        v_k * np.sin(v_lon),
        v_k * np.cos(v_lon),
        (v_r - R).astype(np.float32)
    ]).astype(np.float32)
    faces_pv = np.column_stack([np.full(nf, 3), faces]).ravel().astype(np.int64)
    return pv.PolyData(sp_verts, faces_pv)


def stereo_coords(lats, lons, R=1737400.0):
    """Convert lat/lon arrays to south polar stereo x, y."""
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    k = 2.0 * R * np.tan(np.pi/4.0 + lat_rad/2.0)
    return k * np.sin(lon_rad), k * np.cos(lon_rad)


def add_grid(pl, R_m=1737400.0, lat_step=2.5, lon_step=30,
             lat_range=(-88, -80), color='gray', opacity=0.4, width=1.5):
    """Add lat/lon grid lines to a PyVista plotter."""
    for lon_deg in np.arange(0, 360, lon_step):
        lr = np.radians(lon_deg)
        lat_pts = np.linspace(np.radians(lat_range[0]), np.radians(lat_range[1]), 100)
        k = 2.0 * R_m * np.tan(np.pi/4 + lat_pts/2)
        pts = np.column_stack([k*np.sin(lr), k*np.cos(lr), np.zeros(100)+50])
        pl.add_mesh(pv.Spline(pts, n_points=100), color=color, opacity=opacity, line_width=width)
    for lat_deg in np.arange(lat_range[0], lat_range[1]+0.1, lat_step):
        lr = np.radians(lat_deg)
        lp = np.linspace(0, 2*np.pi, 360)
        k = 2.0 * R_m * np.tan(np.pi/4 + lr/2)
        pts = np.column_stack([k*np.sin(lp), k*np.cos(lp), np.zeros(360)+50])
        pl.add_mesh(pv.Spline(pts, n_points=360), color=color, opacity=opacity, line_width=width)


def setup_camera(pl, sp_x, sp_y):
    """Standard south polar overhead camera."""
    cx = (sp_x.min()+sp_x.max())/2
    cy = (sp_y.min()+sp_y.max())/2
    pl.camera_position = [(cx, cy, 50000), (cx, cy, 0), (0, 1, 0)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = max(sp_x.max()-sp_x.min(), sp_y.max()-sp_y.min()) * 0.55


def _outlined_text(draw, pos, text, fill, font, outline=(0,0,0), width=2):
    x, y = pos
    for dx in range(-width, width+1):
        for dy in range(-width, width+1):
            if dx or dy:
                draw.text((x+dx, y+dy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def annotate_image(img_path, sp_x, sp_y, R_m=1737400.0,
                   text_color=(255,255,255), title=None, title_color=(0,0,0),
                   legend=None, named_sites=None):
    """Add labels, title, legend, and named site markers to a rendered image."""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    W, H = img.size
    scale = max(sp_x.max()-sp_x.min(), sp_y.max()-sp_y.min()) * 0.55
    cx_s = (sp_x.min()+sp_x.max())/2
    cy_s = (sp_y.min()+sp_y.max())/2

    def s2p(sx, sy):
        return int(W/2 + (sx-cx_s)/(2*scale)*W), int(H/2 - (sy-cy_s)/(2*scale)*H)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_legend = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = font_sm = font_title = font_legend = ImageFont.load_default()

    outline = (0,0,0) if sum(text_color) > 380 else (255,255,255)

    # Longitude labels
    for lon_deg in range(0, 360, 30):
        k = 2.0*R_m*np.tan(np.pi/4+np.radians(-80)/2)*1.07
        px, py = s2p(k*np.sin(np.radians(lon_deg)), k*np.cos(np.radians(lon_deg)))
        if 20 < px < W-20 and 20 < py < H-20:
            _outlined_text(draw, (px-18, py-10), f"{lon_deg}°", text_color, font_sm, outline)

    # Latitude labels along 45° lon
    for lat_deg in np.arange(-88, -79, 2.5):
        k = 2.0*R_m*np.tan(np.pi/4+np.radians(lat_deg)/2)
        px, py = s2p(k*np.sin(np.radians(45)), k*np.cos(np.radians(45)))
        if 20 < px < W-50 and 20 < py < H-20:
            _outlined_text(draw, (px+8, py-10), f"{int(lat_deg)}°S", text_color, font_sm, outline)

    # Title
    if title:
        t_out = (255,255,255) if sum(title_color) < 380 else (0,0,0)
        _outlined_text(draw, (15, 12), title, title_color, font_title, t_out)

    # Legend
    if legend:
        y0 = H - 25 - len(legend)*30
        for i, (label, c) in enumerate(legend):
            y = y0 + i*30
            draw.rectangle([10, y-2, 310, y+22], fill=(255,255,255,200))
            draw.rectangle([15, y+2, 32, y+17], fill=c, outline=(0,0,0))
            draw.text((40, y), label, fill=c, font=font_legend)

    # Named sites (green star + circle)
    if named_sites:
        for ns in named_sites:
            nlr = np.radians(ns['lat'])
            nk = 2.0*R_m*np.tan(np.pi/4+nlr/2)
            npx, npy = s2p(nk*np.sin(np.radians(ns['lon'])), nk*np.cos(np.radians(ns['lon'])))
            if 10 < npx < W-10 and 10 < npy < H-10:
                pts = []
                for ks in range(10):
                    ang = ks*np.pi/5 - np.pi/2
                    rs = 18 if ks%2==0 else 8
                    pts.append((npx + rs*np.cos(ang), npy + rs*np.sin(ang)))
                draw.polygon(pts, fill=(0, 220, 0), outline=(0, 0, 0))
                draw.ellipse([npx-24, npy-24, npx+24, npy+24], outline=(0, 220, 0), width=3)

    img.save(img_path)


def plot_classification(cfg, sp_mesh, sp_x, sp_y, classification, locations):
    """Plot shadow classification map."""
    pv.global_theme.background = 'white'
    cmap = ListedColormap(['#2166ac', '#f4a582', '#d9d9d9'])
    sp_mesh.cell_data['classification'] = classification

    pl = pv.Plotter(off_screen=True, window_size=[1400, 1400])
    pl.add_mesh(sp_mesh, scalars='classification', cmap=cmap, clim=[-0.5, 2.5],
                show_scalar_bar=False, smooth_shading=False)
    for loc in locations:
        sx, sy = stereo_coords(np.array([loc['lat']]), np.array([loc['lon']]), cfg.body_radius)
        pl.add_mesh(pv.Sphere(radius=300, center=(float(sx[0]), float(sy[0]), 300)), color='red')
    add_grid(pl, cfg.body_radius)
    setup_camera(pl, sp_x, sp_y)

    path = str(cfg.figures_dir / 'classification.png')
    pl.screenshot(path); pl.close()
    annotate_image(path, sp_x, sp_y, cfg.body_radius,
        text_color=(60,60,60), title="Shadow Classification",
        title_color=(0,0,0),
        legend=[("PSR", (33,102,172)), ("Seasonal", (220,130,90)),
                ("Always Lit", (140,140,140)), ("Target", (220,0,0))],
        named_sites=[{'lat': ns.lat, 'lon': ns.lon} for ns in cfg.named_sites])
    return path


def plot_scalar_map(cfg, sp_mesh, sp_x, sp_y, data, title, cmap, clim,
                    bar_title, filename, locations=None):
    """Plot a generic scalar field on the stereographic mesh."""
    pv.global_theme.background = 'white'
    sp_tmp = sp_mesh.copy()
    sp_tmp.cell_data['scalar'] = data

    pl = pv.Plotter(off_screen=True, window_size=[1400, 1400])
    pl.add_mesh(sp_tmp, scalars='scalar', cmap=cmap, clim=clim,
                scalar_bar_args={'title': bar_title, 'color': 'black',
                                 'title_font_size': 14, 'label_font_size': 12,
                                 'position_x': 0.82, 'position_y': 0.15, 'width': 0.12},
                smooth_shading=False)
    if locations:
        for loc in locations:
            sx, sy = stereo_coords(np.array([loc['lat']]), np.array([loc['lon']]), cfg.body_radius)
            pl.add_mesh(pv.Sphere(radius=300, center=(float(sx[0]), float(sy[0]), 300)), color='red')
    add_grid(pl, cfg.body_radius, color='white', opacity=0.3)
    setup_camera(pl, sp_x, sp_y)

    path = str(cfg.figures_dir / filename)
    pl.screenshot(path); pl.close()
    annotate_image(path, sp_x, sp_y, cfg.body_radius, text_color=(255,255,255),
        title=title, title_color=(255,255,255),
        named_sites=[{'lat': ns.lat, 'lon': ns.lon} for ns in cfg.named_sites])
    return path


def plot_seasonal_pattern(cfg, lats, lons, locations, nf):
    """Plot seasonal illumination pattern (fraction illuminated per lunation)."""
    n_lun = cfg.illumination.num_lunations
    lunation_dates = [cfg.time.start_datetime + timedelta(days=i * cfg.time.lunation_days)
                      for i in range(n_lun)]
    frac_all = np.zeros(n_lun)
    frac_targets = np.zeros(n_lun)

    for i in range(n_lun):
        stats_file = cfg.survey_dir / f'lunation_{i:02d}/statistics/direct_solar_avg.bin'
        if not stats_file.exists():
            continue
        avg = read_stats_binary(str(stats_file))
        if len(avg) != nf:
            continue
        frac_all[i] = np.sum(avg > 0) / nf
        if locations:
            n_lit = sum(1 for loc in locations if avg[loc['face_idx']] > 0.1)
            frac_targets[i] = n_lit / len(locations)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(lunation_dates, frac_all*100, 'o-', color='#333', ms=5, lw=1.5, label='All faces')
    ax1.set_ylabel('Illuminated faces (%)')
    y_min = max(0, frac_all.min()*100 - 5)
    y_max = min(100, frac_all.max()*100 + 5)
    ax1.set_ylim(y_min, y_max)

    if locations:
        ax2 = ax1.twinx()
        ax2.plot(lunation_dates, frac_targets*100, 's-', color='#d62728', ms=6, lw=1.5,
                 label='Target locations')
        ax2.set_ylabel('Targets illuminated (%)', color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        ax2.set_ylim(-5, 105)
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.set_xlabel('Date')
    ax1.set_title(f'Seasonal Illumination Pattern')
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', framealpha=0.9)

    path = str(cfg.figures_dir / 'seasonal_pattern.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def plot_all(cfg: SurveyConfig):
    """Generate all survey plots."""
    print("=== Generating Plots ===")
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'

    # Load mesh
    print("  Loading mesh...")
    verts, faces, fc, lats, lons = read_ply_mesh(str(cfg.mesh_path))
    nf = len(faces)
    sp_mesh = make_stereo_mesh(verts, faces, nf, cfg.body_radius)
    sp_x, sp_y = stereo_coords(lats, lons, cfg.body_radius)

    # Load locations
    locations = []
    if cfg.locations_csv.exists():
        with open(cfg.locations_csv) as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                p = line.strip().split(',')
                if len(p) >= 3:
                    dlat = lats - float(p[1])
                    dlon = lons - float(p[2])
                    fidx = int(np.argmin(dlat**2 + dlon**2))
                    locations.append({'name': p[0], 'lat': float(p[1]), 'lon': float(p[2]),
                                      'face_idx': fidx})

    # Classification
    class_file = cfg.survey_dir / "face_classification.bin"
    if class_file.exists():
        classification = read_classification_binary(str(class_file))
        plot_classification(cfg, sp_mesh, sp_x, sp_y, classification, locations)
        print("  Classification map ✓")

    # Illumination fraction
    frac_file = cfg.survey_dir / "illum_total_fraction.bin"
    if frac_file.exists():
        illum_frac = read_stats_binary(str(frac_file))
        plot_scalar_map(cfg, sp_mesh, sp_x, sp_y, illum_frac*100,
                        "Illumination Fraction (%)", 'viridis', [0, 40],
                        '% Time Illuminated', 'illum_fraction.png', locations)
        print("  Illumination fraction map ✓")

    # Continuous duration
    hours_file = cfg.survey_dir / "illum_max_continuous_hours.bin"
    if hours_file.exists():
        max_hrs = read_stats_binary(str(hours_file))
        plot_scalar_map(cfg, sp_mesh, sp_x, sp_y, np.minimum(max_hrs, 200),
                        "Max Continuous Illumination (hr)", 'plasma', [0, 200],
                        'Max Continuous (hr)', 'continuous_duration.png', locations)
        print("  Continuous duration map ✓")

    # Summer/winter temperature
    for lun, label in [(cfg.thermal.lunations[0] if cfg.thermal.lunations else 7, 'summer'),
                        (cfg.thermal.lunations[-1] if len(cfg.thermal.lunations) > 1 else 13, 'winter')]:
        for stat, sname, clim in [('avg', 'Avg T (K)', [25, 250]), ('max', 'Max T (K)', [25, 350])]:
            tfile = cfg.thermal_dir / f"lunation_{lun:02d}/statistics/temperature_{stat}.bin"
            if tfile.exists():
                tdata = read_stats_binary(str(tfile))
                plot_scalar_map(cfg, sp_mesh, sp_x, sp_y, tdata,
                                f"{label.title()} {sname}", 'magma', clim,
                                sname, f'temp_{label}_{stat}.png', locations)
                print(f"  {label.title()} {stat} temperature ✓")

    # Seasonal pattern
    plot_seasonal_pattern(cfg, lats, lons, locations, nf)
    print("  Seasonal pattern ✓")

    print(f"  Figures saved to: {cfg.figures_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_maps.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    plot_all(cfg)
