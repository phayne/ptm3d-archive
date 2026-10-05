#!/usr/bin/env python
"""Separate sub-Jovian ECLIPSE COOLING from trailing-hemisphere ALBEDO DARKENING
in Europa's equilibrium ice shell (Tier 1 -- existing PTM3D output, no new sims).

The longitude-dependent shell-thickness signal has two physical sources:
  * b11^c (sub-Jovian / anti-Jovian) -- an INSOLATION asymmetry. The sub-Jovian
    hemisphere is eclipsed by Jupiter once per orbit (a localized deficit in the
    annual-mean direct insolation), slightly offset by Jupiter-shine (reflected
    sunlight + Jupiter thermal IR).
  * b11^s (leading / trailing)       -- an ALBEDO asymmetry from trailing-
    hemisphere darkening (the spatially variable Bond-albedo map).

This script:
  (T1) runs the O&S pipeline on the uniform-ice (A=0.5, NO darkening) run and the
       albedo-map (WITH darkening) run for 4 rheologies -> harmonics + MoI/TPW.
  (T2) decomposes the per-component annual-mean flux fields (direct solar = eclipse
       imprint, Jupiter reflected, Jupiter IR) and gives a first-order linearized
       attribution of b11^c, with the thermal-inertia caveat made explicit.

Gridding uses the library PTM3DSurfaceTemperature.temperature_2d(), which pads
the lambda = +-180 deg seam periodically, on the package's periodic grid (no
duplicated +-180 column). Thickness (O&S Eq. 2.8) and harmonics come from the
ojakangas_stevenson package; nothing is re-implemented here.

Run with the europa_py39 env (numpy/scipy/matplotlib):
    /Users/paha3326/anaconda3/envs/europa_py39/bin/python scripts/eclipse_vs_albedo_analysis.py
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src", "python"))

from ojakangas_stevenson import constants as C                       # noqa: E402
from ojakangas_stevenson import grid as G                            # noqa: E402
from ojakangas_stevenson.ptm3d_coupling import (                     # noqa: E402
    PTM3DSurfaceTemperature, _read_binary_timestep, _load_ply_face_latlon)
from ojakangas_stevenson.shell_thickness import (                    # noqa: E402
    equilibrium_thickness, dissipation_field, table1_case)

# ---------------------------------------------------------------- configuration
RUNS = {
    "no_darkening": os.path.join(REPO, "output", "europa_ico6_uniform-ice"),
    "darkening":    os.path.join(REPO, "output", "europa_ico6_albedo-map"),
}
ALBEDO = 0.5                 # uniform-ice Bond albedo (planetary_properties.h:108)
EPS = 0.95                   # surface emissivity / IR absorptivity (PTM3D)
SIGMA = 5.670374419e-8       # Stefan-Boltzmann
N_THETA, N_LAM = 361, 720
OUTDIR = os.path.join(REPO, "plots", "ice-shell-dynamics", "eclipse_analysis")

CASES = [("maxwell_h0", "Maxwell H=0"), ("maxwell_h10", "Maxwell H=10"),
         ("flow_ng3", "Flow nG=3"), ("flow_ng52", "Flow nG=5.2")]
# component, label, absorptivity
COMPONENTS = [("direct_solar", "eclipse (direct solar)", 1.0 - ALBEDO),
              ("primary_reflected", "Jupiter reflected", 1.0 - ALBEDO),
              ("primary_ir", "Jupiter IR", EPS)]

THETA1D, LAM1D = G.periodic_grid(N_THETA, N_LAM)


# ---------------------------------------------------------------- gridding (seam-free)
def gridded_temperature(run_dir):
    """Arithmetic time-mean surface temperature, N-S symmetrized, on the grid."""
    st = PTM3DSurfaceTemperature.from_binary_files(
        os.path.join(run_dir, "temperature"), os.path.join(run_dir, "mesh.ply"))
    st.symmetrize_ns()
    return st.temperature_2d(THETA1D, LAM1D)


def gridded_component(run_dir, comp, lat, lon):
    data = _read_binary_timestep(
        os.path.join(run_dir, "statistics", comp + "_avg.bin"), len(lat))
    if data is None:
        raise FileNotFoundError(f"unreadable {comp}_avg.bin in {run_dir}")
    obj = PTM3DSurfaceTemperature.from_arrays(data, lat, lon)
    obj.symmetrize_ns()
    return obj.temperature_2d(THETA1D, LAM1D)


# ---------------------------------------------------------------- O&S thickness + SH
CASE_KEYS = ("maxwell_h0", "maxwell_h10", "flow_ng3", "flow_ng52")


def build_dissipation():
    """Tidal dissipation integral D per rheology (temperature-independent).

    Returns {case: (D, H, rheology)}.
    """
    THETA, LAM = np.meshgrid(THETA1D, LAM1D, indexing="ij")
    out = {}
    for key in CASE_KEYS:
        rheology, H = table1_case(key)
        out[key] = (dissipation_field(rheology, THETA, LAM), H, rheology)
    return out


def solve_b(T_s, D, H):
    """O&S Eq. 2.8 (package implementation)."""
    return equilibrium_thickness(T_s, D, H)


def sh_full(b):
    """Degree-1 & 2 harmonics (km), average (km), and t20/t22 moments (m)."""
    h = G.harmonics(b, THETA1D, LAM1D)
    t20, t22 = h["t20"], h["t22"]
    return dict(avg=G.area_mean(b, THETA1D, LAM1D) / 1000.0, b10=h["b10"] / 1000,
                b11c=h["b11c"] / 1000, b11s=h["b11s"] / 1000, b20=t20 / 1000,
                b22c=t22 / 1000, b22s=h["b22s"] / 1000, t20=t20, t22=t22,
                four_t22=4 * t22 / 1000, t20_2t22=(t20 + 2 * t22) / 1000,
                dBA_C=1.0e-7 * (4 * t22 / 1000), dBC_A=1.0e-7 * ((t20 + 2 * t22) / 1000))


def b11c_only(b):
    return G.harmonics(b, THETA1D, LAM1D)["b11c"] / 1000


# ---------------------------------------------------------------- T1: geophysics
def run_geophysics(diss):
    print("=" * 96)
    print("T1  GEOPHYSICS: no-darkening (uniform A=0.5) vs with-darkening (albedo map)  [seam-free]")
    print("=" * 96)
    geo = {}
    for tag, run_dir in RUNS.items():
        print(f"\n[{tag}] gridding surface temperature ...")
        T = gridded_temperature(run_dir)
        res = {}
        for key, _ in CASES:
            D, H, _ = diss[key]
            b = solve_b(T, D, H)
            res[key] = dict(T=T, b_km=b / 1000, **sh_full(b))
        geo[tag] = res

    hdr = (f'{"case":<14}{"run":<13}{"<b>":>7}{"b11c":>8}{"b11s":>8}{"b10":>7}'
           f'{"b20":>8}{"b22c":>8}{"4t22":>7}{"t20+2t22":>9}'
           f'{"D(B-A)/C":>11}{"D(B-C)/A":>12}{"stable?":>9}')
    print("\n" + hdr)
    print("-" * len(hdr))
    table = {}
    for key, label in CASES:
        for tag in ("no_darkening", "darkening"):
            r = geo[tag][key]
            stable = "STABLE" if r["dBC_A"] < 0 else "unstable"
            print(f'{label:<14}{tag:<13}{r["avg"]:7.2f}{r["b11c"]:+8.3f}'
                  f'{r["b11s"]:+8.3f}{r["b10"]:+7.3f}{r["b20"]:+8.3f}{r["b22c"]:+8.3f}'
                  f'{r["four_t22"]:7.2f}{r["t20_2t22"]:+9.2f}'
                  f'{r["dBA_C"]:11.2e}{r["dBC_A"]:12.2e}{stable:>9}')
            table[f"{key}|{tag}"] = {k: r[k] for k in
                                     ("avg", "b11c", "b11s", "b10", "b20", "b22c",
                                      "b22s", "four_t22", "t20_2t22", "dBA_C", "dBC_A")}
        print()
    return geo, table


# ---------------------------------------------------------------- T2: attribution
def run_attribution(geo, diss):
    print("=" * 96)
    print("T2  FLUX-COMPONENT DECOMPOSITION & b11^c ATTRIBUTION  (uniform-ice run, A=0.5)")
    print("=" * 96)
    run_dir = RUNS["no_darkening"]
    lat, lon = _load_ply_face_latlon(os.path.join(run_dir, "mesh.ply"))
    fmap = {comp: gridded_component(run_dir, comp, lat, lon) for comp, _, _ in COMPONENTS}

    # Equatorial cardinal-point flux table (robust, no linearization)
    ieq = np.argmin(np.abs(THETA1D - np.pi / 2))
    cards = {"sub-Jovian": 0, "leading": 90, "anti-Jovian": 180, "trailing": -90}
    print("\nEquatorial annual-mean flux [W/m^2] at cardinal longitudes:")
    print(f'{"component":<20}' + "".join(f"{k:>13}" for k in cards))
    eclipse_deficit = None
    for comp, _, _ in COMPONENTS:
        vals = {k: fmap[comp][ieq, np.argmin(np.abs(LAM1D - np.radians(L)))]
                for k, L in cards.items()}
        print(f'{comp:<20}' + "".join(f"{vals[k]:13.3f}" for k in cards))
        if comp == "direct_solar":
            non_subj = np.mean([vals["leading"], vals["anti-Jovian"], vals["trailing"]])
            eclipse_deficit = non_subj - vals["sub-Jovian"]
    print(f'\n  Eclipse deficit (mean non-subJ direct - subJ direct) = {eclipse_deficit:.2f} W/m^2 '
          f'({100*eclipse_deficit/16.7:.0f}% of annual-mean insolation)')
    jshine = (fmap["primary_reflected"][ieq, np.argmin(np.abs(LAM1D))]
              + fmap["primary_ir"][ieq, np.argmin(np.abs(LAM1D))])
    print(f'  Jupiter-shine at sub-Jovian (reflected+IR) = {jshine:.3f} W/m^2 (warming, opposes) '
          f'-> {eclipse_deficit/max(jshine,1e-9):.0f}x smaller than eclipse deficit\n')

    # First-order linearized b11^c attribution
    print("First-order linearized b11^c attribution [km]  (dT = absorptivity * (F - <F>_lon)")
    print("/ (4 eps sigma T^3); contribution = b11c(T) - b11c(T - dT)).  Thermally biased:")
    print("the time-averaged flux ignores eclipse-transient buffering, so the residual is large.")
    fanom = {comp: fmap[comp] - fmap[comp].mean(axis=1, keepdims=True) for comp, _, _ in COMPONENTS}
    T_full = geo["no_darkening"]["maxwell_h0"]["T"]
    hdr = (f'{"case":<14}{"eclipse":>10}{"Jup refl":>10}{"Jup IR":>10}'
           f'{"sum":>10}{"total b11c":>12}{"residual":>10}')
    print("\n" + hdr)
    print("-" * len(hdr))
    attrib = {}
    for key, label in CASES:
        D, H, _ = diss[key]
        b_full = solve_b(T_full, D, H)
        tot = b11c_only(b_full)
        parts = {}
        for comp, _, absb in COMPONENTS:
            dT = absb * fanom[comp] / (4 * EPS * SIGMA * T_full**3)
            parts[comp] = tot - b11c_only(solve_b(T_full - dT, D, H))
        ssum = sum(parts.values())
        print(f'{label:<14}{parts["direct_solar"]:+10.4f}{parts["primary_reflected"]:+10.4f}'
              f'{parts["primary_ir"]:+10.4f}{ssum:+10.4f}{tot:+12.4f}{tot-ssum:+10.4f}')
        attrib[key] = dict(eclipse=parts["direct_solar"], jup_reflected=parts["primary_reflected"],
                           jup_ir=parts["primary_ir"], total=tot, residual=tot - ssum)
    print()
    return fmap, attrib, eclipse_deficit, jshine


# ---------------------------------------------------------------- figures
def _mollweide(ax, field, title, cmap, vmin=None, vmax=None, cbar_label=""):
    lat = np.pi / 2 - THETA1D
    LON, LAT = np.meshgrid(LAM1D, lat)
    pcm = ax.pcolormesh(LON, LAT, field, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="auto", rasterized=True)
    ax.set_title(title, fontsize=9.5)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    cb = ax.figure.colorbar(pcm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
    cb.set_label(cbar_label, fontsize=8); cb.ax.tick_params(labelsize=7)


def make_figures(geo, fmap, attrib):
    os.makedirs(OUTDIR, exist_ok=True)
    ieq = np.argmin(np.abs(THETA1D - np.pi / 2))
    lam_deg = np.degrees(LAM1D)

    # F1: surface temperature
    T_nd, T_d = geo["no_darkening"]["maxwell_h0"]["T"], geo["darkening"]["maxwell_h0"]["T"]
    dT = T_d - T_nd
    fig = plt.figure(figsize=(13, 3.7))
    vmax = np.nanmax(np.abs(dT))
    for i, (fld, ttl, cmap, lab, vl) in enumerate([
            (T_nd, "No darkening (uniform A=0.5)", "inferno", "$T_s$ [K]", None),
            (T_d, "With darkening (albedo map)", "inferno", "$T_s$ [K]", None),
            (dT, "Difference (with $-$ without)", "RdBu_r", "$\\Delta T_s$ [K]", vmax)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="mollweide")
        _mollweide(ax, fld, ttl, cmap, vmin=(-vl if vl else None), vmax=vl, cbar_label=lab)
    fig.suptitle("Annual-mean surface temperature (N-S symmetrized)", fontsize=12)
    fig.savefig(os.path.join(OUTDIR, "F1_surface_temperature.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # F2: flux components + equatorial profile
    fig = plt.figure(figsize=(13, 6.8))
    comps = [("direct_solar", "Direct solar (eclipse imprint)", "viridis"),
             ("primary_reflected", "Jupiter reflected", "magma"),
             ("primary_ir", "Jupiter thermal IR", "magma")]
    for i, (comp, ttl, cmap) in enumerate(comps):
        ax = fig.add_subplot(2, 3, i + 1, projection="mollweide")
        _mollweide(ax, fmap[comp], ttl, cmap, cbar_label="W m$^{-2}$")
    axp = fig.add_subplot(2, 1, 2)
    axp.plot(lam_deg, fmap["direct_solar"][ieq], lw=1.8, color="#2A7F4F", label="direct solar")
    axp.set_xlabel("Longitude [deg]   (0 = sub-Jovian, +-180 = anti-Jovian)")
    axp.set_ylabel("Equatorial direct solar [W m$^{-2}$]", color="#2A7F4F")
    axp.axvline(0, color="0.5", lw=0.8, ls="--")
    axp.annotate("eclipse deficit\nat sub-Jovian", xy=(0, fmap["direct_solar"][ieq].min()),
                 xytext=(60, fmap["direct_solar"][ieq].min() + 0.6), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="0.4"))
    axp.set_xlim(-180, 180); axp.grid(True, lw=0.3, alpha=0.4)
    ax2 = axp.twinx()
    ax2.plot(lam_deg, fmap["primary_reflected"][ieq], lw=1.3, color="#B04030", label="Jup reflected")
    ax2.plot(lam_deg, fmap["primary_ir"][ieq], lw=1.3, color="#D08020", label="Jup IR")
    ax2.set_ylabel("Jupiter-shine [W m$^{-2}$]", color="#B04030")
    lines = axp.get_lines() + ax2.get_lines()
    axp.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")
    fig.suptitle("Flux components (uniform-ice run): localized eclipse dip at sub-Jovian; "
                 "Jupiter-shine ~30x smaller", fontsize=11)
    fig.savefig(os.path.join(OUTDIR, "F2_flux_components.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # F3: thickness
    b_nd, b_d = geo["no_darkening"]["maxwell_h0"]["b_km"], geo["darkening"]["maxwell_h0"]["b_km"]
    db = b_d - b_nd
    fig = plt.figure(figsize=(13, 3.7))
    vmax = np.nanmax(np.abs(db))
    for i, (fld, ttl, cmap, lab, vl) in enumerate([
            (b_nd, "No darkening", "gist_earth", "b [km]", None),
            (b_d, "With darkening", "gist_earth", "b [km]", None),
            (db, "Difference (with $-$ without)", "RdBu_r", "$\\Delta b$ [km]", vmax)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="mollweide")
        _mollweide(ax, fld, ttl, cmap, vmin=(-vl if vl else None), vmax=vl, cbar_label=lab)
    fig.suptitle("Equilibrium shell thickness, Maxwell $H$=0", fontsize=12)
    fig.savefig(os.path.join(OUTDIR, "F3_thickness.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # F4: harmonic bars
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [lab for _, lab in CASES]; x = np.arange(len(CASES))
    for ax, harm, ttl in zip(axes, ["b11c", "b11s", "b22c"],
                             [r"$b_{11}^c$ (sub/anti-Jovian, insolation)",
                              r"$b_{11}^s$ (leading/trailing, albedo)",
                              r"$b_{22}^c$ (tidal)"]):
        nd = [geo["no_darkening"][k][harm] for k, _ in CASES]
        dk = [geo["darkening"][k][harm] for k, _ in CASES]
        ax.bar(x - 0.2, nd, 0.4, label="no darkening", color="#4C72B0")
        ax.bar(x + 0.2, dk, 0.4, label="with darkening", color="#C44E52")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=9.5); ax.axhline(0, color="0.4", lw=0.6)
        ax.set_ylabel("km"); ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Thickness harmonics: darkening adds the leading/trailing $b_{11}^s$; "
                 r"the eclipse $b_{11}^c$ is present with or without darkening", fontsize=11)
    fig.savefig(os.path.join(OUTDIR, "F4_harmonics.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # F5: MoI / stability
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mkey, ttl in zip(axes, ["dBA_C", "dBC_A"],
                             [r"$\Delta(B-A)/C$ (tidal, robust)",
                              r"$\Delta(B-C)/A$ (stability lever)"]):
        nd = [geo["no_darkening"][k][mkey] for k, _ in CASES]
        dk = [geo["darkening"][k][mkey] for k, _ in CASES]
        ax.bar(x - 0.2, nd, 0.4, label="no darkening", color="#4C72B0")
        ax.bar(x + 0.2, dk, 0.4, label="with darkening", color="#C44E52")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=9.5); ax.axhline(0, color="k", lw=0.9)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Moment-of-inertia / TPW stability ($\Delta(B-C)/A>0 \Rightarrow$ unstable)", fontsize=12)
    fig.savefig(os.path.join(OUTDIR, "F5_moi_stability.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # F6: attribution waterfall (Maxwell H=0)
    a = attrib["maxwell_h0"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    steps = [("eclipse", a["eclipse"], "#3B6FB0"), ("Jupiter\nreflected", a["jup_reflected"], "#E1812C"),
             ("Jupiter\nIR", a["jup_ir"], "#E1812C"), ("residual\n(thermal\ninertia)", a["residual"], "#999999")]
    running = 0.0
    for i, (lbl, val, col) in enumerate(steps):
        ax.bar(i, val, bottom=running, color=col, edgecolor="k", lw=0.5); running += val
    ax.bar(len(steps), a["total"], color="#55A868", edgecolor="k", lw=0.5)
    ax.set_xticks(list(range(len(steps) + 1)))
    ax.set_xticklabels([s[0] for s in steps] + ["total\n$b_{11}^c$"], fontsize=8)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel(r"contribution to $b_{11}^c$ [km]")
    ax.set_title("First-order $b_{11}^c$ attribution, Maxwell $H$=0\n"
                 "(linearized; large residual reflects eclipse-transient thermal buffering)", fontsize=9.5)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    fig.savefig(os.path.join(OUTDIR, "F6_attribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures -> {OUTDIR}")


# ---------------------------------------------------------------- verification
def verify(geo, fmap):
    print("=" * 96)
    print("VERIFICATION")
    print("=" * 96)
    # b10 should be ~0 after N-S symmetrization
    b10 = geo["no_darkening"]["maxwell_h0"]["b10"]
    print(f"  N-S symmetrization: |b10| = {abs(b10):.4f} km (should be ~0)")
    # albedo separation
    print(f"  no-darkening b11s = {geo['no_darkening']['maxwell_h0']['b11s']:+.3f} km (should be ~0)")
    print(f"  darkening    b11s = {geo['darkening']['maxwell_h0']['b11s']:+.3f} km (leading/trailing)")
    # published comparison (ICO7, Maxwell H=0): b11c=+0.36, b11s=-1.29, b22c=+1.50 (albedo-map)
    d = geo["darkening"]["maxwell_h0"]
    print(f"  ICO6 albedo-map Maxwell H=0: b11c={d['b11c']:+.3f} (pub +0.364), "
          f"b11s={d['b11s']:+.3f} (pub -1.286), b22c={d['b22c']:+.3f} (pub +1.497)")
    # energy balance at equator
    ieq = np.argmin(np.abs(THETA1D - np.pi / 2)); j0 = np.argmin(np.abs(LAM1D))
    absb = (1 - ALBEDO) * fmap["direct_solar"][ieq, j0] + (1 - ALBEDO) * fmap["primary_reflected"][ieq, j0] \
        + EPS * fmap["primary_ir"][ieq, j0]
    Trad = (absb / (EPS * SIGMA))**0.25
    Tact = geo["no_darkening"]["maxwell_h0"]["T"][ieq, j0]
    print(f"  energy balance @ sub-Jovian eq: absorbed={absb:.2f} W/m^2 -> Trad={Trad:.1f}K vs actual={Tact:.1f}K "
          f"(diff from <T^4> != <T>^4 thermal inertia)\n")


# ---------------------------------------------------------------- main
def main():
    print(C.describe_active())
    diss = build_dissipation()
    geo, table = run_geophysics(diss)
    fmap, attrib, ecl_def, jshine = run_attribution(geo, diss)
    verify(geo, fmap)
    make_figures(geo, fmap, attrib)
    os.makedirs(OUTDIR, exist_ok=True)
    dump = {"parameter_set": C.active_parameter_set().as_dict(), "C_R": C.C_R,
            "T1": table, "T2": attrib, "eclipse_deficit_Wm2": ecl_def,
            "jupiter_shine_subjovian_Wm2": jshine, "albedo": ALBEDO, "eps": EPS}
    with open(os.path.join(OUTDIR, "numbers.json"), "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"Numbers -> {os.path.join(OUTDIR, 'numbers.json')}")


if __name__ == "__main__":
    main()
