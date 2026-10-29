# PTM3D × JWST/NIRSpec (Cartwright et al. 2025) Comparison Toolkit

End-to-end pipeline for testing PTM3D Europa exosphere predictions against the
spatially resolved spectral maps in *Cartwright et al. (2025), PSJ 6, 125*.

## Hypotheses

- **H1 (primary)** — equilibrium crystalline ice fraction tracks $T_{\max}$ via
  the Mitchell+Nordheim balance with hemisphere-asymmetric threshold near
  $T^* \approx 125$ K.
- **H2** — 2.0 μm $H_2O$ band center shifts linearly with $T_{\max}$
  (Cartwright two-anchor fit).
- **H3** — observed CO₂ band depth distribution should match a thermal
  cold-trap pattern (poleward) only if no endogenic source is required.

## Pipeline

```bash
# 1. Forward model — predict PTM3D state-derived observables
python scripts/jwst/predict_phi_eq.py \
    --run output/europa_ico6_albedo-map \
    --out data/jwst/cartwright2025/phi_eq_albedo_map.npz \
    --integrate --n-orbits 200            # ~3 minutes

python scripts/jwst/predict_band_center_2um.py \
    --phi-npz data/jwst/cartwright2025/phi_eq_albedo_map.npz \
    --out data/jwst/cartwright2025/band_center_2um.npz

python scripts/jwst/predict_co2_coldtrap.py \
    --phi-npz data/jwst/cartwright2025/phi_eq_albedo_map.npz \
    --out data/jwst/cartwright2025/co2_coldtrap.npz

# 2. Ingest Cartwright observations (digitize from paper figures first)
python scripts/jwst/ingest_cartwright2025.py \
    --csv-dir data/jwst/cartwright2025/digitized \
    --out data/jwst/cartwright2025/observations.npz
# Add --allow-placeholder for end-to-end testing without real data

# 3. End-to-end comparison
python scripts/jwst/compare_jwst_ptm3d.py
# Writes: data/jwst/cartwright2025/comparison_summary.txt
#         docs/figures/jwst/comparison_4panel.png

# 4. Talk figures
python scripts/jwst/plot_sigmoid_theory.py \
    --phi-npz data/jwst/cartwright2025/phi_eq_albedo_map.npz \
    --out docs/figures/jwst/sigmoid_theory.png
python scripts/jwst/plot_phi_C_map.py \
    --phi-npz data/jwst/cartwright2025/phi_eq_albedo_map.npz \
    --out docs/figures/jwst/phi_C_map.png
```

## Module map

| File                                | Role                                               |
|-------------------------------------|----------------------------------------------------|
| `crystallization.py`                | Python port of `ice_crystallization_model.h` (§D.7)|
| `loaders.py`                        | Mesh + per-face binary loaders (wrap `extract_timeseries.py`) |
| `equirect.py`                       | KD-tree-based equirectangular projection + landmarks |
| `predict_phi_eq.py`                 | H1 forward model (φ_eq + integrated φ_C)           |
| `predict_band_center_2um.py`        | H2 forward model                                   |
| `predict_co2_coldtrap.py`           | H3 forward model                                   |
| `ingest_cartwright2025.py`          | Digitized-CSV → unified NPZ ingestion              |
| `compare_jwst_ptm3d.py`             | End-to-end comparison metrics + 4-panel figure     |
| `plot_sigmoid_theory.py`            | Theory slide: φ_eq(T) sigmoids                     |
| `plot_phi_C_map.py`                 | Main equirectangular φ_C + T_max maps              |

## Digitizing Cartwright (2025) maps

Only the raw NIRSpec cubes are deposited (MAST DOI 10.17909/ve8q-sq65). The
band-parameter maps must be digitized from the paper figures:

| Figure | Map                                  | Save as                                 |
|--------|--------------------------------------|-----------------------------------------|
| Fig 6/8 | 3.1 μm Fresnel peak strength        | `digitized/fresnel_31um.csv`            |
| Fig 4   | 2.0 μm band center [μm]             | `digitized/band_center_2um.csv`         |
| Fig 7   | 4.25 μm CO₂ band depth              | `digitized/co2_band_depth.csv`          |

CSV format (header optional):
```
lat_deg, lon_deg_E, value [, uncertainty]
```
Use **IAU east-positive longitude** (matches mesh.ply). Cartwright figures
typically use west-positive or sub-Earth-centered coordinates; convert before
saving.

## Coordinate convention note

Both the run's `mesh.ply` and the Galileo Bond-albedo map place Europa's
**trailing apex at 90°E** (IAU east-positive). The C++ source
`src/cpp/exosphere/ice_crystallization_model.h` hard-codes the trailing apex
at 270°E — a 180° offset from the actual data convention. The Python forward
model uses the correct location (90°E, exposed via `TRAILING_APEX_LON_E`).
The C++ bug should be fixed before any production crystallization run.

## Validation

`crystallization.py` reproduces Tables D6 and D7 of
`docs/appendix-d-exosphere-model-full.md` to within rounding, and the
diurnal integrator converges to the closed-form `phi_eq` at constant T to
machine precision (verified on 200 orbits at T = 110, 120, 125, 130 K).
