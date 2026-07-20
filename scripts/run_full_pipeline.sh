#!/bin/bash
# =============================================================================
# Amundsen Crater Seasonal Shadow Study - Full Pipeline
# Phase 1: Illumination survey + face classification
# Phase 2: Thermal + radiation for identified locations
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== AMUNDSEN CRATER SEASONAL SHADOW STUDY ==="
echo "Project dir: $PROJECT_DIR"
echo "Start time: $(date)"
echo ""

# ============================================
# PHASE 1: Illumination Survey
# ============================================
echo "=============================="
echo "PHASE 1: Illumination Survey"
echo "=============================="

if [ -d "output/survey/lunation_24/statistics" ]; then
    echo "Phase 1 survey already complete (all 25 lunations found). Skipping."
else
    bash scripts/amundsen_seasonal_survey.sh
fi

# ============================================
# PHASE 1b: Classify faces
# ============================================
echo ""
echo "=============================="
echo "PHASE 1b: Face Classification"
echo "=============================="

if [ -f "configs/amundsen_target_locations.csv" ]; then
    echo "Locations file already exists. Skipping classification."
else
    python3 scripts/identify_seasonal_shadow.py \
        --survey-dir output/survey \
        --mesh /Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/dtm/LDEM_80S_20M__amundsen_2Mpts.ply \
        --num-lunations 25 \
        --output-locations configs/amundsen_target_locations.csv
fi

echo ""
echo "Locations file:"
head -10 configs/amundsen_target_locations.csv

# ============================================
# PHASE 2: Thermal + Radiation
# ============================================
echo ""
echo "=============================="
echo "PHASE 2: Thermal + Radiation"
echo "=============================="

bash scripts/amundsen_thermal_series.sh

# ============================================
# PHASE 2b: Combine outputs
# ============================================
echo ""
echo "=============================="
echo "PHASE 2b: Combine Outputs"
echo "=============================="

python3 scripts/format_seasonal_timeseries.py \
    --thermal-dir output/thermal \
    --locations configs/amundsen_target_locations.csv \
    --num-lunations 25 \
    --output output/thermal/amundsen_seasonal_timeseries.csv

echo ""
echo "=== PIPELINE COMPLETE ==="
echo "Final output: output/thermal/amundsen_seasonal_timeseries.csv"
echo "End time: $(date)"
