#!/bin/bash
# =============================================================================
# Amundsen Crater Seasonal Shadow Survey - Phase 1
# Runs illumination-only simulations for 25 lunations (April 2026 - April 2028)
# =============================================================================

set -e

# Paths
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BIN="${BUILD_DIR}/bin/ptm_thermal_sim"
TEMPLATE="${PROJECT_DIR}/configs/amundsen_survey_template.yaml"
OUTPUT_BASE="${PROJECT_DIR}/output/survey"

# Lunation parameters
START_EPOCH="2026-04-01T00:00:00"
LUNATION_DAYS=29.530588
NUM_LUNATIONS=25

# Check prerequisites
if [ ! -f "$BIN" ]; then
    echo "ERROR: ptm_thermal_sim not found at $BIN"
    echo "Build first: cd $BUILD_DIR && /usr/local/bin/cmake --build . --target ptm_thermal_sim -j4"
    exit 1
fi
if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Template config not found at $TEMPLATE"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

# Python helper to compute lunation start dates
# Uses the datetime module to add fractional days to the start epoch
compute_start_dates() {
    python3 -c "
from datetime import datetime, timedelta
start = datetime(2026, 4, 1, 0, 0, 0)
lunation = $LUNATION_DAYS
for i in range($NUM_LUNATIONS):
    dt = start + timedelta(days=i * lunation)
    print(dt.strftime('%Y-%m-%dT%H:%M:%S'))
"
}

echo "=== Amundsen Crater Seasonal Shadow Survey ==="
echo "Start epoch: $START_EPOCH"
echo "Lunation period: $LUNATION_DAYS days"
echo "Number of lunations: $NUM_LUNATIONS"
echo "Output base: $OUTPUT_BASE"
echo ""

# Generate all start dates
DATES=($(compute_start_dates))

for i in $(seq 0 $((NUM_LUNATIONS - 1))); do
    LUNATION_NUM=$(printf "%02d" $i)
    START_TIME="${DATES[$i]}"
    OUTPUT_DIR="${OUTPUT_BASE}/lunation_${LUNATION_NUM}"
    CONFIG_FILE="${OUTPUT_BASE}/lunation_${LUNATION_NUM}.yaml"

    echo "--- Lunation $LUNATION_NUM: $START_TIME ---"

    # Generate config from template
    mkdir -p "$OUTPUT_DIR"
    sed -e "s|{START_TIME}|${START_TIME}|g" \
        -e "s|{OUTPUT_DIR}|${OUTPUT_DIR}|g" \
        "$TEMPLATE" > "$CONFIG_FILE"

    # Run simulation
    echo "  Running illumination survey..."
    "$BIN" --config "$CONFIG_FILE" 2>&1 | tail -5

    # Verify output
    if [ -f "${OUTPUT_DIR}/statistics/direct_solar_avg.bin" ]; then
        echo "  OK: Statistics written to ${OUTPUT_DIR}/statistics/"
    else
        echo "  WARNING: Expected output not found in ${OUTPUT_DIR}/statistics/"
    fi
    echo ""
done

echo "=== Survey complete. Run identify_seasonal_shadow.py to classify faces. ==="
echo "  python3 ${PROJECT_DIR}/scripts/identify_seasonal_shadow.py \\"
echo "    --survey-dir ${OUTPUT_BASE} \\"
echo "    --mesh ${PROJECT_DIR}/../dtm/LDEM_80S_20M__amundsen_2Mpts.ply \\"
echo "    --num-lunations ${NUM_LUNATIONS}"
