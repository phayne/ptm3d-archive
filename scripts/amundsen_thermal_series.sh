#!/bin/bash
# =============================================================================
# Amundsen Crater Thermal + Radiation Series - Phase 2
# Runs thermal+radiation simulations for 25 lunations with location output
# =============================================================================

set -e

# Paths
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BIN="${BUILD_DIR}/bin/ptm_thermal_sim"
TEMPLATE="${PROJECT_DIR}/configs/amundsen_phase2_template.yaml"
OUTPUT_BASE="${PROJECT_DIR}/output/thermal"
LOCATIONS_FILE="${PROJECT_DIR}/configs/amundsen_target_locations.csv"
VIEW_FACTOR_FILE="${PROJECT_DIR}/output/amundsen_vf.hmat"

# Lunation parameters
LUNATION_DAYS=29.530588
NUM_LUNATIONS=23

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
if [ ! -f "$LOCATIONS_FILE" ]; then
    echo "ERROR: Locations file not found at $LOCATIONS_FILE"
    echo "Run Phase 1 first: ./scripts/amundsen_seasonal_survey.sh"
    echo "Then: python3 scripts/identify_seasonal_shadow.py ..."
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

# Compute start dates (same as Phase 1)
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

echo "=== Amundsen Crater Thermal + Radiation Series ==="
echo "Locations file: $LOCATIONS_FILE"
echo "View factor file: $VIEW_FACTOR_FILE"
echo "Number of lunations: $NUM_LUNATIONS"
echo ""

# Step 0: Compute view factors if not already done
if [ ! -f "$VIEW_FACTOR_FILE" ]; then
    echo "--- Computing view factors (first run only) ---"
    DATES=($(compute_start_dates))
    VF_CONFIG="${OUTPUT_BASE}/viewfactor_compute.yaml"
    VF_OUTPUT="${OUTPUT_BASE}/vf_compute"
    mkdir -p "$VF_OUTPUT"
    sed -e "s|{START_TIME}|${DATES[0]}|g" \
        -e "s|{OUTPUT_DIR}|${VF_OUTPUT}|g" \
        -e "s|{VIEW_FACTOR_FILE}|${VIEW_FACTOR_FILE}|g" \
        -e "s|{LOCATIONS_FILE}|${LOCATIONS_FILE}|g" \
        "$TEMPLATE" > "$VF_CONFIG"
    echo "  This may take a long time for 2M faces..."
    "$BIN" --config "$VF_CONFIG" 2>&1 | tail -10
    if [ -f "$VIEW_FACTOR_FILE" ]; then
        echo "  View factors saved to: $VIEW_FACTOR_FILE"
    else
        echo "  ERROR: View factor computation failed"
        exit 1
    fi
    echo ""
fi

# Generate all start dates
DATES=($(compute_start_dates))

for i in $(seq 0 $((NUM_LUNATIONS - 1))); do
    LUNATION_NUM=$(printf "%02d" $i)
    START_TIME="${DATES[$i]}"
    OUTPUT_DIR="${OUTPUT_BASE}/lunation_${LUNATION_NUM}"
    CONFIG_FILE="${OUTPUT_BASE}/lunation_${LUNATION_NUM}.yaml"

    echo "--- Lunation $LUNATION_NUM: $START_TIME ---"

    # Skip if output already exists
    if [ -f "${OUTPUT_DIR}/location_timeseries.csv" ]; then
        echo "  SKIP: Output already exists at ${OUTPUT_DIR}/location_timeseries.csv"
        echo ""
        continue
    fi

    # Generate config from template
    mkdir -p "$OUTPUT_DIR"
    sed -e "s|{START_TIME}|${START_TIME}|g" \
        -e "s|{OUTPUT_DIR}|${OUTPUT_DIR}|g" \
        -e "s|{VIEW_FACTOR_FILE}|${VIEW_FACTOR_FILE}|g" \
        -e "s|{LOCATIONS_FILE}|${LOCATIONS_FILE}|g" \
        "$TEMPLATE" > "$CONFIG_FILE"

    # Run simulation
    echo "  Running thermal + radiation..."
    "$BIN" --config "$CONFIG_FILE" 2>&1 | tail -5

    # Verify output
    if [ -f "${OUTPUT_DIR}/location_timeseries.csv" ]; then
        echo "  OK: Time series written to ${OUTPUT_DIR}/location_timeseries.csv"
    else
        echo "  WARNING: Expected output not found in ${OUTPUT_DIR}/"
    fi
    echo ""
done

echo "=== Thermal series complete. Run format_seasonal_timeseries.py to combine. ==="
echo "  python3 ${PROJECT_DIR}/scripts/format_seasonal_timeseries.py \\"
echo "    --thermal-dir ${OUTPUT_BASE} \\"
echo "    --locations ${LOCATIONS_FILE} \\"
echo "    --num-lunations ${NUM_LUNATIONS} \\"
echo "    --output ${OUTPUT_BASE}/amundsen_seasonal_timeseries.csv"
