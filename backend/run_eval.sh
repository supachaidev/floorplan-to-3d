#!/usr/bin/env bash
#
# Automated evaluation pipeline for floorplan detection.
#
# Downloads CubiCasa5K dataset, converts samples to ground truth,
# and runs accuracy evaluation.
#
# Usage:
#   ./run_eval.sh              # default: 5 samples, OpenCV mode
#   ./run_eval.sh --limit 20   # convert 20 samples
#   ./run_eval.sh --claude      # use Claude Vision API
#   ./run_eval.sh --shuffle          # pick random samples each run
#   ./run_eval.sh --limit 10 --claude --output results.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CUBICASA_DIR="$SCRIPT_DIR/.cubicasa"
DATASET_DIR="$CUBICASA_DIR/CubiCasa5k"
FIXTURES_DIR="$SCRIPT_DIR/tests/fixtures"
LIMIT=5
USE_CLAUDE=false
SHUFFLE=false
OUTPUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --claude)
            USE_CLAUDE=true
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --shuffle)
            SHUFFLE=true
            shift
            ;;
        --clean)
            echo "Cleaning up downloaded dataset and fixtures..."
            rm -rf "$CUBICASA_DIR" "$FIXTURES_DIR"
            echo "Done."
            exit 0
            ;;
        --help|-h)
            echo "Usage: ./run_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --limit N     Number of samples to evaluate (default: 5)"
            echo "  --claude      Use Claude Vision API instead of OpenCV"
            echo "  --shuffle     Randomly select different samples each run"
            echo "  --output FILE Save detailed results to JSON file"
            echo "  --clean       Remove downloaded dataset and fixtures"
            echo "  --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo " Floorplan Detection Evaluation Pipeline"
echo "=========================================="
echo ""

# Step 1: Check Python dependencies
echo "[1/4] Checking dependencies..."
python3 -c "import cv2, numpy" 2>/dev/null || {
    echo "  Installing Python dependencies..."
    pip install -r requirements.txt -q
}
echo "  OK"
echo ""

# Step 2: Download CubiCasa5K if not present
ZENODO_URL="https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
ZIP_PATH="$CUBICASA_DIR/cubicasa5k.zip"

echo "[2/4] Preparing CubiCasa5K dataset..."

# Check if we already have extracted data with model.svg files
SAMPLE_COUNT=$([ -d "$CUBICASA_DIR" ] && find "$CUBICASA_DIR" -name "model.svg" 2>/dev/null | wc -l | tr -d ' ' || echo 0)

if [ "$SAMPLE_COUNT" -gt 0 ]; then
    echo "  Dataset already available ($SAMPLE_COUNT samples found)"
else
    mkdir -p "$CUBICASA_DIR"

    # Download if zip doesn't exist
    if [ ! -f "$ZIP_PATH" ]; then
        echo "  Downloading CubiCasa5K dataset (5.5 GB)..."
        echo "  This may take a while depending on your connection."
        echo ""
        curl -L -o "$ZIP_PATH" --progress-bar "$ZENODO_URL"
        echo ""
    fi

    # Extract
    echo "  Extracting dataset..."
    if ! unzip -q -o "$ZIP_PATH" -d "$CUBICASA_DIR"; then
        echo ""
        echo "  Error: Failed to extract zip file. It may be corrupted or incomplete."
        echo "  Remove the broken file and re-run:"
        echo ""
        echo "    rm $ZIP_PATH"
        echo "    ./run_eval.sh"
        echo ""
        exit 1
    fi

    # Verify extraction
    SAMPLE_COUNT=$([ -d "$CUBICASA_DIR" ] && find "$CUBICASA_DIR" -name "model.svg" 2>/dev/null | wc -l | tr -d ' ' || echo 0)
    if [ "$SAMPLE_COUNT" -eq 0 ]; then
        echo "  Error: extraction failed — no model.svg files found."
        echo "  Check $CUBICASA_DIR for the extracted contents."
        exit 1
    fi
    echo "  Found $SAMPLE_COUNT sample(s)"

    # Remove zip to save disk space
    rm -f "$ZIP_PATH"
    echo "  Removed zip file to save space"
fi
echo ""

# Step 3: Convert samples
echo "[3/4] Converting $LIMIT sample(s) to test fixtures..."

# Find the data directory (could be nested differently after extraction)
DATA_DIR=""
for candidate in "$CUBICASA_DIR/cubicasa5k" "$CUBICASA_DIR/CubiCasa5k/data" "$CUBICASA_DIR/data" "$CUBICASA_DIR"; do
    if [ -d "$candidate" ] && find "$candidate" -maxdepth 4 -name "model.svg" -print -quit 2>/dev/null | grep -q .; then
        DATA_DIR="$candidate"
        break
    fi
done

if [ -z "$DATA_DIR" ]; then
    echo "  Error: could not locate model.svg files in $CUBICASA_DIR"
    exit 1
fi

CONVERT_ARGS=("--cubicasa-dir" "$DATA_DIR" "--output-dir" "$FIXTURES_DIR" "--limit" "$LIMIT")
if [ "$SHUFFLE" = true ]; then
    CONVERT_ARGS+=("--shuffle")
fi
python3 convert_cubicasa.py "${CONVERT_ARGS[@]}"

# Check we have fixtures
FIXTURE_COUNT=$(find "$FIXTURES_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
if [ "$FIXTURE_COUNT" -eq 0 ]; then
    echo "  No test fixtures generated. Check the dataset."
    exit 1
fi
echo ""

# Step 4: Run evaluation
echo "[4/4] Running evaluation..."
echo ""

EVAL_ARGS=("--test-dir" "$FIXTURES_DIR")

if [ "$USE_CLAUDE" = true ]; then
    EVAL_ARGS+=("--force-claude")
fi

if [ -n "$OUTPUT" ]; then
    EVAL_ARGS+=("--output" "$OUTPUT")
fi

python3 evaluate.py "${EVAL_ARGS[@]}"
