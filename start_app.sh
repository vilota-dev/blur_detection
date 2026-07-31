#!/usr/bin/env bash

# Exit immediately if any tracking check fails critically
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===================================================="
echo "Validating Blur Detection Pipeline Deployment..."
echo "===================================================="

# 1. Check if Git submodules (SAM3 and DINOv3) exist
if [ ! -f "sam3/README.md" ] || [ ! -f "dinov3/README.md" ]; then
    echo "CRITICAL: Git submodules (SAM3/DINOv3) are missing or incomplete."
    echo "   Please execute the setup script first:"
    echo "   chmod +x setup.sh && ./setup.sh"
    exit 1
fi

# 2. Check if the setup script was run by verifying the assets folder existence
if [ ! -d "blur_detection/assets" ]; then
    echo "CRITICAL: The application has not been initialized yet."
    echo "   Please execute the setup script first:"
    echo "   chmod +x setup.sh && ./setup.sh"
    exit 1
fi

# 3. Check for missing neural network weights inside the assets directory
REQUIRED_WEIGHTS=(
    "blur_detection/assets/sam3.pt"
    "blur_detection/assets/bpe_simple_vocab_16e6.txt.gz"
    "blur_detection/assets/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    "blur_detection/assets/dino_classifier_head_multiclass_epoch_58.pth"
)

MISSING_WEIGHTS=0
for weight in "${REQUIRED_WEIGHTS[@]}"; do
    if [ ! -f "$weight" ]; then
        echo "MISSING WEIGHT: Could not find $(basename "$weight")"
        MISSING_WEIGHTS=$((MISSING_WEIGHTS + 1))
    fi
done

if [ "$MISSING_WEIGHTS" -gt 0 ]; then
    echo ""
    echo "CRITICAL: $MISSING_WEIGHTS model file(s) are missing from 'blur_detection/assets/'."
    echo "   Please download them from the Sharepoint link provided in the README"
    echo "   and place them in the assets folder before running this script."
    exit 1
fi

# 4. Verify config.yaml exists
if [ ! -f "blur_detection/config.yaml" ]; then
    echo "CRITICAL: 'blur_detection/config.yaml' is missing."
    exit 1
fi

echo "Environment validation passed! Initializing Streamlit UI..."
echo "===================================================="

# 5. Initialize shell workspace for Conda commands safely
eval "$(conda shell.bash hook)"

# 6. Activate environment and boot up the server core
conda activate blur_detection
cd blur_detection
streamlit run main.py