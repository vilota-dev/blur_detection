#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===================================================="
echo "Starting Blur Detection Batch Processor Setup..."
echo "===================================================="

# 1. Synchronize Git submodules if either SAM3 or DINOv3 is missing
if [ ! -f "sam3/README.md" ] || [ ! -f "dinov3/README.md" ]; then
    echo "Initializing submodules (SAM3 & DINOv3)..."
    git submodule update --init --recursive
fi

# 2. Initialize shell workspace for Conda commands safely
eval "$(conda shell.bash hook)"

# 3. Build or update the unified Conda environment
echo "Creating/Updating Conda environment from conda.yaml..."
conda env create -f conda.yaml || conda env update -f conda.yaml --prune

# 4. Install submodules in editable mode so imports work seamlessly
echo "Installing SAM3 and DINOv3 submodules..."
conda activate blur_detection
pip install -e sam3 --no-deps || pip install -e sam3
pip install -e dinov3 --no-deps || pip install -e dinov3

# 5. Create the assets directory structure for model weights
echo "Creating assets directory for model checkpoints..."
mkdir -p blur_detection/assets

echo "----------------------------------------------------"
echo "Base software layer setup complete!"
echo "----------------------------------------------------"
echo "CRITICAL NEXT STEP: You must manually download the"
echo "   neural network weights and place them into:"
echo "   blur_detection/assets/"
echo ""
echo " Required weights filenames:"
echo "  1. sam3.pt"
echo "  2. bpe_simple_vocab_16e6.txt.gz"
echo "  3. dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
echo "  4. dino_classifier_head_multiclass_epoch_58.pth"
echo "----------------------------------------------------"
echo "To start the app later, run:"
echo "   chmod +x start_app.sh && ./start_app.sh"
echo "===================================================="