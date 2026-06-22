#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

echo "===================================================="
echo "Starting Blur Detection Batch Processor Setup..."
echo "===================================================="

# 1. Synchronize Git submodules if not fully recursed
if [ ! -f "sam3/README.md" ] && [ ! -f "dinov3/README.md" ]; then
    echo "Initializing submodules (SAM3 & DINOv3)..."
    git submodule update --init --recursive
fi

# 2. Build the unified Conda environment
echo "Creating Conda environment from conda.yaml..."
conda env create -f conda.yaml --yes

# 3. Create the assets directory structure for model weights
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
echo "   conda activate blur_detection && cd blur_detection && streamlit run main.py"
echo "===================================================="