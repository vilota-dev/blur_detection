# Blur Detection Batch Processor

This repository provides a single-file Streamlit app that:
- Uses SAM3 for segmentation (prompted with "building").
- Applies directional cropping and CLAHE as needed.
- Uses a DINOv3-based classifier head to predict blur labels per image position.
- Exports full consolidated CSV/XLSX and filtered CSV/XLSX, and copies filtered images to a folder.

How to run:
1. Create conda env from `environment.yml` (you may need to adjust dinov3 / sam3 install URLs):

```bash
conda env create -f environment.yml
conda activate blur_detection
pip install -r requirements.txt  # optional
streamlit run app.py
```

2. In the Streamlit UI, set the input folder, output folder, filtered images folder, and path to the trained classifier head (.pth). Optionally set backbone weights path and SAM3 checkpoint in `config.yaml`.

Notes:
- The app expects image filenames to follow `SN-pos` pattern (e.g., `1359-3.jpg`).
- Filtering thresholds and other constants are editable in `config.yaml`.
