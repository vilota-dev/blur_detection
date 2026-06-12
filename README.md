# Blur Detection Batch Processor

An automated, batch-processing computer vision pipeline that combines **Meta's SAM3** (Segment Anything 3) for zero-shot object segmentation and **DINOv3** for high-precision blur and defect classification. 

This tool is wrapped in a user-friendly **Streamlit** interface, allowing users to select input directories, process thousands of images, and automatically generate filtered reports and sorted image datasets based on customizable confidence thresholds.

---

## 📂 Repository Structure

```text
blur_detection_repo/
├── blur_detection/             # Main application directory
│   ├── main.py                 # Streamlit entry point
│   ├── config.yaml             # Pipeline and threshold configuration
│   ├── core/                   # Orchestration (pipeline.py) and config loaders
│   ├── models/                 # ML wrappers (sam3_wrapper.py, dino_classifier.py)
│   ├── utils/                  # Stateless helpers (image_utils.py, file_utils.py)
│   └── ui/                     # Streamlit UI Components (batch_view.py, review_view.py, dev_view.py)
├── dinov3/                     # Git submodule: Meta DINOv3 backbone
├── sam3/                       # Git submodule: Meta SAM3 segmenter
├── models/                     # Directory for model weights (Requires manual download)
├── conda.yaml                  # Unified Conda environment dependencies
└── README.md                   # Project documentation
```

---

## 💻 Hardware & Software Requirements

* **OS:** Linux (Tested on Ubuntu)
* **GPU:** CUDA-compatible NVIDIA GPU with **CUDA 12.6 or higher**. Modern architecture (Ampere, Hopper, Blackwell) is strongly recommended, as the pipeline utilizes `bfloat16` and `TF32` math optimizations.
* **Environment Manager:** **Conda** ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda) is required to resolve specific PyTorch nightly builds and C++ dependencies.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
Because this repository relies on external Meta repositories for the vision models, you **must** clone it with the `--recurse-submodules` flag to fetch SAM3 and DINOv3.

```bash
git clone --recurse-submodules https://github.com/vilota-dev/blur_detection.git
cd blur_detection_repo
```
*(If you already cloned it without submodules, run: `git submodule update --init --recursive`)*

### 2. Download Model Weights
The neural network weights are too large to host on GitHub. You must download them manually from [Models Link](https://vilota.sharepoint.com/:f:/r/sites/allcompany/Shared%20Documents/Production/Shiva%20Production/4.QC%20Check/QC%20evidence/For%20Shiva%20V2%20-%20outdoor%20FLC/blur%20detection%20models?csf=1&web=1&e=yGQnSo) and place them inside the `blur_detection/blur_detection/assets/` directory.

**Required Files:**
1. `sam3.pt` (SAM3 Checkpoint)
2. `bpe_simple_vocab_16e6.txt.gz` (SAM3 BPE Vocab)
3. `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` (DINOv3 Backbone)
4. `dino_classifier_head_multiclass_epoch_58.pth` (Custom Trained Classification Head)

Place all four files directly into the root `blur_detection/assets/` folder.

### 3. Build the Conda Environment
The `conda.yaml` file contains a highly tuned environment specifically built to resolve dependency conflicts between PyTorch nightlies, SAM3's strict NumPy constraints, and OpenCV.

```bash
# Create the environment
conda env create -f conda.yaml

# Activate the environment
conda activate blur_detection
```

---

## ⚠️ Important: Image Naming Convention

The application relies on a strict filename pattern to parse data correctly. **All input images must follow the `SN-pos` format** (Serial Number followed by a hyphen and the Grid Position). 

* **Correct:** `1359-3.jpg` (SN is 1359, Position is 3)
* **Correct:** `4582A-9.bmp` (SN is 4582A, Position is 9)
* **Incorrect:** `image_3.jpg`, `1359_pos3.png`

If images do not follow this convention, the position parsing will fail, and the batch processor will not be able to categorize the outputs correctly.

---

## ⚙️ Configuration (`config.yaml`)

The `blur_detection/config.yaml` file controls the behavior of the crop logic, model confidence gates, and reporting outputs. 

### Key Parameters:
* **`default_input`**: The default folder path the Streamlit UI will look at for raw images.
* **`default_output_root`**: The default root folder path where all organized outputs will be saved.
* **`target_size`**: The `[width, height]` of the final image fed to the DINO classifier. **Do not change this**, as the custom head was trained on this exact spatial dimension.
* **`position_shifts`**: Maps grid cell positions (1, 3, 5, 7, 9) to specific `[dx, dy]` coordinate shifts. Allows you to fine-tune where the crop window anchors relative to the segmented building. 
    * *Negative `dx` shifts left, positive `dx` shifts right.*
* **`confidence_gate`**: The minimum probability threshold (e.g., `0.9`) required to pass an image as `o` (OK). If below this, the image is flagged for review under its highest defect class.

### Advanced Filtering Logic:
The pipeline automatically flags any Serial Number (SN) containing an `n` (Noisy) prediction. You can control how strictly it handles `sn` (Slightly Noisy) predictions:
* **`sn_single_threshold_percent`**: If *any single position* in an SN is flagged as `sn` with a confidence higher than this percentage, the whole SN is pulled for review.
* **`sn_count_threshold_percent` & `sn_count_required`**: (Optional) If an SN gets multiple `sn` predictions, it will be flagged if it hits the required count at the specified confidence level. *(Remove these lines from the YAML to disable multiple-count tracking).*

---

## 🚀 How to Run

Ensure your conda environment is active, navigate to the inner application folder, and launch Streamlit:

```bash
conda activate blur_detection
cd blur_detection
streamlit run main.py
```

1. A web browser will automatically open (usually at `http://localhost:8501`).
2. Use the UI sidebar to verify your Input Folder, Output Folder, and Filtered Images destinations.
3. Click **"Run Batch"**.
4. The UI will display a live progress bar.

---

## 📊 Outputs

Upon completion, the application automatically builds a clean, standardized data layout inside your designated **Result Output Root Folder**:

### 1. Dataset Output Subfolder (`/dataset_output/`)
* **Consolidated Reports** (`consolidated_batch_predictions.csv` / `.xlsx`): Contains raw predictions, confidence percentages, and recommended actions for *every single image* processed in the batch.
* **Filtered Reports** (`predictions_filtered.csv` / `.xlsx`): A strictly filtered ledger containing *only* the Serial Numbers that require human review or flagged FLC validation based on your threshold rules.

### 2. Processed Images Subfolder (`/processed_images/`)
The pipeline automatically generates isolated subfolders named by each unit's **Serial Number (SN)**. Every unit folder contains:
* The processed crop files generated from active position configurations.
* A `metadata.json` document containing the model predictions, confidence levels, and staging tracks for human annotation tools.

---

## 🧑‍💻 Development & Training

If you wish to modify the pipeline, retrain the classifier on new data, or evaluate results programmatically, refer to these development scripts located within the submodules:

* **SAM3 Image Processor (Cropping & Masking)**
  Used to generate the training datasets and run standalone predictions.
  [sam3/building_seg/image_processor_for_dinov3.py](https://github.com/vilota-dev/sam3.git)
* **DINOv3 Classifier Training**
  Jupyter notebook for fine-tuning the DINOv3 multi-class head.
  [dinov3/scripts/dinov3_blur_detection_trainer_v5.ipynb](https://github.com/vilota-dev/dinov3.git)
* **Standalone Predictor**
  Command-line script for running DINOv3 predictions outside of Streamlit.
  [dinov3/scripts/dinov3_blur_detection.py](https://github.com/vilota-dev/dinov3.git)
* **Result Evaluation**
  Notebook for merging batch results and checking accuracy against ground truth.
  [dinov3/scripts/merger_checker.ipynb](https://github.com/vilota-dev/dinov3.git)