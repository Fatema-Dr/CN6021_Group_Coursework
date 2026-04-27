# Task 2: 3D Brain Tumour Segmentation

## Why We Chose the BraTS 2024 GLI Dataset

For this task, we selected the **BraTS 2024 GLI (Adult Glioma) dataset** (Synapse ID: `syn59059776`). Our reasons are as follows:

1. **Coursework Compliance**: The coursework requires 3D MRI scans with corresponding segmentation masks (ground truth labels) to tackle the challenge of 3D data handling, varied tumour sizes, and class imbalance.
2. **Gold Standard**: The BraIN Tumour Segmentation (BraTS) challenge dataset is the industry gold standard for this exact problem.
3. **Relevance and Quality**: The 2024 Adult Glioma dataset is the direct successor to previous popular datasets (like BraTS 2020) but includes more recent, standardized, and high-quality multi-parametric MRI scans (T1n, T1c, T2w, T2f) with expert-annotated ground truth masks.
4. **Rich Features**: It provides multiple distinct tumour sub-regions (Necrotic Core/Non-Enhancing Tumour, Oedema, Enhancing Tumour) which allows us to train a multi-class semantic segmentation model, perfectly fulfilling the coursework's complexity requirements.

## Pipeline Overview

```
python task2_brain_tumour_segmentation.py --auth-token YOUR_SYNAPSE_TOKEN
```

All code is encapsulated in a single file: `task2_brain_tumour_segmentation.py`

---

## Step-by-Step Pipeline Flow

```
STEP 1 — Exploratory Data Analysis (EDA)
    ├── Download BraTS 2024 data via Synapse (if needed)
    ├── Skip download if data already exists
    ├── Analyze patient volumes (shapes, intensities, class distributions)
    └── Outputs: eda_*.png, eda_patient_statistics.csv

STEP 2 — Building datasets
    ├── Load BraTS patient directories
    ├── Patch-based sampling (96×96×96)
    ├── Z-score normalization (ignoring zero voxels)
    └── Foreground-biased sampling (75% tumor-centered)

STEP 3 — Hyperparameter search (optional with --no_hp_search)
    ├── Grid search over LR, base_filters, dice_weight
    └── Best config selected by validation Dice

STEP 4 — Building 3D U-Net
    ├── 3D convolutions with encoder/decoder + skip connections
    ├── Batch Normalization + Dropout
    └── Transfer learning helper (2D→3D weight inflation)

STEP 5 — Training
    ├── Combined Dice (60%) + Cross-Entropy (40%) loss
    ├── AMP mixed precision
    ├── Early stopping + checkpointing
    └── CosineAnnealing LR scheduler

STEP 6 — Loading best checkpoint

STEP 7 — Test evaluation
    └── Per-class Dice Score, IoU, Hausdorff Distance

STEP 8 — Saving visualisations
    ├── training_curves.png
    ├── dice_per_class.png
    └── sample_predictions/*.png
```

---

## How the Code Works

### 1. Configuration & Setup (`Config` class)
All hyperparameters, paths, and settings are defined in a central `Config` class. This includes data paths, modality selection (we use T2-FLAIR and post-contrast T1, as they are the most informative), model parameters, and training settings (like epochs, learning rate, and batch size).

**GPU/CPU Detection**: The device is automatically selected based on `torch.cuda.is_available()`:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### 2. Synapse Data Downloading
The script includes an automated data downloading mechanism. Using `synapseclient` and `synapseutils`, it authenticates via an API token (provided via a `.env` file or CLI argument), downloads the BraTS 2024 GLI dataset, and uses `p7zip` (via a subprocess wrapper) to automatically extract the `.zip` or `.7z` archives.

**Smart Download**: The `_data_exists()` function checks if data is already present before downloading:
```python
def _data_exists(cfg: Config) -> bool:
    if not cfg.DATA_ROOT.exists():
        return False
    patient_dirs = [d for d in cfg.DATA_ROOT.iterdir() if d.is_dir() and "BraTS" in d.name]
    return len(patient_dirs) > 0
```

### 3. Exploratory Data Analysis (EDA)
Before training, we perform comprehensive EDA on the dataset:
- **Patient Statistics**: Volume shapes, intensity means/stds, class voxel counts
- **Class Distribution**: Mean voxel ratios for background, necrotic core, oedema, enhancing
- **Modality Statistics**: T2-FLAIR and T1c intensity scatter plots and histograms
- **Volume Shapes**: D/H/W dimension distributions
- **Tumor Size Distribution**: Total and per-class tumor volume histograms
- **Sample Slices**: Axial slices with ground truth overlays

EDA outputs saved to `outputs/results/`:
- `eda_class_distribution.png`
- `eda_modality_statistics.png`
- `eda_volume_shapes.png`
- `eda_tumor_size_distribution.png`
- `eda_sample_slices.png`
- `eda_patient_statistics.csv`

### 4. Data Loading & Preprocessing (`BraTSDataset` class)
Handling 3D data requires specialized memory management. Instead of loading entire volumes into GPU memory (which would exceed typical VRAM limits), we use a **patch-based sampling approach**:
- **Volume Normalization**: Each volume undergoes Z-score normalization (ignoring zero-value background voxels).
- **Patch Extraction**: Random $96 \times 96 \times 96$ patches are extracted.
- **Foreground-Biased Sampling**: To address class imbalance, 75% of the sampled patches are forced to be centered on a tumour voxel. This ensures the model learns to identify the tumour despite it occupying a tiny fraction of the brain volume.

### 5. 3D Data Augmentation (`Augment3D` class)
To combat limited annotated data and prevent overfitting, we apply volumetric augmentations on-the-fly:
- Random axis-aligned flips
- Random 90° rotations
- Elastic deformations (simulating tissue variance using Gaussian-smoothed displacement fields)
- Intensity scaling and shifting (simulating scanner variability)

### 6. Model Architecture (`UNet3D` class)
The model is a **3D U-Net**, a state-of-the-art architecture for medical image segmentation:
- **Encoder Path**: Uses 3D convolutions and max-pooling to capture global context (where the tumour is).
- **Decoder Path**: Uses trilinear upsampling to recover spatial resolution.
- **Skip Connections**: Concatenates high-resolution features from the encoder with the decoder to preserve fine-grained boundaries.
- **Regularization**: Incorporates Batch Normalization and 3D Dropout to stabilize training and improve generalization.
- **Transfer Learning**: Includes a helper function to inflate pretrained 2D weights into 3D kernels if needed.

### 7. Loss Functions
To handle severe class imbalance, we use a **Combined Loss** function:
- **Soft Dice Loss** (60% weight): Directly optimizes the overlap between predictions and ground truth, treating rare classes equally to common ones.
- **Cross-Entropy Loss** (40% weight): Provides stable, per-voxel gradient signals during early training.

### 8. Training & Evaluation
- **Mixed Precision (AMP)**: Uses `torch.cuda.amp` to reduce VRAM usage and speed up training.
- **Hyperparameter Search**: A lightweight grid search mechanism tests different learning rates, base filters, and loss weights, selecting the best configuration based on validation Dice score.
- **Metrics**: Evaluates performance using standard segmentation metrics: Per-class **Dice Score**, **IoU (Jaccard Index)**, and **Hausdorff Distance**.
- **Early Stopping & Checkpointing**: Halts training if validation metrics stop improving and saves the best model weights.

### 9. Visualizations
The script generates several outputs in the `outputs/` directory:
- `results/training_curves.png`: Loss, Dice, and Learning Rate curves.
- `results/dice_per_class.png`: Bar chart of performance across different tumour regions.
- `sample_predictions/`: Side-by-side axial slice comparisons of the input FLAIR, Ground Truth, and Model Prediction.

---

## Requirements Checklist

| Requirement | Status |
|-------------|--------|
| Download data from Synapse with API key | ✅ |
| Unzip using p7zip | ✅ |
| Skip download if data exists | ✅ |
| GPU/CPU auto-detection | ✅ |
| EDA with visualizations | ✅ |
| 3D U-Net architecture | ✅ |
| Z-score normalization | ✅ |
| Patch extraction | ✅ |
| Foreground-biased sampling | ✅ |
| 3D augmentation | ✅ |
| Dice + CE combined loss | ✅ |
| AMP mixed precision | ✅ |
| Early stopping | ✅ |
| Checkpointing | ✅ |
| Hyperparameter search | ✅ |
| Dice, IoU, Hausdorff metrics | ✅ |

---

## Output Files

All outputs are written to the `outputs/` directory:

```
outputs/
├── checkpoints/
│   ├── best_model.pth      # Best validation-Dice checkpoint
│   └── final_model.pth     # Weights after all epochs
├── results/
│   ├── training_curves.png
│   ├── dice_per_class.png
│   ├── eda_*.png           # EDA visualizations
│   ├── eda_patient_statistics.csv
│   ├── metrics_summary.csv
│   ├── hyperparameter_search.csv
│   └── model_summary.txt
└── sample_predictions/
    └── sample_XX.png       # Prediction visualizations
```

---

## Usage

```bash
# Full pipeline with download
python task2_brain_tumour_segmentation.py --auth-token YOUR_SYNAPSE_TOKEN

# Skip EDA
python task2_brain_tumour_segmentation.py --auth-token YOUR_SYNAPSE_TOKEN --skip-eda

# Run EDA only
python task2_brain_tumour_segmentation.py --auth-token YOUR_SYNAPSE_TOKEN --eda-only

# Skip hyperparameter search
python task2_brain_tumour_segmentation.py --auth-token YOUR_SYNAPSE_TOKEN --no_hp_search

# Override epochs
python task2_brain_tumour_segmentation.py --auth-token YOUR_SYNAPSE_TOKEN --epochs 100

# Evaluate existing checkpoint
python task2_brain_tumour_segmentation.py --eval --checkpoint outputs/checkpoints/best_model.pth
```