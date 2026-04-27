# Task 2: 3D Brain Tumour Segmentation

## Why We Chose the BraTS 2024 GLI Dataset

For this task, we selected the **BraTS 2024 GLI (Adult Glioma) dataset** (Synapse ID: `syn59059776`). Our reasons are as follows:

1. **Coursework Compliance**: The coursework requires 3D MRI scans with corresponding segmentation masks (ground truth labels) to tackle the challenge of 3D data handling, varied tumour sizes, and class imbalance.
2. **Gold Standard**: The BraIN Tumour Segmentation (BraTS) challenge dataset is the industry gold standard for this exact problem.
3. **Relevance and Quality**: The 2024 Adult Glioma dataset is the direct successor to previous popular datasets (like BraTS 2020) but includes more recent, standardized, and high-quality multi-parametric MRI scans (T1n, T1c, T2w, T2f) with expert-annotated ground truth masks.
4. **Rich Features**: It provides multiple distinct tumour sub-regions (Necrotic Core/Non-Enhancing Tumour, Oedema, Enhancing Tumour) which allows us to train a multi-class semantic segmentation model, perfectly fulfilling the coursework's complexity requirements.

## How the Code Works

The implementation is encapsulated in `task2_brain_tumour_segmentation.py` and is built using PyTorch. Below is a breakdown of the pipeline:

### 1. Configuration & Setup (`Config` class)
All hyperparameters, paths, and settings are defined in a central `Config` class. This includes data paths, modality selection (we use T2-FLAIR and post-contrast T1, as they are the most informative), model parameters, and training settings (like epochs, learning rate, and batch size).

### 2. Synapse Data Downloading
The script includes an automated data downloading mechanism. Using `synapseclient` and `synapseutils`, it authenticates via an API token (provided via a `.env` file or CLI argument), downloads the BraTS 2024 GLI dataset, and uses `p7zip` (via a subprocess wrapper) to automatically extract the `.zip` or `.7z` archives.

### 3. Data Loading & Preprocessing (`BraTSDataset` class)
Handling 3D data requires specialized memory management. Instead of loading entire volumes into GPU memory (which would exceed typical VRAM limits), we use a **patch-based sampling approach**:
- **Volume Normalization**: Each volume undergoes Z-score normalization (ignoring zero-value background voxels).
- **Patch Extraction**: Random $96 \times 96 \times 96$ patches are extracted.
- **Foreground-Biased Sampling**: To address class imbalance, 75% of the sampled patches are forced to be centered on a tumour voxel. This ensures the model learns to identify the tumour despite it occupying a tiny fraction of the brain volume.

### 4. 3D Data Augmentation (`Augment3D` class)
To combat limited annotated data and prevent overfitting, we apply volumetric augmentations on-the-fly:
- Random axis-aligned flips
- Random 90° rotations
- Elastic deformations (simulating tissue variance using Gaussian-smoothed displacement fields)
- Intensity scaling and shifting (simulating scanner variability)

### 5. Model Architecture (`UNet3D` class)
The model is a **3D U-Net**, a state-of-the-art architecture for medical image segmentation:
- **Encoder Path**: Uses 3D convolutions and max-pooling to capture global context (where the tumour is).
- **Decoder Path**: Uses trilinear upsampling to recover spatial resolution.
- **Skip Connections**: Concatenates high-resolution features from the encoder with the decoder to preserve fine-grained boundaries.
- **Regularization**: Incorporates Batch Normalization and 3D Dropout to stabilize training and improve generalization.
- **Transfer Learning**: Includes a helper function to inflate pretrained 2D weights into 3D kernels if needed.

### 6. Loss Functions
To handle severe class imbalance, we use a **Combined Loss** function:
- **Soft Dice Loss** (60% weight): Directly optimizes the overlap between predictions and ground truth, treating rare classes equally to common ones.
- **Cross-Entropy Loss** (40% weight): Provides stable, per-voxel gradient signals during early training.

### 7. Training & Evaluation
- **Mixed Precision (AMP)**: Uses `torch.cuda.amp` to reduce VRAM usage and speed up training.
- **Hyperparameter Search**: A lightweight grid search mechanism tests different learning rates, base filters, and loss weights, selecting the best configuration based on validation Dice score.
- **Metrics**: Evaluates performance using standard segmentation metrics: Per-class **Dice Score**, **IoU (Jaccard Index)**, and **Hausdorff Distance**.
- **Early Stopping & Checkpointing**: Halts training if validation metrics stop improving and saves the best model weights.

### 8. Visualizations
The script generates several outputs in the `outputs/` directory:
- `results/training_curves.png`: Loss, Dice, and Learning Rate curves.
- `results/dice_per_class.png`: Bar chart of performance across different tumour regions.
- `sample_predictions/`: Side-by-side axial slice comparisons of the input FLAIR, Ground Truth, and Model Prediction.
