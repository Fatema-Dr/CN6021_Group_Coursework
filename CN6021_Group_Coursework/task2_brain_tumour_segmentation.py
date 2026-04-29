"""
task2_brain_tumour_segmentation.py
===================================
CN6021 — Advanced Topics in AI and Data Science
Task 2: 3D Brain Tumour Segmentation from MRI Scans

Architecture : 3D U-Net (encoder–decoder with skip connections)
Framework    : PyTorch
Dataset      : BraTS 2024 GLI (Adult Glioma) — downloaded via Synapse API
               Synapse ID: syn59059776
               Expected layout (auto-created by download step):
                 data/
                   BraTS2024_GLI/
                     BraTS-GLI-XXXXX-XXX/
                       BraTS-GLI-XXXXX-XXX-t1n.nii.gz
                       BraTS-GLI-XXXXX-XXX-t1c.nii.gz
                       BraTS-GLI-XXXXX-XXX-t2w.nii.gz
                       BraTS-GLI-XXXXX-XXX-t2f.nii.gz
                       BraTS-GLI-XXXXX-XXX-seg.nii.gz
                     ...

Pipeline
--------
 1.  Configuration & reproducibility
 2.  3D data augmentation (rotations, flips, elastic deformations, intensity)
 3.  Custom Dataset + DataLoader (patch-based, memory-efficient)
 4.  3D U-Net architecture
 5.  Loss functions  (Dice, Focal, combined Dice+BCE)
 6.  Evaluation metrics (Dice, IoU, Hausdorff distance)
 7.  Training loop  (GPU, AMP mixed precision, gradient clipping)
 8.  Hyperparameter grid search
 9.  Transfer-learning helpers (2-D → 3-D weight inflation)
10.  Results, visualisations, model checkpoint saving

Outputs (all written to  outputs/ )
--------------------------------------
  checkpoints/best_model.pth          best validation-Dice checkpoint
  checkpoints/final_model.pth         weights after all epochs
  results/training_curves.png
  results/dice_per_class.png
  results/sample_predictions/         slice visualisations for N test cases
  results/metrics_summary.csv
  results/hyperparameter_search.csv

Usage
-----
  # download BraTS 2024 data + full training run
  python task2_brain_tumour_segmentation.py --auth-token YOUR_SYNAPSE_TOKEN

# evaluate a saved checkpoint
  python task2_brain_tumour_segmentation.py --eval --checkpoint outputs/checkpoints/best_model.pth
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Standard library & third-party imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import csv
import time
import math
import random
import argparse
import warnings
from pathlib import Path
from typing  import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib.colors import ListedColormap

import torch
import torch.nn            as nn
import torch.nn.functional as F
from   torch.utils.data   import Dataset, DataLoader
from   torch.amp          import GradScaler, autocast
from   torch.optim        import AdamW
from   torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from   tqdm               import tqdm

# nibabel for NIfTI I/O  (pip install nibabel)
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("WARNING: nibabel not found. Real BraTS data cannot be loaded.")
    print("         Install with:  pip install nibabel\n")

# scipy for elastic deformation & Hausdorff distance
try:
    from scipy.ndimage          import map_coordinates, gaussian_filter
    from scipy.spatial.distance import directed_hausdorff
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not found. Elastic deformation & Hausdorff disabled.")
    print("         Install with:  pip install scipy\n")

# synapseclient for downloading BraTS 2024 data from Synapse
try:
    import synapseclient
    import synapseutils
    SYNAPSE_AVAILABLE = True
except ImportError:
    SYNAPSE_AVAILABLE = False
    print("WARNING: synapseclient not found. Auto-download disabled.")
    print("         Install with:  pip install synapseclient\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_ROOT = Path("data")
    OUTPUT_DIR     = Path("outputs")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    RESULTS_DIR    = OUTPUT_DIR / "results"
    PRED_DIR       = RESULTS_DIR / "sample_predictions"

    # ── Synapse download ──────────────────────────────────────────────────────
    SYNAPSE_ID     = "syn59059776"       # BraTS 2024 GLI (Adult Glioma)

    # ── Data ───────────────────────────────────────────────────────────────────
    # BraTS 2024 modality suffixes:  t1n, t1c, t2w, t2f
    # We use T2-FLAIR (t2f) + post-contrast T1 (t1c) — most informative pair
    MODALITIES     = ["t2f", "t1c"]
    IN_CHANNELS    = len(MODALITIES)     # 2
    # BraTS 2024 segmentation labels:
    #   0 = background, 1 = NCR/NET (necrotic core), 2 = ED (oedema),
    #   3 = ET (enhancing tumour)
    # Labels are already contiguous 0–3, no remapping needed
    NUM_CLASSES    = 4
    CLASS_NAMES    = ["Background", "Necrotic Core", "Oedema", "Enhancing"]
    LABEL_REMAP    = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}   # BraTS 2024: label 4 (ET) → class 3

    # ── Patch-based training ───────────────────────────────────────────────────
    PATCH_SIZE     = (96, 96, 96)        # D × H × W  (128³ needs ≥16 GB VRAM)
    PATCHES_PER_VOL = 2                  # patches sampled per volume per epoch
    FOREGROUND_PROB = 0.75               # prob that patch centre is a tumour voxel

    # ── Model ──────────────────────────────────────────────────────────────────
    BASE_FILTERS   = 16                  # first encoder stage; doubles each level
    DEPTH          = 4                   # encoder levels (4 → 5 resolution scales)
    DROPOUT        = 0.1

    # ── Training Hardware ──────────────────────────────────────────────────────
    BATCH_SIZE     = 2                   # doubled → ~3 GB VRAM (safe on 6 GB card)
    ACCUMULATION_STEPS = 2               # effective batch = 2×2 = 4
    EPOCHS         = 50
    LR             = 1e-3
    WEIGHT_DECAY   = 1e-5
    GRAD_CLIP      = 1.0
    AMP            = True                # automatic mixed precision
    DEEP_SUPERVISION = True              # auxiliary heads for faster convergence
    PATIENCE       = 10                  # early stopping patience (val Dice)
    NUM_WORKERS    = 8                   # 8 of 20 threads — leaves headroom for OS/WSL
    PREFETCH_FACTOR = 4                  # each worker pre-loads 4 batches

    # ── Loss ───────────────────────────────────────────────────────────────────
    DICE_WEIGHT    = 0.6
    BCE_WEIGHT     = 0.4
    FOCAL_GAMMA    = 2.0

    # ── Augmentation ───────────────────────────────────────────────────────────
    AUG_FLIP_PROB     = 0.5
    AUG_ROTATE_PROB   = 0.3
    AUG_ELASTIC_PROB  = 0.1              # halved — elastic deform is the slowest aug
    AUG_INTENSITY_PROB= 0.3

    # ── Reproducibility ────────────────────────────────────────────────────────
    SEED           = 42

    # ── Misc ───────────────────────────────────────────────────────────────────
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_VIS_N     = 5                   # number of test cases to visualise


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False   # trade reproducibility for speed
    torch.backends.cudnn.benchmark     = True    # auto-tune fastest conv algorithm


def make_dirs() -> None:
    for d in [Config.CHECKPOINT_DIR, Config.RESULTS_DIR, Config.PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  3D Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────
class Augment3D:
    """
    Volumetric augmentation applied identically to image and label tensors.
    All operations work on numpy arrays of shape (C, D, H, W) for images
    and (D, H, W) for labels.
    """

    def __init__(self, cfg: Config = Config):
        self.cfg = cfg

    # ── Random axis-aligned flips ──────────────────────────────────────────────
    @staticmethod
    def random_flip(img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for axis in [1, 2, 3]:           # D, H, W axes of (C,D,H,W)
            if random.random() < 0.5:
                img = np.flip(img, axis=axis).copy()
                lbl = np.flip(lbl, axis=axis - 1).copy()   # lbl is (D,H,W)
        return img, lbl

    # ── Random 90° rotations in the axial plane ────────────────────────────────
    @staticmethod
    def random_rotate90(img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k=k, axes=(2, 3)).copy()   # H–W plane
            lbl = np.rot90(lbl, k=k, axes=(1, 2)).copy()
        return img, lbl

    # ── Elastic deformation ────────────────────────────────────────────────────
    @staticmethod
    def elastic_deform(
        img: np.ndarray,
        lbl: np.ndarray,
        alpha: float = 40.0,
        sigma: float = 6.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates tissue deformation: random displacement fields are smoothed
        with a Gaussian kernel and applied via map_coordinates interpolation.
        """
        if not SCIPY_AVAILABLE:
            return img, lbl

        shape = lbl.shape                # (D, H, W)
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma=sigma
        ) * alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma=sigma
        ) * alpha
        dz = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma=sigma
        ) * alpha

        d, h, w    = shape
        z, y, x    = np.meshgrid(np.arange(d), np.arange(h), np.arange(w),
                                  indexing="ij")
        indices    = (np.clip(z + dz, 0, d - 1).ravel(),
                      np.clip(y + dy, 0, h - 1).ravel(),
                      np.clip(x + dx, 0, w - 1).ravel())

        # Deform each modality channel
        img_def = np.stack([
            map_coordinates(img[c], indices, order=1, mode="reflect"
                            ).reshape(shape)
            for c in range(img.shape[0])
        ])
        lbl_def = map_coordinates(lbl.astype(float), indices,
                                   order=0, mode="reflect").reshape(shape)
        return img_def, lbl_def.astype(lbl.dtype)

    # ── Intensity augmentation (image only) ────────────────────────────────────
    @staticmethod
    def intensity_augment(img: np.ndarray) -> np.ndarray:
        """
        Random brightness shift + contrast scaling per channel.
        Simulates scanner variability between acquisition sites.
        """
        for c in range(img.shape[0]):
            shift   = np.random.uniform(-0.1, 0.1)
            scale   = np.random.uniform(0.9,  1.1)
            noise   = np.random.normal(0, 0.02, size=img[c].shape)
            img[c]  = img[c] * scale + shift + noise
        return img

    def __call__(
        self,
        img: np.ndarray,
        lbl: np.ndarray,
        training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not training:
            return img, lbl
        if random.random() < self.cfg.AUG_FLIP_PROB:
            img, lbl = self.random_flip(img, lbl)
        if random.random() < self.cfg.AUG_ROTATE_PROB:
            img, lbl = self.random_rotate90(img, lbl)
        if random.random() < self.cfg.AUG_ELASTIC_PROB:
            img, lbl = self.elastic_deform(img, lbl)
        if random.random() < self.cfg.AUG_INTENSITY_PROB:
            img = self.intensity_augment(img)
        return img, lbl


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset  (real BraTS  OR  synthetic for smoke-testing)
# ─────────────────────────────────────────────────────────────────────────────
def normalise_volume(vol: np.ndarray) -> np.ndarray:
    """
    Percentile-clipped Z-score normalisation using non-zero voxel statistics.
    1. Clip to [0.5th, 99.5th] percentile to remove scanner-specific outliers.
    2. Z-score normalise using the non-zero brain mask so that
       skull-stripped background voxels don't skew the statistics.
    """
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    # Clip outlier intensities (scanner variability)
    p05, p995 = np.percentile(vol[mask], [0.5, 99.5])
    vol = np.clip(vol, p05, p995)
    mu  = vol[mask].mean()
    std = vol[mask].std() + 1e-8
    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = (vol[mask] - mu) / std
    return out


def remap_labels(seg: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """Convert BraTS label values {0,1,2,4} → {0,1,2,3}."""
    out = np.zeros_like(seg)
    for src, dst in mapping.items():
        out[seg == src] = dst
    return out


class BraTSDataset(Dataset):
    """
    Patch-based 3D Dataset for BraTS-style NIfTI data.

    Each __getitem__ returns a randomly sampled 3-D patch of size
    Config.PATCH_SIZE from one patient volume, together with the
    corresponding label patch.

    Foreground-biased sampling:  with probability FOREGROUND_PROB the patch
    centre is placed on a tumour voxel, ensuring the model sees enough
    positive examples despite the severe class imbalance.
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        cfg:          Config     = Config,
        augment:      bool       = False,
    ) -> None:
        self.patient_dirs = patient_dirs
        self.cfg          = cfg
        self.augment      = augment
        self.aug          = Augment3D(cfg)
        self.patch_size   = np.array(cfg.PATCH_SIZE)

    def __len__(self) -> int:
        return len(self.patient_dirs) * self.cfg.PATCHES_PER_VOL

    def _load_patient(self, patient_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and normalise all modality volumes + segmentation mask."""
        name   = patient_dir.name
        imgs   = []
        for mod in self.cfg.MODALITIES:
            nii_path = patient_dir / f"{name}-{mod}.nii.gz"
            if not nii_path.exists():
                nii_path = patient_dir / f"{name}-{mod}.nii"
            vol = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
            imgs.append(normalise_volume(vol))
        img = np.stack(imgs, axis=0)                       # (C, D, H, W)

        seg_path = patient_dir / f"{name}-seg.nii.gz"
        if not seg_path.exists():
            seg_path = patient_dir / f"{name}-seg.nii"
        seg = nib.load(str(seg_path)).get_fdata().astype(np.int64)
        seg = remap_labels(seg, self.cfg.LABEL_REMAP)      # (D, H, W)

        return img, seg

    def _sample_patch_centre(
        self, seg: np.ndarray
    ) -> Tuple[int, int, int]:
        """
        Sample a valid patch centre.  With probability FOREGROUND_PROB,
        the centre is chosen from tumour voxels (foreground-biased sampling).
        """
        ps   = self.patch_size // 2
        d, h, w = seg.shape
        low  = ps
        high = np.array([d, h, w]) - ps

        # Guard: ensure volume is large enough for the patch
        high = np.maximum(high, low + 1)

        if random.random() < self.cfg.FOREGROUND_PROB:
            fg_coords = np.argwhere(seg > 0)
            if len(fg_coords) > 0:
                centre = fg_coords[random.randint(0, len(fg_coords) - 1)]
                centre = np.clip(centre, low, high - 1)
                return tuple(centre)

        return tuple(np.random.randint(low, high))

    def _extract_patch(
        self,
        img: np.ndarray,
        seg: np.ndarray,
        centre: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        ps    = self.patch_size // 2
        z0, y0, x0 = centre
        slices_img = (
            slice(None),
            slice(z0 - ps[0], z0 + ps[0]),
            slice(y0 - ps[1], y0 + ps[1]),
            slice(x0 - ps[2], x0 + ps[2]),
        )
        slices_seg = (
            slice(z0 - ps[0], z0 + ps[0]),
            slice(y0 - ps[1], y0 + ps[1]),
            slice(x0 - ps[2], x0 + ps[2]),
        )
        return img[slices_img].copy(), seg[slices_seg].copy()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_idx = idx % len(self.patient_dirs)
        patient_dir = self.patient_dirs[patient_idx]

        img, seg = self._load_patient(patient_dir)

        centre      = self._sample_patch_centre(seg)
        img_p, seg_p = self._extract_patch(img, seg, centre)

        # Pad if the volume was smaller than the patch size
        img_p, seg_p = self._pad_to_patch(img_p, seg_p)

        # Augmentation
        img_p, seg_p = self.aug(img_p, seg_p, training=self.augment)

        return {
            "image": torch.from_numpy(img_p.astype(np.float32)),
            "label": torch.from_numpy(seg_p.astype(np.int64)),
        }

    def _pad_to_patch(
        self,
        img: np.ndarray,
        seg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        target = self.patch_size
        pad_d  = max(0, target[0] - img.shape[1])
        pad_h  = max(0, target[1] - img.shape[2])
        pad_w  = max(0, target[2] - img.shape[3])
        if pad_d + pad_h + pad_w > 0:
            img = np.pad(img, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)))
            seg = np.pad(seg, ((0,pad_d),(0,pad_h),(0,pad_w)))
        return img[:, :target[0], :target[1], :target[2]], \
               seg[:target[0], :target[1], :target[2]]




# ─────────────────────────────────────────────────────────────────────────────
# 4.  3D U-Net Architecture
# ─────────────────────────────────────────────────────────────────────────────
class SqueezeExcite3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps.
    Learns per-channel attention weights so the network can emphasise
    the most informative modality features at each spatial location.
    Adds ~1% parameter overhead.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite  = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1, 1, 1)
        return x * w


class ConvBlock3D(nn.Module):
    """
    Double conv block: (Conv3d → BN → LeakyReLU) × 2 + SE attention.
    - LeakyReLU prevents dead neurons in sparse medical volumes.
    - Squeeze-and-Excitation learns per-channel importance.
    - Residual connection added when in/out channels match.
    """

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.se = SqueezeExcite3D(out_ch)
        # 1×1×1 projection for residual when channel dims differ
        self.residual = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x)) + self.residual(x)


class EncoderBlock(nn.Module):
    """ConvBlock followed by 2×2×2 max-pool for spatial downsampling."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ConvBlock3D(in_ch, out_ch, dropout)
        self.pool  = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """
    Trilinear upsample → concatenate skip connection → ConvBlock.
    Skip connections (from U-Net encoder) preserve fine-grained spatial
    detail lost during downsampling — critical for accurate tumour boundaries.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="trilinear",
                                align_corners=True)
        self.block = ConvBlock3D(in_ch + skip_ch, out_ch, dropout)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip if spatial dims differ by 1 voxel
        diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
        x    = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        return self.block(torch.cat([x, skip], dim=1))


class UNet3D(nn.Module):
    """

    3D convolutions
        Brain tumours are inherently 3-D structures with inter-slice
        continuity.  3D convolutions capture this volumetric context
        that 2D slice-wise approaches miss.

    Batch Normalisation
        Stabilises training of deep 3D networks where gradient flow
        is challenging; reduces sensitivity to weight initialisation.

    Residual connections within ConvBlocks
        Mitigates vanishing gradients in deeper encoder stages; allows
        gradients to bypass saturated layers during backpropagation.

    Dropout3D (channel-wise)
        Regularises the network given the limited annotated training data;
        channel-wise dropout is more effective than element-wise for conv.

    Trilinear upsampling (vs transposed conv)
        Avoids checkerboard artefacts common with transposed convolutions,
        producing smoother segmentation boundaries.
    """

    def __init__(self, cfg: Config = Config) -> None:
        super().__init__()
        f  = cfg.BASE_FILTERS       # base number of feature maps (16)
        d  = cfg.DROPOUT
        ic = cfg.IN_CHANNELS
        nc = cfg.NUM_CLASSES

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(ic,    f,     dropout=d)
        self.enc2 = EncoderBlock(f,     f*2,   dropout=d)
        self.enc3 = EncoderBlock(f*2,   f*4,   dropout=d)
        self.enc4 = EncoderBlock(f*4,   f*8,   dropout=d)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ConvBlock3D(f*8, f*16, dropout=d)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(f*16, f*8,  f*8,  dropout=d)
        self.dec3 = DecoderBlock(f*8,  f*4,  f*4,  dropout=d)
        self.dec2 = DecoderBlock(f*4,  f*2,  f*2,  dropout=d)
        self.dec1 = DecoderBlock(f*2,  f,    f,    dropout=d)

        # ── Output ────────────────────────────────────────────────────────────
        self.out_conv = nn.Conv3d(f, nc, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x     = self.bottleneck(x)
        x     = self.dec4(x, s4)
        x     = self.dec3(x, s3)
        x     = self.dec2(x, s2)
        x     = self.dec1(x, s1)
        return self.out_conv(x)          # (B, NUM_CLASSES, D, H, W) logits


# ─────────────────────────────────────────────────────────────────────────────
# 4b.  Transfer learning: inflate 2-D pretrained weights → 3-D
# ─────────────────────────────────────────────────────────────────────────────
def inflate_2d_weights_to_3d(
    state_dict_2d: dict,
    model_3d:      nn.Module,
) -> int:
    """
    'Weight inflation' strategy (Carreira & Zisserman, 2017):
    A 2-D convolutional kernel of shape (C_out, C_in, H, W) is replicated
    along a new depth dimension and divided by the depth to preserve the
    activation magnitude:
        W_3d[:, :, d, :, :] = W_2d / depth
    This allows a pretrained 2-D encoder to initialise a 3-D U-Net,
    providing better feature representations than random initialisation
    when annotated 3-D data is scarce.

    Returns the number of layers successfully inflated.
    """
    model_dict     = model_3d.state_dict()
    inflated_count = 0

    for k2d, v2d in state_dict_2d.items():
        if k2d not in model_dict:
            continue
        v3d_shape = model_dict[k2d].shape

        # Only inflate 4-D → 5-D conv kernels
        if v2d.ndim == 4 and v3d_shape[2:] == (v2d.shape[2], v2d.shape[3], v2d.shape[3]):
            depth     = v3d_shape[2]
            inflated  = v2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
            if inflated.shape == v3d_shape:
                model_dict[k2d] = inflated
                inflated_count += 1
        elif v2d.shape == v3d_shape:
            model_dict[k2d] = v2d
            inflated_count += 1

    model_3d.load_state_dict(model_dict, strict=False)
    print(f"  Transfer learning: inflated/copied {inflated_count} weight tensors.")
    return inflated_count


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.
    Dice = 2 * |P ∩ G| / (|P| + |G|)

    Directly optimises the overlap metric used for evaluation.
    Naturally handles class imbalance because it is normalised by the
    total number of predicted + actual positive voxels, so rare classes
    contribute equally to the loss regardless of their volume fraction.

    Smooth factor ε prevents division by zero on empty label patches.
    """

    def __init__(self, smooth: float = 1e-5, ignore_bg: bool = True) -> None:
        super().__init__()
        self.smooth    = smooth
        self.ignore_bg = ignore_bg

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        n_cls    = logits.shape[1]
        probs    = F.softmax(logits, dim=1)               # (B,C,D,H,W)
        targets1h = F.one_hot(targets, n_cls)             # (B,D,H,W,C)
        targets1h = targets1h.permute(0, 4, 1, 2, 3).float()

        start_cls = 1 if self.ignore_bg else 0
        dice_vals = []
        for c in range(start_cls, n_cls):
            p   = probs[:, c].contiguous().view(-1)
            g   = targets1h[:, c].contiguous().view(-1)
            num = 2 * (p * g).sum() + self.smooth
            den = p.sum() + g.sum() + self.smooth
            dice_vals.append(1.0 - num / den)

        return torch.stack(dice_vals).mean()


class FocalLoss(nn.Module):
    """
    Multi-class Focal loss (Lin et al., 2017).
    FL(p) = -α (1 - p_t)^γ  log(p_t)

    Down-weights the loss contribution of easy (well-classified background)
    voxels and focuses training on hard (tumour) voxels — directly addressing
    the severe class imbalance in brain MRI segmentation.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        # Register as buffer so .to(device) moves it automatically
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        probs = F.softmax(logits, dim=1)
        pt   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        fl   = ((1 - pt) ** self.gamma) * ce
        return fl.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination: dice_w * DiceLoss  +  focal_w * FocalLoss.

    The Dice component directly optimises overlap and handles class imbalance.
    The Focal component (Lin et al., 2017) down-weights easy background voxels
    and focuses gradient signal on hard-to-classify tumour boundaries —
    critical given the 99.25% background dominance in BraTS data.

    Class weights are derived from inverse frequency of the EDA-measured
    class distribution: BG=99.26%, NCR=0.008%, ED=0.67%, ET=0.07%.
    """

    def __init__(self, cfg: Config = Config) -> None:
        super().__init__()
        self.dice_loss  = DiceLoss(ignore_bg=True)
        # Inverse-frequency class weights for severe imbalance
        class_weights   = torch.tensor([0.01, 40.0, 1.5, 15.0], dtype=torch.float32)
        self.focal_loss = FocalLoss(gamma=cfg.FOCAL_GAMMA, alpha=class_weights)
        self.dw         = cfg.DICE_WEIGHT
        self.fw         = cfg.BCE_WEIGHT   # reusing the config slot for focal weight

    def forward(
        self, logits: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(logits, list):
            # Deep Supervision Loss
            # out[0] is full resolution, out[1] is 1/2, out[2] is 1/4
            weights = [0.5, 0.3, 0.2]
            total_loss = dl_total = fl_total = 0.0

            for pred, w in zip(logits, weights):
                # Resize targets to match prediction scale if needed
                if pred.shape[2:] != targets.shape[1:]:
                    target_res = F.interpolate(
                        targets.unsqueeze(1).float(),
                        size=pred.shape[2:],
                        mode="nearest"
                    ).squeeze(1).long()
                else:
                    target_res = targets

                dl = self.dice_loss(pred, target_res)
                fl = self.focal_loss(pred, target_res)
                total_loss += w * (self.dw * dl + self.fw * fl)
                dl_total += w * dl
                fl_total += w * fl

            return total_loss, torch.tensor(dl_total), torch.tensor(fl_total)

        # Standard Loss
        dl   = self.dice_loss(logits, targets)
        fl   = self.focal_loss(logits, targets)
        loss = self.dw * dl + self.fw * fl
        return loss, dl, fl


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────
def dice_score(
    pred: np.ndarray,
    gt:   np.ndarray,
    cls:  int,
    smooth: float = 1e-5,
) -> float:
    """Per-class Dice score: 2|P∩G| / (|P|+|G|)."""
    p   = (pred == cls).astype(float)
    g   = (gt   == cls).astype(float)
    num = 2 * (p * g).sum() + smooth
    den = p.sum() + g.sum() + smooth
    return float(num / den)


def iou_score(
    pred: np.ndarray,
    gt:   np.ndarray,
    cls:  int,
    smooth: float = 1e-5,
) -> float:
    """Per-class Intersection over Union (Jaccard index)."""
    p   = (pred == cls).astype(float)
    g   = (gt   == cls).astype(float)
    i   = (p * g).sum()
    u   = p.sum() + g.sum() - i
    return float((i + smooth) / (u + smooth))


def hausdorff_distance(
    pred: np.ndarray,
    gt:   np.ndarray,
    cls:  int,
) -> float:
    """
    Symmetric Hausdorff distance between predicted and ground-truth
    tumour surface for a given class.
    Returns NaN if either mask is empty (avoids invalid distance).
    """
    if not SCIPY_AVAILABLE:
        return float("nan")

    p_pts = np.argwhere(pred == cls).astype(float)
    g_pts = np.argwhere(gt   == cls).astype(float)

    if len(p_pts) == 0 or len(g_pts) == 0:
        return float("nan")

    d1 = directed_hausdorff(p_pts, g_pts)[0]
    d2 = directed_hausdorff(g_pts, p_pts)[0]
    return max(d1, d2)


def evaluate_batch(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    n_cls:   int,
) -> Dict[str, List[float]]:
    """Compute Dice + IoU for every class in a batch."""
    preds = logits.argmax(dim=1).cpu().numpy()
    gts   = targets.cpu().numpy()
    metrics: Dict[str, List[float]] = {"dice": [], "iou": []}
    for cls in range(1, n_cls):        # skip background
        batch_dice = [dice_score(preds[b], gts[b], cls)
                      for b in range(preds.shape[0])]
        batch_iou  = [iou_score(preds[b], gts[b], cls)
                      for b in range(preds.shape[0])]
        metrics["dice"].append(float(np.mean(batch_dice)))
        metrics["iou"].append(float(np.mean(batch_iou)))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  CombinedLoss,
    optimizer:  torch.optim.Optimizer,
    scaler:     GradScaler,
    device:     str,
    cfg:        Config,
) -> Dict[str, float]:
    model.train()
    total_loss = dice_loss_sum = ce_loss_sum = 0.0
    dice_sums  = [0.0] * (cfg.NUM_CLASSES - 1)
    n_batches  = 0

    print(f"   (Buffering first {cfg.NUM_WORKERS} batches on CPU...)")
    pbar = tqdm(loader, desc="   Train", leave=False)
    for i, batch in enumerate(pbar):
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Automatic Mixed Precision forward pass
        with autocast("cuda", enabled=cfg.AMP and device == "cuda"):
            logits       = model(imgs)
            loss, dl, ce = criterion(logits, labels)
            # Scale loss for accumulation
            loss = loss / cfg.ACCUMULATION_STEPS

        # Scaled backward
        scaler.scale(loss).backward()

        if (i + 1) % cfg.ACCUMULATION_STEPS == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss    += loss.item() * cfg.ACCUMULATION_STEPS
        dice_loss_sum += dl.item()
        ce_loss_sum   += ce.item()

        # For metrics, use highest resolution prediction only
        main_logits = logits[0] if isinstance(logits, list) else logits
        m = evaluate_batch(main_logits.detach(), labels, cfg.NUM_CLASSES)
        for j, d in enumerate(m["dice"]):
            dice_sums[j] += d
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item()*cfg.ACCUMULATION_STEPS:.3f}",
                        dice=f"{np.mean(m['dice']):.3f}")

    n = max(n_batches, 1)
    return {
        "loss":      total_loss    / n,
        "dice_loss": dice_loss_sum / n,
        "ce_loss":   ce_loss_sum   / n,
        "dice_cls":  [s / n for s in dice_sums],
        "mean_dice": float(np.mean([s / n for s in dice_sums])),
    }


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: CombinedLoss,
    device:    str,
    cfg:       Config,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    dice_sums  = [0.0] * (cfg.NUM_CLASSES - 1)
    iou_sums   = [0.0] * (cfg.NUM_CLASSES - 1)
    n_batches  = 0

    pbar = tqdm(loader, desc="   Val  ", leave=False)
    for batch in pbar:
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)

        with autocast("cuda", enabled=cfg.AMP and device == "cuda"):
            logits       = model(imgs)
            loss, _, _   = criterion(logits, labels)

        total_loss += loss.item()
        m = evaluate_batch(logits, labels, cfg.NUM_CLASSES)
        for i in range(len(dice_sums)):
            dice_sums[i] += m["dice"][i]
            iou_sums[i]  += m["iou"][i]
        n_batches += 1
        pbar.set_postfix(dice=f"{np.mean(m['dice']):.3f}")

    n = max(n_batches, 1)
    return {
        "loss":      total_loss / n,
        "dice_cls":  [s / n for s in dice_sums],
        "iou_cls":   [s / n for s in iou_sums],
        "mean_dice": float(np.mean([s / n for s in dice_sums])),
        "mean_iou":  float(np.mean([s / n for s in iou_sums])),
    }


def train(
    model:      nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          Config,
) -> Dict[str, list]:
    """Full training loop with AMP, early stopping, checkpointing, and resume."""

    device    = cfg.DEVICE
    criterion = CombinedLoss(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LR,
                      weight_decay=cfg.WEIGHT_DECAY)
    warmup_epochs = 5
    warmup    = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(1, cfg.EPOCHS - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    scaler    = GradScaler("cuda", enabled=cfg.AMP and device == "cuda")

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "val_iou":    [], "lr":       [],
    }

    best_val_dice = -1.0
    patience_ctr  = 0
    start_epoch   = 1
    best_ckpt     = str(cfg.CHECKPOINT_DIR / "best_model.pth")
    last_ckpt     = str(cfg.CHECKPOINT_DIR / "last_checkpoint.pth")

    # ── Resume from last checkpoint if available ──────────────────────────────
    if Path(last_ckpt).exists():
        print(f"\n⟳ Resuming from {last_ckpt}...")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "history" in ckpt:
            history = ckpt["history"]
        best_val_dice = ckpt.get("best_val_dice", -1.0)
        patience_ctr  = ckpt.get("patience_ctr", 0)
        start_epoch   = ckpt["epoch"] + 1
        print(f"   Resumed at epoch {start_epoch} "
              f"(best Dice so far: {best_val_dice:.4f}, "
              f"patience: {patience_ctr}/{cfg.PATIENCE})")

    print(f"\n{'='*65}")
    print(f"Training on {device.upper()}"
          + (f"  [{torch.cuda.get_device_name(0)}]"
             if device == "cuda" else ""))
    print(f"Epochs: {start_epoch}→{cfg.EPOCHS}  |  Batch: {cfg.BATCH_SIZE}"
          f"  |  Patch: {cfg.PATCH_SIZE}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion,
                                   optimizer, scaler, device, cfg)
        val_m   = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_dice"].append(train_m["mean_dice"])
        history["val_dice"].append(val_m["mean_dice"])
        history["val_iou"].append(val_m["mean_iou"])
        history["lr"].append(lr)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{cfg.EPOCHS}]  "
            f"Train loss: {train_m['loss']:.4f}  "
            f"Train Dice: {train_m['mean_dice']:.4f}  |  "
            f"Val loss: {val_m['loss']:.4f}  "
            f"Val Dice: {val_m['mean_dice']:.4f}  "
            f"Val IoU: {val_m['mean_iou']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # Per-class Dice
        cls_str = "  ".join(
            f"{Config.CLASS_NAMES[c+1]}:{val_m['dice_cls'][c]:.3f}"
            for c in range(len(val_m["dice_cls"]))
        )
        print(f"           Per-class val Dice: {cls_str}")

        # Checkpoint: save best model
        if val_m["mean_dice"] > best_val_dice:
            best_val_dice = val_m["mean_dice"]
            patience_ctr  = 0
            save_checkpoint(model, optimizer, epoch, val_m, best_ckpt,
                            scheduler, scaler, history,
                            best_val_dice, patience_ctr)
            print(f"           ✓ New best model saved  (Dice={best_val_dice:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}"
                      f" (no improvement for {cfg.PATIENCE} epochs)")
                break

        # Always save last checkpoint for resume
        save_checkpoint(model, optimizer, epoch, val_m, last_ckpt,
                        scheduler, scaler, history,
                        best_val_dice, patience_ctr)

    # Save final weights
    final_ckpt = str(cfg.CHECKPOINT_DIR / "final_model.pth")
    save_checkpoint(model, optimizer, epoch, val_m, final_ckpt,
                    scheduler, scaler, history,
                    best_val_dice, patience_ctr)
    print(f"\nFinal model saved → {final_ckpt}")
    print(f"Best model saved  → {best_ckpt}"
          f"  (Val Dice = {best_val_dice:.4f})")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   dict,
    path:      str,
    scheduler=None,
    scaler=None,
    history=None,
    best_val_dice: float = -1.0,
    patience_ctr: int = 0,
) -> None:
    """Save a full-state checkpoint for resume support."""
    payload = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optim_state":    optimizer.state_dict(),
        "metrics":        metrics,
        "best_val_dice":  best_val_dice,
        "patience_ctr":   patience_ctr,
        "config": {
            "IN_CHANNELS":  Config.IN_CHANNELS,
            "NUM_CLASSES":  Config.NUM_CLASSES,
            "BASE_FILTERS": Config.BASE_FILTERS,
            "PATCH_SIZE":   Config.PATCH_SIZE,
        },
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if history is not None:
        payload["history"] = history
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module,
                    device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    dice_val = ckpt.get('metrics', {}).get('mean_dice', 'N/A')
    dice_str = f"{dice_val:.4f}" if isinstance(dice_val, float) else str(dice_val)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}"
          f"  (Val Dice = {dice_str})")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────
def hyperparameter_search(
    train_ds: Dataset,
    val_ds:   Dataset,
    cfg:      Config,
) -> dict:
    """
    Lightweight grid search over LR, base filters, and loss weights.
    Each config is trained for a small number of warm-up epochs and
    evaluated by validation Dice.  The best config is returned.
    """
    grid = {
        "lr":           [1e-3, 5e-4],
        "base_filters": [16, 32],
        "dice_weight":  [0.6, 0.8],
    }

    results  = []
    best_cfg = None
    best_val = -1.0

    print(f"\n{'='*65}")
    print("Hyperparameter search")
    print(f"{'='*65}")

    combo_id = 0
    for lr in grid["lr"]:
        for bf in grid["base_filters"]:
            for dw in grid["dice_weight"]:
                combo_id += 1
                print(f"\n  Config {combo_id}: lr={lr}  base_filters={bf}"
                      f"  dice_weight={dw}")

                # Temporarily override config
                cfg.LR           = lr
                cfg.BASE_FILTERS = bf
                cfg.DICE_WEIGHT  = dw
                cfg.BCE_WEIGHT   = 1.0 - dw
                cfg.EPOCHS       = 5        # short warm-up

                set_seed(cfg.SEED)
                model = UNet3D(cfg).to(cfg.DEVICE)
                tl    = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                   shuffle=True,  num_workers=cfg.NUM_WORKERS)
                vl    = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                                   shuffle=False, num_workers=cfg.NUM_WORKERS)

                criterion = CombinedLoss(cfg).to(cfg.DEVICE)
                optimizer = AdamW(model.parameters(), lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
                scaler    = GradScaler("cuda", enabled=cfg.AMP and cfg.DEVICE == "cuda")

                for _ in range(cfg.EPOCHS):
                    train_one_epoch(model, tl, criterion, optimizer,
                                    scaler, cfg.DEVICE, cfg)
                val_m = validate(model, vl, criterion, cfg.DEVICE, cfg)

                print(f"    → Val Dice: {val_m['mean_dice']:.4f}"
                      f"  Val IoU: {val_m['mean_iou']:.4f}")

                results.append({
                    "lr": lr, "base_filters": bf, "dice_weight": dw,
                    "val_dice": round(val_m["mean_dice"], 4),
                    "val_iou":  round(val_m["mean_iou"],  4),
                })

                if val_m["mean_dice"] > best_val:
                    best_val = val_m["mean_dice"]
                    best_cfg = {"lr": lr, "base_filters": bf,
                                "dice_weight": dw}

    # Save search results
    df   = pd.DataFrame(results).sort_values("val_dice", ascending=False)
    path = str(cfg.RESULTS_DIR / "hyperparameter_search.csv")
    df.to_csv(path, index=False)
    print(f"\nHyperparameter search results saved → {path}")
    print(f"Best config: {best_cfg}  (Val Dice = {best_val:.4f})")

    return best_cfg


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Full test-set evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def test_evaluation(
    model:      nn.Module,
    test_loader: DataLoader,
    cfg:         Config,
) -> Dict[str, float]:
    """Compute Dice, IoU, and Hausdorff distance on the test set."""
    device = cfg.DEVICE
    model.eval()

    all_dice = {c: [] for c in range(1, cfg.NUM_CLASSES)}
    all_iou  = {c: [] for c in range(1, cfg.NUM_CLASSES)}
    all_hd   = {c: [] for c in range(1, cfg.NUM_CLASSES)}

    for batch in test_loader:
        imgs   = batch["image"].to(device)
        labels = batch["label"]

        with autocast("cuda", enabled=cfg.AMP and device == "cuda"):
            logits = model(imgs)

        preds = logits.argmax(dim=1).cpu().numpy()
        gts   = labels.numpy()

        for b in range(preds.shape[0]):
            for c in range(1, cfg.NUM_CLASSES):
                all_dice[c].append(dice_score(preds[b], gts[b], c))
                all_iou[c].append(iou_score(preds[b],   gts[b], c))
                all_hd[c].append(hausdorff_distance(preds[b], gts[b], c))

    summary = {}
    print(f"\n{'='*65}")
    print("Test Set Evaluation")
    print(f"{'='*65}")
    print(f"{'Class':<20}  {'Dice':>8}  {'IoU':>8}  {'Hausdorff':>10}")
    print("-" * 50)
    for c in range(1, cfg.NUM_CLASSES):
        d  = float(np.nanmean(all_dice[c]))
        i  = float(np.nanmean(all_iou[c]))
        h  = float(np.nanmean([v for v in all_hd[c] if not math.isnan(v)] or [float("nan")]))
        cname = cfg.CLASS_NAMES[c]
        print(f"{cname:<20}  {d:>8.4f}  {i:>8.4f}  {h:>10.2f}")
        summary[f"dice_{cname}"]      = d
        summary[f"iou_{cname}"]       = i
        summary[f"hausdorff_{cname}"] = h

    mean_d = float(np.mean([summary[k] for k in summary if k.startswith("dice_")]))
    mean_i = float(np.mean([summary[k] for k in summary if k.startswith("iou_")]))
    print("-" * 50)
    print(f"{'Mean (foreground)':<20}  {mean_d:>8.4f}  {mean_i:>8.4f}")
    summary["mean_dice"] = mean_d
    summary["mean_iou"]  = mean_i

    # Save CSV
    csv_path = str(cfg.RESULTS_DIR / "metrics_summary.csv")
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    print(f"\nMetrics saved → {csv_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Visualisations
# ─────────────────────────────────────────────────────────────────────────────
SEG_CMAP = ListedColormap(["black", "red", "green", "blue"])


def plot_training_curves(history: Dict[str, list], cfg: Config) -> None:
    """Save training/validation loss and Dice curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train", color="#2c3e50", lw=2)
    axes[0].plot(history["val_loss"],   label="Val",   color="#e74c3c", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Combined Loss")
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_dice"], label="Train Dice", color="#2c3e50", lw=2)
    axes[1].plot(history["val_dice"],   label="Val Dice",   color="#e74c3c", lw=2)
    axes[1].plot(history["val_iou"],    label="Val IoU",    color="#3498db", lw=2,
                 linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Dice & IoU Curves", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["lr"], color="#8e44ad", lw=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule (Cosine)", fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(cfg.RESULTS_DIR / "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {path}")


def plot_dice_per_class(metrics: Dict[str, float], cfg: Config) -> None:
    """Bar chart of Dice score per tumour sub-region."""
    classes = cfg.CLASS_NAMES[1:]    # skip background
    scores  = [metrics.get(f"dice_{c}", 0.0) for c in classes]
    colours = ["#e74c3c", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, scores, color=colours, edgecolor="black", width=0.5)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Dice Score")
    ax.set_title("Test Dice Score per Tumour Sub-Region", fontweight="bold")
    ax.axhline(metrics.get("mean_dice", 0), color="gray",
               linestyle="--", lw=1.5, label="Mean Dice")
    ax.legend()
    plt.tight_layout()
    path = str(cfg.RESULTS_DIR / "dice_per_class.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dice-per-class chart saved → {path}")


@torch.no_grad()
def save_sample_predictions(
    model:   nn.Module,
    dataset: Dataset,
    cfg:     Config,
    n:       int = 5,
) -> None:
    """
    For the first N samples, save a 3-panel figure showing:
      Col 1 — input FLAIR slice (axial mid-plane)
      Col 2 — ground-truth segmentation
      Col 3 — model prediction
    """
    device = cfg.DEVICE
    model.eval()

    for idx in range(min(n, len(dataset))):
        sample = dataset[idx]
        img    = sample["image"].unsqueeze(0).to(device)
        lbl    = sample["label"].numpy()

        with autocast("cuda", enabled=cfg.AMP and device == "cuda"):
            logit = model(img)
        pred = logit.argmax(dim=1).squeeze(0).cpu().numpy()

        # Pick the axial slice with the most foreground
        fg_per_slice = (lbl > 0).sum(axis=(1, 2))
        mid_z        = int(fg_per_slice.argmax()) if fg_per_slice.max() > 0 \
                       else lbl.shape[0] // 2

        flair_slice = sample["image"][0, mid_z].numpy()
        lbl_slice   = lbl[mid_z]
        pred_slice  = pred[mid_z]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sample {idx + 1}  —  Axial slice z={mid_z}",
                     fontsize=13, fontweight="bold")

        axes[0].imshow(flair_slice, cmap="gray", origin="lower")
        axes[0].set_title("FLAIR Input", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(flair_slice, cmap="gray", origin="lower")
        axes[1].imshow(lbl_slice,  cmap=SEG_CMAP, alpha=0.6,
                       vmin=0, vmax=3, origin="lower")
        axes[1].set_title("Ground Truth", fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(flair_slice, cmap="gray", origin="lower")
        axes[2].imshow(pred_slice, cmap=SEG_CMAP, alpha=0.6,
                       vmin=0, vmax=3, origin="lower")
        axes[2].set_title("Prediction", fontweight="bold")
        axes[2].axis("off")

        # Colour legend
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color="red",   label="Necrotic Core"),
            plt.Rectangle((0, 0), 1, 1, color="green", label="Oedema"),
            plt.Rectangle((0, 0), 1, 1, color="blue",  label="Enhancing"),
        ]
        axes[2].legend(handles=legend_patches, loc="lower right",
                       fontsize=8, framealpha=0.7)

        plt.tight_layout()
        path = str(cfg.PRED_DIR / f"sample_{idx+1:02d}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Sample predictions saved → {cfg.PRED_DIR}/")


def plot_model_architecture_summary(model: nn.Module, cfg: Config) -> None:
    """Save a text summary of the model parameter counts per module."""
    lines   = ["3D U-Net Architecture Summary", "=" * 55]
    total   = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.BatchNorm3d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            total += params
            lines.append(f"  {name:<45} {params:>10,}")
    lines.append("=" * 55)
    lines.append(f"  {'TOTAL PARAMETERS':<45} {total:>10,}")

    path = str(cfg.RESULTS_DIR / "model_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Model summary saved → {path}")
    print(f"  Total parameters: {total:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 11b. Data Downloading 
# ─────────────────────────────────────────────────────────────────────────────
def _data_exists(cfg: Config) -> bool:
    if not cfg.DATA_ROOT.exists():
        return False
    patient_dirs = [d for d in cfg.DATA_ROOT.rglob("BraTS*") if d.is_dir() and "GLI" in d.name and d.name.startswith("BraTS")]
    return len(patient_dirs) > 0

def download_brats_data(cfg: Config, auth_token: Optional[str] = None) -> None:
    if not SYNAPSE_AVAILABLE:
        print("Synapse client not available. Cannot verify/download data via API.")
        return

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token = auth_token or os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        print("No SYNAPSE_AUTH_TOKEN found in environment or .env file.")
        import getpass
        token = getpass.getpass("Please enter your Synapse auth token to verify/download data: ")
        if not token:
            print("No token provided. Cannot verify dataset via API.")
            sys.exit(1)
        os.environ["SYNAPSE_AUTH_TOKEN"] = token

    print("Authenticating with Synapse to verify dataset...")
    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    # 1. First, fetch what is available via the API
    print(f"Checking API for dataset {cfg.SYNAPSE_ID}...")
    children = list(syn.getChildren(cfg.SYNAPSE_ID))
    
    # 2. Check if the local data has the full data matching the API
    missing_zips = False
    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    for child in children:
        name = child['name']
        if name.endswith('.zip'):
            local_path = cfg.DATA_ROOT / name
            # If the zip is missing, check if we at least have extracted patient dirs.
            # We assume if there's a large number of extracted dirs, the data is full.
            if not local_path.exists():
                missing_zips = True

    patient_dirs = [d for d in cfg.DATA_ROOT.rglob("BraTS*") if d.is_dir() and "GLI" in d.name and d.name.startswith("BraTS")]
    data_fully_extracted = len(patient_dirs) > 1500

    if not missing_zips or data_fully_extracted:
        print(f"Data verification passed. Found full dataset (either zip files or {len(patient_dirs)} extracted patients).")
    else:
        print(f"Data is missing or incomplete. Downloading from Synapse API to {cfg.DATA_ROOT}...")
        files = synapseutils.syncFromSynapse(syn, entity=cfg.SYNAPSE_ID, path=str(cfg.DATA_ROOT))

    # After verification/download, extract any local zip files if patient dirs are missing
    patient_dirs = [d for d in cfg.DATA_ROOT.rglob("BraTS*") if d.is_dir() and "GLI" in d.name and d.name.startswith("BraTS")]
    if len(patient_dirs) < 100:  # If we don't have enough extracted patient directories
        local_zips = list(cfg.DATA_ROOT.glob("*.zip"))
        if local_zips:
            import zipfile
            print(f"Extracted patient directories missing. Extracting verified zip archives...")
            for filepath in local_zips:
                print(f"Extracting {filepath} using zipfile...")
                try:
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(cfg.DATA_ROOT)
                except Exception as e:
                    print(f"WARNING: Failed to extract {filepath}: {e}")



# ─────────────────────────────────────────────────────────────────────────────
# 11b. Initial Data Analysis (EDA)
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(cfg: Config, data_root: Path, results_dir: Path) -> None:
    print("\n--- Running Initial Data Analysis (EDA) ---")
    if not data_root.exists():
        print(f"Data root {data_root} does not exist. Cannot run EDA.")
        return

    patient_dirs = sorted(list(set([d for d in data_root.rglob("BraTS*") if d.is_dir() and "GLI" in d.name and d.name.startswith("BraTS")])))
    if len(patient_dirs) == 0:
        print("No patient directories found for EDA.")
        return
        
    print(f"Total patient directories found: {len(patient_dirs)}")
    
    # 1. Visualize 3D Data (Pick a random patient)
    random_dir = random.choice(patient_dirs)
    print(f"Visualizing random subject: {random_dir.name}")
    
    t2f_path = random_dir / f"{random_dir.name}-t2f.nii.gz"
    t1c_path = random_dir / f"{random_dir.name}-t1c.nii.gz"
    seg_path = random_dir / f"{random_dir.name}-seg.nii.gz"
    
    # Fallback to .nii if .nii.gz doesn't exist
    if not t2f_path.exists(): t2f_path = random_dir / f"{random_dir.name}-t2f.nii"
    if not t1c_path.exists(): t1c_path = random_dir / f"{random_dir.name}-t1c.nii"
    if not seg_path.exists(): seg_path = random_dir / f"{random_dir.name}-seg.nii"
    
    if not (t2f_path.exists() and t1c_path.exists() and seg_path.exists()):
        print("Missing required NIfTI files for visualization.")
    else:
        try:
            import nibabel as nib
            t2f = nib.load(str(t2f_path)).get_fdata()
            t1c = nib.load(str(t1c_path)).get_fdata()
            seg = nib.load(str(seg_path)).get_fdata()
            
            # Pick a middle slice in the axial (Z) plane where tumor is prominent
            z_mid = t2f.shape[2] // 2
            tumor_slices = np.sum(seg > 0, axis=(0, 1))
            if np.max(tumor_slices) > 0:
                z_mid = np.argmax(tumor_slices)
                
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(t2f[:, :, z_mid].T, cmap='gray', origin='lower')
            axes[0].set_title('T2-FLAIR')
            axes[1].imshow(t1c[:, :, z_mid].T, cmap='gray', origin='lower')
            axes[1].set_title('T1c')
            
            cmap = ListedColormap(['black', 'red', 'green', 'blue'])
            axes[2].imshow(seg[:, :, z_mid].T, cmap=cmap, origin='lower', vmin=0, vmax=3)
            axes[2].set_title('Segmentation Mask (Red: NCR, Green: ED, Blue: ET)')
            
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(results_dir / "eda_sample_slice.png", dpi=150)
            plt.close()
            print(f"  Saved slice visualization to {results_dir / 'eda_sample_slice.png'}")
        except Exception as e:
            print(f"  Error during visualization: {e}")

    # 2. Compute Class Distribution to quantify Class Imbalance
    print("Computing class distribution (over a subset of 10 patients)...")
    subset = random.sample(patient_dirs, min(10, len(patient_dirs)))
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for p_dir in subset:
        s_path = p_dir / f"{p_dir.name}-seg.nii.gz"
        if not s_path.exists(): s_path = p_dir / f"{p_dir.name}-seg.nii"
        
        if s_path.exists():
            try:
                import nibabel as nib
                seg = nib.load(str(s_path)).get_fdata()
                unique, counts = np.unique(seg, return_counts=True)
                for u, c in zip(unique, counts):
                    if u in class_counts:
                        class_counts[int(u)] += c
            except Exception:
                pass
                
    total_voxels = sum(class_counts.values())
    if total_voxels > 0:
        print("\nClass Imbalance Statistics:")
        for c_idx, name in enumerate(cfg.CLASS_NAMES):
            pct = (class_counts[c_idx] / total_voxels) * 100
            print(f"  {name}: {pct:.4f}% ({class_counts[c_idx]} voxels)")
            
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(cfg.CLASS_NAMES, [class_counts[i] for i in range(4)], color=['gray', 'red', 'green', 'blue'])
        ax.set_yscale('log')  # Log scale due to severe imbalance
        ax.set_ylabel('Voxel Count (Log Scale)')
        ax.set_title('Class Distribution (Demonstrating Severe Imbalance)')
        plt.tight_layout()
        plt.savefig(results_dir / "eda_class_distribution.png", dpi=150)
        plt.close()
        print(f"  Saved class distribution plot to {results_dir / 'eda_class_distribution.png'}")
    print("--- EDA Complete ---\n")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Dataset factory — balanced, stratified subset with metadata
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_TOTAL = 600          # total patients to use
SUBSET_TRAIN = 530          # 530 train + 35 val + 35 test = 600
SUBSET_VAL   = 35
SUBSET_TEST  = 35


def _compute_tumour_volume(seg_path: Path) -> float:
    """Compute total tumour voxel count (labels 1, 2, 3) for stratification."""
    try:
        seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
        return float(np.sum(seg > 0))
    except Exception:
        return 0.0


def _stratified_subset_split(
    patient_dirs: List[Path],
    cfg: Config,
    n_total: int = SUBSET_TOTAL,
    n_train: int = SUBSET_TRAIN,
    n_val: int   = SUBSET_VAL,
    n_test: int  = SUBSET_TEST,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Selects a balanced subset of patients and splits them into train/val/test
    using stratification by tumour volume to ensure each split has a similar
    distribution of small, medium, and large tumours.

    Steps:
      1. Filter to patients that actually have a segmentation mask.
      2. Compute tumour volume for each patient.
      3. Bin patients into volume strata (quartiles).
      4. Sample proportionally from each stratum to build the subset.
      5. Within each stratum, split into train/val/test.
    """
    assert n_train + n_val + n_test == n_total, \
        f"Split sizes must sum to n_total: {n_train}+{n_val}+{n_test} != {n_total}"

    # 1. Filter to patients with segmentation masks
    labelled = []
    for d in patient_dirs:
        seg = d / f"{d.name}-seg.nii.gz"
        if seg.exists():
            labelled.append(d)
    print(f"  Patients with segmentation masks: {len(labelled)} / {len(patient_dirs)}")

    if len(labelled) < n_total:
        print(f"  WARNING: Only {len(labelled)} labelled patients available, "
              f"using all of them instead of {n_total}.")
        n_total = len(labelled)
        # Rescale splits proportionally
        ratio_val  = n_val / (n_train + n_val + n_test)
        ratio_test = n_test / (n_train + n_val + n_test)
        n_val  = max(1, int(n_total * ratio_val))
        n_test = max(1, int(n_total * ratio_test))
        n_train = n_total - n_val - n_test

    # 2. Compute tumour volumes
    print(f"  Computing tumour volumes for stratification ({len(labelled)} patients)...")
    volumes = {}
    for i, d in enumerate(labelled):
        seg_path = d / f"{d.name}-seg.nii.gz"
        volumes[d] = _compute_tumour_volume(seg_path)
        if (i + 1) % 200 == 0:
            print(f"    ... processed {i+1}/{len(labelled)}")
    print(f"    ... done. Volume range: {min(volumes.values()):.0f} – {max(volumes.values()):.0f} voxels")

    # 3. Bin into strata using quartiles
    vol_values = np.array([volumes[d] for d in labelled])
    q25, q50, q75 = np.percentile(vol_values, [25, 50, 75])

    strata: Dict[str, List[Path]] = {
        "small":       [],   # 0 – Q25
        "medium_low":  [],   # Q25 – Q50
        "medium_high": [],   # Q50 – Q75
        "large":       [],   # Q75+
    }
    for d in labelled:
        v = volumes[d]
        if v <= q25:
            strata["small"].append(d)
        elif v <= q50:
            strata["medium_low"].append(d)
        elif v <= q75:
            strata["medium_high"].append(d)
        else:
            strata["large"].append(d)

    print(f"  Strata counts: " + ", ".join(f"{k}={len(v)}" for k, v in strata.items()))

    # 4. Sample proportionally from each stratum, then split
    random.seed(cfg.SEED)
    train_dirs, val_dirs, test_dirs = [], [], []

    for stratum_name, stratum_dirs in strata.items():
        random.shuffle(stratum_dirs)
        # Proportion of the total pool this stratum represents
        proportion = len(stratum_dirs) / len(labelled)
        s_total = max(1, round(n_total * proportion))
        s_val   = max(1, round(n_val * proportion))
        s_test  = max(1, round(n_test * proportion))
        s_train = s_total - s_val - s_test

        # Clamp to available
        s_total = min(s_total, len(stratum_dirs))
        s_train = min(s_train, s_total - s_val - s_test)
        if s_train < 0:
            s_train = 0

        selected = stratum_dirs[:s_total]
        train_dirs.extend(selected[:s_train])
        val_dirs.extend(selected[s_train:s_train + s_val])
        test_dirs.extend(selected[s_train + s_val:s_train + s_val + s_test])

    # 5. Adjust counts to match exact targets (rounding may cause ±1-2 drift)
    all_used = set(train_dirs + val_dirs + test_dirs)
    all_unused = [d for d in labelled if d not in all_used]
    random.shuffle(all_unused)

    while len(train_dirs) + len(val_dirs) + len(test_dirs) < n_total and all_unused:
        train_dirs.append(all_unused.pop())
    while len(train_dirs) > n_train and len(val_dirs) < n_val:
        val_dirs.append(train_dirs.pop())
    while len(train_dirs) > n_train and len(test_dirs) < n_test:
        test_dirs.append(train_dirs.pop())

    random.shuffle(train_dirs)
    random.shuffle(val_dirs)
    random.shuffle(test_dirs)

    return train_dirs, val_dirs, test_dirs


def _save_subset_metadata(
    train_dirs: List[Path],
    val_dirs: List[Path],
    test_dirs: List[Path],
    cfg: Config,
) -> Path:
    """Save a JSON file describing which patients were assigned to each split."""
    import json

    metadata = {
        "description": "BraTS 2024 GLI stratified subset metadata",
        "created_at": str(pd.Timestamp.now()),
        "seed": cfg.SEED,
        "total_patients": len(train_dirs) + len(val_dirs) + len(test_dirs),
        "splits": {
            "train": {
                "count": len(train_dirs),
                "patients": sorted([d.name for d in train_dirs]),
            },
            "validation": {
                "count": len(val_dirs),
                "patients": sorted([d.name for d in val_dirs]),
            },
            "test": {
                "count": len(test_dirs),
                "patients": sorted([d.name for d in test_dirs]),
            },
        },
        "stratification": "tumour_volume_quartiles",
        "notes": (
            "Patients were stratified by total tumour volume (voxels with label > 0) "
            "into 4 quartile-based strata (small, medium_low, medium_high, large). "
            "Each split was sampled proportionally from every stratum to ensure "
            "balanced tumour-size representation across train, validation, and test sets."
        ),
    }

    out_path = cfg.RESULTS_DIR / "subset_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Subset metadata saved → {out_path}")
    return out_path


def build_datasets(
    cfg: Config,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Returns (train_dataset, val_dataset, test_dataset).
    Selects a balanced subset of 600 patients from DATA_ROOT, stratified by
    tumour volume, and writes subset_metadata.json documenting the split.

    Optimization: If subset_metadata.json exists, reloads the previous split
    to save 2-5 minutes of volume calculations.
    """
    import json
    if not cfg.DATA_ROOT.exists():
        raise FileNotFoundError(f"Data root not found: {cfg.DATA_ROOT}. Please ensure data is downloaded.")

    meta_path = cfg.RESULTS_DIR / "subset_metadata.json"
    if meta_path.exists():
        print(f"  Loading existing subset selection from {meta_path}...")
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            # Map patient names back to full paths
            def name_to_path(name):
                # Search for the directory in DATA_ROOT
                matches = list(cfg.DATA_ROOT.rglob(name))
                return matches[0] if matches else None

            tr_dirs   = [name_to_path(p) for p in meta["splits"]["train"]["patients"]]
            val_dirs  = [name_to_path(p) for p in meta["splits"]["validation"]["patients"]]
            test_dirs = [name_to_path(p) for p in meta["splits"]["test"]["patients"]]

            # Verify all paths exist
            if all(d and d.exists() for d in tr_dirs + val_dirs + test_dirs):
                print(f"  Successfully reloaded {len(tr_dirs)+len(val_dirs)+len(test_dirs)} patients.")
                train_ds = BraTSDataset(tr_dirs,   cfg, augment=True)
                val_ds   = BraTSDataset(val_dirs,  cfg, augment=False)
                test_ds  = BraTSDataset(test_dirs, cfg, augment=False)
                return train_ds, val_ds, test_ds
            else:
                print("  Some cached patients are missing from disk. Re-calculating subset...")
        except Exception as e:
            print(f"  Error loading metadata ({e}). Re-calculating subset...")

    patient_dirs = sorted(list(set([
        d for d in cfg.DATA_ROOT.rglob("BraTS*")
        if d.is_dir() and "GLI" in d.name and d.name.startswith("BraTS")
    ])))
    if len(patient_dirs) == 0:
        raise ValueError("No patient directories found in DATA_ROOT")

    print(f"  Total patient directories discovered: {len(patient_dirs)}")

    # Stratified subset selection
    tr_dirs, val_dirs, test_dirs = _stratified_subset_split(patient_dirs, cfg)

    print(f"  Subset — Train: {len(tr_dirs)}"
          f"  Val: {len(val_dirs)}  Test: {len(test_dirs)}"
          f"  Total: {len(tr_dirs) + len(val_dirs) + len(test_dirs)}")

    # Save metadata JSON
    _save_subset_metadata(tr_dirs, val_dirs, test_dirs, cfg)

    train_ds = BraTSDataset(tr_dirs,   cfg, augment=True)
    val_ds   = BraTSDataset(val_dirs,  cfg, augment=False)
    test_ds  = BraTSDataset(test_dirs, cfg, augment=False)
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CN6021 Task 2 — 3D Brain Tumour Segmentation"
    )
    p.add_argument("--auth-token", type=str, default=None,
                   help="Synapse auth token (overrides .env)")
    p.add_argument("--download", action="store_true",
                   help="Download dataset from Synapse before running")
    p.add_argument("--eval", action="store_true",
                   help="Evaluate only (requires --checkpoint)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint for evaluation")
    p.add_argument("--no_hp_search", action="store_true",
                   help="Skip hyperparameter search, use default config")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of training epochs")
    p.add_argument("--skip-eda", action="store_true",
                 help="Skip EDA step")
    p.add_argument("--eda-only", action="store_true",
                 help="Run EDA only and exit")
    p.add_argument("--finetune", action="store_true",
                 help="Fine-tune from best_model.pth with lower LR")
    p.add_argument("--finetune-epochs", type=int, default=25,
                 help="Number of fine-tuning epochs (default: 25)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config()

    if args.epochs:
        cfg.EPOCHS = args.epochs

    set_seed(cfg.SEED)
    make_dirs()

    print("=" * 65)
    print("CN6021 Task 2 — 3D Brain Tumour Segmentation")
    print(f"Device : {cfg.DEVICE.upper()}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 65)

    if args.auth_token:
        os.environ["SYNAPSE_AUTH_TOKEN"] = args.auth_token

    # Ensure dataset is downloaded and available before proceeding
    download_brats_data(cfg, args.auth_token)

    # ── EDA ──────────────────────────────────────────────────────────────────
    if not args.skip_eda:
        print("\nSTEP 1 — Exploratory Data Analysis")
        run_eda(cfg, cfg.DATA_ROOT, cfg.RESULTS_DIR)

    if args.eda_only:
        print("\nEDA complete. Exiting (--eda-only flag).")
        return

    # ── Build datasets ────────────────────────────────────────────────────────
    print("\nSTEP 2 — Building datasets")
    train_ds, val_ds, test_ds = build_datasets(cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=True,
        prefetch_factor=cfg.PREFETCH_FACTOR, persistent_workers=True,
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True,
        prefetch_factor=cfg.PREFETCH_FACTOR, persistent_workers=True,
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True,
        prefetch_factor=cfg.PREFETCH_FACTOR, persistent_workers=True,
    )

    # ── Evaluation-only mode ──────────────────────────────────────────────────
    if args.eval:
        if not args.checkpoint:
            print("ERROR: --eval requires --checkpoint <path>")
            sys.exit(1)
        print(f"\nEvaluation mode — loading {args.checkpoint}")
        model = UNet3D(cfg).to(cfg.DEVICE)
        load_checkpoint(args.checkpoint, model, cfg.DEVICE)
        metrics = test_evaluation(model, test_loader, cfg)
        plot_dice_per_class(metrics, cfg)
        save_sample_predictions(model, test_ds, cfg, n=cfg.SAVE_VIS_N)
        return

    # ── Fine-tune mode ────────────────────────────────────────────────────────
    if args.finetune:
        print("\nSTEP 4 — Building 3D U-Net for fine-tuning")
        model = UNet3D(cfg).to(cfg.DEVICE)

        # Load the best model from initial training as starting point
        ft_last  = str(cfg.CHECKPOINT_DIR / "finetune_last.pth")
        best_init = str(cfg.CHECKPOINT_DIR / "best_model.pth")

        if not Path(ft_last).exists() and not Path(best_init).exists():
            print("ERROR: No best_model.pth found. Run initial training first.")
            sys.exit(1)

        print("\nSTEP 5 — Fine-tuning")
        history = finetune(
            model, train_loader, val_loader, cfg,
            finetune_epochs=args.finetune_epochs,
        )
        plot_training_curves(history, cfg)

        # ── Load best fine-tuned checkpoint for evaluation ────────────────────
        ft_best = str(cfg.CHECKPOINT_DIR / "finetune_best.pth")
        print("\nSTEP 6 — Loading best fine-tuned checkpoint")
        if Path(ft_best).exists():
            load_checkpoint(ft_best, model, cfg.DEVICE)
        elif Path(best_init).exists():
            load_checkpoint(best_init, model, cfg.DEVICE)

        print("\nSTEP 7 — Test evaluation (fine-tuned)")
        metrics = test_evaluation(model, test_loader, cfg)

        print("\nSTEP 8 — Saving visualisations")
        plot_dice_per_class(metrics, cfg)
        save_sample_predictions(model, test_ds, cfg, n=cfg.SAVE_VIS_N)

        print(f"\n{'='*65}")
        print("Fine-tuning complete.")
        print(f"  Best checkpoint  → {cfg.CHECKPOINT_DIR}/finetune_best.pth")
        print(f"  Training curves  → {cfg.RESULTS_DIR}/training_curves.png")
        print(f"  Dice per class   → {cfg.RESULTS_DIR}/dice_per_class.png")
        print(f"  Predictions      → {cfg.PRED_DIR}/")
        print(f"  Metrics CSV      → {cfg.RESULTS_DIR}/metrics_summary.csv")
        print(f"{'='*65}")
        return

    # ── Hyperparameter search ─────────────────────────────────────────────────
    if not args.no_hp_search:
        print("\nSTEP 3 — Hyperparameter search")
        best_hp = hyperparameter_search(train_ds, val_ds, cfg)
        cfg.LR = best_hp["lr"]
        cfg.BASE_FILTERS = best_hp["base_filters"]
        cfg.DICE_WEIGHT = best_hp["dice_weight"]
        cfg.BCE_WEIGHT = 1.0 - cfg.DICE_WEIGHT
        # Restore full epoch count after search
        cfg.EPOCHS = Config.EPOCHS
    else:
        print("\nSTEP 3 — Skipping hyperparameter search (using defaults)")

    # ── Build final model ─────────────────────────────────────────────────────
    print("\nSTEP 4 — Building 3D U-Net")
    set_seed(cfg.SEED)
    model = UNet3D(cfg).to(cfg.DEVICE)
    plot_model_architecture_summary(model, cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nSTEP 5 — Training")
    history = train(model, train_loader, val_loader, cfg)
    plot_training_curves(history, cfg)

    # ── Load best checkpoint for evaluation ───────────────────────────────────
    print("\nSTEP 6 — Loading best checkpoint for test evaluation")
    best_ckpt = str(cfg.CHECKPOINT_DIR / "best_model.pth")
    if Path(best_ckpt).exists():
        load_checkpoint(best_ckpt, model, cfg.DEVICE)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nSTEP 7 — Test evaluation")
    metrics = test_evaluation(model, test_loader, cfg)

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\nSTEP 8 — Saving visualisations")
    plot_dice_per_class(metrics, cfg)
    save_sample_predictions(model, test_ds, cfg, n=cfg.SAVE_VIS_N)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Pipeline complete.")
    print(f"  Best checkpoint  → {cfg.CHECKPOINT_DIR}/best_model.pth")
    print(f"  Final checkpoint → {cfg.CHECKPOINT_DIR}/final_model.pth")
    print(f"  Training curves  → {cfg.RESULTS_DIR}/training_curves.png")
    print(f"  Dice per class   → {cfg.RESULTS_DIR}/dice_per_class.png")
    print(f"  Predictions      → {cfg.PRED_DIR}/")
    print(f"  Metrics CSV      → {cfg.RESULTS_DIR}/metrics_summary.csv")
    print(f"  HP search CSV    → {cfg.RESULTS_DIR}/hyperparameter_search.csv")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# 14.  Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
def finetune(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          Config,
    finetune_epochs: int = 25,
) -> Dict[str, list]:
    """Fine-tune from best_model.pth with lower LR, cosine decay, and resume."""

    device    = cfg.DEVICE
    criterion = CombinedLoss(cfg).to(device)

    # Fine-tune hyperparameters
    ft_lr = 1e-4                     # 10x lower than initial training
    optimizer = AdamW(model.parameters(), lr=ft_lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-7)
    scaler    = GradScaler("cuda", enabled=cfg.AMP and device == "cuda")

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "val_iou":    [], "lr":       [],
    }

    best_val_dice = -1.0
    patience_ctr  = 0
    start_epoch   = 1
    ft_best_ckpt  = str(cfg.CHECKPOINT_DIR / "finetune_best.pth")
    ft_last_ckpt  = str(cfg.CHECKPOINT_DIR / "finetune_last.pth")

    # ── Resume from finetune checkpoint if available ──────────────────────────
    if Path(ft_last_ckpt).exists():
        print(f"\n⟳ Resuming fine-tuning from {ft_last_ckpt}...")
        ckpt = torch.load(ft_last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "history" in ckpt:
            history = ckpt["history"]
        best_val_dice = ckpt.get("best_val_dice", -1.0)
        patience_ctr  = ckpt.get("patience_ctr", 0)
        start_epoch   = ckpt["epoch"] + 1
        print(f"   Resumed at epoch {start_epoch} "
              f"(best Dice so far: {best_val_dice:.4f}, "
              f"patience: {patience_ctr}/{cfg.PATIENCE})")
    else:
        # First fine-tune run — load best model from initial training
        best_init = str(cfg.CHECKPOINT_DIR / "best_model.pth")
        print(f"\n Loading best initial model from {best_init}...")
        ckpt = torch.load(best_init, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        init_dice = ckpt.get("metrics", {}).get("mean_dice", 0.0)
        best_val_dice = init_dice   # set baseline from initial training
        print(f"   Loaded initial best model (Val Dice = {init_dice:.4f})")
        print(f"   Fine-tuning will only save a new best if Dice > {init_dice:.4f}")

    print(f"\n{'='*65}")
    print(f"FINE-TUNING on {device.upper()}"
          + (f"  [{torch.cuda.get_device_name(0)}]"
             if device == "cuda" else ""))
    print(f"Epochs: {start_epoch}→{finetune_epochs}  |  Batch: {cfg.BATCH_SIZE}"
          f"  |  LR: {optimizer.param_groups[0]['lr']:.6f}  |  Patience: {cfg.PATIENCE}")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, finetune_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion,
                                   optimizer, scaler, device, cfg)
        val_m   = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_dice"].append(train_m["mean_dice"])
        history["val_dice"].append(val_m["mean_dice"])
        history["val_iou"].append(val_m["mean_iou"])
        history["lr"].append(lr)

        elapsed = time.time() - t0
        print(
            f"FT Epoch [{epoch:3d}/{finetune_epochs}]  "
            f"Train loss: {train_m['loss']:.4f}  "
            f"Train Dice: {train_m['mean_dice']:.4f}  |  "
            f"Val loss: {val_m['loss']:.4f}  "
            f"Val Dice: {val_m['mean_dice']:.4f}  "
            f"Val IoU: {val_m['mean_iou']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # Per-class Dice
        cls_str = "  ".join(
            f"{Config.CLASS_NAMES[c+1]}:{val_m['dice_cls'][c]:.3f}"
            for c in range(len(val_m["dice_cls"]))
        )
        print(f"           Per-class val Dice: {cls_str}")

        # Checkpoint: save best model
        if val_m["mean_dice"] > best_val_dice:
            best_val_dice = val_m["mean_dice"]
            patience_ctr  = 0
            save_checkpoint(model, optimizer, epoch, val_m, ft_best_ckpt,
                            scheduler, scaler, history,
                            best_val_dice, patience_ctr)
            print(f"           ✓ New fine-tuned best saved  (Dice={best_val_dice:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.PATIENCE:
                print(f"\nEarly stopping (fine-tune) at epoch {epoch}"
                      f" (no improvement for {cfg.PATIENCE} epochs)")
                break

        # Always save last checkpoint for resume
        save_checkpoint(model, optimizer, epoch, val_m, ft_last_ckpt,
                        scheduler, scaler, history,
                        best_val_dice, patience_ctr)

    # Save final fine-tuned weights
    ft_final = str(cfg.CHECKPOINT_DIR / "finetune_final.pth")
    save_checkpoint(model, optimizer, epoch, val_m, ft_final,
                    scheduler, scaler, history,
                    best_val_dice, patience_ctr)
    print(f"\nFine-tune final saved → {ft_final}")
    print(f"Fine-tune best saved  → {ft_best_ckpt}"
          f"  (Val Dice = {best_val_dice:.4f})")

    return history


if __name__ == "__main__":
    main()
