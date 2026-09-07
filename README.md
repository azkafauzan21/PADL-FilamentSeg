# PADL-FilamentSeg

**Physics-Aware Deep Learning for Solar Filament Segmentation**  
*MAGFiLO 1.0 Kaggle 2026 Competition Repository*

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Directory Structure & Isolation Rules](#2-data-directory-structure--isolation-rules)
3. [Installation & Configuration](#3-installation--configuration)
4. [Centralized Execution Guide (CLI)](#4-centralized-execution-guide-cli)
5. [Output Format](#5-output-format)
6. [Monitoring with TensorBoard](#6-monitoring-with-tensorboard)
7. [Reproducibility](#7-reproducibility)

---

## 1. Architecture Overview

### 1.1 Dual-Pipeline Design

PADL-FilamentSeg implements a **sequential three-stage pipeline** that bridges unsupervised heliophysical representation learning with supervised instance segmentation, with an intermediate pre-processing step to resolve I/O bottlenecks from massive FITS files.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PRE-PROCESSING — FITS → NPY Conversion (preprocess.py)                 │
│                                                                         │
│  Input : *.fits (16/32-bit, 2048×2048, from GONG network)              │
│  Ops   : AstropyWarning catch (corrupt skip) → NaN/Inf → 0.0 →         │
│          cv2.INTER_AREA downsample → float32 .npy                       │
│  Output: data/processed/fits/{train,test}/*.npy  (512×512, ~1 MB/file) │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Self-Supervised Pre-training (SimCLR)                        │
│                                                                         │
│  Input : NPY files (float32, 512×512, 1-channel)                        │
│  Model : SolarSimCLR  (ResNet50 with 1-channel conv1 override)          │
│  Loss  : NT-Xent Contrastive Loss                                       │
│  Output: simclr_final.pth  (backbone weights encoding plasma physics)   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  Weight Handoff
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Supervised Fine-Tuning (Mask2Former)                         │
│                                                                         │
│  Input : JPEG images + COCO JSON annotations (Kaggle labeled set)       │
│  Model : FilamentMask2Former  (HF Mask2Former + injected SimCLR backbone)│
│  Loss  : Bipartite Matching Loss (Hungarian + Dice/Focal)               │
│  Output: best_mask2former.pth  (smart-checkpointed by Panoptic Quality) │
└─────────────────────────────────────────────────────────────────────────┘
```

**The critical bridge** — backbone weights from Stage 1 are injected into `FilamentMask2Former.backbone` before supervised training begins, endowing the segmentation decoder with thermodynamic prior knowledge before it ever sees a single annotation.

---

### 1.2 Heliophysics Rationale: Why FITS for Pre-training?

JPEG images distributed in Kaggle competition datasets are compressed to **8-bit**, which irreversibly destroys subtle intensity gradients critical for distinguishing filament optical depth and plasma density. GONG network FITS files, by contrast, encode the **true thermodynamic dynamic range** of the solar chromosphere in 16-bit or 32-bit floating-point.

By pre-training SimCLR on FITS-derived NPY matrices, the backbone learns to detect pixel intensity fluctuations that represent genuine physical energy quantities — not display-optimized pixel values. This gives the model an intrinsic understanding of:

- **Plasma density gradients** encoded in H-alpha absorption strength
- **Thermal topology** of quiescent filament channels vs. surrounding chromospheric network
- **Limb darkening** as an atmospheric effect to discount — not a signal to segment

The FITS files are pre-processed offline to NPY format (512×512) using `cv2.INTER_AREA` downsampling, which preserves the **spatial flux ratio** between pixels by computing the true average over each 4×4 source region. This is physically correct: it is equivalent to integrating the H-alpha emission over a coarser spatial resolution, not interpolating pixel values.

The NT-Xent loss then acts as a **plasma distribution capture mechanism**: it forces the latent space to cluster representations of filament structures with identical thermodynamic topology even under extreme observational flux variation (e.g., equatorial vs. limb-darkened filaments), reducing sensitivity to instrument flux artifacts while preserving morphological invariants.

> **Physics constraint on augmentation:** RandomFlip is **strictly prohibited** across all transforms. Solar filaments exhibit *chirality* (dextral/sinistral magnetic helicity). Flipping an image falsifies the recorded magnetic orientation law, corrupting the physical correlation that the model is learning to encode. Augmentations are limited to translations, contrast shifts, and Gaussian blur.

---

## 2. Data Directory Structure & Isolation Rules

### Expected Layout

```
PADL-FilamentSeg/
├── data/
│   ├── raw/
│   │   ├── fits/
│   │   │   ├── train/          ← ✅ Input for preprocess (Stage 0a)
│   │   │   │   └── *.fits      ← Ribuan file FITS mentah 2048×2048
│   │   │   └── test/           ← ✅ Input for preprocess (Stage 0a)
│   │   │       └── *.fits      ← 180 file FITS (tidak boleh masuk SSL)
│   │   └── MAGFiLO_1.0_Kaggle_2026/
│   │       ├── train/
│   │       │   ├── train_images/    ← 707 JPEG berlabel untuk Stage 2
│   │       │   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json  (46.4 MB)
│   │       └── test/
│   │           └── test_images/    ← 180 JPEG unlabeled untuk inferensi
│   ├── processed/
│   │   └── fits/
│   │       ├── train/          ← ✅ Auto-generated oleh preprocess (Stage 0b)
│   │       │   └── *.npy       ← float32, 512×512, siap untuk SSL
│   │       └── test/           ← ✅ Auto-generated oleh preprocess (Stage 0b)
│   │           └── *.npy
│   ├── train_split.json        ← Auto-generated by extract_metadata (80%)
│   ├── val_split.json          ← Auto-generated by extract_metadata (20%)
│   └── download_fits_targets.csv
├── weights/
│   ├── simclr_final.pth
│   ├── best_mask2former.pth    ← Smart checkpoint (highest val PQ)
│   └── mask2former_final.pth
├── runs/                       ← TensorBoard logs (auto-created)
│   ├── simclr/
│   └── mask2former/
├── data/
│   └── submission.csv          ← Final output of generate_submission
├── config.yaml
└── main.py
```

### ⚠️ Strict Data Isolation Rules

| Rule | Description |
|---|---|
| **NPY Test Isolation** | `data/processed/fits/test/` is **absolutely off-limits** to the SSL training pipeline. The SSL dataloader uses a non-recursive glob on `npy_train_dir` and performs a path guard to raise `RuntimeError` if any `/test/` path contaminates the batch. |
| **FITS → NPY Pipeline** | Raw FITS files are **never read directly** during training. All I/O during SSL goes through pre-processed NPY files. This eliminates the astropy decompression bottleneck from the GPU-bound training loop. |
| **JPEG Test Purity** | `test_images/` contains **only unlabeled images**. The inference pipeline reads exclusively from `jpeg_test_dir` and accepts only `.jpg/.jpeg/.png` extensions. |
| **Anti-Leakage Split** | `extract_metadata` splits by unique `file_name` (physical file), not `image_id`. One physical solar observation can have multiple annotation instances; splitting by `image_id` alone would allow the same physical image to appear in both train and validation sets. |
| **Temporal Stratification** | The train/val split is stratified by **observation year** (4-digit prefix of timestamp). This guarantees that each year of the Solar Cycle 24–25 dataset (2011–2022) is proportionally represented in both partitions, preventing temporal bias. |

---

## 3. Installation & Configuration

### 3.1 Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd PADL-FilamentSeg

# Install Python dependencies
pip install -r requirements.txt

# Install torchmetrics (required for validation metrics)
pip install torchmetrics

# Install tensorboard (required for training visualization)
pip install tensorboard
```

**Full dependency list (`requirements.txt`):**

| Package | Purpose |
|---|---|
| `torch>=2.0.0`, `torchvision` | Core deep learning framework |
| `transformers` | Hugging Face Mask2Former implementation |
| `pycocotools` | RLE encoding/decoding for masks |
| `astropy`, `sunpy` | FITS file I/O (used only in `preprocess.py`, not in training) |
| `opencv-python` | INTER_AREA downsampling in preprocess + JPEG loading in inference |
| `albumentations` | Physics-aware augmentation pipeline |
| `omegaconf` | YAML configuration management |
| `scikit-learn` | Anti-leakage train/val splitting with temporal stratification |
| `pandas`, `numpy` | Data manipulation and array ops |
| `tqdm` | Training progress bars |

---

### 3.2 Configure `config.yaml`

All pipeline parameters are controlled by a single centralized YAML file. Edit it before running any pipeline stage.

```yaml
system:
  seed: 42                    # Global random seed for full reproducibility
  workers: 4                  # DataLoader parallel workers
  fp16_precision: true        # Enable AMP (set false if GPU does not support FP16)
  data_dir: "./data"
  weights_dir: "./weights"

  # --- Raw FITS paths (input for preprocess.py only) ---
  fits_train_dir: "./data/raw/fits/train"
  fits_test_dir:  "./data/raw/fits/test"

  # --- Pre-processed NPY paths (output of preprocess, input for train_simclr) ---
  npy_train_dir:  "./data/processed/fits/train"
  npy_test_dir:   "./data/processed/fits/test"

  # --- JPEG paths for supervised training and inference ---
  jpeg_train_dir: "./data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"
  jpeg_test_dir:  "./data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images"

  # --- COCO annotation paths ---
  coco_annotation_path: "./data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
  train_split_json: "./data/train_split.json"
  val_split_json:   "./data/val_split.json"

physics_parameters:
  percentile_clip_lower: 1    # Lower percentile bound for intensity clipping
  percentile_clip_upper: 99   # Upper percentile bound — removes cosmic ray spikes
  test_size_split: 0.2        # Fraction of images held out for validation

ssl_training:
  backbone: "resnet50"
  latent_dim: 128
  batch_size: 16
  epochs: 100
  learning_rate: 0.0003
  weight_decay: 0.0001
  temperature: 0.1            # NT-Xent softmax temperature

supervised_training:
  batch_size: 4
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0001
  freeze_backbone_epochs: 5   # Epochs with backbone frozen (Phase 1 of fine-tuning)
```

**Key configuration notes:**

- **`fits_train_dir` / `fits_test_dir`**: Used exclusively by `preprocess.py`. After preprocessing is done, these paths are no longer read by any training pipeline.
- **`npy_train_dir`**: The **only** data source for `train_simclr`. Populated by running `--mode preprocess`.
- **`percentile_clip_lower/upper`**: Standard astronomical curation technique. Clips intensity outliers caused by cosmic rays and impulsive solar flare emission.
- **`freeze_backbone_epochs`**: During the first N epochs of Stage 2, the ResNet backbone is frozen. Only the Pixel Decoder and Transformer Decoder are trained. After epoch N, the backbone is unfrozen with a 10× lower learning rate (differential LR) to prevent catastrophic forgetting of SSL representations.
- **`temperature`**: NT-Xent temperature. Lower values create sharper contrastive distributions; `0.1` is empirically stable for solar imagery.

---

## 4. Centralized Execution Guide (CLI)

All pipeline stages are executed through a **single entry point**: `main.py`.

```
usage: main.py [-h] --mode {extract_metadata,preprocess,train_simclr,train_mask2former,generate_submission}
               [--config CONFIG]
               [--workers N] [--batch_size N] [--epochs N] [--lr LR]

PADL-FilamentSeg: Physics-Aware Deep Learning for Solar Filaments

required arguments:
  --mode        Pipeline stage to execute. One of:
                  extract_metadata    — Parse COCO JSON, create anti-leakage train/val split
                  preprocess          — Convert FITS 2048×2048 → NPY 512×512 (run before train_simclr)
                  train_simclr        — Stage 1: SSL pre-training on pre-processed NPY data
                  train_mask2former   — Stage 2: Supervised fine-tuning on JPEG+COCO data
                  generate_submission — Stage 3: Inference and submission CSV generation

optional arguments:
  --config      Path to OmegaConf YAML config file (default: config.yaml)
  --workers N   Override config.system.workers for all modes
  --batch_size N  Override training batch size (mode-routed, see table below)
  --epochs N    Override number of training epochs (mode-routed, see table below)
  --lr LR       Override optimizer learning rate (mode-routed, see table below)
```

### Dynamic Override Routing

The flat CLI flags `--batch_size`, `--epochs`, and `--lr` are **mode-aware**: `main.py` automatically routes them to the correct config block depending on `--mode`. Only arguments explicitly passed are overridden; unspecified flags retain their `config.yaml` values.

| Flag | `train_simclr` routes to | `train_mask2former` routes to | Other modes |
|---|---|---|---|
| `--workers` | `config.system.workers` | `config.system.workers` | `config.system.workers` |
| `--batch_size` | `config.ssl_training.batch_size` | `config.supervised_training.batch_size` | *(ignored)* |
| `--epochs` | `config.ssl_training.epochs` | `config.supervised_training.epochs` | *(ignored)* |
| `--lr` | `config.ssl_training.learning_rate` | `config.supervised_training.learning_rate` | *(ignored)* |

---

### Stage 0a — Extract Metadata & Build Splits

Parse the Kaggle COCO annotation JSON, generate FITS download targets, and create anti-leakage train/val split files. **Must be run once before any training stage.**

```bash
python main.py --mode extract_metadata
```

**What happens internally:**
1. Parses `MAGFiLO_1.0_Annotations_kaggle2026_train.json`.
2. Extracts timestamp and GONG station code from each `file_name` via regex `(\d{14})` and `([A-Z])h\.`.
3. Builds `download_fits_targets.csv` with columns `file_name`, `timestamp`, `year`, `station_code`.
4. Splits images 80/20 using `sklearn.train_test_split` with **temporal stratification** by `year` — ensuring every observation year is proportionally represented in both partitions.
5. Writes COCO-format subset JSONs for each partition.

**Outputs generated:**
```
data/download_fits_targets.csv   # FITS download target list (with station code + year)
data/train_split.json            # 80% of images → training set (COCO subset, year-stratified)
data/val_split.json              # 20% of images → validation set (COCO subset, year-stratified)
```

---

### Stage 0b — Pre-process FITS → NPY

Convert raw FITS files to lightweight, training-ready NPY arrays. **Must be run before `train_simclr`.**

```bash
python main.py --mode preprocess
```

This step is intentionally decoupled from training to eliminate I/O bottlenecks. `astropy.io.fits` decompression and FITS header parsing are expensive operations — running them inside the GPU-bound training loop creates severe CPU/GPU starvation. Pre-processing amortizes this cost to a one-time offline job.

**What happens internally:**
1. Scans all `*.fits` files in `fits_train_dir` and `fits_test_dir`.
2. Opens each file with `astropy.io.fits`. If `AstropyWarning` is raised (truncated / corrupt file), the file is **silently skipped** and logged.
3. Replaces all `NaN` and `Inf` values with `0.0`.
4. Downsamples `2048×2048 → 512×512` using **`cv2.INTER_AREA`** (area-weighted average — spatially conservative for H-alpha flux data).
5. Saves as `float32` `.npy` to the mirrored directory structure under `data/processed/fits/`.
6. **Idempotent**: already-converted files are skipped on re-runs.

> **Known anomaly handled:** The file `data/raw/fits/train/20110126130634Ch.fits` (~712 KB vs. ~2.6 MB typical) was identified during the 2026-09-07 audit as a likely corrupt or truncated file. This file will trigger an `AstropyWarning` and be automatically skipped by the pre-processor.

**Outputs generated:**
```
data/processed/fits/train/*.npy   # float32, 512×512 — input for train_simclr
data/processed/fits/test/*.npy    # float32, 512×512 — available for analysis
```

**Standalone invocation (optional):**
```bash
# Can also be run directly without going through main.py
python pipelines/preprocess.py --config config.yaml
```

---

### Stage 1 — SimCLR Self-Supervised Pre-training

Train the solar physics backbone on pre-processed NPY images using contrastive learning. No labels required.

```bash
# Default: menggunakan seluruh nilai dari config.yaml
python main.py --mode train_simclr

# Override batch size dan learning rate untuk eksperimen cepat
python main.py --mode train_simclr --batch_size=16 --lr=0.0005

# Kurangi workers dan epochs untuk debugging di mesin lokal
python main.py --mode train_simclr --workers=2 --epochs=5 --batch_size=8

# Dengan custom config file
python main.py --mode train_simclr --config config_v2.yaml --lr=0.0003
```

**What happens internally:**
1. Loads all `*.npy` files from `npy_train_dir` (non-recursive, test partition guarded).
2. Applies physics-aware augmentation (no flip, percentile normalization, contrast jitter, blur) via `albumentations`.
3. Trains `SolarSimCLR` (ResNet50, 1-channel input) with NT-Xent loss for `ssl_training.epochs` epochs.
4. Saves a per-epoch checkpoint and a final weight file.

> **Note:** `train_simclr` reads `*.npy` via `np.load()` — O(1) I/O with no decompression overhead. This replaces the previous `astropy.io.fits` path that caused training throughput bottlenecks.

**Outputs:**
```
weights/simclr_epoch_{n}.pth     # Per-epoch checkpoints
weights/simclr_final.pth         # Final backbone weights → input for Stage 2
runs/simclr/                     # TensorBoard logs (SimCLR/NTXentLoss_iter, _epoch)
```

---

### Stage 2 — Mask2Former Supervised Fine-Tuning

Fine-tune the instance segmentation head using labeled JPEG images from the Kaggle dataset. Requires `simclr_final.pth` from Stage 1.

```bash
# Default: menggunakan seluruh nilai dari config.yaml
python main.py --mode train_mask2former

# Override epochs dan batch size (diarahkan ke supervised_training)
python main.py --mode train_mask2former --batch_size=4 --epochs=20

# Fine-tuning dengan learning rate lebih kecil dan lebih sedikit worker
python main.py --mode train_mask2former --lr=0.00005 --workers=2
```

**What happens internally:**
1. Loads `FilamentMask2Former` and injects `simclr_final.pth` backbone weights (`strict=False`).
2. **Phase 1** (epochs 1–`freeze_backbone_epochs`): backbone frozen, only decoder parameters trained.
3. **Phase 2** (subsequent epochs): backbone unfrozen with 10× differential learning rate.
4. After each training epoch, runs a **validation loop** on `val_split.json` to compute:
   - `DiceScore` (via `torchmetrics`) — pixel-level overlap quality
   - `Panoptic Quality (PQ)` — instance-level matching quality (IoU-based greedy matching)
5. **Smart checkpointing**: saves `best_mask2former.pth` **only** when validation PQ exceeds the current best.

**Outputs:**
```
weights/mask2former_epoch_{n}.pth  # Per-epoch checkpoints
weights/mask2former_final.pth      # Final weights (last epoch)
weights/best_mask2former.pth       # Best weights by validation PQ  ← used for inference
runs/mask2former/                  # TensorBoard logs (see Section 6)
```

---

### Stage 3 — Inference & Submission Generation

Run inference on the unlabeled test set and generate a Kaggle-ready `submission.csv`.

```bash
python main.py --mode generate_submission
```

**What happens internally:**
1. Loads `best_mask2former.pth` (falls back to `mask2former_final.pth` if best not found).
2. Reads all JPEG images from `jpeg_test_dir`.
3. For each image: applies percentile normalization → runs model → separates per-query instance masks.
4. Encodes each instance mask to RLE using `pycocotools`.
5. Writes `data/submission.csv` with header `filament_id,segmentation_rle`.

**Output:**
```
data/submission.csv
```

---

### Complete End-to-End Execution Sequence

```bash
# Step 0a: Parse COCO annotations and create year-stratified train/val splits
python main.py --mode extract_metadata

# Step 0b: Convert raw FITS 2048×2048 → NPY 512×512 (one-time offline job)
python main.py --mode preprocess

# Step 1: Pre-train physics backbone on NPY data (GPU recommended)
python main.py --mode train_simclr

# Step 2: Fine-tune segmentation model on labeled JPEG data
python main.py --mode train_mask2former

# Step 3: Generate Kaggle submission CSV
python main.py --mode generate_submission
```

> **Checkpoint note:** Steps 0a and 0b only need to be run **once** per dataset. They are fully idempotent — re-running them will not overwrite existing NPY files or regenerate splits with a different random state (as long as `config.system.seed` is unchanged).

---

## 5. Output Format

### Submission CSV Specification

The inference pipeline produces `data/submission.csv` strictly conforming to the **Kaggle MAGFiLO competition format**.

| Column | Type | Description |
|---|---|---|
| `filament_id` | `string` | Unique identifier: `{image_stem}_{instance_index}` |
| `segmentation_rle` | `string` | Fortran-order Run-Length Encoding of the binary mask |

**Example `submission.csv`:**

```
filament_id,segmentation_rle
20120506124500Ch_1,56 3 120 5 188 7 ...
20120506124500Ch_2,892 2 961 4 1029 6 ...
20120507083200Lh_1,234 8 302 10 370 12 ...
20120507093100Bh_0,
```

**Format guarantees:**

- **Multi-instance separation**: Each independently detected filament query produces a separate row. A single solar image with 3 detected filaments generates rows `{stem}_1`, `{stem}_2`, `{stem}_3`.
- **No-detection sentinel**: Images with no active predictions produce a single row `{stem}_0` with an empty `segmentation_rle` field, ensuring every test image is represented.
- **RLE encoding**: Uses `pycocotools.mask.encode()` with Fortran-contiguous array layout — identical to the evaluator used by the competition judge.
- **Quote-free RLE**: CSV is written with `quoting=csv.QUOTE_NONE` to guarantee RLE strings are not wrapped in `'` or `"` characters, which would cause parsing failures in the evaluator.

### RLE Encoding Details

The binary mask for each filament instance is encoded as follows:

```python
import numpy as np
from pycocotools import mask as mask_utils

# binary_mask: np.ndarray of shape [H, W], dtype uint8, values {0, 1}
mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
encoded = mask_utils.encode(mask_fortran)   # COCO-standard Fortran-order RLE
rle_str = encoded["counts"].decode("utf-8")
```

The resulting RLE string encodes run lengths of alternating 0s and 1s in **column-major (Fortran) order**, compatible with `pycocotools.mask.decode()` for metric computation.

### Evaluation Metric: Panoptic Quality (PQ)

The competition evaluates predictions using **Panoptic Quality**:

$$PQ = \frac{\sum_{(p, g) \in TP} \text{IoU}(p, g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$$

Where a predicted mask $p$ and ground-truth mask $g$ form a True Positive (TP) if and only if $\text{IoU}(p, g) > 0.5$. Unmatched predictions are False Positives (FP) and unmatched ground-truth instances are False Negatives (FN).

The same PQ formula is used **internally during validation** (`train_mask2former.py`) to drive smart checkpointing — ensuring the model saved for inference is the one that maximizes the competition metric, not just training loss.

---

## 6. Monitoring with TensorBoard

Both training stages log metrics to the `runs/` directory using `torch.utils.tensorboard.SummaryWriter`.

```bash
# Launch TensorBoard from the project root
tensorboard --logdir runs/
```

Navigate to `http://localhost:6006` to view real-time curves.

### Available Metrics

| Dashboard Tag | Source | Description |
|---|---|---|
| `SimCLR/NTXentLoss_iter` | `train_simclr` | NT-Xent loss per batch iteration |
| `SimCLR/NTXentLoss_epoch` | `train_simclr` | NT-Xent loss averaged per epoch |
| `Train/BipartiteLoss_iter` | `train_mask2former` | Bipartite matching loss per batch iteration |
| `Train/BipartiteLoss_epoch` | `train_mask2former` | Bipartite matching loss averaged per epoch |
| `Val/DiceScore` | `train_mask2former` | Validation Dice score per epoch |
| `Val/PanopticQuality` | `train_mask2former` | Validation PQ per epoch (drives smart checkpointing) |

---

## 7. Reproducibility

Global stochastic state is locked at startup via `utils/reproducibility.py`:

```python
# Applies automatically when config.system.seed is set
set_seed(seed=42)
```

This freezes: Python `random`, NumPy, PyTorch CPU/CUDA seeds, CuDNN deterministic mode, and `PYTHONHASHSEED`. To reproduce a specific run, ensure `config.system.seed` is set identically and `fp16_precision` matches your hardware configuration.

The temporal stratification in `extract_metadata` is also deterministic: `sklearn.train_test_split` receives `random_state=config.system.seed`, which is initialized **before** `extract_data_routine` is called from `main.py`.

> **Note on CuDNN determinism:** Setting `torch.backends.cudnn.deterministic = True` may reduce throughput on some GPU architectures. Set `fp16_precision: false` if you encounter non-deterministic behavior with AMP enabled.

---

## References

- **Dataset**: MAGFiLO 1.0 — Kaggle 2026 Solar Filament Segmentation Competition
- **Instrument**: GONG (Global Oscillation Network Group) H-alpha network
- **SimCLR**: Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (ICML 2020)
- **Mask2Former**: Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation* (CVPR 2022)
- **Panoptic Quality**: Kirillov et al., *Panoptic Segmentation* (CVPR 2019)
- **Chirality**: Martin et al., *Filament chirality and the solar cycle* (Solar Physics, 1994)
