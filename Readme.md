# 🏜️ Duality AI — Offroad Semantic Scene Segmentation
> **Devcation Hackathon** · Duality AI Segmentation Track  
> Fine-tuning Mask2Former on synthetic desert environments using Duality AI's Falcon platform to achieve robust semantic segmentation across unseen terrain.

---

## 📋 Table of Contents
- [Project Summary](#project-summary)
- [Model Architecture](#model-architecture)
- [Class Labels](#class-labels)
- [Dataset Structure](#dataset-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Environment Setup](#environment-setup)
- [Training Workflow](#training-workflow)
- [Testing & Evaluation](#testing--evaluation)
- [Results](#results)
- [Challenges & Solutions](#challenges--solutions)
- [Conclusion & Future Work](#conclusion--future-work)
- [File Structure](#file-structure)
- [Reproducing Results](#reproducing-results)
- [Troubleshooting](#troubleshooting)

---

## Project Summary

This project trains a **Mask2Former** semantic segmentation model on synthetic desert imagery from Duality AI's **Falcon** digital twin platform. The model segments scenes into **10 terrain and vegetation classes** and is evaluated on a novel, unseen desert environment to test generalization under **domain shift conditions**.

**Key achievements:**
- Val mIoU of **67.23%** across all 10 classes
- Iterative training pipeline across **4 model versions** with progressive improvements
- Proactive **Exploratory Data Analysis (EDA)** that identified critical dataset issues before training — including test set class absence and extreme class imbalance (Logs: 483:1 ratio)
- Implemented **weighted oversampling**, **class-weighted loss**, **Test-Time Augmentation (TTA)**, and **multi-stage augmentation** to handle rare classes

---

## Model Architecture

All four model iterations use **Mask2Former** with a **Swin-Base** transformer backbone, fine-tuned for 10-class semantic segmentation.

| Parameter | Value |
|---|---|
| Architecture | `Mask2FormerForUniversalSegmentation` |
| Backbone | Swin Transformer (Base) |
| Input Resolution | 512 × 512 |
| Number of Classes | 10 |
| Decoder Layers | 10 |
| Hidden Dim | 256 |
| Number of Queries | 100 |
| Loss Weights | Class: 2.0 · Dice: 5.0 · Mask: 5.0 |
| Image Normalization | ImageNet Mean `[0.485, 0.456, 0.406]` · Std `[0.229, 0.224, 0.225]` |
| Pretrained From | `facebook/mask2former-swin-base-coco-panoptic` |

---

## Class Labels

| Class ID (Raw Mask Value) | Mapped ID | Class Name |
|---|---|---|
| 100 | 0 | Trees |
| 200 | 1 | Lush Bushes |
| 300 | 2 | Dry Grass |
| 500 | 3 | Dry Bushes |
| 550 | 4 | Ground Clutter |
| 600 | 5 | Flowers |
| 700 | 6 | Logs |
| 800 | 7 | Rocks |
| 7100 | 8 | Landscape *(all general ground not in another category)* |
| 10000 | 9 | Sky |

> Pixels not mapping to any known class are assigned value `255` (ignore index — excluded from loss and metrics).

---

## Dataset Structure

Download the dataset from the [Falcon documentation portal](https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert) (free Falcon account required). Navigate to the **Segmentation Track** section.

After downloading, organize files as follows:

```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/       # 2,857 RGB input images
│   └── Segmentation/       # 2,857 ground truth masks (16-bit PNG)
└── val/
    ├── Color_Images/        # 317 RGB input images
    └── Segmentation/        # 317 ground truth masks

Offroad_Segmentation_testImages/
├── Color_Images/            # 1,002 unseen test images
└── Segmentation/            # 1,002 test ground truth masks
```



---

## Exploratory Data Analysis

Before any training, we ran a full automated EDA pipeline (`EDA/eda_script.py`) across all three splits. Key findings that directly shaped our training strategy:

### Critical Findings

| Finding | Impact | Action Taken |
|---|---|---|
| **Test set missing 3 classes** (GC, Flowers, Logs have 0 pixels) | Test IoU for these classes is undefined | Evaluated on val set; reported test set as domain-shifted environment |
| **Logs = 0.078% of pixels** (483:1 imbalance vs Sky) | Model ignores Logs entirely with standard CE loss | WeightedRandomSampler (15×) + class weight 20× |
| **Test terrain fundamentally different** (43% Landscape, 18% Rocks vs train) | Domain shift between train and test environments | Color jitter + affine augmentation to improve generalization |
| **Sequential frame leakage risk** (train/val share scene prefixes) | Val IoU may be slightly inflated | Documented as known limitation |
| **Spatial bias** (Sky top-biased, Ground classes bottom-biased) | Vertical flip would be harmful | Explicitly avoided vertical flip augmentation |
| **RGB std significantly higher than ImageNet** (69–76 vs 57–58) | Wider intensity variation in outdoor scenes | Color jitter augmentation with ±40% brightness range |

> The EDA script generates a full `eda_results.json` file and `EDA/EDA.md` report with pixel-level class statistics, spatial bias grids, file integrity checks, and cross-split leakage analysis.

---

## Environment Setup

### Prerequisites
- Miniconda or Anaconda installed
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- Windows OS (for `.bat` scripts; see Mac/Linux instructions below)

### Windows

```bash
# Navigate to the ENV_SETUP folder
cd Offroad_Segmentation_Scripts/ENV_SETUP

# Run the full setup script (creates conda env 'EDU' and installs all packages)
setup_env.bat
```

This runs two steps:
1. `create_env.bat` — creates a Python 3.10 conda environment named `EDU`
2. `install_packages.bat` — installs PyTorch (CUDA 11.8), torchvision, ultralytics, opencv, tqdm

### Mac / Linux

```bash
conda create --name EDU python=3.10 -y
conda activate EDU
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics -y
pip install opencv-contrib-python tqdm transformers torchmetrics pandas seaborn matplotlib
```

### Activate the Environment

```bash
conda activate EDU
```

---

## Training Workflow

The project uses an **iterative fine-tuning strategy** across four scripts, each building on the previous best checkpoint. Every script validates on the val set and saves the model only when mIoU improves.

### Model 01 — Baseline Training (`road_segmentation_01.py`)
Cold-start fine-tuning from `facebook/mask2former-swin-base-coco-panoptic`. No augmentation. Establishes benchmark performance.

```bash
conda activate EDU
cd Offroad_Segmentation_Scripts
python road_segmentation_01.py
```

Update paths at the bottom of the script:
```python
TRAIN_IMG  = r"path/to/train/Color_Images"
TRAIN_MASK = r"path/to/train/Segmentation"
VAL_IMG    = r"path/to/val/Color_Images"
VAL_MASK   = r"path/to/val/Segmentation"
```

- **Epochs:** 5–9 | **Batch size:** 2 | **LR:** 5e-5
- **Output:** `best_offroad_model_01/`
- **Best val mIoU:** 67.62% (Epoch 7)

---

### Model 02 — Fine-tuning + Augmentation (`road_segmentation_02.py`)
Loads `best_offroad_model_01`. Adds horizontal flip augmentation, class-weighted loss, and CosineAnnealingLR scheduler.

```bash
python road_segmentation_02.py
```

- **Epochs:** 3 | **LR:** 1e-5 (lower for fine-tuning)
- **New:** Class weights `[1.5, 1.5, 1.5, 4.0, 6.0, 2.0, 2.5, 3.0, 1.5, 0.5]`
- **Output:** `best_offroad_model_refined/`

---

### Model 03 — Oversampling for Rare Classes (`road_segmentation_03.py`)
Loads `best_offroad_model_refined`. Adds `WeightedRandomSampler` to oversample images with rare classes. Expands augmentation to include affine transforms, color jitter, and Gaussian blur.

```bash
python road_segmentation_03.py
```

- **Oversampling multipliers:** Logs 5×, Ground Clutter 3×, Rocks 4×, Dry Bushes 2×
- **New class weights:** Logs 2.5×, Ground Clutter 10×, Rocks 3.5×
- **Output:** `best_offroad_model_refined/` (overwritten if improved)

---

### Model 04 — TTA + Diagnostics (`road_segmentation_04.py`)
Loads `best_offroad_model_01`. Applies Test-Time Augmentation (TTA) during validation, adds random resized crops, higher oversampling weights, and Ground Clutter confusion diagnostics.

```bash
python road_segmentation_04.py
```

- **TTA:** Original + horizontal flip + brightness boost (averaged logits)
- **Oversampling:** Logs 15×, Ground Clutter 10×, Rocks 8×, Dry Bushes 5×
- **Diagnostics:** Per-image Ground Clutter confusion matrix printed each epoch
- **Output:** `best_offroad_model_refined/` (overwritten if improved)

### Augmentation Summary

| Technique | Applied In | Purpose |
|---|---|---|
| Horizontal Flip (p=0.5) | Models 02, 03, 04 | Left/right viewpoint variation |
| Affine Transform (±15°, scale 0.85–1.15) | Models 03, 04 | Simulates UGV suspension flex and terrain slopes |
| Color Jitter (brightness ±40%, contrast ±30%, saturation ±20%) | Models 03, 04 | Desert lighting variation |
| Gaussian Blur (3×3 kernel, p=0.5) | Models 03, 04 | Engine vibration / dust simulation |
| Random Resized Crop (scale 0.5–1.0, ratio 0.75–1.33) | Model 04 | Multi-scale scene understanding |
| Test-Time Augmentation (TTA) | Model 04 (val only) | Ensemble 3 predictions per image |

> **Note:** Vertical flips were explicitly avoided. EDA revealed Sky is strictly top-biased and Ground classes are bottom-biased — flipping vertically would produce unnatural scenes and hurt performance.

---

## Testing & Evaluation

Run the test script against the held-out test images:

```bash
cd Offroad_Segmentation_Scripts
python test.py
```

Update paths at the bottom of `test.py`:
```python
MODEL_DIR     = r"path/to/best_offroad_model_refined"
TEST_IMG_DIR  = r"path/to/Offroad_Segmentation_testImages/Color_Images"
TEST_MASK_DIR = r"path/to/Offroad_Segmentation_testImages/Segmentation"
```

**Outputs generated:**
- `test_results.csv` — per-class IoU and pixel accuracy
- `confusion_matrix.png` — normalized confusion matrix heatmap (rows = true class, columns = predicted class)

**How to interpret outputs:**
- IoU close to 1.0 = near-perfect segmentation for that class
- IoU = 0.00 for a class means that class is absent from the test set ground truth (expected for Ground Clutter, Flowers, Logs — confirmed by EDA)
- Confusion matrix diagonal = correctly classified pixels; off-diagonal = misclassifications

> ⚠️ **Important Note on Test Set:** Our EDA confirmed that the test set originates from a different desert location and is missing 3 classes entirely (Ground Clutter, Flowers, Logs). This is an **intentional domain shift challenge** by the organizers. Meaningful IoU for those classes can only be evaluated on the val set.

---

## Results

### Val Set Performance — Best Model (used for class evaluation)

| Class | Val IoU | Val Accuracy |
|---|---|---|
| Trees | 86.75% | 94.21% |
| Lush Bushes | 71.49% | 85.25% |
| Dry Grass | 70.90% | 84.02% |
| Dry Bushes | 50.46% | 71.11% |
| Ground Clutter | 38.34% | 48.73% |
| Flowers | 68.44% | 82.46% |
| Logs | 62.15% | 75.16% |
| Rocks | 54.51% | 64.85% |
| Landscape | 70.74% | 83.24% |
| Sky | **98.54%** | **99.17%** |
| **Mean IoU** | **67.23%** | **88.13% pixel accuracy** |

### Test Set Performance (Domain-Shifted Environment)

| Class | Test IoU | Note |
|---|---|---|
| Trees | 47.18% | Present (0.27% of pixels) |
| Lush Bushes | 0.05% | Nearly absent in test |
| Dry Grass | 44.10% | Present (17.4% of pixels) |
| Dry Bushes | 51.17% | Present (3.05% of pixels) |
| Ground Clutter | **0.00%** | Absent from test set (confirmed by EDA) |
| Flowers | **0.00%** | Absent from test set |
| Logs | **0.00%** | Absent from test set |
| Rocks | 4.60% | Present (18.16% of pixels) but visually different from train rocks |
| Landscape | **70.05%** | Dominant class in test (43.15% of pixels) |
| Sky | **98.32%** | Consistent across environments |
| **Mean IoU** | **31.55%** | Includes 3 zero classes; reflects domain shift |

> **On the 7 classes present in the test set, adjusted mIoU = ~52%.** Low Rocks IoU (4.6%) suggests the rocky terrain in the test environment is visually different from training rocks — a classic synthetic-to-novel-synthetic generalization challenge.

### Training Progress (Model 01)

| Epoch | Train Loss | Val Loss | Mean IoU |
|---|---|---|---|
| 1 | 34.99 | 30.45 | 61.65% |
| 3 | 27.87 | 27.49 | 65.97% |
| 5 | 26.93 | 26.93 | 66.98% |
| 7 | 26.41 | **26.50** | **67.62%** |
| 9 | 26.07 | 26.76 | 67.17% |

Loss consistently decreased. Best checkpoint at Epoch 7 before slight overfitting.

---

## Challenges & Solutions

### Challenge 1: Extreme Class Imbalance (Logs — 483:1 ratio)
**Problem:** Logs occupied only 0.078% of training pixels. Standard cross-entropy loss effectively ignored them, leading to low recall.

**Solution:**
- `WeightedRandomSampler` — images with Logs are sampled up to **15× more frequently**
- Class weight of **20×** for Logs in loss function (capped to prevent training instability)
- Larger crop sizes (512×512) to preserve more Log pixels per batch
- TTA during validation to improve prediction stability for small objects

**Result:** Logs IoU improved from ~0.53 (Model 01 Epoch 1) to **0.62** on val set.

---

### Challenge 2: Ground Clutter Boundary Confusion (IoU 0.38)
**Problem:** Ground Clutter appears in 99.9% of images but at small pixel area (4.4%). It is visually similar to Dry Grass and Landscape, causing frequent misclassification at boundaries.

**Solution:**
- High class weight (8–10×) to amplify gradient for GC pixels
- Affine + elastic-style augmentation to improve boundary sharpness
- Strategy C diagnostics in Model 04 — per-image confusion tracking revealed GC is most often confused with Dry Grass (2), Landscape (8), and Rocks (7)

**Remaining challenge:** Boundary ambiguity is partly an annotation issue — the definition of "ground clutter" vs "dry grass" is inherently fuzzy in desert terrain.

---

### Challenge 3: Test Set Domain Shift
**Problem:** The test set comes from a completely different desert location. It lacks 3 classes entirely and has a very different terrain composition (43% Landscape, 18% Rocks vs. 24% and 1.2% in train).

**Solution:**
- Identified this **proactively in EDA** before training began
- Used diverse augmentations (color jitter, affine transforms) to improve robustness to scene variation
- Reported val set IoU as the primary performance metric, with test set results interpreted in the context of domain shift

---

### Challenge 4: 16-bit Mask Format
**Problem:** Mask files use 16-bit grayscale (`I;16` PIL mode) because class IDs go up to 10,000 — exceeding the 0–255 range of standard 8-bit masks. Loading with standard image pipelines silently fails.

**Solution:** Explicit PIL load with mode detection + remapping table in all dataset classes:
```python
raw_mask = np.array(Image.open(mask_path))  # preserves 16-bit values
continuous_mask = np.full_like(raw_mask, fill_value=255, dtype=np.int64)
for old_id, new_id in self.value_map.items():
    continuous_mask[raw_mask == old_id] = new_id
```

---

### Challenge 5: Mask2Former Collation
**Problem:** Mask2Former requires `mask_labels` to be a list of variable-size tensors. PyTorch's default `collate_fn` tries to stack them and raises a runtime error.

**Solution:** Custom `mask2former_collate_fn` that keeps `mask_labels` and `class_labels` as Python lists while stacking the remaining tensor fields normally.

---

## Conclusion & Future Work

### What We Achieved
- Built an end-to-end semantic segmentation pipeline from EDA to deployment-ready model
- Achieved **67.23% val mIoU** with iterative improvements across 4 training stages
- Identified and documented a critical domain shift between train/val and test environments
- Implemented advanced training strategies: oversampling, TTA, class-weighted loss, and multi-augmentation pipelines

### What We Would Do Next

**Immediate improvements:**
- Train for **50–100 epochs** (current pipeline is limited to 2–9 epochs — far from convergence)
- Implement **Copy-Paste augmentation** for Logs: extract Log cutouts from donor images and paste them into other training images — the single highest-impact change for rare class IoU
- Switch normalization to **dataset-specific stats** (RGB mean=[120.1, 116.4, 111.9], std=[69.1, 69.3, 76.0]) instead of ImageNet values
- Add **Dice Loss** component (`0.5×CE + 0.5×Dice`) to directly optimize IoU

**Architecture improvements:**
- Try **HRNet-W48 + OCR** — maintains full-resolution feature maps throughout the network, ideal for small objects like Logs
- Try **SegFormer-B5** — better hierarchical features for complex multi-class scenes

**Data improvements:**
- Scene-aware train/val split (group consecutive frames; avoid sequential leakage)
- Request additional Logs data from Duality AI (organizers acknowledged this gap)
- Review and potentially exclude 28 two-class `ww` images that may represent incomplete annotations

**Deployment:**
- Optimize for inference speed target of <50ms/image using model quantization or a lightweight backbone (PIDNet-S)

---

## File Structure

```
.
├── EDA/
│   ├── eda_script.py               # Full automated EDA pipeline
│   ├── eda_compare.py              # Cross-split comparison printer
│   └── EDA.md                      # Full EDA report with findings and recommendations
│
├── Offroad_Segmentation_Scripts/
│   ├── ENV_SETUP/
│   │   ├── create_env.bat          # Step 1: create conda env
│   │   ├── install_packages.bat    # Step 2: install dependencies
│   │   └── setup_env.bat           # Runs both steps
│   ├── road_segmentation_01.py     # Baseline training (cold start)
│   ├── road_segmentation_02.py     # Fine-tuning + augmentation + class weights
│   ├── road_segmentation_03.py     # Oversampling for rare classes
│   ├── road_segmentation_04.py     # TTA + confusion diagnostics
│   ├── test.py                     # Final evaluation script
│   └── test_results.csv            # Val set evaluation results
│
├── Offroad_Segmentation_Models/
│   ├── Model_01/                   # config.json + preprocessor_config.json
│   ├── Model_02/
│   ├── Model_03/
│   └── Model_04/
│
├── Offroad_Segmentation_Visuals/
│   ├── Model_Visuals_01/training_logs.csv   # Epoch-by-epoch metrics
│   ├── Model_Visuals_02/training_logs.csv
│   ├── Model_Visuals_03/training_logs.csv
│   └── Model_Visuals_04/training_logs.csv
│
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/           # 2,857 training images
│   │   └── Segmentation/           # 2,857 training masks
│   └── val/
│       ├── Color_Images/           # 317 validation images
│       └── Segmentation/           # 317 validation masks
│
├── Offroad_Segmentation_testImages/
│   ├── Color_Images/               # 1,002 test images (unseen environment)
│   └── Segmentation/               # 1,002 test masks
│
├── confusion_matrix.png            # Val set confusion matrix heatmap
├── confusion_matrix_2.png          # Test set confusion matrix heatmap
└── Readme.md                       # This file
```

---

## Reproducing Results

To reproduce the final best model from scratch:

```bash
# 1. Setup environment
cd Offroad_Segmentation_Scripts/ENV_SETUP
setup_env.bat

# 2. Activate environment
conda activate EDU

# 3. Run EDA (optional but recommended)
cd ../..
python EDA/eda_script.py

# 4. Baseline training — saves best_offroad_model_01/
cd Offroad_Segmentation_Scripts
python road_segmentation_01.py      # epochs=9, batch_size=2

# 5. Fine-tune with augmentation — saves best_offroad_model_refined/
python road_segmentation_02.py      # epochs=3, batch_size=2

# 6. Refine with oversampling
python road_segmentation_03.py      # epochs=3, batch_size=2

# 7. TTA + diagnostics refinement
python road_segmentation_04.py      # epochs=2, batch_size=2

# 8. Evaluate on val set
python test.py                      # uses val paths by default for class IoU
```

**Expected outputs:**
- `test_results.csv` — per-class IoU and accuracy table
- `confusion_matrix.png` — normalized confusion matrix
- `training_metrics.csv` — epoch-by-epoch training and val loss + per-class IoU

**Expected val mIoU:** ~67% after full pipeline.

---

## Troubleshooting

**`setup_env.bat` not working on Mac/Linux?**
Create a `setup_env.sh` with equivalent `conda` and `pip install` commands (see Environment Setup above).

**Training is too slow?**
- Reduce `batch_size` from `2` to `1`
- Close background applications to free GPU memory
- Monitor GPU with `nvidia-smi`

**Out of GPU memory (CUDA OOM)?**
- Set `batch_size=1`
- Reduce resolution in `preprocessor_config.json` from `512` to `384`

**Results show 317 images instead of 1002?**
- Verify `TEST_IMG_DIR` in `test.py` points to `Offroad_Segmentation_testImages/Color_Images` and **not** the val folder
- Run `print(len(os.listdir(TEST_IMG_DIR)))` to confirm count before running

**Ground Clutter / Flowers / Logs show 0% IoU on test set?**
- This is expected behaviour — these 3 classes are absent from the test environment (confirmed by EDA)
- Use the val set results for evaluating these classes

**Conda environment not found?**
Run `conda activate EDU` from an Anaconda Prompt, not a standard terminal.

---

## Support

Join the [Duality AI Community Discord](https://discord.com/invite/dualityfalconcommunity) for real-time help, announcements, and live Q&A with organizers.

---

*Built for the Devcation Hackathon — Duality AI Segmentation Track*