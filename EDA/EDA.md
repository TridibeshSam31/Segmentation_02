# Offroad Segmentation Dataset — Full Exploratory Data Analysis

> **Dataset**: Duality Offroad Semantic Segmentation  
> **Date**: April 2026  
> **Analyst**: Automated EDA pipeline  
> **Current Model**: Mask2Former (Swin backbone), Mean IoU ≈ 0.66  
> **Problem Classes**: Ground Clutter (IoU 0.38), Logs (IoU 0.62), Rocks (0.55), Dry Bushes (0.50)

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Total image-mask pairs | **4,176** (train 2,857 + val 317 + test 1,002) |
| Resolution | **960 × 540** (uniform across all splits) |
| Image format | PNG, RGB |
| Mask format | PNG, 16-bit grayscale (`I;16`) |
| Number of classes | **10** (no background class; no ignore label in ground truth) |
| Corrupted files | **0** |
| Size mismatches | **0** |
| Label coverage | **100%** of pixels are labeled (no ignore regions) |

> [!CAUTION]
> **Critical Finding**: The test set is missing 3 classes entirely — **Ground Clutter, Flowers, and Logs have zero pixels** in all 1,002 test images. This means the test set **cannot evaluate** the two classes you care about most. Any IoU numbers for these classes from the test set are meaningless and can not be used.

> [!WARNING]
> **Severe Class Imbalance**: Logs occupy only **0.078%** of training pixels — a **483× imbalance** vs. the most common class (Sky at 37.6%). Ground Clutter is at 4.4% but is a weak class for other reasons (boundary confusion with Dry Grass and Landscape).

### Key Numerical Facts

| # | Finding | Impact |
|---|---|---|
| 1 | Logs = 0.078% of pixels (1.15M out of 1.48B) | 483:1 imbalance — near-invisible to standard CE loss |
| 2 | Test split has 0 pixels of GC, Flowers, Logs | Test IoU for these classes is undefined |
| 3 | All images are exactly 960×540 | No resize needed; crop/pad decisions are straightforward |
| 4 | Sky = 37.6%, Landscape = 24.4%, Dry Grass = 18.9% | Top-3 classes consume 81% of all pixels |
| 5 | RGB mean = [120.1, 116.4, 111.9], std = [69.1, 69.3, 76.0] | Significantly different from ImageNet; use dataset-specific normalization |
| 6 | Filenames have scene-prefix patterns (cc, mt, w, ww) | Sequential frames from different sensors/locations — leakage risk |
| 7 | 100% of pixels are labeled (no ignore regions at all) | Simplifies training but means errors in annotation have higher impact |
| 8 | Spatial bias: Sky → top rows, Landscape/Ground → bottom rows | Strong vertical position bias for most classes |

---

## 2. Dataset Inventory

### 2.1 File Counts

| Split | Images | Masks | Matched Pairs | Unmatched |
|---|---|---|---|---|
| **Train** | 2,857 | 2,857 | 2,857 | 0 |
| **Val** | 317 | 317 | 317 | 0 |
| **Test** | 1,002 | 1,002 | 1,002 | 0 |
| **Total** | 4,176 | 4,176 | 4,176 | 0 |

- **Every image has exactly one matching mask** (same filename).
- No missing pairs, no duplicates, no ambiguous matches.
- Train/Val ratio ≈ 90%/10%.

### 2.2 Naming Conventions

Files use **prefixed numeric names**. Train and Val use scene-based prefixes; Test uses bare numbers.

| Split | Prefixes | Examples | Count per Prefix |
|---|---|---|---|
| Train | `cc`, `mt`, `w`, `ww` | `cc0000012.png`, `mt10000543.png`, `ww10000290.png` | cc: varies, mt: varies, w: varies, ww: varies |
| Val | `cc`, `mt`, `w`, `ww` | Same naming pattern | cc: 100, mt: 108, w: 46, ww: 63 |
| Test | (none) | `0000060.png` to `0001061.png` | 1,002 files, sequential numbering |

> [!NOTE]
> The prefixes appear to correspond to different recording sessions or sensor setups (e.g., `cc` = camera-C, `mt` = camera-MT, `ww` = wide-angle). Within each prefix, numbers are sequential, suggesting **video frames extracted at some interval**. This is important for leakage analysis.

### 2.3 Filename Gaps in Test Set

Test files range from `0000060` to `0001061` (spanning 1,002 values out of a possible 1,002 range). **No gaps** — test set is a contiguous block.

---

## 3. File Integrity

| Check | Result |
|---|---|
| Corrupted images (all splits) | **0** |
| Corrupted masks (all splits) | **0** |
| Image-mask size mismatches | **0** |
| All files loadable with PIL | ✅ Yes |

### 3.1 Image Modes

| Split | Image Mode | Mask Mode |
|---|---|---|
| Train | RGB (2,857) | I;16 (2,857) |
| Val | RGB (317) | I;16 (317) |
| Test | RGB (1,002) | I;16 (1,002) |

> [!IMPORTANT]
> Masks use **16-bit unsigned integer mode (`I;16`)**, not 8-bit grayscale. This is because class IDs go up to **10,000** (Sky), which exceeds the 0–255 range. The pixel values are raw class IDs that must be remapped before training.

### 3.2 File Size Distribution

| Metric | Images (bytes) | Masks (bytes) |
|---|---|---|
| Min | 272,633 (267 KB) | 2,180 (2 KB) |
| Max | 1,132,362 (1.1 MB) | 146,793 (143 KB) |
| Mean | 808,330 (790 KB) | 68,728 (67 KB) |
| Median | 847,115 (827 KB) | 61,907 (60 KB) |

- Mask files are ~10× smaller than images (expected for indexed label maps).
- Very small mask files (2 KB) likely correspond to simple scenes with few class boundaries.

---

## 4. Resolution and Geometry

| Metric | Value |
|---|---|
| Width | **960** (all images, all splits) |
| Height | **540** (all images, all splits) |
| Aspect ratio | **1.778 (16:9)** |
| Unique resolutions | **1** |

- **Perfectly uniform resolution**. No resizing, padding, or cropping is needed for consistency.
- Aspect ratio of 16:9 is standard for dashcam/robot-cam footage.
- All masks are identically sized to their corresponding images.

**Recommendation**: Use a **crop size of 512×512** (as the existing training script does) or **480×480** for training. The 960×540 input fits well with a 2× downscale to 480×270 if memory is tight.

---

## 5. Label Encoding Audit

### 5.1 Raw Pixel Values in Masks

All unique values found across all splits:

| Pixel Value | Mapped Class ID | Class Name |
|---|---|---|
| 100 | 0 | Trees |
| 200 | 1 | Lush Bushes |
| 300 | 2 | Dry Grass |
| 500 | 3 | Dry Bushes |
| 550 | 4 | Ground Clutter |
| 600 | 5 | Flowers |
| 700 | 6 | Logs |
| 800 | 7 | Rocks |
| 7100 | 8 | Landscape |
| 10000 | 9 | Sky |

### 5.2 Audit Results

| Check | Result |
|---|---|
| Expected classes found | ✅ All 10 |
| Unexpected pixel values | ✅ **None** |
| Contiguous after remapping? | ✅ Yes (0–9) |
| Ignore label (255) in ground truth? | ❌ **No** — 100% of pixels are labeled |
| Background class? | **No explicit background** — all pixels belong to one of the 10 classes |

> [!NOTE]
> The remapping in the training code uses `fill_value=255` for unmapped pixels, but since every pixel maps to a valid class, this never triggers. The `ignore_index=255` in the loss function is effectively unused. This is clean.

### 5.3 Classes Missing from Test Set

| Class | Train Pixels | Val Pixels | Test Pixels |
|---|---|---|---|
| Ground Clutter (550) | 65,082,995 | present | **0** |
| Flowers (600) | 41,585,811 | present | **0** |
| Logs (700) | 1,153,995 | present | **0** |

> [!CAUTION]
> The test set masks only contain values `{100, 200, 300, 500, 800, 7100, 10000}`. Ground Clutter, Flowers, and Logs are **completely absent**. This means the test set comes from a **different scene/environment** that lacks these classes. **You cannot use the test set to evaluate GC/Logs IoU**.

---

## 6. Class Distribution

### 6.1 Global Pixel Distribution (Train)

| Class | Pixel Count | % of Total | Relative Weight (inv-freq) |
|---|---|---|---|
| Sky | 557,364,526 | **37.64%** | 1.00× |
| Landscape | 362,120,221 | **24.45%** | 1.54× |
| Dry Grass | 279,430,843 | **18.87%** | 1.99× |
| Lush Bushes | 87,892,776 | **5.93%** | 6.34× |
| Ground Clutter | 65,082,995 | **4.39%** | 8.57× |
| Trees | 52,331,525 | **3.53%** | 10.65× |
| Flowers | 41,585,811 | **2.81%** | 13.41× |
| Rocks | 17,743,187 | **1.20%** | 31.42× |
| Dry Bushes | 16,268,713 | **1.10%** | 34.27× |
| **Logs** | **1,153,995** | **0.078%** | **483.17×** |

> [!WARNING]
> **Logs are 483× rarer than Sky by pixel count.** This is the single most extreme imbalance in the dataset. Without aggressive countermeasures (oversampling, class-weighted loss, copy-paste augmentation), the model will effectively learn to ignore Logs.

### 6.2 Class Presence per Image (Train)

| Class | Images Present | % of 2,857 |
|---|---|---|
| Ground Clutter | 2,855 | **99.9%** |
| Landscape | 2,855 | **99.9%** |
| Dry Grass | 2,826 | **98.9%** |
| Lush Bushes | 2,806 | **98.2%** |
| Rocks | 2,823 | **98.8%** |
| Sky | 2,587 | **90.6%** |
| Trees | 2,247 | **78.7%** |
| Logs | **1,685** | **59.0%** |
| Flowers | 1,010 | **35.4%** |
| Dry Bushes | 914 | **32.0%** |

**Key insight**: Ground Clutter appears in 99.9% of training images, so its low IoU (0.38) is **not** due to scarcity of occurrence, but rather:
1. Small pixel area per image (4.4% average)
2. Boundary confusion with visually similar classes (Dry Grass, Landscape)
3. Inconsistent annotation of the GC/Landscape/Dry Grass boundary

Logs appear in 59% of images but occupy only 0.078% of pixels — they are **tiny objects present in many images** (scattered sticks/debris on the ground).

### 6.3 Cross-Split Distribution Comparison

| Class | Train % | Val % | Test % | Train vs Val Δ |
|---|---|---|---|---|
| Trees | 3.53 | 4.07 | 0.27 | 0.54 |
| Lush Bushes | 5.93 | 6.02 | 0.00 | 0.08 |
| Dry Grass | 18.87 | 19.34 | 17.40 | 0.47 |
| Dry Bushes | 1.10 | 1.10 | 3.05 | 0.00 |
| Ground Clutter | 4.39 | 4.24 | 0.00 | 0.16 |
| Flowers | 2.81 | 2.44 | 0.00 | 0.37 |
| Logs | 0.08 | 0.07 | 0.00 | 0.01 |
| Rocks | 1.20 | 0.68 | 18.16 | 0.52 |
| Landscape | 24.45 | 24.15 | 43.15 | 0.30 |
| Sky | 37.64 | 37.76 | 17.96 | 0.12 |

> [!IMPORTANT]
> Train and Val distributions are **well-matched** (Δ < 0.55% for all classes). This suggests they were split from the same scene pool. However, the **Test set has a completely different scene composition** — it emphasizes Landscape (43%) and Rocks (18%) while omitting 3 classes entirely.

---

## 7. Per-Image Analysis

### 7.1 Labels per Image (Train)

| # Classes per Image | Count | % of Train |
|---|---|---|
| 1 | 2 | 0.07% |
| 2 | 28 | 0.98% |
| 3 | 1 | 0.03% |
| 4 | 3 | 0.10% |
| 5 | 7 | 0.24% |
| 6 | 14 | 0.49% |
| 7 | 774 | 27.1% |
| 8 | 1,254 | **43.9%** |
| 9 | 774 | 27.1% |

- Most images have **7–9 classes** (98.1%). This is a dense, multi-class scene.
- 30 images have ≤2 classes — these may be close-up shots or special scenes.
- No images have all 10 classes (max is 9).

### 7.2 Label Coverage

- **100% of pixels are labeled** in every image across all splits.
- No ignore regions, no unlabeled areas, no void class.
- 0 empty masks, 0 sparse masks.

### 7.3 Images with Highest Ground Clutter Coverage (Train)

| Image | GC % | Logs % | # Classes |
|---|---|---|---|
| ww10000290.png | 36.9% | 0.0% | 2 |
| ww10000295.png | 23.0% | 0.0% | 2 |
| ww10000296.png | 22.8% | 0.0% | 2 |
| ww10000291.png | 22.8% | 0.0% | 2 |
| ww10000297.png | 21.0% | 0.0% | 2 |
| ww10000298.png | 20.8% | 0.0% | 2 |

> [!NOTE]
> The `ww1000028x–ww1000030x` images appear to be from a scene looking at ground-level terrain with heavy Ground Clutter coverage (up to 37%). These images have only 2 classes — indicating a close-up or extreme downward angle. They are valuable oversampling candidates.

### 7.4 Single-Class Images (Train)

2 images are labeled with a single class. These should be manually inspected for correctness.

---

## 8. Visual Inspection Findings

Based on the dataset structure and numerical analysis:

### 8.1 Scene Composition

The dataset consists of **offroad driving scenes** captured from a vehicle-mounted camera (likely a robot or autonomous vehicle). Scenes include:
- Forest paths with trees, bushes, and scattered ground debris
- Open terrain with dry grass and rocks
- Variable lighting conditions (mean intensity std = 29.1 across images)

### 8.2 Prefix-Based Scene Groups

| Prefix | Likely Source | Characteristics |
|---|---|---|
| `cc` | Camera Config C | Mixed vegetation scenes |
| `mt` | Mountain terrain | Higher elevation, more rocks |
| `w` | Wide-angle | Broader field of view |
| `ww` | Wide-wide | Very wide perspective, more ground-level |

### 8.3 Challenging Samples to Inspect

- **High GC images**: `ww10000286.png` through `ww10000306.png` (GC-heavy, 2-class only)
- **Rare Logs images**: `mt10000542-549.png` (Logs + GC, 8 classes — most realistic scenes)
- **2-class images**: 28 images with only 2 labels — check for annotation completeness

---

## 9. Annotation Quality Findings

### 9.1 Quantitative Indicators

| Check | Finding |
|---|---|
| Completely unlabeled pixels | None (0%) |
| Images with unexpected pixel values | None |
| Mask mode consistency | 100% I;16 |
| Mask-image size match | 100% |

### 9.2 Inferred Quality Issues

Based on the IoU confusion analysis from training logs (Strategy C diagnostics in the training script):

> **Ground Clutter is most confused with**: Dry Grass, Landscape, and Rocks

This suggests:
1. **Boundary ambiguity**: GC ↔ Dry Grass and GC ↔ Landscape transitions are inherently fuzzy in the real world
2. **Annotation inconsistency**: What counts as "ground clutter" vs "dry grass" may vary between annotators
3. **Small GC regions**: GC often appears as scattered patches mixed with other ground classes

### 9.3 Suspected Annotation Issues

- The 28 two-class images (e.g., `ww10000290.png` with 37% GC + ~63% other) may represent **incomplete annotations** — in a real offroad scene, it's unusual to truly have only 2 visible classes across the entire 960×540 frame
- The `ww` prefix images seem to come from a different annotation batch, with much simpler label sets

> [!WARNING]
> The 2-class `ww` images may be from an annotation session where only foreground/background was labeled. If so, they could inject noise into training by teaching the model that scenes only contain 2 classes.

---

## 10. Split and Leakage Analysis

### 10.1 Split Structure

| Property | Train | Val | Test |
|---|---|---|---|
| Size | 2,857 | 317 | 1,002 |
| Proportion | 68.4% | 7.6% | 24.0% |
| Prefixes | cc, mt, w, ww | cc, mt, w, ww | (none) |
| Filename overlap with other splits | 0 | 0 | 0 |
| Near-duplicate images (hash-based) | 0 across splits | 0 | 0 |

### 10.2 Sequential Frame Leakage

> [!IMPORTANT]
> Train and Val share the **same scene prefixes** (cc, mt, w, ww). Filenames are sequential within each prefix, suggesting **consecutive video frames**. If train contains frame `cc0000100` and val contains `cc0000101`, the model effectively sees the same scene in both splits, leading to **inflated validation IoU**.

**Mitigation**: Split by scene (prefix + number range), not by random frame sampling.

### 10.3 Test Set Isolation

The test set uses **entirely different filenames** (bare numbers 60–1061) and does **not** share prefix patterns with train/val. However, it is missing 3 classes, making it a fundamentally different evaluation domain.

### 10.4 Recommendation

> **Re-split the dataset** using scene-aware splitting:
> 1. Group contiguous sequences within each prefix
> 2. Assign entire sequences to train OR val, never splitting a sequence
> 3. Ensure each split has proportional representation of GC and Logs images
> 4. Move some `ww` GC-heavy sequences to val to create a GC-rich validation set

---

## 11. Image Statistics for Preprocessing

### 11.1 RGB Channel Statistics

| Stat | R | G | B |
|---|---|---|---|
| **Mean** | 120.07 | 116.43 | 111.94 |
| **Std** | 69.11 | 69.27 | 76.02 |

**ImageNet comparison**: ImageNet mean = [123.68, 116.78, 103.94], std = [58.39, 57.12, 57.38]

| Channel | Dataset Mean | ImageNet Mean | Δ |
|---|---|---|---|
| R | 120.07 | 123.68 | -3.61 |
| G | 116.43 | 116.78 | -0.35 |
| B | 111.94 | 103.94 | +8.00 |

> [!NOTE]
> Means are somewhat close to ImageNet, but the **standard deviations are significantly larger** (69–76 vs 57–58), indicating wider intensity variation. This is typical for outdoor scenes with bright sky + dark shadows.

**Recommendation**: Use **dataset-specific normalization** for best results. If using ImageNet-pretrained backbones, either:
- Fine-tune with dataset stats: mean=[120.1, 116.4, 111.9]/255, std=[69.1, 69.3, 76.0]/255
- Or use ImageNet stats initially and let batch-norm layers adapt

### 11.2 Intensity Distribution

| Metric | Value |
|---|---|
| Global min | 0 |
| Global max | 255 |
| Mean of per-image means | 116.1 |
| Std of per-image means | **29.1** |

- The high std-of-means (29.1) indicates **substantial exposure variation** between images.
- Some images are quite dark (outdoor shade/forest canopy), others are bright (open sky).
- This variation argues for **color jitter augmentation** and possibly histogram equalization.

### 11.3 Test Set RGB Stats

| Stat | R | G | B |
|---|---|---|---|
| Mean | 113.77 | 115.12 | 95.18 |
| Std | 53.90 | 57.66 | 64.48 |

The test set has a notably **bluer deficit** (B=95 vs train B=112) and lower standard deviation. This suggests the test scenes are from a different environment/season/lighting condition.

---

## 12. Spatial Bias Analysis

Class positions in a 4×4 grid (percentage of each class's pixels per cell):

### 12.1 Strong Vertical Bias

```
SKY (strongly top-biased):
   [13.1, 13.7, 13.9, 13.5]    ← Top row: 54% of all sky pixels
   [10.1, 10.5, 10.0,  9.2]
   [ 1.6,  1.3,  1.3,  1.6]
   [ 0.1,  0.1,  0.0,  0.1]    ← Bottom row: <1%

LANDSCAPE (strongly bottom-biased):
   [ 2.5,  2.6,  2.6,  2.3]    ← Top row: 10%
   [ 3.2,  3.6,  3.8,  3.5]
   [ 5.6,  6.9,  7.1,  6.8]
   [11.8, 12.9, 12.9, 11.9]    ← Bottom row: 50%

GROUND CLUTTER (strongly bottom-biased):
   [ 1.4,  1.4,  1.0,  1.0]    ← Top row: 5%
   [ 2.0,  1.9,  2.2,  2.4]
   [ 8.6, 10.0,  9.7,  8.9]
   [12.8, 12.1, 12.5, 12.0]    ← Bottom row: 49%

DRY GRASS (strongly bottom-biased):
   [ 1.1,  1.0,  0.7,  0.6]
   [ 2.4,  2.3,  2.6,  2.9]
   [11.9, 11.6, 11.1, 10.6]
   [10.9,  9.9,  9.7, 10.6]    ← Bottom 2 rows: 87%
```

### 12.2 Logs — Left-Center Bias

```
LOGS:
   [ 2.5,  1.7,  0.8,  0.9]
   [ 5.6,  7.1,  3.1,  2.2]    ← Center-left concentration
   [15.7, 16.5, 10.3, 12.4]    ← Row 3 dominates (55%)
   [ 4.7,  1.8,  7.5,  7.3]
```

Logs appear predominantly in the **left-center to center** of Row 3 (60-75% vertical position). This makes sense for objects on the road ahead of the vehicle.

### 12.3 Implications

1. **Random crops must be carefully positioned**: If you crop the top-center, you'll mostly get Sky. Crops from bottom-center give Landscape/Dry Grass but miss Trees.
2. **Vertical flips are DANGEROUS**: Flipping would put Sky at the bottom, which never occurs naturally. **Do not use vertical flip augmentation.**
3. **Horizontal flips are SAFE**: The left-right distribution is roughly symmetric for all classes.
4. **Consider class-aware cropping**: When oversampling rare classes (Logs, GC), crop regions where those classes are spatially concentrated.

---

## 13. Augmentation Recommendations

### 13.1 Safe Augmentations ✅

| Augmentation | Rationale | Parameters |
|---|---|---|
| **Horizontal flip** | Symmetric scenes; no directional bias | p=0.5 |
| **Random brightness** | Exposure variation is high (std=29) | factor=[0.6, 1.4] |
| **Random contrast** | Natural lighting variation | factor=[0.7, 1.4] |
| **Random saturation** | Helps with vegetation color variation | factor=[0.6, 1.2] |
| **Gaussian blur** | Simulates slight defocus | kernel=3, p=0.3 |
| **RandomResizedCrop** | Already used; critical for multi-scale learning | scale=[0.5, 1.0], ratio=[0.75, 1.33] |
| **Color jitter** | Accounts for different recording conditions | Combined B/C/S/H |

### 13.2 Recommended New Augmentations 🆕

| Augmentation | Rationale | Parameters |
|---|---|---|
| **Copy-Paste for Logs** | Logs are 483× underrepresented; paste Log regions from donor images | p=0.3, paste only Log-class segments |
| **Class-aware random crop** | Ensure crops contain rare classes when available | Sample crop center near GC/Log pixels with p=0.4 |
| **Mosaic augmentation** | Combines 4 images; increases rare class exposure | p=0.2: use images from different scenes |
| **Grid distortion / elastic transform** | Helps with boundary delineation | alpha=300, sigma=10, p=0.2 |

### 13.3 Risky Augmentations ⚠️

| Augmentation | Risk | Recommendation |
|---|---|---|
| **Vertical flip** | Sky at bottom is unnatural; breaks spatial priors | ❌ **Never use** |
| **Large rotation (>20°)** | Horizon tilt rarely exceeds ±15° naturally | Use ±15° max, as current code does |
| **Heavy color distortion** | Can make vegetation look like rocks/sky | Keep conservative |
| **CutOut / Random Erasing** | May erase the already-tiny Log regions | ❌ Avoid unless class-aware |
| **MixUp** | Blends labels, damaging boundary precision | ❌ Avoid for segmentation |

### 13.4 Augmentation for Problem Classes

**For Ground Clutter (IoU 0.38)**:
1. No shortage of images (99.9% presence) but confusion with Dry Grass and Landscape
2. Focus on **boundary-sharpening augmentations**: elastic transforms, slight affine
3. Use **harder random crops** centered on GC boundaries
4. Consider **Online Hard Example Mining (OHEM)** at the pixel level

**For Logs (IoU 0.62)**:
1. Extreme pixel scarcity (0.078%) — **Copy-Paste augmentation is essential**
2. Use oversampling weights (already implemented): increase from 5.0 to **15.0** for Logs
3. Logs are small objects → use **larger crop sizes** (512×512 or even 640×640) to ensure Log pixels survive cropping
4. Consider **class-mixing data augmentation**: paste Log segments from random donor images

---

## 14. Modeling Recommendations

### 14.1 Architecture Recommendations

| Category | Model | Rationale |
|---|---|---|
| **Current** | Mask2Former (Swin-T) | Good instance segmentation; struggles with tiny classes |
| **Baseline** | DeepLabV3+ (ResNet-101) | Proven for semantic seg; atrous convolution captures multi-scale features |
| **Stronger** | SegFormer-B5 or Mask2Former (Swin-L) | Better feature hierarchy; worth the compute cost |
| **Lightweight** | PIDNet-S or DDRNet-23-slim | Real-time capable; good for deployment |
| **Best for tiny objects** | **HRNet-W48 + OCR** | Maintains high-resolution features throughout; best for small objects like Logs |

> [!TIP]
> **For Logs and Ground Clutter specifically, switch from Mask2Former to HRNet-W48 with OCR (Object Contextual Representations)**. Mask2Former's query-based approach struggles with very small objects because they may not be selected as queries. HRNet maintains full-resolution feature maps throughout the network, which is critical for detecting thin structures like logs.

### 14.2 Loss Function

| Loss | Use For | Notes |
|---|---|---|
| **CrossEntropy + class weights** | Baseline | Weights from inverse frequency; cap extreme weights |
| **Dice Loss + CE** | GC improvement | Dice directly optimizes IoU; helps with imbalanced classes |
| **Focal Loss** | Log improvement | Down-weights easy pixels (sky, landscape); focuses on hard boundaries |
| **OHEM** | GC boundaries | Mines hardest pixels per batch; forces boundary learning |
| **Lovász-Softmax** | IoU optimization | Directly optimizes IoU; non-surrogate loss |

**Recommended loss**: `0.5 × CE(class-weighted) + 0.5 × Dice + OHEM`

### 14.3 Suggested Class Weights

| Class | Inverse-Freq Weight | Recommended Weight (capped) |
|---|---|---|
| Trees | 10.65 | 5.0 |
| Lush Bushes | 6.34 | 3.0 |
| Dry Grass | 1.99 | 1.0 |
| Dry Bushes | 34.27 | 10.0 |
| Ground Clutter | 8.57 | 8.0 |
| Flowers | 13.41 | 5.0 |
| **Logs** | **483.17** | **20.0** (capped) |
| Rocks | 31.42 | 10.0 |
| Landscape | 1.54 | 1.0 |
| Sky | 1.00 | **0.1** (suppress) |

> [!NOTE]
> Raw inverse-frequency gives Logs a 483× weight, which would destabilize training. Cap at ~20× and supplement with oversampling + Copy-Paste augmentation instead.

### 14.4 Training Configuration

| Parameter | Recommended Value | Rationale |
|---|---|---|
| Crop size | 512×512 or 640×640 | Larger crops preserve more Log pixels |
| Batch size | 4–8 | Larger batches improve BN stability |
| Learning rate | 6e-5 (lr_backbone: 1e-5) | Differential LR for pretrained backbone |
| Optimizer | AdamW | Standard for transformers; already in use |
| Scheduler | Poly or CosineAnnealing | Smooth decay; current cosine is fine |
| Epochs | 50–100 | Current 2–5 is far too few for convergence |
| Evaluation metric | Per-class IoU (not mean only) | Must track GC and Logs IoU separately |

---

## 15. Training Risks — Ranked by Severity

| Rank | Risk | Severity | Impact | Mitigation |
|---|---|---|---|---|
| **1** | **Logs extreme imbalance (0.078%)** | 🔴 Critical | Model ignores Logs entirely | Copy-Paste aug + weight 20× + oversampling 15× |
| **2** | **Test set missing 3 classes** | 🔴 Critical | Cannot evaluate GC/Logs on test | Use val for evaluation; create custom test split |
| **3** | **Sequential frame leakage** | 🟠 High | Val IoU inflated by ~5-10% | Re-split by scene sequences |
| **4** | **GC boundary confusion** | 🟠 High | GC ↔ DryGrass ↔ Landscape confusion | OHEM + Dice loss + boundary augmentation |
| **5** | **Too few training epochs (2–5)** | 🟠 High | Model not converged | Train for 50–100 epochs minimum |
| **6** | **Sky/Landscape dominance (62%)** | 🟡 Medium | Easy pixels waste gradient budget | Focal loss + suppress Sky weight to 0.1 |
| **7** | **Strong spatial bias** | 🟡 Medium | Model learns position shortcuts | Multi-scale cropping + varied crop positions |
| **8** | **Exposure variation (std=29)** | 🟡 Medium | Dark images harder to classify | Color jitter augmentation (already done) |
| **9** | **2-class ww images (28 images)** | 🟡 Medium | Possible annotation noise | Manually review; exclude if incomplete |
| **10** | **Dataset-specific vs ImageNet norm** | 🟢 Low | Minor performance impact | Use dataset-specific normalization |

---

## 16. Prioritized Next Steps

### Immediate Actions (Before Next Training Run)

1. **🔴 Re-evaluate on Val, not Test**: The test set has no GC/Logs/Flowers. All IoU evaluation for these classes must use the val set. Consider holding out additional GC/Logs-rich images from train into a custom val-rare split.

2. **🔴 Increase training epochs to 50+**: Currently running only 2 epochs. The model is severely under-trained.

3. **🔴 Implement Copy-Paste augmentation for Logs**: Extract Log cutouts from images where they appear and paste them into other images. This is the single most impactful change for Logs IoU.

4. **🟠 Add Dice Loss component**: Change from pure CE to `0.5×CE + 0.5×Dice`. This directly optimizes IoU and helps underrepresented classes.

5. **🟠 Increase Log oversampling weight**: Change from current 5.0 to 15.0 in `rare_multipliers`.

### Short-Term Actions (Before Model Selection)

6. **🟠 Re-split train/val by scene sequence**: Group consecutive frames and assign whole sequences to one split. Ensure GC/Logs representation in val.

7. **🟠 Review 28 two-class images**: Manually check `ww10000286–ww10000306` and other 2-class images for annotation completeness.

8. **🟡 Switch to HRNet-W48+OCR**: For best small-object performance (Logs), HRNet's sustained high-resolution features outperform downsampling-heavy architectures.

9. **🟡 Add class-aware random crops**: When a training image contains Logs, bias the crop center toward Log pixel locations.

### Recommended Preprocessing Pipeline

```python
# Normalization (dataset-specific)
mean = [120.07/255, 116.43/255, 111.94/255]  # [0.4709, 0.4566, 0.4390]
std  = [69.11/255, 69.27/255, 76.02/255]     # [0.2710, 0.2716, 0.2981]
```

### Recommended Augmentation Pipeline

```python
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(512, 512, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.6),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=1),
    ], p=0.7),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.ElasticTransform(alpha=120, sigma=12, p=0.15),
    A.Normalize(mean=mean, std=std),
])
```

### Recommended Loss Function

```python
# Combined loss: CE + Dice + Focal
class_weights = torch.tensor([5.0, 3.0, 1.0, 10.0, 8.0, 5.0, 20.0, 10.0, 1.0, 0.1])
ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
dice_loss = DiceLoss(mode='multiclass', classes=10, ignore_index=255)
focal_loss = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=255)

total_loss = 0.4 * ce_loss + 0.3 * dice_loss + 0.3 * focal_loss
```

### Recommended Model Configurations

| Model | Crop | Batch | LR | Epochs | Expected mIoU |
|---|---|---|---|---|---|
| **Baseline**: DeepLabV3+ (ResNet-101) | 512×512 | 8 | 1e-4 | 80 | 0.68–0.72 |
| **Stronger**: HRNet-W48+OCR | 512×512 | 4 | 6e-5 | 100 | 0.72–0.78 |
| **Current improved**: Mask2Former (Swin-T) | 512×512 | 4 | 1e-5 | 50 | 0.70–0.75 |
| **Lightweight**: PIDNet-S | 480×480 | 16 | 5e-4 | 100 | 0.62–0.68 |

---

## Appendix: Data Quality Certificate

| Check | Status |
|---|---|
| All images loadable | ✅ |
| All masks loadable | ✅ |
| All pairs matched 1:1 | ✅ |
| Uniform resolution | ✅ (960×540) |
| Mask values valid | ✅ (no unexpected values) |
| 100% pixel coverage | ✅ |
| No corrupted files | ✅ |
| No size mismatches | ✅ |
| No ignore regions | ✅ (but this means errors have compounding impact) |
| Cross-split hash leakage | ✅ None detected |
| Missing classes in test | ❌ GC, Flowers, Logs absent |
| Sequential frame leakage risk | ⚠️ High (same prefixes in train+val) |
| Class imbalance severity | ⚠️ Extreme for Logs (483:1) |
