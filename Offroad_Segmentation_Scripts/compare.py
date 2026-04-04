import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ==========================================
# CONFIG
# ==========================================
MASK_DIRS = {
    "Train": r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Segmentation",
    "Val":   r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Segmentation",
    "Test":  r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_testImages\Segmentation"
}

VALUE_MAP = {
    100:   "Tree",
    200:   "Lush Bushes",
    300:   "Dry Grass",
    500:   "Dry Bushes",
    550:   "Ground Clutter",
    600:   "Flowers",
    700:   "Logs",
    800:   "Rocks",
    7100:  "Landscape",
    10000: "Sky"
}
CLASS_NAMES = list(VALUE_MAP.values())
NUM_CLASSES  = len(CLASS_NAMES)

# ==========================================
# SCAN ALL SPLITS
# ==========================================
# Aggregated stats per split
stats = {}   # split -> dict of metrics

for split_name, mask_dir in MASK_DIRS.items():
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff', '.jpg'))
    ])
    N = len(mask_files)
    print(f"\n📂 Scanning [{split_name}] — {N} images from {mask_dir}")

    # Per-class counters
    present_count   = defaultdict(int)   # images containing class
    pixel_count     = defaultdict(int)   # raw pixel count
    total_pixels    = 0                  # total pixels scanned
    unknown_vals    = defaultdict(int)   # unknown pixel value -> count
    classes_per_img = []                 # how many classes each image has

    for fname in tqdm(mask_files, desc=f"  {split_name}"):
        mask_np = np.array(Image.open(os.path.join(mask_dir, fname)))
        unique_vals, val_counts = np.unique(mask_np, return_counts=True)
        val_freq = dict(zip(unique_vals.tolist(), val_counts.tolist()))
        total_pixels += mask_np.size

        img_class_count = 0
        for pixel_val, class_name in VALUE_MAP.items():
            if pixel_val in val_freq:
                present_count[class_name]  += 1
                pixel_count[class_name]    += val_freq[pixel_val]
                img_class_count += 1

        classes_per_img.append(img_class_count)

        for v in unique_vals:
            if v not in VALUE_MAP:
                unknown_vals[int(v)] += val_freq[v]

    classes_per_img = np.array(classes_per_img)

    stats[split_name] = {
        "N":               N,
        "total_pixels":    total_pixels,
        "present_count":   present_count,
        "pixel_count":     pixel_count,
        "classes_per_img": classes_per_img,
        "unknown_vals":    unknown_vals,
    }

# ==========================================
# REPORT
# ==========================================
DIV  = "=" * 80
DIV2 = "-" * 80

def pct(a, b): return (a / b * 100) if b else 0

for split_name, s in stats.items():
    N             = s["N"]
    total_px      = s["total_pixels"]
    cpi           = s["classes_per_img"]
    present_count = s["present_count"]
    pixel_count   = s["pixel_count"]

    print(f"\n{DIV}")
    print(f"  SPLIT: {split_name.upper()}  —  {N} images  |  {total_px:,} total pixels")
    print(DIV)

    # ── 1. Class presence table ──────────────────────────────────────────────
    print(f"\n{'#':<4} {'Class':<18} {'Images w/ class':>16} {'Coverage %':>11} "
          f"{'Total Pixels':>14} {'Pixel Share %':>14} {'Avg px/img':>12} {'Status'}")
    print(DIV2)

    rows = []
    for pixel_val, class_name in VALUE_MAP.items():
        cnt     = present_count[class_name]
        px      = pixel_count[class_name]
        cov_pct = pct(cnt, N)
        px_pct  = pct(px, total_px)
        avg_px  = px / cnt if cnt else 0
        status  = "ALL"    if cnt == N else \
                  "RARE"   if cov_pct < 20 else \
                  "COMMON" if cov_pct >= 60 else "PARTIAL"
        rows.append((pixel_val, class_name, cnt, px, cov_pct, px_pct, avg_px, status))

    # Sort by coverage descending
    rows.sort(key=lambda x: x[4], reverse=True)

    for i, (pv, cn, cnt, px, cov, px_pct, avg_px, status) in enumerate(rows, 1):
        icon = "✅" if status == "ALL" else "🟡" if status == "COMMON" else "🔶" if status == "PARTIAL" else "❌"
        print(f"{i:<4} {cn:<18} {cnt:>7}/{N:<7}  {cov:>9.1f}%  {px:>14,}  {px_pct:>12.2f}%  {avg_px:>10,.0f}   {icon} {status}")

    # ── 2. Dataset-level summary ─────────────────────────────────────────────
    print(f"\n{DIV2}")
    print(" DATASET-LEVEL SUMMARY")
    print(DIV2)

    fully_covered  = [cn for cn in CLASS_NAMES if present_count[cn] == N]
    common         = [cn for cn in CLASS_NAMES if N * 0.6 <= present_count[cn] < N]
    partial        = [cn for cn in CLASS_NAMES if N * 0.2 <= present_count[cn] < N * 0.6]
    rare           = [cn for cn in CLASS_NAMES if 0 < present_count[cn] < N * 0.2]
    absent         = [cn for cn in CLASS_NAMES if present_count[cn] == 0]

    print(f"  ✅ ALL images   ({len(fully_covered):>2}/10): {', '.join(fully_covered) or '—'}")
    print(f"  🟡 COMMON ≥60%  ({len(common):>2}/10): {', '.join(common) or '—'}")
    print(f"  🔶 PARTIAL 20-60% ({len(partial):>2}/10): {', '.join(partial) or '—'}")
    print(f"  ⚠️  RARE   <20%  ({len(rare):>2}/10): {', '.join(rare) or '—'}")
    print(f"  ❌ ABSENT   0%  ({len(absent):>2}/10): {', '.join(absent) or '—'}")

    # ── 3. Classes-per-image distribution ────────────────────────────────────
    print(f"\n{DIV2}")
    print(" CLASSES PER IMAGE DISTRIBUTION")
    print(DIV2)
    print(f"  Min classes in a single image : {cpi.min()}")
    print(f"  Max classes in a single image : {cpi.max()}")
    print(f"  Mean classes per image        : {cpi.mean():.2f}")
    print(f"  Median classes per image      : {np.median(cpi):.1f}")
    print(f"  Std deviation                 : {cpi.std():.2f}")
    print()

    bin_labels = list(range(1, NUM_CLASSES + 2))
    for k in range(1, NUM_CLASSES + 1):
        count_k = int(np.sum(cpi == k))
        if count_k == 0:
            continue
        bar = "█" * int(count_k / N * 40)
        print(f"  {k:>2} classes: {count_k:>4} images  ({pct(count_k,N):>5.1f}%)  {bar}")

    # ── 4. Class co-occurrence matrix (text) ─────────────────────────────────
    print(f"\n{DIV2}")
    print(" CLASS CO-OCCURRENCE  (% of images where both classes appear together)")
    print(DIV2)

    # Rebuild per-image class presence as binary matrix
    mask_files = sorted([
        f for f in os.listdir(MASK_DIRS[split_name])
        if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff', '.jpg'))
    ])
    presence_matrix = np.zeros((N, NUM_CLASSES), dtype=bool)
    for idx, fname in enumerate(mask_files):
        mask_np = np.array(Image.open(os.path.join(MASK_DIRS[split_name], fname)))
        unique_vals = set(np.unique(mask_np).tolist())
        for ci, (pv, cn) in enumerate(VALUE_MAP.items()):
            if pv in unique_vals:
                presence_matrix[idx, ci] = True

    # Print header
    short = [cn[:6] for cn in CLASS_NAMES]
    print("            " + "  ".join(f"{s:>6}" for s in short))
    for i, cn_i in enumerate(CLASS_NAMES):
        row = f"{cn_i[:10]:<10}  "
        for j, cn_j in enumerate(CLASS_NAMES):
            if i == j:
                row += f"{'—':>6}  "
            else:
                both = np.sum(presence_matrix[:, i] & presence_matrix[:, j])
                row += f"{pct(both, N):>5.0f}%  "
        print(row)

    # ── 5. Class imbalance ratio ──────────────────────────────────────────────
    print(f"\n{DIV2}")
    print(" CLASS IMBALANCE (pixel share relative to uniform 10% baseline)")
    print(DIV2)
    baseline = 10.0
    for pv, cn in VALUE_MAP.items():
        px     = pixel_count[cn]
        share  = pct(px, total_px)
        ratio  = share / baseline
        bar    = "█" * int(min(ratio * 10, 50))
        flag   = "⚠️ " if ratio < 0.1 or ratio > 5 else "  "
        print(f"  {flag}{cn:<18}  {share:>6.2f}%  ratio={ratio:>5.2f}x  {bar}")

    # ── 6. Unknown pixel values ───────────────────────────────────────────────
    if s["unknown_vals"]:
        print(f"\n{DIV2}")
        print(" ⚠️  UNKNOWN PIXEL VALUES (not in value_map)")
        print(DIV2)
        for val, cnt in sorted(s["unknown_vals"].items()):
            print(f"  Pixel value {val:>6}  →  {cnt:>10,} pixels total")
    else:
        print(f"\n  ✅ No unknown pixel values found in {split_name}.")

print(f"\n{DIV}")
print("  ANALYSIS COMPLETE")
print(DIV)