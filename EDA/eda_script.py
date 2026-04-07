import os
import sys
import json
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import traceback
import hashlib

# ============================================
# CONFIGURATION
# ============================================
BASE = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT"
SPLITS = {
    "train": {
        "images": os.path.join(BASE, "Offroad_Segmentation_Training_Dataset", "train", "Color_Images"),
        "masks": os.path.join(BASE, "Offroad_Segmentation_Training_Dataset", "train", "Segmentation"),
    },
    "val": {
        "images": os.path.join(BASE, "Offroad_Segmentation_Training_Dataset", "val", "Color_Images"),
        "masks": os.path.join(BASE, "Offroad_Segmentation_Training_Dataset", "val", "Segmentation"),
    },
    "test": {
        "images": os.path.join(BASE, "Offroad_Segmentation_testImages", "Color_Images"),
        "masks": os.path.join(BASE, "Offroad_Segmentation_testImages", "Segmentation"),
    },
}

VALUE_MAP = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    550: 4,    # Ground Clutter
    600: 5,    # Flowers
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9,  # Sky
}
CLASS_NAMES = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
               "Flowers", "Logs", "Rocks", "Landscape", "Sky"]

OUTPUT_FILE = os.path.join(BASE, "eda_results.json")

# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def analyze_split(split_name, img_dir, mask_dir):
    """Full analysis of a single split."""
    print(f"\n{'='*60}")
    print(f"  ANALYZING SPLIT: {split_name.upper()}")
    print(f"{'='*60}")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
    
    img_set = set(img_files)
    mask_set = set(mask_files)
    
    # Pairing analysis
    matched = img_set & mask_set
    imgs_without_mask = img_set - mask_set
    masks_without_img = mask_set - img_set
    
    result = {
        "split": split_name,
        "num_images": len(img_files),
        "num_masks": len(mask_files),
        "num_matched_pairs": len(matched),
        "images_without_mask": sorted(list(imgs_without_mask))[:20],
        "masks_without_img": sorted(list(masks_without_img))[:20],
        "images_without_mask_count": len(imgs_without_mask),
        "masks_without_img_count": len(masks_without_img),
    }
    
    print(f"  Images: {len(img_files)}, Masks: {len(mask_files)}, Matched pairs: {len(matched)}")
    if imgs_without_mask:
        print(f"  ⚠ Images without mask: {len(imgs_without_mask)}")
    if masks_without_img:
        print(f"  ⚠ Masks without image: {len(masks_without_img)}")
    
    # Filename numbering analysis
    img_numbers = []
    for f in img_files:
        try:
            num = int(os.path.splitext(f)[0])
            img_numbers.append(num)
        except:
            pass
    if img_numbers:
        result["filename_range"] = {"min": min(img_numbers), "max": max(img_numbers)}
        result["filename_gaps"] = []
        expected = set(range(min(img_numbers), max(img_numbers)+1))
        actual = set(img_numbers)
        gaps = sorted(expected - actual)
        if gaps:
            # Summarize gap ranges
            gap_ranges = []
            start = gaps[0]
            end = gaps[0]
            for g in gaps[1:]:
                if g == end + 1:
                    end = g
                else:
                    gap_ranges.append(f"{start}-{end}" if start != end else str(start))
                    start = g
                    end = g
            gap_ranges.append(f"{start}-{end}" if start != end else str(start))
            result["filename_gaps"] = gap_ranges
            print(f"  Filename range: {min(img_numbers)}-{max(img_numbers)}")
            print(f"  Gaps in numbering: {gap_ranges[:10]}{'...' if len(gap_ranges) > 10 else ''}")
    
    # File integrity, modes, sizes, resolutions
    corrupted_images = []
    corrupted_masks = []
    img_modes = Counter()
    mask_modes = Counter()
    img_sizes_bytes = []
    mask_sizes_bytes = []
    img_resolutions = []
    mask_resolutions = []
    size_mismatches = []
    
    # Class distribution accumulators
    total_pixels_per_class = np.zeros(10, dtype=np.int64)
    class_presence_per_image = np.zeros(10, dtype=np.int64)  # how many images contain each class
    all_unique_mask_values = set()
    unexpected_values = set()
    
    # Per-image stats
    per_image_class_coverage = []  # list of dicts
    empty_mask_images = []
    single_class_images = []
    sparse_label_images = []  # less than 5% labeled
    
    # RGB stats for preprocessing
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixel_count = 0
    
    # Spatial bias (divide image into 4x4 grid)
    GRID = 4
    spatial_class_count = np.zeros((10, GRID, GRID), dtype=np.int64)
    
    # Intensity stats
    min_intensities = []
    max_intensities = []
    mean_intensities = []
    
    # Hash for near-duplicate detection
    image_hashes = {}
    
    paired_files = sorted(list(matched))
    total = len(paired_files)
    
    for idx, fname in enumerate(paired_files):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{total}...", flush=True)
        
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        
        # Image integrity
        try:
            img = Image.open(img_path)
            img.load()
            img_modes[img.mode] += 1
            img_sizes_bytes.append(os.path.getsize(img_path))
            w, h = img.size
            img_resolutions.append((w, h))
            
            # RGB stats
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb, dtype=np.float64)
            pixel_sum += img_np.sum(axis=(0, 1))
            pixel_sq_sum += (img_np ** 2).sum(axis=(0, 1))
            total_pixel_count += w * h
            
            min_intensities.append(img_np.min())
            max_intensities.append(img_np.max())
            mean_intensities.append(img_np.mean())
            
            # Image hash (downscale to 16x16 grayscale for fast comparison)
            thumb = img_rgb.resize((16, 16)).convert("L")
            thumb_hash = hashlib.md5(np.array(thumb).tobytes()).hexdigest()
            if thumb_hash not in image_hashes:
                image_hashes[thumb_hash] = []
            image_hashes[thumb_hash].append((split_name, fname))
            
        except Exception as e:
            corrupted_images.append({"file": fname, "error": str(e)})
            continue
        
        # Mask integrity
        try:
            mask = Image.open(mask_path)
            mask.load()
            mask_modes[mask.mode] += 1
            mask_sizes_bytes.append(os.path.getsize(mask_path))
            mw, mh = mask.size
            mask_resolutions.append((mw, mh))
            
            if (w, h) != (mw, mh):
                size_mismatches.append({"file": fname, "img_size": (w, h), "mask_size": (mw, mh)})
            
        except Exception as e:
            corrupted_masks.append({"file": fname, "error": str(e)})
            continue
        
        # Mask analysis
        mask_np = np.array(mask)
        unique_vals = np.unique(mask_np)
        all_unique_mask_values.update(unique_vals.tolist())
        
        # Check for unexpected values
        for v in unique_vals:
            if v not in VALUE_MAP:
                unexpected_values.add(int(v))
        
        # Map to contiguous labels
        continuous_mask = np.full_like(mask_np, fill_value=255, dtype=np.int64)
        for old_id, new_id in VALUE_MAP.items():
            continuous_mask[mask_np == old_id] = new_id
        
        total_mask_pixels = continuous_mask.size
        labeled_pixels = np.sum(continuous_mask != 255)
        
        # Per-class pixel counts
        class_coverage = {}
        classes_present = []
        for c in range(10):
            count = np.sum(continuous_mask == c)
            total_pixels_per_class[c] += count
            if count > 0:
                class_presence_per_image[c] += 1
                classes_present.append(c)
            class_coverage[CLASS_NAMES[c]] = float(count) / total_mask_pixels * 100
        
        per_image_class_coverage.append({
            "file": fname,
            "coverage": class_coverage,
            "num_classes": len(classes_present),
            "labeled_pct": float(labeled_pixels) / total_mask_pixels * 100,
            "classes_present": classes_present,
        })
        
        if labeled_pixels == 0:
            empty_mask_images.append(fname)
        elif len(classes_present) == 1:
            single_class_images.append(fname)
        if labeled_pixels / total_mask_pixels < 0.05:
            sparse_label_images.append(fname)
        
        # Spatial bias (4x4 grid)
        h_mask, w_mask = continuous_mask.shape
        cell_h = h_mask // GRID
        cell_w = w_mask // GRID
        for gi in range(GRID):
            for gj in range(GRID):
                cell = continuous_mask[gi*cell_h:(gi+1)*cell_h, gj*cell_w:(gj+1)*cell_w]
                for c in range(10):
                    spatial_class_count[c, gi, gj] += np.sum(cell == c)
    
    # Compute derived stats
    total_all_class_pixels = total_pixels_per_class.sum()
    class_pct = (total_pixels_per_class / max(total_all_class_pixels, 1) * 100).tolist()
    
    # RGB mean/std
    rgb_mean = (pixel_sum / max(total_pixel_count, 1)).tolist()
    rgb_std = (np.sqrt(pixel_sq_sum / max(total_pixel_count, 1) - (pixel_sum / max(total_pixel_count, 1))**2)).tolist()
    
    # Resolution stats
    widths = [r[0] for r in img_resolutions]
    heights = [r[1] for r in img_resolutions]
    aspect_ratios = [w/h for w, h in img_resolutions]
    
    # Spatial bias normalization (per-class, normalized to percentage in each cell)
    spatial_bias = {}
    for c in range(10):
        total_c = spatial_class_count[c].sum()
        if total_c > 0:
            normalized = (spatial_class_count[c] / total_c * 100).tolist()
        else:
            normalized = np.zeros((GRID, GRID)).tolist()
        spatial_bias[CLASS_NAMES[c]] = normalized
    
    # Near duplicates
    near_duplicates = {h: files for h, files in image_hashes.items() if len(files) > 1}
    
    # Assemble result
    result.update({
        "corrupted_images": corrupted_images,
        "corrupted_masks": corrupted_masks,
        "image_modes": dict(img_modes),
        "mask_modes": dict(mask_modes),
        "image_file_size_stats": {
            "min": min(img_sizes_bytes) if img_sizes_bytes else 0,
            "max": max(img_sizes_bytes) if img_sizes_bytes else 0,
            "mean": float(np.mean(img_sizes_bytes)) if img_sizes_bytes else 0,
            "median": float(np.median(img_sizes_bytes)) if img_sizes_bytes else 0,
        },
        "mask_file_size_stats": {
            "min": min(mask_sizes_bytes) if mask_sizes_bytes else 0,
            "max": max(mask_sizes_bytes) if mask_sizes_bytes else 0,
            "mean": float(np.mean(mask_sizes_bytes)) if mask_sizes_bytes else 0,
            "median": float(np.median(mask_sizes_bytes)) if mask_sizes_bytes else 0,
        },
        "resolution": {
            "width": {"min": min(widths), "max": max(widths), "mean": float(np.mean(widths)), "std": float(np.std(widths)), "unique_count": len(set(widths))},
            "height": {"min": min(heights), "max": max(heights), "mean": float(np.mean(heights)), "std": float(np.std(heights)), "unique_count": len(set(heights))},
            "aspect_ratio": {"min": min(aspect_ratios), "max": max(aspect_ratios), "mean": float(np.mean(aspect_ratios)), "std": float(np.std(aspect_ratios))},
            "unique_resolutions": len(set(img_resolutions)),
            "most_common_resolution": Counter(img_resolutions).most_common(3),
        },
        "size_mismatches": size_mismatches[:20],
        "size_mismatch_count": len(size_mismatches),
        "all_unique_mask_values": sorted([int(v) for v in all_unique_mask_values]),
        "unexpected_mask_values": sorted([int(v) for v in unexpected_values]),
        "class_pixel_counts": {CLASS_NAMES[i]: int(total_pixels_per_class[i]) for i in range(10)},
        "class_pixel_pct": {CLASS_NAMES[i]: round(class_pct[i], 4) for i in range(10)},
        "class_presence_frequency": {CLASS_NAMES[i]: int(class_presence_per_image[i]) for i in range(10)},
        "class_presence_pct": {CLASS_NAMES[i]: round(int(class_presence_per_image[i]) / max(total, 1) * 100, 2) for i in range(10)},
        "empty_mask_images": empty_mask_images[:20],
        "empty_mask_count": len(empty_mask_images),
        "single_class_images": single_class_images[:20],
        "single_class_count": len(single_class_images),
        "sparse_label_images": sparse_label_images[:20],
        "sparse_label_count": len(sparse_label_images),
        "rgb_stats": {
            "mean": [round(v, 4) for v in rgb_mean],
            "std": [round(v, 4) for v in rgb_std],
        },
        "intensity_stats": {
            "global_min": float(min(min_intensities)) if min_intensities else 0,
            "global_max": float(max(max_intensities)) if max_intensities else 0,
            "mean_of_means": float(np.mean(mean_intensities)) if mean_intensities else 0,
            "std_of_means": float(np.std(mean_intensities)) if mean_intensities else 0,
        },
        "spatial_bias_4x4": spatial_bias,
        "near_duplicate_groups": len(near_duplicates),
        "near_duplicate_examples": {h: files[:5] for h, files in list(near_duplicates.items())[:10]},
    })
    
    # Per-image stats summary: find hardest images (most classes, rarest class coverage)
    # Sort by coverage of ground_clutter + logs
    rare_class_images = []
    for pimg in per_image_class_coverage:
        gc_cov = pimg["coverage"].get("Ground Clutter", 0)
        log_cov = pimg["coverage"].get("Logs", 0)
        if gc_cov > 0 or log_cov > 0:
            rare_class_images.append({
                "file": pimg["file"],
                "ground_clutter_pct": round(gc_cov, 4),
                "logs_pct": round(log_cov, 4),
                "num_classes": pimg["num_classes"],
            })
    rare_class_images.sort(key=lambda x: x["ground_clutter_pct"] + x["logs_pct"], reverse=True)
    result["top_rare_class_images"] = rare_class_images[:30]
    result["images_with_ground_clutter"] = sum(1 for r in rare_class_images if r["ground_clutter_pct"] > 0)
    result["images_with_logs"] = sum(1 for r in rare_class_images if r["logs_pct"] > 0)
    
    # Distribution of num_classes per image
    num_classes_dist = Counter(pimg["num_classes"] for pimg in per_image_class_coverage)
    result["num_classes_per_image_distribution"] = dict(sorted(num_classes_dist.items()))
    
    # Labeled percentage stats
    labeled_pcts = [pimg["labeled_pct"] for pimg in per_image_class_coverage]
    result["labeled_pct_stats"] = {
        "min": round(min(labeled_pcts), 4) if labeled_pcts else 0,
        "max": round(max(labeled_pcts), 4) if labeled_pcts else 0,
        "mean": round(float(np.mean(labeled_pcts)), 4) if labeled_pcts else 0,
        "images_below_90pct": sum(1 for p in labeled_pcts if p < 90),
        "images_below_50pct": sum(1 for p in labeled_pcts if p < 50),
    }
    
    print(f"  ✅ Split analysis complete.")
    return result, image_hashes


def check_cross_split_leakage(all_hashes):
    """Check for near-duplicate images across splits."""
    print(f"\n{'='*60}")
    print(f"  CROSS-SPLIT LEAKAGE ANALYSIS")
    print(f"{'='*60}")
    
    cross_split = []
    for h, files in all_hashes.items():
        splits_in_group = set(f[0] for f in files)
        if len(splits_in_group) > 1:
            cross_split.append({
                "hash": h,
                "files": files[:5],
                "splits": sorted(list(splits_in_group)),
            })
    
    print(f"  Cross-split near-duplicate groups: {len(cross_split)}")
    return cross_split[:50]


def check_sequential_leakage(splits_data):
    """Check if consecutive frames are split across train and val."""
    print(f"\n  Checking sequential frame leakage...")
    
    train_nums = set()
    val_nums = set()
    test_nums = set()
    
    for split_name, data in splits_data.items():
        frange = data.get("filename_range", {})
        if not frange:
            continue
        nums = set()
        # Re-derive from matched pairs count and range
        # Actually, let me just use the file list approach
    
    # Get file numbers per split
    for split_name, paths in SPLITS.items():
        files = sorted(os.listdir(paths["images"]))
        nums = []
        for f in files:
            try:
                nums.append(int(os.path.splitext(f)[0]))
            except:
                pass
        if split_name == "train":
            train_nums = set(nums)
        elif split_name == "val":
            val_nums = set(nums)
        elif split_name == "test":
            test_nums = set(nums)
    
    # Check consecutive frame overlap
    leakage_pairs = []
    for tn in train_nums:
        for offset in [-1, 1, -2, 2]:
            neighbor = tn + offset
            if neighbor in val_nums:
                leakage_pairs.append({"train": tn, "val": neighbor, "distance": abs(offset)})
            if neighbor in test_nums:
                leakage_pairs.append({"train": tn, "test": neighbor, "distance": abs(offset)})
    
    for vn in val_nums:
        for offset in [-1, 1, -2, 2]:
            neighbor = vn + offset
            if neighbor in test_nums:
                leakage_pairs.append({"val": vn, "test": neighbor, "distance": abs(offset)})
    
    print(f"  Sequential leakage pairs (±2 frames): {len(leakage_pairs)}")
    
    # Also check overlap between sets
    train_val_overlap = train_nums & val_nums
    train_test_overlap = train_nums & test_nums
    val_test_overlap = val_nums & test_nums
    
    return {
        "sequential_leakage_pairs": leakage_pairs[:50],
        "total_sequential_leakage": len(leakage_pairs),
        "train_val_id_overlap": sorted(list(train_val_overlap))[:20],
        "train_test_id_overlap": sorted(list(train_test_overlap))[:20],
        "val_test_id_overlap": sorted(list(val_test_overlap))[:20],
        "train_range": {"min": min(train_nums) if train_nums else None, "max": max(train_nums) if train_nums else None},
        "val_range": {"min": min(val_nums) if val_nums else None, "max": max(val_nums) if val_nums else None},
        "test_range": {"min": min(test_nums) if test_nums else None, "max": max(test_nums) if test_nums else None},
    }


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  OFFROAD SEGMENTATION DATASET - FULL EDA")
    print("=" * 60)
    
    all_results = {}
    all_hashes = {}  # Combined hash -> [(split, file), ...]
    
    for split_name, paths in SPLITS.items():
        try:
            split_result, split_hashes = analyze_split(split_name, paths["images"], paths["masks"])
            all_results[split_name] = split_result
            
            # Merge hashes for cross-split analysis
            for h, files in split_hashes.items():
                if h not in all_hashes:
                    all_hashes[h] = []
                all_hashes[h].extend(files)
        except Exception as e:
            print(f"  ❌ Error analyzing {split_name}: {e}")
            traceback.print_exc()
    
    # Cross-split analysis
    cross_split_leakage = check_cross_split_leakage(all_hashes)
    sequential_leakage = check_sequential_leakage(all_results)
    
    all_results["cross_split_analysis"] = {
        "near_duplicate_leakage": cross_split_leakage,
        "sequential_leakage": sequential_leakage,
    }
    
    # Save results
    print(f"\n📄 Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ EDA COMPLETE. Results saved to {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  QUICK SUMMARY")
    print("=" * 60)
    for split_name in ["train", "val", "test"]:
        if split_name not in all_results:
            continue
        r = all_results[split_name]
        print(f"\n  [{split_name.upper()}]")
        print(f"    Pairs: {r['num_matched_pairs']}")
        print(f"    Corrupted imgs/masks: {len(r['corrupted_images'])}/{len(r['corrupted_masks'])}")
        print(f"    Image modes: {r['image_modes']}")
        print(f"    Mask modes: {r['mask_modes']}")
        print(f"    Resolution unique count: {r['resolution']['unique_resolutions']}")
        print(f"    Size mismatches: {r['size_mismatch_count']}")
        print(f"    Unexpected mask values: {r['unexpected_mask_values']}")
        print(f"    Empty masks: {r['empty_mask_count']}")
        print(f"    RGB mean: {r['rgb_stats']['mean']}")
        print(f"    RGB std: {r['rgb_stats']['std']}")
        print(f"    Class distribution (%):")
        for cname in CLASS_NAMES:
            pct = r['class_pixel_pct'].get(cname, 0)
            presence = r['class_presence_pct'].get(cname, 0)
            print(f"      {cname:20s}: {pct:7.3f}% of pixels, present in {presence:5.1f}% of images")
