import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# ==========================================
# 1. Test Dataset Class
# ==========================================
class OffroadTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

        self.value_map = {
            100: 0,   # Trees
            200: 1,   # Lush Bushes
            300: 2,   # Dry Grass
            500: 3,   # Dry Bushes
            550: 4,   # Ground Clutter
            600: 5,   # Flowers
            700: 6,   # Logs
            800: 7,   # Rocks
            7100: 8,  # Landscape
            10000: 9  # Sky
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(img_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))

        # Convert raw pixel values to 0-9 class IDs (255 is ignore_index)
        continuous_mask = np.full_like(raw_mask, fill_value=255, dtype=np.int64)
        for old_id, new_id in self.value_map.items():
            continuous_mask[raw_mask == old_id] = new_id

        # Processor handles the resizing to 518x518
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        # Unpack
        processed_inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Keep the original mask and image size for accurate IoU calculation
        processed_inputs["original_mask"] = torch.tensor(continuous_mask, dtype=torch.long)
        processed_inputs["target_size"] = torch.tensor([raw_mask.shape[0], raw_mask.shape[1]])
        
        return processed_inputs

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "original_masks": [item["original_mask"] for item in batch],
        "target_sizes": [item["target_size"] for item in batch]
    }

# ==========================================
# 2. Fast Confusion Matrix Calculator
# ==========================================
def fast_hist(a, b, n):
    """
    Creates a 1D histogram and reshapes it to a 2D confusion matrix.
    a: True labels, b: Predicted labels, n: Number of classes
    """
    k = (a >= 0) & (a < n) # Ignore index 255
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

# ==========================================
# 3. Main Evaluation Function
# ==========================================
def evaluate_model(model_dir, test_img_dir, test_mask_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating on: {device.type.upper()}")

    print(f"Loading model from {model_dir}...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(device)
    model.eval()

    test_dataset = OffroadTestDataset(test_img_dir, test_mask_dir, processor)
    # Batch size 1 is safest for exact image-to-image boundary reconstruction
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    class_names = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter", 
                   "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
    num_classes = len(class_names)
    
    # Initialize an empty confusion matrix
    total_hist = np.zeros((num_classes, num_classes))

    print(f"Starting evaluation of {len(test_dataset)} images...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Images"):
            pixel_values = batch["pixel_values"].to(device)
            target_sizes = [tuple(size.tolist()) for size in batch["target_sizes"]]
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            
            # Upscale predictions back to the EXACT original image resolution
            predicted_maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
            
            # Calculate metrics for this batch
            for pred_mask, true_mask in zip(predicted_maps, batch["original_masks"]):
                pred_np = pred_mask.cpu().numpy().flatten()
                true_np = true_mask.numpy().flatten()
                
                # Accumulate the confusion matrix
                total_hist += fast_hist(true_np, pred_np, num_classes)

    # ==========================================
    # Calculate Final Metrics from the Histogram
    # ==========================================
    # Pixel Accuracy: sum of diagonal / sum of all pixels
    pixel_acc = np.diag(total_hist).sum() / total_hist.sum()
    
    # Class-wise Accuracy: diagonal / sum of rows
    class_acc = np.diag(total_hist) / total_hist.sum(axis=1)
    
    # Class-wise IoU: intersection / (true_set + predicted_set - intersection)
    intersection = np.diag(total_hist)
    union = total_hist.sum(axis=1) + total_hist.sum(axis=0) - intersection
    ious = intersection / np.maximum(union, 1) # Avoid division by zero
    
    mean_iou = np.nanmean(ious)

    # ==========================================
    # Print and Save Results
    # ==========================================
    print("\n" + "="*50)
    print("🏆 FINAL TEST RESULTS")
    print("="*50)
    print(f"Overall Pixel Accuracy: {pixel_acc * 100:.2f}%")
    print(f"Mean IoU (mIoU):        {mean_iou * 100:.2f}%")
    print("-" * 50)
    print(f"{'Class Name':<18} | {'IoU (%)':<8} | {'Accuracy (%)':<8}")
    print("-" * 50)
    
    results_list = []
    for i, name in enumerate(class_names):
        print(f"{name:<18} | {ious[i]*100:>6.2f}% | {class_acc[i]*100:>6.2f}%")
        results_list.append({"Class": name, "IoU": ious[i], "Accuracy": class_acc[i]})

    # Save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv("test_results.csv", index=False)
    print("\n✅ Metrics saved to test_results.csv")

    # ==========================================
    # Plot Confusion Matrix
    # ==========================================
    # Normalize by row (true classes) to get percentages
    hist_percent = total_hist / total_hist.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(hist_percent, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("✅ Confusion matrix heatmap saved to confusion_matrix.png")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    MODEL_DIR = MODEL_DIR = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Scripts\best_offroad_model_refined"
    TEST_IMG_DIR = TEST_IMG_DIR = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Color_Images"
    TEST_MASK_DIR = TEST_MASK_DIR = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Segmentation"
    
    evaluate_model(MODEL_DIR, TEST_IMG_DIR, TEST_MASK_DIR)