import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from torchmetrics.classification import MulticlassJaccardIndex

# ==========================================
# 1. Dataset Class (Updated to return original mask for eval)
# ==========================================
class DualityOffroadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

        self.value_map = {
            100: 0,      # Trees
            200: 1,      # Lush Bushes
            300: 2,      # Dry Grass
            500: 3,      # Dry Bushes
            550: 4,      # Ground Clutter
            600: 5,      # Flowers
            700: 6,      # Logs
            800: 7,      # Rocks
            7100: 8,     # Landscape
            10000: 9     # Sky
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(img_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))

        continuous_mask = np.full_like(raw_mask, fill_value=255, dtype=np.int64)
        for old_id, new_id in self.value_map.items():
            continuous_mask[raw_mask == old_id] = new_id

        inputs = self.processor(
            images=image,
            segmentation_maps=continuous_mask,
            return_tensors="pt"
        )
        
        # Safely unpack the processor outputs (Fixes the AttributeError)
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                processed_inputs[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], torch.Tensor):
                processed_inputs[k] = v[0]
            else:
                processed_inputs[k] = v
        
        # Keep the 2D mask for validation IoU calculations
        # (Great architectural choice for evaluating spatial/obstacle awareness later)
        processed_inputs["original_mask"] = torch.tensor(continuous_mask, dtype=torch.long)
        
        return processed_inputs

# ==========================================
# 2. Custom Collate Function
# ==========================================
def mask2former_collate_fn(batch):
    """
    Mask2Former needs mask_labels to be a list of tensors of varying sizes.
    Standard PyTorch dataloaders try to stack them and crash. This fixes that.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([
            item.get("pixel_mask", torch.ones_like(item["pixel_values"][0])) 
            for item in batch
        ])
    original_masks = torch.stack([item["original_mask"] for item in batch])
    
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "original_mask": original_masks
    }

# ==========================================
# 3. Training and Evaluation Loop
# ==========================================
def train_model(train_dir_img, train_dir_mask, val_dir_img, val_dir_mask, epochs=10, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Initialize Processor and Model
    processor = Mask2FormerImageProcessor(reduce_labels=False, ignore_index=255,size={"height": 512, "width": 512})
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic",
        ignore_mismatched_sizes=True,
        num_labels=10
    ).to(device)

    # Dataloaders
    train_dataset = DualityOffroadDataset(train_dir_img, train_dir_mask, processor)
    val_dataset = DualityOffroadDataset(val_dir_img, val_dir_mask, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mask2former_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mask2former_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # Metrics Trackers
    # ignore_index=0 means we don't care if it perfectly segments the background
    iou_metric_mean = MulticlassJaccardIndex(num_classes=10, ignore_index=255, average="macro").to(device)
    iou_metric_none = MulticlassJaccardIndex(num_classes=10, ignore_index=255, average="none").to(device)

    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
        "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]

    # CSV Setup
    csv_filename = "training_metrics.csv"
    columns = ["Epoch", "Train_Loss", "Val_Loss", "Mean_IoU"] + [f"IoU_{name.replace(' ', '_')}" for name in class_names]
    metrics_df = pd.DataFrame(columns=columns)

    best_iou = 0.0

    print("🔥 Starting Training Loop...")
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                pixel_mask=batch["pixel_mask"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]]
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        iou_metric_mean.reset()
        iou_metric_none.reset()

        print("🔍 Running Validation...")
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    pixel_mask=batch["pixel_mask"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]]
                )
                val_loss += outputs.loss.item()

                # Get final 2D segmentation map predictions
                target_sizes = [mask.shape for mask in batch["original_mask"]]
                predicted_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                
                # Stack maps and calculate IoU
                preds = torch.stack(predicted_maps).to(device)
                targets = batch["original_mask"].to(device)
                
                iou_metric_mean.update(preds, targets)
                iou_metric_none.update(preds, targets)

        avg_val_loss = val_loss / len(val_loader)
        mean_iou = iou_metric_mean.compute().item()
        class_ious = iou_metric_none.compute().cpu().numpy()

        # --- PRINT METRICS ---
        print("-" * 50)
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Mean IoU: {mean_iou:.4f}")
        print("Individual Class IoUs:")
        for i, name in enumerate(class_names):
            print(f"  - {name}: {class_ious[i]:.4f}")
        print("-" * 50)

        # --- SAVE TO CSV ---
        row_data = [epoch+1, avg_train_loss, avg_val_loss, mean_iou] + class_ious.tolist()
        metrics_df.loc[len(metrics_df)] = row_data
        metrics_df.to_csv(csv_filename, index=False)

        # --- SAVE BEST MODEL ---
        if mean_iou > best_iou:
            best_iou = mean_iou
            print(f"⭐ New Best Mean IoU! Saving model to 'best_offroad_model_01.pth'...")
            model.save_pretrained("best_offroad_model_01")
            # Saves the processor settings (like your custom 512x512 size)
            processor.save_pretrained("best_offroad_model_01")

    print(f"🎉 Training Complete! All metrics saved to {csv_filename}")


if __name__ == "__main__":
    # ---> UPDATE THESE 4 PATHS FOR YOUR LOCAL SETUP <---
    TRAIN_IMG = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Color_Images"
    TRAIN_MASK = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Segmentation"

    VAL_IMG = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Color_Images"
    VAL_MASK = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Segmentation"

    train_model(
        train_dir_img=TRAIN_IMG, 
        train_dir_mask=TRAIN_MASK, 
        val_dir_img=VAL_IMG, 
        val_dir_mask=VAL_MASK, 
        epochs=5,          # Adjust epochs based on your time
        batch_size=2        # If you run out of GPU memory, change this to 1
    )