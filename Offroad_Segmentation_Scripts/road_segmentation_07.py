import os
import torch
import numpy as np
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from torchmetrics.classification import MulticlassJaccardIndex

# Set environment variable for library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==========================================
# 1. Dataset Class (Added Augmentation)
# ==========================================
class DualityOffroadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.train = train
        
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

        self.value_map = {
            100: 0, 200: 1, 300: 2, 500: 3, 550: 4,
            600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(img_path).convert("RGB")
        raw_mask = Image.open(mask_path)
        
        # Simple Augmentation: Horizontal Flip for Training
        if self.train and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            raw_mask = raw_mask.transpose(Image.FLIP_LEFT_RIGHT)

        mask_np = np.array(raw_mask)
        continuous_mask = np.full_like(mask_np, fill_value=255, dtype=np.int64)
        for old_id, new_id in self.value_map.items():
            continuous_mask[mask_np == old_id] = new_id

        inputs = self.processor(
            images=image,
            segmentation_maps=continuous_mask,
            return_tensors="pt"
        )
        
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                processed_inputs[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], torch.Tensor):
                processed_inputs[k] = v[0]
            else:
                processed_inputs[k] = v
        
        processed_inputs["original_mask"] = torch.tensor(continuous_mask, dtype=torch.long)
        return processed_inputs

# ==========================================
# 2. Custom Collate Function
# ==========================================
def mask2former_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item.get("pixel_mask", torch.ones_like(item["pixel_values"][0])) for item in batch])
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
# 3. Refined Training Loop
# ==========================================
def refine_model(train_dir_img, train_dir_mask, val_dir_img, val_dir_mask, epochs=3, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Continuing training on: {device}")

    # Load from YOUR best checkpoint
    model_path = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Scripts\best_offroad_model_01"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found. Please ensure you have the previous model saved.")
        return

    processor = Mask2FormerImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
    class_weights = torch.tensor([1.5, 1.5, 1.5, 4.0, 6.0, 2.0, 2.5, 3.0, 1.5, 0.5]).to(device)

    # Dataloaders (Train now includes Augmentation)
    train_dataset = DualityOffroadDataset(train_dir_img, train_dir_mask, processor, train=True)
    val_dataset = DualityOffroadDataset(val_dir_img, val_dir_mask, processor, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mask2former_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mask2former_collate_fn)

    # Lower Learning Rate for Fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    # Scheduler to lower LR gradually
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    iou_metric_mean = MulticlassJaccardIndex(num_classes=10, ignore_index=255, average="macro").to(device)
    iou_metric_none = MulticlassJaccardIndex(num_classes=10, ignore_index=255, average="none").to(device)

    class_names = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
    csv_filename = "training_metrics.csv"
    columns = ["Epoch", "Train_Loss", "Val_Loss", "Mean_IoU"] + [f"IoU_{name.replace(' ', '_')}" for name in class_names]
    metrics_df = pd.DataFrame(columns=columns)

    print("🔥 Starting Refinement Loop...")
    best_iou = 0.6542
    for epoch in range(epochs):
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
            current_batch_labels = torch.cat(batch["class_labels"]).to(device)
            batch_importance = class_weights[current_batch_labels].mean()
            weighted_loss = outputs.loss * batch_importance
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {weighted_loss.item():.4f}")


        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        iou_metric_mean.reset()
        iou_metric_none.reset()

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    pixel_mask=batch["pixel_mask"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]]
                )
                val_loss += outputs.loss.item()
                target_sizes = [mask.shape for mask in batch["original_mask"]]
                predicted_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                preds = torch.stack(predicted_maps).to(device)
                targets = batch["original_mask"].to(device)
                iou_metric_mean.update(preds, targets)
                iou_metric_none.update(preds, targets)

        avg_val_loss = val_loss / len(val_loader)
        mean_iou = iou_metric_mean.compute().item()
        class_ious = iou_metric_none.compute().cpu().numpy()

        print("-" * 50)
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Mean IoU: {mean_iou:.4f}")
        print("Individual Class IoUs:")
        for i, name in enumerate(class_names):
            print(f"  - {name}: {class_ious[i]:.4f}")
        print("-" * 50)

        row_data = [epoch+1, avg_train_loss, avg_val_loss, mean_iou] + class_ious.tolist()
        metrics_df.loc[len(metrics_df)] = row_data
        metrics_df.to_csv(csv_filename, index=False)

        # SAVE BEST MODEL
        if mean_iou > best_iou:
            best_iou = mean_iou
            model.save_pretrained("best_offroad_model_refined")
            processor.save_pretrained("best_offroad_model_refined")
            print(f"⭐ Refined Best Model Saved (IoU: {best_iou:.4f})")

    print(f"🎉 Refinement Complete! Metrics saved to {csv_filename}")

if __name__ == "__main__":
    # Ensure these paths are correct for your machine
    TRAIN_IMG = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Color_Images"
    TRAIN_MASK = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Segmentation"
    VAL_IMG = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Color_Images"
    VAL_MASK = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Segmentation"

    refine_model(TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK, epochs=3, batch_size=2)