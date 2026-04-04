import os
import torch
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import Counter
import torchvision.transforms as transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==========================================
# 1. Dataset Class (Original Augmentations)
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
            100: 0, #tree
            200: 1, #lush bushes
            300: 2, #dry grass
            500: 3, #dry bushes
            550: 4, #ground clutter
            600: 5, #flowers
            700: 6, #logs
            800: 7, #rocks  
            7100: 8, #landscape
            10000: 9 #sky
        }


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(img_path).convert("RGB")
        raw_mask = Image.open(mask_path)
        
        if self.train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                raw_mask = TF.hflip(raw_mask)
            
            if random.random() > 0.5:
                angle = random.uniform(-15, 15) 
                scale = random.uniform(0.85, 1.15) 
                image = TF.affine(image, angle=angle, translate=[0,0], scale=scale, shear=0, 
                                  interpolation=TF.InterpolationMode.BILINEAR)
                raw_mask = TF.affine(raw_mask, angle=angle, translate=[0,0], scale=scale, shear=0, 
                                     interpolation=TF.InterpolationMode.NEAREST)

            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.6, 1.4))
                image = TF.adjust_contrast(image, random.uniform(0.7, 1.4))
                image = TF.adjust_saturation(image, random.uniform(0.6, 1.2))

            if random.random() > 0.5:
                image = TF.gaussian_blur(image, kernel_size=[3, 3])
            
            if self.train and random.random() > 0.4:
                # Use 'transforms' here instead of 'TF'
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.5, 1.0), ratio=(0.75, 1.33)
                )
                
                # The functional calls for applying the crop remain the same
                image = TF.resized_crop(
                    image, i, j, h, w, size=(512, 512),
                    interpolation=TF.InterpolationMode.BILINEAR
                )
                raw_mask = TF.resized_crop(
                    raw_mask, i, j, h, w, size=(512, 512),
                    interpolation=TF.InterpolationMode.NEAREST # CRITICAL: no class ID blending
                )
                
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
        processed_inputs["image_name"] = img_name # <-- ADDED FOR TTA
        
        return processed_inputs

# ==========================================
# 2. Custom Collate Function
# ==========================================
def mask2former_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item.get("pixel_mask", torch.ones_like(item["pixel_values"][0])) for item in batch])
    original_masks = [item["original_mask"] for item in batch]
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    image_names = [item["image_name"] for item in batch] # <-- ADDED FOR TTA
    
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "original_mask": original_masks,
        "image_names": image_names # <-- ADDED FOR TTA
    }

# ==========================================
# 3. Test-Time Augmentation (TTA) Function
# ==========================================
def predict_with_tta(model, processor, image, device, target_size):
    img_orig = image
    img_flipped = TF.hflip(image)
    img_bright = TF.adjust_brightness(image, 1.3)

    variations = [img_orig, img_flipped, img_bright]
    all_logits = []

    for i, img_var in enumerate(variations):
        inputs = processor(images=img_var, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        mask_cls_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        mask_pred_probs = outputs.masks_queries_logits.sigmoid()
        
        semantic_logits = torch.einsum("bqc,bqhw->bchw", mask_cls_probs, mask_pred_probs)
        
        semantic_logits = F.interpolate(
            semantic_logits, size=target_size, mode="bilinear", align_corners=False
        )
        
        if i == 1:
            semantic_logits = torch.flip(semantic_logits, dims=[3])
            
        all_logits.append(semantic_logits)

    avg_logits = torch.mean(torch.stack(all_logits), dim=0)
    final_mask = torch.argmax(avg_logits, dim=1).squeeze(0)
    
    return final_mask

# ==========================================
# 4. Oversampling Weight Calculator
# ==========================================
def get_oversampling_weights(mask_dir):
    print("📊 Scanning dataset to calculate oversampling weights...")
    mask_names = sorted(os.listdir(mask_dir))
    weights = []
    rare_multipliers = {500: 5.0, 550: 10.0, 700: 15.0, 800: 8.0}
    
    for mask_name in tqdm(mask_names, desc="Analyzing Training Masks"):
        mask_path = os.path.join(mask_dir, mask_name)
        mask_np = np.array(Image.open(mask_path))
        weight = 0.5 
        for rare_val, multiplier in rare_multipliers.items():
            if np.any(mask_np == rare_val):
                weight += multiplier
        weights.append(weight)
        
    return torch.DoubleTensor(weights)

# ==========================================
# 5. Training Loop with TTA & Diagnostics
# ==========================================
def refine_model(train_dir_img, train_dir_mask, val_dir_img, val_dir_mask, epochs=5, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Continuing training on: {device}")

    model_path = "best_offroad_model_01"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found.")
        return

    processor = Mask2FormerImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
    class_weights = torch.tensor([3.0, 2.0, 1.0, 8.0, 8.0, 3.0, 20.0, 8.0, 1.0, 0.1]).to(device)

    train_dataset = DualityOffroadDataset(train_dir_img, train_dir_mask, processor, train=True)
    val_dataset = DualityOffroadDataset(val_dir_img, val_dir_mask, processor, train=False)
    sample_weights = get_oversampling_weights(train_dir_mask)
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=mask2former_collate_fn, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mask2former_collate_fn)

    # Original single learning rate optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    iou_metric_mean = MulticlassJaccardIndex(num_classes=10, ignore_index=255, average="macro").to(device)
    iou_metric_none = MulticlassJaccardIndex(num_classes=10, ignore_index=255, average="none").to(device)

    class_names = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
    csv_filename = "training_metrics.csv"
    columns = ["Epoch", "Train_Loss", "Val_Loss", "Mean_IoU"] + [f"IoU_{name.replace(' ', '_')}" for name in class_names]
    metrics_df = pd.DataFrame(columns=columns)

    print("🔥 Starting Refinement Loop...")
    best_iou = 0.67
    
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
            
            # Applying class weights
            current_batch_labels = torch.cat(batch["class_labels"]).to(device)
            if len(current_batch_labels) > 0:
                batch_importance = class_weights[current_batch_labels].mean()
            else:
                batch_importance = 1.0 
                
            weighted_loss = outputs.loss * batch_importance
            weighted_loss.backward()
            optimizer.step()
            
            train_loss += weighted_loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {weighted_loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # ==========================================
        # VALIDATION (WITH TTA)
        # ==========================================
        model.eval()
        val_loss = 0.0
        iou_metric_mean.reset()
        iou_metric_none.reset()

        with torch.no_grad():
            for batch in val_loader:
                # 1. Standard pass to get Validation Loss
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    pixel_mask=batch["pixel_mask"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]]
                )
                val_loss += outputs.loss.item()
                
                # 2. TTA Pass for actual predictions
                target_sizes = [mask.shape for mask in batch["original_mask"]]
                img_paths = [os.path.join(val_dir_img, name) for name in batch["image_names"]]
                
                predicted_maps = []
                for img_path, target_size in zip(img_paths, target_sizes):
                    raw_image = Image.open(img_path).convert("RGB")
                    final_mask = predict_with_tta(model, processor, raw_image, device, target_size)
                    predicted_maps.append(final_mask)
                    
                preds = torch.stack(predicted_maps).to(device)
                targets = torch.stack(batch["original_mask"]).to(device)
                
                iou_metric_mean.update(preds, targets)
                iou_metric_none.update(preds, targets)

        avg_val_loss = val_loss / len(val_loader)
        mean_iou = iou_metric_mean.compute().item()
        class_ious = iou_metric_none.compute().cpu().numpy()

        # ==========================================
        # STRATEGY C: DIAGNOSTICS FOR GROUND CLUTTER
        # ==========================================
        GC_CLASS = 4 
        print("\n🔍 Strategy C Diagnostics (Last Val Batch):")
        for i, (pred, target) in enumerate(zip(predicted_maps, batch['original_mask'])):
            gc_mask = (target == GC_CLASS)
            if gc_mask.sum() < 100:
                continue
            wrong = (pred.cpu() != target) & gc_mask
            confused_as = pred.cpu()[wrong] 
            if len(confused_as) > 0:
                top = Counter(confused_as.tolist()).most_common(3)
                print(f'  [Image {batch["image_names"][i]}] GC confused as: {[(class_names[c], n) for c, n in top]}')
            if i >= 5:
                break 

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

        if mean_iou >= best_iou:
            best_iou = mean_iou
            model.save_pretrained("best_offroad_model_refined")
            processor.save_pretrained("best_offroad_model_refined")
            print(f"⭐ Refined Best Model Saved (IoU: {best_iou:.4f})")

    print(f"🎉 Refinement Complete! Metrics saved to {csv_filename}")

if __name__ == "__main__":
    TRAIN_IMG = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Color_Images"
    TRAIN_MASK = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Segmentation"
    VAL_IMG = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Color_Images"
    VAL_MASK = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Segmentation"

    refine_model(TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK, epochs=2, batch_size=2)