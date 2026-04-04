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
import torchvision.transforms as transforms
from collections import Counter
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==========================================
# 1. Augmentation & Loss Helpers
# ==========================================
class CopyPasteLogAugmentation:
    def __init__(self, image_dir, mask_dir, value_map, log_class_id=6, donor_pool_size=50):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.value_map, self.log_class_id = value_map, log_class_id
        self.log_original_id = 700
        self.donor_pool = self._build_donor_pool(donor_pool_size)

    def _build_donor_pool(self, max_donors):
        mask_names = sorted(os.listdir(self.mask_dir))
        donors = []
        for mask_name in mask_names:
            if len(donors) >= max_donors: break
            mask_np = np.array(Image.open(os.path.join(self.mask_dir, mask_name)))
            if np.any(mask_np == self.log_original_id):
                donors.append(mask_name)
        print(f"[Copy-Paste] Donor pool ready with {len(donors)} images.")
        return donors

    def __call__(self, image, mask, p=0.3):
        if random.random() > p or not self.donor_pool: return image, mask
        donor_mask_raw = np.array(Image.open(os.path.join(self.mask_dir, random.choice(self.donor_pool))))
        donor_image = np.array(Image.open(os.path.join(self.image_dir, random.choice(self.donor_pool))).convert("RGB").resize(image.size))
        
        d_mask = np.full_like(donor_mask_raw, 255, dtype=np.int64)
        for old, new in self.value_map.items(): d_mask[donor_mask_raw == old] = new
        
        d_mask_res = cv2.resize(d_mask.astype(np.float32), image.size, interpolation=cv2.INTER_NEAREST).astype(np.int64)
        log_m = (d_mask_res == self.log_class_id)
        
        img_np = np.array(image)
        img_np[log_m] = donor_image[log_m]
        mask[log_m] = self.log_class_id
        return Image.fromarray(img_np), mask

class TamedCompositeLoss(torch.nn.Module):
    def __init__(self, weights, ignore_index=255):
        super().__init__()
        self.weights, self.ignore_index = weights, ignore_index

    def forward(self, preds, targets):
        # 0.7 CrossEntropy + 0.2 Dice + 0.1 Focal
        ce = F.cross_entropy(preds, targets, weight=self.weights, ignore_index=self.ignore_index)
        
        # Dice Calculation
        probs = F.softmax(preds, dim=1)
        targets_oh = F.one_hot(targets.clamp(0, 9), 10).permute(0, 3, 1, 2).float()
        valid = (targets != self.ignore_index).float().unsqueeze(1)
        inter = (probs * targets_oh * valid).sum(dim=(2, 3))
        union = (probs * valid).sum(dim=(2, 3)) + (targets_oh * valid).sum(dim=(2, 3))
        dice = 1.0 - ((2. * inter + 1) / (union + 1)).mean()
        
        return 0.7 * ce + 0.2 * dice

# ==========================================
# 2. Dataset (The Bulletproof Version)
# ==========================================
class DualityOffroadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, train=False):
        self.image_dir, self.mask_dir, self.processor, self.train = image_dir, mask_dir, processor, train
        self.image_names = sorted(os.listdir(image_dir))
        self.value_map = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
        if train: self.cp_aug = CopyPasteLogAugmentation(image_dir, mask_dir, self.value_map)

    def __len__(self): return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        raw_m = Image.open(os.path.join(self.mask_dir, img_name))
        
        m_np = np.array(raw_m)
        cont_m = np.full_like(m_np, 255, dtype=np.int64)
        for old, new in self.value_map.items(): cont_m[m_np == old] = new

        if self.train:
            image, cont_m = self.cp_aug(image, cont_m, p=0.3)
            raw_m = Image.fromarray(cont_m.astype(np.uint8))
            
            if random.random() > 0.5:
                image, raw_m = TF.hflip(image), TF.hflip(raw_m)
            
            if random.random() > 0.4: # Resize/Crop Fix
                i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(0.75, 1.33))
                image = TF.resized_crop(image, i, j, h, w, size=(512, 512), interpolation=TF.InterpolationMode.BILINEAR)
                raw_m = TF.resized_crop(raw_m, i, j, h, w, size=(512, 512), interpolation=TF.InterpolationMode.NEAREST)
            else:
                image = TF.resize(image, (512, 512), interpolation=TF.InterpolationMode.BILINEAR)
                raw_m = TF.resize(raw_m, (512, 512), interpolation=TF.InterpolationMode.NEAREST)
            cont_m = np.array(raw_m).astype(np.int64)

        # Bulletproof Mask Labels Generation (Manual Bypass of HF Bug)
        inputs = self.processor(images=image, return_tensors="pt")
        processed = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        classes = np.unique(cont_m)
        classes = classes[classes != 255]
        m_labels, c_labels = [], []
        for cls in classes:
            m_labels.append(cont_m == cls)
            c_labels.append(cls)
            
        processed["mask_labels"] = torch.tensor(np.stack(m_labels), dtype=torch.float32) if len(m_labels) > 0 else torch.zeros((0, 512, 512))
        processed["class_labels"] = torch.tensor(c_labels, dtype=torch.long) if len(c_labels) > 0 else torch.zeros((0,), dtype=torch.long)
        processed["original_mask"] = torch.tensor(cont_m, dtype=torch.long)
        processed["image_name"] = img_name
        return processed

# ==========================================
# 3. Training Logic
# ==========================================
def mask2former_collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "original_mask": [item["original_mask"] for item in batch],
        "image_names": [item["image_name"] for item in batch]
    }

def get_semantic_logits(outputs, size):
    cls_p = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    mask_p = outputs.masks_queries_logits.sigmoid()
    logits = torch.einsum("bqc,bqhw->bchw", cls_p, mask_p)
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

def refine_model(train_img, train_mask, val_img, val_mask, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "best_offroad_model_01"
    
    # Dataset-Specific Normalization
    stats = {"image_mean": [0.47, 0.45, 0.44], "image_std": [0.27, 0.27, 0.30]}
    processor = Mask2FormerImageProcessor.from_pretrained(path, **stats)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(path).to(device)

    # NO ENCODER FREEZING in Script 11
    weights = torch.tensor([3., 2., 1., 8., 8., 3., 20., 8., 1., 0.1]).to(device)
    loss_fn = TamedCompositeLoss(weights)
    
    train_ds = DualityOffroadDataset(train_img, train_mask, processor, train=True)
    val_ds = DualityOffroadDataset(val_img, val_mask, processor, train=False)
    
    # Simplified weight calc for speed
    s_weights = torch.ones(len(train_ds)) 
    loader = DataLoader(train_ds, batch_size=2, collate_fn=mask2former_collate_fn, sampler=WeightedRandomSampler(s_weights, len(s_weights)))
    v_loader = DataLoader(val_ds, batch_size=2, collate_fn=mask2former_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    metric = MulticlassJaccardIndex(10, ignore_index=255).to(device)

    print("🔥 Script 11: Stabilizing Gradients...")
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            out = model(pixel_values=batch["pixel_values"].to(device),
                        mask_labels=[m.to(device) for m in batch["mask_labels"]],
                        class_labels=[c.to(device) for c in batch["class_labels"]])
            
            targets = torch.stack(batch["original_mask"]).to(device)
            logits = get_semantic_logits(out, targets.shape[-2:])
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            if i % 20 == 0: print(f"E{epoch+1} | B{i} | Loss: {loss.item():.4f}")

        model.eval()
        metric.reset()
        with torch.no_grad():
            for batch in v_loader:
                out = model(pixel_values=batch["pixel_values"].to(device))
                targets = torch.stack(batch["original_mask"]).to(device)
                logits = get_semantic_logits(out, targets.shape[-2:])
                metric.update(logits.argmax(1), targets)
        
        iou = metric.compute().item()
        print(f"✅ Epoch {epoch+1} Mean IoU: {iou:.4f}")
        if iou >= 0.67:
            model.save_pretrained("best_offroad_model_refined_v11")
            print("⭐ Improvement Saved!")

if __name__ == "__main__":
    # Update these paths to your local setup
    T_I = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Color_Images"
    T_M = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\train\Segmentation"
    V_I = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Color_Images"
    V_M = r"C:\Users\Tridibesh Samantroy\OneDrive\Desktop\IIT\Offroad_Segmentation_Training_Dataset\val\Segmentation"
    refine_model(T_I, T_M, V_I, V_M)