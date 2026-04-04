"""
Segmentation Training Script
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm
import random
import torchvision.transforms.functional as TF

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions (KEPT AS IS)
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion (KEPT AS IS)
# ============================================================================

value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# 1. NEW: Dice Loss Component (Fix Strategy #1)
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target):
        inputs = F.softmax(inputs, dim=1)
        target_one_hot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(inputs * target_one_hot, dims)
        cardinality = torch.sum(inputs + target_one_hot, dims)
        
        dice = (2. * intersection / (cardinality + 1e-7)).mean()
        return 1 - dice


# ============================================================================
# Dataset (MODIFIED: Fix Strategy #4 & #6)
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, size=(518, 518), is_train=True):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.size = size
        self.is_train = is_train
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = convert_mask(Image.open(mask_path))

        # POINT 4: Synchronized Augmentation
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random Crop and Resize (ensures boundaries are learned)
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.75, 1.33))
            image = TF.resized_crop(image, i, j, h, w, self.size, Image.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)

            # Color Jitter (only on image)
            if random.random() > 0.3:
                image = transforms.ColorJitter(0.2, 0.2, 0.2)(image)
        else:
            image = TF.resize(image, self.size, Image.BILINEAR)
            mask = TF.resize(mask, self.size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)
        
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ============================================================================
# Model (MODIFIED: Fix Strategy #2)
# ============================================================================

class SegmentationHeadUPNet(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        # Learnable Upsampling Blocks
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.upconv1(x)))
        x = F.relu(self.bn3(self.upconv2(x)))

        return self.classifier(x)


# ============================================================================
# Metrics (KEPT AS IS)
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index: continue
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0: iou_per_class.append(float('nan'))
        else: iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class)

def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())
    return np.mean(dice_per_class)

def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Plotting Functions (KEPT AS IS)
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")

def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # POINT 6: Resolution Increase (518 is 37 * 14 patches)
    h, w = 518, 518
    batch_size = 16
    lr = 1e-4
    n_epochs = 40

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    train_loader = DataLoader(MaskDataset(data_dir, transform=transform, size=(h, w), is_train=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MaskDataset(val_dir, transform=transform, size=(h, w), is_train=False), batch_size=batch_size, shuffle=False)

    BACKBONE_SIZE = "base"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    print("Checking embedding dimensions...")
    with torch.no_grad():
        # Create a tiny dummy image to see what the backbone outputs
        dummy_img = torch.zeros(1, 3, h, w).to(device)
        dummy_output = backbone_model.forward_features(dummy_img)["x_norm_patchtokens"]
        n_embedding = dummy_output.shape[-1] 
    
    print(f"Backbone loaded. Embedding dimension: {n_embedding}")
    classifier = SegmentationHeadUPNet(
        in_channels=n_embedding,  # This will now correctly be 768
        out_channels=n_classes, 
        tokenW=w//14, 
        tokenH=h//14
    ).to(device)
    # POINT 1 & 5: Hybrid Loss + Weights
    class_weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 3.0, 4.0, 3.0, 3.0, 1.0, 0.05]).to(device)
    ce_loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fct = DiceLoss(n_classes=n_classes)

    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    # POINT 3: ReduceLR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    history = {k: [] for k in ['train_loss', 'val_loss', 'train_iou', 'val_iou', 'train_dice', 'val_dice', 'train_pixel_acc', 'val_pixel_acc']}
    best_val_iou = 0.0

    print("\nStarting training...")
    for epoch in range(n_epochs):
        classifier.train()
        t_metrics = {'loss': [], 'iou': [], 'dice': [], 'acc': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            logits = classifier(feat)
            preds = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            
            # POINT 1: Combined Loss
            loss = 0.5 * ce_loss_fct(preds, labels) + 0.5 * dice_loss_fct(preds, labels)
            
            loss.backward(); optimizer.step(); optimizer.zero_grad()

            t_metrics['loss'].append(loss.item())
            with torch.no_grad():
                t_metrics['iou'].append(compute_iou(preds, labels))
                t_metrics['dice'].append(compute_dice(preds, labels))
                t_metrics['acc'].append(compute_pixel_accuracy(preds, labels))

        # Validation
        classifier.eval()
        v_metrics = {'loss': [], 'iou': [], 'dice': [], 'acc': []}
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(feat)
                preds = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                
                loss = 0.5 * ce_loss_fct(preds, labels) + 0.5 * dice_loss_fct(preds, labels)
                v_metrics['loss'].append(loss.item())
                v_metrics['iou'].append(compute_iou(preds, labels))
                v_metrics['dice'].append(compute_dice(preds, labels))
                v_metrics['acc'].append(compute_pixel_accuracy(preds, labels))

        # Update History
        for k in history.keys():
            m = t_metrics if 'train' in k else v_metrics
            history[k].append(np.nanmean(m[k.split('_')[-1] if 'iou' in k else k.split('_')[-1]]))

        cur_iou = history['val_iou'][-1]
        scheduler.step(cur_iou)

        if cur_iou > best_val_iou:
            best_val_iou = cur_iou
            torch.save(classifier.state_dict(), os.path.join(script_dir, "segmentation_head_best.pth"))

        print(f"Epoch {epoch+1} - Val IoU: {cur_iou:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")
        

    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)
    torch.save(classifier.state_dict(), os.path.join(script_dir, "segmentation_head_v2.pth"))
    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    main()