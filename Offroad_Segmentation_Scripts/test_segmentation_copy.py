"""
Segmentation Validation Script (FIXED)
Compatible with SegmentationHeadUPNet (training script)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

plt.switch_backend('Agg')


# =========================================================
# MASK + COLOR
# =========================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

class_names = [
    'Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
    'Ground Clutter','Logs','Rocks','Landscape','Sky'
]

n_classes = len(value_map)

color_palette = np.array([
    [0,0,0],[34,139,34],[0,255,0],[210,180,140],[139,90,43],
    [128,128,0],[139,69,19],[128,128,128],[160,82,45],[135,206,235]
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n_classes):
        out[mask == i] = color_palette[i]
    return out


# =========================================================
# DATASET (FIXED MASK HANDLING)
# =========================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, name))
        mask = convert_mask(mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            mask = (mask * 255).long()   # ✅ FIXED

        return img, mask, name


# =========================================================
# MODEL (MATCH TRAINING EXACTLY)
# =========================================================

class SegmentationHeadUPNet(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.upconv1(x)))
        x = F.relu(self.bn3(self.upconv2(x)))

        return self.classifier(x)


# =========================================================
# METRICS
# =========================================================

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    ious = []
    for c in range(n_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append((inter / union).item())

    return np.nanmean(ious), ious


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="segmentation_head.pth")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # SAME SIZE AS TRAINING
    w = int(((960/2)//14)*14)
    h = int(((540/2)//14)*14)

    transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor()
    ])

    dataset = MaskDataset(args.data_dir, transform, mask_transform)
    loader = DataLoader(dataset, batch_size=1)

    # BACKBONE (MATCH TRAINING)
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    # GET EMBEDDING
    sample,_ ,_ = dataset[0]
    sample = sample.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = backbone.forward_features(sample)["x_norm_patchtokens"]
    dim = emb.shape[2]

    # LOAD MODEL
    model = SegmentationHeadUPNet(dim, n_classes, w//14, h//14).to(device)

    # Graceful fallback if the model file isn't found during initial testing
    if os.path.exists(args.model_path):
        state = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file {args.model_path} not found. Running with uninitialized weights to test pipeline.")

    model.eval()

    # RUN
    all_ious = []

    with torch.no_grad():
        for imgs, labels, names in tqdm(loader, desc="Generating Predictions"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feat)

            out = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear")

            # 1. Compute metrics
            iou, _ = compute_iou(out, labels.squeeze(1))
            all_ious.append(iou)

            # 2. Extract and Save the Colored Predictions (NEW CODE)
            # Convert logits to class predictions (Shape: Batch Size x H x W)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            
            for i in range(preds.shape[0]):
                color_mask = mask_to_color(preds[i])
                # Convert RGB to BGR for OpenCV saving
                color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(args.output_dir, names[i])
                cv2.imwrite(save_path, color_mask_bgr)

    print(f"\nSaved all image predictions to: {os.path.abspath(args.output_dir)}")
    print("FINAL Mean IoU:", np.nanmean(all_ious))


if __name__ == "__main__":
    main()