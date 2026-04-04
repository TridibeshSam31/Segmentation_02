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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ==========================================
# [FIX] Copy-Paste Augmentation Helper
# ==========================================
class CopyPasteLogAugmentation:
    def __init__(self, image_dir, mask_dir, value_map, log_class_id=6, donor_pool_size=50):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.value_map = value_map
        self.log_class_id = log_class_id
        self.log_original_id = 700
        self.donor_pool = self._build_donor_pool(donor_pool_size)

    def _build_donor_pool(self, max_donors):
        mask_names = sorted(os.listdir(self.mask_dir))
        donors = []
        for mask_name in mask_names:
            if len(donors) >= max_donors:
                break
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask_np = np.array(Image.open(mask_path))
            if np.any(mask_np == self.log_original_id):
                donors.append((mask_name, mask_name))
        print(f"[Copy-Paste] Built donor pool with {len(donors)} log-containing images")
        return donors

    def __call__(self, image, mask, p=0.3):
        if random.random() > p or len(self.donor_pool) == 0:
            return image, mask

        donor_img_name, donor_mask_name = random.choice(self.donor_pool)
        donor_img_path = os.path.join(self.image_dir, donor_img_name)
        donor_mask_path = os.path.join(self.mask_dir, donor_mask_name)

        donor_image = Image.open(donor_img_path).convert("RGB")
        donor_mask_raw = np.array(Image.open(donor_mask_path))

        donor_mask = np.full_like(donor_mask_raw, fill_value=255, dtype=np.int64)
        for old_id, new_id in self.value_map.items():
            donor_mask[donor_mask_raw == old_id] = new_id

        log_mask = (donor_mask == self.log_class_id)
        if not log_mask.any():
            return image, mask

        image_np = np.array(image)
        donor_image_np = np.array(donor_image.resize(image.size, Image.BILINEAR))
        donor_mask_resized = cv2.resize(
            donor_mask.astype(np.float32),
            (image.size[0], image.size[1]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int64)
        log_mask_resized = (donor_mask_resized == self.log_class_id)

        image_np[log_mask_resized] = donor_image_np[log_mask_resized]
        mask[log_mask_resized] = self.log_class_id

        return Image.fromarray(image_np), mask


# ==========================================
# [FIX] Class-Aware Cropping Helper
# ==========================================
def get_class_aware_crop_params(mask, target_classes=[6, 4], crop_size=512, bias_prob=0.4):
    h, w = mask.shape

    if random.random() < bias_prob:
        combined_mask = np.zeros_like(mask, dtype=bool)
        for cls_id in target_classes:
            combined_mask |= (mask == cls_id)

        if combined_mask.any():
            coords = np.argwhere(combined_mask)
            if len(coords) > 0:
                center_idx = random.randint(0, len(coords) - 1)
                center_y, center_x = coords[center_idx]

                top = max(0, center_y - crop_size // 2)
                left = max(0, center_x - crop_size // 2)

                if top + crop_size > h:
                    top = h - crop_size
                if left + crop_size > w:
                    left = w - crop_size

                top = max(0, top)
                left = max(0, left)

                return top, left, crop_size, crop_size

    top = random.randint(0, max(0, h - crop_size))
    left = random.randint(0, max(0, w - crop_size))
    return top, left, crop_size, crop_size


# ==========================================
# 1. Dataset Class
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
            100: 0,    # tree
            200: 1,    # lush bushes
            300: 2,    # dry grass
            500: 3,    # dry bushes
            550: 4,    # ground clutter
            600: 5,    # flowers
            700: 6,    # logs
            800: 7,    # rocks
            7100: 8,   # landscape
            10000: 9   # sky
        }

        # [FIX] Both copy_paste_aug and albu_transform are now at the correct
        # indentation level inside __init__, not nested inside each other.
        if self.train:
            self.copy_paste_aug = CopyPasteLogAugmentation(
                image_dir, mask_dir, self.value_map, log_class_id=6
            )

        if self.train:
            # [FIX] is_check_shapes=False added as a safety net, but the real fix
            # is ensuring image_np and continuous_mask are always in sync before this call.
            self.albu_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0, scale_limit=0.15, rotate_limit=15, p=0.5,
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255
                ),
                A.ElasticTransform(alpha=120, sigma=12, p=0.15),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.GaussianBlur(blur_limit=(3, 3), p=0.5)
            ], is_check_shapes=True)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = Image.open(img_path).convert("RGB")
        raw_mask = Image.open(mask_path)

        mask_np = np.array(raw_mask)
        continuous_mask = np.full_like(mask_np, fill_value=255, dtype=np.int32)
        for old_id, new_id in self.value_map.items():
            continuous_mask[mask_np == old_id] = new_id

        if self.train:
            # Step 1: Copy-paste augmentation (operates on PIL image)
            image, continuous_mask = self.copy_paste_aug(image, continuous_mask, p=0.3)

            # Step 2: Horizontal flip (keep PIL + numpy in sync)
            if random.random() > 0.5:
                image = TF.hflip(image)
                continuous_mask = cv2.flip(continuous_mask, 1)

            # Step 3: Rotation and scale (keep PIL + numpy in sync)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                scale = random.uniform(0.85, 1.15)
                image = TF.affine(
                    image, angle=angle, translate=[0, 0], scale=scale, shear=0,
                    interpolation=TF.InterpolationMode.BILINEAR
                )
                image_np_tmp = np.array(image)
                h_tmp, w_tmp = image_np_tmp.shape[:2]
                center = (w_tmp // 2, h_tmp // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, scale)
                continuous_mask = cv2.warpAffine(
                    continuous_mask, matrix, (w_tmp, h_tmp),
                    flags=cv2.INTER_NEAREST, borderValue=255
                )
                # Sync PIL after affine
                image = Image.fromarray(image_np_tmp)

            # Step 4: Class-aware random crop
            crop_size = 512
            image_np = np.array(image)  # fresh numpy from synced PIL

            if image_np.shape[0] >= crop_size and image_np.shape[1] >= crop_size:
                top, left, h_crop, w_crop = get_class_aware_crop_params(
                    continuous_mask, target_classes=[6, 4],
                    crop_size=crop_size, bias_prob=0.4
                )
                image_np = image_np[top:top + h_crop, left:left + w_crop]
                continuous_mask = continuous_mask[top:top + h_crop, left:left + w_crop]

                if image_np.shape[0] != crop_size or image_np.shape[1] != crop_size:
                    image_np = cv2.resize(image_np, (crop_size, crop_size),
                                          interpolation=cv2.INTER_LINEAR)
                    continuous_mask = cv2.resize(continuous_mask, (crop_size, crop_size),
                                                 interpolation=cv2.INTER_NEAREST)
            else:
                image_np = cv2.resize(image_np, (crop_size, crop_size),
                                      interpolation=cv2.INTER_LINEAR)
                continuous_mask = cv2.resize(continuous_mask, (crop_size, crop_size),
                                             interpolation=cv2.INTER_NEAREST)

            # [FIX] Always sync PIL from image_np AFTER the entire crop block.
            # This was the primary bug: the old code only updated `image` in the else
            # branch, so `image_np = np.array(image)` below would silently re-read
            # the pre-crop PIL image, giving albumentations mismatched dimensions.
            image = Image.fromarray(image_np)

            # Step 5: Albumentations (elastic + color)
            # image_np is now guaranteed to match continuous_mask in size
            image_np = np.array(image)
            transformed = self.albu_transform(image=image_np, mask=continuous_mask)
            image_np = transformed['image']
            continuous_mask = transformed['mask']
            image = Image.fromarray(image_np)

            # Step 6: Extra color jitter (kept as backup alongside albumentations)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.6, 1.4))
                image = TF.adjust_contrast(image, random.uniform(0.7, 1.4))
                image = TF.adjust_saturation(image, random.uniform(0.6, 1.2))

            # Step 7: Gaussian blur
            if random.random() > 0.5:
                image = TF.gaussian_blur(image, kernel_size=[3, 3])

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
        processed_inputs["image_name"] = img_name

        return processed_inputs


# ==========================================
# 2. Custom Collate Function
# ==========================================
def mask2former_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([
        item.get("pixel_mask", torch.ones_like(item["pixel_values"][0]))
        for item in batch
    ])
    original_masks = [item["original_mask"] for item in batch]
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    image_names = [item["image_name"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "original_mask": original_masks,
        "image_names": image_names
    }


# ==========================================
# 3. Loss Functions
# ==========================================
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        probs = F.softmax(predictions, dim=1)
        num_classes = predictions.shape[1]

        # [FIX] targets may contain 255 (ignore); clamp before one_hot to avoid index error
        targets_clamped = targets.clamp(0, num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        valid_mask = (targets != self.ignore_index).float().unsqueeze(1)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        log_probs = F.log_softmax(predictions, dim=1)

        B, C, H, W = predictions.shape
        log_probs = log_probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)

        valid_mask = targets_flat != self.ignore_index
        targets_valid = targets_flat[valid_mask]
        log_probs_valid = log_probs[valid_mask]

        if targets_valid.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)

        log_pt = log_probs_valid.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets_valid]
            focal_loss = -alpha_t * focal_term * log_pt
        else:
            focal_loss = -focal_term * log_pt

        return focal_loss.mean()


def apply_ohem(predictions, targets, loss_tensor, top_k_ratio=0.25, ignore_index=255):
    targets_flat = targets.view(-1)
    valid_mask = targets_flat != ignore_index

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)

    valid_losses = loss_tensor[valid_mask]
    k = max(1, int(valid_losses.numel() * top_k_ratio))
    hard_losses, _ = torch.topk(valid_losses, k)

    return hard_losses.mean()


class CompositeLoss(torch.nn.Module):
    def __init__(self, class_weights, num_classes=10, ignore_index=255):
        super(CompositeLoss, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=ignore_index)

        self.ce_weight = 0.4
        self.dice_weight = 0.3
        self.focal_weight = 0.3

    def forward(self, predictions, targets, apply_ohem_flag=True):
        ce_loss_per_pixel = F.cross_entropy(
            predictions, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        if apply_ohem_flag:
            ce_loss = apply_ohem(
                predictions, targets, ce_loss_per_pixel.view(-1),
                top_k_ratio=0.25, ignore_index=self.ignore_index
            )
        else:
            ce_loss = ce_loss_per_pixel.mean()

        dice_loss = self.dice_loss(predictions, targets)
        focal_loss = self.focal_loss(predictions, targets)

        total_loss = (self.ce_weight * ce_loss +
                      self.dice_weight * dice_loss +
                      self.focal_weight * focal_loss)

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
            'focal_loss': focal_loss.item()
        }


# ==========================================
# 4. Semantic Logits Extractor
# ==========================================
def get_semantic_logits_from_mask2former(outputs, target_size):
    mask_cls_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    mask_pred_probs = outputs.masks_queries_logits.sigmoid()
    semantic_logits = torch.einsum("bqc,bqhw->bchw", mask_cls_probs, mask_pred_probs)
    semantic_logits = F.interpolate(
        semantic_logits, size=target_size, mode="bilinear", align_corners=False
    )
    return semantic_logits


# ==========================================
# 5. Test-Time Augmentation (TTA)
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
# 6. Oversampling Weight Calculator
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
# 7. Training Loop
# ==========================================
def refine_model(train_dir_img, train_dir_mask, val_dir_img, val_dir_mask,
                 epochs=5, batch_size=2, is_test_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Continuing training on: {device}")

    model_path = "best_offroad_model_01"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found.")
        return

    processor = Mask2FormerImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)

    # Class order: Trees(0), Lush Bushes(1), Dry Grass(2), Dry Bushes(3),
    #              Ground Clutter(4), Flowers(5), Logs(6), Rocks(7), Landscape(8), Sky(9)
    class_weights = torch.tensor([1.5, 1.5, 1.0, 1.5, 8.0, 1.5, 20.0, 1.5, 1.5, 0.1]).to(device)
    print(f"[Class Weights] Logs: 20.0, Ground Clutter: 8.0, Sky: 0.1")

    train_dataset = DualityOffroadDataset(train_dir_img, train_dir_mask, processor, train=True)
    val_dataset = DualityOffroadDataset(val_dir_img, val_dir_mask, processor, train=False)
    sample_weights = get_oversampling_weights(train_dir_mask)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=mask2former_collate_fn, sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=mask2former_collate_fn
    )

    # Freeze encoder
    for param in model.model.pixel_level_module.encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    composite_loss_fn = CompositeLoss(
        class_weights=class_weights, num_classes=10, ignore_index=255
    )
    composite_loss_fn.to(device)

    if is_test_mode:
        print("[Evaluation] Test Mode: Ignoring Logs and Ground Clutter in IoU computation")
    else:
        print("[Evaluation] Validation Mode: Evaluating all classes")

    iou_metric_mean = MulticlassJaccardIndex(num_classes=10, ignore_index=255, average="macro").to(device)
    iou_metric_none = MulticlassJaccardIndex(num_classes=10, ignore_index=255, average="none").to(device)

    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
        "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]
    csv_filename = "training_metrics_updated.csv"
    columns = (
        ["Epoch", "Train_Loss", "CE_Loss", "Dice_Loss", "Focal_Loss", "Val_Loss", "Mean_IoU"]
        + [f"IoU_{name.replace(' ', '_')}" for name in class_names]
    )
    metrics_df = pd.DataFrame(columns=columns)

    print("🔥 Starting Refinement Loop with Composite Loss...")
    best_iou = 0.67

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_ce_loss = 0.0
        train_dice_loss = 0.0
        train_focal_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                pixel_mask=batch["pixel_mask"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]]
            )

            target_masks = torch.stack(batch["original_mask"]).to(device)
            target_size = target_masks.shape[-2:]
            semantic_logits = get_semantic_logits_from_mask2former(outputs, target_size)

            total_loss, loss_components = composite_loss_fn(
                semantic_logits, target_masks, apply_ohem_flag=True
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_ce_loss += loss_components['ce_loss']
            train_dice_loss += loss_components['dice_loss']
            train_focal_loss += loss_components['focal_loss']

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {total_loss.item():.4f} | CE: {loss_components['ce_loss']:.4f} | "
                    f"Dice: {loss_components['dice_loss']:.4f} | Focal: {loss_components['focal_loss']:.4f}"
                )

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_ce_loss = train_ce_loss / len(train_loader)
        avg_dice_loss = train_dice_loss / len(train_loader)
        avg_focal_loss = train_focal_loss / len(train_loader)

        # ---- Validation ----
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

                target_masks = torch.stack(batch["original_mask"]).to(device)
                target_size = target_masks.shape[-2:]
                semantic_logits = get_semantic_logits_from_mask2former(outputs, target_size)

                batch_val_loss, _ = composite_loss_fn(
                    semantic_logits, target_masks, apply_ohem_flag=False
                )
                val_loss += batch_val_loss.item()

                target_sizes = [mask.shape for mask in batch["original_mask"]]
                img_paths = [os.path.join(val_dir_img, name) for name in batch["image_names"]]

                predicted_maps = []
                for img_path, t_size in zip(img_paths, target_sizes):
                    raw_image = Image.open(img_path).convert("RGB")
                    final_mask = predict_with_tta(model, processor, raw_image, device, t_size)
                    predicted_maps.append(final_mask)

                preds = torch.stack(predicted_maps).to(device)
                targets = torch.stack(batch["original_mask"]).to(device)

                iou_metric_mean.update(preds, targets)
                iou_metric_none.update(preds, targets)

        avg_val_loss = val_loss / len(val_loader)
        mean_iou = iou_metric_mean.compute().item()
        class_ious = iou_metric_none.compute().cpu().numpy()

        if is_test_mode:
            print("[Evaluation] Excluding Logs(6) and Ground Clutter(4) from test metrics")
            class_ious[4] = np.nan
            class_ious[6] = np.nan
            valid_ious = class_ious[~np.isnan(class_ious)]
            mean_iou = valid_ious.mean() if len(valid_ious) > 0 else 0.0

        # Ground Clutter diagnostics
        GC_CLASS = 4
        print("\n🔍 Ground Clutter Diagnostics (Last Val Batch):")
        for i, (pred, target) in enumerate(zip(predicted_maps, batch['original_mask'])):
            gc_mask = (target == GC_CLASS)
            if gc_mask.sum() < 100:
                continue
            wrong = (pred.cpu() != target) & gc_mask
            confused_as = pred.cpu()[wrong]
            if len(confused_as) > 0:
                top = Counter(confused_as.tolist()).most_common(3)
                print(
                    f'  [{batch["image_names"][i]}] GC confused as: '
                    f'{[(class_names[c], n) for c, n in top]}'
                )
            if i >= 5:
                break

        print("-" * 50)
        print(f"Epoch {epoch+1} Summary:")
        print(
            f"Train Loss: {avg_train_loss:.4f} "
            f"(CE: {avg_ce_loss:.4f}, Dice: {avg_dice_loss:.4f}, Focal: {avg_focal_loss:.4f})"
        )
        print(f"Val Loss: {avg_val_loss:.4f} | Mean IoU: {mean_iou:.4f}")
        print("Individual Class IoUs:")
        for i, name in enumerate(class_names):
            if is_test_mode and i in [4, 6]:
                print(f"  - {name}: N/A (excluded in test mode)")
            else:
                iou_val = class_ious[i]
                print(f"  - {name}: {iou_val:.4f}" if not np.isnan(iou_val) else f"  - {name}: N/A")
        print("-" * 50)

        row_data = (
            [epoch + 1, avg_train_loss, avg_ce_loss, avg_dice_loss, avg_focal_loss,
             avg_val_loss, mean_iou]
            + class_ious.tolist()
        )
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

    refine_model(TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK, epochs=10, batch_size=4, is_test_mode=False)