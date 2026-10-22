"""
Approach 2: Lightweight Deep Learning Classification Pipeline.
Implements Transfer Learning using MobileNetV2 with ImageNet pretrained weights.
Includes:
- Training-only data augmentations (flip, rotation, affine, jitter)
- Balanced class-weighted CrossEntropyLoss
- Two-stage training: Head warm-up followed by fine-tuning of upper inverted residual blocks
- Early stopping & best validation checkpointing
- Training history tracking and curve plotting
"""
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CLASS_NAMES, NUM_CLASSES, RANDOM_SEED, VAL_SPLIT_RATIO,
    IMAGE_SIZE_CNN, BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS,
    HEAD_LR, FINETUNE_LR, WEIGHT_DECAY, EARLY_STOPPING_PATIENCE,
    IMAGENET_MEAN, IMAGENET_STD,
    MODELS_DIR, TRAINING_CURVES_DIR
)
from utils import (
    seed_everything, load_image_rgb,
    get_stratified_train_val_test, get_class_weights
)


class CitrusDataset(Dataset):
    """PyTorch Dataset for orange condition classification images."""
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = load_image_rgb(row["file_path"])
        tensor_img = self.transform(image)
        label = int(row["class_idx"])
        return tensor_img, label


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train (augmented) and validation/test (deterministic) transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE_CNN),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE_CNN),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


def build_mobilenet_v2(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build MobileNetV2 with ImageNet pretrained weights and customized classifier head."""
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze feature backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head (1280 features -> num_classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_cnn_pipeline():
    """Execute deep learning training and validation pipeline."""
    seed_everything(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_CURVES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"APPROACH 2: DEEP LEARNING (MobileNetV2) PIPELINE ON {device}")
    print("=" * 60)

    # 1. Dataset Splits
    train_df, val_df, test_df = get_stratified_train_val_test(VAL_SPLIT_RATIO, RANDOM_SEED)
    print(f"Splits loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_transform, val_transform = get_transforms()
    train_dataset = CitrusDataset(train_df, train_transform)
    val_dataset = CitrusDataset(val_df, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )

    # 2. Balanced Class Weights for CrossEntropyLoss
    class_weights = get_class_weights(train_df["class_idx"].values, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Computed Class Weights: {dict(zip(CLASS_NAMES, [round(w.item(), 3) for w in class_weights]))}")

    # 3. Model Setup
    model = build_mobilenet_v2(NUM_CLASSES).to(device)

    # 4. Two-Stage Training:
    # Stage 1: Warmup head (3 epochs)
    # Stage 2: Fine-tune top blocks (features[14:]) (7 epochs)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = MODELS_DIR / "best_mobilenetv2.pt"
    patience_counter = 0
    total_train_start = time.time()

    print("\n--- Phase 1: Warmup Classification Head (Backbone Frozen) ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # Unfreeze top feature layers at epoch 4 for fine-tuning
        if epoch == 4:
            print("\n--- Phase 2: Fine-Tuning Top Backbone Blocks (features[14:]) ---")
            for i, block in enumerate(model.features):
                if i >= 14:
                    for param in block.parameters():
                        param.requires_grad = True
            # Recreate optimizer with lower learning rate for fine-tuning
            optimizer = torch.optim.AdamW([
                {"params": model.classifier.parameters(), "lr": FINETUNE_LR * 2},
                {"params": [p for p in model.features[14:].parameters() if p.requires_grad], "lr": FINETUNE_LR}
            ], weight_decay=WEIGHT_DECAY)

        # Training Loop
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_train_preds.extend(preds)
            all_train_targets.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_dataset)
        train_acc = accuracy_score(all_train_targets, all_train_preds)

        # Validation Loop
        model.eval()
        val_running_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_val_preds.extend(preds)
                all_val_targets.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_dataset)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_macro_f1 = f1_score(all_val_targets, all_val_preds, average="macro")
        epoch_dur = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))

        print(
            f"Epoch [{epoch:02d}/{MAX_EPOCHS:02d}] ({epoch_dur:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro-F1: {val_macro_f1:.4f}"
        )

        # Checkpointing based on Validation Macro F1
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES
            }, best_model_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Macro F1 = {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE and epoch >= 6:
                print(f"Early stopping triggered at epoch {epoch} (No improvement for {EARLY_STOPPING_PATIENCE} epochs).")
                break

    total_train_time = time.time() - total_train_start
    print(f"\nTraining completed in {total_train_time:.2f}s! Best Validation Macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # 5. Measure Latency on Validation set
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Latency test (warmup 5 iterations, then average 50 images)
    dummy_input = torch.randn(1, 3, *IMAGE_SIZE_CNN).to(device)
    for _ in range(5):
        _ = model(dummy_input)

    inf_times = []
    with torch.no_grad():
        for i in range(min(50, len(val_dataset))):
            img_tensor, _ = val_dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            t0 = time.time()
            _ = model(img_tensor)
            inf_times.append((time.time() - t0) * 1000)

    avg_latency_ms = float(np.mean(inf_times))
    print(f"Average CNN Inference Latency (CPU): {avg_latency_ms:.2f} ms per image")

    # 6. Save Training Curves
    plt.figure(figsize=(13, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    plt.plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="#d62728")
    plt.axvline(best_epoch, linestyle="--", color="gray", label=f"Best Checkpoint (Ep {best_epoch})")
    plt.title("MobileNetV2 Training and Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Accuracy / F1 subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], "o-", label="Train Accuracy", color="#1f77b4")
    plt.plot(epochs_range, history["val_acc"], "s-", label="Val Accuracy", color="#2ca02c")
    plt.plot(epochs_range, history["val_macro_f1"], "^-", label="Val Macro F1", color="#ff7f0e")
    plt.axvline(best_epoch, linestyle="--", color="gray", label=f"Best Checkpoint (Ep {best_epoch})")
    plt.title("MobileNetV2 Accuracy & Macro F1 Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curves_path = TRAINING_CURVES_DIR / "mobilenetv2_curves.png"
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {curves_path}")

    # 7. Save CNN Metadata
    meta_path = MODELS_DIR / "cnn_model_metadata.json"
    meta = {
        "architecture": "MobileNetV2 (ImageNet Pretrained)",
        "input_resolution": list(IMAGE_SIZE_CNN),
        "total_train_time_sec": round(total_train_time, 2),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(float(best_val_f1), 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "class_names": CLASS_NAMES,
        "history": history
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Saved metadata to: {meta_path}")

    return history


if __name__ == "__main__":
    train_cnn_pipeline()
