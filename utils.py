"""
Utility functions for dataset loading, preprocessing, seeding, and plotting.
"""
import os
import random
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RANDOM_SEED, CLASS_NAMES, CLASS_TO_IDX,
    TRAIN_DIR, TEST_DIR, VAL_SPLIT_RATIO
)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image_rgb(image_path: str | Path) -> Image.Image:
    """
    Safely load an image, convert RGBA / Grayscale to 3-channel RGB.
    Ensures alpha channel is safely composited against a white background or stripped.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    with Image.open(path) as img:
        img_mode = img.mode
        if img_mode == "RGBA":
            # Composite RGBA over white background to avoid transparent black artifacts
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            return background
        elif img_mode != "RGB":
            return img.convert("RGB")
        else:
            return img.copy()


def scan_dataset(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Scan dataset folder (train and test directories) and construct a metadata DataFrame.
    """
    from config import TRAIN_DIR, TEST_DIR

    records = []
    splits = [("train", TRAIN_DIR), ("test", TEST_DIR)]

    for split_name, split_dir in splits:
        if not split_dir.exists():
            continue
        for class_name in CLASS_NAMES:
            class_folder = split_dir / class_name
            if not class_folder.is_dir():
                continue
            for file_path in class_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    records.append({
                        "file_path": str(file_path),
                        "filename": file_path.name,
                        "class_name": class_name,
                        "class_idx": CLASS_TO_IDX[class_name],
                        "original_split": split_name,
                        "extension": file_path.suffix.lower()
                    })

    df = pd.DataFrame(records)
    return df


def get_stratified_train_val_test(
    val_ratio: float = VAL_SPLIT_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constructs the 3 splits:
    - Train split (~85% of original train folder)
    - Val split (~15% of original train folder, stratified)
    - Test split (untouched 100% of original test folder)
    """
    full_df = scan_dataset()

    train_val_df = full_df[full_df["original_split"] == "train"].copy().reset_index(drop=True)
    test_df = full_df[full_df["original_split"] == "test"].copy().reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(train_val_df, train_val_df["class_idx"]))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def get_class_weights(labels: np.ndarray, num_classes: int = len(CLASS_NAMES)) -> torch.Tensor:
    """
    Compute balanced class weights for CrossEntropyLoss.
    weight = total_samples / (num_classes * count_per_class)
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels
    )
    return torch.tensor(weights, dtype=torch.float32)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path
) -> None:
    """
    Render and save a normalized & raw confusion matrix heatmap.
    """
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} - Raw Counts", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=10)
    axes[0].set_ylabel("True Label", fontsize=10)

    # Normalized percentages
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title} - Normalized (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=10)
    axes[1].set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
