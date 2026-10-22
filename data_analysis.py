"""
Dataset Analysis Pipeline for Orange Condition Classification Dataset.
Performs data integrity verification, duplicate detection, perceptual hash audit,
color/outlier analysis, split validation, and visual chart generation.
"""
import os
import json
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_DIR, TRAIN_DIR, TEST_DIR, CLASS_NAMES, CLASS_TO_IDX,
    DATASET_ANALYSIS_DIR, RANDOM_SEED, VAL_SPLIT_RATIO
)
from utils import load_image_rgb, get_stratified_train_val_test, seed_everything


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Calculate difference hash (dHash) for perceptual image comparison."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(gray.getdata(), dtype=np.int32).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "".join(f"{b:02x}" for b in np.packbits(diff.flatten()))


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Compute Hamming distance between two hex string hashes."""
    b1 = bin(int(hex_hash1, 16))[2:].zfill(64)
    b2 = bin(int(hex_hash2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))


def analyze_dataset() -> Dict[str, Any]:
    """Execute end-to-end dataset analysis."""
    seed_everything(RANDOM_SEED)
    DATASET_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("STARTING COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 60)

    # 1. Image Discovery and Integrity Check
    records = []
    corrupted_files = []
    md5_hashes = {}
    exact_duplicates = []

    all_splits = [("train", TRAIN_DIR), ("test", TEST_DIR)]

    for split_name, split_path in all_splits:
        for class_name in CLASS_NAMES:
            folder = split_path / class_name
            if not folder.exists():
                continue
            for f in folder.iterdir():
                if not f.is_file() or f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                # Verify image readable
                try:
                    with Image.open(f) as img:
                        img.verify()
                except Exception as e:
                    corrupted_files.append((str(f), str(e)))
                    continue

                # Compute MD5
                with open(f, "rb") as fp:
                    data = fp.read()
                    md5 = hashlib.md5(data).hexdigest()
                    if md5 in md5_hashes:
                        exact_duplicates.append((str(f), md5_hashes[md5]))
                    else:
                        md5_hashes[md5] = str(f)

                # Read image properties
                with Image.open(f) as img:
                    w, h = img.size
                    mode = img.mode

                # Safe load for perceptual analysis and color stats
                rgb_img = load_image_rgb(f)
                dhash_val = compute_dhash(rgb_img)
                np_img = np.array(rgb_img)

                # Mean color channels (RGB)
                mean_r = float(np.mean(np_img[:, :, 0]))
                mean_g = float(np.mean(np_img[:, :, 1]))
                mean_b = float(np.mean(np_img[:, :, 2]))

                # Mean HSV
                hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
                mean_h = float(np.mean(hsv_img[:, :, 0]))
                mean_s = float(np.mean(hsv_img[:, :, 1]))
                mean_v = float(np.mean(hsv_img[:, :, 2]))

                records.append({
                    "file_path": str(f),
                    "filename": f.name,
                    "split": split_name,
                    "class_name": class_name,
                    "class_idx": CLASS_TO_IDX[class_name],
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "extension": f.suffix.lower(),
                    "md5": md5,
                    "dhash": dhash_val,
                    "mean_r": mean_r,
                    "mean_g": mean_g,
                    "mean_b": mean_b,
                    "mean_h": mean_h,
                    "mean_s": mean_s,
                    "mean_v": mean_v,
                })

    df = pd.DataFrame(records)
    total_images = len(df)
    print(f"Total valid images scanned: {total_images}")
    print(f"Total corrupted images: {len(corrupted_files)}")
    print(f"Total exact byte-level duplicates: {len(exact_duplicates)}")

    # 2. Near-Duplicate & Cross-Split Leakage Audit
    train_records = df[df["split"] == "train"].to_dict("records")
    test_records = df[df["split"] == "test"].to_dict("records")

    near_duplicate_audit = []
    print("\nAuditing cross-split near-duplicates (train vs test)...")

    for te in test_records:
        test_img_np = np.array(load_image_rgb(te["file_path"]).resize((224, 224)))
        for tr in train_records:
            dist = hamming_distance(te["dhash"], tr["dhash"])
            if dist <= 4:  # Candidate perceptual threshold
                train_img_np = np.array(load_image_rgb(tr["file_path"]).resize((224, 224)))
                # Calculate Structural Similarity Index (SSIM)
                ssim_score = ssim(
                    test_img_np, train_img_np,
                    channel_axis=2, data_range=255
                )
                mse_score = float(np.mean((test_img_np.astype("float") - train_img_np.astype("float")) ** 2))

                # Evaluation determination:
                # If ssim > 0.85 and dist <= 2, it is a sequential burst or identical fruit.
                # If ssim < 0.70, it's merely a coincidentally similar circular citrus outline.
                is_genuine_leak = (ssim_score > 0.80 and dist <= 3)

                near_duplicate_audit.append({
                    "test_file": te["filename"],
                    "test_class": te["class_name"],
                    "test_path": te["file_path"],
                    "train_file": tr["filename"],
                    "train_class": tr["class_name"],
                    "train_path": tr["file_path"],
                    "hamming_distance": dist,
                    "ssim": round(float(ssim_score), 4),
                    "mse": round(mse_score, 2),
                    "same_class": te["class_name"] == tr["class_name"],
                    "leakage_assessment": "Confirmed Near-Duplicate Fruit" if is_genuine_leak else "Geometry/Background Similarity"
                })

    audit_df = pd.DataFrame(near_duplicate_audit)
    audit_csv_path = DATASET_ANALYSIS_DIR / "near_duplicates_audit.csv"
    audit_df.to_csv(audit_csv_path, index=False)
    confirmed_leaks = audit_df[audit_df["leakage_assessment"] == "Confirmed Near-Duplicate Fruit"]
    print(f"Total potential pairs flagged (hamming <= 4): {len(audit_df)}")
    print(f"Confirmed genuine near-duplicate fruits cross-split: {len(confirmed_leaks)}")

    # 3. Class Counts and Splits
    train_df, val_df, test_df = get_stratified_train_val_test(VAL_SPLIT_RATIO, RANDOM_SEED)
    class_dist_data = {
        "class": CLASS_NAMES,
        "train_count": [sum(train_df["class_name"] == c) for c in CLASS_NAMES],
        "val_count": [sum(val_df["class_name"] == c) for c in CLASS_NAMES],
        "test_count": [sum(test_df["class_name"] == c) for c in CLASS_NAMES],
        "total_count": [sum(df["class_name"] == c) for c in CLASS_NAMES]
    }
    dist_df = pd.DataFrame(class_dist_data)
    dist_csv_path = DATASET_ANALYSIS_DIR / "class_distribution_summary.csv"
    dist_df.to_csv(dist_csv_path, index=False)

    # 4. Generate Visualizations
    # A) Class Distribution Chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    plt.bar(x - width, dist_df["train_count"], width, label="Train (85% of train dir)", color="#2b5c8f")
    plt.bar(x, dist_df["val_count"], width, label="Val (15% of train dir)", color="#469b88")
    plt.bar(x + width, dist_df["test_count"], width, label="Test (held-out test dir)", color="#d95f02")

    for i in range(len(CLASS_NAMES)):
        plt.text(x[i] - width, dist_df["train_count"][i] + 4, str(dist_df["train_count"][i]), ha="center", fontsize=9)
        plt.text(x[i], dist_df["val_count"][i] + 4, str(dist_df["val_count"][i]), ha="center", fontsize=9)
        plt.text(x[i] + width, dist_df["test_count"][i] + 4, str(dist_df["test_count"][i]), ha="center", fontsize=9)

    plt.xlabel("Condition Class", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Images", fontsize=11, fontweight="bold")
    plt.title("Orange Condition Dataset Split Distribution (Stratified)", fontsize=13, fontweight="bold")
    plt.xticks(x, [c.capitalize() for c in CLASS_NAMES])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    dist_chart_path = DATASET_ANALYSIS_DIR / "class_distribution.png"
    plt.savefig(dist_chart_path, dpi=300)
    plt.close()

    # B) Sample Image Grid (4 classes x 4 sample images = 16 images)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for row_idx, cls in enumerate(CLASS_NAMES):
        cls_df = df[df["class_name"] == cls].sample(4, random_state=RANDOM_SEED).reset_index(drop=True)
        for col_idx in range(4):
            img_path = cls_df.loc[col_idx, "file_path"]
            img = load_image_rgb(img_path)
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_title(f"{cls} ({cls_df.loc[col_idx, 'width']}x{cls_df.loc[col_idx, 'height']})", fontsize=10)
            ax.axis("off")

    plt.suptitle("Representative Dataset Samples Across All 4 Classes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    sample_grid_path = DATASET_ANALYSIS_DIR / "sample_grid.png"
    plt.savefig(sample_grid_path, dpi=300)
    plt.close()

    # C) Color Distribution Plots (HSV Hue & Saturation Violin / Boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="class_name", y="mean_h", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Mean Hue (H) Distribution by Class", fontweight="bold")
    axes[0].set_ylabel("Mean Hue (0-179)")
    axes[0].set_xlabel("Condition Class")

    sns.boxplot(x="class_name", y="mean_s", data=df, ax=axes[1], palette="Set2")
    axes[1].set_title("Mean Saturation (S) Distribution by Class", fontweight="bold")
    axes[1].set_ylabel("Mean Saturation (0-255)")
    axes[1].set_xlabel("Condition Class")

    plt.tight_layout()
    color_dist_path = DATASET_ANALYSIS_DIR / "color_distributions.png"
    plt.savefig(color_dist_path, dpi=300)
    plt.close()

    # 5. Compile Statistics Summary
    stats = {
        "dataset_name": "Orange Post-Harvest Quality & Condition Dataset",
        "total_images": total_images,
        "corrupted_images_count": len(corrupted_files),
        "exact_duplicates_count": len(exact_duplicates),
        "classes": {
            c: {
                "total": int(sum(df["class_name"] == c)),
                "train": int(sum(train_df["class_name"] == c)),
                "val": int(sum(val_df["class_name"] == c)),
                "test": int(sum(test_df["class_name"] == c)),
                "percentage": round(float(sum(df["class_name"] == c) / total_images * 100), 2)
            }
            for c in CLASS_NAMES
        },
        "extensions": dict(Counter(df["extension"])),
        "modes": dict(Counter(df["mode"])),
        "resolutions": {
            "most_common": str(Counter(zip(df["width"], df["height"])).most_common(3)),
            "min_resolution": [int(df["width"].min()), int(df["height"].min())],
            "max_resolution": [int(df["width"].max()), int(df["height"].max())]
        },
        "data_leakage_audit": {
            "potential_pairs_hamming_le_4": len(audit_df),
            "confirmed_near_duplicate_fruits": len(confirmed_leaks),
            "cross_split_leakage_note": "A small number of fruit samples in the original train and test folders show high perceptual similarity (sequential burst photos of the same fruit). The held-out test folder is evaluated both overall and with explicit leakage reporting."
        }
    }

    stats_json_path = DATASET_ANALYSIS_DIR / "dataset_statistics.json"
    with open(stats_json_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\nDataset Analysis Completed Successfully!")
    print(f"Outputs generated in: {DATASET_ANALYSIS_DIR}")
    return stats


if __name__ == "__main__":
    analyze_dataset()
