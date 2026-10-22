"""
Approach 1: Traditional Computer Vision Classification Pipeline.
Extracts handcrafted color histograms (HSV), channel moments, and Local Binary Patterns (LBP) texture.
Trains and tunes RBF-SVM and Random Forest classifiers using training and validation splits.
Saves the best model, feature scaler, and metadata.
"""
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from config import (
    CLASS_NAMES, RANDOM_SEED, VAL_SPLIT_RATIO,
    HIST_BINS_HSV, LBP_RADIUS, LBP_POINTS, LBP_METHOD,
    MODELS_DIR
)
from utils import seed_everything, load_image_rgb, get_stratified_train_val_test


def extract_image_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a comprehensive 558-dimensional handcrafted feature vector:
    1. 3D HSV Color Histogram (8x8x8 = 512 features)
    2. Color statistical moments (Mean, Std, Skew across RGB & HSV = 18 features)
    3. Multi-scale Uniform Local Binary Patterns (LBP) texture histograms (10 + 18 = 28 features)
    """
    # 1. 3D HSV Color Histogram
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist_hsv = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        HIST_BINS_HSV, [0, 180, 0, 256, 0, 256]
    )
    hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()  # 512 dim

    # 2. Color Statistical Moments (RGB and HSV)
    moments = []
    for ch in range(3):
        # RGB channel
        c_rgb = image_rgb[:, :, ch].flatten().astype(float)
        moments.extend([np.mean(c_rgb), np.std(c_rgb), float(skew(c_rgb))])
        # HSV channel
        c_hsv = hsv[:, :, ch].flatten().astype(float)
        moments.extend([np.mean(c_hsv), np.std(c_hsv), float(skew(c_hsv))])
    color_moments = np.array(moments, dtype=np.float32)  # 18 dim

    # 3. Multi-scale LBP Texture Features
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Scale 1: radius 1, points 8 (uniform has 10 bins)
    lbp1 = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist_lbp1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 10), density=True)

    # Scale 2: radius 2, points 16 (uniform has 18 bins)
    lbp2 = local_binary_pattern(gray, P=16, R=2, method="uniform")
    hist_lbp2, _ = np.histogram(lbp2.ravel(), bins=18, range=(0, 18), density=True)

    lbp_features = np.hstack([hist_lbp1, hist_lbp2]).astype(np.float32)  # 28 dim

    # Concatenate all features into single feature vector
    feature_vector = np.hstack([hist_hsv, color_moments, lbp_features])
    return feature_vector


def extract_features_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for all images in a DataFrame."""
    X_list = []
    y_list = []

    for _, row in df.iterrows():
        img = load_image_rgb(row["file_path"]).resize((256, 256))
        img_np = np.array(img)
        feats = extract_image_features(img_np)
        X_list.append(feats)
        y_list.append(row["class_idx"])

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y


def train_traditional_pipeline():
    """Execute training, tuning, and validation for Traditional CV."""
    seed_everything(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("APPROACH 1: TRADITIONAL COMPUTER VISION PIPELINE")
    print("=" * 60)

    # 1. Load Stratified Data Splits
    train_df, val_df, test_df = get_stratified_train_val_test(VAL_SPLIT_RATIO, RANDOM_SEED)
    print(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Extract Features
    print("Extracting handcrafted features (HSV Hist + Moments + LBP)...")
    start_feat_time = time.time()
    X_train, y_train = extract_features_dataset(train_df)
    X_val, y_val = extract_features_dataset(val_df)
    feat_time = time.time() - start_feat_time
    print(f"Feature extraction completed in {feat_time:.2f}s. Feature shape: {X_train.shape}")

    # 3. Fit Preprocessing Scaler (strictly on train data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Train & Tune Candidate Classifiers
    candidates = {
        "SVM_RBF_C1": SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED),
        "SVM_RBF_C5": SVC(C=5.0, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED),
        "SVM_RBF_C10": SVC(C=10.0, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED),
        "RandomForest_150": RandomForestClassifier(n_estimators=150, max_depth=15, class_weight="balanced", random_state=RANDOM_SEED)
    }

    best_val_f1 = -1.0
    best_name = None
    best_model = None
    best_fit_time = 0.0

    print("\nEvaluating candidate models on validation split:")
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train_scaled, y_train)
        fit_dur = time.time() - t0

        val_preds = clf.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_preds)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro")

        print(f"  {name:<18} -> Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f} | Train Time: {fit_dur:.3f}s")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_name = name
            best_model = clf
            best_fit_time = fit_dur

    print(f"\nSelected Best Traditional Model: {best_name} (Val Macro F1 = {best_val_f1:.4f})")

    # 5. Measure Validation Inference Latency
    num_val = len(X_val_scaled)
    t_inf_start = time.time()
    for i in range(num_val):
        _ = best_model.predict_proba(X_val_scaled[i:i+1])
    total_inf_time = time.time() - t_inf_start
    avg_latency_ms = (total_inf_time / num_val) * 1000
    print(f"Average Inference Latency per sample (CPU): {avg_latency_ms:.2f} ms")

    # 6. Save Model Artifact
    model_bundle = {
        "model_name": best_name,
        "model_type": "Traditional_CV",
        "classifier": best_model,
        "scaler": scaler,
        "feature_dim": X_train.shape[1],
        "class_names": CLASS_NAMES,
        "train_time_sec": best_fit_time,
        "feature_extraction_time_sec": feat_time,
        "avg_latency_ms": avg_latency_ms,
        "best_val_macro_f1": best_val_f1
    }

    model_save_path = MODELS_DIR / "traditional_model.joblib"
    joblib.dump(model_bundle, model_save_path)
    print(f"Saved traditional model bundle to: {model_save_path}")

    # Also save metadata json
    meta_json_path = MODELS_DIR / "traditional_model_metadata.json"
    meta = {
        "model_name": best_name,
        "model_type": "Support Vector Classifier (RBF Kernel)" if "SVM" in best_name else "Random Forest",
        "feature_dimensions": X_train.shape[1],
        "class_names": CLASS_NAMES,
        "val_macro_f1": round(float(best_val_f1), 4),
        "train_time_sec": round(float(best_fit_time), 3),
        "inference_latency_ms_per_image": round(float(avg_latency_ms), 3)
    }
    with open(meta_json_path, "w") as f:
        json.dump(meta, f, indent=4)

    return model_bundle


if __name__ == "__main__":
    train_traditional_pipeline()
