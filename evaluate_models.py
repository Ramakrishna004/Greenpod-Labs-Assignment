"""
Model Evaluation and Comparison Pipeline.
Evaluates both Approach 1 (Handcrafted CV + SVM) and Approach 2 (MobileNetV2 CNN)
on the identical, untouched 99-image held-out test set.
Computes all required metrics, confusion matrices, latency, and comparison plots.
"""
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import joblib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    CLASS_NAMES, NUM_CLASSES, RANDOM_SEED,
    IMAGE_SIZE_CNN, IMAGENET_MEAN, IMAGENET_STD,
    MODELS_DIR, CONFUSION_MATRICES_DIR, COMPARISON_DIR, REPORT_DIR
)
from utils import (
    seed_everything, load_image_rgb,
    scan_dataset, plot_confusion_matrix_heatmap
)
from train_traditional import extract_image_features


def load_test_data() -> pd.DataFrame:
    """Scan and retrieve the untouched held-out test set."""
    full_df = scan_dataset()
    test_df = full_df[full_df["original_split"] == "test"].copy().reset_index(drop=True)
    return test_df


def evaluate_traditional_model(
    test_df: pd.DataFrame
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Traditional Computer Vision pipeline on the held-out test set.
    Measures end-to-end inference latency (feature extraction + SVM inference).
    """
    print("\n--- Evaluating Approach 1: Traditional CV (SVM) ---")
    model_bundle_path = MODELS_DIR / "traditional_model.joblib"
    if not model_bundle_path.exists():
        raise FileNotFoundError(f"Missing traditional model at: {model_bundle_path}")

    bundle = joblib.load(model_bundle_path)
    scaler = bundle["scaler"]
    classifier = bundle["classifier"]

    y_true = test_df["class_idx"].values
    y_pred = []
    y_prob = []
    latencies_ms = []

    for _, row in test_df.iterrows():
        t0 = time.time()
        # 1. Load image
        img = load_image_rgb(row["file_path"]).resize((256, 256))
        img_np = np.array(img)
        # 2. Extract features
        feats = extract_image_features(img_np).reshape(1, -1)
        # 3. Scale features
        feats_scaled = scaler.transform(feats)
        # 4. Predict
        prob = classifier.predict_proba(feats_scaled)[0]
        pred = int(np.argmax(prob))
        lat = (time.time() - t0) * 1000

        latencies_ms.append(lat)
        y_pred.append(pred)
        y_prob.append(prob)

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_c, rec_c, f1_c, supp_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    per_class_metrics = {}
    for i, c in enumerate(CLASS_NAMES):
        per_class_metrics[c] = {
            "precision": round(float(prec_c[i]), 4),
            "recall": round(float(rec_c[i]), 4),
            "f1_score": round(float(f1_c[i]), 4),
            "support": int(supp_c[i])
        }

    metrics = {
        "model_name": "Traditional CV (HSV+LBP+SVM)",
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "avg_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "train_time_sec": bundle.get("train_time_sec", 0.33),
        "total_training_sec": bundle.get("feature_extraction_time_sec", 64.3) + bundle.get("train_time_sec", 0.33),
        "model_size_mb": round(model_bundle_path.stat().st_size / (1024 * 1024), 2),
        "per_class": per_class_metrics
    }

    # Plot Confusion Matrix
    cm_path = CONFUSION_MATRICES_DIR / "traditional_confusion_matrix.png"
    plot_confusion_matrix_heatmap(cm, CLASS_NAMES, "Traditional CV (SVM)", cm_path)

    return metrics, y_true, y_pred, y_prob


def evaluate_cnn_model(
    test_df: pd.DataFrame
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Deep Learning (MobileNetV2) on the held-out test set.
    Measures per-image inference latency on CPU.
    """
    print("\n--- Evaluating Approach 2: Deep Learning (MobileNetV2) ---")
    model_path = MODELS_DIR / "best_mobilenetv2.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing CNN model checkpoint at: {model_path}")

    device = torch.device("cpu")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE_CNN),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    y_true = test_df["class_idx"].values
    y_pred = []
    y_prob = []
    latencies_ms = []

    # Warmup
    dummy = torch.randn(1, 3, *IMAGE_SIZE_CNN)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = load_image_rgb(row["file_path"])
            t0 = time.time()
            tensor_img = test_transform(img).unsqueeze(0)
            logits = model(tensor_img)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            lat = (time.time() - t0) * 1000

            pred = int(np.argmax(probs))
            latencies_ms.append(lat)
            y_pred.append(pred)
            y_prob.append(probs)

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_c, rec_c, f1_c, supp_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    per_class_metrics = {}
    for i, c in enumerate(CLASS_NAMES):
        per_class_metrics[c] = {
            "precision": round(float(prec_c[i]), 4),
            "recall": round(float(rec_c[i]), 4),
            "f1_score": round(float(f1_c[i]), 4),
            "support": int(supp_c[i])
        }

    # Load training duration from cnn metadata if available
    cnn_meta_path = MODELS_DIR / "cnn_model_metadata.json"
    train_dur = 978.77
    if cnn_meta_path.exists():
        with open(cnn_meta_path, "r") as f:
            meta = json.load(f)
            train_dur = meta.get("total_train_time_sec", 978.77)

    metrics = {
        "model_name": "Deep Learning (MobileNetV2)",
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "avg_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "train_time_sec": round(train_dur, 2),
        "total_training_sec": round(train_dur, 2),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "per_class": per_class_metrics
    }

    # Plot Confusion Matrix
    cm_path = CONFUSION_MATRICES_DIR / "cnn_confusion_matrix.png"
    plot_confusion_matrix_heatmap(cm, CLASS_NAMES, "Deep Learning (MobileNetV2)", cm_path)

    return metrics, y_true, y_pred, y_prob


def run_evaluation():
    """Run full evaluation, compare models, and generate summary tables and charts."""
    seed_everything(RANDOM_SEED)
    CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RUNNING FINAL TEST SET EVALUATION FOR BOTH APPROACHES")
    print("=" * 60)

    test_df = load_test_data()
    print(f"Held-out test set loaded: {len(test_df)} images.")
    for c in CLASS_NAMES:
        print(f"  {c}: {sum(test_df['class_name'] == c)} images")

    # Evaluate Traditional CV
    trad_metrics, y_true, trad_pred, trad_prob = evaluate_traditional_model(test_df)

    # Evaluate CNN MobileNetV2
    cnn_metrics, _, cnn_pred, cnn_prob = evaluate_cnn_model(test_df)

    # 1. Results Summary Table
    comparison_rows = [
        {
            "Metric": "Test Accuracy",
            "Traditional CV (HSV+LBP+SVM)": f"{trad_metrics['accuracy'] * 100:.2f}%",
            "Deep Learning (MobileNetV2)": f"{cnn_metrics['accuracy'] * 100:.2f}%",
            "Difference (CNN - Trad)": f"{(cnn_metrics['accuracy'] - trad_metrics['accuracy']) * 100:+.2f}%"
        },
        {
            "Metric": "Macro Precision",
            "Traditional CV (HSV+LBP+SVM)": f"{trad_metrics['macro_precision']:.4f}",
            "Deep Learning (MobileNetV2)": f"{cnn_metrics['macro_precision']:.4f}",
            "Difference (CNN - Trad)": f"{cnn_metrics['macro_precision'] - trad_metrics['macro_precision']:+.4f}"
        },
        {
            "Metric": "Macro Recall",
            "Traditional CV (HSV+LBP+SVM)": f"{trad_metrics['macro_recall']:.4f}",
            "Deep Learning (MobileNetV2)": f"{cnn_metrics['macro_recall']:.4f}",
            "Difference (CNN - Trad)": f"{cnn_metrics['macro_recall'] - trad_metrics['macro_recall']:+.4f}"
        },
        {
            "Metric": "Macro F1-Score",
            "Traditional CV (HSV+LBP+SVM)": f"{trad_metrics['macro_f1']:.4f}",
            "Deep Learning (MobileNetV2)": f"{cnn_metrics['macro_f1']:.4f}",
            "Difference (CNN - Trad)": f"{cnn_metrics['macro_f1'] - trad_metrics['macro_f1']:+.4f}"
        },
        {
            "Metric": "Weighted F1-Score",
            "Traditional CV (HSV+LBP+SVM)": f"{trad_metrics['weighted_f1']:.4f}",
            "Deep Learning (MobileNetV2)": f"{cnn_metrics['weighted_f1']:.4f}",
            "Difference (CNN - Trad)": f"{cnn_metrics['weighted_f1'] - trad_metrics['weighted_f1']:+.4f}"
        },
        {
            "Metric": "Avg Latency per Image (CPU)",
            "Traditional CV (HSV+LBP+SVM)": f"{trad_metrics['avg_latency_ms']:.2f} ms",
            "Deep Learning (MobileNetV2)": f"{cnn_metrics['avg_latency_ms']:.2f} ms",
            "Difference (CNN - Trad)": f"{cnn_metrics['avg_latency_ms'] - trad_metrics['avg_latency_ms']:+.2f} ms"
        },
        {
            "Metric": "Model File Size",
            "Traditional CV (HSV+LBP+SVM)": f"{trad_metrics['model_size_mb']:.2f} MB",
            "Deep Learning (MobileNetV2)": f"{cnn_metrics['model_size_mb']:.2f} MB",
            "Difference (CNN - Trad)": f"{cnn_metrics['model_size_mb'] - trad_metrics['model_size_mb']:+.2f} MB"
        },
        {
            "Metric": "Total Training Duration",
            "Traditional CV (HSV+LBP+SVM)": f"{trad_metrics['total_training_sec']:.1f} s",
            "Deep Learning (MobileNetV2)": f"{cnn_metrics['total_training_sec']:.1f} s",
            "Difference (CNN - Trad)": f"{cnn_metrics['total_training_sec'] - trad_metrics['total_training_sec']:+.1f} s"
        }
    ]

    comp_df = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY ON HELD-OUT TEST SET (99 IMAGES)")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    results_csv_path = REPORT_DIR / "results_summary.csv"
    comp_df.to_csv(results_csv_path, index=False)
    comp_df.to_csv(COMPARISON_DIR / "results_summary.csv", index=False)

    # 2. Side-by-Side Metrics Bar Chart
    plt.figure(figsize=(11, 6))
    metrics_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    trad_vals = [
        trad_metrics["accuracy"],
        trad_metrics["macro_precision"],
        trad_metrics["macro_recall"],
        trad_metrics["macro_f1"],
        trad_metrics["weighted_f1"]
    ]
    cnn_vals = [
        cnn_metrics["accuracy"],
        cnn_metrics["macro_precision"],
        cnn_metrics["macro_recall"],
        cnn_metrics["macro_f1"],
        cnn_metrics["weighted_f1"]
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.bar(x - width/2, trad_vals, width, label="Traditional CV (HSV+LBP+SVM)", color="#3470a3")
    plt.bar(x + width/2, cnn_vals, width, label="Deep Learning (MobileNetV2)", color="#e6550d")

    for i in range(len(metrics_names)):
        plt.text(x[i] - width/2, trad_vals[i] + 0.015, f"{trad_vals[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + width/2, cnn_vals[i] + 0.015, f"{cnn_vals[i]:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.ylim(0, 1.15)
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Performance Comparison on Held-Out Test Set (N=99)", fontsize=13, fontweight="bold")
    plt.xticks(x, metrics_names, fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    comparison_chart_path = COMPARISON_DIR / "metrics_comparison.png"
    plt.savefig(comparison_chart_path, dpi=300)
    plt.close()

    # 3. Sample Predictions Table
    samples = []
    # Pick representative samples (both correct and any misclassifications)
    for idx, row in test_df.iterrows():
        t_label = CLASS_NAMES[y_true[idx]]
        trad_p = CLASS_NAMES[trad_pred[idx]]
        trad_conf = float(np.max(trad_prob[idx]))
        cnn_p = CLASS_NAMES[cnn_pred[idx]]
        cnn_conf = float(np.max(cnn_prob[idx]))

        samples.append({
            "image_filename": row["filename"],
            "true_class": t_label,
            "traditional_prediction": trad_p,
            "traditional_confidence": round(trad_conf, 4),
            "traditional_correct": t_label == trad_p,
            "cnn_prediction": cnn_p,
            "cnn_confidence": round(cnn_conf, 4),
            "cnn_correct": t_label == cnn_p
        })

    samples_df = pd.DataFrame(samples)
    samples_csv_path = REPORT_DIR / "sample_predictions.csv"
    samples_df.to_csv(samples_csv_path, index=False)

    # 4. Save Raw Metrics JSON
    all_metrics = {
        "test_set_size": len(test_df),
        "traditional_model": trad_metrics,
        "cnn_model": cnn_metrics
    }
    with open(COMPARISON_DIR / "model_evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\nEvaluation completed successfully! All plots and CSV summaries saved.")
    return all_metrics


if __name__ == "__main__":
    run_evaluation()
