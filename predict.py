"""
Inference and Prediction Pipeline for Orange Condition Classification.
Supports both Traditional CV (HSV+LBP+SVM) and Deep Learning (MobileNetV2).
Outputs predicted class, confidence scores, top probabilities, and optional visualization.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CLASS_NAMES, NUM_CLASSES, CLASS_DESCRIPTIONS,
    IMAGE_SIZE_CNN, IMAGENET_MEAN, IMAGENET_STD,
    MODELS_DIR, SAMPLE_PREDICTIONS_DIR
)
from utils import load_image_rgb
from train_traditional import extract_image_features


def load_traditional_bundle():
    """Load the trained Traditional CV model and scaler bundle."""
    bundle_path = MODELS_DIR / "traditional_model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Traditional model file not found at: {bundle_path}. Please run train_traditional.py first."
        )
    return joblib.load(bundle_path)


def load_cnn_model(device: torch.device):
    """Load the trained MobileNetV2 checkpoint."""
    model_path = MODELS_DIR / "best_mobilenetv2.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model checkpoint not found at: {model_path}. Please run train_cnn.py first."
        )

    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_image(
    image_path: str | Path,
    model_type: str = "cnn",
    save_viz: bool = True,
    output_viz_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run inference on an input image using either 'cnn' or 'traditional' model.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")

    # Validate image readability
    try:
        raw_pil = load_image_rgb(path)
    except Exception as e:
        raise ValueError(f"Failed to load or decode image at '{path}': {str(e)}")

    model_type = model_type.lower()
    t_start = time.time()

    if model_type in ["cnn", "mobilenet", "dl"]:
        device = torch.device("cpu")
        model = load_cnn_model(device)
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_CNN),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        tensor_img = transform(raw_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    elif model_type in ["traditional", "trad", "svm"]:
        bundle = load_traditional_bundle()
        scaler = bundle["scaler"]
        classifier = bundle["classifier"]

        resized_np = np.array(raw_pil.resize((256, 256)))
        feats = extract_image_features(resized_np).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        probs = classifier.predict_proba(feats_scaled)[0]
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'cnn' or 'traditional'.")

    latency_ms = (time.time() - t_start) * 1000

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Top class probabilities
    prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    result = {
        "image_path": str(path.resolve()),
        "filename": path.name,
        "model_used": "MobileNetV2 (CNN)" if "cnn" in model_type else "Traditional CV (SVM)",
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "description": CLASS_DESCRIPTIONS.get(pred_class, ""),
        "probabilities": prob_dict,
        "top_predictions": sorted_probs,
        "latency_ms": round(latency_ms, 2)
    }

    # Save visualization
    if save_viz:
        SAMPLE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        if output_viz_path is None:
            viz_name = f"prediction_{path.stem}_{model_type}.png"
            output_viz_path = SAMPLE_PREDICTIONS_DIR / viz_name

        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1.2]})

        # Image view
        ax_img.imshow(raw_pil)
        ax_img.set_title(f"Input: {path.name}", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        # Probability Bar chart
        classes_plot = [item[0] for item in sorted_probs]
        scores_plot = [item[1] for item in sorted_probs]
        colors = ["#2ca02c" if c == "fresh" else ("#1f77b4" if c == "grenning" else "#d62728") for c in classes_plot]

        bars = ax_bar.barh(classes_plot[::-1], scores_plot[::-1], color=colors[::-1])
        ax_bar.set_xlim(0, 1.05)
        ax_bar.set_xlabel("Probability / Confidence", fontsize=10, fontweight="bold")
        ax_bar.set_title(f"Prediction: {pred_class.upper()} ({confidence*100:.1f}%)", fontsize=12, fontweight="bold")

        for bar in bars:
            width = bar.get_width()
            ax_bar.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%",
                        va="center", ha="left", fontsize=9, fontweight="bold")

        plt.suptitle(f"Model: {result['model_used']} | Latency: {latency_ms:.1f} ms", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_viz_path, dpi=300)
        plt.close()
        result["visualization_path"] = str(output_viz_path.resolve())

    return result


def main():
    parser = argparse.ArgumentParser(description="Classify an orange image condition using saved models.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "traditional"],
                        help="Model to use for inference ('cnn' or 'traditional').")
    parser.add_argument("--no-viz", action="store_true", help="Do not save output prediction visualization.")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for prediction plot.")

    args = parser.parse_args()

    try:
        res = predict_image(
            image_path=args.image,
            model_type=args.model,
            save_viz=not args.no_viz,
            output_viz_path=Path(args.output) if args.output else None
        )

        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"File:            {res['filename']}")
        print(f"Model:           {res['model_used']}")
        print(f"Predicted Class: {res['predicted_class'].upper()}")
        print(f"Confidence:      {res['confidence'] * 100:.2f}%")
        print(f"Description:     {res['description']}")
        print(f"Inference Time:  {res['latency_ms']:.2f} ms")
        print("\nClass Probabilities:")
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        for cls_name, prob in res["top_predictions"]:
            bar = "#" * int(prob * 30)
            print(f"  {cls_name:<10}: {prob*100:6.2f}% {bar}")
        if "visualization_path" in res:
            print(f"\nVisualization saved to: {res['visualization_path']}")
        print("=" * 60)

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
