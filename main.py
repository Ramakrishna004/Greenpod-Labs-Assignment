"""
Main Entry Point for the Orange Condition Classification Pipeline.
Supports end-to-end execution: analysis, traditional training, CNN training, evaluation, and prediction.

Usage examples:
    python main.py --mode all
    python main.py --mode analyze
    python main.py --mode train-traditional
    python main.py --mode train-cnn
    python main.py --mode evaluate
    python main.py --mode predict --image "../data/orange_dataset/test/fresh/f (1).png" --model cnn
"""
import argparse
import sys
from pathlib import Path

from data_analysis import analyze_dataset
from train_traditional import train_traditional_pipeline
from train_cnn import train_cnn_pipeline
from evaluate_models import run_evaluation
from predict import predict_image


def main():
    parser = argparse.ArgumentParser(description="End-to-End Orange Condition Classification Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "analyze", "train-traditional", "train-cnn", "evaluate", "predict"],
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument("--image", type=str, default=None, help="Image path for prediction mode.")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "traditional"],
                        help="Model to use for prediction (default: cnn)")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GREENPOD LABS - COMPUTER VISION CLASSIFICATION PIPELINE")
    print(f"Target Mode: {args.mode}")
    print("=" * 70)

    if args.mode == "analyze":
        analyze_dataset()
    elif args.mode == "train-traditional":
        train_traditional_pipeline()
    elif args.mode == "train-cnn":
        train_cnn_pipeline()
    elif args.mode == "evaluate":
        run_evaluation()
    elif args.mode == "predict":
        if not args.image:
            print("Error: --image argument is required when mode is 'predict'.", file=sys.stderr)
            sys.exit(1)
        res = predict_image(args.image, model_type=args.model)
        print(f"\nPrediction for {res['filename']}: {res['predicted_class'].upper()} ({res['confidence']*100:.2f}%)")
        print(f"Model: {res['model_used']} | Latency: {res['latency_ms']:.2f} ms")
    elif args.mode == "all":
        print("\n[Step 1/4] Running Comprehensive Dataset Analysis...")
        analyze_dataset()

        print("\n[Step 2/4] Training Approach 1 (Traditional CV - HSV+LBP+SVM)...")
        train_traditional_pipeline()

        print("\n[Step 3/4] Training Approach 2 (Deep Learning - MobileNetV2)...")
        train_cnn_pipeline()

        print("\n[Step 4/4] Evaluating Both Models on Held-out Test Set...")
        run_evaluation()

        print("\nPipeline execution complete! All artifacts and reports are ready.")


if __name__ == "__main__":
    main()
