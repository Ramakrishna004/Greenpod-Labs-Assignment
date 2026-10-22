# Greenpod Labs — Computer Vision Quality & Condition Classification

End-to-end computer vision pipeline developed for **Greenpod Labs** to detect and classify post-harvest fruit conditions using both **Traditional Computer Vision (Handcrafted Color/Texture Features + SVM)** and **Deep Learning (Transfer Learning with MobileNetV2)**.

> **Dataset Disclosure Note**:
> Following assignment guidelines to *"never invent class names or fabricate data"*, this project classifies the ground-truth classes present in the provided dataset archive (`Orange Dataset.zip`):
> 1. `blackspot` (Citrus Black Spot - *Guignardia citricarpa*)
> 2. `canker` (Citrus Canker - *Xanthomonas axonopodis*)
> 3. `fresh` (Healthy, unblemished citrus fruit)
> 4. `grenning` (Citrus Greening / Huanglongbing - *Candidatus Liberibacter*)

---

## 1. Project Architecture

```text
avocado_ripening_classification/
├── README.md                      # Comprehensive instructions and usage guide
├── requirements.txt               # Pinned Python 3.12 dependencies
├── config.py                      # Global configuration and hyperparameter settings
├── utils.py                       # Safe image loader, dataset scanning, stratification
├── data_analysis.py               # Exploratory data analysis, duplicate/leakage audit
├── train_traditional.py           # Feature extraction (HSV + Moments + LBP) & SVM tuning
├── train_cnn.py                   # PyTorch MobileNetV2 transfer learning pipeline
├── evaluate_models.py             # Head-to-head evaluation on untouched held-out test set
├── predict.py                     # CLI inference script for new images with confidence bars
├── main.py                        # Central pipeline orchestrator
├── tests/
│   └── test_pipeline.py           # Automated unit test suite
├── models/
│   ├── traditional_model.joblib   # Serialized StandardScaler + RBF-SVM bundle (1.34 MB)
│   ├── traditional_model_metadata.json
│   ├── best_mobilenetv2.pt        # PyTorch best validation checkpoint (8.74 MB)
│   └── cnn_model_metadata.json
├── outputs/
│   ├── dataset_analysis/          # Class distribution, sample grid, SSIM leakage audit
│   ├── confusion_matrices/        # Raw & normalized confusion matrix heatmaps
│   ├── training_curves/           # MobileNetV2 training/validation loss & accuracy curves
│   ├── sample_predictions/        # Visual prediction plots with confidence bars
│   └── comparison/                # Metric bar chart and results summary
└── report/
    ├── assignment_report.md       # Comprehensive scientific assignment report
    ├── results_summary.csv        # Side-by-side metric comparison table
    └── sample_predictions.csv     # Per-sample predictions on held-out test set
```

---

## 2. Prerequisites & Installation

- **OS**: Windows 10/11 (or Linux / macOS)
- **Python**: Python 3.12 (64-bit recommended)
- **Hardware**: Runs efficiently on CPU (inference latency: ~93 ms for Traditional CV, ~108 ms for MobileNetV2).

### Setup Environment

```powershell
# From the project root directory:
cd avocado_ripening_classification

# Install dependencies
py -3.12 -m pip install -r requirements.txt
```

---

## 3. Dataset Configuration

By default, the pipeline expects the extracted dataset at:
```text
../data/orange_dataset/
├── train/
│   ├── blackspot/  (184 images)
│   ├── canker/     (179 images)
│   ├── fresh/      (281 images)
│   └── grenning/   (347 images)
└── test/
    ├── blackspot/   (22 images)
    ├── canker/      (22 images)
    ├── fresh/       (33 images)
    └── grenning/    (22 images)
```

To specify a custom dataset location, set the `DATA_DIR` environment variable:
```powershell
$env:DATA_DIR = "C:\path\to\your\dataset"
```

---

## 4. Execution Commands

### Run Complete End-to-End Workflow
To run analysis, train both models, evaluate on held-out test data, and generate all reports:
```powershell
py -3.12 main.py --mode all
```

### Run Individual Stages

1. **Dataset Analysis & Leakage Audit**:
   ```powershell
   py -3.12 main.py --mode analyze
   # Or directly:
   py -3.12 data_analysis.py
   ```
   *Generates class distribution charts, 4x4 sample grid, HSV distribution plots, and a 19-pair SSIM near-duplicate audit.*

2. **Train Approach 1 (Traditional Computer Vision)**:
   ```powershell
   py -3.12 main.py --mode train-traditional
   # Or directly:
   py -3.12 train_traditional.py
   ```
   *Extracts 558-dimensional HSV color histograms, channel moments, and multi-scale LBP texture features. Tunes RBF-SVM with balanced class weights.*

3. **Train Approach 2 (Deep Learning - MobileNetV2)**:
   ```powershell
   py -3.12 main.py --mode train-cnn
   # Or directly:
   py -3.12 train_cnn.py
   ```
   *Trains MobileNetV2 using transfer learning with head warmup and upper block fine-tuning, training-only data augmentation, and early stopping.*

4. **Evaluate Both Models on Held-out Test Set**:
   ```powershell
   py -3.12 main.py --mode evaluate
   # Or directly:
   py -3.12 evaluate_models.py
   ```
   *Evaluates both models on identical 99 test images, outputs confusion matrices, metrics comparison chart, and results CSV.*

5. **Run Predictions on Individual Images**:
   ```powershell
   # Deep Learning inference:
   py -3.12 predict.py --image "../data/orange_dataset/test/fresh/f (1).png" --model cnn

   # Traditional CV inference:
   py -3.12 predict.py --image "../data/orange_dataset/test/blackspot/b (1).jpg" --model traditional
   ```
   *Returns predicted condition class, probabilities for all 4 classes, inference latency, and saves a visualization plot to `outputs/sample_predictions/`.*

6. **Run Unit Tests**:
   ```powershell
   py -3.12 -m unittest discover -s tests
   ```

---

## 5. Measured Test Set Performance Summary

Evaluated on the untouched held-out test set ($N=99$ images):

| Metric | Traditional CV (HSV+LBP+SVM) | Deep Learning (MobileNetV2) | Selected Winner |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **98.99%** | 94.95% | **Traditional CV** (+4.04%) |
| **Macro Precision** | **0.9891** | 0.9475 | **Traditional CV** |
| **Macro Recall** | **0.9886** | 0.9432 | **Traditional CV** |
| **Macro F1-Score** | **0.9886** | 0.9447 | **Traditional CV** |
| **Weighted F1-Score** | **0.9899** | 0.9492 | **Traditional CV** |
| **Inference Latency (CPU)** | **93.38 ms** | 108.63 ms | **Traditional CV** (15 ms faster) |
| **Model Size** | **1.34 MB** | 8.74 MB | **Traditional CV** (6.5x smaller) |
| **Training Duration** | **64.6 s** | 978.8 s | **Traditional CV** (15x faster) |

---

## 6. Troubleshooting

- **NumPy / SciPy ABI Incompatibility**: Ensure `numpy<2.0.0` (specifically `1.26.4`) is installed alongside `scipy 1.13+` and `opencv-python<4.11`.
- **Transparency in PNG images**: The custom `load_image_rgb()` in `utils.py` automatically detects RGBA modes and composites alpha channels against clean white backgrounds.
- **Windows Multiprocessing**: PyTorch `num_workers` is set to `0` in `config.py` to prevent Windows spawn worker deadlocks.

---

## 7. Comprehensive Assignment Report & Mandatory Answers

The complete, scientific assignment report is located at [`report/assignment_report.md`](report/assignment_report.md). It covers:
- **Full Dataset Analysis**: Format distributions, 32 RGBA alpha-compositing, color spaces, and 19-pair SSIM near-duplicate cross-split screening.
- **Architectural Documentation**: Deep-dive into 558-dimensional handcrafted features (HSV + Moments + LBP) and MobileNetV2 transfer learning.
- **Head-to-Head Empirical Benchmarking**: Side-by-side metric tables, confusion matrices, latency measurements, and biological error analysis.
- **Answers to All 4 Mandatory Questions**:
  1. *Production Performance*: Expected real-world accuracy degradation, conveyor belt lighting, and 2-week shadow validation.
  2. *Model Drawbacks*: Subtle lesion boundary overlap (Black Spot vs Canker), whole-image receptive field limits, and 2D surface occlusion.
  3. *Future Improvements*: Instance segmentation (YOLOv8/Mask R-CNN), 360° inspection rigs, hybrid ensemble architectures, and INT8 edge quantization.
  4. *Time Allocation & Technical Challenges*: Complete chronological breakdown of development effort (35% data auditing, 25% environment, 25% modeling, 15% evaluation).
- **Production Readiness & Limitations**: Cautionary declaration, confidence culling thresholds (<80% human review), and deployment roadmap.
