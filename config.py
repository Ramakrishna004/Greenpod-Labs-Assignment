"""
Configuration settings for the Orange Condition Classification Pipeline.
Note: Dataset contains 4 classes: blackspot, canker, fresh, grenning.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Dataset paths
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "data" / "orange_dataset"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Project Artifact Directories
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_ANALYSIS_DIR = OUTPUTS_DIR / "dataset_analysis"
CONFUSION_MATRICES_DIR = OUTPUTS_DIR / "confusion_matrices"
TRAINING_CURVES_DIR = OUTPUTS_DIR / "training_curves"
SAMPLE_PREDICTIONS_DIR = OUTPUTS_DIR / "sample_predictions"
COMPARISON_DIR = OUTPUTS_DIR / "comparison"
REPORT_DIR = PROJECT_ROOT / "report"

# Class Configuration (alphabetically sorted matching ImageFolder convention)
CLASS_NAMES = ["blackspot", "canker", "fresh", "grenning"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

# Class Descriptions / Botanical context
CLASS_DESCRIPTIONS = {
    "blackspot": "Citrus Black Spot (Guignardia citricarpa) - fungal lesions on peel",
    "canker": "Citrus Canker (Xanthomonas axonopodis) - necrotic corky lesions with oily margins",
    "fresh": "Healthy / Fresh Orange Fruit - intact, unblemished peel",
    "grenning": "Citrus Greening / Huanglongbing (HLB) - asymmetric mottling, green rind retention"
}

# Reproducibility Seed
RANDOM_SEED = 42

# Data Splitting
# From original train folder: 85% train, 15% validation stratified
VAL_SPLIT_RATIO = 0.15

# Traditional CV Feature Extraction Parameters
IMAGE_SIZE_TRADITIONAL = (256, 256)
HIST_BINS_HSV = (8, 8, 8)  # 512-dim color histogram
LBP_RADIUS = 2
LBP_POINTS = 16
LBP_METHOD = "uniform"

# CNN Model Parameters
IMAGE_SIZE_CNN = (224, 224)  # MobileNetV2 native input resolution
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 is safest on Windows multiprocessing
MAX_EPOCHS = 15
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

# Normalization parameters (ImageNet standards)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
