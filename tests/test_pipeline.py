"""
Automated Unit Tests for Orange Condition Classification Pipeline.
Verifies config integrity, image loading, feature extraction dimensions, model loading, and predictions.
"""
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CLASS_NAMES, NUM_CLASSES, MODELS_DIR, DATA_DIR,
    IMAGE_SIZE_CNN
)
from utils import load_image_rgb, seed_everything
from train_traditional import extract_image_features
from predict import predict_image


class TestClassificationPipeline(unittest.TestCase):

    def test_01_config_and_classes(self):
        """Test configuration invariants."""
        self.assertEqual(len(CLASS_NAMES), 4)
        self.assertEqual(CLASS_NAMES, ["blackspot", "canker", "fresh", "grenning"])
        self.assertEqual(NUM_CLASSES, 4)

    def test_02_image_loader_rgba_conversion(self):
        """Test load_image_rgb properly handles synthetic RGBA images."""
        rgba_img = Image.new("RGBA", (100, 100), (255, 128, 0, 200))
        temp_path = Path("temp_test_rgba.png")
        rgba_img.save(temp_path)

        try:
            loaded_rgb = load_image_rgb(temp_path)
            self.assertEqual(loaded_rgb.mode, "RGB")
            self.assertEqual(loaded_rgb.size, (100, 100))
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_03_feature_extraction_dimensions(self):
        """Verify handcrafted CV feature extraction vector dimension (558 dims)."""
        dummy_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        feats = extract_image_features(dummy_rgb)
        self.assertEqual(feats.shape, (558,))
        self.assertFalse(np.isnan(feats).any())

    def test_04_traditional_model_persistence(self):
        """Verify traditional model bundle exists, loads, and predicts."""
        model_path = MODELS_DIR / "traditional_model.joblib"
        self.assertTrue(model_path.exists(), f"Model missing: {model_path}")

        bundle = joblib.load(model_path)
        self.assertIn("classifier", bundle)
        self.assertIn("scaler", bundle)
        self.assertEqual(bundle["feature_dim"], 558)

    def test_05_cnn_model_persistence(self):
        """Verify MobileNetV2 checkpoint exists and loads."""
        model_path = MODELS_DIR / "best_mobilenetv2.pt"
        self.assertTrue(model_path.exists(), f"Checkpoint missing: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint["num_classes"], 4)

    def test_06_predict_pipeline_execution(self):
        """Test predict_image end-to-end with held-out test sample."""
        sample_path = DATA_DIR / "test" / "fresh" / "f (1).png"
        self.assertTrue(sample_path.exists(), f"Sample image missing: {sample_path}")

        # Test CNN prediction
        res_cnn = predict_image(sample_path, model_type="cnn", save_viz=False)
        self.assertIn(res_cnn["predicted_class"], CLASS_NAMES)
        self.assertGreaterEqual(res_cnn["confidence"], 0.0)
        self.assertLessEqual(res_cnn["confidence"], 1.0)

        # Test Traditional prediction
        res_trad = predict_image(sample_path, model_type="traditional", save_viz=False)
        self.assertIn(res_trad["predicted_class"], CLASS_NAMES)
        self.assertGreaterEqual(res_trad["confidence"], 0.0)
        self.assertLessEqual(res_trad["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
