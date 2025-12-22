import json
import os

import joblib

from config import RF_MODEL_PATH, SCALER_PATH, METRICS_PATH


class ModelLoader:
    """Load and manage ML models"""

    def __init__(self):
        self.rf_model = None
        self.scaler = None
        self.metrics = None
        self.is_loaded = False

    def load_models(self):
        """Load Random Forest model and scaler"""
        try:
            # Load Random Forest model
            if os.path.exists(RF_MODEL_PATH):
                self.rf_model = joblib.load(RF_MODEL_PATH)
                print(f"✓ Random Forest model loaded from {RF_MODEL_PATH}")
            else:
                print(f"⚠ Model not found at {RF_MODEL_PATH}")
                print("  Run ml_training/train_model.py first to train the model")
                return False

            # Load scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                print(f"✓ Scaler loaded from {SCALER_PATH}")
            else:
                print(f"⚠ Scaler not found, predictions may be inaccurate")

            # Load metrics
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, 'r') as f:
                    self.metrics = json.load(f)
                print(f"✓ Model metrics loaded")

            self.is_loaded = True
            return True

        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False

    def get_model(self):
        """Get the loaded Random Forest model"""
        if not self.is_loaded:
            self.load_models()
        return self.rf_model

    def get_scaler(self):
        """Get the loaded scaler"""
        if not self.is_loaded:
            self.load_models()
        return self.scaler

    def get_metrics(self):
        """Get model performance metrics"""
        if not self.is_loaded:
            self.load_models()
        return self.metrics


# Global model loader instance
model_loader = ModelLoader()