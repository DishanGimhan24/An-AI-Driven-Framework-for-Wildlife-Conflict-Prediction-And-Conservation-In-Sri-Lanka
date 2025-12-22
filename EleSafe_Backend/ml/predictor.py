import numpy as np

from config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM
from data_processing.feature_extractor import feature_extractor
from ml.model_loader import model_loader
from utils.response_formatter import get_risk_level


class RiskPredictor:
    """Run risk predictions using trained model"""

    def __init__(self):
        self.model_loader = model_loader
        self.feature_extractor = feature_extractor

    def predict_risk(self, latitude, longitude, date_str):
        """
        Predict risk for a given location and date
        Returns: dict with risk_score, risk_level, and features
        """
        try:
            # Extract features
            features_dict = self.feature_extractor.extract_features(latitude, longitude, date_str)

            # Get feature names in correct order
            feature_names = self.feature_extractor.get_feature_names()

            # Create feature array
            feature_array = np.array([[features_dict[name] for name in feature_names]])

            # Scale features if scaler is available
            scaler = self.model_loader.get_scaler()
            if scaler is not None:
                feature_array = scaler.transform(feature_array)

            # Get model
            model = self.model_loader.get_model()
            if model is None:
                return {
                    'error': 'Model not loaded. Train model first.',
                    'risk_score': None,
                    'risk_level': None
                }

            # Predict probability
            risk_score = model.predict_proba(feature_array)[0][1]  # Probability of conflict

            # Get risk level
            risk_level = get_risk_level(
                risk_score,
                RISK_THRESHOLD_HIGH,
                RISK_THRESHOLD_MEDIUM
            )

            # Return results
            return {
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'features': features_dict,
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'date': date_str
            }

        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'risk_score': None,
                'risk_level': None
            }

    def predict_batch(self, locations):
        """
        Predict risk for multiple locations
        locations: list of dicts with latitude, longitude, date
        """
        results = []

        for loc in locations:
            prediction = self.predict_risk(
                loc['latitude'],
                loc['longitude'],
                loc['date']
            )
            results.append(prediction)

        return results

    def get_feature_importance(self):
        """Get feature importance from model"""
        model = self.model_loader.get_model()

        if model is None:
            return None

        feature_names = self.feature_extractor.get_feature_names()
        importances = model.feature_importances_

        # Create sorted list of feature importance
        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {'feature': name, 'importance': float(imp)}
            for name, imp in feature_importance
        ]


# Global predictor instance
risk_predictor = RiskPredictor()