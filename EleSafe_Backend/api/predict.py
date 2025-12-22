from flask import Blueprint, request
from pydantic import ValidationError

from ml.predictor import risk_predictor
from utils.response_formatter import success_response, error_response
from utils.validators import PredictRequest

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
def predict_risk():
    """
    Predict wildlife conflict risk for a specific location and date
    Uses trained Random Forest model (Model 1)

    Request body:
    {
        "latitude": 6.123,
        "longitude": 80.456,
        "date": "2025-12-17"
    }

    Response:
    {
        "status": "success",
        "data": {
            "risk_score": 0.78,
            "risk_level": "HIGH",
            "location": {...},
            "date": "2025-12-17",
            "features": {...}
        }
    }
    """
    try:
        # Validate request
        data = PredictRequest(**request.json)

        # Get prediction from model
        prediction = risk_predictor.predict_risk(
            data.latitude,
            data.longitude,
            data.date
        )

        # Check for errors
        if 'error' in prediction:
            return error_response(prediction['error'], 500)

        return success_response(prediction, "Risk prediction completed")

    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)


@predict_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict risk for multiple locations

    Request body:
    {
        "locations": [
            {"latitude": 6.123, "longitude": 80.456, "date": "2025-12-17"},
            {"latitude": 6.234, "longitude": 80.567, "date": "2025-12-18"}
        ]
    }
    """
    try:
        data = request.json
        locations = data.get('locations', [])

        if not locations:
            return error_response("No locations provided", 400)

        # Get predictions
        predictions = risk_predictor.predict_batch(locations)

        return success_response({
            'predictions': predictions,
            'count': len(predictions)
        }, "Batch prediction completed")

    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)


@predict_bp.route('/model/feature-importance', methods=['GET'])
def get_feature_importance():
    """
    Get feature importance from trained model
    Shows which features are most important for predictions
    """
    try:
        importance = risk_predictor.get_feature_importance()

        if importance is None:
            return error_response("Model not loaded", 404)

        return success_response({
            'feature_importance': importance
        }, "Feature importance retrieved")

    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)