from datetime import datetime, timedelta

from flask import Blueprint, request
from pydantic import ValidationError

from utils.response_formatter import success_response, error_response
from utils.validators import ForecastRequest

forecast_bp = Blueprint('forecast', __name__)


@forecast_bp.route('/forecast', methods=['POST'])
def forecast_risk():
    """
    Forecast risk for the next N days (7-30 days)

    Request body:
    {
        "latitude": 6.123,
        "longitude": 80.456,
        "forecast_days": 7
    }

    Note: This endpoint requires LSTM model (Model 2)
    Currently returns placeholder data
    """
    try:
        # Validate request
        data = ForecastRequest(**request.json)

        # TODO: Implement LSTM model prediction
        # For now, return placeholder forecast

        today = datetime.now()
        forecast_data = []

        for i in range(data.forecast_days):
            forecast_date = today + timedelta(days=i + 1)

            # Placeholder: decreasing risk over time
            risk_score = max(0.3, 0.75 - (i * 0.05))

            forecast_data.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'day': i + 1,
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score >= 0.7 else 'MEDIUM' if risk_score >= 0.4 else 'LOW'
            })

        return success_response({
            'location': {
                'latitude': data.latitude,
                'longitude': data.longitude
            },
            'forecast': forecast_data,
            'forecast_days': data.forecast_days,
            'note': 'LSTM model not implemented yet - showing placeholder data'
        }, "Forecast generated")

    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)