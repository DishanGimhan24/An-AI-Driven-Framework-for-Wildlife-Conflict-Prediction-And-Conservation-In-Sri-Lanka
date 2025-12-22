from flask import Flask, jsonify
from flask_cors import CORS

from api.forecast import forecast_bp
# Import blueprints
from api.predict import predict_bp
# from api.hotspots import hotspots_bp
from api.historical import historical_bp
from config import FLASK_HOST, FLASK_PORT, DEBUG
from data_processing.data_loader import data_loader
from ml.model_loader import model_loader

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Register blueprints
app.register_blueprint(predict_bp, url_prefix='/api')
# app.register_blueprint(hotspots_bp, url_prefix='/api')
app.register_blueprint(historical_bp, url_prefix='/api')
app.register_blueprint(forecast_bp, url_prefix='/api')


@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'message': 'Wildlife Conflict Prediction API',
        'status': 'running',
        'version': '1.0.0'
    })


@app.route('/api/health')
def health():
    """Detailed health check"""
    return jsonify({
        'api': 'running',
        'model_loaded': model_loader.is_loaded,
        'data_loaded': data_loader.elephant_distribution is not None
    })


@app.route('/api/status')
def status():
    """System status and loaded data info"""
    status_info = {
        'model': {
            'loaded': model_loader.is_loaded,
            'metrics': model_loader.get_metrics() if model_loader.is_loaded else None
        },
        'datasets': {
            'elephant_distribution': data_loader.elephant_distribution is not None,
            'protected_areas': data_loader.protected_areas is not None,
            'power_fences': data_loader.power_fences is not None,
            'roads': data_loader.roads is not None,
            'railways': data_loader.railways is not None,
            'rainfall_data': data_loader.rainfall_data is not None,
            'tracking_data': data_loader.tracking_data is not None
        }
    }

    return jsonify(status_info)


def initialize_system():
    """Initialize data and models on startup"""
    print("\n" + "=" * 50)
    print("Wildlife Conflict Prediction System - Starting")
    print("=" * 50 + "\n")

    # Load datasets
    print("Loading datasets...")
    data_loader.load_all()

    # Load ML models
    print("\nLoading ML models...")
    model_loader.load_models()

    print("\n" + "=" * 50)
    print("System Ready")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    # Initialize system
    initialize_system()

    # Run Flask app
    print(f"Starting Flask server on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)