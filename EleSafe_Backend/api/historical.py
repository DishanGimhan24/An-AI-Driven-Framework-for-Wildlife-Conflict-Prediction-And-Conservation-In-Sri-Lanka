import os

import pandas as pd
from flask import Blueprint, request

from config import ELEPHANT_DEATHS_DIR
from data_processing.data_loader import data_loader
from utils.response_formatter import success_response, error_response

historical_bp = Blueprint('historical', __name__)


@historical_bp.route('/historical-conflicts', methods=['GET'])
def get_historical_conflicts():
    """
    Get historical conflict data

    Query parameters:
    - start_date: YYYY-MM-DD
    - end_date: YYYY-MM-DD
    - region: optional region filter
    """
    try:
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        region = request.args.get('region')

        # Load tracking data
        if data_loader.tracking_data is None:
            return error_response("Historical data not available", 404)

        df = data_loader.tracking_data.copy()

        # Filter by date if provided
        if start_date and 'Date' in df.columns:
            df = df[df['Date'] >= start_date]

        if end_date and 'Date' in df.columns:
            df = df[df['Date'] <= end_date]

        # Convert to list of dicts
        conflicts = df.to_dict('records')

        return success_response({
            'conflicts': conflicts,
            'count': len(conflicts),
            'start_date': start_date,
            'end_date': end_date
        }, "Historical conflicts retrieved")

    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)


@historical_bp.route('/conflict-stats', methods=['GET'])
def get_conflict_stats():
    """
    Get statistical summary of conflicts
    """
    try:
        if data_loader.tracking_data is None:
            return error_response("Historical data not available", 404)

        df = data_loader.tracking_data

        stats = {
            'total_conflicts': len(df),
            'by_month': {},
            'by_year': {}
        }

        # Calculate stats if date column exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year

            # Monthly distribution
            stats['by_month'] = df['Month'].value_counts().to_dict()

            # Yearly distribution
            stats['by_year'] = df['Year'].value_counts().to_dict()

        return success_response(stats, "Conflict statistics retrieved")

    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)


@historical_bp.route('/elephant-deaths', methods=['GET'])
def get_elephant_deaths():
    """
    Get elephant death records

    Query parameters:
    - year: filter by year (2022-2025)
    """
    try:
        year = request.args.get('year')

        # Look for PDF files in Elephant Deaths folder
        deaths_data = []

        if os.path.exists(ELEPHANT_DEATHS_DIR):
            pdf_files = [f for f in os.listdir(ELEPHANT_DEATHS_DIR) if f.endswith('.pdf')]

            deaths_data = {
                'available_years': [],
                'files': pdf_files,
                'note': 'PDF files need to be parsed for detailed data'
            }

            # Extract years from filenames
            for filename in pdf_files:
                if '2022' in filename:
                    deaths_data['available_years'].append(2022)
                elif '2023' in filename:
                    deaths_data['available_years'].append(2023)
                elif '2024' in filename:
                    deaths_data['available_years'].append(2024)
                elif '2025' in filename:
                    deaths_data['available_years'].append(2025)

        return success_response(deaths_data, "Elephant death records retrieved")

    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)