import os
from datetime import datetime

from config import POPULATION_DIR
from data_processing.data_loader import data_loader
from data_processing.spatial_utils import (
    calculate_distance_to_nearest,
    point_in_polygon,
    calculate_buffer_zone,
    get_density_at_point
)


class FeatureExtractor:
    """Extract features for ML model from location and date"""

    def __init__(self):
        self.data_loader = data_loader

    def extract_features(self, latitude, longitude, date_str):
        """
        Extract all features for a given location and date
        Returns: dict of features
        """
        features = {}

        # Basic features
        features['latitude'] = latitude
        features['longitude'] = longitude

        # Date features
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        features['month'] = date_obj.month
        features['day_of_year'] = date_obj.timetuple().tm_yday
        features['season'] = self._get_season(date_obj.month)

        # Spatial features
        features.update(self._extract_spatial_features(latitude, longitude))

        # Environmental features
        features.update(self._extract_environmental_features(latitude, longitude, date_obj))

        return features

    def _get_season(self, month):
        """
        Get season for Sri Lanka
        Dry season: May-September (5-9)
        Wet season: October-April (10-4)
        """
        if 5 <= month <= 9:
            return 1  # Dry
        else:
            return 0  # Wet

    def _extract_spatial_features(self, lat, lon):
        """Extract spatial features from shapefiles"""
        features = {}

        # Distance to protected areas
        if self.data_loader.protected_areas is not None:
            features['dist_to_protected_area_km'] = calculate_distance_to_nearest(
                lat, lon, self.data_loader.protected_areas
            )
            features['in_protected_area'] = point_in_polygon(
                lat, lon, self.data_loader.protected_areas
            )
            features['buffer_zone'] = calculate_buffer_zone(
                lat, lon, self.data_loader.protected_areas, buffer_km=5
            )
        else:
            features['dist_to_protected_area_km'] = 0
            features['in_protected_area'] = 0
            features['buffer_zone'] = 0

        # Distance to roads
        if self.data_loader.roads is not None:
            features['dist_to_road_km'] = calculate_distance_to_nearest(
                lat, lon, self.data_loader.roads
            )
            features['near_road'] = 1 if features['dist_to_road_km'] < 2 else 0
        else:
            features['dist_to_road_km'] = 0
            features['near_road'] = 0

        # Distance to railways
        if self.data_loader.railways is not None:
            features['dist_to_railway_km'] = calculate_distance_to_nearest(
                lat, lon, self.data_loader.railways
            )
            features['near_railway'] = 1 if features['dist_to_railway_km'] < 2 else 0
        else:
            features['dist_to_railway_km'] = 0
            features['near_railway'] = 0

        # Distance to electric fences
        if self.data_loader.power_fences is not None:
            features['dist_to_fence_km'] = calculate_distance_to_nearest(
                lat, lon, self.data_loader.power_fences
            )
        else:
            features['dist_to_fence_km'] = 0

        # Elephant distribution zone
        if self.data_loader.elephant_distribution is not None:
            features['in_elephant_zone'] = point_in_polygon(
                lat, lon, self.data_loader.elephant_distribution
            )
        else:
            features['in_elephant_zone'] = 0

        # Population density
        population_raster = os.path.join(POPULATION_DIR, 'lka_pd_2020_1km.tif')
        if os.path.exists(population_raster):
            features['population_density'] = get_density_at_point(lat, lon, population_raster)
            features['high_human_presence'] = 1 if features['population_density'] > 100 else 0
        else:
            features['population_density'] = 0
            features['high_human_presence'] = 0

        return features

    def _extract_environmental_features(self, lat, lon, date_obj):
        """Extract environmental features (rainfall, NDVI)"""
        features = {}

        # Rainfall features
        if self.data_loader.rainfall_data is not None:
            rainfall_info = self._get_rainfall_for_location_date(lat, lon, date_obj)
            features.update(rainfall_info)
        else:
            features['rainfall_7day'] = 0
            features['rainfall_30day'] = 0
            features['is_dry_period'] = 0

        # NDVI features (placeholder - implement based on actual data format)
        features['ndvi_current'] = 0.4  # Placeholder
        features['ndvi_previous_month'] = 0.38  # Placeholder
        features['vegetation_decreasing'] = 1 if features['ndvi_current'] < features['ndvi_previous_month'] else 0

        return features

    def _get_rainfall_for_location_date(self, lat, lon, date_obj):
        """Get rainfall data for location and date"""
        # This is simplified - actual implementation needs spatial interpolation
        # For now, use nearest grid point or average

        df = self.data_loader.rainfall_data

        # Filter by approximate date
        date_str = date_obj.strftime('%Y-%m-%d')

        # Calculate 7-day and 30-day rainfall
        # Placeholder implementation
        rainfall_7day = 12.5  # mm
        rainfall_30day = 45.2  # mm

        return {
            'rainfall_7day': rainfall_7day,
            'rainfall_30day': rainfall_30day,
            'is_dry_period': 1 if rainfall_7day < 5 else 0
        }

    def get_feature_names(self):
        """Return list of all feature names in order"""
        return [
            'latitude', 'longitude', 'month', 'day_of_year', 'season',
            'dist_to_protected_area_km', 'in_protected_area', 'buffer_zone',
            'dist_to_road_km', 'near_road',
            'dist_to_railway_km', 'near_railway',
            'dist_to_fence_km',
            'in_elephant_zone',
            'population_density', 'high_human_presence',
            'rainfall_7day', 'rainfall_30day', 'is_dry_period',
            'ndvi_current', 'ndvi_previous_month', 'vegetation_decreasing'
        ]


# Global feature extractor instance
feature_extractor = FeatureExtractor()