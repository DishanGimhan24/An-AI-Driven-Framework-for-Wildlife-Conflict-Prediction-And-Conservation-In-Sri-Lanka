import geopandas as gpd
import pandas as pd
import os
from config import (
    ELEPHANT_DISTRIBUTION_DIR,
    PROTECTED_AREAS_DIR,
    POWER_FENCE_DIR,
    ROAD_RAILWAY_DIR,
    POPULATION_DIR,
    RAINFALL_DIR,
    NDVI_DIR,
    ELEPHANT_TRACKING_DIR
)


class DataLoader:
    """Load and cache all datasets"""

    def __init__(self):
        self.elephant_distribution = None
        self.protected_areas = None
        self.power_fences = None
        self.roads = None
        self.railways = None
        self.population_raster = None
        self.rainfall_data = None
        self.ndvi_data = None
        self.tracking_data = None

    def load_all(self):
        """Load all required datasets"""
        print("Loading datasets...")

        # Load shapefiles
        self.load_elephant_distribution()
        self.load_protected_areas()
        self.load_power_fences()
        self.load_roads_railways()

        # Load CSV data
        self.load_rainfall_data()
        self.load_ndvi_data()
        self.load_tracking_data()

        print("All datasets loaded successfully")

    def load_elephant_distribution(self):
        """Load elephant distribution shapefile"""
        shapefile_path = os.path.join(ELEPHANT_DISTRIBUTION_DIR, 'Survey-ElephantDistribution.shp')
        if os.path.exists(shapefile_path):
            self.elephant_distribution = gpd.read_file(shapefile_path)
            print(f"✓ Elephant distribution loaded: {len(self.elephant_distribution)} features")
        else:
            print(f"⚠ Elephant distribution shapefile not found")

    def load_protected_areas(self):
        """Load protected areas shapefiles"""
        # Try to find .shp files directly or extract from zip
        shp_files = []

        # Check for direct .shp files
        for root, dirs, files in os.walk(PROTECTED_AREAS_DIR):
            for file in files:
                if file.endswith('.shp'):
                    shp_files.append(os.path.join(root, file))

        # If no .shp found, try to extract from .zip files
        if not shp_files:
            import zipfile
            zip_files = [f for f in os.listdir(PROTECTED_AREAS_DIR) if f.endswith('.zip')]

            for zip_file in zip_files:
                zip_path = os.path.join(PROTECTED_AREAS_DIR, zip_file)
                extract_path = os.path.join(PROTECTED_AREAS_DIR, 'extracted')
                os.makedirs(extract_path, exist_ok=True)

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)

                    # Find .shp files in extracted folder
                    for root, dirs, files in os.walk(extract_path):
                        for file in files:
                            if file.endswith('.shp'):
                                shp_files.append(os.path.join(root, file))
                except Exception as e:
                    print(f"⚠ Error extracting {zip_file}: {e}")

        # Load first shapefile found
        if shp_files:
            try:
                self.protected_areas = gpd.read_file(shp_files[0])
                print(f"✓ Protected areas loaded: {len(self.protected_areas)} features")
            except Exception as e:
                print(f"⚠ Error loading protected areas: {e}")
        else:
            print(f"⚠ Protected areas shapefile not found")

    def load_power_fences(self):
        """Load power fence data from all provinces"""
        fence_gdfs = []

        # Load from each province folder
        for province in os.listdir(POWER_FENCE_DIR):
            province_path = os.path.join(POWER_FENCE_DIR, province)
            if os.path.isdir(province_path):
                shp_files = [f for f in os.listdir(province_path) if f.endswith('.shp')]
                for shp_file in shp_files:
                    try:
                        gdf = gpd.read_file(os.path.join(province_path, shp_file))
                        # Set CRS if missing, assume Sri Lanka local CRS then convert to WGS84
                        if gdf.crs is None:
                            gdf.set_crs("EPSG:4326", inplace=True)
                        else:
                            gdf = gdf.to_crs("EPSG:4326")
                        fence_gdfs.append(gdf)
                    except Exception as e:
                        print(f"⚠ Error loading {shp_file}: {e}")

        if fence_gdfs:
            self.power_fences = gpd.GeoDataFrame(pd.concat(fence_gdfs, ignore_index=True), crs="EPSG:4326")
            print(f"✓ Power fences loaded: {len(self.power_fences)} features")
        else:
            print(f"⚠ Power fence data not found")

    def load_roads_railways(self):
        """Load roads and railways shapefiles"""
        import os

        # Set environment variable to restore missing .shx files
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'

        # Load roads
        roads_path = os.path.join(ROAD_RAILWAY_DIR, 'gis_osm_roads_free_1.shp')
        if os.path.exists(roads_path):
            try:
                self.roads = gpd.read_file(roads_path)
                print(f"✓ Roads loaded: {len(self.roads)} features")
            except Exception as e:
                print(f"⚠ Error loading roads: {e}")
        else:
            print(f"⚠ Roads shapefile not found at {roads_path}")

        # Load railways
        railways_path = os.path.join(ROAD_RAILWAY_DIR, 'gis_osm_railways_free_1.shp')
        if os.path.exists(railways_path):
            try:
                self.railways = gpd.read_file(railways_path)
                print(f"✓ Railways loaded: {len(self.railways)} features")
            except Exception as e:
                print(f"⚠ Error loading railways: {e}")
        else:
            print(f"⚠ Railways shapefile not found at {railways_path}")

    def load_rainfall_data(self):
        """Load rainfall CSV data"""
        csv_files = [f for f in os.listdir(RAINFALL_DIR) if f.endswith('.csv')]

        if csv_files:
            # Load all years and combine
            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(os.path.join(RAINFALL_DIR, csv_file), on_bad_lines='skip')
                    dfs.append(df)
                    print(f"  ✓ Loaded {csv_file}")
                except Exception as e:
                    print(f"  ⚠ Error loading {csv_file}: {e}")

            if dfs:
                self.rainfall_data = pd.concat(dfs, ignore_index=True)
                print(f"✓ Rainfall data loaded: {len(self.rainfall_data)} records")
            else:
                print(f"⚠ No rainfall data loaded")
        else:
            print(f"⚠ Rainfall data not found")

    def load_ndvi_data(self):
        """Load NDVI data (implementation depends on format)"""
        # Placeholder - implement based on actual NDVI data format
        print("⚠ NDVI data loader not implemented yet")

    def load_tracking_data(self):
        """Load elephant tracking data"""
        excel_files = [f for f in os.listdir(ELEPHANT_TRACKING_DIR) if f.endswith(('.xls', '.xlsx'))]

        if excel_files:
            dfs = []
            for excel_file in excel_files:
                try:
                    df = pd.read_excel(os.path.join(ELEPHANT_TRACKING_DIR, excel_file))
                    dfs.append(df)
                except Exception as e:
                    print(f"⚠ Error loading {excel_file}: {e}")

            if dfs:
                self.tracking_data = pd.concat(dfs, ignore_index=True)
                print(f"✓ Tracking data loaded: {len(self.tracking_data)} records")
        else:
            print(f"⚠ Tracking data not found")


# Global data loader instance
data_loader = DataLoader()