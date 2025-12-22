import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import ELEPHANT_TRACKING_DIR, ELEPHANT_DEATHS_DIR


def load_positive_samples():
    """
    Load positive samples (conflict locations) from tracking data
    Returns: DataFrame with latitude, longitude, date, conflict=1
    """
    print("Loading positive samples from conflict data...")

    positive_samples = []

    # Load elephant tracking data (conflicts)
    if os.path.exists(ELEPHANT_TRACKING_DIR):
        excel_files = [f for f in os.listdir(ELEPHANT_TRACKING_DIR)
                       if f.endswith(('.xls', '.xlsx'))]

        for excel_file in excel_files:
            try:
                file_path = os.path.join(ELEPHANT_TRACKING_DIR, excel_file)
                df = pd.read_excel(file_path)

                # Print columns to debug
                print(f"  Columns in {excel_file}: {df.columns.tolist()[:5]}...")

                # Try different column name variations
                # X = Longitude, Y = Latitude
                lat_cols = [col for col in df.columns if col.strip().upper() == 'Y' or 'lat' in col.lower()]
                lon_cols = [col for col in df.columns if
                            col.strip().upper() == 'X' or 'lon' in col.lower() or 'lng' in col.lower()]
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

                if lat_cols and lon_cols:
                    lat_col = lat_cols[0]
                    lon_col = lon_cols[0]
                    date_col = date_cols[0] if date_cols else None

                    print(f"  Using: Lat={lat_col}, Lon={lon_col}")

                    # Extract data
                    extracted_count = 0
                    for idx, row in df.iterrows():
                        try:
                            # X is longitude, Y is latitude
                            if lat_col.strip().upper() == 'Y':
                                lat = float(row[lat_col])
                                lon = float(row[lon_col])
                            else:
                                lat = float(row[lat_col])
                                lon = float(row[lon_col])

                            # Skip invalid coordinates
                            if pd.isna(lat) or pd.isna(lon):
                                continue
                            if not (5.9 <= lat <= 9.9 and 79.5 <= lon <= 81.9):
                                continue

                            date = row[date_col] if date_col else pd.Timestamp('2020-01-01')

                            positive_samples.append({
                                'latitude': lat,
                                'longitude': lon,
                                'date': date,
                                'conflict': 1
                            })
                            extracted_count += 1
                        except:
                            continue

                print(f"  ✓ Loaded {excel_file}: {extracted_count} samples")

            except Exception as e:
                print(f"  ⚠ Error loading {excel_file}: {e}")

    if not positive_samples:
        print("⚠ No positive samples found!")
        print("Creating dummy positive samples for demonstration...")
        # Create some dummy samples
        import numpy as np
        np.random.seed(42)
        for _ in range(800):
            positive_samples.append({
                'latitude': np.random.uniform(6.0, 8.5),
                'longitude': np.random.uniform(80.0, 81.5),
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 730)),
                'conflict': 1
            })

    df = pd.DataFrame(positive_samples)
    print(f"\nTotal positive samples: {len(df)}")

    return df


if __name__ == '__main__':
    df = load_positive_samples()
    print(df.head())
    print(f"\nShape: {df.shape}")