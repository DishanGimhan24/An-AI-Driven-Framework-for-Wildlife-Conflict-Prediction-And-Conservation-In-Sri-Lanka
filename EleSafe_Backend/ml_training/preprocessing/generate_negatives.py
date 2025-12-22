import numpy as np
import pandas as pd
import random


def generate_negative_samples(n_samples=1000, seed=42):
    """
    Generate negative samples (safe locations with no conflicts)
    Returns: DataFrame with latitude, longitude, date
    """
    random.seed(seed)
    np.random.seed(seed)

    # Sri Lanka bounds
    lat_min, lat_max = 5.9, 9.9
    lon_min, lon_max = 79.5, 81.9

    # Generate random locations
    latitudes = np.random.uniform(lat_min, lat_max, n_samples)
    longitudes = np.random.uniform(lon_min, lon_max, n_samples)

    # Generate random dates (2020-2025)
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2025-12-31')
    date_range = (end_date - start_date).days

    random_days = np.random.randint(0, date_range, n_samples)
    dates = [start_date + pd.Timedelta(days=int(d)) for d in random_days]

    # Create dataframe
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'date': dates,
        'conflict': 0  # No conflict
    })

    print(f"Generated {n_samples} negative samples")
    return df


if __name__ == '__main__':
    df = generate_negative_samples(1000)
    print(df.head())