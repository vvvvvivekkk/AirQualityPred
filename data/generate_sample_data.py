"""
Generate a realistic sample air quality dataset for training.
Creates synthetic data with seasonal patterns, trends, and correlations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_air_quality_data(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    freq: str = "1H",
    output_path: str = None,
) -> pd.DataFrame:
    """
    Generate synthetic air quality data with realistic patterns.

    Args:
        start_date: Start date for the dataset
        end_date: End date for the dataset
        freq: Frequency of observations (1H = hourly)
        output_path: Path to save the CSV file

    Returns:
        DataFrame with air quality and weather data
    """
    np.random.seed(42)

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(date_range)

    # Time-based features for patterns
    hours = np.array([d.hour for d in date_range])
    day_of_year = np.array([d.timetuple().tm_yday for d in date_range])
    month = np.array([d.month for d in date_range])

    # --- Weather features ---
    # Temperature: seasonal + daily cycle + noise
    temp_seasonal = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_daily = 5 * np.sin(2 * np.pi * (hours - 6) / 24)
    temperature = temp_seasonal + temp_daily + np.random.normal(0, 2, n)

    # Humidity: inversely related to temperature + noise
    humidity = 70 - 0.8 * (temperature - 15) + np.random.normal(0, 8, n)
    humidity = np.clip(humidity, 20, 100)

    # Wind speed: slight daily pattern + random
    wind_speed = (
        3
        + 2 * np.sin(2 * np.pi * (hours - 14) / 24)
        + np.random.exponential(1.5, n)
    )
    wind_speed = np.clip(wind_speed, 0.1, 25)

    # --- Pollutant concentrations ---
    # PM2.5: rush hour peaks + seasonal + weather influence
    rush_hour_factor = 1.5 * np.exp(-0.5 * ((hours - 8) / 2) ** 2) + 1.2 * np.exp(
        -0.5 * ((hours - 18) / 2) ** 2
    )
    winter_factor = 1 + 0.5 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    wind_suppression = np.exp(-0.1 * wind_speed)
    humidity_factor = 1 + 0.005 * humidity

    pm25_base = 25 * winter_factor * (1 + rush_hour_factor) * wind_suppression * humidity_factor
    pm25 = pm25_base + np.random.gamma(2, 3, n)
    pm25 = np.clip(pm25, 1, 500)

    # PM10: correlated with PM2.5 but higher
    pm10 = pm25 * np.random.uniform(1.3, 2.5, n) + np.random.normal(5, 3, n)
    pm10 = np.clip(pm10, 2, 600)

    # NO2: traffic-driven
    no2_traffic = 20 * np.exp(-0.5 * ((hours - 8) / 2.5) ** 2) + 15 * np.exp(
        -0.5 * ((hours - 18) / 2.5) ** 2
    )
    no2 = 15 + no2_traffic * wind_suppression + np.random.gamma(2, 2, n)
    no2 = np.clip(no2, 1, 200)

    # CO: correlated with NO2
    co = 0.3 + 0.02 * no2 + 0.01 * pm25 + np.random.exponential(0.1, n)
    co = np.clip(co, 0.1, 10)

    # SO2: industrial + seasonal
    so2_industrial = 5 + 3 * np.sin(2 * np.pi * (hours - 10) / 24)
    so2 = so2_industrial * winter_factor + np.random.gamma(1.5, 2, n)
    so2 = np.clip(so2, 0.5, 100)

    # Build DataFrame
    df = pd.DataFrame(
        {
            "datetime": date_range,
            "PM2.5": np.round(pm25, 2),
            "PM10": np.round(pm10, 2),
            "NO2": np.round(no2, 2),
            "CO": np.round(co, 2),
            "SO2": np.round(so2, 2),
            "temperature": np.round(temperature, 2),
            "humidity": np.round(humidity, 2),
            "wind_speed": np.round(wind_speed, 2),
        }
    )

    # Add some realistic missing values (~1%)
    for col in ["PM2.5", "PM10", "NO2", "CO", "SO2", "temperature", "humidity", "wind_speed"]:
        mask = np.random.random(n) < 0.01
        df.loc[mask, col] = np.nan

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "air_quality.csv")

    df.to_csv(output_path, index=False)
    print(f"[✓] Generated {len(df)} records → {output_path}")
    print(f"    Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print(f"    Missing values:\n{df.isnull().sum().to_string()}")

    return df


if __name__ == "__main__":
    generate_air_quality_data()
