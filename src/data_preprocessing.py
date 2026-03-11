"""
Data preprocessing module for Air Quality Prediction System.
Handles loading, cleaning, feature engineering, and preparing data
for the Temporal Fusion Transformer model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# AQI breakpoints for PM2.5 (µg/m³)
AQI_BREAKPOINTS = [
    (0, 12.0, "Good"),
    (12.1, 35.4, "Moderate"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, "Unhealthy"),
    (150.5, 250.4, "Very Unhealthy"),
    (250.5, 500.0, "Hazardous"),
]


def get_aqi_category(pm25_value: float) -> str:
    """Return AQI category string based on PM2.5 concentration."""
    if pd.isna(pm25_value):
        return "Unknown"
    for low, high, category in AQI_BREAKPOINTS:
        if low <= pm25_value <= high:
            return category
    return "Hazardous" if pm25_value > 500 else "Good"


def load_data(filepath: str) -> pd.DataFrame:
    """Load csv and parse datetime column."""
    logger.info(f"Loading data from {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["datetime"])
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using forward-fill, back-fill, then interpolation."""
    logger.info(f"Missing values before handling:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Step 1: Linear interpolation for numeric columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # Step 2: Forward fill then backward fill for any remaining
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    remaining = df.isnull().sum().sum()
    if remaining > 0:
        logger.warning(f"Still {remaining} missing values after handling")
        df = df.dropna()
    else:
        logger.info("All missing values handled successfully")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from the datetime column."""
    logger.info("Creating time-based features")
    dt = df["datetime"]

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for hour and month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add AQI category column based on PM2.5."""
    df["aqi_category"] = df["PM2.5"].apply(get_aqi_category)
    logger.info(f"AQI distribution:\n{df['aqi_category'].value_counts()}")
    return df


def normalize_features(df: pd.DataFrame, target_cols: list = None, fit: bool = True, scaler: StandardScaler = None):
    """
    Normalize numeric features using StandardScaler.

    Args:
        df: Input DataFrame
        target_cols: Columns to normalize (if None, all numeric except time indices)
        fit: Whether to fit the scaler or just transform
        scaler: Pre-fitted scaler (used during inference)

    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    if target_cols is None:
        target_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "temperature", "humidity", "wind_speed"]

    existing_cols = [c for c in target_cols if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[existing_cols] = scaler.fit_transform(df[existing_cols])
        logger.info(f"Fitted and transformed {len(existing_cols)} columns")
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        df[existing_cols] = scaler.transform(df[existing_cols])
        logger.info(f"Transformed {len(existing_cols)} columns with pre-fitted scaler")

    return df, scaler


def create_sequences(df: pd.DataFrame, input_length: int = 168, forecast_horizon: int = 24):
    """
    Create time-series sequences for the TFT model.

    Args:
        df: Preprocessed DataFrame
        input_length: Number of past time steps (168 = 7 days of hourly data)
        forecast_horizon: Number of future steps to predict (24 = 1 day)

    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    feature_cols = [
        "PM2.5", "PM10", "NO2", "CO", "SO2",
        "temperature", "humidity", "wind_speed",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dow_sin", "dow_cos", "is_weekend",
    ]

    existing_features = [c for c in feature_cols if c in df.columns]
    target_col = "PM2.5"

    data = df[existing_features].values
    targets = df[target_col].values

    X, y = [], []
    total = len(data) - input_length - forecast_horizon + 1

    for i in range(total):
        X.append(data[i : i + input_length])
        y.append(targets[i + input_length : i + input_length + forecast_horizon])

    logger.info(f"Created {len(X)} sequences (input={input_length}, horizon={forecast_horizon})")
    return np.array(X), np.array(y)


def prepare_tft_data(df: pd.DataFrame):
    """
    Prepare data in the format required by pytorch-forecasting TimeSeriesDataSet.
    Adds a time_idx and group_id column.
    """
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["group_id"] = "station_1"  # Single station for this demo

    return df


def preprocess_pipeline(filepath: str, normalize: bool = True):
    """
    Full preprocessing pipeline.

    Args:
        filepath: Path to the raw CSV file
        normalize: Whether to normalize features

    Returns:
        Tuple of (processed DataFrame, scaler or None)
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    df = load_data(filepath)
    df = handle_missing_values(df)
    df = add_time_features(df)
    df = add_aqi_category(df)

    scaler = None
    if normalize:
        df, scaler = normalize_features(df)

    df = prepare_tft_data(df)

    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info("PREPROCESSING COMPLETE")

    return df, scaler


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "air_quality.csv")

    df, scaler = preprocess_pipeline(data_path)
    print(f"\nProcessed data shape: {df.shape}")
    print(f"\nSample:\n{df.head()}")
    print(f"\nAQI categories:\n{df['aqi_category'].value_counts()}")
