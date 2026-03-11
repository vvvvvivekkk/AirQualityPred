"""Data preprocessing for Air Quality Prediction System."""

import os
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "datetime",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "temperature",
    "humidity",
    "wind_speed",
]

# Use PM25 internally to avoid dot (.) issues in pytorch-forecasting.
CANONICAL_MAP = {
    "PM2.5": "PM25",
    "PM2_5": "PM25",
    "pm2.5": "PM25",
    "pm25": "PM25",
    "PM10": "PM10",
    "NO2": "NO2",
    "CO": "CO",
    "SO2": "SO2",
    "temperature": "temperature",
    "humidity": "humidity",
    "wind_speed": "wind_speed",
    "datetime": "datetime",
}

AQI_BREAKPOINTS = [
    (0, 12.0, "Good"),
    (12.1, 35.4, "Moderate"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, "Unhealthy"),
    (150.5, 250.4, "Very Unhealthy"),
    (250.5, 500.0, "Hazardous"),
]


def get_aqi_category(pm25_value: float) -> str:
    """Return AQI category based on PM2.5 concentration."""
    if pd.isna(pm25_value):
        return "Unknown"
    for low, high, category in AQI_BREAKPOINTS:
        if low <= pm25_value <= high:
            return category
    return "Hazardous" if pm25_value > 500 else "Good"


def _normalize_column_name(name: str) -> str:
    raw = str(name).strip()
    compact = raw.replace(" ", "").replace("-", "_")
    return CANONICAL_MAP.get(raw, CANONICAL_MAP.get(compact, raw))


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and enforce canonical names used across the project."""
    renamed = {_c: _normalize_column_name(_c) for _c in df.columns}
    out = df.rename(columns=renamed)
    return out


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and parse datetime."""
    logger.info("Loading data from %s", filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = clean_column_names(df)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that all required source columns exist after normalization."""
    expected = ["datetime", "PM25", "PM10", "NO2", "CO", "SO2", "temperature", "humidity", "wind_speed"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values with interpolation + ffill/bfill."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Missing values before handling:\n%s", missing)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    remaining = int(df.isnull().sum().sum())
    if remaining:
        logger.warning("Still %d missing values, dropping incomplete rows", remaining)
        df = df.dropna().reset_index(drop=True)
    else:
        logger.info("All missing values handled successfully")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create standard calendar and cyclical time features."""
    logger.info("Creating time-based features")
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add AQI category from PM25."""
    df["aqi_category"] = df["PM25"].apply(get_aqi_category)
    logger.info("AQI distribution:\n%s", df["aqi_category"].value_counts())
    return df


def normalize_features(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """Normalize pollutant/weather numeric columns."""
    if target_cols is None:
        target_cols = ["PM25", "PM10", "NO2", "CO", "SO2", "temperature", "humidity", "wind_speed"]

    existing_cols = [c for c in target_cols if c in df.columns]
    if not existing_cols:
        return df, scaler

    if fit:
        scaler = StandardScaler()
        df[existing_cols] = scaler.fit_transform(df[existing_cols])
        logger.info("Fitted and transformed %d columns", len(existing_cols))
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        df[existing_cols] = scaler.transform(df[existing_cols])
        logger.info("Transformed %d columns with existing scaler", len(existing_cols))

    return df, scaler


def prepare_tft_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data structure for TimeSeriesDataSet."""
    df = df.sort_values("datetime").reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=int)
    df["group_id"] = "station_1"
    return df


def preprocess_pipeline(filepath: str, normalize: bool = True):
    """End-to-end preprocessing pipeline."""
    logger.info("%s", "=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("%s", "=" * 60)

    df = load_data(filepath)
    validate_required_columns(df)
    df = handle_missing_values(df)
    df = add_time_features(df)
    df = add_aqi_category(df)

    scaler = None
    if normalize:
        df, scaler = normalize_features(df, fit=True)

    df = prepare_tft_data(df)

    logger.info("Final dataset shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("PREPROCESSING COMPLETE")

    return df, scaler


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "data", "air_quality.csv")
    frame, _ = preprocess_pipeline(path)
    print(frame.head())
