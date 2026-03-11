"""Prediction service for air quality forecasting."""

import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import get_aqi_category, preprocess_pipeline

AQI_COLORS = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
}


class AirQualityPredictor:
    """Loads trained TFT model and serves predictions."""

    def __init__(self, checkpoint_path: Optional[str] = None, data_path: Optional[str] = None):
        self.model = None
        self.data = None
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path or os.path.join(PROJECT_ROOT, "data", "air_quality.csv")
        self._load_model()
        self._load_data()

    def _discover_checkpoint(self) -> Optional[str]:
        model_dir = os.path.join(PROJECT_ROOT, "models")
        if not os.path.isdir(model_dir):
            return None

        ckpts = sorted(
            [f for f in os.listdir(model_dir) if f.endswith(".ckpt")],
            key=lambda x: os.path.getmtime(os.path.join(model_dir, x)),
            reverse=True,
        )
        return os.path.join(model_dir, ckpts[0]) if ckpts else None

    def _load_model(self) -> None:
        if self.checkpoint_path is None:
            self.checkpoint_path = self._discover_checkpoint()
        if not self.checkpoint_path:
            logger.warning("No checkpoint found. Using statistical fallback predictions.")
            return

        try:
            from pytorch_forecasting import TemporalFusionTransformer

            self.model = TemporalFusionTransformer.load_from_checkpoint(self.checkpoint_path)
            self.model.eval()
            logger.info("Model loaded from %s", self.checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to load model checkpoint (%s). Fallback mode enabled.", exc)
            self.model = None

    def _load_data(self) -> None:
        try:
            # Keep original value scale for fallback/historical visualization.
            self.data, _ = preprocess_pipeline(self.data_path, normalize=False)
            logger.info("Prediction context loaded: %d rows", len(self.data))
        except Exception as exc:
            logger.error("Failed to load data context: %s", exc)
            self.data = None

    def _record(self, ts: datetime, step: int, pm25: float, pm10: float, no2: float, co: float, so2: float) -> Dict:
        category = get_aqi_category(pm25)
        return {
            "datetime": ts.isoformat(),
            "hour": step,
            "PM2.5": round(float(max(pm25, 0.0)), 2),
            "PM10": round(float(max(pm10, 0.0)), 2),
            "NO2": round(float(max(no2, 0.0)), 2),
            "CO": round(float(max(co, 0.0)), 2),
            "SO2": round(float(max(so2, 0.0)), 2),
            "aqi_category": category,
            "aqi_color": AQI_COLORS.get(category, "#808080"),
        }

    def _synthetic_prediction(self, hours: int, base_datetime: Optional[datetime] = None) -> List[Dict]:
        base = base_datetime or datetime.now()
        out = []
        for step in range(1, hours + 1):
            ts = base + timedelta(hours=step)
            hour = ts.hour

            morning_peak = 1.1 * np.exp(-0.5 * ((hour - 8) / 2.2) ** 2)
            evening_peak = 1.3 * np.exp(-0.5 * ((hour - 18) / 2.5) ** 2)
            peak = morning_peak + evening_peak

            pm25 = 24 + 16 * peak + np.random.normal(0, 2.0)
            pm10 = pm25 * 1.7 + np.random.normal(0, 3.0)
            no2 = 22 + 10 * peak + np.random.normal(0, 2.0)
            co = 0.6 + 0.015 * no2 + np.random.normal(0, 0.03)
            so2 = 7 + np.random.normal(0, 1.3)

            out.append(self._record(ts, step, pm25, pm10, no2, co, so2))
        return out

    def _fallback_prediction(self, hours: int, base_datetime: Optional[datetime] = None) -> List[Dict]:
        if self.data is None or self.data.empty:
            return self._synthetic_prediction(hours, base_datetime)

        base = base_datetime or datetime.now()
        recent = self.data.tail(168).copy()
        out = []

        for step in range(1, hours + 1):
            ts = base + timedelta(hours=step)
            hour = ts.hour
            slice_df = recent[recent["hour"] == hour]
            if slice_df.empty:
                slice_df = recent

            pm25 = float(slice_df["PM25"].mean())
            pm10 = float(slice_df["PM10"].mean())
            no2 = float(slice_df["NO2"].mean())
            co = float(slice_df["CO"].mean())
            so2 = float(slice_df["SO2"].mean())

            noise = np.random.normal(1.0, 0.04)
            out.append(self._record(ts, step, pm25 * noise, pm10 * noise, no2 * noise, co * noise, so2 * noise))

        return out

    def _model_predict(self, hours: int, base_datetime: Optional[datetime] = None) -> List[Dict]:
        """Try model-based PM2.5 prediction, then derive other pollutants proportionally."""
        if self.model is None or self.data is None or self.data.empty:
            return self._fallback_prediction(hours, base_datetime)

        try:
            from pytorch_forecasting import TimeSeriesDataSet

            base = base_datetime or datetime.now()
            encoder_length = 168
            pred_len = min(max(hours, 1), 24)

            context = self.data.tail(encoder_length + pred_len).copy()
            context["time_idx"] = np.arange(len(context), dtype=int)
            context["group_id"] = context["group_id"].astype(str)
            context["is_weekend"] = context["is_weekend"].astype(str)

            ds = TimeSeriesDataSet(
                context,
                time_idx="time_idx",
                target="PM25",
                group_ids=["group_id"],
                max_encoder_length=encoder_length,
                min_encoder_length=encoder_length,
                max_prediction_length=pred_len,
                min_prediction_length=pred_len,
                static_categoricals=["group_id"],
                time_varying_known_categoricals=["is_weekend"],
                time_varying_known_reals=["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"],
                time_varying_unknown_reals=["PM25", "PM10", "NO2", "CO", "SO2", "temperature", "humidity", "wind_speed"],
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
            )

            dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
            preds = self.model.predict(dl).detach().cpu().numpy().reshape(-1)

            out = []
            for idx, pm25 in enumerate(preds[:hours], start=1):
                ts = base + timedelta(hours=idx)
                pm25v = float(pm25)
                out.append(
                    self._record(
                        ts,
                        idx,
                        pm25v,
                        pm25v * 1.75,
                        pm25v * 0.58,
                        pm25v * 0.02,
                        pm25v * 0.16,
                    )
                )

            if len(out) < hours:
                extra = self._fallback_prediction(hours - len(out), base + timedelta(hours=len(out)))
                for i, item in enumerate(extra, start=len(out) + 1):
                    item["hour"] = i
                out.extend(extra)

            return out
        except Exception as exc:
            logger.warning("Model inference failed (%s). Using fallback.", exc)
            return self._fallback_prediction(hours, base_datetime)

    def predict_hours(self, hours: int = 6, base_datetime: Optional[datetime] = None) -> List[Dict]:
        hours = max(1, min(int(hours), 72))
        return self._model_predict(hours, base_datetime)

    def predict_day(self, base_datetime: Optional[datetime] = None) -> List[Dict]:
        return self.predict_hours(24, base_datetime)

    def predict_range(self, start_date: str, end_date: str) -> List[Dict]:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        total_hours = int((end - start).total_seconds() // 3600)
        if total_hours <= 0:
            raise ValueError("end_date must be greater than start_date")

        total_hours = min(total_hours, 72)
        return self.predict_hours(total_hours, start)

    def get_historical_data(self, hours: int = 168) -> List[Dict]:
        if self.data is None or self.data.empty:
            return []
        tail = self.data.tail(max(1, int(hours))).copy()
        rows = []
        for _, row in tail.iterrows():
            rows.append(
                {
                    "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                    "PM2.5": round(float(row["PM25"]), 2),
                    "PM10": round(float(row["PM10"]), 2),
                    "NO2": round(float(row["NO2"]), 2),
                    "CO": round(float(row["CO"]), 2),
                    "SO2": round(float(row["SO2"]), 2),
                }
            )
        return rows


_predictor = None


def get_predictor() -> AirQualityPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AirQualityPredictor()
    return _predictor


if __name__ == "__main__":
    p = AirQualityPredictor()
    for item in p.predict_hours(6):
        print(item)
