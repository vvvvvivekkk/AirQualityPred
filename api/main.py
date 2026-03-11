"""FastAPI backend for Air Quality Prediction System."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import AQI_BREAKPOINTS
from src.predict import AQI_COLORS, AirQualityPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Air Quality Prediction API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

_predictor: Optional[AirQualityPredictor] = None


class HourPredictionRequest(BaseModel):
    hours: int = Field(default=6, ge=1, le=72)
    base_datetime: Optional[str] = None


class DayPredictionRequest(BaseModel):
    date: Optional[str] = None


class RangePredictionRequest(BaseModel):
    start_date: str
    end_date: str


def _get_predictor() -> AirQualityPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = AirQualityPredictor()
            logger.info("Predictor initialized")
        except Exception as exc:
            logger.error("Predictor init failed: %s", exc)
            raise HTTPException(status_code=503, detail="Model service unavailable")
    return _predictor


@app.get("/")
def health():
    return {
        "service": "Air Quality Prediction API",
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs",
        "frontend": "/app",
    }


@app.get("/app")
def frontend_app():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.post("/predict-hour")
def predict_hour(req: HourPredictionRequest):
    predictor = _get_predictor()
    base_dt = None
    if req.base_datetime:
        try:
            base_dt = datetime.fromisoformat(req.base_datetime)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid base_datetime format") from exc

    predictions = predictor.predict_hours(req.hours, base_dt)
    return {
        "status": "success",
        "prediction_type": "hourly",
        "count": len(predictions),
        "predictions": predictions,
    }


@app.post("/predict-day")
def predict_day(req: Optional[DayPredictionRequest] = None):
    predictor = _get_predictor()
    base_dt = None
    if req and req.date:
        try:
            base_dt = datetime.fromisoformat(req.date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date format") from exc

    predictions = predictor.predict_day(base_dt)
    return {
        "status": "success",
        "prediction_type": "daily",
        "count": len(predictions),
        "predictions": predictions,
    }


@app.post("/predict-range")
def predict_range(req: RangePredictionRequest):
    predictor = _get_predictor()
    try:
        predictions = predictor.predict_range(req.start_date, req.end_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "success",
        "prediction_type": "range",
        "count": len(predictions),
        "predictions": predictions,
    }


@app.get("/historical")
def historical(hours: int = 168):
    predictor = _get_predictor()
    data = predictor.get_historical_data(hours=hours)
    return {
        "status": "success",
        "count": len(data),
        "historical": data,
    }


@app.get("/aqi-categories")
def aqi_categories():
    return {
        "categories": [
            {
                "category": name,
                "pm25_low": low,
                "pm25_high": high,
                "color": AQI_COLORS.get(name, "#808080"),
            }
            for low, high, name in AQI_BREAKPOINTS
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
