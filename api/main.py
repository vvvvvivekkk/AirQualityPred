"""
FastAPI Prediction Service for Air Quality Prediction System.
Provides REST API endpoints for air quality forecasting.

Endpoints:
    - GET  /                → Health check & API docs link
    - POST /predict-hour    → Predict next N hours
    - POST /predict-day     → Predict next 24 hours
    - POST /predict-range   → Predict custom date range
    - GET  /aqi-categories  → List AQI categories
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import AirQualityPredictor, AQI_COLORS
from src.data_preprocessing import AQI_BREAKPOINTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Air Quality Prediction API",
    description=(
        "REST API for predicting air quality levels using a Temporal Fusion Transformer model. "
        "Provides hourly, daily, and custom range predictions for PM2.5, PM10, NO2, CO, and SO2."
    ),
    version="1.0.0",
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

# ─── Lazy-loaded predictor ────────────────────────────────────
_predictor: Optional[AirQualityPredictor] = None


def get_predictor() -> AirQualityPredictor:
    """Get or initialize the predictor singleton."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = AirQualityPredictor()
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise HTTPException(status_code=503, detail="Model service unavailable")
    return _predictor


# ─── Request / Response Models ────────────────────────────────

class HourPredictionRequest(BaseModel):
    """Request body for hourly prediction."""
    hours: int = Field(default=6, ge=1, le=72, description="Number of hours to predict (1-72)")
    base_datetime: Optional[str] = Field(
        default=None,
        description="Base datetime for prediction start (ISO format). Defaults to now.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "hours": 6,
                "base_datetime": "2024-06-15T10:00:00",
            }
        }


class DayPredictionRequest(BaseModel):
    """Request body for daily prediction."""
    date: Optional[str] = Field(
        default=None,
        description="Date to predict from (YYYY-MM-DD). Defaults to today.",
    )

    class Config:
        json_schema_extra = {
            "example": {"date": "2024-06-15"}
        }


class RangePredictionRequest(BaseModel):
    """Request body for range prediction."""
    start_date: str = Field(..., description="Start date/time (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    end_date: str = Field(..., description="End date/time (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2024-06-15",
                "end_date": "2024-06-18",
            }
        }


class PredictionItem(BaseModel):
    datetime: str
    hour: int
    PM25: float = Field(..., alias="PM2.5")
    PM10: float
    NO2: float
    CO: float
    SO2: float
    aqi_category: str
    aqi_color: str

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    status: str = "success"
    prediction_type: str
    count: int
    predictions: list


# ─── Endpoints ────────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def root():
    """Health check and API information."""
    return {
        "service": "Air Quality Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/predict-hour", "/predict-day", "/predict-range", "/aqi-categories"],
    }


@app.post("/predict-hour", response_model=PredictionResponse, tags=["Predictions"])
async def predict_hour(request: HourPredictionRequest):
    """
    Predict air quality for the next N hours.

    - **hours**: Number of hours to predict (1-72, default: 6)
    - **base_datetime**: Optional start datetime (ISO format)
    """
    try:
        predictor = get_predictor()

        base_dt = None
        if request.base_datetime:
            try:
                base_dt = datetime.fromisoformat(request.base_datetime)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")

        predictions = predictor.predict_hours(request.hours, base_dt)

        return PredictionResponse(
            prediction_type="hourly",
            count=len(predictions),
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-day", response_model=PredictionResponse, tags=["Predictions"])
async def predict_day(request: DayPredictionRequest):
    """
    Predict air quality for the next 24 hours.

    - **date**: Optional date to predict from (YYYY-MM-DD, defaults to today)
    """
    try:
        predictor = get_predictor()

        base_dt = None
        if request.date:
            try:
                base_dt = datetime.fromisoformat(request.date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        predictions = predictor.predict_day(base_dt)

        return PredictionResponse(
            prediction_type="daily",
            count=len(predictions),
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-range", response_model=PredictionResponse, tags=["Predictions"])
async def predict_range(request: RangePredictionRequest):
    """
    Predict air quality for a custom date range.

    - **start_date**: Start date/time (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
    - **end_date**: End date/time (max 7 days from start)
    """
    try:
        predictor = get_predictor()
        predictions = predictor.predict_range(request.start_date, request.end_date)

        return PredictionResponse(
            prediction_type="range",
            count=len(predictions),
            predictions=predictions,
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/aqi-categories", tags=["Reference"])
async def aqi_categories():
    """Get AQI category definitions and color mappings."""
    categories = []
    for low, high, name in AQI_BREAKPOINTS:
        categories.append({
            "category": name,
            "pm25_low": low,
            "pm25_high": high,
            "color": AQI_COLORS.get(name, "#808080"),
        })
    return {"categories": categories}


# ─── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
