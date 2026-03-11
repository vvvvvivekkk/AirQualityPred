# Air Quality Prediction System

End-to-end air quality prediction using a Temporal Fusion Transformer (TFT), FastAPI backend, and a static HTML/CSS/JavaScript frontend.

## Project Structure

```
air_quality_prediction/
├── data/
├── models/
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── api/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
```

## Functional Coverage

- FR11: TFT model implemented in `src/train_model.py`.
- FR12: Training uses historical columns: datetime, PM2.5, PM10, NO2, CO, SO2, temperature, humidity, wind_speed.
- FR13: Predictions supported for next few hours, next day (24h), and up to 72 hours.
- FR14: RMSE and MAE calculated in `src/evaluate_model.py`.
- FR15: FastAPI service in `api/main.py` with lazy model loading.
- FR16: API endpoints for health, hourly, day, and range predictions.

## API Endpoints

- `GET /` health check
- `GET /app` frontend entry page
- `POST /predict-hour`
- `POST /predict-day`
- `POST /predict-range`
- `GET /historical?hours=168`
- `GET /aqi-categories`

## Example Requests

### Predict next few hours

```bash
curl -X POST http://127.0.0.1:8000/predict-hour \
  -H "Content-Type: application/json" \
  -d '{"hours": 6}'
```

### Predict next day

```bash
curl -X POST http://127.0.0.1:8000/predict-day \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Predict custom range

```bash
curl -X POST http://127.0.0.1:8000/predict-range \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2026-03-12", "end_date": "2026-03-14"}'
```

## Example Response (predict-hour)

```json
{
  "status": "success",
  "prediction_type": "hourly",
  "count": 6,
  "predictions": [
    {
      "datetime": "2026-03-12T10:00:00",
      "hour": 1,
      "PM2.5": 34.21,
      "PM10": 56.14,
      "NO2": 20.41,
      "CO": 0.92,
      "SO2": 8.14,
      "aqi_category": "Moderate",
      "aqi_color": "#FFFF00"
    }
  ]
}
```

## Steps to Run

1. Install dependencies

```bash
cd air_quality_prediction
pip install -r requirements.txt
```

2. Generate sample data (if needed)

```bash
python data/generate_sample_data.py
```

3. Train TFT model

```bash
python src/train_model.py --epochs 30 --batch-size 64
```

4. Evaluate model

```bash
python src/evaluate_model.py
```

5. Run API + frontend server

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

6. Open pages

- Frontend: `http://127.0.0.1:8000/app`
- API docs: `http://127.0.0.1:8000/docs`

## Notes on Bug Fixes

- Fixed KeyError on `PM2.5` by normalizing to canonical internal `PM25` and returning `PM2.5` in API output.
- Preprocessing now sanitizes/validates column names before training and inference.
- Predictor robustly loads latest checkpoint if available; falls back safely when unavailable.
- API consistently returns JSON payloads for all prediction endpoints.
- Frontend fetch logic uses the same host/port as backend to avoid cross-origin and URL mismatch issues.
