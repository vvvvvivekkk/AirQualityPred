# 🌍 Air Quality Prediction System

An end-to-end air quality prediction system using the **Temporal Fusion Transformer (TFT)** model. Predicts PM2.5, PM10, NO₂, CO, and SO₂ levels for multiple time horizons with a FastAPI backend and interactive Streamlit dashboard.

---

## 📁 Project Structure

```
air_quality_prediction/
│
├── data/
│   ├── air_quality.csv              # Dataset (generated or real)
│   └── generate_sample_data.py      # Sample data generator
│
├── models/
│   ├── tft_model.pth                # Trained model weights
│   ├── training_metadata.json       # Training run metadata
│   ├── evaluation_report.json       # Evaluation metrics
│   └── metrics_log.txt              # Metrics history log
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data cleaning & feature engineering
│   ├── train_model.py               # TFT model training pipeline
│   ├── evaluate_model.py            # Model evaluation (RMSE, MAE)
│   └── predict.py                   # Prediction service module
│
├── api/
│   └── main.py                      # FastAPI REST API
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
cd air_quality_prediction
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python data/generate_sample_data.py
```

This creates `data/air_quality.csv` with ~26,000 hourly records (3 years of synthetic data).

### 3. Train the Model

```bash
python src/train_model.py --epochs 30 --batch-size 64
```

**CLI Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/air_quality.csv` | Path to dataset |
| `--epochs` | 30 | Max training epochs |
| `--batch-size` | 64 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--encoder-length` | 168 | Past time steps (7 days) |
| `--prediction-length` | 24 | Forecast horizon (1 day) |
| `--gpus` | 0 | GPU count (0 = CPU) |

### 4. Evaluate the Model

```bash
python src/evaluate_model.py
```

Outputs RMSE and MAE metrics overall and per forecast horizon.

### 5. Start the API Server

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

### 6. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard opens at: **http://localhost:8501**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict-hour` | Predict next N hours |
| `POST` | `/predict-day` | Predict next 24 hours |
| `POST` | `/predict-range` | Predict custom date range |
| `GET` | `/aqi-categories` | AQI category definitions |

### Example: Predict Next 6 Hours

```bash
curl -X POST http://localhost:8000/predict-hour \
  -H "Content-Type: application/json" \
  -d '{"hours": 6}'
```

### Example: Predict Custom Range

```bash
curl -X POST http://localhost:8000/predict-range \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2024-06-15", "end_date": "2024-06-18"}'
```

---

## 📊 Dataset Format

The CSV should contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | datetime | Timestamp (hourly) |
| `PM2.5` | float | Fine particulate matter (µg/m³) |
| `PM10` | float | Coarse particulate matter (µg/m³) |
| `NO2` | float | Nitrogen dioxide (µg/m³) |
| `CO` | float | Carbon monoxide (mg/m³) |
| `SO2` | float | Sulfur dioxide (µg/m³) |
| `temperature` | float | Temperature (°C) |
| `humidity` | float | Relative humidity (%) |
| `wind_speed` | float | Wind speed (m/s) |

---

## 🏷️ AQI Categories

Based on PM2.5 concentration:

| Category | PM2.5 Range | Color |
|----------|-------------|-------|
| Good | 0 - 12.0 | 🟢 |
| Moderate | 12.1 - 35.4 | 🟡 |
| Unhealthy for Sensitive Groups | 35.5 - 55.4 | 🟠 |
| Unhealthy | 55.5 - 150.4 | 🔴 |
| Very Unhealthy | 150.5 - 250.4 | 🟣 |
| Hazardous | 250.5 - 500.0 | 🟤 |

---

## 📈 Evaluation Metrics

The system evaluates model performance using:
- **RMSE** (Root Mean Squared Error) — penalizes large errors
- **MAE** (Mean Absolute Error) — average absolute prediction error

Metrics are computed:
- Overall across all predictions
- Per forecast horizon (hour 1, 6, 12, 24)

---

## 🛠️ Tech Stack

- **Model**: Temporal Fusion Transformer (PyTorch Forecasting)
- **Training**: PyTorch Lightning
- **API**: FastAPI + Uvicorn
- **Dashboard**: Streamlit + Plotly
- **Data**: Pandas, NumPy, Scikit-learn
