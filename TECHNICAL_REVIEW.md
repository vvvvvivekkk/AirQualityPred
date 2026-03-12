# Air Quality Prediction System: Technical Review

## 1. Project Overview

This project is an end-to-end air quality forecasting system built around a Temporal Fusion Transformer model. It predicts short-horizon pollutant levels, serves those predictions through a FastAPI backend, and visualizes them in a browser dashboard.

Its main purpose is to forecast air quality measurements such as PM2.5, PM10, NO2, CO, and SO2 from historical time-series data plus weather features. The system is shaped like a practical ML application rather than a research notebook: it has preprocessing, training, evaluation, inference, an API layer, and a frontend.

### Key Features

1. Canonical preprocessing pipeline in `src/data_preprocessing.py`
2. TFT training pipeline in `src/train_model.py`
3. Evaluation with RMSE and MAE in `src/evaluate_model.py`
4. Inference service with checkpoint loading and fallback logic in `src/predict.py`
5. FastAPI prediction endpoints in `api/main.py`
6. Static dashboard UI in `frontend/index.html`, `frontend/script.js`, and `frontend/style.css`

## 2. Tech Stack

### Programming Languages

1. Python for ML, preprocessing, inference, and backend
2. HTML, CSS, and JavaScript for the frontend

### Frameworks and Libraries

1. PyTorch for tensor computation and model execution
2. PyTorch Forecasting for Temporal Fusion Transformer and `TimeSeriesDataSet`
3. Lightning / PyTorch Lightning for the training loop and callbacks
4. Scikit-learn for `StandardScaler` and evaluation metrics
5. Pandas and NumPy for tabular and numerical processing
6. FastAPI for backend APIs
7. Uvicorn as the ASGI server
8. Plotly for frontend visualizations
9. Matplotlib appears in dependencies but is not used by the current code paths

### Databases

1. None
2. The project uses CSV files and filesystem artifacts instead of a database

### APIs and External Services

1. No third-party APIs are used
2. The only API surface is the project's own FastAPI service

### Frontend, Backend, and Infrastructure

1. Frontend: static HTML/CSS/JS with Plotly
2. Backend: FastAPI served by Uvicorn
3. Infrastructure: local filesystem artifacts under `models/` and training logs under `lightning_logs/`
4. No Docker, CI/CD, cloud deployment, or orchestration files are present

## 3. Architecture

The system follows a standard layered ML application architecture.

### Layers

1. Data layer
   - CSV dataset in `data/air_quality.csv`
   - Synthetic data generator in `data/generate_sample_data.py`

2. Processing and model layer
   - Preprocessing normalizes schema, imputes missing values, adds time features, categorizes AQI, optionally scales features, and builds TFT-ready columns
   - Training builds `TimeSeriesDataSet` objects and trains a Temporal Fusion Transformer
   - Evaluation produces offline metrics and a JSON report

3. Serving layer
   - Predictor object loads the latest checkpoint if available
   - If model loading or inference fails, it falls back to historical-statistical or synthetic predictions

4. API layer
   - FastAPI exposes prediction and historical endpoints
   - Static frontend is mounted and served from the same backend

5. Presentation layer
   - Browser dashboard fetches API responses and renders charts with Plotly

### Module Interaction

1. `src/data_preprocessing.py` is the shared dependency for training and inference
2. `src/train_model.py` consumes preprocessing output and writes artifacts
3. `src/evaluate_model.py` evaluates trained models
4. `src/predict.py` loads artifacts and provides inference APIs
5. `api/main.py` wraps predictor methods as HTTP endpoints
6. `frontend/script.js` calls those endpoints and updates the UI

### Data Flow

1. Raw CSV is loaded and column names are normalized
2. Missing values are interpolated and back/forward-filled
3. Calendar and cyclical features are derived from `datetime`
4. The dataset is converted into TFT-compatible format with `time_idx` and `group_id`
5. Training produces checkpoint and metadata files
6. API server initializes `AirQualityPredictor` lazily
7. Frontend requests predictions or historical data
8. API returns JSON payloads for charts and summary cards

### Design Patterns Used

1. Pipeline pattern for preprocessing and training stages
2. Lazy initialization for the predictor singleton in `src/predict.py` and `api/main.py`
3. Graceful degradation via fallback prediction modes
4. Artifact-based state management rather than a database-backed model registry

## 4. Algorithms

The project mixes conventional data processing algorithms with a deep learning forecasting model.

### Implemented Algorithms

1. Linear interpolation for missing values
   - Used in preprocessing to fill numeric gaps
   - Complexity is roughly linear in the number of rows per column
   - Chosen because time-series gaps are usually best handled with ordered interpolation rather than row dropping

2. Forward fill and backward fill
   - Used after interpolation to cover edge cases at the beginning or end of the series
   - Linear complexity
   - Chosen to guarantee no remaining nulls in numeric features

3. Cyclical time encoding
   - Sine and cosine transforms of hour, month, and day-of-week
   - Constant work per row, overall linear in dataset size
   - Chosen because periodic time features are better represented on a circle than as raw integers

4. AQI categorization by breakpoints
   - Implemented as a breakpoint scan over PM2.5 values
   - Constant time per row due to fixed breakpoint count
   - Used to map continuous PM2.5 into human-readable health categories

5. Statistical fallback forecasting
   - In fallback mode, the predictor takes recent historical data, filters by hour-of-day, averages pollutant levels, and adds mild random noise
   - Roughly linear in the fallback window size
   - Chosen to keep the app functional even if checkpoint loading or model inference fails

6. Synthetic forecasting
   - Uses Gaussian-shaped morning and evening traffic peaks plus noise
   - Constant work per forecasted hour
   - Intended as a last-resort demo-safe forecast generator

### Why These Were Chosen

1. They are simple, robust, and appropriate for a single-station time series
2. They keep the application usable even when the ML stack is unavailable
3. They support a demo-to-production transition path without changing the public API

## 5. AI / Machine Learning Components

### Primary Model

1. Temporal Fusion Transformer, created in `src/train_model.py`

### Libraries Used

1. PyTorch
2. PyTorch Forecasting
3. Lightning / PyTorch Lightning
4. Scikit-learn

### Model Architecture Explanation

1. The model is a Temporal Fusion Transformer for multivariate time-series forecasting
2. It uses:
   - variable selection networks
   - LSTM encoder/decoder blocks
   - interpretable multi-head attention
   - gated residual networks
   - quantile output heads
3. The current configuration uses:
   - `hidden_size = 32`
   - `attention_head_size = 2`
   - `dropout = 0.1`
   - `hidden_continuous_size = 16`
   - `output_size = 7`
4. Output size 7 corresponds to quantile forecasts under `QuantileLoss`

### What the Model Predicts

1. The training target is `PM25`, the normalized internal name for PM2.5
2. The API returns `PM2.5` externally for compatibility and readability
3. In live inference, model output is used directly for PM2.5 and the other pollutants are derived proportionally in `src/predict.py`
4. That means only PM2.5 is truly model-predicted; PM10, NO2, CO, and SO2 are heuristic projections at inference time

## 6. Training Details

### Training Pipeline

1. Load CSV and preprocess with normalization enabled
2. Build TFT-compatible dataset with encoder and prediction windows
3. Split data into train, validation, and test partitions by time index
4. Build `TimeSeriesDataSet` objects
5. Create dataloaders
6. Build TFT model from dataset schema
7. Train with Lightning callbacks
8. Save best checkpoint, state dict, and metadata

### Training Configuration from Code

1. Default epochs: 30
2. Batch size: 64
3. Learning rate: 0.001
4. Encoder length: 168
5. Prediction horizon: 24
6. Device selection: CPU unless GPU is available and requested

### Loss Function

1. `QuantileLoss` from PyTorch Forecasting

### Optimizer

1. The code does not explicitly set the optimizer
2. It most likely relies on the default optimizer configuration generated by `TemporalFusionTransformer.from_dataset`, which is typically Adam-based unless overridden
3. That inference is based on framework behavior, not on a custom optimizer definition in this repository

### Callbacks and Controls

1. `EarlyStopping` on validation loss
2. `ModelCheckpoint` to persist the best checkpoint
3. `LearningRateMonitor` for epoch-level learning rate tracking

### Evaluation Metrics

1. RMSE
2. MAE
3. Per-horizon metrics when tensor shapes allow it

### Important Caveat

1. The main block of `src/evaluate_model.py` retrains a model for 2 epochs before evaluating it
2. That means the script's direct entry point does not evaluate the previously trained best checkpoint by default

## 7. Dataset

### Training Data

1. Primary dataset file is `data/air_quality.csv`
2. Synthetic data can be generated by `data/generate_sample_data.py`

### Dataset Structure

1. `datetime`
2. `PM2.5`
3. `PM10`
4. `NO2`
5. `CO`
6. `SO2`
7. `temperature`
8. `humidity`
9. `wind_speed`

### Preprocessing and Feature Engineering

1. Column name normalization maps `PM2.5` and variants to `PM25` internally
2. Required-column validation ensures expected schema exists
3. Numeric missing values are interpolated and filled
4. Time features:
   - `hour`
   - `day_of_week`
   - `day_of_month`
   - `month`
   - `day_of_year`
   - `week_of_year`
   - `is_weekend`
5. Cyclical encodings:
   - `hour_sin`
   - `hour_cos`
   - `month_sin`
   - `month_cos`
   - `dow_sin`
   - `dow_cos`
6. AQI category is derived from `PM25`
7. `time_idx` and `group_id` are added for `TimeSeriesDataSet`

### Data Sources

1. No external real-world data integration is present
2. The included generator creates synthetic but structured air-quality data with seasonal, traffic, and weather-like patterns
3. Based on repository contents, the system is currently oriented toward experimentation, coursework, or a demonstrable prototype rather than a production data ingestion pipeline

## 8. Environment Variables and Configuration

### Environment Variables

1. No environment variables are used in the current codebase
2. No use of `os.environ`, `getenv`, dotenv, or secret-loading code was found in the repository code paths reviewed

### What Controls Behavior Instead

1. Hardcoded constants in source files
2. CLI arguments in `src/train_model.py`
3. Filesystem-based model discovery in `src/predict.py`

### Configuration Inputs Present

1. Training CLI arguments:
   - `data`
   - `epochs`
   - `batch-size`
   - `lr`
   - `encoder-length`
   - `prediction-length`
   - `gpus`
2. API runtime options via Uvicorn CLI
3. Predictor checkpoint auto-discovery by latest modified `.ckpt` file

### Secrets and Tokens

1. None found
2. No API keys, secrets, database passwords, or tokens are present in the repository code reviewed

## 9. Deployment

### Current Run Model

1. Local execution with Python and Uvicorn
2. Frontend is served by the backend under the same host

### Practical Startup Sequence

1. Install dependencies from `requirements.txt`
2. Train the model using `src/train_model.py`
3. Start Uvicorn with the project root on the import path
4. Open the dashboard at `/app`

### Deployment Assets Not Present

1. No Dockerfile
2. No docker-compose file
3. No GitHub Actions or CI workflows
4. No Terraform, Kubernetes, or cloud-specific deployment configuration

### Operational Note

1. The README launch command works when executed from the project root
2. From the workspace root, the backend needed an explicit app-dir to resolve the `api` module

## 10. Folder Structure Explanation

### `api/`

1. Contains the FastAPI entry point
2. Defines health, prediction, historical, and AQI endpoints
3. Mounts the frontend static files

### `data/`

1. Stores the CSV dataset
2. Contains the synthetic data generator

### `frontend/`

1. Static client application
2. HTML defines controls and chart containers
3. JavaScript calls the backend and renders Plotly charts
4. CSS provides layout and styling

### `models/`

1. Stores checkpoint files
2. Stores state dict file
3. Stores evaluation and training metadata
4. Acts as the project's artifact registry

### `src/`

1. Core business logic for preprocessing, training, evaluation, and prediction
2. This is the main ML/application layer

### `lightning_logs/`

1. Lightning-generated training logs
2. Useful for debugging and experiment inspection

### `test_fix.py`

1. A lightweight manual verification script
2. Confirms that the PM2.5 to PM25 canonicalization fix works and dataset construction succeeds
3. This is not a real automated test suite

### `README.md`

1. Project documentation and run instructions
2. Broadly accurate, though a few operational nuances are not documented

### `requirements.txt`

1. Python dependency list
2. Includes both ML and backend/frontend support packages

## 11. Step-by-Step Execution Flow

### Runtime Flow

1. Uvicorn starts the FastAPI app from `api/main.py`
2. The frontend is made available under `/app` and `/frontend`
3. The browser loads `frontend/index.html`
4. `frontend/script.js` immediately requests predictions and historical data
5. When the first prediction endpoint is called, the API lazily initializes `AirQualityPredictor` from `src/predict.py`
6. The predictor tries to discover and load the latest checkpoint from `models/`
7. It also preprocesses the CSV dataset without normalization to keep values on their original scale for history and fallback logic
8. If the checkpoint loads successfully, the predictor constructs a `TimeSeriesDataSet` and runs model inference
9. If model inference fails, it falls back to recent-hour averaging
10. If no usable data exists, it falls back again to synthetic traffic-pattern generation
11. The API returns structured JSON including pollutant values, AQI category, and AQI color
12. The frontend updates summary cards and renders Plotly charts for PM2.5, multi-pollutant comparison, historical versus predicted PM2.5, and AQI category distribution

### Training Flow

1. Training script preprocesses the dataset with scaling enabled
2. It creates train, validation, and test splits
3. It trains the TFT model with Lightning
4. It saves the best checkpoint, a state dict, and metadata
5. Evaluation can then produce RMSE and MAE reports

## 12. Improvements

### Optimizations

1. Evaluate the real best checkpoint directly instead of retraining inside `src/evaluate_model.py`
2. Add explicit model version selection rather than always choosing the most recently modified checkpoint
3. Predict all pollutants directly instead of deriving non-PM2.5 pollutants by fixed ratios
4. Add true multi-sample validation and testing because current validation and test dataset sizes are extremely small in practice
5. Persist the scaler if normalized inference is ever required beyond the current fallback design

### Security Improvements

1. Restrict CORS instead of allowing all origins with credentials in `api/main.py`
2. Add input validation hardening for date range edge cases and malformed payloads
3. Separate demo mode from production mode so synthetic fallback is not mistaken for model-backed predictions

### Scalability Suggestions

1. Introduce a database or object storage for artifacts and prediction history if the system grows
2. Move training and serving into separate processes or services
3. Add background job orchestration for training runs
4. Cache repeated predictions for common short-range queries
5. Add containerization and deployment automation

### Code Quality Improvements

1. Add automated tests for preprocessing, API endpoints, checkpoint discovery, and fallback paths
2. Remove unused dependencies like Matplotlib if it remains unused
3. Make configuration environment-driven instead of hardcoded
4. Fix README runtime instructions to clarify required working directory or app-dir behavior
5. Log whether a response came from model inference, fallback statistics, or synthetic generation
6. Add reproducibility controls such as fixed training seeds and structured experiment tracking

## Assessment

This is a solid prototype or technical demo of an ML forecasting application. The strongest parts are the clear layering, the usable end-to-end flow, and the defensive fallback behavior. The weakest parts are evaluation rigor, configuration management, and production hardening. The most important architectural limitation is that the served multi-pollutant output is partly heuristic, not fully model-driven.

The system is understandable to a new developer because the code is separated by responsibility and the runtime path is straightforward. For a technical review, the main points to emphasize are:

1. The project is end-to-end complete
2. The ML component is real and integrated into a serving stack
3. Evaluation and deployment maturity still need work
4. The architecture is strong enough to evolve into a more production-ready design