"""
Model evaluation module for Air Quality Prediction System.
Computes RMSE, MAE, and generates evaluation reports.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE and MAE evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with RMSE and MAE
    """
    # Flatten if multi-dimensional
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Remove any NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "n_samples": int(len(y_true)),
    }


def evaluate_model(model, test_dataloader, trainer=None):
    """
    Evaluate a trained TFT model on the test set.

    Args:
        model: Trained TemporalFusionTransformer model
        test_dataloader: Test DataLoader
        trainer: PyTorch Lightning Trainer (optional)

    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 60)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("=" * 60)

    model.eval()

    # Get predictions
    try:
        predictions = model.predict(test_dataloader, return_y=True, trainer_kwargs={"accelerator": "cpu"})
        y_pred = predictions.output.cpu().numpy()
        y_true = predictions.y[0].cpu().numpy() if isinstance(predictions.y, (list, tuple)) else predictions.y.cpu().numpy()
    except Exception as e:
        logger.warning(f"Standard prediction failed: {e}. Trying manual prediction loop.")
        y_pred_list, y_true_list = [], []
        with torch.no_grad():
            for batch in test_dataloader:
                x, y = batch
                pred = model(x)
                # TFT returns dict; get the main prediction (median quantile)
                if isinstance(pred, dict):
                    pred_vals = pred["prediction"][:, :, 3]  # Median quantile (index 3 of 7)
                else:
                    pred_vals = pred
                y_true_vals = y[0] if isinstance(y, (list, tuple)) else y
                y_pred_list.append(pred_vals.cpu().numpy())
                y_true_list.append(y_true_vals.cpu().numpy())

        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)

    # Handle quantile output — take median
    if y_pred.ndim == 3:
        y_pred = y_pred[:, :, 3]  # Median quantile

    # Overall metrics
    overall_metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Overall — RMSE: {overall_metrics['rmse']}, MAE: {overall_metrics['mae']}")

    # Per-horizon metrics
    horizon_metrics = {}
    n_horizons = min(y_pred.shape[1] if y_pred.ndim > 1 else 1, 24)

    if y_pred.ndim > 1 and y_true.ndim > 1:
        for h in range(n_horizons):
            h_metrics = compute_metrics(y_true[:, h], y_pred[:, h])
            horizon_metrics[f"hour_{h+1}"] = h_metrics
            if (h + 1) in [1, 6, 12, 24]:
                logger.info(f"  Hour {h+1:2d} — RMSE: {h_metrics['rmse']}, MAE: {h_metrics['mae']}")

    # Build evaluation report
    report = {
        "evaluated_at": datetime.now().isoformat(),
        "overall": overall_metrics,
        "per_horizon": horizon_metrics,
        "prediction_shape": list(y_pred.shape),
    }

    # Save report
    report_path = os.path.join(PROJECT_ROOT, "models", "evaluation_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved to {report_path}")

    # Log to metrics file
    log_path = os.path.join(PROJECT_ROOT, "models", "metrics_log.txt")
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evaluation - {report['evaluated_at']}\n")
        f.write(f"  Overall RMSE: {overall_metrics['rmse']}\n")
        f.write(f"  Overall MAE: {overall_metrics['mae']}\n")
        f.write(f"  Test samples: {overall_metrics['n_samples']}\n")

    return report, y_true, y_pred


def print_evaluation_summary(report: dict):
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 60)
    print("   AIR QUALITY MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Evaluated at: {report['evaluated_at']}")
    print(f"\n  Overall Metrics:")
    print(f"    RMSE : {report['overall']['rmse']}")
    print(f"    MAE  : {report['overall']['mae']}")
    print(f"    Samples: {report['overall']['n_samples']}")

    if report.get("per_horizon"):
        print(f"\n  Per-Horizon RMSE / MAE:")
        for horizon, metrics in report["per_horizon"].items():
            if any(k in horizon for k in ["hour_1 ", "hour_6 ", "hour_12", "hour_24"]) or horizon in [
                "hour_1", "hour_6", "hour_12", "hour_24"
            ]:
                print(f"    {horizon:10s} — RMSE: {metrics['rmse']:8.4f}  MAE: {metrics['mae']:8.4f}")

    print("=" * 60)


if __name__ == "__main__":
    from src.train_model import train_model

    print("[1] Training model...")
    model, trainer, test_dl, training_ds = train_model(max_epochs=5)

    print("\n[2] Evaluating model...")
    report, y_true, y_pred = evaluate_model(model, test_dl, trainer)

    print_evaluation_summary(report)
    print("\n[✓] Evaluation complete!")
