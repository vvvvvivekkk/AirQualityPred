"""Evaluate TFT model performance with RMSE and MAE."""

import json
import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "n_samples": int(len(y_true))}


def _extract_predictions(model, test_dataloader):
    """Get y_true and y_pred from pytorch-forecasting outputs."""
    pred_obj = model.predict(test_dataloader, return_y=True, trainer_kwargs={"accelerator": "cpu"})

    y_pred = pred_obj.output.detach().cpu().numpy() if hasattr(pred_obj.output, "detach") else np.asarray(pred_obj.output)
    y_true = pred_obj.y[0] if isinstance(pred_obj.y, (tuple, list)) else pred_obj.y
    y_true = y_true.detach().cpu().numpy() if hasattr(y_true, "detach") else np.asarray(y_true)

    # Quantile output shape is usually [batch, horizon, quantiles]. Use median quantile (index 3).
    if y_pred.ndim == 3:
        y_pred = y_pred[:, :, 3]

    return y_true, y_pred


def evaluate_model(model, test_dataloader):
    logger.info("%s", "=" * 60)
    logger.info("EVALUATING MODEL")
    logger.info("%s", "=" * 60)

    y_true, y_pred = _extract_predictions(model, test_dataloader)

    overall = compute_metrics(y_true, y_pred)
    logger.info("Overall RMSE: %.4f | MAE: %.4f", overall["rmse"], overall["mae"])

    per_horizon = {}
    if y_true.ndim > 1 and y_pred.ndim > 1:
        horizon_n = min(y_true.shape[1], y_pred.shape[1], 72)
        for idx in range(horizon_n):
            key = f"hour_{idx + 1}"
            per_horizon[key] = compute_metrics(y_true[:, idx], y_pred[:, idx])

    report = {
        "evaluated_at": datetime.now().isoformat(),
        "overall": overall,
        "per_horizon": per_horizon,
        "prediction_shape": list(np.asarray(y_pred).shape),
    }

    out_path = os.path.join(PROJECT_ROOT, "models", "evaluation_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report saved: %s", out_path)
    return report, y_true, y_pred


def print_evaluation_summary(report):
    print("\n" + "=" * 60)
    print("AIR QUALITY MODEL EVALUATION")
    print("=" * 60)
    print(f"Evaluated at: {report['evaluated_at']}")
    print(f"Overall RMSE: {report['overall']['rmse']}")
    print(f"Overall MAE : {report['overall']['mae']}")
    print(f"Samples     : {report['overall']['n_samples']}")


if __name__ == "__main__":
    from src.train_model import train_model

    model, _, test_dl, _ = train_model(max_epochs=2)
    result, _, _ = evaluate_model(model, test_dl)
    print_evaluation_summary(result)
