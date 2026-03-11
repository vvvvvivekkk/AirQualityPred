"""
Temporal Fusion Transformer (TFT) model training module.
Uses PyTorch Forecasting for the TFT implementation with PyTorch Lightning trainer.
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

try:
    # Prefer unified Lightning package to match PyTorch Forecasting model base classes.
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
except ImportError:
    # Backward compatibility for environments that still use pytorch_lightning.
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import preprocess_pipeline


def build_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 168,
    max_prediction_length: int = 24,
    training_cutoff: float = 0.8,
    validation_cutoff: float = 0.9,
):
    """
    Build PyTorch Forecasting TimeSeriesDataSet for TFT.

    Args:
        df: Preprocessed DataFrame with time_idx and group_id columns
        max_encoder_length: Number of past steps the model sees (168h = 7 days)
        max_prediction_length: Number of future steps to predict (24h = 1 day)
        training_cutoff: Fraction of data for training
        validation_cutoff: Fraction of data for training + validation

    Returns:
        Tuple of (training dataset, validation dataset, test dataset)
    """
    n = df["time_idx"].max() + 1
    train_cutoff = int(n * training_cutoff)
    val_cutoff = int(n * validation_cutoff)

    logger.info(f"Dataset split — Train: 0-{train_cutoff}, Val: {train_cutoff}-{val_cutoff}, Test: {val_cutoff}-{n}")

    # Define known & unknown reals, categoricals
    # Note: PM2.5 is renamed to PM25 in preprocessing to avoid '.' characters
    time_varying_known_reals = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dow_sin", "dow_cos",
    ]
    time_varying_unknown_reals = [
        "PM25", "PM10", "NO2", "CO", "SO2",
        "temperature", "humidity", "wind_speed",
    ]
    static_categoricals = ["group_id"]
    time_varying_known_categoricals = ["is_weekend"]

    # Filter columns that actually exist
    available_cols = set(df.columns)
    time_varying_known_reals = [c for c in time_varying_known_reals if c in available_cols]
    time_varying_unknown_reals = [c for c in time_varying_unknown_reals if c in available_cols]

    # Ensure correct dtypes
    df["group_id"] = df["group_id"].astype(str)
    df["is_weekend"] = df["is_weekend"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)

    # Training dataset
    training = TimeSeriesDataSet(
        df[df.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="PM25",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Validation dataset (from training parameters)
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[(df.time_idx > train_cutoff) & (df.time_idx <= val_cutoff)],
        predict=True,
        stop_randomization=True,
    )

    # Test dataset
    test = TimeSeriesDataSet.from_dataset(
        training,
        df[df.time_idx > val_cutoff],
        predict=True,
        stop_randomization=True,
    )

    logger.info(f"Training samples: {len(training)}, Validation: {len(validation)}, Test: {len(test)}")
    return training, validation, test


def create_dataloaders(training, validation, test, batch_size: int = 64, num_workers: int = 0):
    """Create DataLoaders from TimeSeriesDataSets."""
    train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_dl = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    test_dl = test.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    return train_dl, val_dl, test_dl


def build_tft_model(training: TimeSeriesDataSet, learning_rate: float = 0.001):
    """
    Build a Temporal Fusion Transformer model.

    Args:
        training: Training TimeSeriesDataSet (needed for model configuration)
        learning_rate: Initial learning rate

    Returns:
        TemporalFusionTransformer model
    """
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    logger.info(f"TFT model created — Parameters: {model.size() / 1e3:.1f}K")
    return model


def train_model(
    data_path: str = None,
    max_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    max_encoder_length: int = 168,
    max_prediction_length: int = 24,
    gpus: int = 0,
):
    """
    Full training pipeline: preprocess → build dataset → train TFT.

    Args:
        data_path: Path to raw CSV data
        max_epochs: Maximum training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        max_encoder_length: Encoder sequence length
        max_prediction_length: Prediction horizon
        gpus: Number of GPUs (0 = CPU)

    Returns:
        Tuple of (trained model, trainer, test dataloader, training dataset)
    """
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "air_quality.csv")

    # Preprocess
    df, scaler = preprocess_pipeline(data_path)

    # Build datasets
    training_ds, validation_ds, test_ds = build_dataset(
        df,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )

    # Dataloaders
    train_dl, val_dl, test_dl = create_dataloaders(
        training_ds, validation_ds, test_ds,
        batch_size=batch_size,
    )

    # Build model
    model = build_tft_model(training_ds, learning_rate=learning_rate)

    # Callbacks
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min"),
        ModelCheckpoint(
            dirpath=model_dir,
            filename="tft_best_{epoch}_{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    accelerator = "gpu" if gpus > 0 and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    logger.info("=" * 60)
    logger.info("STARTING TFT TRAINING")
    logger.info(f"Epochs: {max_epochs}, Batch: {batch_size}, LR: {learning_rate}")
    logger.info(f"Encoder length: {max_encoder_length}, Prediction length: {max_prediction_length}")
    logger.info("=" * 60)

    # Train
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Best model checkpoint: {best_model_path}")
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    else:
        best_model = model

    # Save final model as .pth
    pth_path = os.path.join(model_dir, "tft_model.pth")
    torch.save(best_model.state_dict(), pth_path)
    logger.info(f"Model weights saved to {pth_path}")

    # Save training metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "encoder_length": max_encoder_length,
        "prediction_length": max_prediction_length,
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score or 0),
        "best_checkpoint": best_model_path,
    }
    meta_path = os.path.join(model_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Training metadata saved to {meta_path}")

    # Log metrics
    _log_metrics(metadata)

    return best_model, trainer, test_dl, training_ds


def _log_metrics(metadata: dict):
    """Log model metrics to a metrics file."""
    log_path = os.path.join(PROJECT_ROOT, "models", "metrics_log.txt")
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Training Run - {metadata['trained_at']}\n")
        f.write(f"{'='*60}\n")
        for k, v in metadata.items():
            f.write(f"  {k}: {v}\n")
    logger.info(f"Metrics logged to {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TFT Air Quality Model")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--encoder-length", type=int, default=168, help="Encoder sequence length")
    parser.add_argument("--prediction-length", type=int, default=24, help="Prediction horizon")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs")

    args = parser.parse_args()

    model, trainer, test_dl, training_ds = train_model(
        data_path=args.data,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_encoder_length=args.encoder_length,
        max_prediction_length=args.prediction_length,
        gpus=args.gpus,
    )

    print("\n[✓] Training complete!")

