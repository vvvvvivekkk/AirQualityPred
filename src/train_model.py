"""Train Temporal Fusion Transformer for air quality forecasting."""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import preprocess_pipeline


def build_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 168,
    max_prediction_length: int = 24,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """Create train/validation/test TimeSeriesDataSet objects."""
    n = int(df["time_idx"].max()) + 1
    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))

    logger.info(
        "Dataset split - Train: 0-%s, Val: %s-%s, Test: %s-%s",
        train_cutoff,
        train_cutoff,
        val_cutoff,
        val_cutoff,
        n,
    )

    known_reals = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
    unknown_reals = ["PM25", "PM10", "NO2", "CO", "SO2", "temperature", "humidity", "wind_speed"]

    available = set(df.columns)
    known_reals = [c for c in known_reals if c in available]
    unknown_reals = [c for c in unknown_reals if c in available]

    work_df = df.copy()
    work_df["group_id"] = work_df["group_id"].astype(str)
    work_df["is_weekend"] = work_df["is_weekend"].astype(str)
    work_df["time_idx"] = work_df["time_idx"].astype(int)

    training = TimeSeriesDataSet(
        work_df[work_df.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="PM25",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_prediction_length=max_prediction_length,
        static_categoricals=["group_id"],
        time_varying_known_categoricals=["is_weekend"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        work_df[(work_df.time_idx > train_cutoff) & (work_df.time_idx <= val_cutoff)],
        predict=True,
        stop_randomization=True,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        work_df[work_df.time_idx > val_cutoff],
        predict=True,
        stop_randomization=True,
    )

    logger.info("Samples - Train: %d, Validation: %d, Test: %d", len(training), len(validation), len(test))
    return training, validation, test


def create_dataloaders(training, validation, test, batch_size: int = 64):
    train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_dl = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    return train_dl, val_dl, test_dl


def build_tft_model(training: TimeSeriesDataSet, learning_rate: float = 1e-3) -> TemporalFusionTransformer:
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    logger.info("TFT model created - Parameters: %.1fK", model.size() / 1e3)
    return model


def train_model(
    data_path: str = None,
    max_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_encoder_length: int = 168,
    max_prediction_length: int = 24,
    gpus: int = 0,
):
    """Run full training pipeline and persist artifacts."""
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "air_quality.csv")

    df, _ = preprocess_pipeline(data_path, normalize=True)

    training_ds, validation_ds, test_ds = build_dataset(
        df,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )
    train_dl, val_dl, test_dl = create_dataloaders(training_ds, validation_ds, test_ds, batch_size=batch_size)

    model = build_tft_model(training_ds, learning_rate=learning_rate)

    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(
            dirpath=model_dir,
            filename="tft_best_{epoch}_{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    accelerator = "gpu" if gpus > 0 and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    logger.info("%s", "=" * 60)
    logger.info("STARTING TFT TRAINING")
    logger.info(
        "Epochs: %d, Batch: %d, LR: %.6f, Encoder: %d, Horizon: %d",
        max_epochs,
        batch_size,
        learning_rate,
        max_encoder_length,
        max_prediction_length,
    )
    logger.info("%s", "=" * 60)

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path) if best_model_path else model

    # Save files for inference/recovery
    torch.save(best_model.state_dict(), os.path.join(model_dir, "tft_model.pth"))

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "encoder_length": max_encoder_length,
        "prediction_length": max_prediction_length,
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score or 0.0),
        "best_checkpoint": best_model_path,
    }

    with open(os.path.join(model_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(model_dir, "metrics_log.txt"), "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Training Run - {metadata['trained_at']}\n")
        for key, value in metadata.items():
            f.write(f"  {key}: {value}\n")

    logger.info("Training artifacts saved in %s", model_dir)
    return best_model, trainer, test_dl, training_ds


def main():
    parser = argparse.ArgumentParser(description="Train TFT Air Quality Model")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-length", type=int, default=168)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_encoder_length=args.encoder_length,
        max_prediction_length=args.prediction_length,
        gpus=args.gpus,
    )
    print("\n[OK] Training complete")


if __name__ == "__main__":
    main()
