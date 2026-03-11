#!/usr/bin/env python
"""Quick test to verify the PM2.5 -> PM25 fix works."""
import sys
sys.path.insert(0, '.')

from src.data_preprocessing import preprocess_pipeline
from src.train_model import build_dataset

print("Loading data...")
df, scaler = preprocess_pipeline('data/air_quality.csv')
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Has PM2.5 column: {'PM2.5' in df.columns}")
print(f"Has PM25 column: {'PM25' in df.columns}")

print("\nBuilding datasets...")
train, val, test = build_dataset(df)
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

print("\nSUCCESS: Fix verified!")

