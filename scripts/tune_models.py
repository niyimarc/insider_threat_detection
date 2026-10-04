"""
Small hyperparameter comparison for the two PyTorch models, selected
PURELY by validation reconstruction loss -- never by test-set detection
metrics (AUROC/AUPRC/recall). This matters for leakage: every malicious
user is routed to the test split by design (src/preprocessing/split.py),
so validation is benign-only and has no detection signal to tune
against at all; picking a config because it scored better on TEST would
mean the reported test numbers are no longer an honest held-out
evaluation (Chapter 3.6's leakage-avoidance design would be violated).
Val loss is the only safe proxy available during model selection.

This does NOT overwrite checkpoints/*.pt -- it's purely informational.
Once you see which config wins, update scripts/train_models.py's model
constructor arguments yourself and re-run it to produce the real
checkpoint.

Trains each config with a smaller epoch/patience budget than the real
training run, just to rank configs quickly -- the winning config should
still be retrained with the full budget (see train_models.py) for the
final checkpoint.

Run from the project root:
    python scripts/tune_models.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.vocab import build_vocab
from src.training.scaling import build_scaler
from src.training.dataset import SessionDataset, collate_sessions
from src.preprocessing.feature_groups import get_scaled_columns
from src.training.train import train_model
from src.models.transformer_autoencoder import DualChannelTransformerAutoencoder, reconstruction_loss as transformer_loss
from src.models.lstm_autoencoder import LSTMAutoencoder, reconstruction_loss as lstm_loss

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
BATCH_SIZE = 32
SWEEP_NUM_EPOCHS = 40
SWEEP_PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALED_COLUMNS = get_scaled_columns()

LSTM_CONFIGS = [
    {"hidden_dim": 64, "dropout": 0.0, "lr": 1e-3},   # current default, for comparison
    {"hidden_dim": 128, "dropout": 0.0, "lr": 1e-3},  # more capacity
    {"hidden_dim": 64, "dropout": 0.2, "lr": 1e-3},   # regularization
    {"hidden_dim": 128, "dropout": 0.0, "lr": 5e-4},  # more capacity, gentler lr
]

TRANSFORMER_CONFIGS = [
    {"presence_dim": 16, "magnitude_dim": 48, "nhead": 4, "num_layers": 2, "dropout": 0.1, "lr": 1e-3},  # current default
    {"presence_dim": 32, "magnitude_dim": 96, "nhead": 4, "num_layers": 2, "dropout": 0.1, "lr": 1e-3},  # more capacity, same ratio
    {"presence_dim": 16, "magnitude_dim": 48, "nhead": 4, "num_layers": 3, "dropout": 0.1, "lr": 1e-3},  # deeper
    {"presence_dim": 16, "magnitude_dim": 48, "nhead": 4, "num_layers": 2, "dropout": 0.3, "lr": 1e-3},  # more regularization
]


def main():
    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    train_df = events_df[events_df["split"] == "train"]
    val_df = events_df[events_df["split"] == "validation"]

    file_type_vocab = build_vocab(train_df["file_type"])
    file_extension_vocab = build_vocab(train_df["file_extension"])
    scaler = build_scaler(train_df, SCALED_COLUMNS)

    train_dataset = SessionDataset(train_df, file_type_vocab, file_extension_vocab, scaler)
    val_dataset = SessionDataset(val_df, file_type_vocab, file_extension_vocab, scaler)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sessions)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sessions)
    print(f"{len(train_dataset)} train sessions, {len(val_dataset)} validation sessions\n")

    print("=== LSTM autoencoder: hyperparameter comparison (ranked by best val loss) ===")
    lstm_results = []
    for config in LSTM_CONFIGS:
        print(f"\n--- config: {config} ---")
        model = LSTMAutoencoder(
            file_type_vocab_size=len(file_type_vocab),
            file_extension_vocab_size=len(file_extension_vocab),
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        )
        _, history = train_model(
            model, train_loader, val_loader, lstm_loss,
            num_epochs=SWEEP_NUM_EPOCHS, patience=SWEEP_PATIENCE, lr=config["lr"],
            device=DEVICE,
        )
        lstm_results.append((config, history["best_val_loss"]))

    print("\n--- LSTM ranking ---")
    for config, best_val_loss in sorted(lstm_results, key=lambda r: r[1]):
        print(f"  best_val_loss={best_val_loss:.4f}  {config}")

    print("\n=== Transformer autoencoder: hyperparameter comparison (ranked by best val loss) ===")
    transformer_results = []
    for config in TRANSFORMER_CONFIGS:
        print(f"\n--- config: {config} ---")
        model = DualChannelTransformerAutoencoder(
            file_type_vocab_size=len(file_type_vocab),
            file_extension_vocab_size=len(file_extension_vocab),
            presence_dim=config["presence_dim"],
            magnitude_dim=config["magnitude_dim"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        _, history = train_model(
            model, train_loader, val_loader, transformer_loss,
            num_epochs=SWEEP_NUM_EPOCHS, patience=SWEEP_PATIENCE, lr=config["lr"],
            device=DEVICE,
        )
        transformer_results.append((config, history["best_val_loss"]))

    print("\n--- Transformer ranking ---")
    for config, best_val_loss in sorted(transformer_results, key=lambda r: r[1]):
        print(f"  best_val_loss={best_val_loss:.4f}  {config}")


if __name__ == "__main__":
    main()
