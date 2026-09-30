import os
import sys
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader

from src.training.vocab import build_vocab, save_vocab
from src.training.dataset import SessionDataset, collate_sessions
from src.training.train import train_model
from src.models.transformer_autoencoder import (
    DualChannelTransformerAutoencoder,
    reconstruction_loss as transformer_loss,
)
from src.models.lstm_autoencoder import LSTMAutoencoder, reconstruction_loss as lstm_loss
from src.models.static_vector_baseline import build_session_vectors, StaticVectorAnomalyDetector

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")

BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 5


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    train_df = events_df[events_df["split"] == "train"]
    val_df = events_df[events_df["split"] == "validation"]

    # Per Chapter 3.4, training is benign-only by construction of the
    # split (see notebooks/13_verify_dataset_on_real_data.py -- train
    # and validation both showed a 0.0000 malicious rate already), so
    # no additional filtering is needed here.

    print("Building vocabularies from TRAIN split only ...")
    file_type_vocab = build_vocab(train_df["file_type"])
    file_extension_vocab = build_vocab(train_df["file_extension"])
    save_vocab(file_type_vocab, os.path.join(CHECKPOINT_DIR, "file_type_vocab.json"))
    save_vocab(file_extension_vocab, os.path.join(CHECKPOINT_DIR, "file_extension_vocab.json"))

    print("Building datasets and loaders ...")
    train_dataset = SessionDataset(train_df, file_type_vocab, file_extension_vocab)
    val_dataset = SessionDataset(val_df, file_type_vocab, file_extension_vocab)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sessions
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sessions
    )
    print(f"{len(train_dataset)} train sessions, {len(val_dataset)} validation sessions")

    # --- Proposed model: dual-channel Transformer autoencoder ---
    print("\n=== Training proposed model (Transformer autoencoder) ===")
    transformer_model = DualChannelTransformerAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
    )
    transformer_model, transformer_history = train_model(
        transformer_model, train_loader, val_loader, transformer_loss,
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
    )
    torch.save(
        transformer_model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "transformer_autoencoder.pt"),
    )
    with open(os.path.join(CHECKPOINT_DIR, "transformer_autoencoder_history.json"), "w") as f:
        json.dump(transformer_history, f, indent=2)

    # --- Baseline 1: LSTM autoencoder ---
    print("\n=== Training baseline 1 (LSTM autoencoder) ===")
    lstm_model = LSTMAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
    )
    lstm_model, lstm_history = train_model(
        lstm_model, train_loader, val_loader, lstm_loss,
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
    )
    torch.save(lstm_model.state_dict(), os.path.join(CHECKPOINT_DIR, "lstm_autoencoder.pt"))
    with open(os.path.join(CHECKPOINT_DIR, "lstm_autoencoder_history.json"), "w") as f:
        json.dump(lstm_history, f, indent=2)

    # --- Baseline 2: static session-vector Isolation Forest ---
    print("\n=== Fitting baseline 2 (static-vector Isolation Forest) ===")
    train_vectors, _ = build_session_vectors(train_df, file_type_vocab, file_extension_vocab)
    static_detector = StaticVectorAnomalyDetector()
    static_detector.fit(train_vectors)
    joblib.dump(static_detector, os.path.join(CHECKPOINT_DIR, "static_vector_detector.joblib"))

    print(f"\nAll three models trained and saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()