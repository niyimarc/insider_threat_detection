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
from src.training.scaling import build_scaler, save_scaler
from src.training.dataset import SessionDataset, collate_sessions, NUMERIC_FEATURE_COLUMNS
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
# Raised from 5: the LSTM baseline's val loss was still trending down
# (0.2601 -> 0.1788 in its last ~10 epochs) when patience=5 cut it off,
# suggesting it hadn't actually plateaued yet -- a more generous budget
# lets both models run closer to their real best-val-loss epoch before
# stopping, rather than stopping on a short run of noisy non-improving
# epochs.
PATIENCE = 10

# outside_working_hours is already 0/1 -- no need to standardize it like
# the genuinely continuous, wildly-scaled columns.
SCALED_COLUMNS = [c for c in NUMERIC_FEATURE_COLUMNS if c != "outside_working_hours"]

# Trains on the GPU when one is available (e.g. a Colab GPU runtime) and
# falls back to CPU otherwise, so the same script works on both.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cpu_state_dict(model):
    """Checkpoints are always saved with CPU tensors, so they load fine on
    a CPU-only machine (the evaluation scripts) regardless of where they
    were trained -- a state dict saved from GPU tensors fails to load on
    a machine without CUDA unless map_location is passed everywhere."""
    return {key: value.cpu() for key, value in model.state_dict().items()}


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Training device: {DEVICE}")
    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    train_df = events_df[events_df["split"] == "train"]
    val_df = events_df[events_df["split"] == "validation"]

    print("Building vocabularies from TRAIN split only ...")
    file_type_vocab = build_vocab(train_df["file_type"])
    file_extension_vocab = build_vocab(train_df["file_extension"])
    save_vocab(file_type_vocab, os.path.join(CHECKPOINT_DIR, "file_type_vocab.json"))
    save_vocab(file_extension_vocab, os.path.join(CHECKPOINT_DIR, "file_extension_vocab.json"))

    print("Building numeric feature scaler from TRAIN split only ...")
    scaler = build_scaler(train_df, SCALED_COLUMNS)
    save_scaler(scaler, os.path.join(CHECKPOINT_DIR, "numeric_scaler.json"))
    print(scaler)

    print("Building datasets and loaders ...")
    train_dataset = SessionDataset(train_df, file_type_vocab, file_extension_vocab, scaler)
    val_dataset = SessionDataset(val_df, file_type_vocab, file_extension_vocab, scaler)
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
        num_epochs=NUM_EPOCHS, patience=PATIENCE, device=DEVICE,
    )
    torch.save(
        _cpu_state_dict(transformer_model),
        os.path.join(CHECKPOINT_DIR, "transformer_autoencoder.pt"),
    )
    with open(os.path.join(CHECKPOINT_DIR, "transformer_autoencoder_history.json"), "w") as f:
        json.dump(transformer_history, f, indent=2)

    # --- Baseline 1: LSTM autoencoder ---
    print("\n=== Training baseline 1 (LSTM autoencoder) ===")
    # hidden_dim raised from the default 64 to 128: scripts/tune_models.py's
    # comparison showed this config reaching a best val loss of 0.1196 vs.
    # 0.2068 for hidden_dim=64 (same lr, same epoch budget) -- a clear,
    # consistent win at every epoch along the way, not just at the end.
    lstm_model = LSTMAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
        hidden_dim=128,
    )
    lstm_model, lstm_history = train_model(
        lstm_model, train_loader, val_loader, lstm_loss,
        num_epochs=NUM_EPOCHS, patience=PATIENCE, device=DEVICE,
    )
    torch.save(_cpu_state_dict(lstm_model), os.path.join(CHECKPOINT_DIR, "lstm_autoencoder.pt"))
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