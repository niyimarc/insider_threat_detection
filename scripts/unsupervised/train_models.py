"""
Trains every registered model (src/unsupervised/models/registry.py) on the TRAIN
split and saves the checkpoints, vocabularies, scaler, feature
configuration and run metadata into one run folder.

    INSIDER_CHECKPOINT_DIR   run folder (default "checkpoints")
    INSIDER_SEED             training seed (default 42)
    INSIDER_FEATURE_GROUPS   feature groups to use (default: all)
    INSIDER_MODELS           comma-separated subset of models (default: all)
    INSIDER_REQUIRE_GPU=1    refuse to start on a CPU-only runtime

Example:
    INSIDER_CHECKPOINT_DIR=checkpoints/all_seed1 INSIDER_SEED=1 python scripts/unsupervised/train_models.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.unsupervised.models.registry import get_model_specs
from src.unsupervised.training.vocab import build_vocab, save_vocab
from src.unsupervised.training.scaling import build_scaler, save_scaler
from src.unsupervised.training.dataset import SessionDataset, collate_sessions
from src.unsupervised.training.feature_config import save_feature_config
from src.common.run_config import get_checkpoint_dir, get_seed, set_global_seed, save_run_metadata
from src.unsupervised.training.train_registry import train_registered_model
from src.common.preprocessing.feature_groups import get_scaled_columns

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
# Folder and seed come from INSIDER_CHECKPOINT_DIR / INSIDER_SEED (see
# src/common/run_config.py) so ablations and repeated runs don't
# overwrite each other.
CHECKPOINT_DIR = get_checkpoint_dir()
SEED = get_seed()

BATCH_SIZE = 32
NUM_EPOCHS = 100
# Raised from 5: the LSTM baseline's val loss was still trending down when
# patience=5 cut it off, suggesting it hadn't actually plateaued yet.
PATIENCE = 10

# 0/1 flag columns (see feature_groups.BINARY_NUMERIC_COLUMNS) are not
# standardised -- only the genuinely continuous, wildly-scaled columns are.
SCALED_COLUMNS = get_scaled_columns()

# Trains on the GPU when one is available (e.g. a Colab GPU runtime) and
# falls back to CPU otherwise, so the same script works on both.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set INSIDER_REQUIRE_GPU=1 (the Colab commands do) to refuse to start a
# multi-hour run on the CPU by accident -- e.g. after a Colab runtime
# reset silently went back to a CPU runtime.
REQUIRE_GPU = os.environ.get("INSIDER_REQUIRE_GPU", "").strip() not in ("", "0")


def main():
    if REQUIRE_GPU and DEVICE != "cuda":
        raise SystemExit(
            "INSIDER_REQUIRE_GPU is set but no GPU is available (torch.cuda.is_available() "
            "is False). On Colab: Runtime -> Change runtime type -> T4 GPU, then "
            "Runtime -> Disconnect and delete runtime, reconnect, and re-run the setup cell."
        )
    specs = get_model_specs()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_global_seed(SEED)

    print(f"Training device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Checkpoint folder: {CHECKPOINT_DIR}")
    print(f"Models: {[s.name for s in specs]}")
    print(f"Run metadata saved to {save_run_metadata(CHECKPOINT_DIR, SEED, DEVICE)}")
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

    # Record which feature groups/columns these checkpoints are trained
    # with, so evaluation can verify it is using the same configuration.
    feature_config_path = save_feature_config(CHECKPOINT_DIR)
    print(f"Feature configuration saved to {feature_config_path}")

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

    for spec in specs:
        print(f"\n=== {spec.display_name} [{spec.name}] ===")
        # Re-seed before every model, so a model's result does not depend
        # on which other models were trained before it in this run.
        set_global_seed(SEED)
        train_registered_model(
            spec,
            train_df=train_df, train_loader=train_loader, val_loader=val_loader,
            file_type_vocab=file_type_vocab, file_extension_vocab=file_extension_vocab,
            checkpoint_dir=CHECKPOINT_DIR, seed=SEED, device=DEVICE,
            num_epochs=NUM_EPOCHS, patience=PATIENCE,
        )

    print(f"\nAll models trained and saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
