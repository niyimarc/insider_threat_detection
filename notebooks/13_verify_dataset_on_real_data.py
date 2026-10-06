import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.unsupervised.training.vocab import build_vocab
from src.unsupervised.training.dataset import SessionDataset, collate_sessions

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")


def main():
    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    print(f"{len(events_df)} events loaded\n")

    train_df = events_df[events_df["split"] == "train"]
    val_df = events_df[events_df["split"] == "validation"]
    test_df = events_df[events_df["split"] == "test"]

    # Vocabs are built ONLY from train, then reused for val/test -- this
    # is the real-data equivalent of what test_encode_falls_back_to_unknown
    # verified on toy data.
    file_type_vocab = build_vocab(train_df["file_type"])
    file_extension_vocab = build_vocab(train_df["file_extension"])
    print(f"file_type_vocab ({len(file_type_vocab)} entries): {file_type_vocab}")
    print(f"file_extension_vocab: {len(file_extension_vocab)} entries "
          f"(includes the reserved <unknown> slot)\n")

    train_dataset = SessionDataset(train_df, file_type_vocab, file_extension_vocab)
    val_dataset = SessionDataset(val_df, file_type_vocab, file_extension_vocab)
    test_dataset = SessionDataset(test_df, file_type_vocab, file_extension_vocab)

    for name, dataset in [("train", train_dataset), ("validation", val_dataset), ("test", test_dataset)]:
        labels = torch.tensor([dataset[i]["label"] for i in range(len(dataset))])
        lengths = [dataset[i]["numeric"].shape[0] for i in range(len(dataset))]
        print(
            f"{name:10s}: {len(dataset):6d} sessions | "
            f"malicious rate {labels.mean().item():.4f} | "
            f"session length min/mean/max = {min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}"
        )

    print("\nPulling one real batch through DataLoader + collate_sessions ...")
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_sessions)
    batch = next(iter(loader))
    for key, value in batch.items():
        print(f"  {key:14s} shape={tuple(value.shape)} dtype={value.dtype}")

    assert not torch.isnan(batch["numeric"]).any(), "NaNs found in numeric features!"
    assert batch["mask"].dtype == torch.bool
    print("\nNo NaNs in numeric features. Smoke test passed.")


if __name__ == "__main__":
    main()
