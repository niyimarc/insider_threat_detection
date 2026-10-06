"""
One uniform way to get a per-session anomaly score out of ANY registered
model (see src/unsupervised/models/registry.py): a trained PyTorch autoencoder, the
Isolation Forest, or a no-training heuristic baseline.

Scores always come back as a numpy array of length n_sessions, in the
canonical session order (events sorted by SESSION_KEY_COLUMNS, then
grouped) -- the same order SessionDataset and the static-vector baseline
use -- and HIGHER always means MORE anomalous.

Heavy libraries (torch, scikit-learn, joblib) are imported lazily inside
the functions that need them.
"""

import os

import numpy as np
import pandas as pd

from src.common.preprocessing.sessionize import SESSION_KEY_COLUMNS
from src.unsupervised.training.feature_config import verify_feature_config
from src.common.run_config import RUN_METADATA_FILENAME


def session_table(events_df):
    """
    One row per session in canonical order, with the columns user,
    session_start, session_end, n_events and window_label (the
    session_is_malicious value, 0/1).
    """
    ordered = events_df.sort_values(SESSION_KEY_COLUMNS)
    grouped = ordered.groupby(SESSION_KEY_COLUMNS, sort=False)
    table = grouped.size().rename("n_events").reset_index()
    table["window_label"] = (
        grouped["session_is_malicious"].first().astype(int).values
    )
    return table


def load_run_artifacts(run_dir):
    """
    Loads what training saved next to the checkpoints (vocabularies and
    scaler) and checks the run's feature configuration against the
    current settings.
    """
    from src.unsupervised.training.vocab import load_vocab
    from src.unsupervised.training.scaling import load_scaler

    verify_feature_config(run_dir)
    return {
        "file_type_vocab": load_vocab(os.path.join(run_dir, "file_type_vocab.json")),
        "file_extension_vocab": load_vocab(os.path.join(run_dir, "file_extension_vocab.json")),
        "scaler": load_scaler(os.path.join(run_dir, "numeric_scaler.json")),
    }


def run_seed(run_dir):
    """The training seed recorded for a run, or None if it has no metadata."""
    import json
    path = os.path.join(run_dir, RUN_METADATA_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("seed")


def has_checkpoint(spec, run_dir):
    """Whether a run folder contains what this model needs (heuristics need nothing)."""
    if spec.kind == "heuristic":
        return True
    return os.path.exists(os.path.join(run_dir, spec.checkpoint_file))


def score_heuristic(spec, events_df):
    """Session mean of one column, or the session's event count."""
    ordered = events_df.sort_values(SESSION_KEY_COLUMNS)
    grouped = ordered.groupby(SESSION_KEY_COLUMNS, sort=False)
    if spec.heuristic_reduce == "count":
        return grouped.size().to_numpy(dtype=float)
    return grouped[spec.heuristic_column].mean().to_numpy(dtype=float)


def score_sessions(spec, events_df, run_dir=None, artifacts=None, batch_size=64):
    """
    Per-session anomaly scores from one registered model.

    run_dir/artifacts are needed for trained models (torch, static) and
    ignored for heuristics. `artifacts` comes from load_run_artifacts and
    can be reused across models of the same run.
    """
    if spec.kind == "heuristic":
        missing = [c for c in spec.required_columns if c not in events_df.columns]
        if missing:
            raise ValueError(f"{spec.name} needs column(s) {missing}, which the dataset lacks.")
        return score_heuristic(spec, events_df)

    if run_dir is None:
        raise ValueError(f"{spec.name} is a trained model: a run_dir is required.")
    artifacts = artifacts or load_run_artifacts(run_dir)
    checkpoint_path = os.path.join(run_dir, spec.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} not found: train {spec.name} first.")

    file_type_vocab = artifacts["file_type_vocab"]
    file_extension_vocab = artifacts["file_extension_vocab"]

    if spec.kind == "torch":
        import torch
        from torch.utils.data import DataLoader
        from src.unsupervised.training.dataset import SessionDataset, collate_sessions
        from src.unsupervised.evaluation.score import score_pytorch_model

        dataset = SessionDataset(events_df, file_type_vocab, file_extension_vocab, artifacts["scaler"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sessions)
        model = spec.build(len(file_type_vocab), len(file_extension_vocab))
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        scores, _ = score_pytorch_model(model, loader, spec.score_fn)
        return np.asarray(scores, dtype=float)

    if spec.kind == "static":
        import joblib
        from src.unsupervised.models.static_vector_baseline import build_session_vectors

        detector = joblib.load(checkpoint_path)
        vectors, _ = build_session_vectors(events_df, file_type_vocab, file_extension_vocab)
        return np.asarray(detector.anomaly_score(vectors), dtype=float)

    raise ValueError(f"Unsupported model kind {spec.kind!r}")
