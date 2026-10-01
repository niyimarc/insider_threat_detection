"""
Runs a trained model over a set of sessions and returns one raw
anomaly score per session, plus the matching true labels -- the
shared first step before calibration (percentile_calibrate) and
metrics (compute_threshold_metrics / compute_threshold_free_metrics)
in src/evaluation/metrics.py.
"""

import torch


def score_pytorch_model(model, loader, score_fn):
    """
    Runs `model` over every batch in `loader` (a DataLoader built with
    shuffle=False, so order matches the underlying SessionDataset's
    session_users list) and returns (scores, labels) as flat Python
    lists in that same order.

    score_fn is the model-specific session_anomaly_score function (see
    src/models/transformer_autoencoder.py or lstm_autoencoder.py).
    """
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            scores = score_fn(outputs, batch)
            all_scores.extend(scores.tolist())
            all_labels.extend(batch["label"].tolist())
    return all_scores, all_labels


def score_static_model(detector, vectors):
    """Runs the Isolation Forest baseline over a set of pre-built
    session vectors (see src/models/static_vector_baseline.py)."""
    return detector.anomaly_score(vectors).tolist()