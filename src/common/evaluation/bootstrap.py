"""
User-level bootstrap for evaluation metrics.

Sessions of the same user are not independent (the same person, the
same habits), so resampling SESSIONS would give intervals that are far
too narrow. Here whole USERS are resampled with replacement, and every
session of a sampled user comes along. With only ~35 insiders on the
development partition this is the interval that reflects how much the
result depends on which insiders happened to be in the sample.

Several models, and several training seeds of one model, are evaluated on
the SAME resamples, which allows paired comparisons ("does model A beat
the single-feature baseline?") with an interval on the difference.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auroc(labels, scores):
    """AUROC, or NaN when only one class (or constant scores) is present."""
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels, scores):
    labels = np.asarray(labels)
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


METRICS = {"auroc": safe_auroc, "auprc": safe_auprc}


def make_user_resamples(session_users, n_boot=500, seed=0):
    """
    n_boot index arrays; each is the concatenated session indices of a
    sample of users drawn with replacement (a user drawn twice
    contributes its sessions twice).
    """
    users = np.asarray(session_users)
    unique_users = np.unique(users)
    index_by_user = {u: np.flatnonzero(users == u) for u in unique_users}
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n_boot):
        sample = rng.choice(unique_users, size=len(unique_users), replace=True)
        resamples.append(np.concatenate([index_by_user[u] for u in sample]))
    return resamples


def metric_over_resamples(labels, scores, resamples, metric):
    """
    Value of `metric` on every resample. `scores` is either a 1-D array
    (one model run) or a 2-D array (n_runs, n_sessions), in which case
    the metric is averaged over the runs within each resample (the
    seed-mean). Resamples with only one class give NaN.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    runs = scores[None, :] if scores.ndim == 1 else scores
    values = np.full(len(resamples), np.nan)
    for i, idx in enumerate(resamples):
        sample_labels = labels[idx]
        per_run = [metric(sample_labels, run[idx]) for run in runs]
        values[i] = np.mean(per_run)
    return values


def interval(values, level=95):
    """(low, high) percentile interval, ignoring NaN resamples."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    tail = (100 - level) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def paired_difference(values_a, values_b):
    """Per-resample difference a - b (NaN where either is NaN)."""
    return np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
