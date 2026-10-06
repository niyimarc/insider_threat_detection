"""
Standardizes numeric features (z-score: (x - mean) / std) so that
features on wildly different scales -- e.g. seconds_since_device_connect
can be tens of thousands, while copies_since_device_connect is a small
integer count -- don't dominate the MSE reconstruction loss purely by
virtue of their raw magnitude, independent of how meaningful their
deviation actually is.

Following the exact same principle as src/unsupervised/training/vocab.py: statistics
are computed ONLY from the training split, then reused unchanged for
validation/test, so a session's normalized value means the same thing
regardless of which split it came from.
"""

import json


def build_scaler(df, columns):
    """
    Computes per-column (mean, std) from df[columns] -- fit on the
    TRAINING split only. Returns {column: {"mean": ..., "std": ...}}.
    A std of 0 (a constant column) is replaced with 1 to avoid
    dividing by zero -- the column just keeps its constant, unscaled
    value in that case, rather than blowing up.
    """
    scaler = {}
    for column in columns:
        values = df[column].astype(float)
        mean = float(values.mean())
        std = float(values.std())
        scaler[column] = {"mean": mean, "std": std if std > 0 else 1.0}
    return scaler


def apply_scaler(series, column, scaler):
    """Applies a previously-built scaler to one column's values."""
    stats = scaler[column]
    return (series.astype(float) - stats["mean"]) / stats["std"]


def save_scaler(scaler, path):
    with open(path, "w") as f:
        json.dump(scaler, f, indent=2)


def load_scaler(path):
    with open(path) as f:
        return json.load(f)