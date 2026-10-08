"""
Turns a user-day table into WINDOW matrices for the sequence model.

Row i of the result describes day i from the user's own history: its
SEQ_LEN most recent active days, the day itself last. Each step carries the
day's input columns and a 0/1 mask channel (0 = padding, because the user
has fewer than SEQ_LEN earlier rows). The window is flattened to
(n, SEQ_LEN * (d + 1)) so the whole cross-validation harness (user-level
folds, sample weights, subsetting rows) works on it unchanged.

CAUSAL by construction: a window only ever reads rows of the SAME user that
come at or before the day being scored, never a later day and never another
user. Because every window stays inside one user, user-level folds cannot
leak through it.
"""

import numpy as np
import pandas as pd

from src.supervised.dataset import feature_matrix
from src.supervised.models import SEQ_LEN


def build_sequence_matrix(table, columns, seq_len=SEQ_LEN):
    """(n, seq_len * (len(columns) + 1)) window matrix, rows in the table's own order."""
    x = feature_matrix(table, columns)
    n, d = x.shape
    users = table["user"].to_numpy()
    days = pd.to_datetime(table["day"]).to_numpy()

    frame = pd.DataFrame({"user": users, "day": days})
    order = frame.sort_values(["user", "day"], kind="mergesort").index.to_numpy()   # positions
    sorted_users = users[order]
    rank = pd.Series(np.arange(n)).groupby(sorted_users).cumcount().to_numpy()        # earlier rows of the user
    xs = x[order]
    position = np.arange(n)

    window = np.full((n, seq_len, d + 1), np.nan)
    window[:, :, d] = 0.0
    for offset in range(seq_len):
        has_row = rank >= offset                      # the row `offset` places earlier is the same user's
        step = seq_len - 1 - offset
        window[has_row, step, :d] = xs[position[has_row] - offset]
        window[has_row, step, d] = 1.0

    result = np.empty_like(window)
    result[order] = window                            # row order[k] of the table gets window[k]
    return result.reshape(n, seq_len * (d + 1))
