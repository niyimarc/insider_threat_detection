"""
The USER-DAY table for the supervised detector.

One row per (user, calendar day) on which the user had at least one
file-copy event. (Days with no file event are not rows: this is the file
access log's own view of the user, with the other signals attached to
those days. Say so in the thesis.)

Features: every registered numeric feature (src/common/preprocessing/
feature_groups.py) is summarised over the day's events -- mean and max
(just the mean for 0/1 flags) -- plus the day's event and session counts.
Each column keeps the signal group of the feature it came from, so the
ablation (file only / + device / + logon / + http) is just a choice of
groups, with no environment variable involved.

Labels, kept in separate columns and NEVER used as inputs:
  window_label  1 if any event of the day falls in the user's insider
                time window (the original labelling; the training label)
  answer-key labels are attached by the caller (see
  src/common/evaluation/labelings.answer_key_day_labels).

Honest note on the inputs: the per-user session deviation features
(`user_session_volume_deviation`, `user_session_frequency_deviation`) are
leave-one-out against the SAME user's other sessions anywhere in the
data, so they are not strictly causal (they may use that user's later
sessions). They never use any other user's data or any label, and the
folds are split by user, so there is no cross-user leakage -- but state
this when describing the features. The http features use only past data.
"""

import numpy as np
import pandas as pd

from src.common.preprocessing.feature_groups import (
    BINARY_NUMERIC_COLUMNS,
    NUMERIC_FEATURE_REGISTRY,
    feature_group_of,
    resolve_groups,
)

DAY_KEY = ["user", "day"]
WINDOW_LABEL = "window_label"

# Day-level count features and the signal group they belong to.
COUNT_COLUMNS = (("day_n_events", "file"), ("day_n_sessions", "file"))


def _aggregated_columns_of(column):
    """The day-table columns produced from one registered feature."""
    if column in BINARY_NUMERIC_COLUMNS:
        return [f"{column}__mean"]
    return [f"{column}__mean", f"{column}__max"]


def build_user_day_table(events_df):
    """
    events_df: file-event rows (the processed dataset, or a subset of
    users) with the registered feature columns, `user`, `date`,
    `session_start` and `is_malicious`.
    Returns a DataFrame sorted by (user, day) with columns:
    user, day, window_label, day_n_events, day_n_sessions, and
    `<feature>__mean` / `<feature>__max` for every registered feature.
    """
    needed = [c for c, _ in NUMERIC_FEATURE_REGISTRY] + ["user", "date", "session_start", "is_malicious"]
    missing = [c for c in needed if c not in events_df.columns]
    if missing:
        raise ValueError(
            f"Events are missing column(s) {missing}. If these are feature columns, the dataset "
            "predates them: re-run scripts/common/build_dataset.py."
        )

    work = events_df[needed].copy()
    work["day"] = pd.to_datetime(work["date"]).dt.normalize()
    for column, _ in NUMERIC_FEATURE_REGISTRY:
        work[column] = work[column].astype(float)
    work["is_malicious"] = work["is_malicious"].astype(bool)

    grouped = work.groupby(DAY_KEY, sort=True)
    pieces = {
        "day_n_events": grouped.size(),
        "day_n_sessions": grouped["session_start"].nunique(),
        WINDOW_LABEL: grouped["is_malicious"].any().astype(int),
    }
    for column, _ in NUMERIC_FEATURE_REGISTRY:
        pieces[f"{column}__mean"] = grouped[column].mean()
        if column not in BINARY_NUMERIC_COLUMNS:
            pieces[f"{column}__max"] = grouped[column].max()

    table = pd.DataFrame(pieces).reset_index()
    table["day_n_events"] = table["day_n_events"].astype(float)
    table["day_n_sessions"] = table["day_n_sessions"].astype(float)
    return table


def day_feature_columns(groups):
    """
    Day-table input columns for the given signal groups, in registry
    order (counts first). `groups` is validated like the registry's
    (the mandatory "file" group must be present).
    """
    enabled = resolve_groups(groups)
    columns = [c for c, g in COUNT_COLUMNS if g in enabled]
    for column, group in NUMERIC_FEATURE_REGISTRY:
        if group in enabled:
            columns += _aggregated_columns_of(column)
    return columns


def group_of_day_column(column):
    """The signal group a day-table input column belongs to."""
    for name, group in COUNT_COLUMNS:
        if column == name:
            return group
    base = column.rsplit("__", 1)[0]
    return feature_group_of(base)


def attach_labels(day_table, answer_key_days=None):
    """
    Adds `answer_key_label` (from answer_key_day_labels) to the day
    table by (user, day). Days missing from the answer-key frame get 0.
    """
    table = day_table.copy()
    if answer_key_days is None:
        table["answer_key_label"] = 0
        return table
    merged = table.merge(answer_key_days, on=DAY_KEY, how="left")
    merged["answer_key_label"] = merged["answer_key_label"].fillna(0).astype(int)
    return merged


def feature_matrix(day_table, columns):
    """Float matrix of the chosen columns (NaN kept: models decide how to impute)."""
    return day_table[list(columns)].to_numpy(dtype=float, copy=True)
