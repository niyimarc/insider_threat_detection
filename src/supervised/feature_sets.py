"""
A FEATURE SET tells the supervised experiment which columns of a day table
are inputs, which signal group each belongs to, which no-training
heuristics to report, and how to describe the labels. Two are defined:

  FILE_EVENT_DAYS   the first table: only days with a file copy, event features
                    summarised over each day (src/supervised/dataset.py)
  ALL_DAYS          every day the user was active in any log (logon, device or
                    file), with day-level logon and device signals and
                    per-user baselines (src/supervised/all_days.py)

The signal-group names are the same ("file", "device", "logon", "http"), so
the same ablation (file; + device; + logon; + http) applies to both.
The heuristics are fixed in advance, from the dataset's readme and
scenarios.txt, not chosen from results.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

from src.common.preprocessing.feature_groups import resolve_groups
from src.supervised.dataset import day_feature_columns, group_of_day_column
from src.supervised.day_signals import DEVICE_DAY_COLUMNS, LOGON_DAY_COLUMNS

_FILE_EVENT_WINDOW_DESCRIPTION = (
    "A user-day is malicious if any of its file events falls inside the user's insider time "
    "window. This is the TRAINING label of the supervised models."
)
_FILE_EVENT_ANSWER_KEY_DESCRIPTION = (
    "A user-day is malicious if it has a file event on a day with an answer-key http event "
    "for that insider, or during an answer-key removable-device session. Evaluation only: the "
    "supervised models were NOT trained on it. It is built from the same kind of events the "
    "http and device features measure, so it favours those features by construction."
)
_ALL_DAYS_WINDOW_DESCRIPTION = (
    "A user-day is malicious if the day falls inside the user's insider time window (whether or "
    "not the user copied a file that day). This is the TRAINING label of the supervised models."
)
_ALL_DAYS_ANSWER_KEY_DESCRIPTION = (
    "A user-day is malicious if the answer key lists an injected http event for that insider on that "
    "day, or a removable-device session of that insider covers the day. Evaluation only: the "
    "supervised models were NOT trained on it. It is built from the same kind of events the "
    "http and device features measure, so it favours those features by construction."
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns_fn: Callable            # (groups) -> list of input column names
    group_fn: Callable              # (column) -> signal group
    heuristics: Dict[str, Tuple[str, str]]   # key -> (column, display name); higher = more anomalous
    window_description: str
    answer_key_description: str
    unit_label: str = "user-days"


# ---- first table: file-event days -------------------------------------------------
FILE_EVENT_DAYS = FeatureSet(
    name="file_event_days",
    columns_fn=day_feature_columns,
    group_fn=group_of_day_column,
    heuristics={
        "single_feature_http": ("http_offbaseline_share_7d__mean", "Heuristic: day mean of http_offbaseline_share_7d"),
        "day_event_count": ("day_n_events", "Heuristic: events in the day (more = more anomalous)"),
    },
    window_description=_FILE_EVENT_WINDOW_DESCRIPTION,
    answer_key_description=_FILE_EVENT_ANSWER_KEY_DESCRIPTION,
)

# ---- all-days table ---------------------------------------------------------------
_ALL_DAYS_FILE = (
    "day_n_events", "day_n_sessions", "day_has_file_events",
    "seconds_since_last_event__mean", "seconds_since_last_event__max",
    "user_session_volume_deviation__mean", "user_session_volume_deviation__max",
    "user_session_frequency_deviation__mean", "user_session_frequency_deviation__max",
)
_ALL_DAYS_DEVICE = (
    "seconds_since_device_connect__mean", "seconds_since_device_connect__max",
    "copies_since_device_connect__mean", "copies_since_device_connect__max",
) + DEVICE_DAY_COLUMNS
_ALL_DAYS_LOGON = ("outside_working_hours__mean",) + LOGON_DAY_COLUMNS
_ALL_DAYS_HTTP = ("http_new_domains_today", "http_new_domains_7d", "http_offbaseline_share_7d")

ALL_DAYS_COLUMNS_BY_GROUP = {
    "file": _ALL_DAYS_FILE,
    "device": _ALL_DAYS_DEVICE,
    "logon": _ALL_DAYS_LOGON,
    "http": _ALL_DAYS_HTTP,
}


def all_days_feature_columns(groups):
    """Input columns of the all-days table for the given signal groups (the file group is mandatory)."""
    enabled = resolve_groups(groups)
    columns = []
    for group in ("file", "device", "logon", "http"):
        if group in enabled:
            columns += list(ALL_DAYS_COLUMNS_BY_GROUP[group])
    return columns


def all_days_group_of_column(column):
    for group, columns in ALL_DAYS_COLUMNS_BY_GROUP.items():
        if column in columns:
            return group
    raise KeyError(f"'{column}' is not an input column of the all-days table.")


def without_columns(feature_set, excluded):
    """
    The same feature set with some input columns removed (an ablation). Heuristics are unchanged.
    Unknown names are rejected, so a typo cannot silently ablate nothing.
    """
    excluded = tuple(excluded)
    all_columns = set(feature_set.columns_fn(["file", "device", "logon", "http"]))
    unknown = [c for c in excluded if c not in all_columns]
    if unknown:
        raise ValueError(f"Not input columns of the {feature_set.name} feature set: {unknown}")
    return FeatureSet(
        name=f"{feature_set.name}-without-{'-'.join(excluded)}" if excluded else feature_set.name,
        columns_fn=lambda groups: [c for c in feature_set.columns_fn(groups) if c not in excluded],
        group_fn=feature_set.group_fn,
        heuristics=feature_set.heuristics,
        window_description=feature_set.window_description,
        answer_key_description=feature_set.answer_key_description,
        unit_label=feature_set.unit_label,
    )


ALL_DAYS = FeatureSet(
    name="all_days",
    columns_fn=all_days_feature_columns,
    group_fn=all_days_group_of_column,
    heuristics={
        "single_feature_http": ("http_offbaseline_share_7d", "Heuristic: day value of http_offbaseline_share_7d"),
        "day_event_count": ("day_n_events", "Heuristic: file copies in the day (more = more anomalous)"),
        "device_rate": ("dv_n_connects_logratio",
                        "Heuristic: removable-device connects vs the user's own recent rate"),
        "after_hours_rate": ("lg_after_hours_logratio",
                             "Heuristic: after-hours logon events vs the user's own recent rate"),
    },
    window_description=_ALL_DAYS_WINDOW_DESCRIPTION,
    answer_key_description=_ALL_DAYS_ANSWER_KEY_DESCRIPTION,
)
