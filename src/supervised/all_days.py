"""
The ALL-DAYS user-day table.

Rows: every (user, calendar day) on which the user was active in the logon,
device or file log (the union), not only days with a file copy. Insider
behaviour in r4.2 shows up mostly in device and logon activity, much of it on
days with no file copy, so the first (file-event-days) table could not see it.

Columns:
  * file group     the file-event-day summaries of src/supervised/dataset.py
                   (0 / NaN on days with no file copy) plus `day_has_file_events`
  * device group   those event-attached device features, plus the day-level
                   device signals and their baselines (src/supervised/day_signals.py)
  * logon group    `outside_working_hours__mean`, plus the day-level logon signals
                   and their baselines
  * http group     the three http features of that calendar day (0.0 when the
                   user browsed nothing that day -- the same neutral value as
                   before)
  window_label / answer_key_label   evaluation and training labels, NEVER inputs

Leakage rules are those of the individual pieces: baselines use only each
user's previous active days; the http features use only past visits; no
feature uses a label or the answer key.
"""

import numpy as np
import pandas as pd

from src.common.preprocessing.http_features import HTTP_FEATURE_COLUMNS
from src.supervised.dataset import DAY_KEY, WINDOW_LABEL, build_user_day_table
from src.supervised.day_signals import (
    DEVICE_RAW_COLUMNS,
    LOGON_RAW_COLUMNS,
    add_causal_baselines,
    device_day_counts,
    logon_day_counts,
)

ANSWER_KEY_COLUMN = "answer_key_label"
_LOGON_COUNT_COLUMNS = ("lg_n_logons", "lg_n_distinct_pcs", "lg_after_hours_events", "lg_other_pc_events")


def window_labels_for_days(days, windows):
    """
    0/1 array aligned with `days` (a DataFrame with user and day): 1 if the day
    falls inside the user's insider window (whole calendar days from the
    window's start day to its end day, inclusive). `windows` has user, start, end.
    """
    labels = np.zeros(len(days), dtype=int)
    users = days["user"].to_numpy()
    day_values = pd.to_datetime(days["day"]).to_numpy()
    for _, window in windows.iterrows():
        inside = (
            (users == window["user"])
            & (day_values >= window["start"].normalize().to_datetime64())
            & (day_values <= window["end"].normalize().to_datetime64())
        )
        labels[inside] = 1
    return labels


def build_all_user_days(logon_df, device_df, file_events_df, http_user_day, windows, answer_key_days, users):
    """
    logon_df, device_df: the raw logon/device logs (user, pc, date, activity).
    file_events_df:      processed file events (the registered feature columns, user, date,
                         session_start, is_malicious).
    http_user_day:       output of build_http_user_day_features (user, day, 3 http columns).
    windows:             insider windows (user, start, end).
    answer_key_days:     output of labelings.answer_key_labeled_days (user, day, answer_key_label).
    users:               the users to include.
    Returns a DataFrame sorted by (user, day).
    """
    users = set(users)
    logon_df = logon_df[logon_df["user"].isin(users)]
    device_df = device_df[device_df["user"].isin(users)]
    file_events_df = file_events_df[file_events_df["user"].isin(users)]

    logon_day = logon_day_counts(logon_df)
    device_day = device_day_counts(device_df)
    file_day = build_user_day_table(file_events_df).drop(columns=[WINDOW_LABEL])

    universe = pd.concat(
        [logon_day[DAY_KEY], device_day[DAY_KEY], file_day[DAY_KEY]], ignore_index=True
    ).drop_duplicates()
    table = (
        universe.merge(logon_day, on=DAY_KEY, how="left")
        .merge(device_day, on=DAY_KEY, how="left")
        .merge(file_day, on=DAY_KEY, how="left")
    )

    for column in _LOGON_COUNT_COLUMNS + DEVICE_RAW_COLUMNS:
        table[column] = table[column].fillna(0.0)
    table["lg_is_weekend"] = (table["day"].dt.dayofweek >= 5).astype(float)
    table["day_n_events"] = table["day_n_events"].fillna(0.0)
    table["day_n_sessions"] = table["day_n_sessions"].fillna(0.0)
    table["day_has_file_events"] = (table["day_n_events"] > 0).astype(float)

    http = http_user_day[["user", "day", *HTTP_FEATURE_COLUMNS]]
    table = table.merge(http, on=DAY_KEY, how="left")
    for column in HTTP_FEATURE_COLUMNS:
        table[column] = table[column].fillna(0.0)

    table = add_causal_baselines(table)

    table[WINDOW_LABEL] = window_labels_for_days(table, windows)
    positives = answer_key_days[DAY_KEY + [ANSWER_KEY_COLUMN]]
    table = table.merge(positives, on=DAY_KEY, how="left")
    table[ANSWER_KEY_COLUMN] = table[ANSWER_KEY_COLUMN].fillna(0).astype(int)
    return table.sort_values(DAY_KEY).reset_index(drop=True)
