"""
EMAIL signals for the all-days supervised table.

The raw per user-day counts come from src/common/preprocessing/email_features.py (one
chronological pass over email.csv, envelope only). This module joins them onto the table
and adds the same kind of causal per-user baselines as the logon and device signals:
log-ratio of today against the same user's PREVIOUS active days (trailing window, today
excluded, NaN until MIN_HISTORY previous days), never z-scores.

The list is fixed in advance from the dataset readme and scenarios.txt (scenario 3's mass
email, uploads to outside addresses): no answer-key event and no result was used to choose
or tune it. Days without email get zeros, the same neutral value used for http.
"""

import numpy as np

from src.common.preprocessing.email_features import EMAIL_RAW_COLUMNS
from src.supervised.day_signals import _trailing_mean_of_previous

EMAIL_DERIVED_COLUMNS = ("em_new_recipient_share",)
_BASELINES = {                      # new column -> raw column it compares with its own recent past
    "em_n_sent_logratio": "em_n_sent",
    "em_n_recipients_logratio": "em_n_recipients",
    "em_n_external_logratio": "em_n_external_recipients",
    "em_max_recipients_logratio": "em_max_recipients",
    "em_attachments_logratio": "em_attachments",
}
EMAIL_BASELINE_COLUMNS = tuple(_BASELINES)
EMAIL_DAY_COLUMNS = EMAIL_RAW_COLUMNS + EMAIL_DERIVED_COLUMNS + EMAIL_BASELINE_COLUMNS


def add_email_features(table, email_user_day):
    """
    table: one row per (user, active day) with `user` and `day`. email_user_day: output of
    build_email_user_day. Returns a copy sorted by (user, day) with EMAIL_DAY_COLUMNS added;
    rows are neither added nor dropped (a day with email but no logon/device/file activity
    is not a row of the table).
    """
    email = email_user_day[["user", "day", *EMAIL_RAW_COLUMNS]]
    out = table.merge(email, on=["user", "day"], how="left")
    for column in EMAIL_RAW_COLUMNS:
        out[column] = out[column].fillna(0.0)
    out = out.sort_values(["user", "day"]).reset_index(drop=True)
    distinct = out["em_n_distinct_recipients"].to_numpy()
    out["em_new_recipient_share"] = np.where(distinct > 0, out["em_n_new_recipients"] / np.maximum(distinct, 1.0), 0.0)
    for new_column, raw_column in _BASELINES.items():
        out[new_column] = np.log1p(out[raw_column]) - np.log1p(_trailing_mean_of_previous(out, raw_column))
    return out
