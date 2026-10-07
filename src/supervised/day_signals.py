"""
DAY-LEVEL logon and device signals for the all-days supervised table.

Why these exist: the first supervised table only had days with a file copy,
and its device/logon features were attached to those file events. But the
CERT r4.2 scenarios live mostly in the device and logon logs (after-hours
logins, removable-drive use at markedly higher rates than the user's own,
first-ever drive use), and the dataset's own readme says that a user's start
time, workday length, after-hours habit and thumb-drive rate persist from
day to day, so DEVIATION FROM THE USER'S OWN BASELINE is the intended signal.

The feature list below was fixed in advance from that readme and from
scenarios.txt -- no answer-key event and no result was looked at to choose
or tune it. Rules:
  * RAW day counts come from the log of that day only.
  * BASELINE features compare today with the same user's PREVIOUS active
    days only (trailing window, today excluded): nothing from the future.
    Until a user has MIN_HISTORY previous active days the deviation is NaN
    (unknown), and the "first time ever" flags stay 0 until FIRST_EVENT_MIN_HISTORY
    previous days exist, so the start of the log does not fire everything.
  * The baselines are differences or log-ratios, never z-scores: a user
    whose daily values are (almost) constant would give an exploding z-score
    (this happened with the http volume feature; see http_features.py).
  * "Primary PC" is the user's most frequent logon PC in their first
    PRIMARY_PC_DAYS calendar days of data (a fixed warm-up, as in http).
  * Working hours are the generic 07:00-19:00 window; each user's own
    pattern enters through the baseline features, not through this constant.
"""

import numpy as np
import pandas as pd

WORK_START_HOUR = 7.0
WORK_END_HOUR = 19.0
PRIMARY_PC_DAYS = 30
BASELINE_WINDOW = 28          # previous ACTIVE days in the trailing baseline
MIN_HISTORY = 7               # previous active days needed before a deviation is defined
FIRST_EVENT_MIN_HISTORY = 14  # previous active days needed before a "first time" flag can fire
MAX_CONNECTED_MINUTES = 720.0  # one connection counts at most 12 hours (power-down artefacts)

LOGON_RAW_COLUMNS = (
    "lg_n_logons", "lg_first_hour", "lg_last_hour", "lg_span_hours", "lg_n_distinct_pcs",
    "lg_is_weekend", "lg_after_hours_events", "lg_other_pc_events",
)
LOGON_BASELINE_COLUMNS = (
    "lg_first_hour_dev", "lg_span_dev", "lg_n_logons_logratio", "lg_after_hours_logratio",
    "lg_first_after_hours_day",
)
DEVICE_RAW_COLUMNS = (
    "dv_n_connects", "dv_after_hours_connects", "dv_n_distinct_pcs", "dv_connected_minutes",
    "dv_unmatched_connects",
)
DEVICE_BASELINE_COLUMNS = (
    "dv_n_connects_logratio", "dv_connected_minutes_dev", "dv_prior_use_days", "dv_first_use_today",
    "dv_first_after_hours_connect_day",
)
LOGON_DAY_COLUMNS = LOGON_RAW_COLUMNS + LOGON_BASELINE_COLUMNS
DEVICE_DAY_COLUMNS = DEVICE_RAW_COLUMNS + DEVICE_BASELINE_COLUMNS


def _hours(dates):
    return dates.dt.hour + dates.dt.minute / 60.0 + dates.dt.second / 3600.0


def logon_day_counts(logon_df):
    """
    One row per (user, day) that has any logon/logoff event, with the
    LOGON_RAW_COLUMNS. `logon_df` needs user, pc, date (datetime), activity.
    """
    df = logon_df[["user", "pc", "date", "activity"]].copy()
    df["day"] = df["date"].dt.normalize()
    df["hour"] = _hours(df["date"])
    df["is_logon"] = df["activity"] == "Logon"
    df["after_hours"] = (df["hour"] < WORK_START_HOUR) | (df["hour"] >= WORK_END_HOUR)

    first_day = df.groupby("user")["day"].transform("min")
    warmup = df[df["is_logon"] & (df["day"] < first_day + pd.Timedelta(days=PRIMARY_PC_DAYS))]
    primary_pc = warmup.groupby("user")["pc"].agg(lambda pcs: pcs.mode().iloc[0])
    primary_for_row = df["user"].map(primary_pc)
    df["other_pc_logon"] = df["is_logon"] & primary_for_row.notna() & (df["pc"] != primary_for_row)

    grouped = df.groupby(["user", "day"], sort=True)
    out = pd.DataFrame({
        "lg_n_logons": grouped["is_logon"].sum().astype(float),
        "lg_first_hour": grouped["hour"].min(),
        "lg_last_hour": grouped["hour"].max(),
        "lg_n_distinct_pcs": grouped["pc"].nunique().astype(float),
        "lg_after_hours_events": grouped["after_hours"].sum().astype(float),
        "lg_other_pc_events": grouped["other_pc_logon"].sum().astype(float),
    })
    out["lg_span_hours"] = out["lg_last_hour"] - out["lg_first_hour"]
    out = out.reset_index()
    out["lg_is_weekend"] = (out["day"].dt.dayofweek >= 5).astype(float)
    return out[["user", "day", *LOGON_RAW_COLUMNS]]


def device_day_counts(device_df):
    """
    One row per (user, day) with any Connect/Disconnect event, with the
    DEVICE_RAW_COLUMNS. A Disconnect that directly follows a Connect on the
    same user+PC closes a connection; its length is attributed to the
    Disconnect's day. Connects with no Disconnect are counted as unmatched.
    """
    df = device_df[["user", "pc", "date", "activity"]].sort_values(["user", "pc", "date"]).copy()
    df["day"] = df["date"].dt.normalize()
    df["hour"] = _hours(df["date"])
    df["is_connect"] = df["activity"] == "Connect"
    df["after_hours_connect"] = df["is_connect"] & (
        (df["hour"] < WORK_START_HOUR) | (df["hour"] >= WORK_END_HOUR)
    )

    by_pc = df.groupby(["user", "pc"], sort=False)
    previous_activity = by_pc["activity"].shift(1)
    previous_date = by_pc["date"].shift(1)
    df["closes_connection"] = (df["activity"] == "Disconnect") & (previous_activity == "Connect")
    minutes = (df["date"] - previous_date).dt.total_seconds() / 60.0
    df["connected_minutes"] = minutes.where(df["closes_connection"], 0.0).clip(upper=MAX_CONNECTED_MINUTES)

    grouped = df.groupby(["user", "day"], sort=True)
    out = pd.DataFrame({
        "dv_n_connects": grouped["is_connect"].sum().astype(float),
        "dv_after_hours_connects": grouped["after_hours_connect"].sum().astype(float),
        "dv_n_distinct_pcs": grouped["pc"].nunique().astype(float),
        "dv_connected_minutes": grouped["connected_minutes"].sum(),
        "closed": grouped["closes_connection"].sum().astype(float),
    })
    out["dv_unmatched_connects"] = (out["dv_n_connects"] - out["closed"]).clip(lower=0.0)
    out = out.drop(columns=["closed"]).reset_index()
    return out[["user", "day", *DEVICE_RAW_COLUMNS]]


def _trailing_mean_of_previous(table, column):
    """Mean of the user's previous BASELINE_WINDOW active-day rows (today excluded); NaN with < MIN_HISTORY."""
    return table.groupby("user")[column].transform(
        lambda s: s.shift(1).rolling(BASELINE_WINDOW, min_periods=MIN_HISTORY).mean()
    )


def add_causal_baselines(table):
    """
    Adds LOGON_BASELINE_COLUMNS and DEVICE_BASELINE_COLUMNS to a table with
    one row per (user, active day) and the raw logon/device columns (counts
    already 0 on days without that log). Uses only each user's previous rows.
    The table is sorted by (user, day) first; the result keeps that order.
    """
    table = table.sort_values(["user", "day"]).reset_index(drop=True).copy()
    n_prior_days = table.groupby("user").cumcount()      # active days strictly before today

    def log_ratio(column):
        return np.log1p(table[column]) - np.log1p(_trailing_mean_of_previous(table, column))

    table["lg_first_hour_dev"] = table["lg_first_hour"] - _trailing_mean_of_previous(table, "lg_first_hour")
    table["lg_span_dev"] = table["lg_span_hours"] - _trailing_mean_of_previous(table, "lg_span_hours")
    table["lg_n_logons_logratio"] = log_ratio("lg_n_logons")
    table["lg_after_hours_logratio"] = log_ratio("lg_after_hours_events")
    table["dv_n_connects_logratio"] = log_ratio("dv_n_connects")
    table["dv_connected_minutes_dev"] = (
        table["dv_connected_minutes"] - _trailing_mean_of_previous(table, "dv_connected_minutes")
    )

    def prior_days_with(column):
        flag = (table[column] > 0).astype(float)
        return flag.groupby(table["user"]).cumsum() - flag, flag

    prior_after_hours, today_after_hours = prior_days_with("lg_after_hours_events")
    table["lg_first_after_hours_day"] = (
        (today_after_hours > 0) & (prior_after_hours == 0) & (n_prior_days >= FIRST_EVENT_MIN_HISTORY)
    ).astype(float)

    prior_use, today_use = prior_days_with("dv_n_connects")
    table["dv_prior_use_days"] = prior_use
    table["dv_first_use_today"] = (
        (today_use > 0) & (prior_use == 0) & (n_prior_days >= FIRST_EVENT_MIN_HISTORY)
    ).astype(float)

    prior_ah_connect, today_ah_connect = prior_days_with("dv_after_hours_connects")
    table["dv_first_after_hours_connect_day"] = (
        (today_ah_connect > 0) & (prior_ah_connect == 0) & (n_prior_days >= FIRST_EVENT_MIN_HISTORY)
    ).astype(float)
    return table
