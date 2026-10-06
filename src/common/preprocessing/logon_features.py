"""
Features derived from logon.csv (signal group "logon" in
src/common/preprocessing/feature_groups.py): how a file copy relates to the
user's own usual working hours and working days.

(Session construction from logon/logoff pairs is in sessionize.py; this
module is only about behavioural features.)
"""

import pandas as pd

DEFAULT_TYPICAL_START_HOUR = 7.0
DEFAULT_TYPICAL_END_HOUR = 19.0


def _time_of_day_hours(date_series):
    return date_series.dt.hour + date_series.dt.minute / 60.0


def _build_user_hour_windows(logon_df, start_percentile=10, end_percentile=90):
    logons = logon_df[logon_df["activity"] == "Logon"]
    logoffs = logon_df[logon_df["activity"] == "Logoff"]

    typical_start = _time_of_day_hours(logons["date"]).groupby(
        logons["user"]
    ).quantile(start_percentile / 100)

    typical_end = _time_of_day_hours(logoffs["date"]).groupby(
        logoffs["user"]
    ).quantile(end_percentile / 100)

    return typical_start, typical_end


def _build_user_working_days(logon_df):
    logons = logon_df[logon_df["activity"] == "Logon"]
    return logons.groupby("user")["date"].apply(
        lambda dates: set(dates.dt.dayofweek.unique())
    )


def add_outside_hours_feature(file_df, logon_df):
    """
    Adds `outside_working_hours`: True if this file event happened on a
    day of week this user has never logged on before, OR outside their
    own typical logon/logoff hour window (10th-90th percentile of their
    historical Logon/Logoff times). Users with no logon history at all
    fall back to a generic 07:00-19:00 window, with no day check applied.
    """
    file_df = file_df.copy()

    typical_start, typical_end = _build_user_hour_windows(logon_df)
    working_days = _build_user_working_days(logon_df)

    event_hour = _time_of_day_hours(file_df["date"])
    event_day = file_df["date"].dt.dayofweek

    user_start = file_df["user"].map(typical_start).fillna(DEFAULT_TYPICAL_START_HOUR)
    user_end = file_df["user"].map(typical_end).fillna(DEFAULT_TYPICAL_END_HOUR)
    time_flag = (event_hour < user_start) | (event_hour > user_end)

    def _is_unusual_day(user, day):
        days = working_days.get(user)
        return False if days is None else day not in days

    day_flag = pd.Series(
        [_is_unusual_day(u, d) for u, d in zip(file_df["user"], event_day)],
        index=file_df.index,
    )

    file_df["outside_working_hours"] = time_flag | day_flag
    return file_df
