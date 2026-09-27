import pandas as pd

FILE_TYPE_BY_HEADER = {
    "D0-CF-11-E0-A1-B1-1A-E1": "office_document",
    "25-50-44-46-2D": "pdf",
    "50-4B-03-04-14": "zip",
    "FF-D8": "jpg",
    "4D-5A-90-00-03-00-00-00-04-00-00-00-FF-FF-00-00-B8-00-00-00-00-00-00-00-40-00": "exe",
}


def extract_file_type(file_df):
    """
    Categorise each file event by its real type, inferred from the
    hexadecimal header at the start of `content` — not from the filename
    extension, which the release's own documentation implies can be
    inconsistent with the file's actual type.
    """
    file_df = file_df.copy()
    header = file_df["content"].str.split(" ").str[0]
    file_df["file_type"] = header.map(FILE_TYPE_BY_HEADER).fillna("other")
    return file_df

def extract_file_extension(file_df):
    """
    Adds `file_extension`: the filename's extension, lower-cased. r4.2's
    file.csv stores only a bare filename with no directory structure, so
    this is the only path-like information actually available.
    """
    file_df = file_df.copy()
    file_df["file_extension"] = file_df["filename"].str.split(".").str[-1].str.lower()
    return file_df


def add_time_since_previous_event(file_df):
    """
    Adds `seconds_since_last_event`: time elapsed since this same user's
    previous file-copy event, in seconds. A user's very first event has
    no previous event to compare against, so it gets 0.
    """
    file_df = file_df.sort_values(["user", "date"]).reset_index(drop=True)
    file_df["seconds_since_last_event"] = (
        file_df.groupby("user")["date"].diff().dt.total_seconds().fillna(0)
    )
    return file_df

def add_device_session_features(file_df, device_df):
    """
    Adds two features derived from device.csv:
      - seconds_since_device_connect: time since the enclosing removable
        device connection began, for the same user and PC.
      - copies_since_device_connect: how many files had already been
        copied earlier within that same connection (0 for the first).

    A file event with no preceding Connect event for its user+PC (e.g.
    the device log is missing an earlier connect) is treated as the
    start of its own connection: 0 seconds since connect, position 0.
    """
    file_df = file_df.sort_values("date").reset_index(drop=True)

    connects = device_df[device_df["activity"] == "Connect"].copy()
    connects = connects.sort_values("date").reset_index(drop=True)
    connects = connects.rename(columns={"date": "connect_time"})

    merged = pd.merge_asof(
        file_df,
        connects[["user", "pc", "connect_time"]],
        left_on="date",
        right_on="connect_time",
        by=["user", "pc"],
        direction="backward",
    )

    # Fallback: no preceding connect found for this user+pc — treat the
    # file event itself as the start of an (unlogged) connection.
    missing_connect = merged["connect_time"].isna()
    merged.loc[missing_connect, "connect_time"] = merged.loc[missing_connect, "date"]

    merged["seconds_since_device_connect"] = (
        (merged["date"] - merged["connect_time"]).dt.total_seconds()
    )
    merged["copies_since_device_connect"] = (
        merged.groupby(["user", "pc", "connect_time"]).cumcount()
    )

    return merged.drop(columns=["connect_time"])

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