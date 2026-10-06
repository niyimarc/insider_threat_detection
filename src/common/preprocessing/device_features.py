"""
Features derived from device.csv (signal group "device" in
src/common/preprocessing/feature_groups.py): removable-device connection
behaviour around each file copy.
"""

import pandas as pd


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
