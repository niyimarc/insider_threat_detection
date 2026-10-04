"""
Features derived from file.csv itself (signal group "file" in
src/preprocessing/feature_groups.py): file type, file extension and
the timing of a user's file-copy events.

Session-level file features (volume and frequency relative to a user's
own baseline) live in file_session_features.py, because they need
sessions to exist first.
"""

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
