"""
Two ways of deciding which test SESSIONS count as malicious. Final
results are always reported under BOTH, side by side.

  "window"      the original labelling (Chapter 3): a session is malicious
                if it falls inside its user's insider time window from
                insiders.csv. For the long scenario-2 windows this marks
                many ordinary sessions as malicious.
  "answer_key"  a tighter, event-level labelling from the answer key's own
                records: a session is malicious if it contains a file
                event on a day when the answer key lists an injected http
                event for that insider, or while one of the insider's
                answer-key removable-device sessions (Connect -> Disconnect)
                was open.

IMPORTANT for reporting: the answer_key labelling was introduced AFTER
seeing the window-labelled results, because window labels dilute
scenario 2. It is an evaluation labelling only -- no feature, model or
threshold is built from the answer key. State this wherever it is used,
and never report only the more flattering of the two.
"""

import numpy as np
import pandas as pd
import glob
import os

from src.common.preprocessing.labeling import DEFAULT_ANSWERS_DIR
from src.common.preprocessing.sessionize import SESSION_KEY_COLUMNS

WINDOW_LABELING = "window"
ANSWER_KEY_LABELING = "answer_key"
LABELINGS = (WINDOW_LABELING, ANSWER_KEY_LABELING)

_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def load_answer_key_events(answers_dir=DEFAULT_ANSWERS_DIR):
    """
    Reads every answers/r4.2-<scenario>/*.csv file and returns a dict:
      "http_days":        DataFrame(user, day, scenario) -- days with >= 1 injected http event
      "device_intervals": DataFrame(user, scenario, start, end) -- each Connect->Disconnect pair
    """
    http_rows, device_events = [], []
    for path in sorted(glob.glob(os.path.join(answers_dir, "r4.2-*", "*.csv"))):
        scenario = int(os.path.basename(os.path.dirname(path)).split("-")[1])
        with open(path) as f:  # text mode: "\r\n" and "\n" line endings both read as "\n"
            for line in f:
                line = line.rstrip("\r\n")  # belt and braces: never leave a stray "\r" on the last field
                if line.startswith("http,"):
                    parts = line.split(",", 5)  # url/content may contain commas
                    if len(parts) >= 5:
                        date = pd.to_datetime(parts[2], format=_DATE_FORMAT)
                        http_rows.append((parts[3], date.normalize(), scenario))
                elif line.startswith("device,"):
                    parts = line.split(",")
                    if len(parts) >= 6:
                        device_events.append(
                            (pd.to_datetime(parts[2], format=_DATE_FORMAT), parts[3], parts[4], parts[5], scenario)
                        )

    http_days = pd.DataFrame(http_rows, columns=["user", "day", "scenario"]).drop_duplicates()

    intervals = []
    open_connect = {}
    for date, user, pc, activity, scenario in sorted(device_events, key=lambda e: e[0]):
        key = (user, pc)
        if activity == "Connect":
            open_connect[key] = (date, scenario)
        elif activity == "Disconnect" and key in open_connect:
            start, sc = open_connect.pop(key)
            intervals.append({"user": user, "scenario": sc, "start": start, "end": date})
    device_intervals = pd.DataFrame(intervals, columns=["user", "scenario", "start", "end"])

    return {"http_days": http_days, "device_intervals": device_intervals}


def answer_key_session_labels(events_df, key):
    """
    0/1 array, one entry per session in canonical order (events sorted by
    SESSION_KEY_COLUMNS, then grouped): 1 if any event of the session
    falls on an answer-key http day for its user or inside an
    answer-key device interval for its user.
    """
    ordered = events_df.sort_values(SESSION_KEY_COLUMNS).copy()
    ordered["_day"] = pd.to_datetime(ordered["date"]).dt.normalize()

    http_keys = (
        key["http_days"][["user", "day"]].drop_duplicates()
        .rename(columns={"day": "_day"}).assign(_hit_http=True)
    )
    merged = ordered.merge(http_keys, on=["user", "_day"], how="left")  # left merge keeps row order
    # np.array(..., copy=True): newer pandas returns READ-ONLY arrays from
    # .to_numpy() (copy-on-write), and `hit` is modified in place below.
    hit = np.array(merged["_hit_http"].notna(), dtype=bool, copy=True)

    dates = pd.to_datetime(merged["date"]).to_numpy()
    users = merged["user"].to_numpy()
    for user, intervals in key["device_intervals"].groupby("user"):
        idx = np.flatnonzero(users == user)
        if len(idx) == 0:
            continue
        user_dates = dates[idx]
        in_interval = np.zeros(len(idx), dtype=bool)
        for _, interval in intervals.iterrows():
            in_interval |= (user_dates >= interval["start"].to_datetime64()) & (
                user_dates <= interval["end"].to_datetime64()
            )
        hit[idx] |= in_interval

    merged["_hit"] = hit
    return merged.groupby(SESSION_KEY_COLUMNS, sort=False)["_hit"].any().to_numpy().astype(int)
