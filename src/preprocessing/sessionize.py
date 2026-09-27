import pandas as pd


def _build_sessions_for_user(user_logon_df):
    """
    Given one user's logon/logoff rows (sorted by date), returns a list
    of (start, end) tuples — one per real session — collapsing repeated
    Logons (screen unlocks) into whichever session is already open, and
    ignoring stray Logoffs with no open session.
    """
    sessions = []
    current_start = None

    for _, row in user_logon_df.iterrows():
        if row["activity"] == "Logon":
            if current_start is None:
                current_start = row["date"]
            # else: repeated Logon while a session is open — a screen
            # unlock continuing the same session, not a new one.
        elif row["activity"] == "Logoff":
            if current_start is not None:
                sessions.append((current_start, row["date"]))
                current_start = None
            # else: stray Logoff with no open session — ignore (dirty data).

    return sessions


def build_all_sessions(logon_df):
    """Returns a DataFrame with one row per real session: user, start, end."""
    logon_df = logon_df.sort_values(["user", "date"])
    all_sessions = []

    for user, group in logon_df.groupby("user"):
        for start, end in _build_sessions_for_user(group):
            all_sessions.append({"user": user, "start": start, "end": end})

    return pd.DataFrame(all_sessions)

def assign_sessions_to_file_events(file_df, sessions_df):
    """
    Assigns each file event to the session (start, end) it falls inside,
    for that same user. A file event that doesn't fall inside any known
    session becomes its own singleton session (start = end = its own
    timestamp), so no event is ever silently dropped.
    """
    file_df = file_df.sort_values(["user", "date"]).reset_index(drop=True)
    sessions_df = sessions_df.sort_values(["user", "start"]).reset_index(drop=True)

    merged = pd.merge_asof(
        file_df,
        sessions_df.rename(columns={"start": "session_start", "end": "session_end"}),
        left_on="date",
        right_on="session_start",
        by="user",
        direction="backward",
    )

    # merge_asof only guarantees the event is AFTER session_start — we
    # still need to check it's also BEFORE session_end, otherwise it
    # might have matched a session that had already closed.
    valid = merged["session_end"].notna() & (merged["date"] <= merged["session_end"])

    # Fallback: no valid enclosing session — treat this event as its
    # own singleton session.
    merged.loc[~valid, "session_start"] = merged.loc[~valid, "date"]
    merged.loc[~valid, "session_end"] = merged.loc[~valid, "date"]

    return merged

def label_sessions(file_events_with_sessions):
    """
    Adds `session_is_malicious`: True for every event belonging to a
    session that contains at least one malicious-labelled event.
    """
    file_events_with_sessions = file_events_with_sessions.copy()
    file_events_with_sessions["session_is_malicious"] = (
        file_events_with_sessions
        .groupby(["user", "session_start", "session_end"])["is_malicious"]
        .transform("any")
    )
    return file_events_with_sessions