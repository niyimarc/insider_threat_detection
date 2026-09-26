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