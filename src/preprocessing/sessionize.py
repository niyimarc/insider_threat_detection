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
    file_df = file_df.sort_values("date").reset_index(drop=True)
    sessions_df = sessions_df.sort_values("start").reset_index(drop=True)

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

def add_user_session_volume_deviation_feature(file_events_with_sessions):
    """
    Adds `user_session_volume_deviation`: how this session's event count
    compares to this SAME USER's own typical session, computed from
    their OTHER sessions only (leave-one-out), then broadcast to every
    event row in the session.

    A value of 1.0 means this session's length matches this user's own
    average; 3.0 means three times their usual volume. This is computed
    per-user -- directly reflecting the release's own documentation that
    an unusual number of file copies RELATIVE TO A USER'S NORMAL PATTERN
    is the signal to watch for, not raw volume compared to everyone else.

    A user with only one session has no "other" sessions to compare
    against, and gets the neutral fallback value 1.0 ("looks typical")
    rather than an invented signal from nothing.
    """
    df = file_events_with_sessions.copy()

    session_lengths = (
        df.groupby(["user", "session_start", "session_end"]).size()
        .rename("session_length")
        .reset_index()
    )

    user_totals = session_lengths.groupby("user")["session_length"].transform("sum")
    user_counts = session_lengths.groupby("user")["session_length"].transform("count")

    # Leave-one-out mean: (sum of all this user's sessions - this one) / (count - 1)
    leave_one_out_mean = (user_totals - session_lengths["session_length"]) / (user_counts - 1)
    deviation = session_lengths["session_length"] / leave_one_out_mean

    # Users with only one session (count == 1) have no "other" sessions
    # to compare against -- fall back to the neutral "looks typical"
    # value instead of a division by zero.
    deviation = deviation.where(user_counts > 1, 1.0)
    deviation = deviation.replace([float("inf"), float("-inf")], 1.0).fillna(1.0)

    session_lengths["user_session_volume_deviation"] = deviation

    return df.merge(
        session_lengths[["user", "session_start", "session_end", "user_session_volume_deviation"]],
        on=["user", "session_start", "session_end"],
        how="left",
    )

def add_user_session_frequency_deviation_feature(file_events_with_sessions):
    """
    Adds `user_session_frequency_deviation`: how the GAP since this
    user's previous session compares to this same user's own typical
    gap between sessions, computed from their OTHER gaps only
    (leave-one-out), then broadcast to every event row in the session.

    A value well below 1.0 means this session started much SOONER
    after the user's last session than usual for them -- i.e. they are
    showing up more often than normal. This is meant to catch elevated
    activity that shows up as more FREQUENT ordinary-looking sessions
    rather than one abnormally large session, which
    add_user_session_volume_deviation_feature cannot see.

    A user's very first session has no previous session to measure a
    gap from, and a user with fewer than 3 total sessions has too few
    gaps to form a reliable leave-one-out baseline -- both fall back to
    the neutral value 1.0 ("looks typical") rather than an invented
    signal.
    """
    df = file_events_with_sessions.copy()

    sessions = (
        df[["user", "session_start", "session_end"]]
        .drop_duplicates()
        .sort_values(["user", "session_start"])
        .reset_index(drop=True)
    )

    sessions["gap_seconds"] = (
        sessions.groupby("user")["session_start"].diff().dt.total_seconds()
    )

    user_gap_totals = sessions.groupby("user")["gap_seconds"].transform(
        lambda gaps: gaps.sum(skipna=True)
    )
    user_gap_counts = sessions.groupby("user")["gap_seconds"].transform(
        lambda gaps: gaps.count()
    )

    # Leave-one-out mean of this user's OTHER gaps (excluding this one).
    leave_one_out_mean = (user_gap_totals - sessions["gap_seconds"]) / (user_gap_counts - 1)
    deviation = sessions["gap_seconds"] / leave_one_out_mean

    # Fall back to the neutral value whenever there's no real gap to
    # compare (this user's first session) or too few other gaps to form
    # a meaningful baseline (fewer than 2 other gaps, i.e. fewer than 3
    # total sessions for this user).
    has_enough_data = sessions["gap_seconds"].notna() & (user_gap_counts > 1)
    deviation = deviation.where(has_enough_data, 1.0)
    deviation = deviation.replace([float("inf"), float("-inf")], 1.0).fillna(1.0)

    sessions["user_session_frequency_deviation"] = deviation

    return df.merge(
        sessions[["user", "session_start", "session_end", "user_session_frequency_deviation"]],
        on=["user", "session_start", "session_end"],
        how="left",
    )