import pandas as pd
from src.preprocessing.sessionize import (
    assign_sessions_to_file_events,
    build_all_sessions,
    label_sessions,
)


def test_repeated_logons_collapse_into_one_session():
    logon_df = pd.DataFrame({
        "user": ["AAA0001", "AAA0001", "AAA0001"],
        "date": pd.to_datetime([
            "2010-01-04 07:46:00",
            "2010-01-04 12:59:09",  # screen-unlock Logon, no Logoff before it
            "2010-01-04 17:03:00",
        ]),
        "activity": ["Logon", "Logon", "Logoff"],
    })

    sessions = build_all_sessions(logon_df)

    assert len(sessions) == 1
    assert sessions.iloc[0]["start"] == pd.Timestamp("2010-01-04 07:46:00")
    assert sessions.iloc[0]["end"] == pd.Timestamp("2010-01-04 17:03:00")


def test_two_separate_sessions_stay_separate():
    logon_df = pd.DataFrame({
        "user": ["AAA0001"] * 4,
        "date": pd.to_datetime([
            "2010-01-04 08:00:00", "2010-01-04 12:00:00",
            "2010-01-04 13:00:00", "2010-01-04 17:00:00",
        ]),
        "activity": ["Logon", "Logoff", "Logon", "Logoff"],
    })

    sessions = build_all_sessions(logon_df)
    assert len(sessions) == 2


def test_stray_logoff_with_no_open_session_is_ignored():
    logon_df = pd.DataFrame({
        "user": ["AAA0001", "AAA0001"],
        "date": pd.to_datetime(["2010-01-04 09:00:00", "2010-01-04 17:00:00"]),
        "activity": ["Logoff", "Logoff"],  # first one has nothing to close
    })

    sessions = build_all_sessions(logon_df)
    assert len(sessions) == 0


def test_sessions_do_not_mix_across_users():
    logon_df = pd.DataFrame({
        "user": ["AAA0001", "BBB0002", "AAA0001", "BBB0002"],
        "date": pd.to_datetime([
            "2010-01-04 08:00:00", "2010-01-04 09:00:00",
            "2010-01-04 17:00:00", "2010-01-04 18:00:00",
        ]),
        "activity": ["Logon", "Logon", "Logoff", "Logoff"],
    })

    sessions = build_all_sessions(logon_df)
    assert len(sessions) == 2
    assert set(sessions["user"]) == {"AAA0001", "BBB0002"}


def test_assign_sessions_matches_event_inside_window():
    sessions_df = pd.DataFrame({
        "user": ["AAA0001"],
        "start": pd.to_datetime(["2010-01-04 08:00:00"]),
        "end": pd.to_datetime(["2010-01-04 17:00:00"]),
    })
    file_df = pd.DataFrame({
        "user": ["AAA0001"],
        "date": pd.to_datetime(["2010-01-04 12:00:00"]),
    })

    result = assign_sessions_to_file_events(file_df, sessions_df)

    assert result.iloc[0]["session_start"] == pd.Timestamp("2010-01-04 08:00:00")
    assert result.iloc[0]["session_end"] == pd.Timestamp("2010-01-04 17:00:00")


def test_assign_sessions_falls_back_for_event_between_sessions():
    sessions_df = pd.DataFrame({
        "user": ["AAA0001", "AAA0001"],
        "start": pd.to_datetime(["2010-01-04 08:00:00", "2010-01-05 08:00:00"]),
        "end": pd.to_datetime(["2010-01-04 17:00:00", "2010-01-05 17:00:00"]),
    })
    file_df = pd.DataFrame({
        "user": ["AAA0001"],
        "date": pd.to_datetime(["2010-01-04 20:00:00"]),  # after day 1's
                                                            # session ended,
                                                            # before day 2's
                                                            # session started
    })

    result = assign_sessions_to_file_events(file_df, sessions_df)

    # Should fall back to its own singleton session, not get wrongly
    # attached to day 1's already-closed session.
    assert result.iloc[0]["session_start"] == pd.Timestamp("2010-01-04 20:00:00")
    assert result.iloc[0]["session_end"] == pd.Timestamp("2010-01-04 20:00:00")


def test_label_sessions_marks_whole_session_malicious():
    file_events = pd.DataFrame({
        "user": ["AAA0001", "AAA0001", "AAA0001"],
        "session_start": pd.to_datetime(["2010-01-04 08:00:00"] * 3),
        "session_end": pd.to_datetime(["2010-01-04 17:00:00"] * 3),
        "is_malicious": [False, True, False],
    })

    result = label_sessions(file_events)
    assert list(result["session_is_malicious"]) == [True, True, True]


def test_label_sessions_leaves_benign_sessions_alone():
    file_events = pd.DataFrame({
        "user": ["AAA0001", "AAA0001"],
        "session_start": pd.to_datetime(["2010-01-04 08:00:00"] * 2),
        "session_end": pd.to_datetime(["2010-01-04 17:00:00"] * 2),
        "is_malicious": [False, False],
    })

    result = label_sessions(file_events)
    assert list(result["session_is_malicious"]) == [False, False]


def test_label_sessions_does_not_mix_different_sessions():
    file_events = pd.DataFrame({
        "user": ["AAA0001", "AAA0001"],
        "session_start": pd.to_datetime(["2010-01-04 08:00:00", "2010-01-05 08:00:00"]),
        "session_end": pd.to_datetime(["2010-01-04 17:00:00", "2010-01-05 17:00:00"]),
        "is_malicious": [True, False],
    })

    result = label_sessions(file_events)
    assert list(result["session_is_malicious"]) == [True, False]
