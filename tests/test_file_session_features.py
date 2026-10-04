import pandas as pd
import pytest
from src.preprocessing.file_session_features import (
    add_user_session_volume_deviation_feature,
    add_user_session_frequency_deviation_feature,
)


def test_user_session_volume_deviation_flags_unusually_large_session():
    # User AAA's three sessions: 2, 2, and 20 events. The 20-event
    # session should compare itself against the OTHER two (avg 2),
    # giving a deviation of 10x -- clearly unusual for this person.
    sessions = []
    for session_index, event_count in enumerate([2, 2, 20]):
        for _ in range(event_count):
            sessions.append({
                "user": "AAA0001",
                "session_start": pd.Timestamp("2010-01-01") + pd.Timedelta(days=session_index),
                "session_end": pd.Timestamp("2010-01-01") + pd.Timedelta(days=session_index, hours=1),
            })
    df = pd.DataFrame(sessions)

    result = add_user_session_volume_deviation_feature(df)

    large_session_deviation = result[result["session_start"] == pd.Timestamp("2010-01-03")][
        "user_session_volume_deviation"
    ].iloc[0]
    assert large_session_deviation == pytest.approx(10.0)  # 20 / avg(2, 2)

    small_session_deviation = result[result["session_start"] == pd.Timestamp("2010-01-01")][
        "user_session_volume_deviation"
    ].iloc[0]
    assert small_session_deviation == pytest.approx(2 / 11)  # 2 / avg(2, 20)


def test_user_session_volume_deviation_falls_back_to_neutral_for_single_session_user():
    df = pd.DataFrame({
        "user": ["BBB0002"] * 5,
        "session_start": [pd.Timestamp("2010-01-01")] * 5,
        "session_end": [pd.Timestamp("2010-01-01 01:00:00")] * 5,
    })

    result = add_user_session_volume_deviation_feature(df)

    assert (result["user_session_volume_deviation"] == 1.0).all()


def test_user_session_frequency_deviation_flags_bunched_up_sessions():
    # User AAA's four sessions, 2 days apart, 2 days apart, then 20 days
    # apart. The two 2-day gaps should each compare against the OTHER
    # two gaps (2 and 20 -> avg 11), giving 2/11 -- much MORE frequent
    # than usual. The 20-day gap compares against the other two 2-day
    # gaps (avg 2), giving 20/2 = 10 -- much LESS frequent than usual.
    session_starts = [
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2010-01-03"),   # +2 days
        pd.Timestamp("2010-01-05"),   # +2 days
        pd.Timestamp("2010-01-25"),   # +20 days
    ]
    rows = [
        {"user": "AAA0001", "session_start": start, "session_end": start + pd.Timedelta(hours=1)}
        for start in session_starts
    ]
    df = pd.DataFrame(rows)

    result = add_user_session_frequency_deviation_feature(df)

    deviation_by_start = result.set_index("session_start")["user_session_frequency_deviation"]

    assert deviation_by_start[pd.Timestamp("2010-01-01")] == 1.0  # first session, no gap at all
    assert deviation_by_start[pd.Timestamp("2010-01-03")] == pytest.approx(2 / 11)
    assert deviation_by_start[pd.Timestamp("2010-01-05")] == pytest.approx(2 / 11)
    assert deviation_by_start[pd.Timestamp("2010-01-25")] == pytest.approx(10.0)


def test_user_session_frequency_deviation_falls_back_for_too_few_sessions():
    # Only 2 sessions total -- one real gap, but no OTHER gap to compare
    # it against, so both sessions should fall back to the neutral value.
    df = pd.DataFrame({
        "user": ["BBB0002", "BBB0002"],
        "session_start": [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-01-05")],
        "session_end": [pd.Timestamp("2010-01-01 01:00:00"), pd.Timestamp("2010-01-05 01:00:00")],
    })

    result = add_user_session_frequency_deviation_feature(df)

    assert (result["user_session_frequency_deviation"] == 1.0).all()
