import pandas as pd
from src.preprocessing.logon_features import add_outside_hours_feature


def test_outside_hours_flags_unusual_time_and_unusual_day():
    logon_df = pd.DataFrame({
        "user": ["AAA0001"] * 6,
        "date": pd.to_datetime([
            "2010-06-07 08:00:00", "2010-06-07 17:00:00",  # Monday
            "2010-06-08 08:00:00", "2010-06-08 17:00:00",  # Tuesday
            "2010-06-09 08:00:00", "2010-06-09 17:00:00",  # Wednesday
        ]),
        "activity": ["Logon", "Logoff"] * 3,
    })
    file_df = pd.DataFrame({
        "user": ["AAA0001", "AAA0001", "AAA0001"],
        "date": pd.to_datetime([
            "2010-06-07 10:00:00",  # Monday, normal hour -> False
            "2010-06-07 22:00:00",  # Monday, late night -> True (time)
            "2010-06-12 10:00:00",  # Saturday, never worked -> True (day)
        ]),
    })

    result = add_outside_hours_feature(file_df, logon_df)
    assert list(result["outside_working_hours"]) == [False, True, True]


def test_outside_hours_falls_back_to_default_window_when_no_logon_history():
    logon_df = pd.DataFrame({
        "user": ["placeholder"], "date": pd.to_datetime(["2010-01-01 08:00:00"]),
        "activity": ["Logon"],
    }).iloc[0:0]

    file_df = pd.DataFrame({
        "user": ["BBB0002", "BBB0002"],
        "date": pd.to_datetime([
            "2010-06-07 10:00:00",  # within default 07:00-19:00 -> False
            "2010-06-07 22:00:00",  # outside default window -> True
        ]),
    })

    result = add_outside_hours_feature(file_df, logon_df)
    assert list(result["outside_working_hours"]) == [False, True]
