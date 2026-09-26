import pandas as pd
from src.preprocessing.features import (
    add_outside_hours_feature,
    extract_file_type, 
    extract_file_extension, 
    add_time_since_previous_event, 
    add_device_session_features,
    )


def test_extract_file_type_known_and_unknown_headers():
    file_df = pd.DataFrame({
        "content": [
            "D0-CF-11-E0-A1-B1-1A-E1 some content keywords here",
            "25-50-44-46-2D some content keywords here",
            "FF-D8 some content keywords here",
            "AA-BB-CC a completely unrecognised header",
        ],
    })

    result = extract_file_type(file_df)

    assert list(result["file_type"]) == ["office_document", "pdf", "jpg", "other"]

def test_extract_file_extension():
    file_df = pd.DataFrame({"filename": ["EYPC9Y08.doc", "AU75JV6U.JPG", "noext"]})
    result = extract_file_extension(file_df)
    assert list(result["file_extension"]) == ["doc", "jpg", "noext"]


def test_time_since_previous_event_respects_user_boundaries():
    file_df = pd.DataFrame({
        "user": ["AAA0001", "AAA0001", "BBB0002"],
        "date": pd.to_datetime([
            "2010-06-01 10:00:00",
            "2010-06-01 10:05:00",  # 5 minutes after AAA0001's first event
            "2010-06-01 09:00:00",  # BBB0002's first event — earlier in time,
                                     # but a DIFFERENT user, so this must be 0,
                                     # not a negative number from comparing
                                     # against AAA0001's second event.
        ]),
    })

    result = add_time_since_previous_event(file_df)
    result = result.set_index("user")

    assert result.loc["AAA0001", "seconds_since_last_event"].tolist() == [0, 300]
    assert result.loc["BBB0002", "seconds_since_last_event"] == 0

def test_add_device_session_features_normal_case():
    device_df = pd.DataFrame({
        "user": ["AAA0001"],
        "pc": ["PC-1"],
        "date": pd.to_datetime(["2010-06-01 09:00:00"]),
        "activity": ["Connect"],
    })
    file_df = pd.DataFrame({
        "user": ["AAA0001", "AAA0001"],
        "pc": ["PC-1", "PC-1"],
        "date": pd.to_datetime([
            "2010-06-01 09:05:00",
            "2010-06-01 09:10:00",
        ]),
    })

    result = add_device_session_features(file_df, device_df)

    assert list(result["seconds_since_device_connect"]) == [300, 600]
    assert list(result["copies_since_device_connect"]) == [0, 1]


def test_add_device_session_features_missing_connect_falls_back_to_zero():
    device_df = pd.DataFrame({
        "user": ["placeholder"],
        "pc": ["placeholder"],
        "date": pd.to_datetime(["2010-01-01"]),
        "activity": ["Connect"],
    }).iloc[0:0]  # zero rows, but same column dtypes as a normal DataFrame

    file_df = pd.DataFrame({
        "user": ["AAA0001"],
        "pc": ["PC-1"],
        "date": pd.to_datetime(["2010-06-01 09:05:00"]),
    })

    result = add_device_session_features(file_df, device_df)

    assert result["seconds_since_device_connect"].iloc[0] == 0
    assert result["copies_since_device_connect"].iloc[0] == 0

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