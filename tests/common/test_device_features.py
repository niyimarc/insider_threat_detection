import pandas as pd
from src.common.preprocessing.device_features import add_device_session_features


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
