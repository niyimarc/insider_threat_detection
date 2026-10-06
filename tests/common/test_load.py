import pandas as pd
from src.common.preprocessing.load import load_file_events, load_logon_events, load_device_events


def test_file_events_loaded_correctly():
    df = load_file_events()
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert set(df.columns) == {"id", "date", "user", "pc", "filename", "content"}


def test_logon_events_loaded_correctly():
    df = load_logon_events()
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert set(df["activity"].unique()) == {"Logon", "Logoff"}


def test_device_events_loaded_correctly():
    df = load_device_events()
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert set(df["activity"].unique()) == {"Connect", "Disconnect"}