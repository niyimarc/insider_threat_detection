import numpy as np
import pandas as pd
import pytest

from src.common.preprocessing.feature_groups import NUMERIC_FEATURE_REGISTRY, available_groups
from src.supervised.dataset import (
    DAY_KEY,
    attach_labels,
    build_user_day_table,
    day_feature_columns,
    feature_matrix,
    group_of_day_column,
)


def _events():
    """U1 has two days (2 events + 1 event, the second day malicious); U2 has one day."""
    rows = []

    def add(user, ts, session_start, malicious, **overrides):
        row = {"user": user, "date": pd.Timestamp(ts), "session_start": pd.Timestamp(session_start),
               "is_malicious": malicious}
        for column, _ in NUMERIC_FEATURE_REGISTRY:
            row[column] = 1.0
        row.update(overrides)
        rows.append(row)

    add("U1", "2010-06-01 09:00", "2010-06-01 08:00", False, seconds_since_last_event=0.0, outside_working_hours=0.0)
    add("U1", "2010-06-01 09:10", "2010-06-01 08:00", False, seconds_since_last_event=600.0, outside_working_hours=1.0)
    add("U1", "2010-06-02 23:00", "2010-06-02 22:30", True, seconds_since_last_event=30.0,
        http_offbaseline_share_7d=0.4)
    add("U2", "2010-06-01 10:00", "2010-06-01 09:00", False)
    add("U2", "2010-06-01 15:00", "2010-06-01 14:00", False)  # second session the same day
    return pd.DataFrame(rows)


def test_one_row_per_user_day_with_counts_and_labels():
    table = build_user_day_table(_events())
    assert list(table[DAY_KEY].itertuples(index=False, name=None)) == [
        ("U1", pd.Timestamp("2010-06-01")), ("U1", pd.Timestamp("2010-06-02")), ("U2", pd.Timestamp("2010-06-01")),
    ]
    by_day = table.set_index(DAY_KEY)
    assert by_day.loc[("U1", pd.Timestamp("2010-06-01")), "day_n_events"] == 2
    assert by_day.loc[("U1", pd.Timestamp("2010-06-01")), "day_n_sessions"] == 1
    assert by_day.loc[("U2", pd.Timestamp("2010-06-01")), "day_n_sessions"] == 2
    assert by_day["window_label"].tolist() == [0, 1, 0]


def test_features_are_mean_and_max_with_flags_summarised_by_their_mean_only():
    table = build_user_day_table(_events()).set_index(DAY_KEY)
    first = table.loc[("U1", pd.Timestamp("2010-06-01"))]
    assert first["seconds_since_last_event__mean"] == 300.0
    assert first["seconds_since_last_event__max"] == 600.0
    assert first["outside_working_hours__mean"] == 0.5
    assert "outside_working_hours__max" not in table.columns
    assert table.loc[("U1", pd.Timestamp("2010-06-02")), "http_offbaseline_share_7d__mean"] == 0.4


def test_day_label_is_true_if_any_event_of_the_day_is_malicious():
    events = _events()
    events.loc[0, "is_malicious"] = True  # one malicious event among two on U1's first day
    table = build_user_day_table(events).set_index(DAY_KEY)
    assert table.loc[("U1", pd.Timestamp("2010-06-01")), "window_label"] == 1


def test_columns_follow_the_selected_signal_groups():
    file_only = day_feature_columns(("file",))
    everything = day_feature_columns(available_groups())
    assert set(file_only) < set(everything)
    assert "day_n_events" in file_only and "day_n_sessions" in file_only
    assert not any(c.startswith("http_") for c in file_only)
    assert any(c.startswith("http_offbaseline_share_7d") for c in everything)
    assert len(everything) == len(set(everything))


def test_the_mandatory_file_group_cannot_be_dropped():
    with pytest.raises(ValueError):
        day_feature_columns(("device",))


def test_every_input_column_maps_back_to_its_signal_group():
    for column in day_feature_columns(available_groups()):
        assert group_of_day_column(column) in available_groups()
    assert group_of_day_column("http_new_domains_7d__mean") == "http"
    assert group_of_day_column("seconds_since_device_connect__max") == "device"
    assert group_of_day_column("day_n_events") == "file"


def test_labels_are_never_input_columns():
    columns = day_feature_columns(available_groups())
    assert "window_label" not in columns and "answer_key_label" not in columns


def test_missing_columns_raise_a_rebuild_hint():
    events = _events().drop(columns=["http_new_domains_7d"])
    with pytest.raises(ValueError, match="build_dataset"):
        build_user_day_table(events)


def test_answer_key_labels_attach_by_user_and_day():
    table = build_user_day_table(_events())
    key_days = pd.DataFrame({
        "user": ["U1", "U2"], "day": pd.to_datetime(["2010-06-02", "2010-06-01"]), "answer_key_label": [1, 0],
    })
    labelled = attach_labels(table, key_days).set_index(DAY_KEY)
    assert labelled["answer_key_label"].tolist() == [0, 1, 0]
    assert (attach_labels(table)["answer_key_label"] == 0).all()


def test_feature_matrix_is_a_writable_float_copy():
    table = build_user_day_table(_events())
    columns = day_feature_columns(("file",))
    x = feature_matrix(table, columns)
    assert x.shape == (3, len(columns)) and x.dtype == float
    x[0, 0] = -1.0  # must not raise (read-only arrays under newer pandas) and must not alter the table
    assert table[columns[0]].iloc[0] != -1.0


def test_input_frame_is_not_modified():
    events = _events()
    before = events.copy()
    build_user_day_table(events)
    pd.testing.assert_frame_equal(events, before)
