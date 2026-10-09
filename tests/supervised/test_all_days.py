import numpy as np
import pandas as pd
import pytest

from src.common.preprocessing.feature_groups import NUMERIC_FEATURE_REGISTRY, available_groups
from src.supervised.all_days import build_all_user_days, window_labels_for_days
from src.supervised.dataset import DAY_KEY
from src.supervised.experiment import run_experiment
from src.supervised.feature_sets import (
    ALL_DAYS,
    ALL_DAYS_COLUMNS_BY_GROUP,
    FILE_EVENT_DAYS,
    all_days_feature_columns,
    all_days_group_of_column,
)


def _ts(text):
    return pd.Timestamp(text)


def _logon(user, pc, ts, activity="Logon"):
    return {"user": user, "pc": pc, "date": _ts(ts), "activity": activity}


def _raw_frames():
    logon = pd.DataFrame([
        _logon("U1", "PC-1", "2010-01-04 08:00:00"), _logon("U1", "PC-1", "2010-01-04 17:00:00", "Logoff"),
        _logon("U1", "PC-1", "2010-01-05 08:00:00"), _logon("U1", "PC-1", "2010-01-05 17:00:00", "Logoff"),
        _logon("U1", "PC-1", "2010-01-06 08:00:00"), _logon("U1", "PC-1", "2010-01-06 17:00:00", "Logoff"),
        _logon("U2", "PC-2", "2010-01-04 08:30:00"), _logon("U2", "PC-2", "2010-01-04 17:30:00", "Logoff"),
        _logon("U3", "PC-3", "2010-01-04 08:00:00"),                          # a user we will not include
    ])
    device = pd.DataFrame([
        _logon("U1", "PC-1", "2010-01-05 12:00:00", "Connect"), _logon("U1", "PC-1", "2010-01-05 12:20:00", "Disconnect"),
        _logon("U1", "PC-1", "2010-01-07 12:00:00", "Connect"),               # a device day with NO logon that day
    ])
    rows = []
    for ts in ("2010-01-05 12:05:00", "2010-01-05 12:10:00"):
        row = {"user": "U1", "date": _ts(ts), "session_start": _ts("2010-01-05 08:00:00"), "is_malicious": True}
        for column, _ in NUMERIC_FEATURE_REGISTRY:
            row[column] = 1.0
        rows.append(row)
    files = pd.DataFrame(rows)
    http = pd.DataFrame({
        "user": ["U1", "U1"], "day": pd.to_datetime(["2010-01-05", "2010-01-06"]),
        "http_new_domains_today": [0.5, 0.0], "http_new_domains_7d": [1.0, 1.0], "http_offbaseline_share_7d": [0.4, 0.2],
    })
    windows = pd.DataFrame({"user": ["U1"], "start": [_ts("2010-01-05 08:00:00")], "end": [_ts("2010-01-06 17:00:00")]})
    key_days = pd.DataFrame({"user": ["U1"], "day": pd.to_datetime(["2010-01-05"]), "answer_key_label": [1]})
    return logon, device, files, http, windows, key_days


def _build(users=("U1", "U2")):
    logon, device, files, http, windows, key_days = _raw_frames()
    return build_all_user_days(logon, device, files, http, windows, key_days, users)


def test_the_table_has_a_row_for_every_active_day_not_only_file_days():
    table = _build()
    days = set(zip(table["user"], table["day"].dt.strftime("%Y-%m-%d")))
    assert days == {("U1", "2010-01-04"), ("U1", "2010-01-05"), ("U1", "2010-01-06"), ("U1", "2010-01-07"),
                    ("U2", "2010-01-04")}          # 01-07 is a device-only day; U3 is excluded
    assert table.duplicated(DAY_KEY).sum() == 0
    assert table[DAY_KEY].equals(table[DAY_KEY].sort_values(DAY_KEY).reset_index(drop=True))


def test_email_columns_are_added_without_changing_rows_or_the_existing_columns():
    from src.common.preprocessing.email_features import EMAIL_RAW_COLUMNS
    from src.supervised.email_day_signals import EMAIL_DAY_COLUMNS
    logon, device, files, http, windows, key_days = _raw_frames()
    email = pd.DataFrame([{**{c: 0.0 for c in EMAIL_RAW_COLUMNS}, "user": u, "day": pd.Timestamp(d), "em_n_sent": 4.0}
                          for u, d in (("U1", "2010-01-05"), ("U3", "2010-01-04"))])
    plain = build_all_user_days(logon, device, files, http, windows, key_days, ("U1", "U2"))
    with_email = build_all_user_days(logon, device, files, http, windows, key_days, ("U1", "U2"), email_user_day=email)
    assert with_email[DAY_KEY].equals(plain[DAY_KEY])
    assert set(EMAIL_DAY_COLUMNS) <= set(with_email.columns) and not set(EMAIL_DAY_COLUMNS) & set(plain.columns)
    pd.testing.assert_frame_equal(with_email[plain.columns], plain)            # every old column unchanged
    assert with_email.set_index(DAY_KEY).loc[("U1", _ts("2010-01-05")), "em_n_sent"] == 4.0
    assert (with_email["user"] != "U3").all()


def test_days_without_a_file_copy_get_zero_counts_and_missing_file_summaries():
    table = _build().set_index(["user", "day"])
    no_file = table.loc[("U1", _ts("2010-01-04"))]
    assert no_file["day_n_events"] == 0 and no_file["day_has_file_events"] == 0
    assert np.isnan(no_file["seconds_since_last_event__mean"])
    with_file = table.loc[("U1", _ts("2010-01-05"))]
    assert with_file["day_n_events"] == 2 and with_file["day_has_file_events"] == 1
    assert with_file["dv_n_connects"] == 1 and with_file["dv_connected_minutes"] == 20.0


def test_device_only_day_has_zero_logon_counts_but_keeps_its_device_activity():
    row = _build().set_index(["user", "day"]).loc[("U1", _ts("2010-01-07"))]
    assert row["lg_n_logons"] == 0 and np.isnan(row["lg_first_hour"])
    assert row["dv_n_connects"] == 1


def test_window_labels_cover_whole_days_including_days_without_file_copies():
    table = _build().set_index(["user", "day"])
    assert table.loc[("U1", _ts("2010-01-04")), "window_label"] == 0
    assert table.loc[("U1", _ts("2010-01-05")), "window_label"] == 1
    assert table.loc[("U1", _ts("2010-01-06")), "window_label"] == 1      # no file copy that day, still in the window
    assert table.loc[("U1", _ts("2010-01-07")), "window_label"] == 0
    assert table.loc[("U2", _ts("2010-01-04")), "window_label"] == 0


def test_answer_key_labels_are_attached_by_user_and_day():
    table = _build().set_index(["user", "day"])
    assert table["answer_key_label"].tolist().count(1) == 1
    assert table.loc[("U1", _ts("2010-01-05")), "answer_key_label"] == 1


def test_http_features_join_by_day_with_zero_for_unseen_days():
    table = _build().set_index(["user", "day"])
    assert table.loc[("U1", _ts("2010-01-05")), "http_offbaseline_share_7d"] == 0.4
    assert table.loc[("U2", _ts("2010-01-04")), "http_offbaseline_share_7d"] == 0.0
    assert table.loc[("U1", _ts("2010-01-04")), "http_new_domains_7d"] == 0.0


def test_only_the_requested_users_are_included():
    assert set(_build(users=("U2",))["user"]) == {"U2"}


def test_every_input_column_exists_belongs_to_one_group_and_is_not_a_label():
    table = _build()
    columns = all_days_feature_columns(available_groups())
    assert len(columns) == len(set(columns))
    for column in columns:
        assert column in table.columns, column
        assert all_days_group_of_column(column) in available_groups()
    assert "window_label" not in columns and "answer_key_label" not in columns
    assert set(all_days_feature_columns(("file",))) == set(ALL_DAYS_COLUMNS_BY_GROUP["file"])
    with pytest.raises(ValueError):
        all_days_feature_columns(("device",))                 # the file group cannot be dropped


def test_without_columns_removes_only_the_named_inputs_and_rejects_unknown_names():
    from src.supervised.feature_sets import ALL_DAYS, without_columns
    groups = ("file", "device", "logon", "http")
    ablated = without_columns(ALL_DAYS, ["dv_prior_use_days"])
    before, after = ALL_DAYS.columns_fn(groups), ablated.columns_fn(groups)
    assert "dv_prior_use_days" in before and "dv_prior_use_days" not in after
    assert [c for c in before if c != "dv_prior_use_days"] == after
    assert ablated.heuristics == ALL_DAYS.heuristics
    assert without_columns(ALL_DAYS, []).columns_fn(groups) == before
    with pytest.raises(ValueError):
        without_columns(ALL_DAYS, ["not_a_column"])


def test_inputs_are_not_modified():
    logon, device, files, http, windows, key_days = _raw_frames()
    before = [f.copy() for f in (logon, device, files, http, windows, key_days)]
    build_all_user_days(logon, device, files, http, windows, key_days, ("U1", "U2"))
    for original, copy in zip((logon, device, files, http, windows, key_days), before):
        pd.testing.assert_frame_equal(original, copy)


def test_window_labels_for_days_handles_several_windows():
    days = pd.DataFrame({"user": ["A", "A", "B", "B"], "day": pd.to_datetime(["2010-01-01", "2010-01-10"] * 2)})
    windows = pd.DataFrame({"user": ["A", "B"], "start": [_ts("2010-01-09 10:00:00"), _ts("2010-01-01 00:00:00")],
                            "end": [_ts("2010-01-12 00:00:00"), _ts("2010-01-02 00:00:00")]})
    assert window_labels_for_days(days, windows).tolist() == [0, 1, 1, 0]


# ---------------- the experiment with the all-days feature set ----------------

def _synthetic_all_days_table(seed=0):
    rng = np.random.default_rng(seed)
    columns = all_days_feature_columns(available_groups())
    rows, scenario_by_user = [], {}
    for u in range(24):
        user = f"U{u:02d}"
        insider = u < 12
        if insider:
            scenario_by_user[user] = 1 + u % 3
        start = (14 + 7 * (u % 3)) if insider else 99
        for d in range(40):
            malicious = int(insider and d >= start)
            row = {c: float(rng.normal()) for c in columns}
            if malicious:
                row["http_offbaseline_share_7d"] += 3.0
                row["dv_n_connects_logratio"] += 3.0
            if d < 3:                                   # no baseline yet: the real columns are NaN here
                row["dv_n_connects_logratio"] = np.nan
                row["lg_after_hours_logratio"] = np.nan
            row.update({"user": user, "day": pd.Timestamp("2010-06-01") + pd.Timedelta(days=d),
                        "window_label": malicious, "answer_key_label": int(malicious and d % 2 == 0)})
            rows.append(row)
    return pd.DataFrame(rows), scenario_by_user


@pytest.fixture(scope="module")
def all_days_experiment():
    table, scenarios = _synthetic_all_days_table()
    return run_experiment(
        table, scenarios, model_names=("logistic",), config_names=("file+device+logon", "all"),
        n_folds=4, n_repeats=1, n_boot=40, seed=0, feature_set=ALL_DAYS,
    )


def test_all_days_feature_set_reports_its_own_heuristics_and_labels(all_days_experiment):
    results, markdown = all_days_experiment
    models = results["labelings"]["window"]["models"]
    for key in ALL_DAYS.heuristics:
        assert key in models
    assert "whether or not the user copied a file that day" in markdown
    assert "Insiders only, date-stratified" in markdown


def test_heuristic_scores_with_missing_baselines_are_scored_not_dropped(all_days_experiment):
    results, _ = all_days_experiment
    models = results["labelings"]["window"]["models"]
    assert models["device_rate"]["auroc"]["mean"] > 0.8        # the planted device-rate signal
    assert models["single_feature_http"]["auroc"]["mean"] > 0.8
    assert 0.3 < models["after_hours_rate"]["auroc"]["mean"] < 0.7   # nothing planted there


def test_the_default_feature_set_is_unchanged():
    assert set(FILE_EVENT_DAYS.heuristics) == {"single_feature_http", "day_event_count"}
