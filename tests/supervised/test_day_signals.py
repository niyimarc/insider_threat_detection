import numpy as np
import pandas as pd
import pytest

from src.supervised.day_signals import (
    BASELINE_WINDOW,
    DEVICE_DAY_COLUMNS,
    FIRST_EVENT_MIN_HISTORY,
    LOGON_DAY_COLUMNS,
    MIN_HISTORY,
    add_causal_baselines,
    device_day_counts,
    logon_day_counts,
)


def _logons():
    rows = [
        # Mon 2010-01-04: ordinary day
        ("U1", "PC-1", "2010-01-04 08:00:00", "Logon"), ("U1", "PC-1", "2010-01-04 17:00:00", "Logoff"),
        # Tue 2010-01-05: early logon on another PC, evening logon on the usual PC
        ("U1", "PC-9", "2010-01-05 06:00:00", "Logon"), ("U1", "PC-1", "2010-01-05 20:30:00", "Logon"),
        ("U1", "PC-1", "2010-01-05 21:00:00", "Logoff"),
        # Sat 2010-01-09
        ("U1", "PC-1", "2010-01-09 10:00:00", "Logon"),
    ]
    return pd.DataFrame(rows, columns=["user", "pc", "date", "activity"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_logon_day_counts():
    out = logon_day_counts(_logons()).set_index("day")
    ordinary = out.loc[pd.Timestamp("2010-01-04")]
    assert ordinary["lg_n_logons"] == 1 and ordinary["lg_first_hour"] == 8.0 and ordinary["lg_last_hour"] == 17.0
    assert ordinary["lg_span_hours"] == 9.0 and ordinary["lg_n_distinct_pcs"] == 1
    assert ordinary["lg_after_hours_events"] == 0 and ordinary["lg_other_pc_events"] == 0 and ordinary["lg_is_weekend"] == 0

    odd = out.loc[pd.Timestamp("2010-01-05")]
    assert odd["lg_n_logons"] == 2 and odd["lg_first_hour"] == 6.0 and odd["lg_last_hour"] == 21.0
    assert odd["lg_after_hours_events"] == 3          # 06:00, 20:30 and 21:00 are outside 07:00-19:00
    assert odd["lg_n_distinct_pcs"] == 2
    assert odd["lg_other_pc_events"] == 1             # the 06:00 logon is on PC-9, not the user's primary PC-1

    assert out.loc[pd.Timestamp("2010-01-09"), "lg_is_weekend"] == 1
    assert set(logon_day_counts(_logons()).columns) == {"user", "day", *LOGON_DAY_COLUMNS[:8]}


def test_logon_day_counts_does_not_modify_its_input():
    frame = _logons()
    before = frame.copy()
    logon_day_counts(frame)
    pd.testing.assert_frame_equal(frame, before)


def _devices():
    rows = [
        ("U1", "PC-1", "2010-01-05 09:00:00", "Connect"), ("U1", "PC-1", "2010-01-05 09:30:00", "Disconnect"),
        ("U1", "PC-1", "2010-01-05 20:00:00", "Connect"),                       # never disconnected, after hours
        ("U1", "PC-1", "2010-01-06 08:00:00", "Disconnect"),                    # closes the overnight connect: 12 h
        ("U1", "PC-1", "2010-01-07 08:00:00", "Disconnect"),                    # stray disconnect with no connect
    ]
    return pd.DataFrame(rows, columns=["user", "pc", "date", "activity"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_device_day_counts_pairs_connects_with_disconnects():
    out = device_day_counts(_devices()).set_index("day")
    first = out.loc[pd.Timestamp("2010-01-05")]
    assert first["dv_n_connects"] == 2 and first["dv_after_hours_connects"] == 1
    assert first["dv_n_distinct_pcs"] == 1
    assert first["dv_connected_minutes"] == 30.0
    assert first["dv_unmatched_connects"] == 1          # the 20:00 connect had no disconnect that day

    overnight = out.loc[pd.Timestamp("2010-01-06")]
    assert overnight["dv_n_connects"] == 0 and overnight["dv_connected_minutes"] == 720.0   # 12 h, attributed to the disconnect's day

    stray = out.loc[pd.Timestamp("2010-01-07")]
    assert stray["dv_connected_minutes"] == 0.0 and stray["dv_n_connects"] == 0


def test_a_very_long_connection_is_capped():
    frame = pd.DataFrame(
        [("U1", "PC-1", "2010-01-05 09:00:00", "Connect"), ("U1", "PC-1", "2010-01-09 09:00:00", "Disconnect")],
        columns=["user", "pc", "date", "activity"],
    ).assign(date=lambda d: pd.to_datetime(d["date"]))
    out = device_day_counts(frame)
    assert out["dv_connected_minutes"].max() == 720.0


# ---------------- causal baselines ----------------

def _table(n_days=40, user="U1", **overrides):
    days = pd.date_range("2010-02-01", periods=n_days, freq="D")
    data = {
        "user": user, "day": days,
        "lg_n_logons": 2.0, "lg_first_hour": 8.0, "lg_span_hours": 9.0, "lg_after_hours_events": 0.0,
        "dv_n_connects": 1.0, "dv_connected_minutes": 30.0, "dv_after_hours_connects": 0.0,
    }
    frame = pd.DataFrame({k: (v if k in ("day",) else v) for k, v in data.items()})
    for column, values in overrides.items():
        frame[column] = values
    return frame


def test_baselines_are_nan_until_enough_history_and_zero_for_constant_users():
    out = add_causal_baselines(_table())
    assert out["lg_first_hour_dev"].iloc[:MIN_HISTORY].isna().all()
    assert out["lg_first_hour_dev"].iloc[MIN_HISTORY:].eq(0.0).all()
    assert out["dv_n_connects_logratio"].iloc[MIN_HISTORY:].eq(0.0).all()
    # a constant user must give finite values, never an exploding z-score
    assert np.isfinite(out[["lg_first_hour_dev", "lg_span_dev", "lg_n_logons_logratio", "dv_connected_minutes_dev"]]
                       .iloc[MIN_HISTORY:].to_numpy()).all()


def test_a_spike_stands_out_against_the_users_own_previous_days():
    connects = np.ones(40)
    connects[30] = 9.0
    out = add_causal_baselines(_table(dv_n_connects=connects))
    assert out["dv_n_connects_logratio"].iloc[30] == pytest.approx(np.log1p(9.0) - np.log1p(1.0))
    assert out["dv_n_connects_logratio"].iloc[29] == 0.0
    # the spike enters later baselines only after it has happened
    assert out["dv_n_connects_logratio"].iloc[31] < 0


def test_baselines_never_use_the_future():
    base = add_causal_baselines(_table())
    changed_future = _table().copy()
    changed_future.loc[30:, "dv_n_connects"] = 50.0
    changed_future.loc[30:, "lg_first_hour"] = 3.0
    after = add_causal_baselines(changed_future)
    columns = ["dv_n_connects_logratio", "lg_first_hour_dev", "dv_prior_use_days", "dv_first_use_today"]
    pd.testing.assert_frame_equal(base.loc[:29, columns], after.loc[:29, columns])


def test_first_ever_device_use_fires_once_and_only_after_the_warmup():
    first_use_day = FIRST_EVENT_MIN_HISTORY + 6
    connects = np.zeros(40)
    connects[first_use_day:] = 1.0
    out = add_causal_baselines(_table(dv_n_connects=connects))
    flagged = np.flatnonzero(out["dv_first_use_today"].to_numpy() == 1.0)
    assert flagged.tolist() == [first_use_day]
    assert out["dv_prior_use_days"].iloc[first_use_day + 3] == 3

    early = np.zeros(40)
    early[3:] = 1.0                                       # a drive "first used" while there is almost no history
    assert add_causal_baselines(_table(dv_n_connects=early))["dv_first_use_today"].sum() == 0


def test_first_after_hours_day_fires_once():
    after_hours = np.zeros(40)
    after_hours[25] = 2.0
    after_hours[30] = 1.0
    out = add_causal_baselines(_table(lg_after_hours_events=after_hours))
    assert np.flatnonzero(out["lg_first_after_hours_day"].to_numpy() == 1.0).tolist() == [25]


def test_users_do_not_share_baselines():
    quiet = _table(user="A")
    busy = _table(user="B", dv_n_connects=10.0)
    out = add_causal_baselines(pd.concat([quiet, busy], ignore_index=True))
    for user in ("A", "B"):
        rows = out[out["user"] == user]
        assert rows["dv_n_connects_logratio"].iloc[MIN_HISTORY:].eq(0.0).all()


def test_all_baseline_and_raw_columns_are_produced():
    out = add_causal_baselines(_table())
    for column in LOGON_DAY_COLUMNS[8:] + DEVICE_DAY_COLUMNS[5:]:
        assert column in out.columns
    assert BASELINE_WINDOW >= MIN_HISTORY
