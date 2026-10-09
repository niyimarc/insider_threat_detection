import numpy as np
import pandas as pd
import pytest

from src.common.preprocessing.email_features import EMAIL_RAW_COLUMNS
from src.supervised.email_day_signals import EMAIL_BASELINE_COLUMNS, EMAIL_DAY_COLUMNS, add_email_features


def _table(n_days=12):
    days = pd.date_range("2010-01-04", periods=n_days)
    return pd.DataFrame({"user": ["U1"] * n_days, "day": days})


def _email(n_sent_by_day):
    rows = []
    for day, n in n_sent_by_day.items():
        row = {c: 0.0 for c in EMAIL_RAW_COLUMNS}
        row.update({"user": "U1", "day": pd.Timestamp(day), "em_n_sent": float(n), "em_n_recipients": 2.0 * n,
                    "em_n_distinct_recipients": 2.0, "em_n_new_recipients": 1.0})
        rows.append(row)
    return pd.DataFrame(rows)


def test_columns_are_added_without_adding_or_dropping_rows():
    table = _table()
    out = add_email_features(table, _email({"2010-01-05": 3, "2010-02-20": 1}))     # the 2nd email day is not a table row
    assert len(out) == len(table)
    assert set(EMAIL_DAY_COLUMNS) <= set(out.columns)
    assert out.loc[out["day"] == "2010-01-05", "em_n_sent"].item() == 3.0
    assert out.loc[out["day"] == "2010-01-06", "em_n_sent"].item() == 0.0            # no email: neutral zero


def test_new_recipient_share_is_new_over_distinct_and_zero_on_quiet_days():
    out = add_email_features(_table(), _email({"2010-01-05": 3}))
    day = out.set_index("day")
    assert day.loc["2010-01-05", "em_new_recipient_share"] == pytest.approx(0.5)
    assert day.loc["2010-01-06", "em_new_recipient_share"] == 0.0


def test_baselines_use_only_previous_days_and_wait_for_history():
    base = {f"2010-01-{d:02d}": 2 for d in range(4, 12)}                  # 2 emails a day for 8 days
    out = add_email_features(_table(), _email({**base, "2010-01-12": 20}))
    day = out.set_index("day")
    assert day.loc["2010-01-08", "em_n_sent_logratio"] != day.loc["2010-01-08", "em_n_sent_logratio"]   # NaN: < 7 earlier days
    spike = day.loc["2010-01-12", "em_n_sent_logratio"]
    assert spike == pytest.approx(np.log1p(20) - np.log1p(2.0))                                         # vs the previous 7 days
    # causality: changing a LATER day cannot change an earlier day's baseline
    later = add_email_features(_table(), _email({**base, "2010-01-12": 99, "2010-01-13": 50})).set_index("day")
    assert later.loc["2010-01-11", "em_n_sent_logratio"] == day.loc["2010-01-11", "em_n_sent_logratio"]


def test_baseline_columns_are_part_of_the_day_columns_and_unique():
    assert set(EMAIL_BASELINE_COLUMNS) <= set(EMAIL_DAY_COLUMNS)
    assert len(EMAIL_DAY_COLUMNS) == len(set(EMAIL_DAY_COLUMNS))
    assert not set(EMAIL_DAY_COLUMNS) & {"window_label", "answer_key_label"}


def test_input_frames_are_not_modified():
    table, email = _table(), _email({"2010-01-05": 3})
    before = (table.copy(), email.copy())
    add_email_features(table, email)
    pd.testing.assert_frame_equal(table, before[0])
    pd.testing.assert_frame_equal(email, before[1])
