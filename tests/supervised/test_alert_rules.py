import numpy as np
import pandas as pd
import pytest

from src.supervised.alert_rules import (
    apply_alert_rule, assign_user_folds, cross_fitted_alert_analysis, rank_average,
)
from src.supervised.user_level import alert_analysis, windows_from_labels


def test_rank_average_is_the_mean_of_percentile_ranks_and_ignores_the_scale():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([40.0, 10.0, 30.0, 20.0]) * 1000
    expected = (np.array([0.25, 0.5, 0.75, 1.0]) + np.array([1.0, 0.25, 0.75, 0.5])) / 2
    np.testing.assert_allclose(rank_average([a, b]), expected)
    np.testing.assert_allclose(rank_average([a, a * 10 + 3]), rank_average([a]))
    with pytest.raises(ValueError):
        rank_average([a, a[:2]])


def test_single_rule_is_the_raw_flags_and_persist_needs_two_of_the_last_three_active_days():
    users = np.array(["u"] * 6 + ["v"] * 3)
    days = pd.to_datetime(["2010-01-0%d" % d for d in (1, 2, 3, 4, 5, 6)] + ["2010-01-01", "2010-01-02", "2010-01-03"])
    raw = np.array([1, 0, 0, 1, 1, 0,   1, 0, 0], dtype=bool)
    np.testing.assert_array_equal(apply_alert_rule(users, days, raw, "single"), raw)
    persist = apply_alert_rule(users, days, raw, "persist_2of3")
    # u: day1 isolated; day4 isolated (days 2,3 not flagged); day5 flagged and day4 flagged -> yes; day6 not flagged
    np.testing.assert_array_equal(persist, [False, False, False, False, True, False, False, False, False])
    assert (persist <= raw).all()
    with pytest.raises(ValueError):
        apply_alert_rule(users, days, raw, "other")


def test_persist_rule_respects_input_order_and_users_independently():
    users = np.array(["b", "a", "b", "a", "b"])
    days = pd.to_datetime(["2010-01-03", "2010-01-01", "2010-01-01", "2010-01-02", "2010-01-02"])
    raw = np.array([1, 1, 1, 1, 0], dtype=bool)                  # a: days 1,2 flagged; b: days 1,3 flagged
    out = apply_alert_rule(users, days, raw, "persist_2of3")
    # a's second flagged day (01-02) alerts; b: 01-03 has flags on 01-01 and 01-03 -> alerts
    np.testing.assert_array_equal(out, [True, False, False, True, False])


def test_folds_split_insiders_and_benign_users_evenly_and_reproducibly():
    users = [f"I{i}" for i in range(10)] + [f"B{i}" for i in range(15)]
    folds = assign_user_folds(users, [f"I{i}" for i in range(10)], n_folds=5, seed=0)
    assert set(folds.values()) == set(range(5))
    for f in range(5):
        members = [u for u, v in folds.items() if v == f]
        assert sum(u.startswith("I") for u in members) == 2 and sum(u.startswith("B") for u in members) == 3
    assert folds == assign_user_folds(users, [f"I{i}" for i in range(10)], n_folds=5, seed=0)


def _toy(n_benign=10, n_insiders=5, n_days=20, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i in range(n_benign + n_insiders):
        user = f"B{i}" if i < n_benign else f"I{i}"
        for d in range(n_days):
            insider_day = user.startswith("I") and d >= 10
            rows.append((user, pd.Timestamp("2010-01-01") + pd.Timedelta(days=d),
                         rng.normal() + (6.0 if insider_day else 0.0)))
            labels.append(int(insider_day))
    frame = pd.DataFrame(rows, columns=["user", "day", "score"])
    windows = windows_from_labels(frame["user"], frame["day"], labels)
    return frame, windows


def test_cross_fitted_cutoffs_use_only_benign_users_of_other_folds():
    frame, windows = _toy()
    result = cross_fitted_alert_analysis(frame["user"], frame["day"], frame["score"], windows, 0.95, "single")
    folds = assign_user_folds(frame["user"], windows, n_folds=5, seed=0)
    benign = ~frame["user"].isin(windows).to_numpy()
    fold_of_row = frame["user"].map(folds).to_numpy()
    for f, threshold in result["fold_thresholds"].items():
        expected = np.quantile(frame["score"].to_numpy()[benign & (fold_of_row != f)], 0.95)
        assert threshold == pytest.approx(expected)
    assert result["summary"]["n_insiders"] == 5 and result["summary"]["n_benign"] == 10
    assert result["summary"]["insiders_detected"] == 5                       # planted shift of 6 sd


def test_persistence_cannot_raise_the_number_of_alert_days_and_lowers_false_alarms():
    frame, windows = _toy(n_benign=20, seed=1)
    single = cross_fitted_alert_analysis(frame["user"], frame["day"], frame["score"], windows, 0.9, "single")
    persist = cross_fitted_alert_analysis(frame["user"], frame["day"], frame["score"], windows, 0.9, "persist_2of3")
    s, p = single["summary"], persist["summary"]
    assert p["benign_alert_days_per_user"] < s["benign_alert_days_per_user"]       # strictly: some isolated flags exist
    assert p["benign_alert_days_per_user"] <= s["benign_alert_days_per_user"]
    assert p["benign_users_alerted"] <= s["benign_users_alerted"]
    assert p["insiders_detected"] >= 4                                        # still catches the planted shift


def test_alert_analysis_accepts_explicit_flags_and_keeps_default_behaviour():
    frame, windows = _toy()
    default = alert_analysis(frame["user"], frame["day"], frame["score"], windows)
    flags = (frame["score"] >= default["threshold"]).to_numpy()
    explicit = alert_analysis(frame["user"], frame["day"], frame["score"], windows, alert_flags=flags)
    assert explicit["summary"] == default["summary"] and np.isnan(explicit["threshold"])
    with pytest.raises(ValueError):
        alert_analysis(frame["user"], frame["day"], frame["score"], windows, alert_flags=flags[:-1])
