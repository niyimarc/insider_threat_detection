import numpy as np
import pandas as pd
import pytest

from src.supervised.dataset import WINDOW_LABEL
from src.supervised.user_level import (
    alert_analysis,
    precision_at_k,
    ranked_user_table,
    user_level_auroc,
    user_scores,
    windows_from_labels,
)


def test_user_scores_take_the_mean_of_the_highest_days_or_the_single_highest():
    users = ["A"] * 5 + ["B"] * 2
    scores = [1, 9, 5, 3, 7, 4, 2]
    top = user_scores(users, scores, k=2)
    assert top["A"] == pytest.approx((9 + 7) / 2) and top["B"] == pytest.approx((4 + 2) / 2)   # B has only 2 days
    assert user_scores(users, scores, rule="max").to_dict() == {"A": 9.0, "B": 4.0}
    with pytest.raises(ValueError):
        user_scores(users, scores, rule="median")


def test_user_level_auroc_is_one_for_perfect_separation_zero_for_reversed_and_pairs_models():
    insiders = [f"I{i}" for i in range(6)]
    benign = [f"B{i}" for i in range(4)]
    good = pd.Series({**{u: 10.0 + i for i, u in enumerate(insiders)}, **{u: float(i) for i, u in enumerate(benign)}})
    bad = pd.Series({**{u: float(i) for i, u in enumerate(insiders)}, **{u: 10.0 + i for i, u in enumerate(benign)}})
    tied = pd.Series({u: 1.0 for u in insiders + benign})
    res = user_level_auroc({"good": good, "bad": bad, "tied": tied}, insiders, n_boot=50, seed=0, reference="tied")
    assert res["good"]["auroc"] == 1.0 and res["good"]["ci_low"] == 1.0
    assert res["bad"]["auroc"] == 0.0
    assert res["tied"]["auroc"] == 0.5
    assert res["good"]["auroc_minus_reference"]["mean"] == pytest.approx(0.5)
    assert res["good"]["auroc_minus_reference"]["ci_excludes_zero"] is True
    assert "auroc_minus_reference" not in res["tied"]


def test_user_level_auroc_without_both_groups_is_nan_not_an_error():
    res = user_level_auroc({"m": pd.Series({"A": 1.0, "B": 2.0})}, insider_users=["A", "B"])
    assert np.isnan(res["m"]["auroc"])


def _matched_period_data():
    days = pd.date_range("2010-01-01", periods=10)
    rows, good = [], []
    def add(user, score_of_day):
        for d, day in enumerate(days, start=1):
            rows.append((user, day))
            good.append(score_of_day(d))
    add("I1", lambda d: 10.0 if 3 <= d <= 5 else 0.0)
    add("I2", lambda d: 10.0 if 6 <= d <= 8 else 0.0)
    add("B1", lambda d: 5.0)
    add("B2", lambda d: 20.0 if 6 <= d <= 8 else 0.0)
    frame = pd.DataFrame(rows, columns=["user", "day"])
    windows = {"I1": (days[2], days[4]), "I2": (days[5], days[7])}
    return frame, np.array(good), windows


def test_matched_period_compares_each_insider_with_benign_users_on_the_same_dates():
    from src.supervised.user_level import matched_period_auroc
    frame, good, windows = _matched_period_data()
    res = matched_period_auroc(frame["user"], frame["day"], {"good": good, "flat": np.ones(len(good))},
                               windows, n_boot=40)
    # I1 beats B1 and B2 on days 3-5; I2 beats B1 but loses to B2 on days 6-8 -> 3 of 4 pairs
    assert res["good"]["auroc"] == pytest.approx(0.75)
    assert res["flat"]["auroc"] == pytest.approx(0.5)               # ties count one half
    assert res["good"]["coverage"] == 1.0


def test_a_score_that_depends_only_on_the_date_cannot_separate_insiders_in_the_matched_period_check():
    from src.supervised.user_level import matched_period_auroc
    frame, _, windows = _matched_period_data()
    date_only = (frame["day"] - frame["day"].min()).dt.days.to_numpy().astype(float)
    res = matched_period_auroc(frame["user"], frame["day"], {"date": date_only}, windows, n_boot=40)
    assert res["date"]["auroc"] == pytest.approx(0.5)


def test_matched_period_skips_benign_users_without_enough_days_in_the_window():
    from src.supervised.user_level import matched_period_auroc
    frame, good, windows = _matched_period_data()
    extra = pd.DataFrame({"user": ["B3"], "day": [pd.Timestamp("2010-01-01")]})        # B3 is active on day 1 only
    frame2 = pd.concat([frame, extra], ignore_index=True)
    good2 = np.r_[good, 99.0]
    res = matched_period_auroc(frame2["user"], frame2["day"], {"g": good2}, windows, n_boot=40)
    assert res["g"]["auroc"] == pytest.approx(0.75)                  # B3 contributes nothing
    assert res["g"]["coverage"] == pytest.approx(4 / 6)


def test_matched_period_paired_difference_and_missing_groups():
    from src.supervised.user_level import matched_period_auroc
    frame, good, windows = _matched_period_data()
    res = matched_period_auroc(frame["user"], frame["day"], {"good": good, "flat": np.ones(len(good))},
                               windows, n_boot=40, reference="flat")
    assert res["good"]["auroc_minus_reference"]["mean"] > 0.1       # about 0.25; resampling moves it a little
    none = matched_period_auroc(frame["user"], frame["day"], {"g": good}, windows={}, n_boot=10)
    assert np.isnan(none["g"]["auroc"])


def test_precision_at_k_reports_the_random_baseline():
    ranked = pd.Series({"I1": 9, "B1": 8, "I2": 7, "B2": 1, "B3": 0, "I3": -1}, dtype=float)
    p = precision_at_k(ranked, {"I1", "I2", "I3"}, ks=(2, 3))
    assert p[2]["insiders_in_top_k"] == 1 and p[3]["insiders_in_top_k"] == 2
    assert p[2]["baseline_share"] == pytest.approx(0.5)


def test_windows_come_from_the_first_and_last_labelled_day():
    users = ["A", "A", "A", "B"]
    days = pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-05", "2010-01-01"])
    windows = windows_from_labels(users, days, [0, 1, 1, 0])
    assert windows == {"A": (pd.Timestamp("2010-01-02"), pd.Timestamp("2010-01-05"))}


def _alert_data():
    rows = []
    for user in ("B1", "B2"):                                    # benign: scores 0..9 over 10 days
        for d in range(10):
            rows.append((user, pd.Timestamp("2010-01-01") + pd.Timedelta(days=d), float(d)))
    for d in range(10):                                          # I1: window days 4-7, big scores from day 6
        rows.append(("I1", pd.Timestamp("2010-01-01") + pd.Timedelta(days=d), 100.0 if d in (2, 6, 7) else 0.0))
    for d in range(10):                                          # I2: never scores high
        rows.append(("I2", pd.Timestamp("2010-01-01") + pd.Timedelta(days=d), 0.5))
    frame = pd.DataFrame(rows, columns=["user", "day", "score"])
    windows = {"I1": (pd.Timestamp("2010-01-05"), pd.Timestamp("2010-01-08")),
               "I2": (pd.Timestamp("2010-01-05"), pd.Timestamp("2010-01-08"))}
    return frame, windows


def test_alert_cutoff_comes_from_benign_users_only_and_the_summary_counts_are_right():
    frame, windows = _alert_data()
    a = alert_analysis(frame["user"], frame["day"], frame["score"], windows, quantile=0.9)
    assert a["threshold"] == pytest.approx(np.quantile(np.r_[np.arange(10.0), np.arange(10.0)], 0.9))
    s = a["summary"]
    assert (s["n_insiders"], s["n_benign"], s["insiders_detected"]) == (2, 2, 1)
    assert s["detection_rate"] == 0.5
    assert a["per_user"]["I1"]["days_to_first_alert"] == 2            # window starts Jan 5, first in-window alert Jan 7
    assert a["per_user"]["I1"]["alert_before_window"] is True         # the day-2 spike precedes the window
    assert a["per_user"]["I2"]["first_alert_in_window"] is None
    assert s["insiders_alerted_before_window"] == 1
    assert s["benign_users_alerted"] == 2                             # each benign user has a day at the 90th percentile


def test_alert_analysis_needs_benign_users():
    frame, windows = _alert_data()
    windows = {**windows, "B1": (pd.Timestamp("2010-01-01"), pd.Timestamp("2010-01-02")),
               "B2": (pd.Timestamp("2010-01-01"), pd.Timestamp("2010-01-02"))}
    with pytest.raises(ValueError):
        alert_analysis(frame["user"], frame["day"], frame["score"], windows)


def test_ranked_table_orders_by_score_and_marks_insiders():
    scores = pd.Series({"A": 0.2, "B": 0.9, "C": 0.5})
    frame, windows = _alert_data()
    alerts = alert_analysis(frame["user"], frame["day"], frame["score"], windows, quantile=0.9)
    table = ranked_user_table(scores, alerts, {"A": 2})
    assert table["user"].tolist() == ["B", "C", "A"] and table["rank"].tolist() == [1, 2, 3]
    assert table.set_index("user").loc["A", "is_insider"] and table.set_index("user").loc["A", "scenario"] == 2
    assert not table.set_index("user").loc["B", "is_insider"]
