import numpy as np
import pandas as pd
import pytest

from src.common.preprocessing.http_features import (
    HTTP_FEATURE_COLUMNS,
    build_http_user_day_features,
    add_http_features,
)


def _domain_daily(rows):
    """rows: (user, 'YYYY-MM-DD', domain, visits)"""
    return pd.DataFrame(rows, columns=["user", "day", "domain", "visits"]).assign(
        day=lambda d: pd.to_datetime(d["day"])
    )


def _day(table, user, day):
    row = table[(table["user"] == user) & (table["day"] == pd.Timestamp(day))]
    assert len(row) == 1
    return row.iloc[0]


def _steady_user(user="U1", days=40, domains=("a.com", "b.com")):
    """A user who browses the same two domains every day from 2010-01-01."""
    return [
        (user, d.strftime("%Y-%m-%d"), dom, 10)
        for d in pd.date_range("2010-01-01", periods=days)
        for dom in domains
    ]


def test_domains_first_seen_during_warmup_do_not_count_as_new():
    table = build_http_user_day_features(_domain_daily(_steady_user()), warmup_days=14)
    assert (table["http_new_domains_today"] == 0).all()
    assert (table["http_new_domains_7d"] == 0).all()


def test_a_domain_first_visited_after_warmup_counts_as_new_on_that_day_only():
    rows = _steady_user() + [("U1", "2010-02-01", "newsite.com", 3)]
    table = build_http_user_day_features(_domain_daily(rows), warmup_days=14)
    assert _day(table, "U1", "2010-02-01")["http_new_domains_today"] == pytest.approx(np.log1p(1))
    assert _day(table, "U1", "2010-01-31")["http_new_domains_today"] == 0
    # revisiting it the next day is no longer new
    rows += [("U1", "2010-02-02", "newsite.com", 3)]
    table = build_http_user_day_features(_domain_daily(rows), warmup_days=14)
    assert _day(table, "U1", "2010-02-02")["http_new_domains_today"] == 0


def test_trailing_window_keeps_novelty_for_seven_days_then_drops_it():
    rows = _steady_user(days=50) + [("U1", "2010-02-01", "newsite.com", 3)]
    table = build_http_user_day_features(_domain_daily(rows), warmup_days=14, trailing_days=7)
    for offset, expected in [(0, 1), (3, 1), (6, 1), (7, 0)]:
        day = pd.Timestamp("2010-02-01") + pd.Timedelta(days=offset)
        assert _day(table, "U1", day)["http_new_domains_7d"] == pytest.approx(np.log1p(expected))


def test_novelty_features_use_only_the_past():
    # Adding browsing AFTER a given day must not change that day's novelty values.
    base = _steady_user(days=40) + [("U1", "2010-02-01", "newsite.com", 3)]
    later = base + [("U1", "2010-02-05", "other-new.com", 9)]
    t_base = build_http_user_day_features(_domain_daily(base), warmup_days=14)
    t_later = build_http_user_day_features(_domain_daily(later), warmup_days=14)
    for day in ["2010-01-20", "2010-01-31", "2010-02-01", "2010-02-02"]:
        for column in ("http_new_domains_today", "http_new_domains_7d"):
            assert _day(t_base, "U1", day)[column] == _day(t_later, "U1", day)[column]


def test_users_are_independent():
    rows = _steady_user("U1") + _steady_user("U2") + [("U2", "2010-02-01", "newsite.com", 3)]
    table = build_http_user_day_features(_domain_daily(rows), warmup_days=14)
    assert _day(table, "U1", "2010-02-01")["http_new_domains_today"] == 0
    assert _day(table, "U2", "2010-02-01")["http_new_domains_today"] > 0


def _share_user(extra_days_from="2010-02-10", days=60, extra_visits=5):
    """5 baseline domains every day; from `extra_days_from` also a new site."""
    rows = []
    for d in pd.date_range("2010-01-01", periods=days):
        for k in range(5):
            rows.append(("U1", d.strftime("%Y-%m-%d"), f"d{k}.com", 10))
        if d >= pd.Timestamp(extra_days_from):
            rows.append(("U1", d.strftime("%Y-%m-%d"), "newjob.com", extra_visits))
    return rows


def test_offbaseline_share_is_zero_for_a_user_who_only_visits_baseline_domains():
    table = build_http_user_day_features(_domain_daily(_steady_user()))
    assert (table["http_offbaseline_share_7d"] == 0).all()


def test_offbaseline_share_stays_high_while_a_new_site_keeps_being_visited():
    table = build_http_user_day_features(_domain_daily(_share_user()), warmup_days=14)
    # After a full week of daily visits the 7-day share is 5 / (50 + 5)
    late = _day(table, "U1", "2010-02-20")["http_offbaseline_share_7d"]
    assert late == pytest.approx(5 / 55)
    # ...whereas the NOVELTY features have long since returned to 0 for today
    assert _day(table, "U1", "2010-02-20")["http_new_domains_today"] == 0
    assert _day(table, "U1", "2010-02-20")["http_offbaseline_share_7d"] > 0


def test_offbaseline_share_decays_to_zero_after_the_visits_stop():
    rows = [r for r in _share_user() if not (r[2] == "newjob.com" and r[1] > "2010-02-12")]
    table = build_http_user_day_features(_domain_daily(rows), warmup_days=14)
    assert _day(table, "U1", "2010-02-12")["http_offbaseline_share_7d"] > 0
    assert _day(table, "U1", "2010-02-19")["http_offbaseline_share_7d"] == 0  # 7 days later


def test_offbaseline_share_ignores_domains_first_seen_during_warmup():
    table = build_http_user_day_features(_domain_daily(_share_user(extra_days_from="2010-01-05")), warmup_days=14)
    assert (table["http_offbaseline_share_7d"] == 0).all()


def test_offbaseline_share_is_always_between_zero_and_one():
    heavy = _share_user(extra_days_from="2010-02-10", days=60, extra_visits=500)
    values = build_http_user_day_features(_domain_daily(heavy), warmup_days=14)["http_offbaseline_share_7d"]
    assert values.min() >= 0 and values.max() <= 1
    assert values.max() > 0.8  # 500 of ~550 daily visits go to the new site


def test_offbaseline_share_does_not_use_future_visits():
    short = _share_user(days=50)
    long = _share_user(days=60)
    t_short = build_http_user_day_features(_domain_daily(short), warmup_days=14)
    t_long = build_http_user_day_features(_domain_daily(long), warmup_days=14)
    for day in ["2010-01-20", "2010-02-10", "2010-02-15", "2010-02-19"]:
        assert _day(t_short, "U1", day)["http_offbaseline_share_7d"] == \
            _day(t_long, "U1", day)["http_offbaseline_share_7d"]


def test_daily_visit_volume_does_not_influence_any_feature():
    # Daily http volume is constant per user in CERT r4.2, so no feature
    # may depend on it (it could only encode generation artifacts).
    quiet = _steady_user(days=40)
    loud = [(u, d, dom, v * 50) for (u, d, dom, v) in quiet]
    a = build_http_user_day_features(_domain_daily(quiet))
    b = build_http_user_day_features(_domain_daily(loud))
    pd.testing.assert_frame_equal(a, b)


def test_inactive_calendar_days_get_neutral_values():
    rows = [("U1", "2010-01-01", "a.com", 10), ("U1", "2010-01-05", "a.com", 10),
            ("U1", "2010-01-09", "a.com", 10), ("U1", "2010-01-10", "a.com", 12)]
    table = build_http_user_day_features(_domain_daily(rows))
    gap_day = _day(table, "U1", "2010-01-03")
    assert gap_day["http_new_domains_today"] == 0 and gap_day["http_new_domains_7d"] == 0


def test_output_has_exactly_the_registered_columns():
    table = build_http_user_day_features(_domain_daily(_steady_user()))
    assert list(table.columns) == ["user", "day", *HTTP_FEATURE_COLUMNS]
    assert not table[list(HTTP_FEATURE_COLUMNS)].isna().any().any()


def test_http_feature_columns_match_the_registry():
    from src.common.preprocessing.feature_groups import NUMERIC_FEATURE_REGISTRY
    registered = [c for c, g in NUMERIC_FEATURE_REGISTRY if g == "http"]
    assert registered == list(HTTP_FEATURE_COLUMNS)


def test_add_http_features_matches_on_user_and_day_and_preserves_row_order():
    rows = _steady_user(days=40) + [("U1", "2010-02-01", "newsite.com", 3)]
    table = build_http_user_day_features(_domain_daily(rows), warmup_days=14)
    file_df = pd.DataFrame({
        "user": ["U1", "U1", "U2"],
        "date": pd.to_datetime(["2010-02-01 23:59:00", "2010-01-20 08:00:00", "2010-02-01 10:00:00"]),
        "filename": ["x", "y", "z"],
    })
    result = add_http_features(file_df, table)
    assert result["filename"].tolist() == ["x", "y", "z"]
    assert result.loc[0, "http_new_domains_today"] == pytest.approx(np.log1p(1))
    assert result.loc[1, "http_new_domains_today"] == 0
    # U2 has no browsing record at all: neutral zeros, not NaN
    assert (result.loc[2, list(HTTP_FEATURE_COLUMNS)] == 0).all()
    assert not result[list(HTTP_FEATURE_COLUMNS)].isna().any().any()
    assert "_day" not in result.columns
