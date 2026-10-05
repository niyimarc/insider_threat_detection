import numpy as np
import pandas as pd
import pytest

from src.preprocessing.http_features import (
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


def test_volume_deviation_flags_a_heavy_day_and_is_leave_one_out():
    rows = [("U1", d.strftime("%Y-%m-%d"), "a.com", 10 + (i % 2)) for i, d in
            enumerate(pd.date_range("2010-01-01", periods=30))]
    rows.append(("U1", "2010-02-01", "a.com", 200))
    table = build_http_user_day_features(_domain_daily(rows))
    heavy = _day(table, "U1", "2010-02-01")["http_volume_deviation"]
    normal = _day(table, "U1", "2010-01-10")["http_volume_deviation"]
    assert heavy > 2.5         # ~log(201/11.5): far above the user's usual volume
    # An ordinary day stays close to 0. (It is not exactly 0: its baseline
    # is the mean of the OTHER days, which includes the one heavy day.)
    assert abs(normal) < 0.6
    assert heavy > 5 * abs(normal)


def test_volume_deviation_stays_bounded_for_users_with_perfectly_regular_counts():
    # Regression: the z-score version divided by a ~zero standard deviation
    # and produced values around +/-1e7 here.
    rows = [("U1", d.strftime("%Y-%m-%d"), "a.com", 95) for d in pd.date_range("2010-01-01", periods=40)]
    rows.append(("U1", "2010-03-01", "a.com", 94))
    rows.append(("U1", "2010-03-02", "a.com", 300))
    table = build_http_user_day_features(_domain_daily(rows))
    values = table["http_volume_deviation"]
    assert values.abs().max() <= 4.0
    assert abs(_day(table, "U1", "2010-03-01")["http_volume_deviation"]) < 0.1
    assert abs(_day(table, "U1", "2010-01-15")["http_volume_deviation"]) < 0.1
    assert _day(table, "U1", "2010-03-02")["http_volume_deviation"] > 0.9  # the 300-visit day stands out


def test_volume_deviation_is_clipped():
    rows = [("U1", d.strftime("%Y-%m-%d"), "a.com", 1) for d in pd.date_range("2010-01-01", periods=30)]
    rows.append(("U1", "2010-02-15", "a.com", 100000))
    table = build_http_user_day_features(_domain_daily(rows))
    assert _day(table, "U1", "2010-02-15")["http_volume_deviation"] == 4.0


def test_volume_deviation_is_neutral_with_too_few_days():
    rows = [("U1", "2010-01-01", "a.com", 5), ("U1", "2010-01-02", "a.com", 500)]
    table = build_http_user_day_features(_domain_daily(rows))
    assert (table["http_volume_deviation"] == 0).all()


def test_inactive_calendar_days_get_neutral_values():
    rows = [("U1", "2010-01-01", "a.com", 10), ("U1", "2010-01-05", "a.com", 10),
            ("U1", "2010-01-09", "a.com", 10), ("U1", "2010-01-10", "a.com", 12)]
    table = build_http_user_day_features(_domain_daily(rows))
    gap_day = _day(table, "U1", "2010-01-03")
    assert gap_day["http_volume_deviation"] == 0 and gap_day["http_new_domains_today"] == 0


def test_output_has_exactly_the_registered_columns():
    table = build_http_user_day_features(_domain_daily(_steady_user()))
    assert list(table.columns) == ["user", "day", *HTTP_FEATURE_COLUMNS]
    assert not table[list(HTTP_FEATURE_COLUMNS)].isna().any().any()


def test_http_feature_columns_match_the_registry():
    from src.preprocessing.feature_groups import NUMERIC_FEATURE_REGISTRY
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
