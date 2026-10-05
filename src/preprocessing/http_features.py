"""
http features (signal group "http" in src/preprocessing/feature_groups.py).

Each file event is given a description of the same user's web browsing on
that day, built from the small per-user-per-day tables produced by
scripts/precompute_http_daily.py (see src/preprocessing/http_aggregate.py).

Design (from training-users-only exploration, notebooks/14_explore_http.py):
the whole log has only about 1,000 domains and a typical user touches
~150 of them. On 95% of days at most ~6% of a user's visits go to a
domain they had never visited before, so a burst of NEW domains is a
strong user-specific signal -- strongest for a slow behaviour change
such as the scenario-2 pattern, which an individual file copy cannot show.

Features (all attached to every file event on that user-day):

  http_new_domains_today   log1p(number of domains the user visits today
                           for the first time ever)
  http_new_domains_7d      the same, summed over the trailing 7 days
                           (today and the 6 days before it)
  http_volume_deviation    how today's visit count compares to this user's
                           OTHER browsing days (leave-one-out z-score)

Rules that keep this leak-free:
  * No labels, no answer key, no hand-made domain categories ("job
    sites", "leak sites"). Novelty is defined purely from the user's own
    history, so it needs no list of "bad" domains.
  * The two novelty features are POINT-IN-TIME: they use only visits up
    to and including the event's day, never the future.
  * A user's first WARMUP_DAYS days of browsing are treated as
    establishing their baseline: domains first seen then do not count as
    new (otherwise every domain would be "new" at the start of the log).
  * http_volume_deviation is leave-one-out over the user's whole
    timeline, like the existing session-volume feature -- it is NOT
    point-in-time. This is a documented limitation.

A file event on a day with no recorded browsing gets 0.0 for all three
(the neutral value).
"""

import numpy as np
import pandas as pd

HTTP_FEATURE_COLUMNS = (
    "http_new_domains_today",
    "http_new_domains_7d",
    "http_volume_deviation",
)

WARMUP_DAYS = 14
TRAILING_DAYS = 7
MIN_ACTIVE_DAYS_FOR_DEVIATION = 3
STD_FLOOR = 1e-6


def load_http_domain_daily(processed_dir):
    """Reads the (user, day, domain, visits) table made by
    scripts/precompute_http_daily.py."""
    import os
    path = os.path.join(processed_dir, "http_domain_daily.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/precompute_http_daily.py` first "
            "(one pass over the raw http.csv)."
        )
    return pd.read_parquet(path)


def build_http_user_day_features(domain_daily, warmup_days=WARMUP_DAYS, trailing_days=TRAILING_DAYS):
    """
    domain_daily: DataFrame with columns user, day, domain, visits (the
    output of the aggregation step).

    Returns a DataFrame with one row per (user, calendar day) between a
    user's first and last browsing day, with the three HTTP_FEATURE_COLUMNS.
    """
    df = domain_daily[["user", "day", "domain", "visits"]].copy()
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()

    # --- daily visit totals on active days ---
    active = df.groupby(["user", "day"], sort=False)["visits"].sum().rename("visits").reset_index()

    # --- domains first seen per day, excluding each user's warm-up period ---
    first_day_by_user = df.groupby("user")["day"].min().rename("user_first_day")
    first_seen = df.groupby(["user", "domain"], sort=False)["day"].min().rename("first_seen").reset_index()
    first_seen = first_seen.join(first_day_by_user, on="user")
    first_seen = first_seen[first_seen["first_seen"] > first_seen["user_first_day"] + pd.Timedelta(days=warmup_days)]
    new_per_day = (
        first_seen.groupby(["user", "first_seen"]).size().rename("new_domains").reset_index()
        .rename(columns={"first_seen": "day"})
    )
    active = active.merge(new_per_day, on=["user", "day"], how="left")
    active["new_domains"] = active["new_domains"].fillna(0.0)

    # --- leave-one-out z-score of the visit count over the user's active days ---
    n = active.groupby("user")["visits"].transform("count")
    s = active.groupby("user")["visits"].transform("sum")
    q = active.assign(sq=active["visits"] ** 2).groupby("user")["sq"].transform("sum")
    m = n - 1
    safe_m = m.where(m > 0, np.nan)
    loo_mean = (s - active["visits"]) / safe_m
    loo_var = ((q - active["visits"] ** 2) / safe_m) - loo_mean ** 2
    loo_std = np.sqrt(loo_var.clip(lower=0)).clip(lower=STD_FLOOR)
    deviation = (active["visits"] - loo_mean) / loo_std
    deviation = deviation.where(n >= MIN_ACTIVE_DAYS_FOR_DEVIATION, 0.0)
    active["http_volume_deviation"] = deviation.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # --- trailing-window novelty over CALENDAR days (inactive days count as 0) ---
    pieces = []
    for user, g in active.groupby("user", sort=False):
        series = g.set_index("day")["new_domains"].sort_index()
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
        series = series.reindex(full_index, fill_value=0.0)
        trailing = series.rolling(window=trailing_days, min_periods=1).sum()
        pieces.append(pd.DataFrame({
            "user": user, "day": full_index,
            "new_domains": series.values, "new_domains_trailing": trailing.values,
        }))
    calendar = pd.concat(pieces, ignore_index=True)

    calendar = calendar.merge(
        active[["user", "day", "http_volume_deviation"]], on=["user", "day"], how="left"
    )
    calendar["http_volume_deviation"] = calendar["http_volume_deviation"].fillna(0.0)

    calendar["http_new_domains_today"] = np.log1p(calendar["new_domains"])
    calendar["http_new_domains_7d"] = np.log1p(calendar["new_domains_trailing"])

    return calendar[["user", "day", *HTTP_FEATURE_COLUMNS]]


def add_http_features(file_df, http_user_day):
    """
    Adds the HTTP_FEATURE_COLUMNS to every file event, by matching the
    event's (user, calendar day) to the table from
    build_http_user_day_features. Events on days with no browsing record
    get 0.0. Row order is preserved.
    """
    df = file_df.copy()
    df["_day"] = pd.to_datetime(df["date"]).dt.normalize()
    table = http_user_day.rename(columns={"day": "_day"})
    merged = df.merge(table, on=["user", "_day"], how="left")
    for column in HTTP_FEATURE_COLUMNS:
        merged[column] = merged[column].fillna(0.0)
    return merged.drop(columns=["_day"])
