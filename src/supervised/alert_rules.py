"""
ALERT RULES and the ENSEMBLE, for the detector's weakest part: turning day scores into alerts.

Decided before looking at any result of these rules (all variants are reported, none picked):

  ensemble     RANK_AVERAGE: each model's day scores become percentile ranks over all scored rows and
               the ranks are averaged. (A deployment would need a stored reference distribution for the
               ranks; this is the development-style version.)
  alert rules  "single"        a day alerts if its score is at or above the cutoff (the v1 rule);
               "persist_2of3"  a day alerts if it is at or above the cutoff AND at least one of the user's
                               2 previous ACTIVE days was too (2 of the last 3, this day included) -- an
                               isolated noisy day is no longer an alert. Its alerts are a subset of "single".
  cutoff       the `quantile` quantile of benign users' day scores, CROSS-FITTED: users are split into
               folds and a fold's users are judged with a cutoff computed only from benign users in the
               OTHER folds. This tests whether a cutoff set on some users carries over to others (the
               thing that failed for gradient boosting on the locked users), instead of fixing the false
               alarm rate by construction on the very users being judged.

Pure functions: no file access.
"""

import numpy as np
import pandas as pd

from src.supervised.user_level import alert_analysis

ALERT_RULES = ("single", "persist_2of3")


def rank_average(score_arrays):
    """Mean of the percentile ranks (0..1) of several score arrays over the same rows."""
    arrays = [np.asarray(a, dtype=float) for a in score_arrays]
    if not arrays or any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("Need one or more score arrays of equal length.")
    ranks = [pd.Series(a).rank(pct=True, method="average").to_numpy() for a in arrays]
    return np.mean(ranks, axis=0)


def apply_alert_rule(users, days, raw_flags, rule):
    """Alert flags per row from the per-day 'score at or above cutoff' flags, under the chosen rule."""
    if rule not in ALERT_RULES:
        raise ValueError(f"rule must be one of {ALERT_RULES}")
    raw = np.asarray(raw_flags, dtype=bool)
    if rule == "single":
        return raw.copy()
    frame = pd.DataFrame({"user": np.asarray(users), "day": pd.to_datetime(np.asarray(days)),
                          "flag": raw.astype(int)}).sort_values(["user", "day"], kind="stable")
    grouped = frame.groupby("user")["flag"]
    count = frame["flag"] + grouped.shift(1).fillna(0) + grouped.shift(2).fillna(0)
    alert = (frame["flag"].to_numpy() == 1) & (count.to_numpy() >= 2)
    return pd.Series(alert, index=frame.index).sort_index().to_numpy()


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval (low, high) for a proportion; (nan, nan) when n is 0."""
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denominator = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def assign_user_folds(users, insider_users, n_folds=5, seed=0):
    """{user: fold}: insiders and benign users are each dealt round-robin to the folds after a shuffle."""
    rng = np.random.default_rng(seed)
    insider_users = set(insider_users)
    everyone = sorted(pd.unique(np.asarray(users)))
    fold = {}
    for group in ([u for u in everyone if u in insider_users], [u for u in everyone if u not in insider_users]):
        order = list(rng.permutation(len(group)))
        for position, index in enumerate(order):
            fold[group[index]] = position % n_folds
    return fold


def cross_fitted_alert_analysis(users, days, scores, windows, quantile, rule, n_folds=5, seed=0):
    """
    alert_analysis with cutoffs that never see the users they judge (see the module docstring).
    `windows` = windows_from_labels(...). Returns the alert_analysis dict plus "fold_thresholds".
    """
    users_a = np.asarray(users)
    scores_a = np.asarray(scores, dtype=float)
    fold_of_user = assign_user_folds(users_a, windows, n_folds=n_folds, seed=seed)
    fold = pd.Series(users_a).map(fold_of_user).to_numpy()
    benign = ~pd.Series(users_a).isin(windows).to_numpy()
    raw = np.zeros(len(users_a), dtype=bool)
    thresholds = {}
    for f in range(n_folds):
        in_fold = fold == f
        reference = benign & ~in_fold
        if not in_fold.any():
            continue
        if not reference.any():
            raise ValueError("A fold has no benign users outside it to set the cutoff.")
        thresholds[f] = float(np.quantile(scores_a[reference], quantile))
        raw[in_fold] = scores_a[in_fold] >= thresholds[f]
    flags = apply_alert_rule(users_a, days, raw, rule)
    result = alert_analysis(users_a, days, scores_a, windows, quantile=quantile, alert_flags=flags)
    result["fold_thresholds"] = thresholds
    return result
