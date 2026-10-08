"""
From suspicious DAYS to suspicious USERS.

The day-level models give every user-day a score. A security team does not
review days in isolation: it needs to know WHICH PEOPLE to look at, and WHEN
an alert would first have fired. This module turns out-of-fold day scores into
that, with rules fixed in advance (nothing here is tuned on the results):

  user score    the mean of the user's TOP_K_DAYS highest day scores
                ("how bad are this person's worst days"); the single highest
                day is reported as a secondary rule.
  alert         a day whose score is at or above the ALERT_QUANTILE quantile
                of BENIGN users' day scores. The cutoff is taken from benign
                users only, never from the insiders being evaluated.
  detected      an insider with at least one alert inside their own window;
  time to       days from the first window day to the first alert inside the
  detection     window.

CAVEATS that every report must repeat: the development partition has few
benign users, so false-alarm figures are coarse; the share of insiders in the
data is far higher than in any real organisation, so precision-at-k is only
meaningful against its printed baseline; the user-level insider-vs-benign
comparison can also reflect "what kind of user an insider is", which the
calendar and per-user-clock controls only partly cover.

Pure functions: no file access.
"""

import numpy as np
import pandas as pd

TOP_K_DAYS = 7
ALERT_QUANTILE = 0.99
RULES = ("top_k_mean", "max")


def user_scores(users, scores, k=TOP_K_DAYS, rule="top_k_mean"):
    """pandas Series (index = user): each user's score under the chosen rule."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    frame = pd.DataFrame({"user": np.asarray(users), "score": np.asarray(scores, dtype=float)})

    def aggregate(values):
        ordered = np.sort(values.to_numpy())[::-1]
        return float(ordered[:k].mean()) if rule == "top_k_mean" else float(ordered[0])

    return frame.groupby("user")["score"].agg(aggregate)


def _pairwise_auroc(insider_scores, benign_scores):
    """P(an insider outscores a benign user), ties count one half."""
    a = np.asarray(insider_scores, dtype=float)[:, None]
    b = np.asarray(benign_scores, dtype=float)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def user_level_auroc(score_by_model, insider_users, n_boot=1000, seed=0, reference=None):
    """
    score_by_model: {name: Series(user -> user score)}, all over the same users.
    Returns {name: {"auroc", "ci_low", "ci_high", ["auroc_minus_reference": {...}]}}:
    the probability that a random insider outscores a random benign user, with a
    95% interval from resampling insiders and benign users separately (the same
    resamples for every model, so differences are paired).
    """
    insider_users = set(insider_users)
    names = list(score_by_model)
    users = list(score_by_model[names[0]].index)
    insiders = [u for u in users if u in insider_users]
    benign = [u for u in users if u not in insider_users]
    if not insiders or not benign:
        return {name: {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")} for name in names}

    arrays = {n: (score_by_model[n].loc[insiders].to_numpy(), score_by_model[n].loc[benign].to_numpy()) for n in names}
    rng = np.random.default_rng(seed)
    draws = [(rng.integers(0, len(insiders), len(insiders)), rng.integers(0, len(benign), len(benign)))
             for _ in range(n_boot)]
    boot = {n: np.array([_pairwise_auroc(arrays[n][0][i], arrays[n][1][j]) for i, j in draws]) for n in names}

    out = {}
    for n in names:
        low, high = np.percentile(boot[n], [2.5, 97.5])
        out[n] = {"auroc": _pairwise_auroc(*arrays[n]), "ci_low": float(low), "ci_high": float(high)}
    if reference in boot:
        for n in names:
            if n == reference:
                continue
            diff = boot[n] - boot[reference]
            low, high = np.percentile(diff, [2.5, 97.5])
            out[n]["auroc_minus_reference"] = {
                "mean": float(diff.mean()), "ci_low": float(low), "ci_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
    return out


def precision_at_k(user_score_series, insider_users, ks=(5, 10)):
    """
    {k: {"insiders_in_top_k", "k", "baseline_share"}}: how many of the k highest-scored
    users are insiders, next to the share of insiders among ALL users (what a random
    pick would give).
    """
    insider_users = set(insider_users)
    ranked = user_score_series.sort_values(ascending=False)
    baseline = float(np.mean([u in insider_users for u in ranked.index]))
    out = {}
    for k in ks:
        k = min(k, len(ranked))
        out[int(k)] = {
            "k": int(k),
            "insiders_in_top_k": int(sum(u in insider_users for u in ranked.index[:k])),
            "baseline_share": baseline,
        }
    return out


def windows_from_labels(users, days, window_label):
    """{user: (first window day, last window day)} for users with any window-labelled day."""
    frame = pd.DataFrame({"user": np.asarray(users), "day": pd.to_datetime(np.asarray(days)),
                          "y": np.asarray(window_label).astype(int)})
    frame = frame[frame["y"] == 1]
    grouped = frame.groupby("user")["day"].agg(["min", "max"])
    return {u: (row["min"], row["max"]) for u, row in grouped.iterrows()}


def alert_analysis(users, days, scores, windows, quantile=ALERT_QUANTILE):
    """
    The alert simulation (see the module docstring). `windows` = windows_from_labels(...);
    users with no window are benign. Returns {"threshold", "summary": {...}, "per_user": {user: {...}}}.
    """
    frame = pd.DataFrame({"user": np.asarray(users), "day": pd.to_datetime(np.asarray(days)),
                          "score": np.asarray(scores, dtype=float)})
    benign_mask = ~frame["user"].isin(windows)
    if not benign_mask.any():
        raise ValueError("Need benign users to set the alert cutoff.")
    threshold = float(np.quantile(frame.loc[benign_mask, "score"].to_numpy(), quantile))
    frame["alert"] = frame["score"] >= threshold

    per_user = {}
    for user, rows in frame.groupby("user"):
        alert_days = rows.loc[rows["alert"], "day"].sort_values()
        entry = {"n_alert_days": int(len(alert_days)),
                 "first_alert": alert_days.iloc[0] if len(alert_days) else None}
        if user in windows:
            start, end = windows[user]
            in_window = alert_days[(alert_days >= start) & (alert_days <= end)]
            entry.update({
                "is_insider": True,
                "window_start": start,
                "window_end": end,
                "first_alert_in_window": in_window.iloc[0] if len(in_window) else None,
                "days_to_first_alert": int((in_window.iloc[0] - start).days) if len(in_window) else None,
                "alert_before_window": bool((alert_days < start).any()),
            })
        else:
            entry["is_insider"] = False
        per_user[user] = entry

    insiders = {u: e for u, e in per_user.items() if e["is_insider"]}
    benign = {u: e for u, e in per_user.items() if not e["is_insider"]}
    detected = [e for e in insiders.values() if e["first_alert_in_window"] is not None]
    delays = [e["days_to_first_alert"] for e in detected]
    summary = {
        "quantile": quantile,
        "n_insiders": len(insiders),
        "n_benign": len(benign),
        "insiders_detected": len(detected),
        "detection_rate": len(detected) / len(insiders) if insiders else float("nan"),
        "median_days_to_first_alert": float(np.median(delays)) if delays else float("nan"),
        "share_detected_within_7_days": (sum(d <= 6 for d in delays) / len(insiders)) if insiders else float("nan"),
        "insiders_alerted_before_window": int(sum(e["alert_before_window"] for e in insiders.values())),
        "benign_users_alerted": int(sum(e["n_alert_days"] > 0 for e in benign.values())),
        "benign_alert_days_per_user": float(np.mean([e["n_alert_days"] for e in benign.values()])) if benign else float("nan"),
    }
    return {"threshold": threshold, "summary": summary, "per_user": per_user}


def ranked_user_table(user_score_series, alerts, insider_scenarios, frame=None, activity_columns=(), top_days=3):
    """
    DataFrame, one row per user, best-scored first: rank, user_score, first_alert, n_alert_days,
    the top suspicious days with their scores (and the day's raw activity if `frame` is given:
    a table with user, day, score and the activity columns), and -- for EVALUATION ONLY, a real
    deployment does not know them -- is_insider and scenario.
    """
    rows = []
    ordered = user_score_series.sort_values(ascending=False)
    for rank, (user, value) in enumerate(ordered.items(), start=1):
        info = alerts["per_user"].get(user, {})
        row = {
            "rank": rank, "user": user, "user_score": float(value),
            "n_alert_days": info.get("n_alert_days", 0),
            "first_alert": info.get("first_alert"),
            "is_insider": user in insider_scenarios,
            "scenario": insider_scenarios.get(user, 0),
        }
        if frame is not None:
            mine = frame[frame["user"] == user].sort_values("score", ascending=False).head(top_days)
            parts = []
            for _, r in mine.iterrows():
                activity = ", ".join(f"{c}={r[c]:.3g}" for c in activity_columns if c in r.index and pd.notna(r[c]))
                parts.append(f"{pd.Timestamp(r['day']).date()} (score {r['score']:.3g}; {activity})")
            row["top_days"] = " | ".join(parts)
        rows.append(row)
    return pd.DataFrame(rows)
