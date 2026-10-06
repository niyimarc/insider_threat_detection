"""
Day-level check of the http signal, on the DEVELOPMENT partition only.

The session-level test (diagnose_signal_strength.py) uses window labels:
every file session inside an insider's 1.5-2.5 month window counts as
malicious, even though the injected http activity touches far fewer days.
Here the unit is the USER-DAY instead, and a day counts as an "injected
day" if the answer key lists at least one http event for that insider on
it.

The answer key is used ONLY to label days for this evaluation. No feature
is built from it (no domain lists, no thresholds).

Scores compared (higher = more anomalous):
  http_new_domains_today, http_new_domains_7d     the registered features
  offbaseline_share_7d  CANDIDATE, not registered: share of the trailing
        7 days' visits that go to domains first seen AFTER the user's
        14-day warm-up. Unlike novelty it stays high while a user keeps
        visiting a new site, not only on the first visit.

Comparisons (AUROC, 0.5 = no separation):
  vs benign   injected days vs every day of the benign dev users
  vs own      injected days vs the same insider's other days
  vs own-win  injected days vs the same insider's other days INSIDE their
              labelled window (the hardest and fairest comparison)

IMPORTANT for reporting: this day-level labelling was introduced after
seeing the session-level results. If it is used, report both labelings
side by side and say why the day-level one was added.

Run from the project root (needs data/processed/http_domain_daily.parquet
and file_events_processed.parquet, and data/raw/answers):
    python scripts/diagnose_http_daylevel.py
"""

import glob
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.preprocessing.http_features import WARMUP_DAYS, TRAILING_DAYS, build_http_user_day_features
from src.preprocessing.labeling import load_malicious_windows

PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
ANSWERS_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "answers")


def load_injected_http_days():
    """(user, day, scenario) for every day the answer key lists >=1 http event."""
    rows = []
    for path in glob.glob(os.path.join(ANSWERS_DIR, "r4.2-*", "*.csv")):
        scenario = int(os.path.basename(os.path.dirname(path)).split("-")[1])
        with open(path, newline="") as f:
            for line in f:
                if not line.startswith("http,"):
                    continue
                parts = line.rstrip("\n").split(",", 5)
                if len(parts) < 5:
                    continue
                rows.append((parts[3], pd.to_datetime(parts[2], format="%m/%d/%Y %H:%M:%S").normalize(), scenario))
    return pd.DataFrame(rows, columns=["user", "day", "scenario"]).drop_duplicates()


def offbaseline_share_7d(domain_daily):
    """Candidate feature: trailing-7-day share of visits to post-warm-up domains."""
    df = domain_daily[["user", "day", "domain", "visits"]].copy()
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    first_day = df.groupby("user")["day"].min().rename("user_first_day")
    first_seen = df.groupby(["user", "domain"])["day"].min().rename("first_seen").reset_index().join(first_day, on="user")
    first_seen["offbaseline"] = first_seen["first_seen"] > first_seen["user_first_day"] + pd.Timedelta(days=WARMUP_DAYS)
    df = df.merge(first_seen[["user", "domain", "offbaseline"]], on=["user", "domain"])
    df["off_visits"] = df["visits"] * df["offbaseline"]
    daily = df.groupby(["user", "day"])[["visits", "off_visits"]].sum().reset_index()

    pieces = []
    for user, g in daily.groupby("user", sort=False):
        g = g.set_index("day").sort_index()[["visits", "off_visits"]]
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(idx, fill_value=0)
        roll = g.rolling(TRAILING_DAYS, min_periods=1).sum()
        share = (roll["off_visits"] / roll["visits"].where(roll["visits"] > 0)).fillna(0.0)
        pieces.append(pd.DataFrame({"user": user, "day": idx, "offbaseline_share_7d": share.values}))
    return pd.concat(pieces, ignore_index=True)


def auroc(pos_values, neg_values):
    if len(pos_values) == 0 or len(neg_values) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos_values)), np.zeros(len(neg_values))]
    s = np.r_[pos_values, neg_values]
    return roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5


def main():
    events = pd.read_parquet(os.path.join(PROCESSED, "file_events_processed.parquet"),
                             columns=["user", "split", "test_partition"])
    dev_users = set(events.loc[(events["split"] == "test") & (events["test_partition"] == "dev"), "user"])
    windows = load_malicious_windows()
    window_by_user = windows.set_index("user")[["scenario", "start", "end"]].to_dict("index")
    dev_insiders = {u for u in dev_users if u in window_by_user}
    dev_benign = dev_users - dev_insiders
    print(f"dev users: {len(dev_users)} ({len(dev_insiders)} insiders, {len(dev_benign)} benign)")

    domain_daily = pd.read_parquet(os.path.join(PROCESSED, "http_domain_daily.parquet"))
    domain_daily = domain_daily[domain_daily["user"].isin(dev_users)]
    table = build_http_user_day_features(domain_daily).merge(
        offbaseline_share_7d(domain_daily), on=["user", "day"], how="left")

    injected = load_injected_http_days()
    injected = injected[injected["user"].isin(dev_insiders)]
    table = table.merge(injected.assign(injected=True)[["user", "day", "injected"]], on=["user", "day"], how="left")
    table["injected"] = table["injected"].fillna(False).astype(bool)
    table["scenario"] = table["user"].map(lambda u: window_by_user.get(u, {}).get("scenario", 0))
    table["in_window"] = [
        (u in window_by_user) and (window_by_user[u]["start"] <= d <= window_by_user[u]["end"])
        for u, d in zip(table["user"], table["day"])
    ]

    print(f"\nuser-days: {len(table)}; injected days (dev insiders): {int(table['injected'].sum())} "
          f"by scenario {table[table['injected']].groupby('scenario').size().to_dict()}")
    print(f"share of in-window insider days that are injected days: "
          f"{table.loc[table['in_window'], 'injected'].mean():.1%}")

    features = ["http_new_domains_today", "http_new_domains_7d", "offbaseline_share_7d"]
    benign_days = table[table["user"].isin(dev_benign)]
    print(f"\n{'scenario / feature':44s} {'n_inj':>6s} {'vs benign':>10s} {'vs own':>8s} {'vs own-win':>11s}")
    print("-" * 82)
    for scenario in (None, 1, 2, 3):
        inj = table[table["injected"] & ((table["scenario"] == scenario) if scenario else True)]
        label = f"scenario {scenario}" if scenario else "all scenarios"
        for feat in features:
            own = table[(~table["injected"]) & table["user"].isin(inj["user"].unique())]
            own_win = own[own["in_window"]]
            print(f"{label + ' / ' + feat:44s} {len(inj):6d} "
                  f"{auroc(inj[feat].values, benign_days[feat].values):10.3f} "
                  f"{auroc(inj[feat].values, own[feat].values):8.3f} "
                  f"{auroc(inj[feat].values, own_win[feat].values):11.3f}")
        print()
    print("0.5 = no separation. Dev partition only; injected-day labels come from the answer key "
          "and are used for evaluation only.")


if __name__ == "__main__":
    main()
