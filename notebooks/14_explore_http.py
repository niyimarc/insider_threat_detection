"""
Exploration of the http signal -- using TRAINING USERS ONLY.

Why training users only: what we learn here decides how the http
features are designed. If that were learned while looking at insiders
(or at the test users), any later improvement could be a product of
the design having peeked at the evaluation data. Training users are
benign by construction (every insider is placed in the test split), so
this script can only ever describe NORMAL browsing -- which is exactly
what an anomaly detector needs to know -- and cannot be steered by the
answer key. The answer key is deliberately never opened here.

Needs the small tables from `python scripts/precompute_http_daily.py`
and the processed dataset from `python scripts/build_dataset.py`.

Run from the project root:
    python notebooks/14_explore_http.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")


def main():
    events = pd.read_parquet(os.path.join(PROCESSED_DIR, "file_events_processed.parquet"),
                             columns=["user", "split"])
    train_users = set(events.loc[events["split"] == "train", "user"].unique())
    print(f"Training users with file activity: {len(train_users)}")

    domain = pd.read_parquet(os.path.join(PROCESSED_DIR, "http_domain_daily.parquet"))
    hour = pd.read_parquet(os.path.join(PROCESSED_DIR, "http_hour_daily.parquet"))
    all_http_users = domain["user"].nunique()

    domain = domain[domain["user"].isin(train_users)]
    hour = hour[hour["user"].isin(train_users)]
    print(f"http users overall: {all_http_users}; training users with http: {domain['user'].nunique()}")
    print(f"training http visits: {int(domain['visits'].sum()):,} over "
          f"{domain['day'].nunique()} distinct days "
          f"({domain['day'].min().date()} to {domain['day'].max().date()})")

    print("\n=== 1. Volume per user per day (training users, active days only) ===")
    per_day = hour.groupby(["user", "day"])["visits"].sum()
    print(per_day.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(1).to_string())

    print("\n=== 2. Hour-of-day profile (share of visits) ===")
    by_hour = hour.groupby("hour")["visits"].sum()
    print((by_hour / by_hour.sum()).round(4).to_string())

    print("\n=== 3. Domain popularity ===")
    users_per_domain = domain.groupby("domain")["user"].nunique()
    n_domains = len(users_per_domain)
    print(f"distinct domains: {n_domains:,}")
    n_train = domain["user"].nunique()
    for k in (1, 2, 5, 10):
        share = (users_per_domain <= k).mean()
        print(f"  domains visited by <= {k:2d} training user(s): {share:6.1%}")
    visits_per_domain = domain.groupby("domain")["visits"].sum().sort_values(ascending=False)
    top = visits_per_domain.head(15)
    print("\nTop 15 domains by visits:")
    print((top / visits_per_domain.sum()).round(4).to_string())
    cum = (visits_per_domain.cumsum() / visits_per_domain.sum())
    for n in (10, 100, 1000):
        print(f"  top {n:4d} domains cover {cum.iloc[min(n, len(cum)) - 1]:6.1%} of visits")

    print("\n=== 4. How much do users differ from each other? ===")
    per_user = domain.groupby("user")["visits"].sum()
    print("visits per user:")
    print(per_user.describe(percentiles=[0.05, 0.5, 0.95]).round(0).to_string())
    distinct_per_user = domain.groupby("user")["domain"].nunique()
    print("distinct domains per user:")
    print(distinct_per_user.describe(percentiles=[0.05, 0.5, 0.95]).round(0).to_string())

    print("\n=== 5. Novelty: share of a user's daily visits to domains the user "
          "had NOT visited before that day (training users) ===")
    d = domain.sort_values(["user", "day"])
    first_day = d.groupby(["user", "domain"])["day"].transform("min")
    d["is_new_domain_visit"] = d["day"] == first_day
    new_visits = d[d["is_new_domain_visit"]].groupby(["user", "day"])["visits"].sum()
    total_visits = d.groupby(["user", "day"])["visits"].sum()
    novelty = (new_visits.reindex(total_visits.index).fillna(0) / total_visits)
    print(novelty.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3).to_string())

    print("\nNote: nothing above used labels, insiders or the answer key.")


if __name__ == "__main__":
    main()
