"""
Model-free test: does each registered feature, used directly as an
anomaly score, separate malicious from benign sessions? No training, no
checkpoints -- it only reads the processed dataset, so it takes seconds.

Why: if a raw feature has no separation here, no model built on top of it
can be expected to find signal, and more training time is wasted. If a
raw feature DOES separate but a trained model does not, the model is the
problem.

For every numeric feature, each session is summarised by its mean and its
max over the session's events, and the AUROC of that summary is computed
for three comparisons, on the DEVELOPMENT test partition only (never the
locked one -- see src/evaluation/partition.py):

  A. all malicious sessions vs benign users' sessions
  B. malicious sessions vs the SAME insiders' non-malicious sessions
     (removes differences between users: the fairest comparison)
  C. per scenario, malicious vs benign users

AUROC 0.5 = no separation. Below 0.5 means the feature is LOWER in
malicious sessions (still informative if a model can use direction).

Run from the project root:
    python scripts/diagnose_signal_strength.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.partition import select_test_df
from src.preprocessing.feature_groups import NUMERIC_FEATURE_REGISTRY
from src.preprocessing.labeling import load_malicious_windows
from src.training.dataset import SESSION_KEY_COLUMNS

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")


def session_table(test_df, scenario_map):
    columns = [c for c, _ in NUMERIC_FEATURE_REGISTRY]
    grouped = test_df.groupby(SESSION_KEY_COLUMNS, sort=True)
    means = grouped[columns].mean().add_suffix("__mean")
    maxes = grouped[columns].max().add_suffix("__max")
    meta = grouped.agg(
        label=("session_is_malicious", "first"),
        n_events=("date", "size"),
    )
    table = pd.concat([meta, means, maxes], axis=1).reset_index()
    table["label"] = table["label"].astype(int)
    table["scenario"] = table["user"].map(scenario_map).fillna(0).astype(int)
    table["insider"] = table["user"].isin(scenario_map)
    return table


def auroc(labels, values):
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels) or len(np.unique(values)) < 2:
        return float("nan")
    return roc_auc_score(labels, values)


def main():
    events = pd.read_parquet(DATA_PATH)
    test_df = select_test_df(events)
    scenario_map = load_malicious_windows().set_index("user")["scenario"].to_dict()
    t = session_table(test_df, scenario_map)

    y = t["label"].values
    insider = t["insider"].values
    print(f"\n{len(t)} dev sessions, {int(y.sum())} malicious "
          f"(scenario 1/2/3 = {[int(((t['scenario'] == s) & (y == 1)).sum()) for s in (1, 2, 3)]}), "
          f"{int(insider.sum())} from insiders, {int((~insider).sum())} from benign users\n")

    summaries = [c for c in t.columns if c.endswith("__mean") or c.endswith("__max")]
    summaries.append("n_events")
    group_of = {c: g for c, g in NUMERIC_FEATURE_REGISTRY}

    header = f"{'feature (session summary)':48s} {'group':7s} {'A all':>7s} {'B own':>7s} {'s1':>6s} {'s2':>6s} {'s3':>6s}"
    print(header)
    print("-" * len(header))
    for col in summaries:
        base = col.split("__")[0]
        a = auroc(y, t[col].values)
        b = auroc(y[insider], t[col].values[insider])
        per_scenario = []
        for s in (1, 2, 3):
            keep = (~insider) | ((t["scenario"].values == s) & (y == 1))
            per_scenario.append(auroc((y == 1)[keep], t[col].values[keep]))
        print(f"{col:48s} {group_of.get(base, '-'):7s} {a:7.3f} {b:7.3f} "
              + " ".join(f"{v:6.3f}" for v in per_scenario))

    print("\nA: all malicious vs benign users | B: malicious vs the same insiders' other sessions | "
          "s1-s3: that scenario vs benign users")
    print("0.5 = no separation. This is the dev partition; nothing here uses the locked insiders.")


if __name__ == "__main__":
    main()
