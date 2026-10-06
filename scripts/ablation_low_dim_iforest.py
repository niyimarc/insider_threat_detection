"""
Cheap, GPU-free ablation: an Isolation Forest on a handful of per-session
NUMERIC summaries (the mean of each registered feature over the session),
with one run per feature-group combination, 10 seeds each.

Why this exists: the full static-vector baseline isolates on 100+
dimensions (vocabulary fractions etc.), where one weak signal can be
drowned out. This keeps only the registered numeric features, so whether
a signal group helps shows up directly -- and it takes seconds per run,
so every configuration can be repeated over many seeds.

Protocol (same rules as the rest of the project):
  * fit on TRAIN users' sessions only (benign by construction); features
    are standardised with TRAIN statistics only;
  * scored on the DEVELOPMENT test partition only (see
    src/evaluation/partition.py) -- the locked partition is never read
    here;
  * unsupervised: no labels are used to fit anything.

Reported per configuration, as mean +/- std over seeds:
  pooled      all malicious sessions vs benign users' sessions
  own         malicious vs the same insiders' other sessions
  s1 / s2 / s3  that scenario vs benign users (s1 has 5 sessions and s3
              15 on dev -- treat as anecdotal)
and, for the full configuration, a 95% user-level bootstrap interval of
the pooled AUROC (users are resampled, not sessions, because sessions of
one user are not independent).

Run from the project root:
    python scripts/ablation_low_dim_iforest.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from src.evaluation.partition import select_test_df
from src.preprocessing.feature_groups import NUMERIC_FEATURE_REGISTRY
from src.preprocessing.labeling import load_malicious_windows

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
SESSION_KEYS = ["user", "session_start", "session_end"]
SEEDS = list(range(10))
N_BOOTSTRAP = 300


def columns_for(groups):
    return [c for c, g in NUMERIC_FEATURE_REGISTRY if g in groups]


def session_summaries(df, columns):
    """One row per session: mean of each feature over its events, plus the
    user and the session's malicious label."""
    grouped = df.groupby(SESSION_KEYS, sort=True)
    means = grouped[columns].mean()
    meta = grouped.agg(label=("session_is_malicious", "first"))
    out = pd.concat([means, meta], axis=1).reset_index()
    out["label"] = out["label"].astype(int)
    return out


def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return roc_auc_score(labels, scores)


def metrics(table, scores, scenario_map):
    y = table["label"].values
    insider = table["user"].isin(scenario_map).values
    scenario = table["user"].map(scenario_map).fillna(0).astype(int).values
    result = {"pooled": safe_auroc(y, scores), "own": safe_auroc(y[insider], scores[insider])}
    for s in (1, 2, 3):
        keep = (~insider) | ((scenario == s) & (y == 1))
        result[f"s{s}"] = safe_auroc((y == 1)[keep], scores[keep])
    return result


def user_bootstrap_ci(table, scores, seed=0):
    rng = np.random.default_rng(seed)
    users = table["user"].unique()
    by_user = {u: np.flatnonzero(table["user"].values == u) for u in users}
    y = table["label"].values
    values = []
    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(users, size=len(users), replace=True)
        idx = np.concatenate([by_user[u] for u in sample])
        value = safe_auroc(y[idx], scores[idx])
        if not np.isnan(value):
            values.append(value)
    return np.percentile(values, [2.5, 97.5])


def main():
    events = pd.read_parquet(DATA_PATH)
    train_df = events[events["split"] == "train"]
    dev_df = select_test_df(events)
    scenario_map = load_malicious_windows().set_index("user")["scenario"].to_dict()

    configs = {
        "file only": ["file"],
        "file + device + logon": ["file", "device", "logon"],
        "file + device + logon + http": ["file", "device", "logon", "http"],
        "http only": ["http"],
    }

    results_table = []
    full_scores = None
    full_table = None
    for name, groups in configs.items():
        columns = columns_for(groups)
        train_t = session_summaries(train_df, columns)
        dev_t = session_summaries(dev_df, columns)
        mean, std = train_t[columns].mean(), train_t[columns].std().replace(0, 1.0)
        x_train = ((train_t[columns] - mean) / std).values
        x_dev = ((dev_t[columns] - mean) / std).values

        per_seed, score_sum = [], np.zeros(len(dev_t))
        for seed in SEEDS:
            forest = IsolationForest(n_estimators=200, random_state=seed).fit(x_train)
            scores = -forest.score_samples(x_dev)
            per_seed.append(metrics(dev_t, scores, scenario_map))
            score_sum += scores
        frame = pd.DataFrame(per_seed)
        results_table.append((name, len(columns), frame.mean(), frame.std()))
        if name == "file + device + logon + http":
            full_scores, full_table = score_sum / len(SEEDS), dev_t

    print(f"\ntrain sessions: {len(train_t)}, dev sessions: {len(dev_t)} "
          f"({int(dev_t['label'].sum())} malicious); {len(SEEDS)} seeds per configuration\n")
    header = f"{'configuration':32s} {'#feat':>5s} " + " ".join(f"{k:>13s}" for k in ("pooled", "own", "s1", "s2", "s3"))
    print(header)
    print("-" * len(header))
    for name, n_feat, mean, std in results_table:
        cells = " ".join(f"{mean[k]:6.3f}+-{std[k]:.3f}" for k in ("pooled", "own", "s1", "s2", "s3"))
        print(f"{name:32s} {n_feat:5d} {cells}")

    low, high = user_bootstrap_ci(full_table, full_scores)
    print(f"\nfull configuration, pooled AUROC of the seed-averaged score: "
          f"{safe_auroc(full_table['label'].values, full_scores):.3f} "
          f"(95% user-level bootstrap interval {low:.3f} to {high:.3f}, {N_BOOTSTRAP} resamples)")
    print("\nAUROC 0.5 = no separation. Dev partition only; the locked partition was not read.")


if __name__ == "__main__":
    main()
