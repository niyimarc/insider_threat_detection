"""
Diagnostic: why are all three models scoring below random (AUROC < 0.5)?

Read-only. Doesn't retrain or overwrite anything. It answers three questions:
  1. Can simple session properties (length, gaps, copy counts) on their
     own separate malicious from benign test sessions, and in which
     direction?
  2. Which of the Transformer's four score components (presence,
     magnitude, file_type, file_extension) drives the below-random
     AUROC?
  3. Is the total anomaly score mostly just tracking session length?

Run from the project root:
    python scripts/unsupervised/diagnose_scores.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.unsupervised.training.vocab import load_vocab
from src.unsupervised.training.scaling import load_scaler
from src.unsupervised.training.feature_config import verify_feature_config
from src.common.run_config import get_checkpoint_dir
from src.unsupervised.training.dataset import SessionDataset, collate_sessions, SESSION_KEY_COLUMNS
from src.common.preprocessing.labeling import load_malicious_windows
from src.common.evaluation.partition import select_test_df
from src.unsupervised.models.transformer_autoencoder import (
    DualChannelTransformerAutoencoder,
    PRESENCE_COLUMN_INDEX,
    MAGNITUDE_NUMERIC_INDICES,
)

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
CHECKPOINT_DIR = get_checkpoint_dir()


def auroc(labels, values):
    return roc_auc_score(labels, values)


def session_table(test_df, scenario_map):
    """One row per test session, in the SAME order as SessionDataset."""
    rows = []
    for (user, start, end), s in test_df.sort_values(SESSION_KEY_COLUMNS).groupby(
        SESSION_KEY_COLUMNS, sort=False
    ):
        s = s.sort_values("date")
        rows.append({
            "user": user,
            "label": float(s["session_is_malicious"].iloc[0]),
            "scenario": scenario_map.get(user, 0),
            "malicious_user": user in scenario_map,
            "n_events": len(s),
            "first_gap_s": s["seconds_since_last_event"].iloc[0],
            "mean_gap_s": s["seconds_since_last_event"].mean(),
            "median_gap_s": s["seconds_since_last_event"].median(),
            "max_copies_since_connect": s["copies_since_device_connect"].max(),
            "mean_secs_since_connect": s["seconds_since_device_connect"].mean(),
            "frac_outside_hours": s["outside_working_hours"].astype(float).mean(),
            "singleton_session": start == end,
        })
    return pd.DataFrame(rows)


def transformer_component_scores(model, loader):
    """Same maths as session_anomaly_score, but keeps the four parts separate."""
    model.eval()
    parts = {"presence": [], "magnitude": [], "file_type": [], "file_extension": []}
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            numeric, mask = batch["numeric"], batch["mask"]
            lengths = mask.sum(dim=1).clamp(min=1)

            p_t = numeric[..., PRESENCE_COLUMN_INDEX:PRESENCE_COLUMN_INDEX + 1]
            p = ((out["presence"] - p_t) ** 2).squeeze(-1)
            parts["presence"] += ((p * mask).sum(1) / lengths).tolist()

            m_t = numeric[..., MAGNITUDE_NUMERIC_INDICES]
            m = ((out["magnitude_numeric"] - m_t) ** 2).mean(-1)
            parts["magnitude"] += ((m * mask).sum(1) / lengths).tolist()

            for key, logits in [("file_type", "file_type_logits"), ("file_extension", "file_extension_logits")]:
                ce = nn.functional.cross_entropy(out[logits].transpose(1, 2), batch[key], reduction="none")
                parts[key] += ((ce * mask).sum(1) / lengths).tolist()
    df = pd.DataFrame(parts)
    df["total"] = df.sum(axis=1)
    return df


def main():
    events_df = pd.read_parquet(DATA_PATH)
    test_df = select_test_df(events_df)

    windows = load_malicious_windows()
    scenario_map = windows.set_index("user")["scenario"].to_dict()

    print("Building per-session table for TEST split ...")
    t = session_table(test_df, scenario_map)
    y = t["label"].values
    print(f"{len(t)} sessions, {int(y.sum())} malicious ({y.mean():.1%})\n")

    print("=== 1. Simple session properties: malicious vs benign (medians) ===")
    cols = ["n_events", "first_gap_s", "mean_gap_s", "median_gap_s",
            "max_copies_since_connect", "mean_secs_since_connect", "frac_outside_hours"]
    print(t.groupby("label")[cols].median().T.rename(columns={0.0: "benign", 1.0: "malicious"}))
    print(f"\nsingleton (no enclosing logon session) share: "
          f"benign={t.loc[t.label == 0, 'singleton_session'].mean():.1%} "
          f"malicious={t.loc[t.label == 1, 'singleton_session'].mean():.1%}")

    print("\nAUROC of each raw property used directly as an 'anomaly score'")
    print("(> 0.5 = higher value means more likely malicious; < 0.5 = the opposite):")
    for c in cols:
        print(f"  {c:28s} {auroc(y, t[c].values):.3f}")

    print("\n=== 2. Transformer score, broken into its four components ===")
    file_type_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_type_vocab.json"))
    file_extension_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_extension_vocab.json"))
    verify_feature_config(CHECKPOINT_DIR)
    scaler = load_scaler(os.path.join(CHECKPOINT_DIR, "numeric_scaler.json"))
    ds = SessionDataset(test_df, file_type_vocab, file_extension_vocab, scaler)
    assert ds.session_users == t["user"].tolist(), "session order mismatch"
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_sessions)

    model = DualChannelTransformerAutoencoder(len(file_type_vocab), len(file_extension_vocab))
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "transformer_autoencoder.pt")))
    comp = transformer_component_scores(model, loader)

    for c in comp.columns:
        print(f"  {c:16s} AUROC={auroc(y, comp[c].values):.3f}  "
              f"median benign={comp.loc[y == 0, c].median():.3f}  "
              f"malicious={comp.loc[y == 1, c].median():.3f}")

    print("\n=== 3. Is the score just tracking session length / gaps? ===")
    for c in ["n_events", "mean_gap_s", "first_gap_s"]:
        rho, _ = spearmanr(comp["total"], t[c])
        print(f"  Spearman(total score, {c}) = {rho:+.3f}")

    print("\n=== 4. Within malicious users only (their own benign vs malicious sessions) ===")
    mu = t["malicious_user"].values
    print(f"  {mu.sum()} sessions from malicious users, {int(y[mu].sum())} malicious")
    print(f"  total-score AUROC within malicious users: {auroc(y[mu], comp['total'].values[mu]):.3f}")
    print(f"  n_events AUROC within malicious users:    {auroc(y[mu], t['n_events'].values[mu]):.3f}")

    print("\n=== 5. Per scenario: median total score vs benign ===")
    print(f"  benign: {comp.loc[y == 0, 'total'].median():.3f}")
    for sc in [1, 2, 3]:
        sel = (y == 1) & (t["scenario"].values == sc)
        if sel.any():
            print(f"  scenario {sc} (n={sel.sum()}): {comp.loc[sel, 'total'].median():.3f}  "
                  f"median n_events={t.loc[sel, 'n_events'].median():.0f}")


if __name__ == "__main__":
    main()
