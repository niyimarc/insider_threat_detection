"""
Diagnostic: is scenario 2's below-random pooled AUROC explained by its
~1.5-2.5 MONTH insider-threat windows diluting the session-level label
(see src/preprocessing/labeling.py's label_file_events/label_sessions),
rather than by a modeling failure?

Read-only. Doesn't retrain, doesn't touch checkpoints or processed data.
For each of the three trained models, this recomputes AUROC/AUPRC on
three different subsets of the test split, all compared against the
SAME benign sessions:

  1. all malicious sessions (= the pooled number already in
     evaluation_results.json -- reported here again just for a
     side-by-side baseline)
  2. scenario 1 + scenario 3 malicious sessions only (short windows --
     label should closely track the actual act)
  3. scenario 2 malicious sessions only (long windows -- label mostly
     tracks "this user, sometime in a 2-month span")

If (2) is well above 0.5 and (3) stays near 0.5, that confirms the
window-dilution theory: the model has real signal, but scenario 2's
coarse label swamps it in the pooled metric.

Run from the project root:
    python scripts/diagnose_scenario2_dilution.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.training.vocab import load_vocab
from src.training.scaling import load_scaler
from src.training.dataset import SessionDataset, collate_sessions, SESSION_KEY_COLUMNS
from src.preprocessing.labeling import load_malicious_windows
from src.models.transformer_autoencoder import (
    DualChannelTransformerAutoencoder,
    session_anomaly_score as transformer_score_fn,
)
from src.models.lstm_autoencoder import LSTMAutoencoder, session_anomaly_score as lstm_score_fn
from src.models.static_vector_baseline import build_session_vectors
from src.evaluation.score import score_pytorch_model, score_static_model

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
BATCH_SIZE = 64


def get_session_users(events_df):
    """Mirrors SessionDataset's own grouping exactly, for the
    static-vector baseline, which has no SessionDataset of its own."""
    return [
        session_df["user"].iloc[0]
        for _, session_df in events_df.sort_values(SESSION_KEY_COLUMNS).groupby(
            SESSION_KEY_COLUMNS, sort=False
        )
    ]


def auroc_auprc_on_subset(scores, labels, keep_mask):
    """AUROC/AUPRC computed only over sessions where keep_mask is True --
    e.g. "benign sessions OR scenario-2 malicious sessions", discarding
    everything else, so scenario 2 is compared against the SAME benign
    population as every other subset rather than a different one."""
    scores = np.asarray(scores)[keep_mask]
    labels = np.asarray(labels)[keep_mask]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None, None  # no malicious or no benign left to compare against
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def report(name, test_scores, test_labels, test_users, scenario_map):
    test_scores = np.asarray(test_scores)
    test_labels = np.asarray(test_labels)
    scenario_of_session = np.array([scenario_map.get(u, 0) for u in test_users])

    benign = test_labels == 0
    malicious = test_labels == 1

    print(f"\n=== {name} ===")
    n_mal = int(malicious.sum())
    n_s1 = int((malicious & (scenario_of_session == 1)).sum())
    n_s2 = int((malicious & (scenario_of_session == 2)).sum())
    n_s3 = int((malicious & (scenario_of_session == 3)).sum())
    print(f"  malicious test sessions: {n_mal}  (scenario 1={n_s1}, 2={n_s2}, 3={n_s3})")

    all_keep = benign | malicious
    auroc_all, auprc_all = auroc_auprc_on_subset(test_scores, test_labels, all_keep)
    print(f"  [1] all malicious vs benign            AUROC={auroc_all:.4f}  AUPRC={auprc_all:.4f}")

    short_window_keep = benign | (malicious & np.isin(scenario_of_session, [1, 3]))
    auroc_short, auprc_short = auroc_auprc_on_subset(test_scores, test_labels, short_window_keep)
    if auroc_short is not None:
        print(f"  [2] scenario 1+3 only vs benign         AUROC={auroc_short:.4f}  AUPRC={auprc_short:.4f}")
    else:
        print("  [2] scenario 1+3 only vs benign         (no sessions in this subset)")

    long_window_keep = benign | (malicious & (scenario_of_session == 2))
    auroc_long, auprc_long = auroc_auprc_on_subset(test_scores, test_labels, long_window_keep)
    if auroc_long is not None:
        print(f"  [3] scenario 2 only vs benign            AUROC={auroc_long:.4f}  AUPRC={auprc_long:.4f}")
    else:
        print("  [3] scenario 2 only vs benign            (no sessions in this subset)")

    return {
        "model": name,
        "all": auroc_all,
        "scenario_1_and_3_only": auroc_short,
        "scenario_2_only": auroc_long,
    }


def main():
    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    test_df = events_df[events_df["split"] == "test"]

    print("Loading vocabularies and scaler saved during training ...")
    file_type_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_type_vocab.json"))
    file_extension_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_extension_vocab.json"))
    scaler = load_scaler(os.path.join(CHECKPOINT_DIR, "numeric_scaler.json"))

    print("Building user -> scenario map from insiders.csv ...")
    malicious_windows = load_malicious_windows()
    scenario_map = malicious_windows.set_index("user")["scenario"].to_dict()

    test_dataset = SessionDataset(test_df, file_type_vocab, file_extension_vocab, scaler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sessions)
    print(f"{len(test_dataset)} test sessions")

    summary = []

    # --- Proposed model ---
    transformer_model = DualChannelTransformerAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
    )
    transformer_model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "transformer_autoencoder.pt"))
    )
    test_scores, test_labels = score_pytorch_model(transformer_model, test_loader, transformer_score_fn)
    summary.append(report(
        "proposed_transformer_autoencoder", test_scores, test_labels, test_dataset.session_users, scenario_map
    ))

    # --- Baseline 1 ---
    lstm_model = LSTMAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
    )
    lstm_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "lstm_autoencoder.pt")))
    test_scores, test_labels = score_pytorch_model(lstm_model, test_loader, lstm_score_fn)
    summary.append(report(
        "baseline_lstm_autoencoder", test_scores, test_labels, test_dataset.session_users, scenario_map
    ))

    # --- Baseline 2 ---
    static_detector = joblib.load(os.path.join(CHECKPOINT_DIR, "static_vector_detector.joblib"))
    test_vectors, test_labels = build_session_vectors(test_df, file_type_vocab, file_extension_vocab)
    test_users = get_session_users(test_df)
    test_scores = score_static_model(static_detector, test_vectors)
    summary.append(report(
        "baseline_static_vector_isolation_forest", test_scores, test_labels.tolist(), test_users, scenario_map
    ))

    print("\n=== Summary ===")
    print(f"{'model':42s} {'all':>8s} {'s1+3 only':>10s} {'s2 only':>8s}")
    for row in summary:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"
        print(f"{row['model']:42s} {fmt(row['all']):>8s} {fmt(row['scenario_1_and_3_only']):>10s} {fmt(row['scenario_2_only']):>8s}")


if __name__ == "__main__":
    main()
