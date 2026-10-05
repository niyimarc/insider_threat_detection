"""
Combines all three already-trained models' test scores into one ensemble
score, to check whether they pick up different/complementary signal.

IMPORTANT -- how this stays leak-free: the three raw scores live on
totally different scales (LSTM reconstruction loss ~0.2, Transformer
~0.003, Isolation Forest's -score_samples is unbounded), so they can't
just be averaged directly -- whichever happens to have the largest
magnitude would dominate. Instead each model's raw score is converted to
a 0-100 percentile using percentile_calibrate() against that SAME
model's own VALIDATION (benign-only) score distribution -- exactly the
same calibration src/evaluation/metrics.py already uses for thresholding
-- and only THEN averaged. The reference distribution is validation, not
test, so nothing here is tuned against test labels. The combination rule
itself (unweighted mean of the three calibrated percentiles) is fixed in
advance, not chosen by checking which weighting scores best on test --
picking weights that way would be leakage.

Read-only. Doesn't retrain or overwrite checkpoints.

Run from the project root (needs checkpoints from a successful
`python scripts/train_models.py` run with the current NUMERIC_FEATURE_COLUMNS):
    python scripts/ensemble_scores.py
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

from src.training.vocab import load_vocab
from src.training.scaling import load_scaler
from src.training.feature_config import verify_feature_config
from src.training.dataset import SessionDataset, collate_sessions, SESSION_KEY_COLUMNS
from src.preprocessing.labeling import load_malicious_windows
from src.models.transformer_autoencoder import (
    DualChannelTransformerAutoencoder,
    session_anomaly_score as transformer_score_fn,
)
from src.models.lstm_autoencoder import LSTMAutoencoder, session_anomaly_score as lstm_score_fn
from src.models.static_vector_baseline import build_session_vectors
from src.evaluation.score import score_pytorch_model, score_static_model
from src.evaluation.metrics import percentile_calibrate
from src.evaluation.partition import select_test_df

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
BATCH_SIZE = 64


def get_session_users(events_df):
    return [
        session_df["user"].iloc[0]
        for _, session_df in events_df.sort_values(SESSION_KEY_COLUMNS).groupby(
            SESSION_KEY_COLUMNS, sort=False
        )
    ]


def auroc_auprc_on_subset(scores, labels, keep_mask):
    from sklearn.metrics import roc_auc_score, average_precision_score
    scores = np.asarray(scores)[keep_mask]
    labels = np.asarray(labels)[keep_mask]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None, None
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def report_subsets(name, scores, labels, users, scenario_map):
    labels = np.asarray(labels)
    scenario_of_session = np.array([scenario_map.get(u, 0) for u in users])
    benign = labels == 0
    malicious = labels == 1

    all_keep = benign | malicious
    short_keep = benign | (malicious & np.isin(scenario_of_session, [1, 3]))
    long_keep = benign | (malicious & (scenario_of_session == 2))

    auroc_all, auprc_all = auroc_auprc_on_subset(scores, labels, all_keep)
    auroc_short, _ = auroc_auprc_on_subset(scores, labels, short_keep)
    auroc_long, _ = auroc_auprc_on_subset(scores, labels, long_keep)

    print(f"  {name:30s} all AUROC={auroc_all:.4f} AUPRC={auprc_all:.4f}  "
          f"s1+3={auroc_short if auroc_short is None else f'{auroc_short:.4f}'}  "
          f"s2={auroc_long if auroc_long is None else f'{auroc_long:.4f}'}")


def main():
    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    val_df = events_df[events_df["split"] == "validation"]
    test_df = select_test_df(events_df)

    file_type_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_type_vocab.json"))
    file_extension_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_extension_vocab.json"))
    verify_feature_config(CHECKPOINT_DIR)
    scaler = load_scaler(os.path.join(CHECKPOINT_DIR, "numeric_scaler.json"))

    malicious_windows = load_malicious_windows()
    scenario_map = malicious_windows.set_index("user")["scenario"].to_dict()

    val_dataset = SessionDataset(val_df, file_type_vocab, file_extension_vocab, scaler)
    test_dataset = SessionDataset(test_df, file_type_vocab, file_extension_vocab, scaler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sessions)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sessions)

    calibrated_test_scores = {}
    test_labels = None
    test_users = test_dataset.session_users

    # --- Proposed model ---
    transformer_model = DualChannelTransformerAutoencoder(
        file_type_vocab_size=len(file_type_vocab), file_extension_vocab_size=len(file_extension_vocab)
    )
    transformer_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "transformer_autoencoder.pt")))
    val_scores, _ = score_pytorch_model(transformer_model, val_loader, transformer_score_fn)
    test_scores, test_labels = score_pytorch_model(transformer_model, test_loader, transformer_score_fn)
    calibrated_test_scores["transformer"] = percentile_calibrate(test_scores, val_scores)

    # --- Baseline 1 ---
    lstm_model = LSTMAutoencoder(
        file_type_vocab_size=len(file_type_vocab), file_extension_vocab_size=len(file_extension_vocab)
    )
    lstm_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "lstm_autoencoder.pt")))
    val_scores, _ = score_pytorch_model(lstm_model, val_loader, lstm_score_fn)
    test_scores, _ = score_pytorch_model(lstm_model, test_loader, lstm_score_fn)
    calibrated_test_scores["lstm"] = percentile_calibrate(test_scores, val_scores)

    # --- Baseline 2 ---
    static_detector = joblib.load(os.path.join(CHECKPOINT_DIR, "static_vector_detector.joblib"))
    val_vectors, _ = build_session_vectors(val_df, file_type_vocab, file_extension_vocab)
    test_vectors, static_labels = build_session_vectors(test_df, file_type_vocab, file_extension_vocab)
    static_users = get_session_users(test_df)
    val_scores = score_static_model(static_detector, val_vectors)
    test_scores = score_static_model(static_detector, test_vectors)
    calibrated_test_scores["static"] = percentile_calibrate(test_scores, val_scores)

    # The static baseline's session order comes from a plain groupby over
    # test_df, not from SessionDataset -- confirm it lines up with the two
    # PyTorch models' order before averaging scores element-wise.
    assert static_users == test_users, "session order mismatch between static baseline and SessionDataset"
    assert static_labels.tolist() == test_labels, "label mismatch between static baseline and SessionDataset"

    print("\n=== Individual models (calibrated against validation, for reference) ===")
    for name, scores in calibrated_test_scores.items():
        report_subsets(name, scores, test_labels, test_users, scenario_map)

    ensemble = np.mean(
        [calibrated_test_scores["transformer"], calibrated_test_scores["lstm"], calibrated_test_scores["static"]],
        axis=0,
    )
    print("\n=== Ensemble (unweighted mean of calibrated percentiles) ===")
    report_subsets("ensemble (all 3)", ensemble, test_labels, test_users, scenario_map)

    # Also check the two pairwise combinations, in case one model is
    # mostly just adding noise to the other two.
    pair_lstm_static = np.mean([calibrated_test_scores["lstm"], calibrated_test_scores["static"]], axis=0)
    report_subsets("ensemble (lstm+static)", pair_lstm_static, test_labels, test_users, scenario_map)

    pair_transformer_static = np.mean(
        [calibrated_test_scores["transformer"], calibrated_test_scores["static"]], axis=0
    )
    report_subsets("ensemble (transformer+static)", pair_transformer_static, test_labels, test_users, scenario_map)


if __name__ == "__main__":
    main()
