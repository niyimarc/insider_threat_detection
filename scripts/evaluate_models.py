import os
import sys
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

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
from src.evaluation.metrics import (
    diagnose_length_bias,
    percentile_calibrate_by_length,
    compute_threshold_metrics,
    compute_threshold_free_metrics,
    per_scenario_recall,
)

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
BATCH_SIZE = 64
THRESHOLD_PERCENTILES = [90, 95, 99]
LENGTH_BUCKETS = 5


def get_session_users(events_df):
    """Mirrors SessionDataset's own grouping exactly, so the resulting
    user list lines up 1:1 with build_session_vectors' output order for
    the static-vector baseline (which has no SessionDataset of its own)."""
    return [
        session_df["user"].iloc[0]
        for _, session_df in events_df.sort_values(SESSION_KEY_COLUMNS).groupby(
            SESSION_KEY_COLUMNS, sort=False
        )
    ]


def _report(name, val_scores, val_lengths, test_scores, test_lengths, test_labels, test_users, scenario_map):
    print(f"\n=== Evaluating {name} ===")

    # Check the length confound BEFORE fixing it, on benign test sessions
    # only (label == 0) -- mixing in malicious sessions here would
    # conflate "correlated with length" with "correlated with being
    # malicious", which is exactly the distinction we need to keep clean.
    benign_mask = [label == 0 for label in test_labels]
    benign_scores = [s for s, keep in zip(test_scores, benign_mask) if keep]
    benign_lengths = [l for l, keep in zip(test_lengths, benign_mask) if keep]
    bias = diagnose_length_bias(benign_scores, benign_lengths)
    print(f"length-vs-score correlation (benign test sessions): {bias['correlation']:.3f} (p={bias['p_value']:.4f})")

    calibrated = percentile_calibrate_by_length(
        test_scores, test_lengths, val_scores, val_lengths, num_buckets=LENGTH_BUCKETS
    )

    threshold_free = compute_threshold_free_metrics(test_labels, test_scores)
    print(f"AUROC = {threshold_free['auroc']:.4f} | AUPRC = {threshold_free['auprc']:.4f}")

    result = {
        "model": name,
        "length_bias_correlation": bias["correlation"],
        "threshold_free": threshold_free,
        "thresholds": {},
    }
    for pct in THRESHOLD_PERCENTILES:
        metrics = compute_threshold_metrics(test_labels, calibrated, threshold_percentile=pct)
        predictions = metrics.pop("predictions")
        print(
            f"  @{pct}th percentile: precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
            f"FPR={metrics['false_positive_rate']:.3f}"
        )
        scenario_breakdown = per_scenario_recall(test_labels, predictions, test_users, scenario_map)
        print(f"    per-scenario recall: {scenario_breakdown}")
        result["thresholds"][pct] = {**metrics, "per_scenario_recall": scenario_breakdown}

    return result


def main():
    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    val_df = events_df[events_df["split"] == "validation"]
    test_df = events_df[events_df["split"] == "test"]

    print("Loading vocabularies and scaler saved during training ...")
    file_type_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_type_vocab.json"))
    file_extension_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, "file_extension_vocab.json"))
    verify_feature_config(CHECKPOINT_DIR)
    scaler = load_scaler(os.path.join(CHECKPOINT_DIR, "numeric_scaler.json"))

    print("Building user -> scenario map from insiders.csv ...")
    malicious_windows = load_malicious_windows()
    scenario_map = malicious_windows.set_index("user")["scenario"].to_dict()

    val_dataset = SessionDataset(val_df, file_type_vocab, file_extension_vocab, scaler)
    test_dataset = SessionDataset(test_df, file_type_vocab, file_extension_vocab, scaler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sessions)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sessions)
    print(f"{len(val_dataset)} validation sessions, {len(test_dataset)} test sessions")

    val_lengths = [val_dataset[i]["numeric"].shape[0] for i in range(len(val_dataset))]
    test_lengths = [test_dataset[i]["numeric"].shape[0] for i in range(len(test_dataset))]

    all_results = []

    # --- Proposed model ---
    transformer_model = DualChannelTransformerAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
    )
    transformer_model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "transformer_autoencoder.pt"))
    )
    val_scores, _ = score_pytorch_model(transformer_model, val_loader, transformer_score_fn)
    test_scores, test_labels = score_pytorch_model(transformer_model, test_loader, transformer_score_fn)
    all_results.append(_report(
        "proposed_transformer_autoencoder", val_scores, val_lengths,
        test_scores, test_lengths, test_labels, test_dataset.session_users, scenario_map,
    ))

    # --- Baseline 1 ---
    lstm_model = LSTMAutoencoder(
        file_type_vocab_size=len(file_type_vocab),
        file_extension_vocab_size=len(file_extension_vocab),
    )
    lstm_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "lstm_autoencoder.pt")))
    val_scores, _ = score_pytorch_model(lstm_model, val_loader, lstm_score_fn)
    test_scores, test_labels = score_pytorch_model(lstm_model, test_loader, lstm_score_fn)
    all_results.append(_report(
        "baseline_lstm_autoencoder", val_scores, val_lengths,
        test_scores, test_lengths, test_labels, test_dataset.session_users, scenario_map,
    ))

    # --- Baseline 2 ---
    static_detector = joblib.load(os.path.join(CHECKPOINT_DIR, "static_vector_detector.joblib"))
    val_vectors, _ = build_session_vectors(val_df, file_type_vocab, file_extension_vocab)
    test_vectors, test_labels = build_session_vectors(test_df, file_type_vocab, file_extension_vocab)
    test_users = get_session_users(test_df)

    # session_to_vector always puts event count (session length) first.
    static_val_lengths = val_vectors[:, 0].tolist()
    static_test_lengths = test_vectors[:, 0].tolist()

    val_scores = score_static_model(static_detector, val_vectors)
    test_scores = score_static_model(static_detector, test_vectors)
    all_results.append(_report(
        "baseline_static_vector_isolation_forest", val_scores, static_val_lengths,
        test_scores, static_test_lengths, test_labels.tolist(), test_users, scenario_map,
    ))

    results_path = os.path.join(CHECKPOINT_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()