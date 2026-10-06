"""
Evaluates every registered model (src/unsupervised/models/registry.py) from one run
folder: AUROC/AUPRC on raw scores, plus threshold metrics at the 90th /
95th / 99th percentile of length-calibrated scores and per-scenario
recall.

This is the quick per-run report. For the final, interval-carrying
comparison across seeds and both labelings, use scripts/unsupervised/final_evaluation.py.

    INSIDER_CHECKPOINT_DIR   run folder (default "checkpoints")
    INSIDER_TEST_PARTITION   dev (default) | locked | all
    INSIDER_MODELS           comma-separated subset of models

Example:
    INSIDER_CHECKPOINT_DIR=checkpoints/all_seed1 python scripts/unsupervised/evaluate_models.py
"""

import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.unsupervised.models.registry import get_model_specs
from src.common.preprocessing.labeling import load_malicious_windows
from src.common.evaluation.partition import select_test_df, get_partition_name
from src.unsupervised.evaluation.scoring import session_table, load_run_artifacts, score_sessions, has_checkpoint
from src.common.evaluation.metrics import (
    diagnose_length_bias,
    percentile_calibrate_by_length,
    compute_threshold_metrics,
    compute_threshold_free_metrics,
    per_scenario_recall,
)
from src.common.run_config import get_checkpoint_dir

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
CHECKPOINT_DIR = get_checkpoint_dir()
BATCH_SIZE = 64
THRESHOLD_PERCENTILES = [90, 95, 99]
LENGTH_BUCKETS = 5


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
    test_df = select_test_df(events_df)

    print(f"Loading vocabularies and scaler from {CHECKPOINT_DIR} ...")
    artifacts = load_run_artifacts(CHECKPOINT_DIR)  # also verifies the feature configuration

    scenario_map = load_malicious_windows().set_index("user")["scenario"].to_dict()

    val_table = session_table(val_df)
    test_table = session_table(test_df)
    print(f"{len(val_table)} validation sessions, {len(test_table)} test sessions")
    val_lengths = val_table["n_events"].tolist()
    test_lengths = test_table["n_events"].tolist()
    test_labels = test_table["window_label"].tolist()
    test_users = test_table["user"].tolist()

    all_results = []
    for spec in get_model_specs():
        if not has_checkpoint(spec, CHECKPOINT_DIR):
            print(f"\n[skipped] {spec.name}: no checkpoint in {CHECKPOINT_DIR}")
            continue
        missing = [c for c in spec.required_columns if c not in events_df.columns]
        if missing:
            print(f"\n[skipped] {spec.name}: dataset lacks column(s) {missing}")
            continue
        val_scores = score_sessions(spec, val_df, CHECKPOINT_DIR, artifacts, BATCH_SIZE)
        test_scores = score_sessions(spec, test_df, CHECKPOINT_DIR, artifacts, BATCH_SIZE)
        all_results.append(_report(
            spec.name, val_scores.tolist(), val_lengths,
            test_scores.tolist(), test_lengths, test_labels, test_users, scenario_map,
        ))

    # One results file per partition, so a locked-partition run can never
    # overwrite (or be confused with) the development results.
    results_path = os.path.join(CHECKPOINT_DIR, f"evaluation_results_{get_partition_name()}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()
