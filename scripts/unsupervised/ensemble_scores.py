"""
Combines the TRAINED registered models' scores into ensembles, to check
whether they pick up different/complementary signal.

How this stays leak-free: raw scores live on totally different scales
(LSTM reconstruction loss ~0.1, Transformer ~0.003, Isolation Forest
unbounded), so each model's raw score is first converted to a 0-100
percentile against that SAME model's own VALIDATION (benign-only) score
distribution -- the same calibration src/common/evaluation/metrics.py uses for
thresholding -- and only then averaged. The reference distribution is
validation, not test, and the combination rule (unweighted mean) is fixed
in advance: choosing weights by checking test labels would be leakage.

Heuristic baselines are not part of the ensemble (they are baselines, not
trained detectors). Read-only: nothing is retrained or overwritten.

    INSIDER_CHECKPOINT_DIR   run folder (default "checkpoints")
    INSIDER_TEST_PARTITION   dev (default) | locked | all
    INSIDER_MODELS           comma-separated subset of models

Run from the project root:
    python scripts/unsupervised/ensemble_scores.py
"""

import itertools
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.unsupervised.models.registry import get_model_specs
from src.common.preprocessing.labeling import load_malicious_windows
from src.common.evaluation.partition import select_test_df
from src.unsupervised.evaluation.scoring import session_table, load_run_artifacts, score_sessions, has_checkpoint
from src.common.evaluation.metrics import percentile_calibrate
from src.common.run_config import get_checkpoint_dir

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
CHECKPOINT_DIR = get_checkpoint_dir()
BATCH_SIZE = 64


def _auroc(scores, labels, keep):
    scores, labels = np.asarray(scores)[keep], np.asarray(labels)[keep]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None, None
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def report_subsets(name, scores, labels, users, scenario_map):
    labels = np.asarray(labels)
    scenario_of_session = np.array([scenario_map.get(u, 0) for u in users])
    benign = labels == 0
    malicious = labels == 1

    auroc_all, auprc_all = _auroc(scores, labels, benign | malicious)
    auroc_short, _ = _auroc(scores, labels, benign | (malicious & np.isin(scenario_of_session, [1, 3])))
    auroc_long, _ = _auroc(scores, labels, benign | (malicious & (scenario_of_session == 2)))

    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    print(f"  {name:40s} all AUROC={fmt(auroc_all)} AUPRC={fmt(auprc_all)}  "
          f"s1+3={fmt(auroc_short)}  s2={fmt(auroc_long)}")


def main():
    print(f"Loading {DATA_PATH} ...")
    events_df = pd.read_parquet(DATA_PATH)
    val_df = events_df[events_df["split"] == "validation"]
    test_df = select_test_df(events_df)

    artifacts = load_run_artifacts(CHECKPOINT_DIR)
    scenario_map = load_malicious_windows().set_index("user")["scenario"].to_dict()

    table = session_table(test_df)
    labels, users = table["window_label"].tolist(), table["user"].tolist()

    calibrated = {}
    for spec in get_model_specs():
        if not spec.is_trainable or not has_checkpoint(spec, CHECKPOINT_DIR):
            continue
        val_scores = score_sessions(spec, val_df, CHECKPOINT_DIR, artifacts, BATCH_SIZE)
        test_scores = score_sessions(spec, test_df, CHECKPOINT_DIR, artifacts, BATCH_SIZE)
        calibrated[spec.name] = percentile_calibrate(test_scores, val_scores)

    if len(calibrated) < 2:
        raise SystemExit(f"Need at least two trained models to ensemble; found {list(calibrated)}.")

    print("\n=== Individual models (calibrated against validation, for reference) ===")
    for name, scores in calibrated.items():
        report_subsets(name, scores, labels, users, scenario_map)

    print("\n=== Ensembles (unweighted mean of calibrated percentiles) ===")
    names = list(calibrated)
    for size in range(len(names), 1, -1):
        for combo in itertools.combinations(names, size):
            scores = np.mean([calibrated[n] for n in combo], axis=0)
            report_subsets("+".join(combo), scores, labels, users, scenario_map)


if __name__ == "__main__":
    main()
