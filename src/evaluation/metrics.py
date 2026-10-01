"""
Evaluation metrics for the anomaly-detection framing established in
Chapter 3.5-3.7: models never see malicious sessions during training, so
detection at test time works by comparing each session's reconstruction
error against a threshold calibrated on VALIDATION (benign-only) data,
rather than by a conventional trained classifier's decision boundary.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def percentile_calibrate(scores, reference_scores):
    """
    For each value in `scores`, returns what percentile of
    `reference_scores` it falls above (0-100) -- following Tuor et al.
    (2017)'s approach, cited in Chapter 3.5, of calibrating against the
    validation reconstruction-error distribution rather than an
    arbitrary fixed cutoff.

    `reference_scores` should be the VALIDATION (benign-only) raw
    anomaly scores -- the "what does normal look like" distribution. A
    calibrated value of 97 means "this session's raw score is worse
    than 97% of normal validation sessions."
    """
    reference_sorted = np.sort(np.asarray(reference_scores))
    # searchsorted gives, for each score, how many reference values are
    # <= it -- exactly the count needed for a percentile rank.
    ranks = np.searchsorted(reference_sorted, scores, side="right")
    return ranks / len(reference_sorted) * 100


def compute_threshold_metrics(labels, calibrated_scores, threshold_percentile):
    """
    Flags any session with calibrated_scores >= threshold_percentile as
    predicted-malicious, then computes precision, recall, F1, detection
    rate (= recall, reported under both names since Chapter 3.7 asks for
    "detection-rate and false-positive-rate pairs" specifically, to
    match how Tuor et al. (2017) and Yumlembam et al. (2025) report
    results) and false positive rate against the true labels.
    """
    labels = np.asarray(labels)
    predictions = (np.asarray(calibrated_scores) >= threshold_percentile).astype(float)

    true_positives = int(((predictions == 1) & (labels == 1)).sum())
    false_positives = int(((predictions == 1) & (labels == 0)).sum())
    false_negatives = int(((predictions == 0) & (labels == 1)).sum())
    true_negatives = int(((predictions == 0) & (labels == 0)).sum())

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0 else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0 else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    false_positive_rate = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0 else 0.0
    )

    return {
        "threshold_percentile": threshold_percentile,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_rate": recall,
        "false_positive_rate": false_positive_rate,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "predictions": predictions,
    }


def compute_threshold_free_metrics(labels, raw_scores):
    """
    AUROC and AUPRC don't need a chosen threshold -- they summarize
    performance across every possible threshold at once. Chapter 3.7
    gives AUPRC particular weight because of r4.2's severe class
    imbalance, where AUROC can look misleadingly good.
    """
    return {
        "auroc": roc_auc_score(labels, raw_scores),
        "auprc": average_precision_score(labels, raw_scores),
    }


def per_scenario_recall(labels, predictions, session_users, scenario_map):
    """
    Breaks recall down by which of r4.2's three scripted scenarios each
    malicious session's user belongs to (Chapter 3.7) -- only malicious
    sessions are considered, since recall is only meaningful for actual
    positives. scenario_map should come from insiders.csv (see
    src/preprocessing/labeling.py's load_malicious_windows), mapping
    user -> scenario number.

    A scenario with zero malicious test sessions returns recall=None
    rather than a misleading 0/0 division.
    """
    df = pd.DataFrame({
        "label": np.asarray(labels),
        "prediction": np.asarray(predictions),
        "user": list(session_users),
    })
    malicious = df[df["label"] == 1].copy()
    malicious["scenario"] = malicious["user"].map(scenario_map)

    results = {}
    for scenario, group in malicious.groupby("scenario"):
        total = len(group)
        detected = int((group["prediction"] == 1).sum())
        results[scenario] = {
            "total_malicious_sessions": total,
            "detected": detected,
            "recall": detected / total if total > 0 else None,
        }
    return results