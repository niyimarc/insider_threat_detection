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
from scipy.stats import spearmanr


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
    src/common/preprocessing/labeling.py's load_malicious_windows), mapping
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


def diagnose_length_bias(scores, lengths):
    """
    Spearman correlation between a session's reconstruction score and
    its length (number of events). A strong correlation here -- before
    any fix -- would mean the model's anomaly score is partly just
    tracking "how many events were in this session" rather than
    "how unusual was this session's behaviour", since an average taken
    over fewer events naturally swings further from the mean than one
    taken over many (basic statistics: variance of a mean shrinks as
    the sample size grows), independent of whether the CONTENT is
    actually unusual.
    """
    correlation, p_value = spearmanr(scores, lengths)
    return {"correlation": correlation, "p_value": p_value}


def build_length_buckets(reference_lengths, num_buckets=5):
    """
    Computes bucket boundaries from the DISTINCT session-length values
    seen in the reference (validation, benign-only) data, using
    quantiles over those unique values rather than the raw (repeated)
    lengths. This matters when lengths are skewed -- e.g. far more
    short sessions than long ones -- since quantiling the raw repeated
    values can let a numerically dominant group collapse the edges
    entirely, leaving no boundary between short and long sessions at all.
    """
    unique_lengths = np.unique(reference_lengths)
    quantiles = np.linspace(0, 1, num_buckets + 1)
    return np.unique(np.quantile(unique_lengths, quantiles))


def assign_length_bucket(lengths, edges):
    """Assigns each length to a bucket index using pre-computed edges
    (from build_length_buckets). Values outside the reference range are
    clipped into the nearest edge bucket rather than erroring."""
    lengths = np.asarray(lengths)
    return np.digitize(lengths, edges[1:-1], right=True)


def percentile_calibrate_by_length(scores, lengths, reference_scores, reference_lengths, num_buckets=5):
    """
    Like percentile_calibrate, but compares each session's score only
    against reference (validation) sessions of SIMILAR LENGTH, instead
    of the whole reference population at once -- this is the actual fix
    for the length confound diagnose_length_bias checks for.
    """
    edges = build_length_buckets(reference_lengths, num_buckets)
    reference_buckets = assign_length_bucket(reference_lengths, edges)
    test_buckets = assign_length_bucket(lengths, edges)

    reference_scores = np.asarray(reference_scores)
    scores = np.asarray(scores)
    calibrated = np.zeros(len(scores))

    for bucket in np.unique(test_buckets):
        bucket_reference_scores = reference_scores[reference_buckets == bucket]
        bucket_mask = test_buckets == bucket
        if len(bucket_reference_scores) == 0:
            # Shouldn't normally happen since buckets are built from this
            # same reference data, but fall back safely if it does.
            bucket_reference_scores = reference_scores
        calibrated[bucket_mask] = percentile_calibrate(scores[bucket_mask], bucket_reference_scores)

    return calibrated