import numpy as np
import pytest
from src.evaluation.metrics import (
    percentile_calibrate,
    compute_threshold_metrics,
    compute_threshold_free_metrics,
    per_scenario_recall,
    diagnose_length_bias,
    build_length_buckets,
    assign_length_bucket,
    percentile_calibrate_by_length, 
)


def test_percentile_calibrate_known_values():
    reference = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # 10 values
    # A score of 5.0 is >= exactly 5 of the 10 reference values -> 50th percentile
    result = percentile_calibrate([5.0], reference)
    assert result[0] == 50.0

    # A score higher than everything in the reference -> 100th percentile
    result = percentile_calibrate([100.0], reference)
    assert result[0] == 100.0

    # A score lower than everything in the reference -> 0th percentile
    result = percentile_calibrate([0.0], reference)
    assert result[0] == 0.0


def test_compute_threshold_metrics_perfect_separation():
    # Malicious sessions score high (above threshold), benign score low.
    labels = [1, 1, 0, 0]
    calibrated_scores = [99.0, 97.0, 10.0, 5.0]

    result = compute_threshold_metrics(labels, calibrated_scores, threshold_percentile=95)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["false_positive_rate"] == 0.0
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0


def test_compute_threshold_metrics_handles_no_positive_predictions():
    # Nothing crosses the threshold -- precision should not divide by zero.
    labels = [1, 0]
    calibrated_scores = [10.0, 5.0]

    result = compute_threshold_metrics(labels, calibrated_scores, threshold_percentile=95)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["true_positives"] == 0
    assert result["false_positives"] == 0


def test_compute_threshold_free_metrics_perfect_separation():
    labels = [1, 1, 0, 0]
    raw_scores = [10.0, 9.0, 2.0, 1.0]

    result = compute_threshold_free_metrics(labels, raw_scores)

    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0


def test_per_scenario_recall_breaks_down_correctly():
    # 4 malicious sessions: 2 from scenario 1 (1 detected), 2 from
    # scenario 2 (both detected). Benign sessions are ignored entirely.
    labels = [1, 1, 1, 1, 0, 0]
    predictions = [1, 0, 1, 1, 0, 1]  # note: benign false positive at index 5 ignored
    session_users = ["user_a", "user_a2", "user_b", "user_b2", "benign_user", "benign_user2"]
    scenario_map = {"user_a": 1, "user_a2": 1, "user_b": 2, "user_b2": 2}

    result = per_scenario_recall(labels, predictions, session_users, scenario_map)

    assert result[1]["total_malicious_sessions"] == 2
    assert result[1]["detected"] == 1
    assert result[1]["recall"] == 0.5

    assert result[2]["total_malicious_sessions"] == 2
    assert result[2]["detected"] == 2
    assert result[2]["recall"] == 1.0


def test_per_scenario_recall_returns_none_for_scenario_with_no_malicious_sessions():
    labels = [1, 0]
    predictions = [1, 0]
    session_users = ["user_a", "benign_user"]
    scenario_map = {"user_a": 1, "user_never_appears": 2}

    result = per_scenario_recall(labels, predictions, session_users, scenario_map)

    assert 2 not in result  # scenario 2 had zero malicious sessions in this data, so it's just absent
    assert result[1]["recall"] == 1.0

def test_diagnose_length_bias_detects_strong_correlation():
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scores = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # perfectly, strongly anti-correlated

    result = diagnose_length_bias(scores, lengths)

    assert result["correlation"] == pytest.approx(-1.0)


def test_build_length_buckets_and_assign_are_consistent():
    reference_lengths = list(range(1, 101))  # 1..100
    edges = build_length_buckets(reference_lengths, num_buckets=4)

    # every reference length should land in a valid bucket index
    buckets = assign_length_bucket(reference_lengths, edges)
    assert buckets.min() >= 0
    assert buckets.max() <= 3


def test_percentile_calibrate_by_length_compares_within_bucket_not_globally():
    # Reference set dominated by long sessions (90) with few short ones
    # (10) -- a pure length confound, no real behavioural difference
    # encoded, just two different score "regimes" tied to length.
    reference_lengths = [2] * 10 + [20] * 90
    reference_scores = [1.0] * 10 + [10.0] * 90

    # A short test session scoring 1.5 is a clear outlier among short
    # reference sessions (which all scored exactly 1.0), but gets
    # swamped globally by the 90 long sessions all scoring 10.0.
    test_lengths = [2]
    test_scores = [1.5]

    naive = percentile_calibrate(test_scores, reference_scores)
    bucketed = percentile_calibrate_by_length(
        test_scores, test_lengths, reference_scores, reference_lengths, num_buckets=2
    )

    assert naive[0] <= 10  # globally, swamped by the 90 long sessions
    assert bucketed[0] == 100  # within its own length bucket, a clear outlier