import numpy as np
import pytest

from src.common.evaluation.bootstrap import make_user_resamples
from src.common.evaluation.stratified import (
    coverage,
    evaluate_stratified,
    format_stratified_markdown,
    stratified_auroc,
    stratified_metric_over_resamples,
)


def _brute_force(labels, scores, strata):
    concordant = total = 0.0
    for stratum in np.unique(strata):
        rows = strata == stratum
        pos, neg = scores[rows & (labels == 1)], scores[rows & (labels == 0)]
        for a in pos:
            for b in neg:
                total += 1
                concordant += 1.0 if a > b else 0.5 if a == b else 0.0
    return concordant / total if total else float("nan")


def test_matches_a_brute_force_pair_count_including_ties_and_one_class_strata():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 60))
        strata = rng.integers(0, int(rng.integers(1, 8)), n)
        labels = (rng.random(n) < rng.random()).astype(int)
        scores = np.round(rng.normal(size=n), int(rng.integers(0, 3)))  # coarse rounding makes many ties
        expected = _brute_force(labels, scores, strata)
        got = stratified_auroc(labels, scores, strata)
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected, abs=1e-12)


def test_a_score_that_only_knows_the_stratum_gets_chance():
    strata = np.repeat(np.arange(10), 20)
    labels = np.where(np.arange(len(strata)) % 2 == 0, (strata >= 5).astype(int), 0)  # labels live in late strata only
    assert stratified_auroc(labels, strata.astype(float), strata) == pytest.approx(0.5)


def test_a_score_that_separates_within_strata_gets_one():
    strata = np.repeat(np.arange(4), 10)
    labels = np.tile([1, 0], 20)
    scores = labels + strata * 100.0     # huge stratum offsets must not matter
    assert stratified_auroc(labels, scores, strata) == pytest.approx(1.0)
    assert stratified_auroc(labels, -scores + 2 * strata * 100.0, strata) == pytest.approx(0.0)


def test_no_stratum_with_both_classes_gives_nan():
    assert np.isnan(stratified_auroc([1, 1, 0, 0], [1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]))
    assert np.isnan(stratified_auroc([], [], []))


def test_coverage_counts_only_strata_with_both_classes():
    labels = np.array([1, 0, 0, 0, 1, 1])
    strata = np.array([0, 0, 1, 1, 2, 2])      # stratum 1 has no positive, stratum 2 no negative
    assert coverage(labels, strata) == {"n_strata": 3, "n_strata_used": 1, "n_positive": 3, "n_positive_used": 1}


def test_resample_version_matches_the_direct_metric_on_the_identity_resample():
    rng = np.random.default_rng(1)
    labels = (rng.random(60) < 0.3).astype(int)
    scores = rng.normal(size=(2, 60))
    strata = rng.integers(0, 4, 60)
    identity = [np.arange(60)]
    got = stratified_metric_over_resamples(labels, scores, strata, identity)[0]
    expected = np.mean([stratified_auroc(labels, run, strata) for run in scores])
    assert got == pytest.approx(expected)


def test_evaluate_stratified_reports_intervals_and_a_paired_difference():
    rng = np.random.default_rng(2)
    users = np.repeat([f"U{i}" for i in range(20)], 15)
    strata = np.tile(np.repeat(np.arange(3), 5), 20)
    labels = (rng.random(len(users)) < 0.3).astype(int)
    good = labels + rng.normal(scale=0.5, size=len(users))
    noise = rng.normal(size=len(users))
    resamples = make_user_resamples(users, n_boot=50, seed=0)
    result = evaluate_stratified(labels, {"good": good, "noise": noise}, strata, resamples, reference="noise")
    assert result["models"]["good"]["auroc"]["mean"] > 0.8
    assert result["models"]["good"]["auroc"]["ci_low"] <= result["models"]["good"]["auroc"]["mean"] <= result["models"]["good"]["auroc"]["ci_high"]
    assert result["models"]["good"]["auroc_minus_reference"]["ci_excludes_zero"] is True
    assert "auroc_minus_reference" not in result["models"]["noise"]
    text = format_stratified_markdown("T", "D", result, {"good": "Good model"})
    assert "Good model" in text and "calendar strata" in text and "(reference)" in text
