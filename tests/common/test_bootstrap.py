import numpy as np
import pytest

from src.common.evaluation.bootstrap import (
    interval,
    make_user_resamples,
    metric_over_resamples,
    paired_difference,
    safe_auprc,
    safe_auroc,
)


def test_safe_metrics_return_nan_when_one_class_only():
    assert np.isnan(safe_auroc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(safe_auroc([1, 1], [0.1, 0.2]))
    assert np.isnan(safe_auprc([0, 0], [0.1, 0.2]))


def test_safe_auroc_matches_the_obvious_cases():
    assert safe_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert safe_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_resamples_are_made_of_whole_users():
    users = np.array(["A", "A", "B", "B", "B", "C"])
    resamples = make_user_resamples(users, n_boot=50, seed=1)
    assert len(resamples) == 50
    sizes = {"A": 2, "B": 3, "C": 1}
    for idx in resamples:
        counts = {u: int((users[idx] == u).sum()) for u in sizes}
        # a user's sessions come all together, a whole number of times
        for u, size in sizes.items():
            assert counts[u] % size == 0
        assert len(idx) >= 1


def test_resamples_are_reproducible_and_seed_dependent():
    users = np.array(list("AABBCCDD"))
    a = make_user_resamples(users, n_boot=10, seed=3)
    b = make_user_resamples(users, n_boot=10, seed=3)
    c = make_user_resamples(users, n_boot=10, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert any(not np.array_equal(x, y) for x, y in zip(a, c))


def _toy(n_users=30, sessions_per_user=6, signal=2.0, seed=0):
    rng = np.random.default_rng(seed)
    users, labels, scores = [], [], []
    for u in range(n_users):
        insider = u < n_users // 3
        for _ in range(sessions_per_user):
            malicious = insider and rng.random() < 0.5
            users.append(f"U{u}")
            labels.append(int(malicious))
            scores.append(rng.normal(loc=signal if malicious else 0.0))
    return np.array(users), np.array(labels), np.array(scores)


def test_interval_contains_the_point_estimate_for_an_informative_score():
    users, labels, scores = _toy(signal=1.5)
    resamples = make_user_resamples(users, n_boot=200, seed=0)
    values = metric_over_resamples(labels, scores, resamples, safe_auroc)
    low, high = interval(values)
    point = safe_auroc(labels, scores)
    assert low < point < high
    assert low > 0.5  # a real signal: the interval excludes chance


def test_interval_for_a_random_score_includes_chance():
    users, labels, _ = _toy()
    scores = np.random.default_rng(5).random(len(labels))
    resamples = make_user_resamples(users, n_boot=200, seed=0)
    low, high = interval(metric_over_resamples(labels, scores, resamples, safe_auroc))
    assert low < 0.5 < high


def test_user_level_interval_is_wider_than_a_session_level_one():
    # Strong user effect: each user has a personal offset, so sessions of a
    # user are NOT independent. Resampling users must give a wider interval
    # than (incorrectly) resampling sessions.
    rng = np.random.default_rng(0)
    users, labels, scores = [], [], []
    for u in range(40):
        offset = rng.normal(scale=2.0)
        insider = u < 15
        for _ in range(8):
            users.append(f"U{u}")
            labels.append(int(insider and rng.random() < 0.6))
            scores.append(offset + rng.normal(scale=0.3))
    users, labels, scores = np.array(users), np.array(labels), np.array(scores)

    user_values = metric_over_resamples(labels, scores, make_user_resamples(users, 300, seed=1), safe_auroc)
    session_resamples = [np.random.default_rng(i).integers(0, len(labels), len(labels)) for i in range(300)]
    session_values = metric_over_resamples(labels, scores, session_resamples, safe_auroc)
    u_low, u_high = interval(user_values)
    s_low, s_high = interval(session_values)
    assert (u_high - u_low) > (s_high - s_low)


def test_seed_mean_over_a_score_matrix():
    users, labels, scores = _toy(signal=1.5)
    matrix = np.stack([scores, scores])  # two identical "seeds"
    resamples = make_user_resamples(users, n_boot=20, seed=0)
    single = metric_over_resamples(labels, scores, resamples, safe_auroc)
    averaged = metric_over_resamples(labels, matrix, resamples, safe_auroc)
    assert np.allclose(single, averaged, equal_nan=True)


def test_interval_ignores_nan_resamples_and_handles_all_nan():
    assert interval([0.4, np.nan, 0.6, 0.5])[0] >= 0.4
    low, high = interval([np.nan, np.nan])
    assert np.isnan(low) and np.isnan(high)


def test_paired_difference_is_elementwise():
    assert np.allclose(paired_difference([0.6, 0.7], [0.5, 0.5]), [0.1, 0.2])
