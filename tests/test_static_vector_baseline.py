import numpy as np
import pandas as pd
import pytest

from src.training.vocab import build_vocab
from src.training.dataset import NUMERIC_FEATURE_COLUMNS
from src.preprocessing.feature_groups import BINARY_NUMERIC_COLUMNS
from src.models.static_vector_baseline import (
    session_to_vector,
    build_session_vectors,
    StaticVectorAnomalyDetector,
)

# session_to_vector summarises every continuous numeric column with 4
# stats (mean/std/min/max), and every 0/1 flag column with a single
# "fraction of events flagged" slot. Both counts come from the feature
# registry, so this test stays correct as signals are added or disabled.
_NUM_CONTINUOUS_COLUMNS = len([c for c in NUMERIC_FEATURE_COLUMNS if c not in BINARY_NUMERIC_COLUMNS])
_NUM_BINARY_COLUMNS = len([c for c in NUMERIC_FEATURE_COLUMNS if c in BINARY_NUMERIC_COLUMNS])


def _toy_events_df():
    session_start = pd.to_datetime(["2010-06-01 09:00:00"] * 2 + ["2010-06-02 09:00:00"] * 3)
    session_end = pd.to_datetime(["2010-06-01 17:00:00"] * 2 + ["2010-06-02 17:00:00"] * 3)
    return pd.DataFrame({
        "user": ["AAA0001", "AAA0001", "BBB0002", "BBB0002", "BBB0002"],
        "session_start": session_start,
        "session_end": session_end,
        "date": pd.to_datetime([
            "2010-06-01 10:00:00", "2010-06-01 10:05:00",
            "2010-06-02 10:00:00", "2010-06-02 10:05:00", "2010-06-02 10:10:00",
        ]),
        "file_type": ["pdf", "zip", "office_document", "pdf", "exe"],
        "file_extension": ["pdf", "zip", "doc", "pdf", "exe"],
        "seconds_since_last_event": [0.0, 5.0, 0.0, 5.0, 5.0],
        "seconds_since_device_connect": [10.0, 15.0, 20.0, 25.0, 30.0],
        "copies_since_device_connect": [0, 1, 0, 1, 2],
        "outside_working_hours": [False, False, True, True, True],
        "user_session_volume_deviation": [1.0, 1.0, 1.0, 1.0, 1.0],
        "user_session_frequency_deviation": [1.0, 1.0, 1.0, 1.0, 1.0],
        "http_new_domains_today": [0.0, 0.0, 0.0, 0.0, 0.0],
        "http_new_domains_7d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "session_is_malicious": [False, False, True, True, True],
    })

def test_session_to_vector_has_expected_length_and_event_count():
    events_df = _toy_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])
    session_a = events_df[events_df["user"] == "AAA0001"]  # 2 events

    vector = session_to_vector(session_a, file_type_vocab, file_extension_vocab)

    # 1 (count) + continuous features * 4 stats + one fraction per 0/1 flag
    # column + len(file_type_vocab) + len(file_extension_vocab)
    expected_length = (
        1 + _NUM_CONTINUOUS_COLUMNS * 4 + _NUM_BINARY_COLUMNS
        + len(file_type_vocab) + len(file_extension_vocab)
    )
    assert vector.shape == (expected_length,)
    assert vector[0] == 2  # session_a has 2 events


@pytest.mark.skipif(
    "outside_working_hours" not in NUMERIC_FEATURE_COLUMNS,
    reason="the logon feature group is disabled in this configuration",
)
def test_session_to_vector_outside_hours_fraction_is_correct():
    events_df = _toy_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])
    session_b = events_df[events_df["user"] == "BBB0002"]  # all 3 outside hours

    vector = session_to_vector(session_b, file_type_vocab, file_extension_vocab)
    outside_hours_fraction_index = 1 + _NUM_CONTINUOUS_COLUMNS * 4  # right after count + continuous stats
    assert vector[outside_hours_fraction_index] == 1.0


def test_build_session_vectors_matches_number_of_sessions_and_labels():
    events_df = _toy_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])

    vectors, labels = build_session_vectors(events_df, file_type_vocab, file_extension_vocab)

    assert vectors.shape[0] == 2  # two sessions
    assert sorted(labels.tolist()) == [0.0, 1.0]  # one benign, one malicious


def test_anomaly_detector_scores_a_clear_outlier_higher_than_normal_points():
    # Sanity check that IsolationForest is actually being used as an
    # anomaly detector, not just wired up to run without error: fit on
    # a tight cluster of "normal" points, then confirm a point far
    # outside that cluster scores as more anomalous than an in-cluster
    # point -- this is the classical-baseline equivalent of the
    # "model can learn" test used for the two PyTorch models.
    rng = np.random.default_rng(42)
    normal_points = rng.normal(loc=0.0, scale=0.1, size=(100, 5))
    detector = StaticVectorAnomalyDetector(n_estimators=50).fit(normal_points)

    in_cluster_point = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    outlier_point = np.array([[50.0, 50.0, 50.0, 50.0, 50.0]])

    in_cluster_score = detector.anomaly_score(in_cluster_point)[0]
    outlier_score = detector.anomaly_score(outlier_point)[0]

    assert outlier_score > in_cluster_score


def test_anomaly_score_returns_one_value_per_input_vector():
    rng = np.random.default_rng(0)
    normal_points = rng.normal(size=(50, 4))
    detector = StaticVectorAnomalyDetector(n_estimators=20).fit(normal_points)

    scores = detector.anomaly_score(rng.normal(size=(10, 4)))

    assert scores.shape == (10,)
    assert np.isfinite(scores).all()