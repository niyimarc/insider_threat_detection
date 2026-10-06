"""
Baseline 2: a static session-vector anomaly detector (Mateo Muñoz et al.,
2026), restricted to file-access-only features per Chapter 3.5, rather
than their original multi-source (login/network/email/file/device)
feature vector.

Unlike the proposed model and Baseline 1, this collapses each session
into ONE fixed-length vector of summary statistics -- discarding event
order and sequence structure entirely -- and scores it with a classical
algorithm rather than a neural network. Its purpose in this study isn't
to be competitive on its own; it's to test whether keeping a session as
an ordered SEQUENCE (as the proposed model and Baseline 1 do) actually
buys anything over just aggregating a session's file-access behaviour
into a flat vector, the way most of the wider UEBA literature already
does.

Because training data contains no malicious sessions at all (see
src/common/preprocessing/split.py), this is fit as a one-class problem: an
Isolation Forest trained only on benign session vectors, exactly
mirroring how the two PyTorch-based models are trained only on benign
sequences.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from src.unsupervised.training.dataset import NUMERIC_FEATURE_COLUMNS, SESSION_KEY_COLUMNS
from src.unsupervised.training.vocab import encode
from src.common.preprocessing.feature_groups import BINARY_NUMERIC_COLUMNS


def session_to_vector(session_df, file_type_vocab, file_extension_vocab):
    """
    Collapses one session's rows into a single fixed-length feature
    vector: event count, summary statistics (mean/std/min/max) for each
    continuous numeric feature, the fraction of events outside working
    hours, and the proportion of events falling into each file_type /
    file_extension vocabulary entry.

    file_type_vocab/file_extension_vocab must be the same vocabularies
    used everywhere else in this project (built from train, reused for
    validation/test) -- see src/unsupervised/training/vocab.py -- so the resulting
    vector's dimensions mean the same thing across splits.
    """
    features = []

    features.append(len(session_df))  # session length (event count)

    continuous_columns = [c for c in NUMERIC_FEATURE_COLUMNS if c not in BINARY_NUMERIC_COLUMNS]
    for column in continuous_columns:
        values = session_df[column].astype(float)
        features.extend([
            values.mean(),
            values.std() if len(values) > 1 else 0.0,
            values.min(),
            values.max(),
        ])

    # 0/1 flag columns (e.g. outside_working_hours) are summarised as the
    # fraction of the session's events that are flagged.
    binary_columns = [c for c in NUMERIC_FEATURE_COLUMNS if c in BINARY_NUMERIC_COLUMNS]
    for column in binary_columns:
        features.append(session_df[column].astype(float).mean())

    # Proportion of this session's events falling into each vocabulary
    # entry -- e.g. if 2 of 5 events are "pdf", that category gets 0.4.
    for column, vocab in [("file_type", file_type_vocab), ("file_extension", file_extension_vocab)]:
        encoded = encode(session_df[column], vocab)
        counts = np.zeros(len(vocab))
        for index in encoded:
            counts[index] += 1
        features.extend(counts / len(session_df))

    return np.array(features, dtype=np.float64)


def build_session_vectors(events_df, file_type_vocab, file_extension_vocab):
    """
    Builds one feature vector per session for an entire split, plus the
    matching array of session_is_malicious labels (for evaluation only
    -- never used for fitting, since training is benign-only).

    Returns (vectors, labels): vectors is (num_sessions, vector_dim),
    labels is (num_sessions,) of 0.0/1.0.
    """
    vectors = []
    labels = []
    for _, session_df in events_df.sort_values(SESSION_KEY_COLUMNS).groupby(
        SESSION_KEY_COLUMNS, sort=False
    ):
        vectors.append(session_to_vector(session_df, file_type_vocab, file_extension_vocab))
        labels.append(float(session_df["session_is_malicious"].iloc[0]))

    return np.stack(vectors), np.array(labels)


class StaticVectorAnomalyDetector:
    """
    Thin wrapper around scikit-learn's IsolationForest, kept in the same
    "fit on benign only, score everything" shape as the two PyTorch
    models, so all three baselines can be trained and evaluated through
    the same task-26/task-27 pipeline logic.
    """

    def __init__(self, random_state=42, **isolation_forest_kwargs):
        self.model = IsolationForest(random_state=random_state, **isolation_forest_kwargs)

    def fit(self, benign_vectors):
        """benign_vectors: (num_benign_sessions, vector_dim), built from
        TRAINING data only (see build_session_vectors)."""
        self.model.fit(benign_vectors)
        return self

    def anomaly_score(self, vectors):
        """
        Returns one anomaly score per session, HIGHER = more anomalous
        -- the same convention used by session_anomaly_score() in the
        two PyTorch models, so all three can be compared/thresholded the
        same way downstream. IsolationForest's own score_samples() uses
        the opposite convention (higher = more normal), hence the
        negation.
        """
        return -self.model.score_samples(vectors)