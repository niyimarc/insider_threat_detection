import numpy as np
import pandas as pd
import pytest

from src.unsupervised.evaluation.scoring import has_checkpoint, score_heuristic, score_sessions, session_table
from src.unsupervised.models.registry import get_spec


def _events():
    rows = []

    def add(user, start, values, malicious=False):
        start = pd.Timestamp(start)
        for i, v in enumerate(values):
            rows.append({
                "user": user, "session_start": start, "session_end": start + pd.Timedelta(hours=8),
                "date": start + pd.Timedelta(minutes=10 * i),
                "http_offbaseline_share_7d": v, "session_is_malicious": int(malicious),
            })
    add("B", "2010-06-02 08:00:00", [0.2, 0.4])                     # deliberately out of user order
    add("A", "2010-06-03 08:00:00", [0.0])
    add("A", "2010-06-01 08:00:00", [0.1, 0.1, 0.4], malicious=True)
    return pd.DataFrame(rows)


def test_session_table_is_in_canonical_order_with_counts_and_labels():
    table = session_table(_events())
    assert list(table["user"]) == ["A", "A", "B"]
    assert list(table["session_start"]) == sorted(table["session_start"][:2]) + [pd.Timestamp("2010-06-02 08:00:00")]
    assert list(table["n_events"]) == [3, 1, 2]
    assert list(table["window_label"]) == [1, 0, 0]


def test_heuristic_scores_follow_the_same_order_as_the_session_table():
    events = _events()
    mean_scores = score_heuristic(get_spec("single_feature_http"), events)
    assert np.allclose(mean_scores, [0.2, 0.0, 0.3])
    length_scores = score_heuristic(get_spec("session_length_only"), events)
    assert np.allclose(length_scores, session_table(events)["n_events"].to_numpy())


def test_score_sessions_handles_heuristics_without_a_run_folder():
    scores = score_sessions(get_spec("single_feature_http"), _events())
    assert scores.shape == (3,)


def test_heuristic_needing_a_missing_column_raises_clearly():
    with pytest.raises(ValueError, match="http_offbaseline_share_7d"):
        score_sessions(get_spec("single_feature_http"), _events().drop(columns=["http_offbaseline_share_7d"]))


def test_trained_models_require_a_run_folder():
    with pytest.raises(ValueError, match="run_dir"):
        score_sessions(get_spec("lstm_ae"), _events())


def test_has_checkpoint(tmp_path):
    assert has_checkpoint(get_spec("single_feature_http"), str(tmp_path))  # heuristics need none
    assert not has_checkpoint(get_spec("lstm_ae"), str(tmp_path))
    (tmp_path / "lstm_autoencoder.pt").write_bytes(b"x")
    assert has_checkpoint(get_spec("lstm_ae"), str(tmp_path))


def test_missing_checkpoint_raises_a_clear_error(tmp_path):
    artifacts = {"file_type_vocab": {"<unknown>": 0}, "file_extension_vocab": {"<unknown>": 0}, "scaler": {}}
    with pytest.raises(FileNotFoundError, match="train lstm_ae first"):
        score_sessions(get_spec("lstm_ae"), _events(), str(tmp_path), artifacts)


def test_session_order_matches_the_training_dataset_order():
    # The whole evaluation relies on score i meaning the same session in
    # every model's output -- including SessionDataset's order.
    from src.unsupervised.training.dataset import SessionDataset
    from src.common.preprocessing.feature_groups import get_numeric_feature_columns
    from src.unsupervised.training.vocab import build_vocab

    events = _events().assign(file_type="pdf", file_extension="pdf")
    for column in get_numeric_feature_columns():
        if column not in events.columns:
            events[column] = 0.0
    dataset = SessionDataset(events, build_vocab(events["file_type"]), build_vocab(events["file_extension"]))
    assert dataset.session_users == list(session_table(events)["user"])
