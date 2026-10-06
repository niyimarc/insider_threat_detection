import pandas as pd
import torch

from src.training.dataset import (
    session_to_tensors,
    NUMERIC_FEATURE_COLUMNS,
    SessionDataset,
    collate_sessions,
)
from src.training.vocab import build_vocab


def _toy_session_df():
    # Three events in one session, deliberately out of date order, to
    # confirm session_to_tensors sorts them before building the tensors.
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2010-06-01 10:05:00",
            "2010-06-01 10:00:00",
            "2010-06-01 10:10:00",
        ]),
        "file_type": ["pdf", "office_document", "zip"],
        "file_extension": ["pdf", "doc", "zip"],
        "seconds_since_last_event": [5.0, 0.0, 10.0],
        "seconds_since_device_connect": [65.0, 60.0, 75.0],
        "copies_since_device_connect": [1, 0, 2],
        "outside_working_hours": [False, False, True],
        "user_session_volume_deviation": [1.0, 1.0, 1.0],
        "user_session_frequency_deviation": [1.0, 1.0, 1.0],
        "http_new_domains_today": [0.0, 0.0, 0.0],
        "http_new_domains_7d": [0.0, 0.0, 0.0],
        "http_offbaseline_share_7d": [0.0, 0.0, 0.0],
        "session_is_malicious": [True, True, True],
    })


def test_session_to_tensors_sorts_events_by_date():
    session_df = _toy_session_df()
    file_type_vocab = build_vocab(session_df["file_type"])
    file_extension_vocab = build_vocab(session_df["file_extension"])

    result = session_to_tensors(session_df, file_type_vocab, file_extension_vocab)

    # after sorting by date: office_document (10:00), pdf (10:05), zip (10:10)
    expected_order = ["office_document", "pdf", "zip"]
    expected_indices = [file_type_vocab[v] for v in expected_order]
    assert result["file_type"].tolist() == expected_indices


def test_session_to_tensors_shapes_and_dtypes():
    session_df = _toy_session_df()
    file_type_vocab = build_vocab(session_df["file_type"])
    file_extension_vocab = build_vocab(session_df["file_extension"])

    result = session_to_tensors(session_df, file_type_vocab, file_extension_vocab)

    num_events = len(session_df)
    assert result["numeric"].shape == (num_events, len(NUMERIC_FEATURE_COLUMNS))
    assert result["numeric"].dtype == torch.float32
    assert result["file_type"].shape == (num_events,)
    assert result["file_type"].dtype == torch.long
    assert result["file_extension"].shape == (num_events,)
    assert result["file_extension"].dtype == torch.long
    assert result["label"].shape == ()  # scalar
    assert result["label"].dtype == torch.float32


def test_session_to_tensors_label_reflects_session_is_malicious():
    malicious_df = _toy_session_df()
    file_type_vocab = build_vocab(malicious_df["file_type"])
    file_extension_vocab = build_vocab(malicious_df["file_extension"])
    malicious_result = session_to_tensors(malicious_df, file_type_vocab, file_extension_vocab)
    assert malicious_result["label"].item() == 1.0

    benign_df = _toy_session_df()
    benign_df["session_is_malicious"] = False
    benign_result = session_to_tensors(benign_df, file_type_vocab, file_extension_vocab)
    assert benign_result["label"].item() == 0.0


def test_session_to_tensors_uses_unknown_for_extension_unseen_in_vocab():
    session_df = _toy_session_df()
    # vocab built without "zip" in it at all -- simulates an extension
    # that appears in validation/test but never in training.
    train_only_df = session_df[session_df["file_extension"] != "zip"]
    file_type_vocab = build_vocab(train_only_df["file_type"])
    file_extension_vocab = build_vocab(train_only_df["file_extension"])

    result = session_to_tensors(session_df, file_type_vocab, file_extension_vocab)

    # after sorting by date: office_document(doc), pdf(pdf), zip(zip - unknown)
    from src.training.vocab import UNKNOWN_TOKEN
    assert result["file_extension"].tolist()[-1] == file_extension_vocab[UNKNOWN_TOKEN]


def _two_session_events_df():
    # Session A (user AAA0001): 2 events, benign.
    # Session B (user BBB0002): 3 events, malicious.
    # Different lengths on purpose, to exercise padding.
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
        "http_offbaseline_share_7d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "session_is_malicious": [False, False, True, True, True],
    })

def test_session_dataset_groups_rows_into_one_sample_per_session():
    events_df = _two_session_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])

    dataset = SessionDataset(events_df, file_type_vocab, file_extension_vocab)

    assert len(dataset) == 2  # two distinct sessions, regardless of event count
    lengths = sorted(dataset[i]["numeric"].shape[0] for i in range(len(dataset)))
    assert lengths == [2, 3]  # session A has 2 events, session B has 3


def test_session_dataset_getitem_label_matches_session():
    events_df = _two_session_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])

    dataset = SessionDataset(events_df, file_type_vocab, file_extension_vocab)
    labels = sorted(dataset[i]["label"].item() for i in range(len(dataset)))
    assert labels == [0.0, 1.0]  # one benign session, one malicious session


def test_collate_sessions_pads_to_longest_session_in_batch():
    events_df = _two_session_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])
    dataset = SessionDataset(events_df, file_type_vocab, file_extension_vocab)

    batch = collate_sessions([dataset[i] for i in range(len(dataset))])

    batch_size = 2
    max_len = 3  # the longer of the two sessions
    assert batch["numeric"].shape == (batch_size, max_len, len(NUMERIC_FEATURE_COLUMNS))
    assert batch["file_type"].shape == (batch_size, max_len)
    assert batch["file_extension"].shape == (batch_size, max_len)
    assert batch["mask"].shape == (batch_size, max_len)
    assert batch["label"].shape == (batch_size,)


def test_collate_sessions_mask_marks_padding_correctly():
    events_df = _two_session_events_df()
    file_type_vocab = build_vocab(events_df["file_type"])
    file_extension_vocab = build_vocab(events_df["file_extension"])
    dataset = SessionDataset(events_df, file_type_vocab, file_extension_vocab)

    # Order samples explicitly: shorter session first, longer second.
    samples = [dataset[i] for i in range(len(dataset))]
    samples.sort(key=lambda s: s["numeric"].shape[0])
    batch = collate_sessions(samples)

    short_row_mask = batch["mask"][0]
    long_row_mask = batch["mask"][1]

    assert short_row_mask.tolist() == [True, True, False]  # padded at the end
    assert long_row_mask.tolist() == [True, True, True]  # no padding needed
