import json
import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.unsupervised.evaluation.scoring import load_run_artifacts, score_sessions, session_table
from src.unsupervised.models.registry import get_model_specs, get_spec
from src.common.preprocessing.feature_groups import get_scaled_columns
from src.unsupervised.training.dataset import SessionDataset, collate_sessions
from src.unsupervised.training.feature_config import save_feature_config
from src.unsupervised.training.scaling import build_scaler, save_scaler
from src.unsupervised.training.train_registry import train_registered_model
from src.unsupervised.training.vocab import build_vocab, save_vocab


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
        "http_offbaseline_share_7d": [0.0, 0.1, 0.0, 0.2, 0.3],
        "session_is_malicious": [False, False, False, False, False],
    })


def test_every_registered_model_trains_saves_loads_and_scores(tmp_path):
    """End-to-end through the registry: train -> checkpoint on disk -> load -> score."""
    run_dir = str(tmp_path)
    events = _toy_events_df()
    file_type_vocab = build_vocab(events["file_type"])
    file_extension_vocab = build_vocab(events["file_extension"])
    scaler = build_scaler(events, get_scaled_columns())
    save_vocab(file_type_vocab, os.path.join(run_dir, "file_type_vocab.json"))
    save_vocab(file_extension_vocab, os.path.join(run_dir, "file_extension_vocab.json"))
    save_scaler(scaler, os.path.join(run_dir, "numeric_scaler.json"))
    save_feature_config(run_dir)

    dataset = SessionDataset(events, file_type_vocab, file_extension_vocab, scaler)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_sessions)

    for spec in get_model_specs():
        history = train_registered_model(
            spec, train_df=events, train_loader=loader, val_loader=loader,
            file_type_vocab=file_type_vocab, file_extension_vocab=file_extension_vocab,
            checkpoint_dir=run_dir, seed=0, device="cpu", num_epochs=1, patience=5,
        )
        if spec.kind == "torch":
            assert "val_loss" in history
            history_file = os.path.splitext(spec.checkpoint_file)[0] + "_history.json"
            with open(os.path.join(run_dir, history_file)) as f:
                assert "train_loss" in json.load(f)
        else:
            assert history is None
        if spec.checkpoint_file:
            assert os.path.exists(os.path.join(run_dir, spec.checkpoint_file))

    artifacts = load_run_artifacts(run_dir)
    expected_sessions = len(session_table(events))
    for spec in get_model_specs():
        scores = score_sessions(spec, events, run_dir, artifacts)
        assert scores.shape == (expected_sessions,), spec.name
        assert np.isfinite(scores).all(), spec.name


def test_scoring_a_saved_torch_model_matches_scoring_it_directly(tmp_path):
    from src.unsupervised.evaluation.score import score_pytorch_model
    from src.unsupervised.training.train_registry import _cpu_state_dict
    import torch

    run_dir = str(tmp_path)
    events = _toy_events_df()
    file_type_vocab = build_vocab(events["file_type"])
    file_extension_vocab = build_vocab(events["file_extension"])
    scaler = build_scaler(events, get_scaled_columns())
    save_vocab(file_type_vocab, os.path.join(run_dir, "file_type_vocab.json"))
    save_vocab(file_extension_vocab, os.path.join(run_dir, "file_extension_vocab.json"))
    save_scaler(scaler, os.path.join(run_dir, "numeric_scaler.json"))
    save_feature_config(run_dir)

    spec = get_spec("lstm_ae")
    torch.manual_seed(0)
    model = spec.build(len(file_type_vocab), len(file_extension_vocab))
    torch.save(_cpu_state_dict(model), os.path.join(run_dir, spec.checkpoint_file))

    loader = DataLoader(SessionDataset(events, file_type_vocab, file_extension_vocab, scaler),
                        batch_size=64, shuffle=False, collate_fn=collate_sessions)
    direct, _ = score_pytorch_model(model, loader, spec.score_fn)
    via_registry = score_sessions(spec, events, run_dir, load_run_artifacts(run_dir))
    assert np.allclose(direct, via_registry, atol=1e-5)
