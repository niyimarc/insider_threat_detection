"""
Runs scripts/final_evaluation.py end to end on a small synthetic dataset
(heuristic baselines only, so no trained checkpoints are needed), with the
parquet read and the answer-key loaders replaced by synthetic data.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.training.feature_config import save_feature_config
from src.training.scaling import save_scaler
from src.training.vocab import save_vocab

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "final_evaluation.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("final_evaluation_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_events(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(14):
        user = f"U{u:02d}"
        insider = u < 6
        for s in range(8):
            start = pd.Timestamp("2010-06-01 08:00:00") + pd.Timedelta(days=s)
            malicious = insider and s >= 3
            for e in range(2):
                rows.append({
                    "user": user, "session_start": start, "session_end": start + pd.Timedelta(hours=8),
                    "date": start + pd.Timedelta(minutes=30 * e),
                    # insiders U00-U03 and benign even users are "dev"; insiders U04-U05 and benign odd users are "locked"
                    "split": "test", "test_partition": "dev" if (u < 4 or (u >= 6 and u % 2 == 0)) else "locked",
                    "session_is_malicious": int(malicious),
                    "http_offbaseline_share_7d": float(rng.random() * 0.2 + (0.3 if malicious else 0.0)),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def fake_project(monkeypatch, tmp_path):
    events = _synthetic_events()
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: events.copy())

    windows = pd.DataFrame({"user": [f"U{u:02d}" for u in range(6)], "scenario": [1, 2, 2, 2, 3, 2]})
    import src.preprocessing.labeling as labeling_module
    monkeypatch.setattr(labeling_module, "load_malicious_windows", lambda *a, **k: windows)

    key = {
        "http_days": pd.DataFrame({"user": ["U01", "U02"], "day": pd.to_datetime(["2010-06-05", "2010-06-06"]),
                                   "scenario": [2, 2]}),
        "device_intervals": pd.DataFrame(columns=["user", "scenario", "start", "end"]),
    }

    run_dir = tmp_path / "run_seed1"
    run_dir.mkdir()
    save_vocab({"<unknown>": 0}, str(run_dir / "file_type_vocab.json"))
    save_vocab({"<unknown>": 0}, str(run_dir / "file_extension_vocab.json"))
    save_scaler({}, str(run_dir / "numeric_scaler.json"))
    save_feature_config(str(run_dir))
    return run_dir, key, tmp_path


def _run(module, monkeypatch, key, argv):
    monkeypatch.setattr(module, "load_malicious_windows",
                        __import__("src.preprocessing.labeling", fromlist=["x"]).load_malicious_windows)
    monkeypatch.setattr(module, "load_answer_key_events", lambda *a, **k: key)
    monkeypatch.setattr(sys, "argv", ["final_evaluation.py"] + argv)
    module.main()


def test_end_to_end_on_the_dev_partition_writes_both_labelings(fake_project, monkeypatch):
    run_dir, key, tmp_path = fake_project
    module = _load_script()
    out_dir = tmp_path / "out"
    _run(module, monkeypatch, key, [
        "--runs", str(run_dir), "--out-dir", str(out_dir), "--tag", "t", "--n-boot", "40",
        "--models", "single_feature_http", "session_length_only",
    ])

    with open(out_dir / "final_evaluation_dev_t.json") as f:
        results = json.load(f)
    assert results["partition"] == "dev"
    assert set(results["labelings"]) == {"window", "answer_key"}
    window = results["labelings"]["window"]
    assert set(window["models"]) == {"single_feature_http", "session_length_only"}
    # the synthetic http feature really does separate the window labels
    assert window["models"]["single_feature_http"]["auroc"]["mean"] > 0.8
    assert window["reference"] == "single_feature_http"
    assert "auroc_minus_reference" in window["models"]["session_length_only"]
    assert results["n_bootstrap_resamples"] == 40

    markdown = (out_dir / "final_evaluation_dev_t.md").read_text()
    assert "Labelling: window" in markdown and "Labelling: answer_key" in markdown
    assert "DEVELOPMENT-partition numbers" in markdown


def test_dev_partition_excludes_the_locked_users(fake_project, monkeypatch):
    run_dir, key, tmp_path = fake_project
    module = _load_script()
    _run(module, monkeypatch, key, [
        "--runs", str(run_dir), "--out-dir", str(tmp_path / "out"), "--n-boot", "10",
        "--models", "single_feature_http",
    ])
    with open(tmp_path / "out" / "final_evaluation_dev_all.json") as f:
        n_dev_users = json.load(f)["n_test_users"]
    events = _synthetic_events()
    assert n_dev_users == events[events["test_partition"] == "dev"]["user"].nunique()
    assert n_dev_users < events["user"].nunique()


def test_locked_partition_is_refused_without_confirmation(fake_project, monkeypatch):
    run_dir, key, tmp_path = fake_project
    module = _load_script()
    with pytest.raises(SystemExit, match="--confirm-final"):
        _run(module, monkeypatch, key, [
            "--runs", str(run_dir), "--partition", "locked", "--out-dir", str(tmp_path / "out"),
            "--models", "single_feature_http",
        ])
    assert not (tmp_path / "out").exists()


def test_locked_partition_runs_with_confirmation_and_is_not_labelled_as_development(fake_project, monkeypatch):
    run_dir, key, tmp_path = fake_project
    module = _load_script()
    _run(module, monkeypatch, key, [
        "--runs", str(run_dir), "--partition", "locked", "--confirm-final", "--out-dir", str(tmp_path / "out"),
        "--n-boot", "10", "--models", "single_feature_http", "session_length_only",
    ])
    markdown = (tmp_path / "out" / "final_evaluation_locked_all.md").read_text()
    assert "DEVELOPMENT-partition numbers" not in markdown
    with open(tmp_path / "out" / "final_evaluation_locked_all.json") as f:
        results = json.load(f)
    assert results["partition"] == "locked"
    assert results["labelings"]["window"]["n_positive"] > 0  # the locked insiders really were evaluated
    events = _synthetic_events()
    locked_users = events[events["test_partition"] == "locked"]["user"].nunique()
    assert results["n_test_users"] == locked_users


def test_runs_with_different_feature_configurations_are_rejected(fake_project, monkeypatch, tmp_path):
    run_dir, key, base = fake_project
    other = base / "run_other"
    other.mkdir()
    with open(other / "feature_config.json", "w") as f:
        json.dump({"enabled_groups": ["file"], "numeric_columns": ["seconds_since_last_event"]}, f)
    module = _load_script()
    with pytest.raises(SystemExit, match="different feature configurations"):
        _run(module, monkeypatch, key, ["--runs", str(run_dir), str(other), "--out-dir", str(base / "out")])


def test_missing_run_folder_is_reported(fake_project, monkeypatch):
    run_dir, key, tmp_path = fake_project
    module = _load_script()
    with pytest.raises(SystemExit, match="Run folder not found"):
        _run(module, monkeypatch, key, ["--runs", str(tmp_path / "nope")])
