import json
import os

import numpy as np
import pytest

from src.common.run_config import (
    CHECKPOINT_DIR_ENV_VAR,
    SEED_ENV_VAR,
    RUN_METADATA_FILENAME,
    get_checkpoint_dir,
    get_seed,
    set_global_seed,
    save_run_metadata,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(CHECKPOINT_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(SEED_ENV_VAR, raising=False)


def test_defaults_are_the_standard_checkpoint_folder_and_seed_42():
    assert os.path.basename(get_checkpoint_dir()) == "checkpoints"
    assert get_seed() == 42


def test_relative_checkpoint_dir_is_resolved_against_the_project_root(monkeypatch):
    monkeypatch.setenv(CHECKPOINT_DIR_ENV_VAR, "checkpoints/all_seed1")
    path = get_checkpoint_dir()
    assert os.path.isabs(path)
    assert path.replace("\\", "/").endswith("checkpoints/all_seed1")


def test_absolute_checkpoint_dir_is_used_as_is(monkeypatch, tmp_path):
    monkeypatch.setenv(CHECKPOINT_DIR_ENV_VAR, str(tmp_path))
    assert get_checkpoint_dir() == str(tmp_path)


def test_seed_from_environment(monkeypatch):
    monkeypatch.setenv(SEED_ENV_VAR, "7")
    assert get_seed() == 7


def test_non_integer_seed_raises(monkeypatch):
    monkeypatch.setenv(SEED_ENV_VAR, "seven")
    with pytest.raises(ValueError):
        get_seed()


def test_set_global_seed_makes_numpy_and_torch_reproducible():
    import torch
    set_global_seed(123)
    a = (np.random.rand(3), torch.rand(3))
    set_global_seed(123)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_save_run_metadata_records_seed_groups_and_columns(tmp_path):
    path = save_run_metadata(str(tmp_path), seed=5, device="cpu")
    assert os.path.basename(path) == RUN_METADATA_FILENAME
    with open(path) as f:
        meta = json.load(f)
    assert meta["seed"] == 5 and meta["device"] == "cpu"
    assert "file" in meta["enabled_groups"]
    assert meta["numeric_columns"]
    assert "git_commit" in meta and "torch" in meta["versions"]
