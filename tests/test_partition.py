import pandas as pd
import pytest

from src.evaluation.partition import select_test_df, get_partition_name, PARTITION_ENV_VAR


def _events():
    return pd.DataFrame({
        "user": ["a", "b", "c", "d"],
        "split": ["train", "test", "test", "test"],
        "test_partition": ["", "dev", "locked", "dev"],
    })


def test_default_partition_is_dev(monkeypatch):
    monkeypatch.delenv(PARTITION_ENV_VAR, raising=False)
    assert get_partition_name() == "dev"
    assert set(select_test_df(_events())["user"]) == {"b", "d"}


def test_locked_requires_explicit_request(monkeypatch):
    monkeypatch.setenv(PARTITION_ENV_VAR, "locked")
    assert set(select_test_df(_events())["user"]) == {"c"}


def test_all_returns_every_test_user(monkeypatch):
    monkeypatch.setenv(PARTITION_ENV_VAR, "all")
    assert set(select_test_df(_events())["user"]) == {"b", "c", "d"}


def test_train_rows_never_selected(monkeypatch):
    for name in ("dev", "locked", "all"):
        monkeypatch.setenv(PARTITION_ENV_VAR, name)
        assert "a" not in set(select_test_df(_events())["user"])


def test_invalid_partition_name_raises(monkeypatch):
    monkeypatch.setenv(PARTITION_ENV_VAR, "bogus")
    with pytest.raises(ValueError):
        get_partition_name()


def test_old_dataset_without_partition_column_fails_clearly(monkeypatch):
    monkeypatch.delenv(PARTITION_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="build_dataset"):
        select_test_df(_events().drop(columns=["test_partition"]))
