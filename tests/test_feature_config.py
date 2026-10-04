import json
import pytest

from src.preprocessing.feature_groups import FEATURE_GROUPS_ENV_VAR, get_numeric_feature_columns
from src.training.feature_config import (
    FEATURE_CONFIG_FILENAME,
    current_feature_config,
    save_feature_config,
    verify_feature_config,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(FEATURE_GROUPS_ENV_VAR, raising=False)


def test_saved_config_records_groups_and_columns(tmp_path):
    path = save_feature_config(str(tmp_path))

    with open(path) as f:
        saved = json.load(f)

    assert saved["enabled_groups"] == ["file", "device", "logon"]
    assert saved["numeric_columns"] == get_numeric_feature_columns()


def test_verify_passes_when_configuration_is_unchanged(tmp_path):
    save_feature_config(str(tmp_path))
    verify_feature_config(str(tmp_path))  # should not raise


def test_verify_raises_when_groups_changed_since_training(tmp_path, monkeypatch):
    save_feature_config(str(tmp_path))  # trained with every group

    monkeypatch.setenv(FEATURE_GROUPS_ENV_VAR, "file,device")  # evaluated with fewer

    with pytest.raises(RuntimeError, match="Feature configuration mismatch"):
        verify_feature_config(str(tmp_path))


def test_verify_raises_when_column_order_changed_even_if_the_set_is_identical(tmp_path):
    # Two columns swapped has identical tensor shapes, so loading the
    # weights would NOT fail on its own -- this is the dangerous case.
    columns = get_numeric_feature_columns()
    swapped = [columns[1], columns[0]] + columns[2:]
    with open(tmp_path / FEATURE_CONFIG_FILENAME, "w") as f:
        json.dump({"enabled_groups": ["file", "device", "logon"], "numeric_columns": swapped}, f)

    with pytest.raises(RuntimeError, match="Feature configuration mismatch"):
        verify_feature_config(str(tmp_path))


def test_verify_warns_but_does_not_fail_for_checkpoints_without_a_config(tmp_path):
    with pytest.warns(UserWarning, match="cannot verify"):
        verify_feature_config(str(tmp_path))


def test_current_feature_config_follows_the_environment(monkeypatch):
    monkeypatch.setenv(FEATURE_GROUPS_ENV_VAR, "file")
    config = current_feature_config()
    assert config["enabled_groups"] == ["file"]
    assert "outside_working_hours" not in config["numeric_columns"]
