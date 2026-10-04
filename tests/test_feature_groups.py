import pandas as pd
import pytest

from src.preprocessing.feature_groups import (
    FEATURE_GROUPS_ENV_VAR,
    NUMERIC_FEATURE_REGISTRY,
    BINARY_NUMERIC_COLUMNS,
    PRESENCE_COLUMN,
    available_groups,
    resolve_groups,
    get_numeric_feature_columns,
    get_scaled_columns,
    get_binary_columns,
    feature_group_of,
    require_columns,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Every test starts from the default (no environment override).
    monkeypatch.delenv(FEATURE_GROUPS_ENV_VAR, raising=False)


def test_default_column_order_matches_the_order_checkpoints_were_trained_with():
    # The registry is append-only: this order is the model input vector
    # order, so changing it silently invalidates every saved checkpoint
    # (same shapes, different meaning). If this test fails because you
    # reordered the registry, retrain everything and update this list on
    # purpose -- do not just edit the test to make it pass.
    assert get_numeric_feature_columns() == [
        "seconds_since_last_event",
        "seconds_since_device_connect",
        "copies_since_device_connect",
        "outside_working_hours",
        "user_session_volume_deviation",
        "user_session_frequency_deviation",
    ]


def test_registry_has_no_duplicate_columns():
    columns = [column for column, _ in NUMERIC_FEATURE_REGISTRY]
    assert len(columns) == len(set(columns))


def test_every_binary_column_is_registered():
    registered = {column for column, _ in NUMERIC_FEATURE_REGISTRY}
    assert set(BINARY_NUMERIC_COLUMNS) <= registered


def test_presence_column_is_registered_in_a_mandatory_group():
    assert PRESENCE_COLUMN in get_numeric_feature_columns(["file"])


def test_available_groups_in_order_of_first_appearance():
    assert available_groups() == ("file", "device", "logon")


def test_selecting_a_subset_keeps_registry_order_and_drops_other_groups():
    columns = get_numeric_feature_columns(["file", "device"])
    assert columns == [
        "seconds_since_last_event",
        "seconds_since_device_connect",
        "copies_since_device_connect",
        "user_session_volume_deviation",
        "user_session_frequency_deviation",
    ]
    assert "outside_working_hours" not in columns


def test_group_order_in_the_request_does_not_change_column_order():
    assert get_numeric_feature_columns(["logon", "file", "device"]) == get_numeric_feature_columns()


def test_file_group_cannot_be_disabled():
    with pytest.raises(ValueError, match="Mandatory"):
        resolve_groups(["device", "logon"])


def test_unknown_group_raises_instead_of_being_silently_ignored():
    with pytest.raises(ValueError, match="Unknown feature group"):
        resolve_groups(["file", "htpp"])  # typo


def test_environment_variable_selects_groups(monkeypatch):
    monkeypatch.setenv(FEATURE_GROUPS_ENV_VAR, "file, device")
    assert resolve_groups() == ("file", "device")


def test_empty_environment_variable_means_all_groups(monkeypatch):
    monkeypatch.setenv(FEATURE_GROUPS_ENV_VAR, "   ")
    assert resolve_groups() == available_groups()


def test_scaled_and_binary_columns_partition_the_numeric_columns():
    scaled = get_scaled_columns()
    binary = get_binary_columns()
    assert set(scaled).isdisjoint(binary)
    assert set(scaled) | set(binary) == set(get_numeric_feature_columns())
    assert "outside_working_hours" in binary


def test_binary_column_disappears_when_its_group_is_disabled():
    assert get_binary_columns(["file", "device"]) == []


def test_feature_group_of():
    assert feature_group_of("copies_since_device_connect") == "device"
    assert feature_group_of("outside_working_hours") == "logon"
    with pytest.raises(KeyError):
        feature_group_of("not_a_feature")


def test_require_columns_passes_when_all_present():
    df = pd.DataFrame({column: [0.0] for column in get_numeric_feature_columns()})
    require_columns(df)  # should not raise


def test_require_columns_names_the_missing_feature_and_its_group():
    columns = [c for c in get_numeric_feature_columns() if c != "outside_working_hours"]
    df = pd.DataFrame({column: [0.0] for column in columns})
    with pytest.raises(ValueError, match="outside_working_hours.*logon|logon.*outside_working_hours"):
        require_columns(df)


def test_require_columns_only_checks_enabled_groups():
    columns = [c for c in get_numeric_feature_columns() if c != "outside_working_hours"]
    df = pd.DataFrame({column: [0.0] for column in columns})
    require_columns(df, groups=["file", "device"])  # logon not needed -> no error
