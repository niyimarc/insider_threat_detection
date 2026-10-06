import pytest

from src.models.registry import (
    MODELS_ENV_VAR,
    ModelSpec,
    get_model_specs,
    get_spec,
    register,
    registered_names,
    _REGISTRY,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(MODELS_ENV_VAR, raising=False)


def test_default_models_in_reporting_order():
    assert registered_names() == [
        "transformer_ae", "lstm_ae", "static_iforest", "single_feature_http", "session_length_only",
    ]


def test_every_registered_model_has_a_complete_spec():
    for spec in get_model_specs():
        if spec.kind == "torch":
            assert spec.build and spec.loss_fn and spec.score_fn and spec.checkpoint_file
        elif spec.kind == "static":
            assert spec.build_static and spec.checkpoint_file
        else:
            assert spec.kind == "heuristic" and not spec.is_trainable


def test_checkpoint_files_are_unique():
    files = [s.checkpoint_file for s in get_model_specs() if s.checkpoint_file]
    assert len(files) == len(set(files))


def test_checkpoint_file_names_match_what_earlier_runs_saved():
    # Existing run folders use these exact names; renaming would orphan them.
    assert get_spec("transformer_ae").checkpoint_file == "transformer_autoencoder.pt"
    assert get_spec("lstm_ae").checkpoint_file == "lstm_autoencoder.pt"
    assert get_spec("static_iforest").checkpoint_file == "static_vector_detector.joblib"


def test_environment_variable_selects_a_subset_and_keeps_registry_order(monkeypatch):
    monkeypatch.setenv(MODELS_ENV_VAR, "static_iforest, transformer_ae")
    assert [s.name for s in get_model_specs()] == ["transformer_ae", "static_iforest"]


def test_unknown_model_name_raises_instead_of_being_dropped(monkeypatch):
    monkeypatch.setenv(MODELS_ENV_VAR, "transformer_ae,not_a_model")
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_specs()
    with pytest.raises(ValueError):
        get_spec("nope")


def test_explicit_names_override_the_environment(monkeypatch):
    monkeypatch.setenv(MODELS_ENV_VAR, "lstm_ae")
    assert [s.name for s in get_model_specs(["static_iforest"])] == ["static_iforest"]


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register(ModelSpec(name="lstm_ae", display_name="x", kind="heuristic"))


def test_incomplete_specs_are_rejected():
    with pytest.raises(ValueError):
        ModelSpec(name="bad_torch", display_name="x", kind="torch")        # no build/loss/score
    with pytest.raises(ValueError):
        ModelSpec(name="bad_static", display_name="x", kind="static")      # no builder
    with pytest.raises(ValueError):
        ModelSpec(name="bad_kind", display_name="x", kind="quantum")
    with pytest.raises(ValueError):
        ModelSpec(name="bad_reduce", display_name="x", kind="heuristic", heuristic_reduce="median")


def test_a_new_model_can_be_registered_and_is_picked_up():
    spec = ModelSpec(name="tmp_test_heuristic", display_name="tmp", kind="heuristic",
                     heuristic_column="seconds_since_last_event")
    register(spec)
    try:
        assert "tmp_test_heuristic" in registered_names()
        assert get_model_specs(["tmp_test_heuristic"])[0] is spec
    finally:
        _REGISTRY.pop("tmp_test_heuristic")


def test_torch_specs_build_real_models():
    import torch.nn as nn
    for name in ("transformer_ae", "lstm_ae"):
        model = get_spec(name).build(7, 9)
        assert isinstance(model, nn.Module)


def test_static_spec_builds_a_detector_with_the_requested_seed():
    detector = get_spec("static_iforest").build_static(123)
    assert detector.model.random_state == 123


def test_the_reference_baseline_is_registered_as_a_heuristic():
    spec = get_spec("single_feature_http")
    assert spec.kind == "heuristic"
    assert spec.heuristic_column == "http_offbaseline_share_7d"
    assert spec.required_columns == ("http_offbaseline_share_7d",)
