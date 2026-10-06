import numpy as np
import pandas as pd
import pytest

from src.common.preprocessing.feature_groups import available_groups
from src.supervised.dataset import WINDOW_LABEL, day_feature_columns
from src.supervised.experiment import GROUP_CONFIGS, HEURISTICS, model_key, run_experiment


def _synthetic_day_table(seed=0):
    """
    24 users (12 insiders, 12 benign), 40 days each. On an insider's malicious days the http
    feature is high; every other column is noise. The answer-key label is a subset of the
    window label.
    """
    rng = np.random.default_rng(seed)
    columns = day_feature_columns(available_groups())
    rows, scenario_by_user = [], {}
    for u in range(24):
        user = f"U{u:02d}"
        insider = u < 12
        if insider:
            scenario_by_user[user] = 1 + u % 3
        for d in range(40):
            malicious = int(insider and d >= 20)
            row = {c: float(rng.normal()) for c in columns}
            row["day_n_events"] = float(rng.integers(1, 6))
            row["day_n_sessions"] = 1.0
            if malicious:
                row["http_offbaseline_share_7d__mean"] += 3.0
                row["http_new_domains_7d__mean"] += 3.0
            row.update({
                "user": user, "day": pd.Timestamp("2010-06-01") + pd.Timedelta(days=d),
                WINDOW_LABEL: malicious,
                "answer_key_label": int(malicious and d % 2 == 0),
            })
            rows.append(row)
    return pd.DataFrame(rows), scenario_by_user


@pytest.fixture(scope="module")
def experiment():
    table, scenarios = _synthetic_day_table()
    results, markdown = run_experiment(
        table, scenarios, config_names=("file+device+logon", "all"),
        n_folds=4, n_repeats=2, n_boot=60, seed=0,
    )
    return results, markdown


def test_results_cover_both_labelings_and_every_model_and_heuristic(experiment):
    results, _ = experiment
    assert results["unit"] == "user-day" and results["training_label"] == "window"
    assert set(results["labelings"]) == {"window", "answer_key"}
    models = results["labelings"]["window"]["models"]
    for heuristic in HEURISTICS:
        assert heuristic in models
    for model in ("logistic", "gradient_boosting"):
        for config in ("file+device+logon", "all"):
            assert model_key(model, config) in models
    assert models[model_key("logistic", "all")]["n_runs"] == 2   # the repeats
    assert models["single_feature_http"]["n_runs"] == 1


def test_planted_http_signal_is_found_when_http_is_enabled_and_not_otherwise(experiment):
    results, _ = experiment
    models = results["labelings"]["window"]["models"]
    with_http = models[model_key("gradient_boosting", "all")]["auroc"]["mean"]
    without_http = models[model_key("gradient_boosting", "file+device+logon")]["auroc"]["mean"]
    assert with_http > 0.9
    assert without_http < 0.7
    assert models["single_feature_http"]["auroc"]["mean"] > 0.9


def test_http_contribution_is_a_paired_positive_difference(experiment):
    results, _ = experiment
    contribution = results["http_contribution"]["window"]["gradient_boosting"]
    assert contribution["mean"] > 0.2
    assert contribution["ci_excludes_zero"] is True


def test_the_report_names_the_unit_the_training_label_and_the_dev_caveat(experiment):
    _, markdown = experiment
    assert "user-days" in markdown
    assert "Labelling: window" in markdown and "Labelling: answer_key" in markdown
    assert "TRAINING label" in markdown
    assert "locked partition has NOT been used" in markdown


def test_results_are_json_serialisable(experiment):
    import json
    results, _ = experiment
    json.dumps(results, default=str)


def test_same_inputs_give_the_same_results():
    table, scenarios = _synthetic_day_table()
    kwargs = dict(model_names=("gradient_boosting",), config_names=("all",), n_folds=4, n_repeats=1, n_boot=20)
    a, _ = run_experiment(table, scenarios, **kwargs)
    b, _ = run_experiment(table, scenarios, **kwargs)
    key = model_key("gradient_boosting", "all")
    assert a["labelings"]["window"]["models"][key]["auroc"] == b["labelings"]["window"]["models"][key]["auroc"]


def test_unknown_names_are_rejected():
    table, scenarios = _synthetic_day_table()
    with pytest.raises(ValueError):
        run_experiment(table, scenarios, model_names=("svm",))
    with pytest.raises(ValueError):
        run_experiment(table, scenarios, config_names=("everything",))
    assert "all" in GROUP_CONFIGS
