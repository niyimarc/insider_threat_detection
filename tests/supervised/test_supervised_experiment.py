import numpy as np
import pandas as pd
import pytest

from src.common.preprocessing.feature_groups import available_groups
from src.supervised.dataset import WINDOW_LABEL, day_feature_columns
from src.supervised.experiment import GROUP_CONFIGS, HEURISTICS, model_key, run_experiment


def _synthetic_day_table(seed=0, window_start=20):
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
            malicious = int(insider and d >= window_start)
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


# ---------------- confound checks ----------------

def test_calendar_only_control_exposes_a_time_confound(experiment):
    # In the synthetic data every insider's window is the LATER half of the period, so the date alone
    # reveals the label for insiders: the control must score clearly above chance.
    results, markdown = experiment
    models = results["labelings"]["window"]["models"]
    for model in ("logistic", "gradient_boosting"):
        assert model_key(model, "calendar_only") in models
    assert models[model_key("gradient_boosting", "calendar_only")]["auroc"]["mean"] > 0.7
    assert "Calendar-only control" in markdown and "CONTROL" in markdown


def test_within_insider_comparison_uses_only_insiders_and_is_paired_against_the_control(experiment):
    results, markdown = experiment
    within = results["within_insider"]
    assert within is not None
    assert within["n_sessions"] == 12 * 40                       # the 12 insiders' days only
    assert within["reference"] == model_key("logistic", "calendar_only")
    entry = within["models"][model_key("gradient_boosting", "all")]
    assert entry["auroc"]["mean"] > 0.95
    assert "auroc_minus_reference" in entry
    assert "within-insider" in markdown


def test_calendar_columns_are_never_inputs_of_the_real_configurations():
    from src.supervised.dataset import CALENDAR_COLUMNS
    for config, groups in GROUP_CONFIGS.items():
        assert not set(CALENDAR_COLUMNS) & set(day_feature_columns(groups))


def test_permutation_importance_points_at_the_planted_signal_group(experiment):
    results, markdown = experiment
    entry = results["permutation_importance"]["gradient_boosting"]
    groups = entry["groups"]
    assert set(groups) == {"file", "device", "logon", "http"}
    assert groups["http"]["auroc_drop"] > 0.3 and groups["http"]["ci_excludes_zero"] is True
    for other in ("file", "device", "logon"):
        assert groups[other]["auroc_drop"] < 0.1
    assert "Permutation importance by signal group" in markdown


def test_importance_is_skipped_when_the_all_signals_configuration_is_not_requested():
    table, scenarios = _synthetic_day_table()
    results, markdown = run_experiment(
        table, scenarios, model_names=("logistic",), config_names=("file",),
        n_folds=4, n_repeats=1, n_boot=20,
    )
    assert results["permutation_importance"] == {}
    assert "Permutation importance by signal group" not in markdown


# ---------------- date-matched evaluation and training ----------------

@pytest.fixture(scope="module")
def date_experiment():
    # insider windows start exactly on a week boundary (day 21), so the weekly strata are clean:
    # weeks 3, 4 and 5 contain both classes, weeks 0-2 contain no malicious day
    table, scenarios = _synthetic_day_table(window_start=21)
    results, markdown = run_experiment(
        table, scenarios, config_names=("file+device+logon", "all"),
        n_folds=4, n_repeats=2, n_boot=60, seed=0,
    )
    return results, markdown


def test_date_matched_results_are_reported_with_their_coverage(date_experiment):
    results, markdown = date_experiment
    section = results["date_matched"]
    assert section["strata_days"] == 7
    cov = section["trained_normally"]["coverage"]
    assert cov["n_strata_used"] == 3 and cov["n_positive_used"] == cov["n_positive"]
    assert "Date-matched evaluation" in markdown
    assert "Trained normally, evaluated date-stratified" in markdown
    assert "Trained date-matched, evaluated date-stratified" in markdown


def test_the_calendar_only_control_falls_to_chance_once_the_date_is_removed(date_experiment):
    results, _ = date_experiment
    # sanity check of the whole correction: with the date removed, a model given only the date has nothing left
    for table in ("trained_normally", "trained_date_matched"):
        models = results["date_matched"][table]["models"]
        for model in ("logistic", "gradient_boosting"):
            value = models[model_key(model, "calendar_only")]["auroc"]["mean"]
            assert 0.35 < value < 0.65, (table, model, value)


def test_a_real_behavioural_signal_survives_date_matching(date_experiment):
    results, _ = date_experiment
    matched = results["date_matched"]["trained_date_matched"]["models"]
    assert matched[model_key("gradient_boosting", "all")]["auroc"]["mean"] > 0.9
    assert matched["single_feature_http"]["auroc"]["mean"] > 0.9
    # without the planted http signal there is nothing left to find
    assert matched[model_key("gradient_boosting", "file+device+logon")]["auroc"]["mean"] < 0.65


def test_date_matched_http_contribution_is_reported_and_positive(date_experiment):
    results, markdown = date_experiment
    d = results["date_matched"]["http_contribution"]["gradient_boosting"]
    assert d["mean"] > 0.2 and d["ci_excludes_zero"] is True
    assert "http contribution, date-matched" in markdown


def test_strata_width_is_configurable():
    table, scenarios = _synthetic_day_table(window_start=21)
    results, _ = run_experiment(
        table, scenarios, model_names=("logistic",), config_names=("all",),
        n_folds=4, n_repeats=1, n_boot=20, strata_days=14,
    )
    assert results["date_matched"]["strata_days"] == 14
