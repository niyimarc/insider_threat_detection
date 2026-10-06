import numpy as np

from src.evaluation.bootstrap import make_user_resamples
from src.evaluation.final_report import evaluate_labeling, format_markdown


def _setup(seed=0, n_users=40, per_user=6):
    rng = np.random.default_rng(seed)
    users, labels, scenario = [], [], []
    for u in range(n_users):
        insider = u < 16
        sc = 1 + (u % 3) if insider else 0
        for _ in range(per_user):
            users.append(f"U{u}")
            scenario.append(sc)
            labels.append(int(insider and rng.random() < 0.6))
    users, labels, scenario = np.array(users), np.array(labels), np.array(scenario)
    insider_mask = scenario > 0
    return rng, users, labels, scenario, insider_mask


def test_informative_model_beats_random_model_and_reference_difference_is_reported():
    rng, users, labels, scenario, insider = _setup()
    informative = labels * 1.5 + rng.normal(size=len(labels))
    reference = rng.random(len(labels))
    matrices = {
        "good": np.stack([informative, informative + rng.normal(scale=0.1, size=len(labels))]),
        "ref": reference[None, :],
    }
    resamples = make_user_resamples(users, n_boot=150, seed=0)
    result = evaluate_labeling(labels, scenario, insider, matrices, resamples, reference="ref")

    good = result["models"]["good"]
    assert good["n_runs"] == 2
    assert good["auroc"]["mean"] > 0.75
    assert good["auroc"]["ci_low"] < good["auroc"]["mean"] < good["auroc"]["ci_high"]
    assert good["auroc"]["std_over_runs"] >= 0
    diff = good["auroc_minus_reference"]
    assert diff["mean"] > 0.1 and diff["ci_excludes_zero"] is True
    assert "auroc_minus_reference" not in result["models"]["ref"]  # the reference isn't compared to itself


def test_prevalence_and_positive_counts_are_reported():
    rng, users, labels, scenario, insider = _setup()
    resamples = make_user_resamples(users, n_boot=10, seed=0)
    result = evaluate_labeling(labels, scenario, insider, {"m": rng.random((1, len(labels)))}, resamples)
    assert result["n_sessions"] == len(labels)
    assert result["n_positive"] == int(labels.sum())
    assert abs(result["prevalence"] - labels.mean()) < 1e-12
    assert sum(result["positives_by_scenario"].values()) == int(labels.sum())


def test_per_scenario_auroc_is_none_when_a_scenario_has_no_positives():
    rng, users, labels, scenario, insider = _setup()
    labels = np.where(scenario == 3, 0, labels)  # scenario 3 has no malicious sessions
    resamples = make_user_resamples(users, n_boot=10, seed=0)
    result = evaluate_labeling(labels, scenario, insider, {"m": rng.random((1, len(labels)))}, resamples)
    per_scenario = result["models"]["m"]["per_scenario_auroc"]
    assert per_scenario[3] is None
    assert per_scenario[1] is not None and per_scenario[2] is not None


def test_a_single_run_has_zero_spread():
    rng, users, labels, scenario, insider = _setup()
    resamples = make_user_resamples(users, n_boot=10, seed=0)
    result = evaluate_labeling(labels, scenario, insider, {"m": rng.random((1, len(labels)))}, resamples)
    assert result["models"]["m"]["auroc"]["std_over_runs"] == 0.0


def test_markdown_table_has_a_row_per_model_and_marks_significant_differences():
    rng, users, labels, scenario, insider = _setup()
    matrices = {"good": (labels * 2.0 + rng.normal(size=len(labels)))[None, :], "ref": rng.random((1, len(labels)))}
    resamples = make_user_resamples(users, n_boot=100, seed=0)
    result = evaluate_labeling(labels, scenario, insider, matrices, resamples, reference="ref")
    text = format_markdown("window", "description here", result, {"good": "Good model", "ref": "Reference"})
    assert "### Labelling: window" in text
    assert "| Good model |" in text and "| Reference |" in text
    assert "(reference)" in text
    assert " *" in text  # the good model's difference interval excludes zero
    assert "chance-level AUPRC" in text
