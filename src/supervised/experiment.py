"""
The supervised experiment, as one function: out-of-fold scores for every
(model, signal-group configuration), the two no-training reference
heuristics, and the same user-level bootstrap report the unsupervised
evaluation uses.

Protocol (state all of this in the write-up):
  * unit: the user-day (see src/supervised/dataset.py)
  * training label: the WINDOW labelling. The answer key is NOT used for
    training; the answer-key labelling is a second, evaluation-only view.
  * cross-validation splits USERS (stratified by scenario); every score is
    out-of-fold; fixed hyperparameters, no tuning
  * scores are pooled over folds and evaluated with the user-level bootstrap
  * the HTTP CONTRIBUTION is a paired test: the same model with and without
    the http signal, on the same resamples
  * the reference heuristics are selected on dev data, so they are optimistic
Pure given a day table: no file access, testable on synthetic data.
"""

import numpy as np

from src.common.evaluation.bootstrap import make_user_resamples
from src.common.evaluation.final_report import evaluate_labeling, format_markdown
from src.supervised.cross_validation import repeated_out_of_fold_scores
from src.supervised.dataset import WINDOW_LABEL, day_feature_columns, feature_matrix
from src.supervised.models import DISPLAY_NAMES, MODEL_NAMES

WINDOW = "window"
ANSWER_KEY = "answer_key"
ANSWER_KEY_COLUMN = "answer_key_label"

# signal-group configurations for the ablation, in the order they are added
GROUP_CONFIGS = {
    "file": ("file",),
    "file+device": ("file", "device"),
    "file+device+logon": ("file", "device", "logon"),
    "all": ("file", "device", "logon", "http"),
}
NO_HTTP_CONFIG = "file+device+logon"
ALL_CONFIG = "all"

HEURISTICS = {
    "single_feature_http": ("http_offbaseline_share_7d__mean", "Heuristic: day mean of http_offbaseline_share_7d"),
    "day_event_count": ("day_n_events", "Heuristic: events in the day (more = more anomalous)"),
}
DEFAULT_REFERENCE = "single_feature_http"

LABELING_DESCRIPTIONS = {
    WINDOW: (
        "A user-day is malicious if any of its file events falls inside the user's insider time "
        "window. This is the TRAINING label of the supervised models."
    ),
    ANSWER_KEY: (
        "A user-day is malicious if it has a file event on a day with an answer-key http event "
        "for that insider, or during an answer-key removable-device session. Evaluation only: the "
        "supervised models were NOT trained on it. It is built from the same kind of events the "
        "http and device features measure, so it favours those features by construction."
    ),
}


def model_key(model_name, config_name):
    return f"{model_name}[{config_name}]"


def run_experiment(
    day_table,
    scenario_by_user,
    model_names=MODEL_NAMES,
    config_names=tuple(GROUP_CONFIGS),
    n_folds=5,
    n_repeats=3,
    n_boot=500,
    seed=0,
    reference=DEFAULT_REFERENCE,
):
    """
    day_table: build_user_day_table output plus `answer_key_label`.
    Returns (results_dict, markdown_text).
    """
    for name in model_names:
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {name!r}")
    for name in config_names:
        if name not in GROUP_CONFIGS:
            raise ValueError(f"Unknown group configuration {name!r}; choose from {list(GROUP_CONFIGS)}")

    users = day_table["user"].to_numpy()
    scenario_of_day = np.array([int(scenario_by_user.get(u, 0)) for u in users])
    insider_mask = np.array([u in scenario_by_user for u in users])
    y_train = day_table[WINDOW_LABEL].to_numpy().astype(int)

    display_names = {}
    score_matrices = {}
    for name, (column, display) in HEURISTICS.items():
        score_matrices[name] = day_table[column].to_numpy(dtype=float)[None, :]
        display_names[name] = display

    for model in model_names:
        for config in config_names:
            columns = day_feature_columns(GROUP_CONFIGS[config])
            x = feature_matrix(day_table, columns)
            key = model_key(model, config)
            score_matrices[key] = repeated_out_of_fold_scores(
                model, users, x, y_train, scenario_by_user,
                n_folds=n_folds, n_repeats=n_repeats, seed=seed,
            )
            display_names[key] = f"{DISPLAY_NAMES[model]}, signals: {config}"

    labelings = {WINDOW: y_train}
    if ANSWER_KEY_COLUMN in day_table.columns:
        labelings[ANSWER_KEY] = day_table[ANSWER_KEY_COLUMN].to_numpy().astype(int)

    resamples = make_user_resamples(users, n_boot=n_boot, seed=0)
    ref = reference if reference in score_matrices else None

    results = {
        "unit": "user-day",
        "n_rows": int(len(day_table)),
        "n_users": int(len(np.unique(users))),
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "n_bootstrap_resamples": n_boot,
        "training_label": WINDOW,
        "group_configs": {k: list(v) for k, v in GROUP_CONFIGS.items() if k in config_names},
        "labelings": {},
        "http_contribution": {},
    }
    markdown = []
    for name, labels in labelings.items():
        if labels.sum() == 0:
            continue
        results["labelings"][name] = evaluate_labeling(
            labels, scenario_of_day, insider_mask, score_matrices, resamples, reference=ref
        )
        markdown.append(format_markdown(name, LABELING_DESCRIPTIONS[name], results["labelings"][name],
                                        display_names, unit="user-days"))

        # paired: the same model with the http signal vs without it
        contribution = {}
        for model in model_names:
            if NO_HTTP_CONFIG in config_names and ALL_CONFIG in config_names:
                without, with_http = model_key(model, NO_HTTP_CONFIG), model_key(model, ALL_CONFIG)
                paired = evaluate_labeling(
                    labels, scenario_of_day, insider_mask,
                    {without: score_matrices[without], with_http: score_matrices[with_http]},
                    resamples, reference=without,
                )
                contribution[model] = paired["models"][with_http]["auroc_minus_reference"]
        results["http_contribution"][name] = contribution
        if contribution:
            lines = [f"#### http contribution under the {name} labelling "
                     f"(AUROC with http minus AUROC without, paired, 95% user-level interval)", ""]
            for model, d in contribution.items():
                mark = " *" if d["ci_excludes_zero"] else ""
                lines.append(f"- {DISPLAY_NAMES[model]}: {d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{mark}")
            lines += ["", "`*` = the interval excludes 0.", ""]
            markdown.append("\n".join(lines))

    markdown.append(
        "_Development-partition numbers (the 50 dev users), cross-validated by user. Features and the "
        "http heuristic were chosen while looking at this partition, so the heuristics are optimistic. "
        "The locked partition has NOT been used for the supervised models._\n"
    )
    return results, "\n".join(markdown)
