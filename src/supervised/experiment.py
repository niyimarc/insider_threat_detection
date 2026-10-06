"""
The supervised experiment, as one function: out-of-fold scores for every
(model, signal-group configuration), the no-training reference
heuristics, the confound checks, and the same user-level bootstrap report
the unsupervised evaluation uses.

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

CONFOUND CHECKS (so a good score cannot be mistaken for detection when it
is something else):
  * CALENDAR-ONLY CONTROL: the same models given only the date (day index
    and weekday). If that scores well, the window labels are recoverable
    from WHEN a day happened (insider windows sit in particular stretches
    of the dataset), and the real scores cannot be read as behaviour.
  * WITHIN-INSIDER COMPARISON: every model scored on insiders' days only
    (window days vs the same insiders' other days). This removes
    "insiders differ from benign users" as an explanation, and is
    reported next to the calendar-only control, because the time
    explanation remains possible there.
  * PERMUTATION IMPORTANCE BY SIGNAL GROUP: how much each group's columns
    matter to the all-signals model on held-out users.
Pure given a day table: no file access, testable on synthetic data.
"""

import numpy as np

from src.common.evaluation.bootstrap import (
    interval,
    make_user_resamples,
    metric_over_resamples,
    paired_difference,
    safe_auroc,
)
from src.common.evaluation.final_report import evaluate_labeling, format_markdown
from src.supervised.cross_validation import (
    grouped_permutation_scores,
    make_user_folds,
    repeated_out_of_fold_scores,
)
from src.supervised.dataset import (
    CALENDAR_COLUMNS,
    WINDOW_LABEL,
    add_calendar_controls,
    day_feature_columns,
    feature_matrix,
    group_of_day_column,
)
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
CALENDAR_CONTROL = "calendar_only"

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
WITHIN_INSIDER_DESCRIPTION = (
    "Only the INSIDERS' user-days are scored: positives are an insider's days inside their window, "
    "negatives are the SAME insiders' other days. Benign users are excluded, so 'insiders differ "
    "from benign users' cannot explain a score. The calendar-only control is the reference: a model "
    "that does not beat it is not showing more than WHEN the window days happened."
)


def model_key(model_name, config_name):
    return f"{model_name}[{config_name}]"


def _column_indices_by_group(columns):
    by_group = {}
    for i, column in enumerate(columns):
        by_group.setdefault(group_of_day_column(column), []).append(i)
    return by_group


def _permutation_importance(table, scenario_by_user, model_names, labels, resamples, n_folds, seed, n_permutations):
    """{model: {group: {"auroc_drop": ..., ci...}}} for the all-signals configuration."""
    columns = day_feature_columns(GROUP_CONFIGS[ALL_CONFIG])
    x = feature_matrix(table, columns)
    group_to_columns = _column_indices_by_group(columns)
    users = table["user"].to_numpy()
    folds = make_user_folds(users, scenario_by_user, n_folds=n_folds, seed=seed)
    importance = {}
    for model in model_names:
        base, permuted = grouped_permutation_scores(
            model, users, x, labels, folds, group_to_columns, seed=seed, n_permutations=n_permutations,
        )
        base_values = metric_over_resamples(labels, base, resamples, safe_auroc)
        per_group = {}
        for group, matrix in permuted.items():
            drop = paired_difference(base_values, metric_over_resamples(labels, matrix, resamples, safe_auroc))
            low, high = interval(drop)
            per_group[group] = {
                "auroc_drop": float(np.nanmean(drop)) if np.isfinite(drop).any() else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        importance[model] = {
            "base_auroc": float(safe_auroc(labels, base)),
            "groups": per_group,
        }
    return importance


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
    n_permutations=5,
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

    day_table = add_calendar_controls(day_table)
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
            key = model_key(model, config)
            score_matrices[key] = repeated_out_of_fold_scores(
                model, users, feature_matrix(day_table, columns), y_train, scenario_by_user,
                n_folds=n_folds, n_repeats=n_repeats, seed=seed,
            )
            display_names[key] = f"{DISPLAY_NAMES[model]}, signals: {config}"
        # the control: the SAME model, given only the date
        key = model_key(model, CALENDAR_CONTROL)
        score_matrices[key] = repeated_out_of_fold_scores(
            model, users, feature_matrix(day_table, CALENDAR_COLUMNS), y_train, scenario_by_user,
            n_folds=n_folds, n_repeats=n_repeats, seed=seed,
        )
        display_names[key] = f"CONTROL: {DISPLAY_NAMES[model]}, calendar only (date and weekday)"

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
        "within_insider": None,
        "permutation_importance": {},
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

    # ---- confound check 1+2: calendar-only control, and the within-insider comparison ----
    markdown.append("## Confound checks\n")
    control_models = [m for m in model_names]
    control_lines = ["#### Calendar-only control (window labelling)", ""]
    window_models = results["labelings"][WINDOW]["models"]
    for model in control_models:
        auroc = window_models[model_key(model, CALENDAR_CONTROL)]["auroc"]
        control_lines.append(
            f"- {DISPLAY_NAMES[model]} given only the date: AUROC {auroc['mean']:.3f} "
            f"[{auroc['ci_low']:.3f}, {auroc['ci_high']:.3f}]"
        )
    control_lines += [
        "",
        "Around 0.5 means the date alone does not reveal the window labels. A clearly higher value means "
        "the labels can be recovered from WHEN a day happened, and the behavioural scores above must be "
        "read against it.",
        "",
    ]
    markdown.append("\n".join(control_lines))

    in_labels = y_train[insider_mask]
    if insider_mask.sum() and 0 < in_labels.sum() < len(in_labels):
        in_resamples = make_user_resamples(users[insider_mask], n_boot=n_boot, seed=0)
        in_matrices = {k: np.atleast_2d(m)[:, insider_mask] for k, m in score_matrices.items()}
        control_ref = model_key(control_models[0], CALENDAR_CONTROL) if control_models else None
        results["within_insider"] = evaluate_labeling(
            in_labels, scenario_of_day[insider_mask], np.ones(int(insider_mask.sum()), dtype=bool),
            in_matrices, in_resamples, reference=control_ref, per_scenario=False,
        )
        markdown.append(format_markdown("within-insider (window labels)", WITHIN_INSIDER_DESCRIPTION,
                                        results["within_insider"], display_names, unit="insider user-days"))

    # ---- confound check 3: permutation importance by signal group ----
    if ALL_CONFIG in config_names:
        importance = _permutation_importance(
            day_table, scenario_by_user, model_names, y_train, resamples, n_folds, seed, n_permutations,
        )
        results["permutation_importance"] = importance
        lines = ["#### Permutation importance by signal group "
                 "(all-signals model, held-out users, window labelling)", "",
                 f"AUROC lost when one group's columns are shuffled across the held-out days "
                 f"({n_permutations} shuffles, paired 95% user-level interval). Larger = the model relies on "
                 "that group more.", ""]
        for model, entry in importance.items():
            lines.append(f"{DISPLAY_NAMES[model]} (AUROC with nothing shuffled: {entry['base_auroc']:.3f})")
            lines.append("")
            for group, d in entry["groups"].items():
                mark = " *" if d["ci_excludes_zero"] else ""
                lines.append(f"- {group}: {d['auroc_drop']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{mark}")
            lines.append("")
        lines += ["`*` = the interval excludes 0. Groups that share information can hide each other "
                  "(shuffling one is partly compensated by another), so small drops do not prove a group is useless.", ""]
        markdown.append("\n".join(lines))

    markdown.append(
        "_Development-partition numbers (the 50 dev users), cross-validated by user. Features and the "
        "http heuristic were chosen while looking at this partition, so the heuristics are optimistic. "
        "The locked partition has NOT been used for the supervised models._\n"
    )
    return results, "\n".join(markdown)
