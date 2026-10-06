"""
Turns per-session scores from several models (each possibly from several
training seeds) into the comparison table the thesis reports, under one
labelling at a time:

  * AUROC and AUPRC, as the mean over training seeds, with a 95%
    USER-LEVEL bootstrap interval (src/common/evaluation/bootstrap.py) and the
    spread of the seeds themselves;
  * a PAIRED difference in AUROC against a reference model (default: the
    single-feature heuristic) with an interval, so "does the deep model
    beat one raw feature?" has a statistical answer;
  * per-scenario AUROC (that scenario's malicious sessions vs benign
    users' sessions); scenarios 1 and 3 have very few sessions -- treat
    those columns as anecdotal;
  * the number of positive sessions and the chance-level AUPRC
    (= prevalence), because AUPRC means little without it.

Pure functions on arrays: no file or model access, so it is testable on
synthetic data.
"""

import numpy as np

from src.common.evaluation.bootstrap import (
    METRICS,
    interval,
    metric_over_resamples,
    paired_difference,
    safe_auroc,
)


def _seed_mean_and_spread(labels, score_matrix, metric):
    per_run = [metric(labels, run) for run in score_matrix]
    return float(np.mean(per_run)), float(np.std(per_run)) if len(per_run) > 1 else 0.0


def evaluate_labeling(labels, scenario_of_session, insider_mask, score_matrices, resamples, reference=None):
    """
    labels:               (n_sessions,) 0/1 under this labelling
    scenario_of_session:  (n_sessions,) scenario number of the session's user, 0 for non-insiders
    insider_mask:         (n_sessions,) True for sessions of insider users
    score_matrices:       {model_name: array (n_runs, n_sessions)}, higher = more anomalous
    resamples:            user-level resamples (bootstrap.make_user_resamples)
    reference:            model name for the paired comparison (or None)
    """
    labels = np.asarray(labels)
    scenario_of_session = np.asarray(scenario_of_session)
    insider_mask = np.asarray(insider_mask)

    result = {
        "n_sessions": int(len(labels)),
        "n_positive": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "positives_by_scenario": {
            int(s): int(((labels == 1) & (scenario_of_session == s)).sum()) for s in (1, 2, 3)
        },
        "reference": reference,
        "models": {},
    }

    boot = {}  # (model, metric) -> per-resample values
    for name, matrix in score_matrices.items():
        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        entry = {"n_runs": int(matrix.shape[0])}
        for metric_name, metric in METRICS.items():
            mean, spread = _seed_mean_and_spread(labels, matrix, metric)
            values = metric_over_resamples(labels, matrix, resamples, metric)
            boot[(name, metric_name)] = values
            low, high = interval(values)
            entry[metric_name] = {"mean": mean, "std_over_runs": spread, "ci_low": low, "ci_high": high}

        per_scenario = {}
        for s in (1, 2, 3):
            keep = (~insider_mask) | ((scenario_of_session == s) & (labels == 1))
            if ((scenario_of_session == s) & (labels == 1)).sum() == 0:
                per_scenario[s] = None
                continue
            per_run = [safe_auroc((labels == 1)[keep], run[keep]) for run in matrix]
            per_scenario[s] = float(np.mean(per_run))
        entry["per_scenario_auroc"] = per_scenario
        result["models"][name] = entry

    if reference is not None and reference in score_matrices:
        for name, entry in result["models"].items():
            if name == reference:
                continue
            diff = paired_difference(boot[(name, "auroc")], boot[(reference, "auroc")])
            low, high = interval(diff)
            entry["auroc_minus_reference"] = {
                "mean": float(np.nanmean(diff)) if np.isfinite(diff).any() else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
    return result


def _fmt_ci(entry):
    return f"{entry['mean']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"


def format_markdown(labeling_name, description, result, display_names=None):
    """A markdown table for one labelling."""
    display_names = display_names or {}
    lines = [
        f"### Labelling: {labeling_name}",
        "",
        description,
        "",
        f"{result['n_sessions']} sessions, {result['n_positive']} malicious "
        f"(prevalence {result['prevalence']:.3f}, which is the chance-level AUPRC); "
        f"malicious sessions by scenario: {result['positives_by_scenario']}.",
        "",
    ]
    ref = result["reference"]
    header = ["Model", "runs", "AUROC [95% CI]", "AUPRC [95% CI]"]
    if ref:
        header.append(f"AUROC minus {display_names.get(ref, ref)} [95% CI]")
    header += ["s1", "s2", "s3"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for name, entry in result["models"].items():
        row = [display_names.get(name, name), str(entry["n_runs"]), _fmt_ci(entry["auroc"]), _fmt_ci(entry["auprc"])]
        if ref:
            if name == ref:
                row.append("(reference)")
            elif "auroc_minus_reference" in entry:
                d = entry["auroc_minus_reference"]
                mark = " *" if d["ci_excludes_zero"] else ""
                row.append(f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{mark}")
            else:
                row.append("n/a")
        for s in (1, 2, 3):
            v = entry["per_scenario_auroc"].get(s)
            row.append("n/a" if v is None else f"{v:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "0.5 AUROC = no separation. `*` = the interval for the difference excludes 0. "
        "Intervals resample whole users. Scenario columns: that scenario's malicious "
        "sessions vs benign users' sessions (s1 and s3 are tiny samples: anecdotal).",
        "",
    ]
    return "\n".join(lines)
