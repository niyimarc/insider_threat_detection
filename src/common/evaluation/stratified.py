"""
Date-STRATIFIED AUROC: a ranking metric that cannot be helped by knowing
WHEN a row happened.

The problem it solves: insider windows sit in particular stretches of the
dataset, and activity drifts over time for everyone. A score that merely
tracks the calendar then looks like detection (a model given nothing but
the date reaches AUROC ~0.74 on the supervised user-day task). Here every
malicious row is compared ONLY with non-malicious rows from the SAME
calendar stratum (default: the same week), so the date carries no
information at the stratum level. Concretely, the metric is the fraction of
(malicious, non-malicious) pairs, restricted to pairs from the same
stratum, in which the malicious row scores higher (ties count half). A
score that is constant within each stratum (depends on the stratum only)
gets 0.5 by construction. A score that varies by day INSIDE a stratum is
not covered by that guarantee: where a window starts or ends mid-stratum
it can land above or below 0.5. Pair this metric with date-matched
TRAINING when the model is given any date-like input.

Strata with only one class contribute nothing (there is no pair in them).
`coverage` reports how many strata and malicious rows are actually used,
so a result is never quoted without knowing what it was computed on.

Pure numpy: fast enough to run inside a 500-resample user-level bootstrap.
"""

import numpy as np

from src.common.evaluation.bootstrap import interval, paired_difference


def _stratum_ids(strata):
    """Dense 0..k-1 ids for arbitrary stratum labels."""
    _, ids = np.unique(np.asarray(strata), return_inverse=True)
    return ids


def stratified_auroc(labels, scores, strata):
    """
    AUROC over same-stratum (malicious, non-malicious) pairs, ties = 0.5.
    NaN if no stratum contains both classes.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    ids = _stratum_ids(strata)
    n = len(labels)
    if n == 0:
        return float("nan")

    order = np.lexsort((scores, ids))               # by stratum, then score
    l, s, g = labels[order], scores[order], ids[order]
    position = np.arange(n)

    new_stratum = np.r_[True, g[1:] != g[:-1]]
    stratum_start = np.maximum.accumulate(np.where(new_stratum, position, 0))
    pos_in_stratum = position - stratum_start       # 0-based rank position inside the stratum
    stratum_index = np.cumsum(new_stratum) - 1

    new_run = new_stratum | np.r_[True, s[1:] != s[:-1]]   # runs of tied scores within a stratum
    run_index = np.cumsum(new_run) - 1
    run_first = pos_in_stratum[new_run]
    run_length = np.bincount(run_index)
    average_rank = run_first[run_index] + (run_length[run_index] - 1) / 2.0 + 1.0  # 1-based, ties averaged

    n_pos = np.bincount(stratum_index, weights=l)
    n_all = np.bincount(stratum_index)
    n_neg = n_all - n_pos
    rank_sum_pos = np.bincount(stratum_index, weights=average_rank * l)
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2.0     # concordant pairs (+0.5 per tie)
    pairs = n_pos * n_neg
    valid = pairs > 0
    if not valid.any():
        return float("nan")
    return float(u_statistic[valid].sum() / pairs[valid].sum())


def coverage(labels, strata):
    """
    What the stratified metric is actually computed on:
    {"n_strata": all strata, "n_strata_used": strata with both classes,
     "n_positive": malicious rows, "n_positive_used": malicious rows in used strata}.
    """
    labels = np.asarray(labels).astype(int)
    ids = _stratum_ids(strata)
    n_pos = np.bincount(ids, weights=labels, minlength=ids.max() + 1 if len(ids) else 0)
    n_all = np.bincount(ids, minlength=n_pos.shape[0])
    used = (n_pos > 0) & (n_all - n_pos > 0)
    return {
        "n_strata": int(len(n_all)),
        "n_strata_used": int(used.sum()),
        "n_positive": int(labels.sum()),
        "n_positive_used": int(n_pos[used].sum()),
    }


def stratified_metric_over_resamples(labels, scores, strata, resamples):
    """
    Stratified AUROC on every user-level resample (see bootstrap.make_user_resamples).
    `scores` is 1-D (one run) or 2-D (n_runs, n_rows); with several runs the
    metric is averaged over the runs inside each resample.
    """
    labels = np.asarray(labels)
    strata = np.asarray(strata)
    scores = np.asarray(scores, dtype=float)
    runs = scores[None, :] if scores.ndim == 1 else scores
    values = np.full(len(resamples), np.nan)
    for i, idx in enumerate(resamples):
        per_run = [stratified_auroc(labels[idx], run[idx], strata[idx]) for run in runs]
        values[i] = np.mean(per_run)
    return values


def evaluate_stratified(labels, score_matrices, strata, resamples, reference=None):
    """
    {model: array (n_runs, n_rows)} -> per model the seed-mean stratified AUROC
    with a 95% user-level bootstrap interval, and (if `reference` is given) the
    PAIRED difference against the reference model on the same resamples.
    """
    labels = np.asarray(labels).astype(int)
    result = {"coverage": coverage(labels, strata), "reference": reference, "models": {}}
    boot = {}
    for name, matrix in score_matrices.items():
        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        per_run = [stratified_auroc(labels, run, strata) for run in matrix]
        values = stratified_metric_over_resamples(labels, matrix, strata, resamples)
        boot[name] = values
        low, high = interval(values)
        result["models"][name] = {
            "n_runs": int(matrix.shape[0]),
            "auroc": {
                "mean": float(np.mean(per_run)),
                "std_over_runs": float(np.std(per_run)) if len(per_run) > 1 else 0.0,
                "ci_low": low,
                "ci_high": high,
            },
        }
    if reference is not None and reference in boot:
        for name, entry in result["models"].items():
            if name == reference:
                continue
            diff = paired_difference(boot[name], boot[reference])
            low, high = interval(diff)
            entry["auroc_minus_reference"] = {
                "mean": float(np.nanmean(diff)) if np.isfinite(diff).any() else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
    return result


def format_stratified_markdown(title, description, result, display_names=None):
    """A markdown table of date-stratified AUROCs."""
    display_names = display_names or {}
    cov = result["coverage"]
    ref = result["reference"]
    lines = [
        f"#### {title}", "",
        description, "",
        f"Computed on {cov['n_strata_used']} of {cov['n_strata']} calendar strata that contain both classes, "
        f"covering {cov['n_positive_used']} of {cov['n_positive']} malicious rows. 0.5 = no separation.", "",
    ]
    header = ["Model", "runs", "date-stratified AUROC [95% CI]"]
    if ref:
        header.append(f"minus {display_names.get(ref, ref)} [95% CI]")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for name, entry in result["models"].items():
        a = entry["auroc"]
        row = [display_names.get(name, name), str(entry["n_runs"]),
               f"{a['mean']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]"]
        if ref:
            if name == ref:
                row.append("(reference)")
            elif "auroc_minus_reference" in entry:
                d = entry["auroc_minus_reference"]
                mark = " *" if d["ci_excludes_zero"] else ""
                row.append(f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{mark}")
            else:
                row.append("n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "`*` = the interval for the difference excludes 0. Intervals resample whole users.", ""]
    return "\n".join(lines)
