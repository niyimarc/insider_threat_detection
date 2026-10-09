"""
Does the EMAIL signal improve the detector? (v2 = frozen v1 features + email features)

This is NOT a clean unseen-user test. The locked users were scored once by the frozen v1 detector, so
every insider in the data has now been seen. This script therefore uses ALL users (dev + locked) in
user-level cross-validation: no user is ever scored by a model that trained on that user, but the
estimate is a development-style, optimistic one. The frozen v1 locked result stays the clean evidence;
what is reported here is the PAIRED difference v2 - v1 on identical folds, rows and resamples, next to an
email-only model and the date / per-user-clock controls.

The email feature list (src/common/preprocessing/email_features.py, src/supervised/email_day_signals.py)
was fixed in advance from the dataset readme and scenarios.txt; no answer-key event or result was used to
choose it. Needs data/processed/email_user_day.parquet (scripts/common/precompute_email_daily.py).

    python scripts/supervised/evaluate_email_signal.py
    python scripts/supervised/evaluate_email_signal.py --models gradient_boosting --n-repeats 1 --tag quick

Outputs (default results/supervised/): email_signal_cv_<tag>.md / .json
"""

import argparse
import json
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.common.evaluation.bootstrap import make_user_resamples
from src.common.evaluation.labelings import answer_key_labeled_days, load_answer_key_events
from src.common.evaluation.partition import select_test_df
from src.common.evaluation.stratified import evaluate_stratified, format_stratified_markdown
from src.common.preprocessing.email_features import OUTPUT_NAME, load_email_user_day
from src.common.preprocessing.http_features import build_http_user_day_features, load_http_domain_daily
from src.common.preprocessing.labeling import load_malicious_windows
from src.common.preprocessing.load import load_device_events, load_logon_events
from src.supervised.all_days import build_all_user_days
from src.supervised.dataset import WINDOW_LABEL, add_calendar_controls
from src.supervised.detector import out_of_fold_day_scores
from src.supervised.email_day_signals import EMAIL_DAY_COLUMNS
from src.supervised.experiment import CONTROLS, GROUP_CONFIGS, ALL_CONFIG
from src.supervised.feature_sets import ALL_DAYS, all_days_plus_email_columns
from src.supervised.models import ALL_MODEL_NAMES, DISPLAY_NAMES
from src.supervised.user_level import (
    ALERT_QUANTILE, alert_analysis, matched_period_auroc, user_level_auroc, user_scores, windows_from_labels,
)

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
DATA_PATH = os.path.join(PROCESSED_DIR, "file_events_processed.parquet")
STRATA_DAYS = 7
VARIANTS = {"v1": "v1 (no email)", "v2": "v2 (+ email)", "email_only": "email features only"}


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _fmt(e):
    return f"{e['auroc']:.3f} [{e['ci_low']:.3f}, {e['ci_high']:.3f}]"


def _diff(e):
    if "auroc_minus_reference" not in e:
        return "(reference)" if e.get("is_reference") else "n/a"
    d = e["auroc_minus_reference"]
    return f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{' *' if d['ci_excludes_zero'] else ''}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["gradient_boosting", "gru"], choices=ALL_MODEL_NAMES)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/supervised")
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(PROCESSED_DIR, OUTPUT_NAME)):
        sys.exit(f"{OUTPUT_NAME} is missing: run scripts/common/precompute_email_daily.py first.")

    print(f"Loading {DATA_PATH} ...")
    events = pd.read_parquet(DATA_PATH)
    test_events = select_test_df(events, partition="all")      # every insider: cross-validation, not a clean test
    users_all = set(test_events["user"].unique())
    print("Loading the raw logs, the http and email features ...")
    logon, device = load_logon_events(), load_device_events()
    http_user_day = build_http_user_day_features(load_http_domain_daily(PROCESSED_DIR))
    email = load_email_user_day(PROCESSED_DIR)
    windows_df = load_malicious_windows()
    key_days = answer_key_labeled_days(load_answer_key_events())
    scenario_by_user = windows_df.set_index("user")["scenario"].to_dict()

    print("Building the all-days table with email ...")
    table = add_calendar_controls(build_all_user_days(
        logon, device, test_events, http_user_day, windows_df[windows_df["user"].isin(users_all)], key_days,
        users_all, email_user_day=email))
    users, days = table["user"].to_numpy(), table["day"].to_numpy()
    insiders = {u for u in np.unique(users) if u in scenario_by_user}
    n_benign = len(np.unique(users)) - len(insiders)
    windows = windows_from_labels(users, days, table[WINDOW_LABEL].to_numpy())
    print(f"{len(table)} user-days, {len(np.unique(users))} users ({len(insiders)} insiders, {n_benign} benign); "
          f"{100 * (table['em_n_sent'] > 0).mean():.0f}% of days have email")

    column_sets = {"v1": ALL_DAYS.columns_fn(GROUP_CONFIGS[ALL_CONFIG]), "v2": all_days_plus_email_columns(),
                   "email_only": list(EMAIL_DAY_COLUMNS)}
    scores, label = {}, {}
    for model in args.models:
        for variant, columns in column_sets.items():
            key = f"{model}/{variant}"
            print(f"Scoring days: {key} ({len(columns)} columns) ...")
            scores[key] = out_of_fold_day_scores(table, scenario_by_user, model, columns, n_folds=args.n_folds,
                                                 n_repeats=args.n_repeats, seed=args.seed, date_matched=True,
                                                 strata_days=STRATA_DAYS)
            label[key] = f"{DISPLAY_NAMES[model]}, {VARIANTS[variant]}"
        for control, (control_columns, text) in CONTROLS.items():
            key = f"{model}/{control}"
            print(f"Scoring days: {key} ...")
            scores[key] = out_of_fold_day_scores(table, scenario_by_user, model, list(control_columns),
                                                 n_folds=args.n_folds, n_repeats=args.n_repeats, seed=args.seed,
                                                 date_matched=True, strata_days=STRATA_DAYS)
            label[key] = f"CONTROL ({text}): {DISPLAY_NAMES[model]}"

    results = {"commit": _git_commit(), "n_users": int(len(np.unique(users))), "n_insiders": len(insiders),
               "n_benign": int(n_benign), "n_email_columns": len(EMAIL_DAY_COLUMNS),
               "n_v1_columns": len(column_sets["v1"]), "n_v2_columns": len(column_sets["v2"])}
    md = [
        f"# Email signal: v2 (+ email) against v1, ALL users, user-level cross-validation (`{args.tag}`)\n",
        f"{results['n_users']} users ({len(insiders)} insiders, {n_benign} benign), {len(table)} user-days. "
        f"{args.n_folds} folds x {args.n_repeats} repeats, date-matched training, {args.n_boot} bootstrap resamples. "
        f"v1 = the frozen detector's {results['n_v1_columns']} input columns; v2 = v1 + {results['n_email_columns']} "
        f"email columns. Commit `{results['commit']}`.\n",
        "**This is NOT a clean unseen-user test.** The locked users were scored once by the frozen v1 detector, so "
        "every insider has been seen; this uses all users in user-level cross-validation (no user is scored by a "
        "model that trained on them) and is development-style and optimistic. Read the PAIRED differences v2 minus v1 "
        "(same folds, rows and resamples); the frozen v1 locked result remains the clean evidence. The email feature "
        "list was fixed in advance from the dataset readme and scenarios.txt.\n",
    ]

    # ---- 1. day level: insiders only, date-stratified ----
    ins_mask = np.array([u in insiders for u in users])
    labels_all = table[WINDOW_LABEL].to_numpy().astype(int)
    strata_all = (table["calendar_day_index"].to_numpy() // STRATA_DAYS).astype(int)
    md.append("## 1. Day level: insiders only, date-stratified (window days vs the same insiders' other days, same week)\n")
    results["day_level"] = {}
    subsets = {"all insiders": ins_mask}
    for s in (1, 2, 3):
        subsets[f"scenario {s} insiders only"] = np.array([scenario_by_user.get(u, 0) == s for u in users])
    for name, mask in subsets.items():
        if not mask.any() or not (0 < labels_all[mask].sum() < mask.sum()):
            md.append(f"#### {name}\n\nNot computable.\n")
            continue
        resamples = make_user_resamples(users[mask], n_boot=args.n_boot, seed=0)
        for model in args.models:
            keys = [k for k in scores if k.startswith(model + "/")]
            res = evaluate_stratified(labels_all[mask], {k: scores[k][mask][None, :] for k in keys},
                                      strata_all[mask], resamples, reference=f"{model}/v1")
            results["day_level"].setdefault(name, {})[model] = res
            md.append(format_stratified_markdown(f"{name}: {DISPLAY_NAMES[model]}", "Window labelling.", res, label))

    # ---- 2. whole-timeline person ranking, 3. matched period ----
    md.append("## 2. Whole-timeline person ranking (user-level AUROC, insiders vs benign users)\n")
    results["user_level"] = {}
    for rule in ("top_k_mean", "max"):
        md.append(f"#### rule: {rule}\n\n| Model | user-level AUROC [95% CI] | minus v1 [95% CI] |\n|---|---|---|")
        for model in args.models:
            keys = [k for k in scores if k.startswith(model + "/")]
            res = user_level_auroc({k: user_scores(users, scores[k], rule=rule) for k in keys}, insiders,
                                   n_boot=args.n_boot, seed=0, reference=f"{model}/v1")
            res[f"{model}/v1"]["is_reference"] = True
            results["user_level"].setdefault(rule, {})[model] = res
            for k in keys:
                md.append(f"| {label[k]} | {_fmt(res[k])} | {_diff(res[k])} |")
        md.append("")
    md.append("## 3. Matched-period check (insider's window vs benign users on the same dates)\n")
    md.append("| Model | matched-period AUROC [95% CI] | minus v1 [95% CI] |\n|---|---|---|")
    results["matched_period"] = {}
    for model in args.models:
        keys = [k for k in scores if k.startswith(model + "/")]
        res = matched_period_auroc(users, days, {k: scores[k] for k in keys}, windows, n_boot=args.n_boot, seed=0,
                                   reference=f"{model}/v1")
        res[f"{model}/v1"]["is_reference"] = True
        results["matched_period"][model] = res
        for k in keys:
            md.append(f"| {label[k]} | {_fmt(res[k])} | {_diff(res[k])} |")
    md.append("")

    # ---- 4. alert simulation, by scenario ----
    md.append(f"## 4. Alert simulation (cutoff = {int(ALERT_QUANTILE * 100)}th percentile of benign users' day scores, per model)\n")
    md.append("| Model | detected: all | scenario 1 | scenario 2 | scenario 3 | median days to first alert | "
              "insiders alerted BEFORE window | benign users alerted |\n|---|---|---|---|---|---|---|---|")
    results["alerts"] = {}
    for k in scores:
        if k.split("/")[1] in CONTROLS:
            continue
        a = alert_analysis(users, days, scores[k], windows)
        s = a["summary"]
        by_scenario = {}
        for sc in (1, 2, 3):
            members = [u for u, e in a["per_user"].items() if e["is_insider"] and scenario_by_user[u] == sc]
            hit = sum(a["per_user"][u]["first_alert_in_window"] is not None for u in members)
            by_scenario[sc] = (hit, len(members))
        results["alerts"][k] = {"threshold": a["threshold"], "by_scenario": {str(c): v for c, v in by_scenario.items()}, **s}
        md.append(f"| {label[k]} | {s['insiders_detected']} of {s['n_insiders']} | "
                  + " | ".join(f"{h} of {n}" for h, n in by_scenario.values())
                  + f" | {s['median_days_to_first_alert']:.0f} | {s['insiders_alerted_before_window']} of {s['n_insiders']} | "
                    f"{s['benign_users_alerted']} of {s['n_benign']} |")
    md.append("\n_All users, cross-validated: development-style numbers, optimistic. Email features describe envelope "
              "data only (counts, recipients, domains, attachments, size, hour), never message text._\n")

    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"email_signal_cv_{args.tag}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    markdown = "\n".join(md)
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write(markdown)
    print("\n" + markdown)
    print(f"Saved {os.path.join(out_dir, stem)}.json and .md")


if __name__ == "__main__":
    main()
