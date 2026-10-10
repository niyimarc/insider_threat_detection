"""
Can PEER BASELINES (LDAP), an ENSEMBLE and PERSISTENT-ALERT RULES make the detector better?

This is NOT a clean unseen-user test: the locked users were scored once by the frozen v1 detector, so every
insider has been seen. Everything here is user-level cross-validation over ALL users (no user is scored by a
model trained on that user), a development-style, optimistic estimate. The frozen v1 locked result remains the
clean evidence. Read the PAIRED differences against v1 (same folds, rows and resamples).

Fixed BEFORE looking at any result (and every variant below is reported, none is picked after the fact):
  feature sets   v1 = the frozen 40 columns;  v1_peer = v1 + 16 LDAP peer columns (src/supervised/peer_signals.py);
                 v2 = v1 + 17 email columns;  v3 = v1 + email + peer.
  models         gradient boosting, GRU, and their ENSEMBLE (mean percentile rank of the two day scores).
  alert rules    "single" (v1 rule) and "persist_2of3" (a flagged day with another flagged day among the 2 previous
                 active days), each at the 99th and the 95th percentile of benign day scores. Cutoffs are
                 CROSS-FITTED: a fold of users is judged with a cutoff computed from benign users of the other folds.
No answer-key event or result chose any column, rule or quantile. The peer means include insiders (a deployment
cannot know who they are).

    python scripts/supervised/evaluate_v3_detector.py
    python scripts/supervised/evaluate_v3_detector.py --models gradient_boosting --n-repeats 1 --tag quick

Needs data/processed/email_user_day.parquet (scripts/common/precompute_email_daily.py) and the raw LDAP folder.
Outputs (default results/supervised/): v3_detector_cv_<tag>.md / .json
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
from src.common.preprocessing.ldap import load_ldap_snapshots
from src.common.preprocessing.load import load_device_events, load_logon_events
from src.supervised.alert_rules import ALERT_RULES, cross_fitted_alert_analysis, rank_average
from src.supervised.all_days import build_all_user_days
from src.supervised.dataset import WINDOW_LABEL, add_calendar_controls
from src.supervised.day_signals import device_day_counts, logon_day_counts
from src.supervised.detector import out_of_fold_day_scores
from src.supervised.experiment import ALL_CONFIG, GROUP_CONFIGS
from src.supervised.feature_sets import (
    ALL_DAYS, all_days_plus_email_columns, all_days_plus_email_peer_columns, all_days_plus_peer_columns,
)
from src.supervised.models import ALL_MODEL_NAMES, DISPLAY_NAMES
from src.supervised.peer_signals import add_peer_features
from src.supervised.user_level import matched_period_auroc, user_level_auroc, user_scores, windows_from_labels

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
DATA_PATH = os.path.join(PROCESSED_DIR, "file_events_processed.parquet")
STRATA_DAYS = 7
ENSEMBLE = "ensemble"
ENSEMBLE_NAME = "Ensemble: mean rank of gradient boosting and GRU"
VARIANTS = {"v1": "v1 (frozen features)", "v1_peer": "v1 + peer baselines", "v2": "v2 (+ email)",
            "v3": "v3 (+ email + peer)"}
ALERT_QUANTILES = (0.99, 0.95)


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
    print("Loading the raw logs, http, email and LDAP ...")
    logon, device = load_logon_events(), load_device_events()
    http_user_day = build_http_user_day_features(load_http_domain_daily(PROCESSED_DIR))
    email = load_email_user_day(PROCESSED_DIR)
    ldap = load_ldap_snapshots()
    windows_df = load_malicious_windows()
    key_days = answer_key_labeled_days(load_answer_key_events())
    scenario_by_user = windows_df.set_index("user")["scenario"].to_dict()

    print("Building the all-days table (email + peers; peers use ALL employees' activity) ...")
    table = build_all_user_days(
        logon, device, test_events, http_user_day, windows_df[windows_df["user"].isin(users_all)], key_days,
        users_all, email_user_day=email)
    table = add_peer_features(table, logon_day_counts(logon), device_day_counts(device), email, ldap)
    table = add_calendar_controls(table)
    users, days = table["user"].to_numpy(), table["day"].to_numpy()
    insiders = {u for u in np.unique(users) if u in scenario_by_user}
    n_benign = len(np.unique(users)) - len(insiders)
    windows = windows_from_labels(users, days, table[WINDOW_LABEL].to_numpy())
    print(f"{len(table)} user-days, {len(np.unique(users))} users ({len(insiders)} insiders, {n_benign} benign)")

    column_sets = {"v1": ALL_DAYS.columns_fn(GROUP_CONFIGS[ALL_CONFIG]), "v1_peer": all_days_plus_peer_columns(),
                   "v2": all_days_plus_email_columns(), "v3": all_days_plus_email_peer_columns()}
    scores, label = {}, {}
    for model in args.models:
        for variant, columns in column_sets.items():
            key = f"{model}/{variant}"
            print(f"Scoring days: {key} ({len(columns)} columns) ...")
            scores[key] = out_of_fold_day_scores(table, scenario_by_user, model, columns, n_folds=args.n_folds,
                                                 n_repeats=args.n_repeats, seed=args.seed, date_matched=True,
                                                 strata_days=STRATA_DAYS)
            label[key] = f"{DISPLAY_NAMES[model]}, {VARIANTS[variant]}"
    families = list(args.models)
    if {"gradient_boosting", "gru"} <= set(args.models):
        for variant in column_sets:
            key = f"{ENSEMBLE}/{variant}"
            scores[key] = rank_average([scores[f"gradient_boosting/{variant}"], scores[f"gru/{variant}"]])
            label[key] = f"{ENSEMBLE_NAME}, {VARIANTS[variant]}"
        families.append(ENSEMBLE)

    results = {"commit": _git_commit(), "n_users": int(len(np.unique(users))), "n_insiders": len(insiders),
               "n_benign": int(n_benign), "columns": {v: len(c) for v, c in column_sets.items()}}
    md = [
        f"# v3 detector: peer baselines, ensemble and persistent alerts, ALL users, user-level cross-validation (`{args.tag}`)\n",
        f"{results['n_users']} users ({len(insiders)} insiders, {n_benign} benign), {len(table)} user-days. "
        f"{args.n_folds} folds x {args.n_repeats} repeats, date-matched training, {args.n_boot} bootstrap resamples. "
        f"Columns: {results['columns']}. Commit `{results['commit']}`.\n",
        "**This is NOT a clean unseen-user test.** The locked users were scored once by the frozen v1 detector, so every "
        "insider has been seen; this is development-style and optimistic. Read the PAIRED differences against v1; the "
        "frozen v1 locked result remains the clean evidence. Features, rules and quantiles were fixed before any result "
        "and ALL variants are shown. With only "
        f"{n_benign} benign users the false-alarm figures are coarse.\n",
    ]

    # ---- 1. day level ----
    ins_mask = np.array([u in insiders for u in users])
    labels_all = table[WINDOW_LABEL].to_numpy().astype(int)
    strata_all = (table["calendar_day_index"].to_numpy() // STRATA_DAYS).astype(int)
    md.append("## 1. Day level: insiders only, date-stratified\n")
    results["day_level"] = {}
    subsets = {"all insiders": ins_mask}
    for s in (1, 2, 3):
        subsets[f"scenario {s} insiders only"] = np.array([scenario_by_user.get(u, 0) == s for u in users])
    for name, mask in subsets.items():
        if not mask.any() or not (0 < labels_all[mask].sum() < mask.sum()):
            md.append(f"#### {name}\n\nNot computable.\n")
            continue
        resamples = make_user_resamples(users[mask], n_boot=args.n_boot, seed=0)
        for family in families:
            keys = [k for k in scores if k.startswith(family + "/")]
            res = evaluate_stratified(labels_all[mask], {k: scores[k][mask][None, :] for k in keys},
                                      strata_all[mask], resamples, reference=f"{family}/v1")
            results["day_level"].setdefault(name, {})[family] = res
            title = ENSEMBLE_NAME if family == ENSEMBLE else DISPLAY_NAMES[family]
            md.append(format_stratified_markdown(f"{name}: {title}", "Window labelling.", res, label))

    # ---- 2. person ranking ----
    md.append("## 2. Whole-timeline person ranking (user-level AUROC, insiders vs benign users)\n")
    results["user_level"] = {}
    for rule in ("top_k_mean", "max"):
        md.append(f"#### rule: {rule}\n\n| Model | user-level AUROC [95% CI] | minus v1 [95% CI] |\n|---|---|---|")
        for family in families:
            keys = [k for k in scores if k.startswith(family + "/")]
            res = user_level_auroc({k: user_scores(users, scores[k], rule=rule) for k in keys}, insiders,
                                   n_boot=args.n_boot, seed=0, reference=f"{family}/v1")
            res[f"{family}/v1"]["is_reference"] = True
            results["user_level"].setdefault(rule, {})[family] = res
            for k in keys:
                md.append(f"| {label[k]} | {_fmt(res[k])} | {_diff(res[k])} |")
        md.append("")

    # ---- 3. matched period ----
    md.append("## 3. Matched-period check (insider's window vs benign users on the same dates)\n")
    md.append("| Model | matched-period AUROC [95% CI] | minus v1 [95% CI] |\n|---|---|---|")
    results["matched_period"] = {}
    for family in families:
        keys = [k for k in scores if k.startswith(family + "/")]
        res = matched_period_auroc(users, days, {k: scores[k] for k in keys}, windows, n_boot=args.n_boot, seed=0,
                                   reference=f"{family}/v1")
        res[f"{family}/v1"]["is_reference"] = True
        results["matched_period"][family] = res
        for k in keys:
            md.append(f"| {label[k]} | {_fmt(res[k])} | {_diff(res[k])} |")
    md.append("")

    # ---- 4. alerts, cross-fitted cutoffs ----
    md.append("## 4. Alert simulation with CROSS-FITTED cutoffs (a fold of users is judged with a cutoff set on benign users of the other folds)\n")
    results["alerts"] = {}
    for rule in ALERT_RULES:
        for quantile in ALERT_QUANTILES:
            md.append(f"#### rule `{rule}`, cutoff = {int(round(quantile * 100))}th percentile of benign day scores\n")
            md.append("| Model | detected: all | scenario 1 | scenario 2 | scenario 3 | median days to first alert | "
                      "insiders alerted BEFORE window | benign users alerted | alert days per benign user |\n"
                      "|---|---|---|---|---|---|---|---|---|")
            for k in scores:
                a = cross_fitted_alert_analysis(users, days, scores[k], windows, quantile, rule,
                                                n_folds=args.n_folds, seed=args.seed)
                s = a["summary"]
                by_scenario = {}
                for sc in (1, 2, 3):
                    members = [u for u, e in a["per_user"].items() if e["is_insider"] and scenario_by_user[u] == sc]
                    hit = sum(a["per_user"][u]["first_alert_in_window"] is not None for u in members)
                    by_scenario[sc] = (hit, len(members))
                results["alerts"][f"{rule}/{quantile}/{k}"] = {
                    "by_scenario": {str(c): v for c, v in by_scenario.items()}, **s}
                md.append(f"| {label[k]} | {s['insiders_detected']} of {s['n_insiders']} | "
                          + " | ".join(f"{h} of {n}" for h, n in by_scenario.values())
                          + f" | {s['median_days_to_first_alert']:.0f} | {s['insiders_alerted_before_window']} of "
                            f"{s['n_insiders']} | {s['benign_users_alerted']} of {s['n_benign']} | "
                            f"{s['benign_alert_days_per_user']:.1f} |")
            md.append("")
    md.append("_All users, cross-validated: development-style numbers, optimistic. Peer baselines use staff role/team "
              "from LDAP and the activity of all employees; email features are envelope data only._\n")

    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"v3_detector_cv_{args.tag}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    markdown = "\n".join(md)
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write(markdown)
    print("\n" + markdown)
    print(f"Saved {os.path.join(out_dir, stem)}.json and .md")


if __name__ == "__main__":
    main()
