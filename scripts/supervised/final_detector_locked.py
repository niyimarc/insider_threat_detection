"""
THE ONE-SHOT FINAL TEST of the frozen supervised insider detector on the LOCKED users.

The method is fixed in FROZEN_SUPERVISED_METHOD.md (read it first; this script refuses to run without it
and records its SHA-256). Both frozen models (gradient boosting and the GRU, all signals, date-matched
training) are trained on the development users only, then score the locked users, who they have never
seen. Whatever comes out is what gets reported. Nothing is tuned here.

    python scripts/supervised/final_detector_locked.py --confirm-final

Reading the locked partition requires --confirm-final. A record file (LOCKED_RUN_RECORD.json) is written
when the run completes; a second run is refused unless you pass --allow-second-run, and the report of a
repeated run says so. (On Colab the results folder disappears with the session: copy the outputs and the
record to Drive or commit them.)

Outputs (default results/supervised/): final_detector_locked_<tag>.md / .json, the ranked-user CSVs
final_detector_users_<tag>_<model>.csv, and LOCKED_RUN_RECORD.json.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.common.evaluation.bootstrap import make_user_resamples
from src.common.evaluation.labelings import answer_key_labeled_days, load_answer_key_events
from src.common.evaluation.partition import select_test_df
from src.common.evaluation.stratified import evaluate_stratified, format_stratified_markdown
from src.common.preprocessing.http_features import build_http_user_day_features, load_http_domain_daily
from src.common.preprocessing.labeling import load_malicious_windows
from src.common.preprocessing.load import load_device_events, load_logon_events
from src.supervised.all_days import build_all_user_days
from src.supervised.dataset import WINDOW_LABEL, add_calendar_controls
from src.supervised.detector import align_calendar_origin, fit_and_score, out_of_fold_day_scores
from src.supervised.experiment import ALL_CONFIG, CONTROLS, GROUP_CONFIGS
from src.supervised.feature_sets import ALL_DAYS
from src.supervised.models import DISPLAY_NAMES
from src.supervised.user_level import (
    ALERT_QUANTILE, TOP_K_DAYS, alert_analysis, matched_period_auroc, precision_at_k, ranked_user_table,
    user_level_auroc, user_scores, windows_from_labels,
)

FROZEN_MODELS = ("gradient_boosting", "gru")
FROZEN_FILE = os.path.join(_PROJECT_ROOT, "FROZEN_SUPERVISED_METHOD.md")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
DATA_PATH = os.path.join(PROCESSED_DIR, "file_events_processed.parquet")
RECORD_NAME = "LOCKED_RUN_RECORD.json"
STRATA_DAYS = 7
ACTIVITY_COLUMNS = ("dv_n_connects", "dv_connected_minutes", "lg_after_hours_events",
                    "http_new_domains_today", "day_n_events")


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _fmt(entry):
    return f"{entry['auroc']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-final", action="store_true", help="required to read the locked partition")
    parser.add_argument("--allow-second-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out-dir", default="results/supervised")
    parser.add_argument("--tag", default="locked")
    args = parser.parse_args()

    if not args.confirm_final:
        sys.exit("Refusing to read the locked partition without --confirm-final. The locked users are for ONE "
                 "final evaluation of the method frozen in FROZEN_SUPERVISED_METHOD.md.")
    if not os.path.exists(FROZEN_FILE):
        sys.exit("FROZEN_SUPERVISED_METHOD.md is missing: the method must be written down before the locked run.")
    out_dir = _resolve(args.out_dir)
    record_path = os.path.join(out_dir, RECORD_NAME)
    repeated = os.path.exists(record_path)
    if repeated and not args.allow_second_run:
        sys.exit(f"{record_path} exists: the locked users were already scored. Pass --allow-second-run only if "
                 "you accept that the report then says it is a repeated run.")
    spec_hash = _sha256(FROZEN_FILE)

    print(f"Loading {DATA_PATH} ...")
    events = pd.read_parquet(DATA_PATH)
    dev_events = select_test_df(events, partition="dev")
    locked_events = select_test_df(events, partition="locked")
    dev_users, locked_users = set(dev_events["user"].unique()), set(locked_events["user"].unique())
    if dev_users & locked_users:
        sys.exit("A user is in both partitions: stop.")

    print("Loading the raw logon and device logs ...")
    logon, device = load_logon_events(), load_device_events()
    print("Building the http user-day features ...")
    http_user_day = build_http_user_day_features(load_http_domain_daily(PROCESSED_DIR))
    windows_df = load_malicious_windows()
    key_days = answer_key_labeled_days(load_answer_key_events())
    scenario_by_user = windows_df.set_index("user")["scenario"].to_dict()

    def build(user_events, users):
        return build_all_user_days(logon, device, user_events, http_user_day,
                                   windows_df[windows_df["user"].isin(users)], key_days, users)

    print("Building the development and locked tables ...")
    dev_table = add_calendar_controls(build(dev_events, dev_users))
    locked_table = align_calendar_origin(add_calendar_controls(build(locked_events, locked_users)), dev_table)

    users, days = locked_table["user"].to_numpy(), locked_table["day"].to_numpy()
    locked_insiders = {u for u in np.unique(users) if u in scenario_by_user}
    n_benign = int(len(np.unique(users)) - len(locked_insiders))
    scenario_counts = pd.Series([scenario_by_user[u] for u in locked_insiders]).value_counts().sort_index().to_dict()
    print(f"LOCKED: {len(np.unique(users))} users ({len(locked_insiders)} insiders {scenario_counts}, {n_benign} benign), "
          f"{len(locked_table)} user-days")

    columns = ALL_DAYS.columns_fn(GROUP_CONFIGS[ALL_CONFIG])
    columns_hash = hashlib.sha256(",".join(columns).encode()).hexdigest()
    locked_windows = windows_from_labels(users, days, locked_table[WINDOW_LABEL].to_numpy())

    # ---- scores: frozen models trained on dev, cutoffs set on dev, then the locked users are scored ----
    scores, thresholds = {}, {}
    for model in FROZEN_MODELS:
        print(f"[{model}] development out-of-fold scores for the alert cutoff ...")
        dev_oof = out_of_fold_day_scores(dev_table, scenario_by_user, model, columns, n_folds=5, n_repeats=3,
                                         seed=0, date_matched=True, strata_days=STRATA_DAYS)
        benign_dev = ~dev_table["user"].isin(scenario_by_user).to_numpy()
        thresholds[model] = float(np.quantile(dev_oof[benign_dev], ALERT_QUANTILE))
        print(f"[{model}] training on the development users and scoring the locked users ...")
        scores[model] = fit_and_score(dev_table, locked_table, model, columns, seeds=tuple(args.seeds),
                                      date_matched=True, strata_days=STRATA_DAYS)
        for control, (control_columns, _) in CONTROLS.items():
            scores[f"{model}/{control}"] = fit_and_score(
                dev_table, locked_table, model, list(control_columns), seeds=tuple(args.seeds),
                date_matched=True, strata_days=STRATA_DAYS)

    names = {k: (DISPLAY_NAMES[k] if "/" not in k else f"CONTROL ({CONTROLS[k.split('/')[1]][1]}): "
                 f"{DISPLAY_NAMES[k.split('/')[0]]}") for k in scores}
    reals = list(FROZEN_MODELS)
    results = {
        "commit": _git_commit(), "frozen_spec_sha256": spec_hash, "input_columns_sha256": columns_hash,
        "n_input_columns": len(columns), "repeated_run": repeated, "seeds": list(args.seeds),
        "locked": {"n_users": int(len(np.unique(users))), "n_insiders": len(locked_insiders), "n_benign": n_benign,
                   "insiders_by_scenario": {str(k): int(v) for k, v in scenario_counts.items()},
                   "n_user_days": int(len(locked_table))},
        "alert_thresholds_from_dev": thresholds,
    }
    md = [
        f"# Frozen insider detector: FINAL evaluation on the LOCKED users (`{args.tag}`)\n",
        ("**THIS IS A REPEATED RUN: the locked users had already been scored.**\n" if repeated else ""),
        f"Locked users: {results['locked']['n_users']} ({results['locked']['n_insiders']} insiders by scenario "
        f"{results['locked']['insiders_by_scenario']}, {n_benign} benign), {results['locked']['n_user_days']} user-days. "
        f"Method fixed in FROZEN_SUPERVISED_METHOD.md (SHA-256 `{spec_hash[:16]}...`). Commit `{results['commit']}`. "
        f"{len(columns)} input columns (hash `{columns_hash[:16]}...`). Models trained on the 50 development users only, "
        f"seeds {list(args.seeds)} averaged; alert cutoffs set on development benign users' out-of-fold scores.\n",
        "Read with care: " + (f"only {n_benign} benign locked users exist, so every comparison against benign users is "
                              "coarse. " if n_benign < 30 else "") +
        "Scenario 2 supplies most malicious days; controls are the same model given only the date or only the user's "
        "own clock. Whatever the numbers are, they are the result.\n",
    ]

    # 1. whole-timeline person ranking
    md.append("## 1. Whole-timeline person ranking (no knowledge of when anything happened)\n")
    results["user_level"] = {}
    for rule in ("top_k_mean", "max"):
        by_model = {k: user_scores(users, scores[k], k=TOP_K_DAYS, rule=rule) for k in scores}
        res = user_level_auroc(by_model, locked_insiders, n_boot=args.n_boot, seed=0, reference="gradient_boosting")
        results["user_level"][rule] = res
        md.append(f"#### rule: {rule}\n\n| Model | user-level AUROC [95% CI] | minus gradient boosting [95% CI] |\n|---|---|---|")
        for k in scores:
            row = f"| {names[k]} | {_fmt(res[k])} |"
            if k == "gradient_boosting":
                row += " (reference) |"
            elif "auroc_minus_reference" in res[k]:
                d = res[k]["auroc_minus_reference"]
                row += f" {d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{' *' if d['ci_excludes_zero'] else ''} |"
            else:
                row += " n/a |"
            md.append(row)
        md.append("")
    md.append("| Model | top 5 | top 10 | insiders among all locked users |\n|---|---|---|---|")
    results["precision_at_k"] = {}
    for k in reals:
        p = precision_at_k(user_scores(users, scores[k]), locked_insiders, ks=(5, 10))
        results["precision_at_k"][k] = p
        md.append(f"| {names[k]} | {p[5]['insiders_in_top_k']} of {p[5]['k']} | {p[10]['insiders_in_top_k']} of "
                  f"{p[10]['k']} | {100 * p[5]['baseline_share']:.0f}% |")
    md.append("")

    # 2. matched period
    md.append("## 2. Matched-period check (insider's window vs benign users on the same dates)\n")
    res = matched_period_auroc(users, days, scores, locked_windows, n_boot=args.n_boot, seed=0,
                               reference="gradient_boosting")
    results["matched_period"] = res
    md.append("| Model | matched-period AUROC [95% CI] | coverage |\n|---|---|---|")
    for k in scores:
        md.append(f"| {names[k]} | {_fmt(res[k])} | {100 * res[k]['coverage']:.0f}% |")
    md.append("")

    # 3. alert simulation with the frozen cutoff
    md.append(f"## 3. Alert simulation (frozen cutoff = {int(ALERT_QUANTILE * 100)}th percentile of benign DEVELOPMENT users' day scores)\n")
    md.append("| Model | insiders detected in their window | median days to first alert | detected within the first week "
              "| insiders alerted BEFORE their window | benign users alerted | alert days per benign user |\n|---|---|---|---|---|---|---|")
    results["alerts"], alert_results = {}, {}
    for k in reals:
        a = alert_analysis(users, days, scores[k], locked_windows, threshold=thresholds[k])
        alert_results[k] = a
        s = a["summary"]
        results["alerts"][k] = {"threshold": a["threshold"], **s}
        md.append(f"| {names[k]} | {s['insiders_detected']} of {s['n_insiders']} ({100 * s['detection_rate']:.0f}%) | "
                  f"{s['median_days_to_first_alert']:.0f} | {100 * s['share_detected_within_7_days']:.0f}% of insiders | "
                  f"{s['insiders_alerted_before_window']} of {s['n_insiders']} | {s['benign_users_alerted']} of "
                  f"{s['n_benign']} | {s['benign_alert_days_per_user']:.1f} |")
    md.append("")

    # 4. day level, insiders only, date-stratified
    ins_mask = np.array([u in locked_insiders for u in users])
    md.append("## 4. Day level: insiders only, date-stratified (window days vs the same insiders' other days, same week)\n")
    if ins_mask.any() and 0 < locked_table[WINDOW_LABEL].to_numpy()[ins_mask].sum() < ins_mask.sum():
        strata = (locked_table["calendar_day_index"].to_numpy() // STRATA_DAYS).astype(int)[ins_mask]
        labels = locked_table[WINDOW_LABEL].to_numpy().astype(int)[ins_mask]
        resamples = make_user_resamples(users[ins_mask], n_boot=args.n_boot, seed=0)
        day_eval = evaluate_stratified(labels, {k: scores[k][ins_mask][None, :] for k in scores}, strata, resamples,
                                       reference="gradient_boosting")
        results["day_level_insiders_only"] = day_eval
        md.append(format_stratified_markdown("Insiders only, date-stratified AUROC", "Window labelling. Controls "
                                             "should be near 0.5.", day_eval, names))
    else:
        md.append("Not computable on these locked users.\n")

    # ranked users
    os.makedirs(out_dir, exist_ok=True)
    activity = [c for c in ACTIVITY_COLUMNS if c in locked_table.columns]
    for k in reals:
        frame = locked_table[["user", "day"] + activity].copy()
        frame["score"] = scores[k]
        report = ranked_user_table(user_scores(users, scores[k]), alert_results[k], scenario_by_user, frame=frame,
                                   activity_columns=activity)
        report.to_csv(os.path.join(out_dir, f"final_detector_users_{args.tag}_{k}.csv"), index=False)
        md.append(f"## Ranked locked users, {names[k]} (top 10 of {len(report)}; full list in the CSV)\n")
        md.append("`is_insider` and `scenario` come from the answer key and are for EVALUATION only.\n")
        md.append("| rank | user | user score | alert days | insider (scenario) |\n|---|---|---|---|---|")
        for _, r in report.head(10).iterrows():
            md.append(f"| {r['rank']} | {r['user']} | {r['user_score']:.3g} | {r['n_alert_days']} | "
                      f"{'yes (' + str(r['scenario']) + ')' if r['is_insider'] else 'no'} |")
        md.append("")
    md.append("_Locked-partition numbers from the frozen method: the final, once-only evaluation. Disclosures are "
              "in FROZEN_SUPERVISED_METHOD.md._\n")

    stem = f"final_detector_locked_{args.tag}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    markdown = "\n".join(md)
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write(markdown)
    with open(record_path, "w") as f:
        json.dump({"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": results["commit"],
                   "frozen_spec_sha256": spec_hash, "tag": args.tag, "repeated_run": repeated}, f, indent=2)
    print("\n" + markdown)
    print(f"Saved {os.path.join(out_dir, stem)}.json / .md, the ranked-user CSVs and {record_path}")


if __name__ == "__main__":
    main()
