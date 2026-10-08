"""
The insider DETECTOR, evaluated on the DEVELOPMENT users: from suspicious days
to suspicious PEOPLE.

For every user-day the chosen model(s) produce an out-of-fold score (no user is
ever scored by a model that saw them). The script then answers the questions a
security team asks:

  * Do the insiders rank above the benign users? (user-level AUROC, with controls)
  * Of the top-ranked users, how many are insiders?
  * If alerts fired whenever a day scored above the 99th percentile of benign
    users' days, how many insiders would be caught, how many days after their
    activity started, and how many benign users would be alerted?

and it writes the ranked user list (with each user's most suspicious days and
what they did on them) so the output can be read like a report.

All rules (user score = mean of the 7 highest day scores; alert cutoff = 99th
percentile of benign users' day scores; window = the insider's own window) are
fixed in src/supervised/user_level.py and were not tuned on these results.
Dev only: there is deliberately no locked-partition option here.

    python scripts/supervised/detect_insiders.py
    python scripts/supervised/detect_insiders.py --models gradient_boosting --n-repeats 1 --tag quick

Outputs (default results/supervised/):
  detector_dev_<tag>.md / .json, and detector_users_<tag>_<model>_<training>.csv
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

from src.common.evaluation.labelings import answer_key_labeled_days, load_answer_key_events
from src.common.evaluation.partition import select_test_df
from src.common.preprocessing.http_features import build_http_user_day_features, load_http_domain_daily
from src.common.preprocessing.labeling import load_malicious_windows
from src.common.preprocessing.load import load_device_events, load_logon_events
from src.supervised.all_days import build_all_user_days
from src.supervised.dataset import WINDOW_LABEL, add_calendar_controls
from src.supervised.detector import out_of_fold_day_scores
from src.supervised.experiment import ALL_CONFIG, CONTROLS, GROUP_CONFIGS
from src.supervised.feature_sets import ALL_DAYS
from src.supervised.models import ALL_MODEL_NAMES, DISPLAY_NAMES
from src.supervised.user_level import (
    ALERT_QUANTILE, TOP_K_DAYS, alert_analysis, matched_period_auroc, precision_at_k, ranked_user_table,
    user_level_auroc, user_scores, windows_from_labels,
)

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
DATA_PATH = os.path.join(PROCESSED_DIR, "file_events_processed.parquet")
TRAININGS = ("date_matched", "normal")
TRAINING_TEXT = {"date_matched": "trained date-matched (calendar removed)", "normal": "trained normally"}
ACTIVITY_COLUMNS = ("dv_n_connects", "dv_connected_minutes", "lg_after_hours_events",
                    "http_new_domains_today", "day_n_events")


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _load_dev_table():
    print(f"Loading {DATA_PATH} ...")
    events = pd.read_parquet(DATA_PATH)
    dev_events = select_test_df(events, partition="dev")  # dev ONLY: never the locked users
    dev_users = set(dev_events["user"].unique())
    print("Loading the raw logon and device logs ...")
    logon, device = load_logon_events(), load_device_events()
    print("Building the http user-day features ...")
    http_user_day = build_http_user_day_features(load_http_domain_daily(PROCESSED_DIR))
    windows = load_malicious_windows()
    key = load_answer_key_events()
    print("Building the all-days table ...")
    table = build_all_user_days(
        logon, device, dev_events, http_user_day,
        windows[windows["user"].isin(dev_users)], answer_key_labeled_days(key), dev_users,
    )
    scenario_by_user = windows.set_index("user")["scenario"].to_dict()
    return add_calendar_controls(table), scenario_by_user


def _label(model, training, control=None):
    base = f"{DISPLAY_NAMES[model]}, {TRAINING_TEXT[training]}"
    return base if control is None else f"CONTROL ({CONTROLS[control][1]}): {base}"


def _fmt_ci(entry):
    return f"{entry['auroc']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["gradient_boosting", "gru"], choices=ALL_MODEL_NAMES)
    parser.add_argument("--training", nargs="+", default=list(TRAININGS), choices=TRAININGS)
    parser.add_argument("--primary-model", default="gru", choices=ALL_MODEL_NAMES)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/supervised")
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()

    table, scenario_by_user = _load_dev_table()
    insiders = {u for u in table["user"].unique() if u in scenario_by_user}
    print(f"{len(table)} user-days, {table['user'].nunique()} users ({len(insiders)} insiders, "
          f"{table['user'].nunique() - len(insiders)} benign)")

    users, days = table["user"].to_numpy(), table["day"].to_numpy()
    windows = windows_from_labels(users, days, table[WINDOW_LABEL].to_numpy())
    columns = ALL_DAYS.columns_fn(GROUP_CONFIGS[ALL_CONFIG])

    day_scores, meta = {}, {}
    for model in args.models:
        for training in args.training:
            matched = training == "date_matched"
            key = f"{model}/{training}"
            print(f"Scoring days: {key} ...")
            day_scores[key] = out_of_fold_day_scores(
                table, scenario_by_user, model, columns, n_folds=args.n_folds, n_repeats=args.n_repeats,
                seed=args.seed, date_matched=matched)
            meta[key] = (model, training, None)
            for control, (control_columns, _) in CONTROLS.items():
                ckey = f"{key}/{control}"
                print(f"Scoring days: {ckey} ...")
                day_scores[ckey] = out_of_fold_day_scores(
                    table, scenario_by_user, model, list(control_columns), n_folds=args.n_folds,
                    n_repeats=args.n_repeats, seed=args.seed, date_matched=matched)
                meta[ckey] = (model, training, control)

    names = {k: _label(*meta[k]) for k in day_scores}
    results = {"commit": _git_commit(), "n_users": int(table["user"].nunique()), "n_insiders": len(insiders),
               "n_benign": int(table["user"].nunique() - len(insiders)), "top_k_days": TOP_K_DAYS,
               "alert_quantile": ALERT_QUANTILE, "user_level": {}, "precision_at_k": {}, "alerts": {}}
    md = [
        f"# Insider detector, development users (`{args.tag}`)\n",
        f"{results['n_users']} users: {results['n_insiders']} insiders and {results['n_benign']} benign. "
        f"{args.n_folds} user-level folds x {args.n_repeats} repeats (every score is out-of-fold: a model never "
        f"scores a user it trained on). Commit: `{results['commit']}`.\n",
        "**Rules, fixed in advance (not tuned on these results).** User score = the mean of the user's "
        f"{TOP_K_DAYS} highest day scores (the single highest day is the secondary rule). Alert = a day scoring at or "
        f"above the {int(ALERT_QUANTILE * 100)}th percentile of BENIGN users' day scores. Detected = at least one "
        "alert inside the insider's own window; time to detection = days from the first window day to the first "
        "in-window alert.\n",
        f"**Read with care.** Only {results['n_benign']} benign users exist in the development partition, so "
        "false-alarm figures are coarse. About "
        f"{100 * results['n_insiders'] / results['n_users']:.0f}% of these users are insiders, far more than in any "
        "organisation, so 'insiders in the top k' must be read against the printed baseline. The insider-vs-benign "
        "comparison can also reflect what kind of user an insider is; the controls (same model given only the date, "
        "or only the user's own clock) show how much of it needs no behaviour at all. Development partition only: the "
        "locked partition has not been used.\n",
    ]

    for rule in ("top_k_mean", "max"):
        md.append(f"## User-level AUROC: do insiders rank above benign users? (rule: {rule})\n")
        md.append("0.5 = no separation. Intervals resample insiders and benign users; differences are paired.\n")
        for training in args.training:
            keys = [k for k in day_scores if meta[k][1] == training]
            by_model = {k: user_scores(users, day_scores[k], rule=rule) for k in keys}
            reference = f"gradient_boosting/{training}" if f"gradient_boosting/{training}" in by_model else None
            res = user_level_auroc(by_model, insiders, n_boot=args.n_boot, seed=args.seed, reference=reference)
            results["user_level"].setdefault(rule, {})[training] = res
            md.append(f"#### {TRAINING_TEXT[training]}\n")
            header = "| Model | user-level AUROC [95% CI] |" + (" minus gradient boosting [95% CI] |" if reference else "")
            md.append(header)
            md.append("|---|---|" + ("---|" if reference else ""))
            for k in keys:
                row = f"| {names[k]} | {_fmt_ci(res[k])} |"
                if reference:
                    if k == reference:
                        row += " (reference) |"
                    elif "auroc_minus_reference" in res[k]:
                        d = res[k]["auroc_minus_reference"]
                        row += f" {d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{' *' if d['ci_excludes_zero'] else ''} |"
                    else:
                        row += " n/a |"
                md.append(row)
            md.append("\n`*` = the interval excludes 0.\n")

    md.append("## Matched-period check: insiders vs benign users over the SAME calendar dates\n")
    md.append(
        "Each insider's score over their own window is compared with every benign user's score over the same "
        "dates (both = mean of the highest k day scores, k = 7 or the number of the insider's active window days "
        "if fewer). A difference in WHEN insiders and benign users were observed cannot help here: a score that "
        "depends only on the date gives exactly 0.5. This is the strictest person-level comparison; it uses the "
        "insider's window, so it answers 'during the malicious period, does the insider outscore everyone else "
        "on those same days?'. 'Coverage' is the share of insider-benign pairs with enough benign days to compare.\n")
    for training in args.training:
        keys = [k for k in day_scores if meta[k][1] == training]
        reference = f"gradient_boosting/{training}" if f"gradient_boosting/{training}" in keys else None
        res = matched_period_auroc(users, days, {k: day_scores[k] for k in keys}, windows,
                                   n_boot=args.n_boot, seed=args.seed, reference=reference)
        results.setdefault("matched_period", {})[training] = res
        md.append(f"#### {TRAINING_TEXT[training]}\n")
        md.append("| Model | matched-period AUROC [95% CI] | coverage |" +
                  (" minus gradient boosting [95% CI] |" if reference else ""))
        md.append("|---|---|---|" + ("---|" if reference else ""))
        for k in keys:
            row = f"| {names[k]} | {_fmt_ci(res[k])} | {100 * res[k]['coverage']:.0f}% |"
            if reference:
                if k == reference:
                    row += " (reference) |"
                elif "auroc_minus_reference" in res[k]:
                    d = res[k]["auroc_minus_reference"]
                    row += f" {d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{' *' if d['ci_excludes_zero'] else ''} |"
                else:
                    row += " n/a |"
            md.append(row)
        md.append("\n`*` = the interval excludes 0.\n")

    md.append("## Of the highest-ranked users, how many are insiders? (rule: top_k_mean)\n")
    md.append("| Model | top 5 | top 10 | insiders among ALL users (random pick) |\n|---|---|---|---|")
    for k in day_scores:
        if meta[k][2] is not None:
            continue
        p = precision_at_k(user_scores(users, day_scores[k]), insiders, ks=(5, 10))
        results["precision_at_k"][k] = p
        md.append(f"| {names[k]} | {p[5]['insiders_in_top_k']} of {p[5]['k']} | {p[10]['insiders_in_top_k']} of "
                  f"{p[10]['k']} | {100 * p[5]['baseline_share']:.0f}% |")
    md.append("")

    md.append(f"## Alert simulation: cutoff = {int(ALERT_QUANTILE * 100)}th percentile of benign users' day scores\n")
    md.append("| Model | insiders detected in their window | median days to first alert | detected within the first week "
              "| insiders alerted BEFORE their window | benign users alerted | alert days per benign user |\n|---|---|---|---|---|---|---|")
    alert_results = {}
    for k in day_scores:
        a = alert_analysis(users, days, day_scores[k], windows)
        alert_results[k] = a
        s = a["summary"]
        results["alerts"][k] = {"threshold": a["threshold"], **s}
        md.append(f"| {names[k]} | {s['insiders_detected']} of {s['n_insiders']} ({100 * s['detection_rate']:.0f}%) | "
                  f"{s['median_days_to_first_alert']:.0f} | {100 * s['share_detected_within_7_days']:.0f}% of insiders | "
                  f"{s['insiders_alerted_before_window']} of {s['n_insiders']} | {s['benign_users_alerted']} of "
                  f"{s['n_benign']} | {s['benign_alert_days_per_user']:.1f} |")
    md.append("\nA control that 'detects' insiders as well as the real model means the alert pattern does not need "
              "behaviour. 'Alerted before their window' are alerts on days the insider had done nothing "
              "scripted yet: they count as noise, not as early warning.\n")

    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    activity = [c for c in ACTIVITY_COLUMNS if c in table.columns]
    primary = f"{args.primary_model}/{args.training[0]}"
    for key in day_scores:
        model, training, control = meta[key]
        if control is not None:
            continue
        frame = table[["user", "day"] + activity].copy()
        frame["score"] = day_scores[key]
        report = ranked_user_table(user_scores(users, day_scores[key]), alert_results[key], scenario_by_user,
                                   frame=frame, activity_columns=activity)
        path = os.path.join(out_dir, f"detector_users_{args.tag}_{model}_{training}.csv")
        report.to_csv(path, index=False)
        if key == primary:
            md.append(f"## Ranked users, {names[key]} (top 15 of {len(report)}; full list in the CSV)\n")
            md.append("`is_insider` and `scenario` come from the answer key and are shown for EVALUATION only; a real "
                      "deployment would not know them. The activity shown is the user's raw activity on that day, not "
                      "a model explanation.\n")
            md.append("| rank | user | user score | alert days | insider (scenario) | most suspicious days |\n|---|---|---|---|---|---|")
            for _, r in report.head(15).iterrows():
                md.append(f"| {r['rank']} | {r['user']} | {r['user_score']:.3g} | {r['n_alert_days']} | "
                          f"{'yes (' + str(r['scenario']) + ')' if r['is_insider'] else 'no'} | {r.get('top_days', '')} |")
            md.append("")
    md.append("_Development-partition numbers. Rules and the alert cutoff were fixed before looking at these results, "
              "but the models and features were developed on these users, so they are optimistic. The locked partition "
              "has NOT been used._\n")

    stem = f"detector_dev_{args.tag}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    markdown = "\n".join(md)
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write(markdown)
    print("\n" + markdown)
    print(f"Saved {os.path.join(out_dir, stem)}.json and .md, plus the ranked-user CSVs")


if __name__ == "__main__":
    main()
