"""
A CLEAN EXTRA TEST of the frozen detector's false alarms, on benign users it has never scored.

The locked test had only 15 benign users, so its false-alarm figures were coarse (gradient boosting alerted
10 of 15, the GRU 2 of 15). About 165 further benign users with file events belong to the TRAINING and
VALIDATION users of the unsupervised phase (split in {train, validation}); the supervised work never trained on,
tuned on or scored them. This script scores them with the frozen detector, exactly as frozen:

  * the two frozen models (gradient boosting, GRU; the 40 v1 columns; date-matched training; seeds 0, 1, 2) are
    trained on the 50 DEVELOPMENT users only;
  * the alert cutoff is the SAVED frozen cutoff (99th percentile of development benign out-of-fold scores) from
    final_detector_locked_<tag>.json, not recomputed or adjusted;
  * the locked users' events are NOT read and are not scored again; their saved per-user results (the CSVs of the
    locked run) are used only for the side-by-side comparison and the user-level AUROC against a larger benign group.

Nothing about the method changes, so this is a continuation of the frozen test, not a new method. The persistence rule
is shown as a secondary, post-frozen variant and is labelled as such. These users all copied files to removable
media at least once (they have file events), so they are not a random sample of ALL benign employees.

    python scripts/supervised/extra_benign_false_alarms.py

Outputs (default results/supervised/): extra_benign_false_alarms_<tag>.md / .json and
extra_benign_users_<tag>_<model>.csv
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
from src.supervised.alert_rules import apply_alert_rule, wilson_interval
from src.supervised.all_days import build_all_user_days
from src.supervised.dataset import add_calendar_controls
from src.supervised.detector import align_calendar_origin, fit_and_score
from src.supervised.experiment import ALL_CONFIG, GROUP_CONFIGS
from src.supervised.feature_sets import ALL_DAYS
from src.supervised.models import DISPLAY_NAMES
from src.supervised.user_level import TOP_K_DAYS, user_level_auroc, user_scores

FROZEN_MODELS = ("gradient_boosting", "gru")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
DATA_PATH = os.path.join(PROCESSED_DIR, "file_events_processed.parquet")
STRATA_DAYS = 7


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _share(k, n):
    low, high = wilson_interval(k, n)
    return f"{k} of {n} ({100 * k / n:.0f}%, 95% CI {100 * low:.0f}% to {100 * high:.0f}%)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out-dir", default="results/supervised")
    parser.add_argument("--frozen-tag", default="locked", help="tag of the saved locked-run results to read")
    parser.add_argument("--tag", default="extra_benign")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    frozen_json = os.path.join(out_dir, f"final_detector_locked_{args.frozen_tag}.json")
    if not os.path.exists(frozen_json):
        sys.exit(f"{frozen_json} is missing: the frozen cutoffs come from the saved locked-run results.")
    with open(frozen_json) as f:
        frozen = json.load(f)
    thresholds = frozen["alert_thresholds_from_dev"]

    print(f"Loading {DATA_PATH} ...")
    events = pd.read_parquet(DATA_PATH)
    dev_events = select_test_df(events, partition="dev")
    extra_events = events[events["split"].isin(["train", "validation"])]
    windows_df = load_malicious_windows()
    scenario_by_user = windows_df.set_index("user")["scenario"].to_dict()
    dev_users, extra_users = set(dev_events["user"].unique()), set(extra_events["user"].unique())
    if extra_users & dev_users or extra_users & set(scenario_by_user):
        sys.exit("Extra users overlap the development users or include an insider: stop.")
    print(f"{len(extra_users)} extra benign users with file events; {len(dev_users)} development users for training.")

    print("Loading the raw logs and the http features ...")
    logon, device = load_logon_events(), load_device_events()
    http_user_day = build_http_user_day_features(load_http_domain_daily(PROCESSED_DIR))
    key_days = answer_key_labeled_days(load_answer_key_events())

    def build(user_events, users):
        return build_all_user_days(logon, device, user_events, http_user_day,
                                   windows_df[windows_df["user"].isin(users)], key_days, users)

    dev_table = add_calendar_controls(build(dev_events, dev_users))
    extra_table = align_calendar_origin(add_calendar_controls(build(extra_events, extra_users)), dev_table)
    users, days = extra_table["user"].to_numpy(), extra_table["day"].to_numpy()
    n_users = len(np.unique(users))
    print(f"Extra benign table: {n_users} users, {len(extra_table)} user-days")

    columns = ALL_DAYS.columns_fn(GROUP_CONFIGS[ALL_CONFIG])
    results = {"commit": _git_commit(), "n_extra_benign_users": int(n_users), "n_user_days": int(len(extra_table)),
               "frozen_thresholds_used": thresholds, "seeds": list(args.seeds), "models": {}}
    os.makedirs(out_dir, exist_ok=True)
    md = [
        f"# Frozen detector: false alarms on {n_users} benign users never scored before (`{args.tag}`)\n",
        f"{n_users} benign users with file events (split train/validation of the unsupervised phase; never used by the "
        f"supervised work), {len(extra_table)} user-days. Frozen models (v1, {len(columns)} columns, date-matched, seeds "
        f"{list(args.seeds)}) trained on the 50 development users only; alert cutoff = the SAVED frozen cutoff from the locked "
        f"run. The locked users' events were not read or scored again. Commit `{results['commit']}`.\n",
        "These users all copied files to removable media at least once, so they are not a random sample of all benign "
        "employees. The persistence rows are a post-frozen variant, shown for information.\n",
    ]

    rows, auroc_rows = [], []
    for model in FROZEN_MODELS:
        print(f"[{model}] training on the development users and scoring the extra benign users ...")
        scores = fit_and_score(dev_table, extra_table, model, columns, seeds=tuple(args.seeds),
                               date_matched=True, strata_days=STRATA_DAYS)
        threshold = float(thresholds[model])
        single = scores >= threshold
        persist = apply_alert_rule(users, days, single, "persist_2of3")
        per_user = pd.DataFrame({"user": users, "single": single, "persist": persist}).groupby("user").agg(
            single_days=("single", "sum"), persist_days=("persist", "sum"), active_days=("single", "size"))
        score_series = user_scores(users, scores, k=TOP_K_DAYS)
        per_user["user_score"] = score_series.reindex(per_user.index)
        per_user.sort_values("user_score", ascending=False).reset_index().to_csv(
            os.path.join(out_dir, f"extra_benign_users_{args.tag}_{model}.csv"), index=False)

        locked_csv = os.path.join(out_dir, f"final_detector_users_{args.frozen_tag}_{model}.csv")
        locked = pd.read_csv(locked_csv) if os.path.exists(locked_csv) else None
        name = DISPLAY_NAMES[model]
        entry = {"threshold": threshold,
                 "extra_users_alerted_single": int((per_user["single_days"] > 0).sum()),
                 "extra_users_alerted_persist": int((per_user["persist_days"] > 0).sum()),
                 "extra_alert_days_per_user_single": float(per_user["single_days"].mean()),
                 "extra_alert_days_per_user_persist": float(per_user["persist_days"].mean()),
                 "extra_share_of_days_alerted_single": float(single.mean())}
        rows.append((name, "extra benign users, single-day rule (frozen)", _share(entry["extra_users_alerted_single"], n_users),
                     f"{entry['extra_alert_days_per_user_single']:.1f}", f"{per_user['single_days'].median():.0f}"))
        rows.append((name, "extra benign users, persistence rule (post-frozen)", _share(entry["extra_users_alerted_persist"], n_users),
                     f"{entry['extra_alert_days_per_user_persist']:.1f}", f"{per_user['persist_days'].median():.0f}"))
        if locked is not None:
            lb = locked[~locked["is_insider"]]
            entry["locked_benign_alerted"] = int((lb["n_alert_days"] > 0).sum())
            entry["locked_benign_n"] = int(len(lb))
            rows.append((name, "the 15 locked benign users, single-day rule (from the locked run)",
                         _share(entry["locked_benign_alerted"], len(lb)), f"{lb['n_alert_days'].mean():.1f}",
                         f"{lb['n_alert_days'].median():.0f}"))
            insiders = locked[locked["is_insider"]]
            insider_set = set(insiders["user"])
            extra_scores = score_series.rename("user_score")
            groups = {"34 locked insiders vs the extra benign users": extra_scores,
                      "34 locked insiders vs ALL benign never used to train (extra + 15 locked)": pd.concat(
                          [extra_scores, lb.set_index("user")["user_score"]])}
            for label, benign_series in groups.items():
                combined = pd.concat([insiders.set_index("user")["user_score"], benign_series])
                res = user_level_auroc({model: combined}, insider_set, n_boot=args.n_boot, seed=0)[model]
                n_b = int((~combined.index.isin(insider_set)).sum())
                auroc_rows.append((name, label, str(n_b), f"{res['auroc']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}]"))
                entry.setdefault("user_level_auroc", {})[label] = res
        results["models"][model] = entry

    md.append("## 1. How many benign users does the frozen detector alert?\n")
    md.append("| Model | Group and rule | Users alerted on at least one day | Alert days per user (mean) | (median) |\n|---|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    if auroc_rows:
        md.append("## 2. Whole-timeline person ranking against a larger benign group\n")
        md.append("Top-7-days rule, 34 locked insiders scored in the locked run against benign users the frozen detector "
                  "never trained on (95% user-level bootstrap).\n")
        md.append("| Model | Comparison | Benign users | User-level AUROC [95% CI] |\n|---|---|---|---|")
        for r in auroc_rows:
            md.append("| " + " | ".join(r) + " |")
        md.append("")
    md.append("_The cutoff is the frozen cutoff, set on development benign users. A high share of extra benign users alerted "
              "means the cutoff does not carry over to new people; a low share is evidence that it does._\n")

    stem = f"extra_benign_false_alarms_{args.tag}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    markdown = "\n".join(md)
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write(markdown)
    print("\n" + markdown)
    print(f"Saved {os.path.join(out_dir, stem)}.json / .md and the per-user CSVs")


if __name__ == "__main__":
    main()
