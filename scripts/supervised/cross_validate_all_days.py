"""
Supervised detection on the ALL-DAYS user-day table, evaluated by user-level
cross-validation on the DEVELOPMENT partition, with every confound control.

The table has a row for every day the user was active in the logon, device or
file log (not only days with a file copy), and adds day-level device and logon
signals with per-user baselines (src/supervised/day_signals.py). Everything
else is as in scripts/supervised/cross_validate.py: window labels for training,
both labellings reported, a calendar-only control, the within-insider
comparison, permutation importance, the date-matched evaluation, and the
decisive insiders-only date-stratified check.

Needs: the processed dataset (build_dataset.py), http_domain_daily.parquet
(precompute_http_daily.py), and the raw logon.csv / device.csv / answer key.
Dev only: there is deliberately no locked-partition option here.

    python scripts/supervised/cross_validate_all_days.py
    python scripts/supervised/cross_validate_all_days.py --models gradient_boosting --tag quick

Outputs (default results/supervised/): supervised_all_days_dev_<tag>.json and .md
"""

import argparse
import json
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.common.evaluation.labelings import answer_key_labeled_days, load_answer_key_events
from src.common.evaluation.partition import select_test_df
from src.common.preprocessing.http_features import build_http_user_day_features, load_http_domain_daily
from src.common.preprocessing.labeling import load_malicious_windows
from src.common.preprocessing.load import load_device_events, load_logon_events
from src.supervised.all_days import build_all_user_days
from src.supervised.experiment import GROUP_CONFIGS, run_experiment
from src.supervised.feature_sets import ALL_DAYS
from src.supervised.models import MODEL_NAMES

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
DATA_PATH = os.path.join(PROCESSED_DIR, "file_events_processed.parquet")


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--configs", nargs="+", default=list(GROUP_CONFIGS), choices=list(GROUP_CONFIGS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strata-days", type=int, default=7)
    parser.add_argument("--out-dir", default="results/supervised")
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()

    print(f"Loading {DATA_PATH} ...")
    events = pd.read_parquet(DATA_PATH)
    dev_events = select_test_df(events, partition="dev")  # dev ONLY: never the locked users
    dev_users = set(dev_events["user"].unique())

    print("Loading the raw logon and device logs ...")
    logon = load_logon_events()
    device = load_device_events()
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

    n_insiders = len({u for u in table["user"].unique() if u in scenario_by_user})
    print(f"{len(table)} user-days, {table['user'].nunique()} users ({n_insiders} insiders); "
          f"{100 * table['day_has_file_events'].mean():.1f}% of the days have a file copy; "
          f"window-malicious days: {int(table['window_label'].sum())}, "
          f"answer-key-malicious days: {int(table['answer_key_label'].sum())}")

    results, markdown = run_experiment(
        table, scenario_by_user,
        model_names=tuple(args.models), config_names=tuple(args.configs),
        n_folds=args.n_folds, n_repeats=args.n_repeats, n_boot=args.n_boot, seed=args.seed,
        strata_days=args.strata_days, feature_set=ALL_DAYS,
    )
    results["partition"] = "dev"
    results["table"] = "all_days"
    results["git_commit"] = _git_commit()
    header = (
        f"# Supervised ALL-DAYS detection: development partition, user-level cross-validation "
        f"(`{args.tag}`)\n\n"
        f"{results['n_users']} users, {results['n_rows']} user-days (every day with logon, device or file "
        f"activity), {args.n_folds} folds x {args.n_repeats} repeats, {args.n_boot} bootstrap resamples. "
        f"Commit: `{results['git_commit']}`.\n"
    )
    markdown = header + "\n" + markdown

    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"supervised_all_days_dev_{args.tag}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write(markdown)
    print("\n" + markdown)
    print(f"Saved {os.path.join(out_dir, stem)}.json and .md")


if __name__ == "__main__":
    main()
