"""
Supervised user-day detection, evaluated by user-level cross-validation on
the DEVELOPMENT partition.

Trains logistic regression and gradient boosting on user-day features with
the WINDOW label, cross-validated over users (stratified by scenario), for
four signal-group configurations (file; +device; +logon; +http), and reports
AUROC/AUPRC with 95% user-level bootstrap intervals under BOTH labellings,
next to two no-training heuristics. Also reports the http signal's paired
contribution for each model.

Deliberately dev-only: there is no locked-partition option here yet. The
locked partition is reserved for one final run of a frozen supervised
method, which will be added when the method is frozen.

    python scripts/supervised/cross_validate.py
    python scripts/supervised/cross_validate.py --models gradient_boosting --configs file all --tag quick

Outputs (default results/supervised/): supervised_cv_dev_<tag>.json and .md
"""

import argparse
import json
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.common.evaluation.labelings import answer_key_day_labels, load_answer_key_events
from src.common.evaluation.partition import select_test_df
from src.common.preprocessing.labeling import load_malicious_windows
from src.supervised.dataset import attach_labels, build_user_day_table
from src.supervised.experiment import GROUP_CONFIGS, run_experiment
from src.supervised.models import MODEL_NAMES

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")


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
    parser.add_argument("--n-repeats", type=int, default=3, help="independent fold assignments (seed-mean)")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strata-days", type=int, default=7,
                        help="width in days of the calendar strata for the date-matched evaluation")
    parser.add_argument("--out-dir", default="results/supervised")
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()

    print(f"Loading {DATA_PATH} ...")
    events = pd.read_parquet(DATA_PATH)
    dev_events = select_test_df(events, partition="dev")  # dev ONLY: never the locked users

    print("Building the user-day table ...")
    day_table = build_user_day_table(dev_events)
    key = load_answer_key_events()
    day_table = attach_labels(day_table, answer_key_day_labels(dev_events, key))
    scenario_by_user = load_malicious_windows().set_index("user")["scenario"].to_dict()

    n_insider_users = len({u for u in day_table["user"].unique() if u in scenario_by_user})
    print(f"{len(day_table)} user-days, {day_table['user'].nunique()} users ({n_insider_users} insiders); "
          f"window-malicious days: {int(day_table['window_label'].sum())}, "
          f"answer-key-malicious days: {int(day_table['answer_key_label'].sum())}")

    results, markdown = run_experiment(
        day_table, scenario_by_user,
        model_names=tuple(args.models), config_names=tuple(args.configs),
        n_folds=args.n_folds, n_repeats=args.n_repeats, n_boot=args.n_boot, seed=args.seed,
        strata_days=args.strata_days,
    )
    results["partition"] = "dev"
    results["git_commit"] = _git_commit()
    header = (
        f"# Supervised user-day detection: development partition, user-level cross-validation "
        f"(`{args.tag}`)\n\n"
        f"{results['n_users']} users, {results['n_rows']} user-days, {args.n_folds} folds x "
        f"{args.n_repeats} repeats, {args.n_boot} bootstrap resamples. Commit: `{results['git_commit']}`.\n"
    )
    markdown = header + "\n" + markdown

    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"supervised_cv_dev_{args.tag}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write(markdown)
    print("\n" + markdown)
    print(f"Saved {os.path.join(out_dir, stem)}.json and .md")


if __name__ == "__main__":
    main()
