"""
The evaluation that produces the thesis results table.

Takes one or more training runs (the same feature configuration, normally
different seeds), scores every registered model on one test partition, and
writes a results table with:
  * AUROC / AUPRC as the mean over seeds, with 95% USER-LEVEL bootstrap
    intervals;
  * a paired comparison against the single-feature heuristic baseline;
  * BOTH labellings side by side (window labels and answer-key labels --
    see src/common/evaluation/labelings.py for why both, and what to disclose);
  * the heuristic baselines (single http feature, session length) next to
    the trained models.

LOCKED PARTITION: the locked test users are for ONE final evaluation of a
method that was fixed beforehand. Reading them requires --confirm-final.
Whatever the locked result is, it is what gets reported.

Examples (run from the project root, once per feature configuration):

    # development partition, three seeds, all feature groups
    python scripts/unsupervised/final_evaluation.py --runs checkpoints/all_seed1 checkpoints/all_seed2 checkpoints/all_seed3

    # the final, once-only run on the locked partition
    python scripts/unsupervised/final_evaluation.py --runs checkpoints/all_seed1 checkpoints/all_seed2 checkpoints/all_seed3 \\
        --partition locked --confirm-final

    # an ablation: runs trained with INSIDER_FEATURE_GROUPS=file,device,logon must be
    # evaluated with the same variable set
    INSIDER_FEATURE_GROUPS=file,device,logon python scripts/unsupervised/final_evaluation.py \\
        --runs checkpoints/nohttp_seed1 checkpoints/nohttp_seed2 --tag nohttp

Outputs (in --out-dir, default results/unsupervised/): final_evaluation_<partition>_<tag>.json and .md
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
from src.common.evaluation.final_report import evaluate_labeling, format_markdown
from src.common.evaluation.labelings import (
    ANSWER_KEY_LABELING,
    WINDOW_LABELING,
    answer_key_session_labels,
    load_answer_key_events,
)
from src.common.evaluation.partition import get_partition_name, select_test_df
from src.unsupervised.evaluation.scoring import (
    has_checkpoint,
    load_run_artifacts,
    run_seed,
    score_sessions,
    session_table,
)
from src.unsupervised.models.registry import get_model_specs
from src.common.preprocessing.labeling import load_malicious_windows
from src.unsupervised.training.feature_config import current_feature_config

DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "file_events_processed.parquet")
DEFAULT_REFERENCE = "single_feature_http"

LABELING_DESCRIPTIONS = {
    WINDOW_LABELING: (
        "A session is malicious if it falls inside its user's insider time window "
        "(the original labelling)."
    ),
    ANSWER_KEY_LABELING: (
        "A session is malicious if it has a file event on a day with an answer-key http event "
        "for that insider, or during an answer-key removable-device session. Introduced after "
        "seeing the window-labelled results (window labels dilute scenario 2); used for "
        "evaluation only -- no feature or model uses the answer key."
    ),
}


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _check_same_feature_config(run_dirs):
    configs = {}
    for run in run_dirs:
        path = os.path.join(run, "feature_config.json")
        if not os.path.exists(path):
            raise SystemExit(f"{run} has no feature_config.json: it was not produced by the current "
                             "train_models.py. Retrain it.")
        with open(path) as f:
            configs[run] = json.load(f)["numeric_columns"]
    first = next(iter(configs.values()))
    different = [r for r, c in configs.items() if c != first]
    if different:
        raise SystemExit(
            f"These runs were trained with different feature configurations than the others: {different}. "
            "Evaluate each feature configuration separately (use --tag), with INSIDER_FEATURE_GROUPS set to match."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run folders (same feature configuration)")
    parser.add_argument("--partition", choices=("dev", "locked", "all"), default=None,
                        help="test partition (default: INSIDER_TEST_PARTITION, else dev)")
    parser.add_argument("--confirm-final", action="store_true",
                        help="required to read the locked (or all) partition")
    parser.add_argument("--out-dir", default="results/unsupervised")
    parser.add_argument("--tag", default="all", help="label for this configuration in the output file names")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="model for the paired AUROC comparison")
    parser.add_argument("--models", nargs="*", default=None, help="subset of registered models")
    args = parser.parse_args()

    partition = args.partition or get_partition_name()
    if partition in ("locked", "all") and not args.confirm_final:
        raise SystemExit(
            f"Refusing to read the '{partition}' partition without --confirm-final. The locked insiders "
            "are for ONE evaluation of a method fixed in advance; whatever the result is, it is reported."
        )

    run_dirs = [_resolve(r) for r in args.runs]
    for run in run_dirs:
        if not os.path.isdir(run):
            raise SystemExit(f"Run folder not found: {run}")
    _check_same_feature_config(run_dirs)

    print(f"Loading {DATA_PATH} ...")
    events = pd.read_parquet(DATA_PATH)
    test_df = select_test_df(events, partition=partition)

    scenario_map = load_malicious_windows().set_index("user")["scenario"].to_dict()
    table = session_table(test_df)
    users = table["user"].to_numpy()
    scenario_of_session = np.array([scenario_map.get(u, 0) for u in users])
    insider_mask = np.array([u in scenario_map for u in users])

    labelings = {WINDOW_LABELING: table["window_label"].to_numpy()}
    key = load_answer_key_events()
    labelings[ANSWER_KEY_LABELING] = answer_key_session_labels(test_df, key)
    for name, labels in labelings.items():
        print(f"labelling {name!r}: {int(labels.sum())} malicious of {len(labels)} sessions")

    # ---- score every model on the chosen partition ----
    specs = get_model_specs(args.models)
    artifacts = {run: load_run_artifacts(run) for run in run_dirs}  # also verifies feature configs
    score_matrices, notes, display_names = {}, [], {}
    for spec in specs:
        display_names[spec.name] = spec.display_name
        missing_cols = [c for c in spec.required_columns if c not in events.columns]
        if missing_cols:
            notes.append(f"{spec.name}: skipped (dataset lacks {missing_cols})")
            continue
        if spec.kind == "heuristic":
            score_matrices[spec.name] = score_sessions(spec, test_df)[None, :]
            continue
        runs_with = [r for r in run_dirs if has_checkpoint(spec, r)]
        if not runs_with:
            notes.append(f"{spec.name}: skipped (no checkpoint in any run)")
            continue
        if len(runs_with) < len(run_dirs):
            notes.append(f"{spec.name}: only {len(runs_with)} of {len(run_dirs)} runs have a checkpoint")
        print(f"scoring {spec.name} on {len(runs_with)} run(s) ...")
        score_matrices[spec.name] = np.stack(
            [score_sessions(spec, test_df, r, artifacts[r]) for r in runs_with]
        )
    for note in notes:
        print(f"[note] {note}")
    if not score_matrices:
        raise SystemExit("No model could be scored.")

    # ---- evaluate under each labelling, on the SAME user resamples ----
    resamples = make_user_resamples(users, n_boot=args.n_boot, seed=0)
    reference = args.reference if args.reference in score_matrices else None
    if reference is None:
        print(f"[note] reference model {args.reference!r} unavailable: no paired comparison")

    results = {
        "partition": partition,
        "runs": [os.path.relpath(r, _PROJECT_ROOT) for r in run_dirs],
        "seeds": [run_seed(r) for r in run_dirs],
        "feature_config": current_feature_config(),
        "git_commit": _git_commit(),
        "n_bootstrap_resamples": args.n_boot,
        "n_test_users": int(len(np.unique(users))),
        "notes": notes,
        "labelings": {},
    }
    markdown = [
        f"# Final evaluation: partition `{partition}`, configuration `{args.tag}`",
        "",
        f"Runs: {results['runs']} (seeds {results['seeds']}). "
        f"Feature groups: {results['feature_config']['enabled_groups']}. "
        f"Commit: `{results['git_commit']}`. {args.n_boot} user-level bootstrap resamples.",
        "",
    ]
    for name, labels in labelings.items():
        if labels.sum() == 0:
            print(f"[note] labelling {name!r} has no malicious sessions on this partition: skipped")
            continue
        results["labelings"][name] = evaluate_labeling(
            labels, scenario_of_session, insider_mask, score_matrices, resamples, reference=reference
        )
        markdown.append(format_markdown(name, LABELING_DESCRIPTIONS[name], results["labelings"][name], display_names))
    if notes:
        markdown += ["Notes: " + "; ".join(notes), ""]
    if partition == "dev":
        markdown += [
            "_These are DEVELOPMENT-partition numbers. Several features and diagnostics were chosen "
            "while looking at this partition, so they are optimistic; the locked partition is the "
            "honest estimate._", "",
        ]

    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"final_evaluation_{partition}_{args.tag}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write("\n".join(markdown))

    print("\n" + "\n".join(markdown))
    print(f"Saved {os.path.join(out_dir, stem)}.json and .md")


if __name__ == "__main__":
    main()
