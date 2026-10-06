"""
Run-level settings shared by every script: where a run's checkpoints live,
which random seed it uses, and a record of what produced it.

  INSIDER_CHECKPOINT_DIR   folder for this run's checkpoints and results.
                           Relative paths are relative to the project root.
                           Default: "checkpoints".
  INSIDER_SEED             random seed for training (default 42).

Why: ablations (with / without a signal group) and repeated runs with
different seeds each need their OWN folder, otherwise the second run
overwrites the first. Example (Colab):

    !INSIDER_CHECKPOINT_DIR=checkpoints/all_seed1 INSIDER_SEED=1 python scripts/unsupervised/train_models.py
    !INSIDER_CHECKPOINT_DIR=checkpoints/all_seed1 python scripts/unsupervised/evaluate_models.py

The training and evaluation scripts must be given the SAME
INSIDER_CHECKPOINT_DIR. Each folder also holds feature_config.json (which
feature groups the models were trained with, verified at load time) and
run_metadata.json (seed, library versions, code version).
"""

import json
import os
import random
import subprocess
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT_DIR_ENV_VAR = "INSIDER_CHECKPOINT_DIR"
SEED_ENV_VAR = "INSIDER_SEED"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_SEED = 42
RUN_METADATA_FILENAME = "run_metadata.json"


def get_checkpoint_dir():
    raw = os.environ.get(CHECKPOINT_DIR_ENV_VAR, "").strip() or DEFAULT_CHECKPOINT_DIR
    return raw if os.path.isabs(raw) else os.path.join(_PROJECT_ROOT, raw)


def get_seed():
    raw = os.environ.get(SEED_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_SEED
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{SEED_ENV_VAR}={raw!r} is not an integer")


def set_global_seed(seed):
    """Seeds python, numpy and torch (CPU and GPU). GPU kernels are not
    guaranteed bit-for-bit reproducible, so repeated runs with the same
    seed can still differ slightly on a GPU."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_run_metadata(checkpoint_dir, seed, device):
    """Writes run_metadata.json next to the checkpoints and returns its path."""
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("torch", "pandas", "sklearn"):
        try:
            module = __import__(name)
            versions[name] = module.__version__
        except Exception:
            versions[name] = "unavailable"

    from src.common.preprocessing.feature_groups import get_numeric_feature_columns, resolve_groups
    metadata = {
        "seed": seed,
        "device": device,
        "git_commit": _git_commit(),
        "enabled_groups": list(resolve_groups()),
        "numeric_columns": get_numeric_feature_columns(),
        "versions": versions,
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, RUN_METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path
