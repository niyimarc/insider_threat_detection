"""
Records which feature groups / columns a set of checkpoints was trained
with, and verifies they match when the checkpoints are loaded again.

Why this exists: the models' input size and column order come from the
feature registry (src/common/preprocessing/feature_groups.py). If the registry
or the INSIDER_FEATURE_GROUPS selection changes between training and
evaluation, a checkpoint can fail to load with an unhelpful tensor-size
error -- or, worse, load fine when two columns merely swap places (same
shapes, different meaning) and silently produce wrong results. Saving
the configuration next to the weights turns both into one clear error.

Same principle as src/unsupervised/training/vocab.py and scaling.py: whatever training
decided is saved with the checkpoint and reused, never re-derived.
"""

import json
import os
import warnings

from src.common.preprocessing.feature_groups import resolve_groups, get_numeric_feature_columns

FEATURE_CONFIG_FILENAME = "feature_config.json"


def current_feature_config():
    """The feature configuration in effect right now."""
    return {
        "enabled_groups": list(resolve_groups()),
        "numeric_columns": get_numeric_feature_columns(),
    }


def save_feature_config(checkpoint_dir):
    """Writes the current feature configuration next to the checkpoints."""
    path = os.path.join(checkpoint_dir, FEATURE_CONFIG_FILENAME)
    with open(path, "w") as f:
        json.dump(current_feature_config(), f, indent=2)
    return path


def verify_feature_config(checkpoint_dir):
    """
    Raises RuntimeError if the checkpoints in `checkpoint_dir` were
    trained with a different set or order of numeric feature columns
    than the current configuration.

    Checkpoints saved before this safeguard existed have no config file.
    That is reported as a warning, not an error, since they remain valid
    for the original feature set; retrain to get a verified checkpoint.
    """
    path = os.path.join(checkpoint_dir, FEATURE_CONFIG_FILENAME)
    if not os.path.exists(path):
        warnings.warn(
            f"No {FEATURE_CONFIG_FILENAME} in {checkpoint_dir}: cannot verify that these "
            "checkpoints match the current feature configuration. Re-run "
            "scripts/unsupervised/train_models.py to record it."
        )
        return

    with open(path) as f:
        saved = json.load(f)
    current = current_feature_config()

    if saved["numeric_columns"] != current["numeric_columns"]:
        raise RuntimeError(
            "Feature configuration mismatch between these checkpoints and the "
            "current settings.\n"
            f"  trained with: groups={saved['enabled_groups']} columns={saved['numeric_columns']}\n"
            f"  current:      groups={current['enabled_groups']} columns={current['numeric_columns']}\n"
            "Set INSIDER_FEATURE_GROUPS to the groups the checkpoints were trained "
            "with, or retrain with scripts/unsupervised/train_models.py."
        )
