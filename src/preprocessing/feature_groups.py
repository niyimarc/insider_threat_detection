"""
Single source of truth for the model's numeric input features, and for
which SIGNAL (log source) each one comes from.

Every numeric feature the models consume is registered below as a
(column_name, signal_group) pair. A signal group corresponds to one raw
log source or one family of features derived from it:

    file    file.csv        file-copy timing, session volume, session frequency
    device  device.csv      removable-device connection features
    logon   logon.csv       working-hours behaviour

Adding a new signal (http, email, ...) is a three-step change:
  1. write src/preprocessing/<signal>_features.py, one function per
     feature, each adding its column(s) to the file-event DataFrame;
  2. append (column, "<signal>") entries to NUMERIC_FEATURE_REGISTRY;
  3. call the new function in scripts/build_dataset.py.
Nothing else needs to change: the dataset, the three models, training
and evaluation all derive their input size and column list from here.

RULES
  * The registry is APPEND-ONLY. The order of NUMERIC_FEATURE_COLUMNS is
    the order of the model's input vector, so reordering or inserting in
    the middle silently changes what every saved checkpoint's weights
    mean (two columns swapped have identical tensor shapes, so loading
    would not even raise an error).
  * The "file" group is mandatory: the research question is file-access
    behaviour, and the Transformer's presence channel is
    `seconds_since_last_event`. Other groups can be switched off, which
    is how ablations (file only / file+device / ...) are run.

SELECTING GROUPS
  By default every registered group is enabled. To train and evaluate a
  subset, set the environment variable before running any script:

      INSIDER_FEATURE_GROUPS=file,device python scripts/train_models.py

  (PowerShell:  $env:INSIDER_FEATURE_GROUPS="file,device")
  The selection is recorded next to the checkpoints (see
  src/training/feature_config.py) and verified when they are loaded, so
  a model can never be evaluated against a different feature set than
  it was trained on.
"""

import os

FEATURE_GROUPS_ENV_VAR = "INSIDER_FEATURE_GROUPS"

# APPEND-ONLY -- see module docstring.
NUMERIC_FEATURE_REGISTRY = (
    ("seconds_since_last_event", "file"),
    ("seconds_since_device_connect", "device"),
    ("copies_since_device_connect", "device"),
    ("outside_working_hours", "logon"),
    ("user_session_volume_deviation", "file"),
    ("user_session_frequency_deviation", "file"),
)

# Columns that are already 0/1 flags: not standardised, and summarised
# as "fraction of events flagged" (rather than mean/std/min/max) by the
# static-vector baseline.
BINARY_NUMERIC_COLUMNS = ("outside_working_hours",)

# Groups that can never be switched off.
MANDATORY_GROUPS = ("file",)

# The column the Transformer's "presence" channel reads. It belongs to a
# mandatory group, so it is always present in NUMERIC_FEATURE_COLUMNS.
PRESENCE_COLUMN = "seconds_since_last_event"


def available_groups():
    """All registered signal groups, in order of first appearance."""
    groups = []
    for _, group in NUMERIC_FEATURE_REGISTRY:
        if group not in groups:
            groups.append(group)
    return tuple(groups)


def resolve_groups(groups=None):
    """
    Returns the enabled signal groups as a tuple in canonical (registry)
    order. `groups` may be an iterable of group names; if None, the
    INSIDER_FEATURE_GROUPS environment variable is used, and if that is
    unset or empty, every registered group is enabled.

    Raises ValueError for an unknown group name or if a mandatory group
    is missing -- a typo must never silently drop a signal.
    """
    known = available_groups()

    if groups is None:
        raw = os.environ.get(FEATURE_GROUPS_ENV_VAR, "").strip()
        groups = [g.strip() for g in raw.split(",") if g.strip()] if raw else list(known)

    groups = list(groups)
    unknown = sorted(set(groups) - set(known))
    if unknown:
        raise ValueError(
            f"Unknown feature group(s) {unknown}. Registered groups: {list(known)}."
        )
    missing = [g for g in MANDATORY_GROUPS if g not in groups]
    if missing:
        raise ValueError(
            f"Mandatory feature group(s) {missing} cannot be disabled "
            f"(requested: {groups})."
        )
    return tuple(g for g in known if g in groups)


def get_numeric_feature_columns(groups=None):
    """Numeric model-input columns for the enabled groups, in registry order."""
    enabled = resolve_groups(groups)
    return [column for column, group in NUMERIC_FEATURE_REGISTRY if group in enabled]


def get_scaled_columns(groups=None):
    """Columns that get standardised (everything except the 0/1 flags)."""
    return [c for c in get_numeric_feature_columns(groups) if c not in BINARY_NUMERIC_COLUMNS]


def get_binary_columns(groups=None):
    """The enabled 0/1 flag columns, in registry order."""
    return [c for c in get_numeric_feature_columns(groups) if c in BINARY_NUMERIC_COLUMNS]


def feature_group_of(column):
    """The signal group a registered column belongs to."""
    for registered_column, group in NUMERIC_FEATURE_REGISTRY:
        if registered_column == column:
            return group
    raise KeyError(f"'{column}' is not a registered numeric feature.")


def require_columns(df, groups=None):
    """
    Raises a clear error if `df` lacks any column the enabled groups
    need. Typical cause: the processed dataset was built before a
    feature was added to the registry -- rebuild it with
    scripts/build_dataset.py.
    """
    needed = get_numeric_feature_columns(groups)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        by_group = {c: feature_group_of(c) for c in missing}
        raise ValueError(
            f"Dataset is missing registered feature column(s) {by_group}. "
            "It was probably built before these features existed -- "
            "re-run scripts/build_dataset.py."
        )
