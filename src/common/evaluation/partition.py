"""
Selects which part of the TEST users an evaluation script looks at.

The test users are split into a "dev" and a "locked" partition when the
dataset is built (see split_test_partition in src/common/preprocessing/split.py).
Everything that is done while designing the method -- choosing features,
ablations, diagnostics -- must use the dev partition. The locked
partition is for the single final evaluation.

Default is "dev", so the locked partition can't be used by accident:
reading it requires setting INSIDER_TEST_PARTITION=locked explicitly.
"""

import os

PARTITION_ENV_VAR = "INSIDER_TEST_PARTITION"
VALID_PARTITIONS = ("dev", "locked", "all")


def get_partition_name():
    name = os.environ.get(PARTITION_ENV_VAR, "dev").strip().lower()
    if name not in VALID_PARTITIONS:
        raise ValueError(
            f"{PARTITION_ENV_VAR}={name!r} is not one of {VALID_PARTITIONS}"
        )
    return name


def select_test_df(events_df, partition=None):
    """Returns the test-split rows of the requested partition (default:
    taken from the environment, which itself defaults to 'dev')."""
    partition = partition or get_partition_name()
    if partition not in VALID_PARTITIONS:
        raise ValueError(f"partition must be one of {VALID_PARTITIONS}, got {partition!r}")

    test_df = events_df[events_df["split"] == "test"]
    if partition == "all":
        selected = test_df
    else:
        if "test_partition" not in events_df.columns:
            raise RuntimeError(
                "The processed dataset has no 'test_partition' column -- it was built "
                "before the dev/locked split existed. Re-run scripts/common/build_dataset.py."
            )
        selected = test_df[test_df["test_partition"] == partition]

    print(f"[evaluation] test partition = {partition!r}: "
          f"{selected['user'].nunique()} users, {len(selected)} events")
    if partition in ("locked", "all"):
        print(f"[evaluation] WARNING: partition {partition!r} includes the locked insiders -- "
              f"use only for the final reported evaluation, never for design decisions.")
    return selected
