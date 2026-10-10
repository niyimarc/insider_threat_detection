"""
LDAP: the monthly staff lists of r4.2 (data/raw/r4.2/LDAP/YYYY-MM.csv, one row per employee on the
payroll that month, with role, team, department, supervisor ...).

Only what the peer baselines need is read: who was employed in which month (people who leave simply
stop appearing) and each person's role and team. Nothing here uses the answer key.
"""

import os

import pandas as pd

from src.common.preprocessing.load import DEFAULT_RAW_DIR

LDAP_DIRNAME = "LDAP"
LDAP_COLUMNS = ("user", "role", "team", "month")


def load_ldap_snapshots(raw_dir=DEFAULT_RAW_DIR):
    """
    DataFrame(user, role, team, month): one row per employee per monthly snapshot, `month` a
    pandas Period('M') taken from the file name. Raises if the folder or a needed column is missing.
    """
    folder = os.path.join(raw_dir, LDAP_DIRNAME)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"LDAP folder not found: {folder}")
    frames = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv"):
            continue
        frame = pd.read_csv(os.path.join(folder, name), usecols=["user_id", "role", "team"])
        frame = frame.rename(columns={"user_id": "user"})
        frame["month"] = pd.Period(name[: -len(".csv")], freq="M")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No LDAP csv files in {folder}")
    return pd.concat(frames, ignore_index=True)[list(LDAP_COLUMNS)]
