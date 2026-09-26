import os
import pandas as pd

# Location of the raw r4.2 data, computed relative to this file's own
# location rather than the caller's working directory — same reasoning
# as the notebooks scripts, just inside a package now instead of a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "data", "raw", "r4.2")
)


def _load_csv_with_dates(filename, raw_dir, expected_columns):
    path = os.path.join(raw_dir, filename)
    df = pd.read_csv(path)

    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing expected columns: {missing}")

    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_file_events(raw_dir=DEFAULT_RAW_DIR):
    """Load file.csv: one row per file copied to removable media."""
    return _load_csv_with_dates(
        "file.csv", raw_dir,
        expected_columns=["id", "date", "user", "pc", "filename", "content"],
    )


def load_logon_events(raw_dir=DEFAULT_RAW_DIR):
    """Load logon.csv: one row per Logon/Logoff event."""
    return _load_csv_with_dates(
        "logon.csv", raw_dir,
        expected_columns=["id", "date", "user", "pc", "activity"],
    )


def load_device_events(raw_dir=DEFAULT_RAW_DIR):
    """Load device.csv: one row per Connect/Disconnect event."""
    return _load_csv_with_dates(
        "device.csv", raw_dir,
        expected_columns=["id", "date", "user", "pc", "activity"],
    )