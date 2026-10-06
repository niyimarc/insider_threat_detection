import os
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANSWERS_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "data", "raw", "answers")
)


def load_malicious_windows(answers_dir=DEFAULT_ANSWERS_DIR, dataset=4.2):
    """Load insiders.csv and filter to just the given CERT release."""
    path = os.path.join(answers_dir, "insiders.csv")
    df = pd.read_csv(path)
    df = df[df["dataset"] == dataset].copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df[["scenario", "user", "start", "end"]].reset_index(drop=True)


def label_file_events(file_df, malicious_windows):
    """
    Add an `is_malicious` column to file_df: True for any row whose
    timestamp falls inside that user's labelled malicious window.
    """
    file_df = file_df.copy()
    file_df["is_malicious"] = False

    for _, window in malicious_windows.iterrows():
        mask = (
            (file_df["user"] == window["user"])
            & (file_df["date"] >= window["start"])
            & (file_df["date"] <= window["end"])
        )
        file_df.loc[mask, "is_malicious"] = True

    return file_df