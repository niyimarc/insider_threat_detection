import pandas as pd
from src.common.preprocessing.labeling import load_malicious_windows, label_file_events


def test_malicious_windows_matches_known_counts():
    windows = load_malicious_windows()
    assert len(windows) == 70
    counts = windows["scenario"].value_counts().to_dict()
    assert counts == {1: 30, 2: 30, 3: 10}


def test_label_file_events_marks_correct_rows():
    # A small, hand-built DataFrame — not real data — so this test checks
    # only the labelling LOGIC, independent of what's actually in file.csv.
    file_df = pd.DataFrame({
        "user": ["AAA0001", "AAA0001", "AAA0001", "BBB0002"],
        "date": pd.to_datetime([
            "2010-06-01 10:00:00",  # before the window — should stay False
            "2010-06-05 12:00:00",  # inside the window — should become True
            "2010-06-10 09:00:00",  # after the window — should stay False
            "2010-06-05 12:00:00",  # different user, same date — should stay False
        ]),
    })
    windows = pd.DataFrame({
        "scenario": [1],
        "user": ["AAA0001"],
        "start": pd.to_datetime(["2010-06-03 00:00:00"]),
        "end": pd.to_datetime(["2010-06-07 00:00:00"]),
    })

    labeled = label_file_events(file_df, windows)

    assert list(labeled["is_malicious"]) == [False, True, False, False]