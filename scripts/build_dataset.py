import os
import sys

# Add project root to path so `src` is importable when running this
# script directly (pytest handles this itself via pytest.ini, but a
# plain `python scripts/build_dataset.py` needs it done explicitly).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.load import load_file_events, load_logon_events, load_device_events
from src.preprocessing.labeling import load_malicious_windows, label_file_events
from src.preprocessing.features import (
    extract_file_type,
    extract_file_extension,
    add_time_since_previous_event,
    add_device_session_features,
    add_outside_hours_feature,
)
from src.preprocessing.sessionize import (
    build_all_sessions,
    assign_sessions_to_file_events,
    label_sessions,
    add_user_session_volume_deviation_feature,
)
from src.preprocessing.split import split_users


def build_dataset():
    print("Loading raw data...")
    file_df = load_file_events()
    logon_df = load_logon_events()
    device_df = load_device_events()

    print("Labelling malicious events...")
    malicious_windows = load_malicious_windows()
    file_df = label_file_events(file_df, malicious_windows)

    print("Engineering features...")
    file_df = extract_file_type(file_df)
    file_df = extract_file_extension(file_df)
    file_df = add_time_since_previous_event(file_df)
    file_df = add_device_session_features(file_df, device_df)
    file_df = add_outside_hours_feature(file_df, logon_df)

    print("Building sessions...")
    sessions_df = build_all_sessions(logon_df)
    file_df = assign_sessions_to_file_events(file_df, sessions_df)
    file_df = label_sessions(file_df)
    file_df = add_user_session_volume_deviation_feature(file_df)

    print("Splitting users...")
    all_users = file_df["user"].unique()
    malicious_users = malicious_windows["user"].unique()
    train_users, val_users, test_users = split_users(all_users, malicious_users)

    file_df["split"] = "unassigned"
    file_df.loc[file_df["user"].isin(train_users), "split"] = "train"
    file_df.loc[file_df["user"].isin(val_users), "split"] = "validation"
    file_df.loc[file_df["user"].isin(test_users), "split"] = "test"
    assert (file_df["split"] != "unassigned").all(), "Some users were not assigned a split"

    print("Saving processed data...")
    output_dir = os.path.join(_PROJECT_ROOT, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "file_events_processed.parquet")
    file_df.to_parquet(output_path, index=False)

    print(f"\nDone. {len(file_df)} events saved to {output_path}")
    print(file_df["split"].value_counts())


if __name__ == "__main__":
    build_dataset()