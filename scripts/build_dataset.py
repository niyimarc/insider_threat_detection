import os
import sys

# Add project root to path so `src` is importable when running this
# script directly (pytest handles this itself via pytest.ini, but a
# plain `python scripts/build_dataset.py` needs it done explicitly).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.load import load_file_events, load_logon_events, load_device_events
from src.preprocessing.labeling import load_malicious_windows, label_file_events
from src.preprocessing.file_features import (
    extract_file_type,
    extract_file_extension,
    add_time_since_previous_event,
)
from src.preprocessing.device_features import add_device_session_features
from src.preprocessing.logon_features import add_outside_hours_feature
from src.preprocessing.http_features import (
    load_http_domain_daily,
    build_http_user_day_features,
    add_http_features,
)
from src.preprocessing.sessionize import (
    build_all_sessions,
    assign_sessions_to_file_events,
    label_sessions,
)
from src.preprocessing.file_session_features import (
    add_user_session_volume_deviation_feature,
    add_user_session_frequency_deviation_feature,
)
from src.preprocessing.feature_groups import (
    NUMERIC_FEATURE_REGISTRY,
    available_groups,
    require_columns,
)
from src.preprocessing.split import split_users, split_test_partition


def build_dataset():
    print("Loading raw data...")
    file_df = load_file_events()
    logon_df = load_logon_events()
    device_df = load_device_events()

    print("Labelling malicious events...")
    malicious_windows = load_malicious_windows()
    file_df = label_file_events(file_df, malicious_windows)

    # Each signal's features are added by its own module. To add a new
    # signal (e.g. http): write src/preprocessing/http_features.py,
    # register its columns in src/preprocessing/feature_groups.py, and
    # call it here -- see the feature_groups module docstring.
    print("Engineering features...")
    # signal group "file"
    file_df = extract_file_type(file_df)
    file_df = extract_file_extension(file_df)
    file_df = add_time_since_previous_event(file_df)
    # signal group "device"
    file_df = add_device_session_features(file_df, device_df)
    # signal group "logon"
    file_df = add_outside_hours_feature(file_df, logon_df)
    # signal group "http" (from the per-user-day tables made by
    # scripts/precompute_http_daily.py)
    http_user_day = build_http_user_day_features(
        load_http_domain_daily(os.path.join(_PROJECT_ROOT, "data", "processed"))
    )
    file_df = add_http_features(file_df, http_user_day)

    print("Building sessions...")
    sessions_df = build_all_sessions(logon_df)
    file_df = assign_sessions_to_file_events(file_df, sessions_df)
    file_df = label_sessions(file_df)
    # signal group "file" (session level -- needs sessions to exist)
    file_df = add_user_session_volume_deviation_feature(file_df)
    file_df = add_user_session_frequency_deviation_feature(file_df)

    # The saved dataset must always contain EVERY registered feature,
    # whichever groups a later training run chooses to enable -- so
    # ablations only need an environment variable, not a rebuild.
    require_columns(file_df, groups=available_groups())
    print("Feature columns in the dataset:")
    for column, group in NUMERIC_FEATURE_REGISTRY:
        print(f"  [{group}] {column}")

    print("Splitting users...")
    all_users = file_df["user"].unique()
    malicious_users = malicious_windows["user"].unique()
    train_users, val_users, test_users = split_users(all_users, malicious_users)

    file_df["split"] = "unassigned"
    file_df.loc[file_df["user"].isin(train_users), "split"] = "train"
    file_df.loc[file_df["user"].isin(val_users), "split"] = "validation"
    file_df.loc[file_df["user"].isin(test_users), "split"] = "test"
    assert (file_df["split"] != "unassigned").all(), "Some users were not assigned a split"

    # Test users are further divided into a development partition (used
    # for every design decision) and a locked partition (evaluated once,
    # at the end). Insiders are stratified by scenario.
    scenario_by_user = malicious_windows.set_index("user")["scenario"].to_dict()
    dev_users, locked_users = split_test_partition(test_users, scenario_by_user)
    file_df["test_partition"] = ""
    file_df.loc[file_df["user"].isin(dev_users), "test_partition"] = "dev"
    file_df.loc[file_df["user"].isin(locked_users), "test_partition"] = "locked"
    print("Test partition (users): "
          f"dev={len(dev_users)} (insiders {len(dev_users & set(scenario_by_user))}), "
          f"locked={len(locked_users)} (insiders {len(locked_users & set(scenario_by_user))})")

    print("Saving processed data...")
    output_dir = os.path.join(_PROJECT_ROOT, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "file_events_processed.parquet")
    file_df.to_parquet(output_path, index=False)

    print(f"\nDone. {len(file_df)} events saved to {output_path}")
    print(file_df["split"].value_counts())


if __name__ == "__main__":
    build_dataset()
