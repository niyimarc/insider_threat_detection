import pandas as pd

df = pd.read_parquet("data/processed/file_events_processed.parquet")

user_splits = df.groupby("user")["split"].first()
print("Users per split:")
print(user_splits.value_counts())

print("\nTotal distinct users:", df["user"].nunique())

# Check no single user's events are split across more than one group —
# each user should have exactly ONE split value across all their rows.
inconsistent = df.groupby("user")["split"].nunique()
print("\nUsers with more than one split value (should be none):",
      (inconsistent > 1).sum())