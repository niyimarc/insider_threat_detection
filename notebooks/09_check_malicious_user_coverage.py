import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
r42_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "r4.2"))
answers_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "answers"))

# The 70 malicious users for r4.2, from the master answer file
insiders = pd.read_csv(os.path.join(answers_dir, "insiders.csv"))
r42_insiders = insiders[insiders["dataset"] == 4.2].copy()

# Load file.csv and parse dates PROPERLY this time
file_df = pd.read_csv(os.path.join(r42_dir, "file.csv"))
file_df["date"] = pd.to_datetime(file_df["date"])
print("Corrected date range:", file_df["date"].min(), "to", file_df["date"].max())

active_file_users = set(file_df["user"].unique())
malicious_users = set(r42_insiders["user"].unique())

covered = malicious_users & active_file_users
missing = malicious_users - active_file_users

print(f"\nMalicious users total: {len(malicious_users)}")
print(f"Malicious users who DO appear in file.csv: {len(covered)}")
print(f"Malicious users who DO NOT appear in file.csv: {len(missing)}")

if missing:
    missing_rows = r42_insiders[r42_insiders["user"].isin(missing)]
    print("\nMissing users, broken down by scenario:")
    print(missing_rows[["scenario", "user"]].sort_values("scenario"))