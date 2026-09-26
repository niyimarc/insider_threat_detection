import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
r42_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "r4.2"))

logon_df = pd.read_csv(os.path.join(r42_dir, "logon.csv"))
logon_df["date"] = pd.to_datetime(logon_df["date"])
logon_df = logon_df.sort_values(["user", "date"]).reset_index(drop=True)

# For each row, look at the PREVIOUS row for the same user
logon_df["prev_activity"] = logon_df.groupby("user")["activity"].shift(1)

double_logons = logon_df[
    (logon_df["activity"] == "Logon") & (logon_df["prev_activity"] == "Logon")
]
print("Total 'Logon directly followed by another Logon' occurrences:", len(double_logons))
print("Number of distinct users affected:", double_logons["user"].nunique())

if len(double_logons) > 0:
    example_user = double_logons["user"].iloc[0]
    example_date = double_logons[double_logons["user"] == example_user]["date"].iloc[0]
    print(f"\nExample - full logon sequence for user {example_user} around {example_date.date()}:")
    same_day = logon_df[
        (logon_df["user"] == example_user)
        & (logon_df["date"].dt.date == example_date.date())
    ]
    print(same_day[["date", "activity"]].to_string(index=False))
else:
    print("\nNo double-logon pattern found in the data.")