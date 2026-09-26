import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
r42_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "r4.2"))
answers_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "answers"))

insiders = pd.read_csv(os.path.join(answers_dir, "insiders.csv"))
r42_insiders = insiders[insiders["dataset"] == 4.2].copy()
r42_insiders["start"] = pd.to_datetime(r42_insiders["start"])
r42_insiders["end"] = pd.to_datetime(r42_insiders["end"])

file_df = pd.read_csv(os.path.join(r42_dir, "file.csv"))
file_df["date"] = pd.to_datetime(file_df["date"])

results = []
for _, row in r42_insiders.iterrows():
    user_events = file_df[file_df["user"] == row["user"]]
    in_window = user_events[
        (user_events["date"] >= row["start"]) & (user_events["date"] <= row["end"])
    ]
    results.append({
        "scenario": row["scenario"],
        "user": row["user"],
        "total_file_events_ever": len(user_events),
        "file_events_during_malicious_window": len(in_window),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n--- Summary by scenario ---")
print(results_df.groupby("scenario")["file_events_during_malicious_window"].describe())