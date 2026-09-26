import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
r42_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "r4.2"))

# 1. Read this release's own documentation first — the actual source of truth
readme_path = os.path.join(r42_dir, "readme.txt")
print("--- r4.2/readme.txt ---")
with open(readme_path, "r", errors="replace") as f:
    print(f.read())

# 2. Load a much bigger sample and double check the columns really are fixed
print("\n--- Bigger sample check ---")
sample = pd.read_csv(os.path.join(r42_dir, "file.csv"), nrows=50000)
print("Columns:", list(sample.columns))
print("Any nulls per column:\n", sample.isnull().sum())

# 3. Search the content column itself for anything resembling an activity word,
# in case operation type is embedded in the text rather than a separate column
keywords = ["open", "write", "copy", "delete"]
for kw in keywords:
    count = sample["content"].str.contains(kw, case=False, na=False).sum()
    print(f"Rows where content contains '{kw}': {count}")