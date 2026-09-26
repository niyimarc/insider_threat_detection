import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
r42_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "r4.2"))

# Check file sizes first, in MB, before loading anything fully
for fname in ["file.csv", "logon.csv", "device.csv", "http.csv", "email.csv"]:
    path = os.path.join(r42_dir, fname)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"{fname}: {size_mb:.1f} MB")

# Peek at just the first 5 rows of file.csv, cheaply, without loading it all
print("\n--- file.csv preview ---")
preview = pd.read_csv(os.path.join(r42_dir, "file.csv"), nrows=5)
print("Columns:", list(preview.columns))
print(preview)