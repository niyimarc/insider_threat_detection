import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
r42_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "r4.2"))

print("=" * 60)
print("FILE.CSV — full verification")
print("=" * 60)
file_df = pd.read_csv(os.path.join(r42_dir, "file.csv"))
print("Total rows:", len(file_df))
print("Columns:", list(file_df.columns))
print("Nulls per column:\n", file_df.isnull().sum())
print("Distinct users:", file_df["user"].nunique())
print("Date range:", file_df["date"].min(), "to", file_df["date"].max())

# Extract just the first token of content (the hex file-header) for every row
file_df["header"] = file_df["content"].str.split(" ").str[0]
print("\nDistinct file headers found:", file_df["header"].nunique())
print(file_df["header"].value_counts().head(15))

# Extract filename extension and cross-check against header,
# to test the readme's claim that "file header correlates with filename extension"
file_df["extension"] = file_df["filename"].str.split(".").str[-1].str.lower()
print("\nExtension vs header (first 20 combinations):")
print(file_df.groupby(["extension", "header"]).size().sort_values(ascending=False).head(20))

print("\n" + "=" * 60)
print("LOGON.CSV — verifying against the readme's claims ourselves")
print("=" * 60)
logon_df = pd.read_csv(os.path.join(r42_dir, "logon.csv"))
print("Columns:", list(logon_df.columns))
print("Unique 'activity' values:", logon_df["activity"].unique())
print(logon_df.head(3))

print("\n" + "=" * 60)
print("DEVICE.CSV — verifying against the readme's claims ourselves")
print("=" * 60)
device_df = pd.read_csv(os.path.join(r42_dir, "device.csv"))
print("Columns:", list(device_df.columns))
print("Unique 'activity' values:", device_df["activity"].unique())
print(device_df.head(3))