import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
insiders_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "answers", "insiders.csv"))

df = pd.read_csv(insiders_path)

# See exactly what distinct dataset values exist, so we're filtering
# against a real observed value rather than a guessed one.
print("Unique dataset values:", sorted(df["dataset"].unique()))

r42 = df[df["dataset"] == 4.2]
print("\nRows matching dataset == 4.2:", len(r42))
print(r42[["scenario", "user", "start", "end"]].to_string(index=False))