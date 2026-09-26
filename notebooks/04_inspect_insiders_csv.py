import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
insiders_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "answers", "insiders.csv"))

df = pd.read_csv(insiders_path)

print("Shape (rows, columns):", df.shape)
print("\nColumn names:", list(df.columns))
print("\nFirst 10 rows:")
print(df.head(10))