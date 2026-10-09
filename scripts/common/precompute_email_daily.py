"""
One chronological pass over email.csv (about 1.3 GB, the text column is never loaded) that writes
the per user-day email features to data/processed/email_user_day.parquet.

    python scripts/common/precompute_email_daily.py

Run it where the raw data lives (on Colab: the Drive copy). Envelope features only: no message
text, no labels, no answer key. Nothing under data/ is committed to git.
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src.common.preprocessing.email_features import OUTPUT_NAME, build_email_user_day
from src.common.preprocessing.load import DEFAULT_RAW_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "processed"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    path = os.path.join(args.raw_dir, "email.csv")
    print(f"Reading {path} ...")
    table = build_email_user_day(path, chunksize=args.chunksize)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, OUTPUT_NAME)
    table.to_parquet(out, index=False)
    print(f"{len(table)} user-days with email, {table['user'].nunique()} users, "
          f"{table['day'].min().date()} to {table['day'].max().date()}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
