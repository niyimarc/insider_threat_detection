"""
Reduces the raw http.csv (about 14 GB) to two small per-user-per-day
tables, once, so exploration and feature building never have to read the
big file again. See src/common/preprocessing/http_aggregate.py for what is kept.

Streams the file in chunks, so memory stays small (a few GB at most).
Uses no labels and no train/test split -- it is a pure aggregation.

Run from the project root (a long job: expect tens of minutes):
    python scripts/common/precompute_http_daily.py
    python scripts/common/precompute_http_daily.py --raw-dir PATH/TO/r4.2 --out-dir PATH/TO/processed

Outputs (parquet): http_domain_daily.parquet, http_hour_daily.parquet
"""

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.common.preprocessing.http_aggregate import (
    HTTP_USECOLS,
    aggregate_chunk,
    combine_counts,
)
from src.common.preprocessing.load import DEFAULT_RAW_DIR

CHUNK_ROWS = 2_000_000
COMPACT_EVERY = 10  # merge partial tables every N chunks to bound memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "processed"))
    args = parser.parse_args()

    path = os.path.join(args.raw_dir, "http.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Reading {path} in chunks of {CHUNK_ROWS:,} rows (columns: {HTTP_USECOLS}) ...")

    domain_parts, hour_parts = [], []
    total_rows, start = 0, time.time()

    reader = pd.read_csv(path, usecols=HTTP_USECOLS, chunksize=CHUNK_ROWS, dtype=str)
    for i, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        d, h = aggregate_chunk(chunk)
        domain_parts.append(d)
        hour_parts.append(h)

        if i % COMPACT_EVERY == 0:
            domain_parts = [combine_counts(domain_parts, ["user", "day", "domain"])]
            hour_parts = [combine_counts(hour_parts, ["user", "day", "hour"])]
            print(f"  {total_rows:>12,} rows | domain table {len(domain_parts[0]):,} rows | "
                  f"{time.time() - start:6.0f}s", flush=True)

    domain_table = combine_counts(domain_parts, ["user", "day", "domain"])
    hour_table = combine_counts(hour_parts, ["user", "day", "hour"])

    domain_path = os.path.join(args.out_dir, "http_domain_daily.parquet")
    hour_path = os.path.join(args.out_dir, "http_hour_daily.parquet")
    domain_table.to_parquet(domain_path, index=False)
    hour_table.to_parquet(hour_path, index=False)

    print(f"\nDone: {total_rows:,} raw rows -> {len(domain_table):,} (user,day,domain) rows, "
          f"{len(hour_table):,} (user,day,hour) rows")
    print(f"  sum of visits (domain table) = {int(domain_table['visits'].sum()):,}  "
          f"(hour table) = {int(hour_table['visits'].sum()):,}  -- these must match")
    assert int(domain_table["visits"].sum()) == int(hour_table["visits"].sum())
    print(f"  {domain_path}\n  {hour_path}")


if __name__ == "__main__":
    main()
