"""
One-pass reduction of the raw http.csv (about 14 GB, 'content' column
included) into two small tables that every later http step uses, so the
big file only ever has to be read once.

  domain table : (user, day, domain)  -> number of page visits
  hour table   : (user, day, hour)    -> number of page visits

Nothing here uses labels, the answer key or the train/test split: it is
a pure, deterministic aggregation of the raw log, applied to ALL users.
Anything that must be learned from training users only (which domains
are "common", per-user baselines) is computed later, from these tables,
inside the feature step.

Only the date, user and url columns are read; 'content' is the bulk of
the file and is not needed.
"""

import pandas as pd

HTTP_USECOLS = ["date", "user", "url"]
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def extract_domain(urls):
    """
    Host part of each URL, lower-cased, without a leading 'www.'.
    'http://www.Example.com/a/b?x=1' -> 'example.com'. Rows with no
    parsable host become the empty string.
    """
    hosts = (
        urls.astype(str)
        .str.replace(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", regex=True)
        .str.split("/", n=1).str[0]
        .str.split("?", n=1).str[0]
        .str.split(":", n=1).str[0]
        .str.lower()
        .str.replace(r"^www\d*\.", "", regex=True)
    )
    return hosts


def aggregate_chunk(chunk):
    """Returns (domain_counts, hour_counts) DataFrames for one chunk of
    raw http rows (columns: date, user, url)."""
    chunk = chunk.dropna(subset=["date", "user", "url"]).copy()
    dates = pd.to_datetime(chunk["date"], format=DATE_FORMAT, errors="coerce")
    keep = dates.notna()
    chunk, dates = chunk[keep], dates[keep]

    day = dates.dt.normalize()
    domain = extract_domain(chunk["url"])

    domain_counts = (
        pd.DataFrame({"user": chunk["user"].values, "day": day.values, "domain": domain.values})
        .groupby(["user", "day", "domain"], sort=False).size().rename("visits").reset_index()
    )
    hour_counts = (
        pd.DataFrame({"user": chunk["user"].values, "day": day.values, "hour": dates.dt.hour.values})
        .groupby(["user", "day", "hour"], sort=False).size().rename("visits").reset_index()
    )
    return domain_counts, hour_counts


def combine_counts(frames, key_columns):
    """Adds up several partial count tables that may share keys."""
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(key_columns, sort=False)["visits"].sum().reset_index()
    )
