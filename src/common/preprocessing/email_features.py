"""
Per user-day EMAIL features, computed in one chronological pass over email.csv.

Only the envelope is used (who sent to how many people, how many are outside the
company, bcc, attachments, size, hour): never the message text. Nothing here uses
a label or the answer key. The company domain is inferred from the data itself
(the most common domain among the `from` addresses), not hard-coded.

Columns of the result (one row per user-day on which the user sent mail):

  em_n_sent                 emails sent that day
  em_n_recipients           recipient slots used (to + cc + bcc, with repeats)
  em_n_distinct_recipients  distinct addresses mailed that day
  em_n_external_recipients  recipient slots whose domain is not the company's
  em_max_recipients         recipients of the biggest single email (mass mail)
  em_n_bcc_emails           emails that used bcc
  em_n_attachment_emails    emails with at least one attachment
  em_attachments            attachments sent
  em_log_size_sum           sum of log(1 + size) over the day's emails
  em_after_hours_sent       emails sent outside 07:00-19:00
  em_n_new_recipients       addresses this user mailed for the FIRST time that day

NOTE on em_n_new_recipients: everyone's first weeks look "all new", a drift
shared by all users. The supervised pipeline trains with date-matched weights
and reports a per-user-clock control for exactly this kind of effect.

The file must be in chronological order (r4.2's is). Days are processed
complete (the last day of a chunk is carried into the next), so distinct and
new-recipient counts are exact.
"""

import numpy as np
import pandas as pd

EMAIL_RAW_COLUMNS = (
    "em_n_sent", "em_n_recipients", "em_n_distinct_recipients", "em_n_external_recipients",
    "em_max_recipients", "em_n_bcc_emails", "em_n_attachment_emails", "em_attachments",
    "em_log_size_sum", "em_after_hours_sent", "em_n_new_recipients",
)
EMAIL_USECOLS = ["date", "user", "to", "cc", "bcc", "from", "size", "attachments"]
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
WORK_START_HOUR, WORK_END_HOUR = 7, 19
OUTPUT_NAME = "email_user_day.parquet"


def infer_company_domain(from_addresses):
    """The most common domain among sender addresses."""
    domains = pd.Series(from_addresses).dropna().astype(str).str.rsplit("@", n=1).str[-1].str.lower()
    if domains.empty:
        raise ValueError("No sender addresses to infer the company domain from.")
    return domains.value_counts().index[0]


def _recipient_rows(df):
    """One row per recipient slot: user, day, addr (to + cc + bcc split on ';')."""
    joined = df["to"].fillna("") + ";" + df["cc"].fillna("") + ";" + df["bcc"].fillna("")
    rec = pd.DataFrame({"user": df["user"].to_numpy(), "day": df["day"].to_numpy(),
                        "email": np.arange(len(df)), "addr": joined.str.split(";").to_numpy()})
    rec = rec.explode("addr")
    rec = rec[rec["addr"].notna() & (rec["addr"] != "")]
    rec["addr"] = rec["addr"].str.lower()
    return rec


def process_complete_days(df, company_domain, seen):
    """
    df: raw email rows of whole days (columns of EMAIL_USECOLS plus a parsed datetime `ts` and a
    normalised `day`). seen: set of "user|address" strings (updated in place). Returns a DataFrame
    (user, day, EMAIL_RAW_COLUMNS).
    """
    df = df.reset_index(drop=True)
    keys = ["user", "day"]
    rec = _recipient_rows(df)
    rec["external"] = (rec["addr"].str.rsplit("@", n=1).str[-1] != company_domain).astype(float)

    per_email = rec.groupby("email").size().rename("n_rcpt")
    mail = df.assign(n_rcpt=per_email.reindex(np.arange(len(df))).fillna(0.0).to_numpy())
    hour = mail["ts"].dt.hour
    mail["after"] = ((hour < WORK_START_HOUR) | (hour >= WORK_END_HOUR)).astype(float)
    mail["has_bcc"] = mail["bcc"].fillna("").ne("").astype(float)
    mail["has_att"] = (mail["attachments"].astype(float) > 0).astype(float)
    mail["att"] = mail["attachments"].astype(float)
    mail["logsize"] = np.log1p(mail["size"].astype(float))
    mail["one"] = 1.0
    out = mail.groupby(keys).agg(
        em_n_sent=("one", "sum"), em_n_recipients=("n_rcpt", "sum"), em_max_recipients=("n_rcpt", "max"),
        em_n_bcc_emails=("has_bcc", "sum"), em_n_attachment_emails=("has_att", "sum"),
        em_attachments=("att", "sum"), em_log_size_sum=("logsize", "sum"), em_after_hours_sent=("after", "sum"),
    )
    out["em_n_distinct_recipients"] = rec.groupby(keys)["addr"].nunique().reindex(out.index).fillna(0.0)
    out["em_n_external_recipients"] = rec.groupby(keys)["external"].sum().reindex(out.index).fillna(0.0)

    first = rec.groupby(["user", "addr"], as_index=False)["day"].min()
    first["key"] = first["user"].astype(str) + "|" + first["addr"]
    new = first[~first["key"].isin(seen)]
    out["em_n_new_recipients"] = new.groupby(keys).size().reindex(out.index).fillna(0.0)
    seen.update(first["key"])

    out = out.reset_index()
    return out[keys + list(EMAIL_RAW_COLUMNS)].astype({c: float for c in EMAIL_RAW_COLUMNS})


def build_email_user_day(path, chunksize=500_000, company_domain=None, progress=print):
    """Reads email.csv in chunks (text column never loaded) and returns the user-day feature table."""
    seen, parts = set(), []
    carry, last_done, max_seen = None, None, None

    def run(frame):
        nonlocal last_done
        if frame.empty:
            return
        first_day = frame["day"].min()
        if last_done is not None and first_day < last_done:
            raise ValueError("email.csv is not in chronological order; the new-recipient feature would be wrong.")
        parts.append(process_complete_days(frame, company_domain, seen))
        last_done = frame["day"].max()

    for i, chunk in enumerate(pd.read_csv(path, usecols=EMAIL_USECOLS, chunksize=chunksize)):
        chunk["ts"] = pd.to_datetime(chunk["date"], format=DATE_FORMAT)
        chunk["day"] = chunk["ts"].dt.normalize()
        if company_domain is None:
            company_domain = infer_company_domain(chunk["from"])
            progress(f"company domain inferred from the data: {company_domain}")
        chunk = chunk.drop(columns=["date", "from"])
        chunk = chunk.sort_values("day", kind="stable")             # tolerate disorder inside a chunk ...
        if max_seen is not None and chunk["day"].min() < max_seen:   # ... but not between chunks
            raise ValueError("email.csv is not in chronological order; the new-recipient feature would be wrong.")
        max_seen = chunk["day"].max()
        frame = chunk if carry is None else pd.concat([carry, chunk], ignore_index=True)
        last_day = frame["day"].max()
        carry = frame[frame["day"] == last_day]
        run(frame[frame["day"] < last_day])
        progress(f"chunk {i + 1} done, {len(seen)} user-address pairs seen")
    if carry is not None:
        run(carry)
    return pd.concat(parts, ignore_index=True).sort_values(["user", "day"]).reset_index(drop=True)


def load_email_user_day(processed_dir):
    import os
    return pd.read_parquet(os.path.join(processed_dir, OUTPUT_NAME))
