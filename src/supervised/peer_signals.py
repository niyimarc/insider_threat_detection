"""
PEER-GROUP baselines for the all-days supervised table.

The per-user baselines of day_signals.py compare a person with their OWN past. These compare a
person with their PEERS on the same days: the other employees with the same role (42 roles) or the
same team (38 teams), taken from the LDAP staff lists. A burst of device use that is normal for a
role, or a company-wide quiet week, then stops looking like a deviation.

For each signal S in PEER_SIGNALS, group G in PEER_GROUPS and window W in PEER_WINDOWS (calendar days):

    pr_<S>_<G>_<W>d = log1p(user's mean of S over the last W days)
                      - log1p(mean of the same quantity over the user's peers in G)

Details fixed in advance:
  * the user is LEFT OUT of their own peer mean, and only people on the LDAP list of that day's month
    count as peers (leavers stop counting);
  * peers come from ALL employees with activity in the logs, not only the users in the table;
  * calendar-day windows, days without activity count as 0 for the user and for peers alike, and the
    first W-1 days of the data are NaN;
  * a peer mean needs at least MIN_PEERS peers, otherwise NaN;
  * only same-day and past activity is used; no label and no answer-key event.

Insiders are not removed from their peers' means (a deployment cannot know who they are); with one insider
in dozens of peers the effect is small.
"""

import numpy as np
import pandas as pd

PEER_SIGNALS = (          # (column, which input frame holds it)
    ("lg_n_logons", "logon"),
    ("lg_after_hours_events", "logon"),
    ("dv_n_connects", "device"),
    ("em_n_sent", "email"),
)
PEER_GROUPS = ("role", "team")
PEER_WINDOWS = (1, 14)
MIN_PEERS = 3

PEER_COLUMNS = tuple(
    f"pr_{signal}_{group}_{window}d"
    for signal, _ in PEER_SIGNALS for group in PEER_GROUPS for window in PEER_WINDOWS
)


def _activity_matrix(frame, column, user_pos, first_day, n_days):
    out = np.zeros((len(user_pos), n_days))
    pos = frame["user"].map(user_pos).to_numpy(dtype=float)
    offset = (pd.to_datetime(frame["day"]) - first_day).dt.days.to_numpy()
    ok = np.isfinite(pos) & (offset >= 0) & (offset < n_days)
    np.add.at(out, (pos[ok].astype(int), offset[ok]), frame[column].to_numpy(dtype=float)[ok])
    return out


def _trailing_mean(x, window):
    """Mean of the last `window` calendar days (today included); NaN for the first window-1 days."""
    if window == 1:
        return x.copy()
    cumulative = np.cumsum(x, axis=1)
    out = cumulative.copy()
    out[:, window:] = cumulative[:, window:] - cumulative[:, :-window]
    out = out / window
    out[:, : window - 1] = np.nan
    return out


def _membership(ldap, user_pos, first_day, n_days):
    """Boolean users x days: True if the user is on the LDAP list that applies to the day's month."""
    member = np.zeros((len(user_pos), n_days), dtype=bool)
    snapshot_months = sorted(ldap["month"].unique())
    by_month = {m: ldap.loc[ldap["month"] == m, "user"].map(user_pos).dropna().astype(int).to_numpy()
                for m in snapshot_months}
    days = pd.date_range(first_day, periods=n_days, freq="D")
    day_months = days.to_period("M")
    for month in day_months.unique():
        earlier = [m for m in snapshot_months if m <= month]
        snapshot = earlier[-1] if earlier else snapshot_months[0]
        columns = np.flatnonzero(np.asarray(day_months == month))
        member[np.ix_(by_month[snapshot], columns)] = True
    return member


def _peer_mean(rolling, member, group_id, n_groups):
    """Leave-one-out mean of `rolling` over the other members of each user's group, per day (NaN if too few)."""
    usable = member & np.isfinite(rolling)
    values = np.where(usable, rolling, 0.0)
    counts = usable.astype(float)
    has_group = group_id >= 0
    onehot = np.zeros((n_groups, len(group_id)))
    onehot[group_id[has_group], np.flatnonzero(has_group)] = 1.0
    group_sum, group_count = onehot @ values, onehot @ counts
    safe_group = np.where(has_group, group_id, 0)
    peers = group_count[safe_group] - counts
    total = group_sum[safe_group] - values
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(peers >= MIN_PEERS, total / np.maximum(peers, 1.0), np.nan)
    mean[~has_group, :] = np.nan
    return mean


def add_peer_features(table, logon_day, device_day, email_user_day, ldap):
    """
    A copy of `table` (columns user, day; row order and index kept) with PEER_COLUMNS added.
    logon_day / device_day / email_user_day: per user-day count frames for ALL employees (output of
    logon_day_counts, device_day_counts, build_email_user_day). ldap: load_ldap_snapshots().
    Rows whose user is not in the LDAP lists get NaN.
    """
    frames = {"logon": logon_day, "device": device_day, "email": email_user_day}
    table_days = pd.to_datetime(table["day"])
    first_day = min([table_days.min()] + [pd.to_datetime(f["day"]).min() for f in frames.values()])
    last_day = max([table_days.max()] + [pd.to_datetime(f["day"]).max() for f in frames.values()])
    n_days = int((last_day - first_day).days) + 1

    users = sorted(ldap["user"].unique())
    user_pos = {u: i for i, u in enumerate(users)}
    member = _membership(ldap, user_pos, first_day, n_days)
    latest = ldap.sort_values("month").groupby("user").last()

    row_user = table["user"].map(user_pos).to_numpy(dtype=float)
    row_day = (table_days - first_day).dt.days.to_numpy()
    row_ok = np.isfinite(row_user)
    rows = row_user[row_ok].astype(int)
    cols = row_day[row_ok]

    group_ids, group_sizes = {}, {}
    for group in PEER_GROUPS:
        codes, labels = pd.factorize(latest[group].reindex(users))
        group_ids[group], group_sizes[group] = codes, len(labels)

    out = table.copy()
    for signal, source in PEER_SIGNALS:
        daily = _activity_matrix(frames[source], signal, user_pos, first_day, n_days)
        for window in PEER_WINDOWS:
            rolling = _trailing_mean(daily, window)
            for group in PEER_GROUPS:
                peer = _peer_mean(rolling, member, group_ids[group], group_sizes[group])
                values = np.full(len(table), np.nan)
                with np.errstate(invalid="ignore"):
                    values[row_ok] = np.log1p(rolling[rows, cols]) - np.log1p(peer[rows, cols])
                out[f"pr_{signal}_{group}_{window}d"] = values
    return out
