import numpy as np
import pandas as pd
import pytest

from src.supervised.peer_signals import PEER_COLUMNS, add_peer_features

DAYS = pd.date_range("2010-02-01", periods=40)          # February and March 2010
ROLE1 = ["A1", "A2", "A3", "A4", "A5"]
ROLE2 = ["B1", "B2", "B3", "B4"]


def _ldap(drop_from_march=()):
    rows = []
    for month in ("2010-02", "2010-03"):
        for user in ROLE1 + ROLE2 + ["C1"]:
            if month == "2010-03" and user in drop_from_march:
                continue
            role = "R1" if user in ROLE1 + ["C1"] else "R2"
            team = {"C1": "T9"}.get(user, "T1" if user in ROLE1 else "T2")
            rows.append({"user": user, "role": role, "team": team, "month": pd.Period(month, freq="M")})
    return pd.DataFrame(rows)


def _counts(value_fn):
    """logon, device and email frames; every user has `value_fn(user, day_index)` of each signal."""
    rows = []
    for user in ROLE1 + ROLE2 + ["C1"]:
        for i, day in enumerate(DAYS):
            rows.append({"user": user, "day": day, "v": float(value_fn(user, i))})
    frame = pd.DataFrame(rows)
    logon = frame.rename(columns={"v": "lg_n_logons"}).assign(lg_after_hours_events=frame["v"])
    device = frame.rename(columns={"v": "dv_n_connects"})
    email = frame.rename(columns={"v": "em_n_sent"})
    return logon, device, email


def _table(users, day_indices):
    return pd.DataFrame([{"user": u, "day": DAYS[i]} for u in users for i in day_indices])


def _run(value_fn, table=None, ldap=None):
    logon, device, email = _counts(value_fn)
    table = _table(ROLE1 + ROLE2, [3, 20]) if table is None else table
    return add_peer_features(table, logon, device, email, _ldap() if ldap is None else ldap)


def _at(out, user, index, column):
    row = out[(out["user"] == user) & (out["day"] == DAYS[index])]
    return row[column].item()


def test_a_spike_is_measured_against_the_other_people_in_the_group_not_the_user_themselves():
    out = _run(lambda u, i: 40 if (u == "A1" and i == 20) else 4)
    assert _at(out, "A1", 20, "pr_lg_n_logons_role_1d") == pytest.approx(np.log1p(40) - np.log1p(4))
    # a peer of A1 sees the spiking colleague inside the peer mean (A1, A3, A4, A5 and C1 are R1: 5 peers each)
    expected_peer_mean = (40 + 4 + 4 + 4 + 4) / 5            # A1, A3, A4, A5, C1 (A2 is left out)
    assert _at(out, "A2", 20, "pr_lg_n_logons_role_1d") == pytest.approx(np.log1p(4) - np.log1p(expected_peer_mean))
    assert _at(out, "A1", 3, "pr_lg_n_logons_role_1d") == pytest.approx(0.0)


def test_other_groups_are_unaffected():
    out = _run(lambda u, i: 40 if (u == "A1" and i == 20) else 4)
    assert _at(out, "B1", 20, "pr_lg_n_logons_role_1d") == pytest.approx(0.0)
    assert _at(out, "B1", 20, "pr_dv_n_connects_team_1d") == pytest.approx(0.0)


def test_a_company_wide_change_is_not_a_deviation():
    out = _run(lambda u, i: 12 if i == 20 else 4)
    for user in ("A1", "B2"):
        assert _at(out, user, 20, "pr_em_n_sent_role_1d") == pytest.approx(0.0)


def test_windowed_features_wait_for_a_full_window_and_average_the_days():
    out = _run(lambda u, i: 10 if (u == "A1" and 7 <= i <= 20) else 2, table=_table(["A1"], [5, 20]))
    assert np.isnan(_at(out, "A1", 5, "pr_lg_n_logons_role_14d"))                    # fewer than 14 days of data
    # days 7..20 are all 10 for A1; the peers (A2..A5 and C1) are at 2 every day
    assert _at(out, "A1", 20, "pr_lg_n_logons_role_14d") == pytest.approx(np.log1p(10.0) - np.log1p(2.0))


def test_groups_with_fewer_than_three_peers_give_nan_for_that_group_only():
    out = _run(lambda u, i: 4, table=_table(["C1"], [20]))
    assert np.isnan(_at(out, "C1", 20, "pr_lg_n_logons_team_1d"))                   # C1 is alone in team T9
    assert _at(out, "C1", 20, "pr_lg_n_logons_role_1d") == pytest.approx(0.0)       # but has 5 role peers


def test_people_who_left_stop_counting_as_peers():
    value = lambda u, i: 100 if (u == "A5" and i >= 28) else 4                      # A5 is huge from March on
    table = _table(["A1"], [3, 35])
    stays = _run(value, table=table, ldap=_ldap())
    leaves = _run(value, table=table, ldap=_ldap(drop_from_march=("A5",)))
    assert _at(stays, "A1", 35, "pr_lg_n_logons_role_1d") < -1.0                    # A5's 100 is in A1's peer mean
    assert _at(leaves, "A1", 35, "pr_lg_n_logons_role_1d") == pytest.approx(0.0)    # A5 no longer a peer
    assert _at(leaves, "A1", 3, "pr_lg_n_logons_role_1d") == pytest.approx(0.0)     # February unchanged


def test_users_missing_from_ldap_get_nan_and_rows_are_unchanged():
    table = pd.concat([_table(["A1", "ZZ"], [3, 20])], ignore_index=True)
    out = _run(lambda u, i: 4, table=table)
    assert len(out) == len(table) and out[["user", "day"]].equals(table)
    assert set(PEER_COLUMNS) <= set(out.columns)
    assert out.loc[out["user"] == "ZZ", list(PEER_COLUMNS)].isna().all().all()


def test_input_frames_are_not_modified_and_no_label_is_used():
    logon, device, email = _counts(lambda u, i: 4)
    table, ldap = _table(["A1"], [3]), _ldap()
    before = [f.copy() for f in (logon, device, email, table, ldap)]
    add_peer_features(table, logon, device, email, ldap)
    for original, copy in zip((logon, device, email, table, ldap), before):
        pd.testing.assert_frame_equal(original, copy)
    assert not any("label" in c for c in PEER_COLUMNS)


def test_only_same_day_and_past_activity_is_used():
    table = _table(ROLE1 + ROLE2, [3, 20, 25])
    base = _run(lambda u, i: 4, table=table)
    later = _run(lambda u, i: 400 if (u == "A2" and i > 20) else 4, table=table)      # only A2's future changes
    early = base["day"].isin([DAYS[3], DAYS[20]]).to_numpy()
    for column in PEER_COLUMNS:
        a, b = base[column].to_numpy(), later[column].to_numpy()
        np.testing.assert_allclose(a[early], b[early], equal_nan=True)
    day25 = (base["day"] == DAYS[25]).to_numpy()                                        # the change IS visible on day 25
    assert not np.allclose(base["pr_lg_n_logons_role_1d"].to_numpy()[day25],
                           later["pr_lg_n_logons_role_1d"].to_numpy()[day25])
