import numpy as np
import pandas as pd

from src.common.evaluation.labelings import answer_key_session_labels, load_answer_key_events


def _write_answer_key(tmp_path):
    folder = tmp_path / "r4.2-2"
    folder.mkdir()
    (folder / "r4.2-2-U1.csv").write_text(
        "http,{AAA},06/28/2010 08:51:08,U1,PC-1,http://monster.com/a.html,words, with a comma, inside\n"
        "device,{B1},08/03/2010 13:55:03,U1,PC-1,Connect\n"
        "device,{B2},08/03/2010 13:58:07,U1,PC-1,Disconnect\n"
        "device,{B3},09/01/2010 10:00:00,U1,PC-1,Connect\n"          # never disconnected: ignored
        "device,{B4},09/05/2010 10:00:00,U1,PC-2,Disconnect\n"       # disconnect with no connect: ignored
        "logon,{C1},06/28/2010 07:00:00,U1,PC-1,Logon\n"             # other record types are ignored
    )
    other = tmp_path / "r4.2-3"
    other.mkdir()
    (other / "r4.2-3-U9.csv").write_text("http,{DDD},07/10/2010 12:00:00,U9,PC-9,http://x.com/y,z\n")
    return str(tmp_path)


def test_answer_key_http_days_and_device_intervals_are_parsed(tmp_path):
    key = load_answer_key_events(_write_answer_key(tmp_path))

    http = key["http_days"]
    assert set(zip(http["user"], http["day"], http["scenario"])) == {
        ("U1", pd.Timestamp("2010-06-28"), 2),
        ("U9", pd.Timestamp("2010-07-10"), 3),
    }
    intervals = key["device_intervals"]
    assert len(intervals) == 1  # only the complete Connect->Disconnect pair
    row = intervals.iloc[0]
    assert row["user"] == "U1" and row["scenario"] == 2
    assert row["start"] == pd.Timestamp("2010-08-03 13:55:03")
    assert row["end"] == pd.Timestamp("2010-08-03 13:58:07")


def _events():
    def session(user, start, events):
        start = pd.Timestamp(start)
        return [
            {"user": user, "session_start": start, "session_end": start + pd.Timedelta(hours=8),
             "date": pd.Timestamp(d), "session_is_malicious": 0}
            for d in events
        ]
    rows = []
    rows += session("U1", "2010-06-28 08:00:00", ["2010-06-28 10:00:00"])                       # on an http day
    rows += session("U1", "2010-07-01 08:00:00", ["2010-07-01 10:00:00"])                       # nothing
    rows += session("U1", "2010-08-03 08:00:00", ["2010-08-03 13:56:00", "2010-08-03 15:00:00"])  # inside a device interval
    rows += session("U1", "2010-08-04 08:00:00", ["2010-08-03 15:00:00"])                       # same day, outside the interval
    rows += session("U2", "2010-06-28 08:00:00", ["2010-06-28 10:00:00"])                       # same day as U1's http day, other user
    return pd.DataFrame(rows)


def test_session_labels_follow_the_answer_key_and_the_canonical_session_order(tmp_path):
    key = load_answer_key_events(_write_answer_key(tmp_path))
    events = _events()
    labels = answer_key_session_labels(events, key)

    ordered = events.sort_values(["user", "session_start", "session_end"])
    sessions = ordered.groupby(["user", "session_start", "session_end"], sort=False).size().reset_index()
    assert len(labels) == len(sessions) == 5
    by_session = {
        (r["user"], r["session_start"].strftime("%Y-%m-%d")): int(l) for (_, r), l in zip(sessions.iterrows(), labels)
    }
    assert by_session[("U1", "2010-06-28")] == 1   # injected http day
    assert by_session[("U1", "2010-07-01")] == 0
    assert by_session[("U1", "2010-08-03")] == 1   # a file copy while the answer-key device session was open
    assert by_session[("U1", "2010-08-04")] == 0   # same day as the device session but outside it
    assert by_session[("U2", "2010-06-28")] == 0   # the http day belongs to a different user


def test_windows_line_endings_are_handled(tmp_path):
    # Regression: a trailing "\r" made "Connect\r" != "Connect", so no device
    # interval was ever built on files with Windows line endings.
    folder = tmp_path / "r4.2-1"
    folder.mkdir()
    with open(folder / "r4.2-1-U5.csv", "w", newline="") as f:
        f.write("device,{A},08/03/2010 13:55:03,U5,PC-5,Connect\r\n"
                "device,{B},08/03/2010 13:58:07,U5,PC-5,Disconnect\r\n"
                "http,{C},08/04/2010 09:00:00,U5,PC-5,http://wikileaks.org/x,content\r\n")
    key = load_answer_key_events(str(tmp_path))
    assert len(key["device_intervals"]) == 1
    assert len(key["http_days"]) == 1


def test_input_frame_is_not_modified(tmp_path):
    key = load_answer_key_events(_write_answer_key(tmp_path))
    events = _events()
    before = events.copy()
    answer_key_session_labels(events, key)
    pd.testing.assert_frame_equal(events, before)


def test_no_answer_key_files_means_no_positive_sessions(tmp_path):
    key = load_answer_key_events(str(tmp_path))
    labels = answer_key_session_labels(_events(), key)
    assert labels.sum() == 0 and len(labels) == 5
