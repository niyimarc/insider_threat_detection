import numpy as np
import pandas as pd
import pytest

from src.common.preprocessing.email_features import (
    EMAIL_RAW_COLUMNS,
    build_email_user_day,
    infer_company_domain,
)

HEADER = "id,date,user,pc,to,cc,bcc,from,size,attachments,content\n"


def _row(date, user, to, cc="", bcc="", sender=None, size=100, attachments=0):
    sender = sender or f"{user.lower()}@dtaa.com"
    return f"{{id}},{date},{user},PC-1,{to},{cc},{bcc},{sender},{size},{attachments},some secret words\n"


def _write(tmp_path, rows):
    path = tmp_path / "email.csv"
    path.write_text(HEADER + "".join(rows))
    return str(path)


def _rows():
    return [
        _row("01/04/2010 08:00:00", "U1", "a@dtaa.com;x@gmail.com"),
        _row("01/04/2010 20:00:00", "U1", "a@dtaa.com", cc="b@dtaa.com", bcc="y@yahoo.com", size=1000, attachments=2),
        _row("01/04/2010 09:00:00", "U2", "a@dtaa.com"),
        _row("01/05/2010 10:00:00", "U1", "a@dtaa.com;z@gmail.com"),
        _row("01/06/2010 07:30:00", "U1", "A@dtaa.com"),                  # upper case: the same address
    ]


def test_company_domain_is_inferred_from_the_senders():
    assert infer_company_domain(["a@dtaa.com", "b@dtaa.com", "c@gmail.com"]) == "dtaa.com"
    with pytest.raises(ValueError):
        infer_company_domain([])


def test_a_days_features_are_exact_even_when_the_day_spans_chunks(tmp_path):
    table = build_email_user_day(_write(tmp_path, _rows()), chunksize=2, progress=lambda *_: None)
    assert list(table.columns) == ["user", "day", *EMAIL_RAW_COLUMNS]
    row = table.set_index(["user", "day"]).loc[("U1", pd.Timestamp("2010-01-04"))]
    assert row["em_n_sent"] == 2 and row["em_n_recipients"] == 5 and row["em_n_distinct_recipients"] == 4
    assert row["em_n_external_recipients"] == 2 and row["em_max_recipients"] == 3
    assert row["em_n_bcc_emails"] == 1 and row["em_n_attachment_emails"] == 1 and row["em_attachments"] == 2
    assert row["em_after_hours_sent"] == 1
    assert row["em_log_size_sum"] == pytest.approx(np.log1p(100) + np.log1p(1000))
    assert row["em_n_new_recipients"] == 4


def test_new_recipients_are_counted_only_the_first_time_an_address_is_mailed(tmp_path):
    table = build_email_user_day(_write(tmp_path, _rows()), chunksize=2, progress=lambda *_: None)
    new = table.set_index(["user", "day"])["em_n_new_recipients"]
    assert new[("U1", pd.Timestamp("2010-01-05"))] == 1        # only z@gmail.com is new
    assert new[("U1", pd.Timestamp("2010-01-06"))] == 0        # A@dtaa.com == a@dtaa.com
    assert new[("U2", pd.Timestamp("2010-01-04"))] == 1        # U2 never mailed a@ before, whatever U1 did


def test_result_does_not_depend_on_the_chunk_size(tmp_path):
    path = _write(tmp_path, _rows())
    small = build_email_user_day(path, chunksize=1, progress=lambda *_: None)
    large = build_email_user_day(path, chunksize=100, progress=lambda *_: None)
    pd.testing.assert_frame_equal(small, large)


def test_a_file_out_of_chronological_order_is_rejected(tmp_path):
    rows = [_row("01/05/2010 10:00:00", "U1", "a@dtaa.com"), _row("01/04/2010 10:00:00", "U1", "a@dtaa.com"),
            _row("01/06/2010 10:00:00", "U1", "a@dtaa.com")]
    with pytest.raises(ValueError):
        build_email_user_day(_write(tmp_path, rows), chunksize=1, progress=lambda *_: None)
