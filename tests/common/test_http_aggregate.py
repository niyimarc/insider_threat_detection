import pandas as pd

from src.common.preprocessing.http_aggregate import extract_domain, aggregate_chunk, combine_counts


def test_extract_domain_handles_common_url_shapes():
    urls = pd.Series([
        "http://www.Example.com/a/b?x=1",
        "https://news.site.org",
        "http://example.com:8080/x",
        "http://www2.example.com/",
    ])
    assert extract_domain(urls).tolist() == [
        "example.com", "news.site.org", "example.com", "example.com",
    ]


def _chunk():
    return pd.DataFrame({
        "date": ["01/04/2010 08:15:00", "01/04/2010 08:45:00", "01/04/2010 23:10:00", "01/05/2010 08:00:00"],
        "user": ["U1", "U1", "U1", "U2"],
        "url": ["http://a.com/1", "http://a.com/2", "http://b.com/", "http://a.com/"],
    })


def test_aggregate_chunk_counts_by_domain_and_hour():
    domain, hour = aggregate_chunk(_chunk())
    d = domain.set_index(["user", "domain"])["visits"]
    assert d[("U1", "a.com")] == 2
    assert d[("U1", "b.com")] == 1
    assert d[("U2", "a.com")] == 1
    h = hour.set_index(["user", "hour"])["visits"]
    assert h[("U1", 8)] == 2
    assert h[("U1", 23)] == 1


def test_both_tables_have_the_same_total():
    domain, hour = aggregate_chunk(_chunk())
    assert domain["visits"].sum() == hour["visits"].sum() == 4


def test_unparsable_rows_are_dropped():
    bad = pd.DataFrame({"date": ["not a date", "01/04/2010 08:00:00"], "user": ["U1", "U1"],
                        "url": ["http://a.com", "http://a.com"]})
    domain, _ = aggregate_chunk(bad)
    assert domain["visits"].sum() == 1


def test_combine_counts_sums_shared_keys():
    a, _ = aggregate_chunk(_chunk())
    combined = combine_counts([a, a], ["user", "day", "domain"])
    assert combined["visits"].sum() == 2 * a["visits"].sum()
    assert len(combined) == len(a)
