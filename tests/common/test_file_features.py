import pandas as pd
from src.common.preprocessing.file_features import (
    extract_file_type,
    extract_file_extension,
    add_time_since_previous_event,
)


def test_extract_file_type_known_and_unknown_headers():
    file_df = pd.DataFrame({
        "content": [
            "D0-CF-11-E0-A1-B1-1A-E1 some content keywords here",
            "25-50-44-46-2D some content keywords here",
            "FF-D8 some content keywords here",
            "AA-BB-CC a completely unrecognised header",
        ],
    })

    result = extract_file_type(file_df)

    assert list(result["file_type"]) == ["office_document", "pdf", "jpg", "other"]


def test_extract_file_extension():
    file_df = pd.DataFrame({"filename": ["EYPC9Y08.doc", "AU75JV6U.JPG", "noext"]})
    result = extract_file_extension(file_df)
    assert list(result["file_extension"]) == ["doc", "jpg", "noext"]


def test_time_since_previous_event_respects_user_boundaries():
    file_df = pd.DataFrame({
        "user": ["AAA0001", "AAA0001", "BBB0002"],
        "date": pd.to_datetime([
            "2010-06-01 10:00:00",
            "2010-06-01 10:05:00",  # 5 minutes after AAA0001's first event
            "2010-06-01 09:00:00",  # BBB0002's first event — earlier in time,
                                     # but a DIFFERENT user, so this must be 0,
                                     # not a negative number from comparing
                                     # against AAA0001's second event.
        ]),
    })

    result = add_time_since_previous_event(file_df)
    result = result.set_index("user")

    assert result.loc["AAA0001", "seconds_since_last_event"].tolist() == [0, 300]
    assert result.loc["BBB0002", "seconds_since_last_event"] == 0
