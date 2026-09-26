import pandas as pd
from src.preprocessing.features import extract_file_type


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