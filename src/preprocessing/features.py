import pandas as pd

FILE_TYPE_BY_HEADER = {
    "D0-CF-11-E0-A1-B1-1A-E1": "office_document",
    "25-50-44-46-2D": "pdf",
    "50-4B-03-04-14": "zip",
    "FF-D8": "jpg",
    "4D-5A-90-00-03-00-00-00-04-00-00-00-FF-FF-00-00-B8-00-00-00-00-00-00-00-40-00": "exe",
}


def extract_file_type(file_df):
    """
    Categorise each file event by its real type, inferred from the
    hexadecimal header at the start of `content` — not from the filename
    extension, which the release's own documentation implies can be
    inconsistent with the file's actual type.
    """
    file_df = file_df.copy()
    header = file_df["content"].str.split(" ").str[0]
    file_df["file_type"] = header.map(FILE_TYPE_BY_HEADER).fillna("other")
    return file_df