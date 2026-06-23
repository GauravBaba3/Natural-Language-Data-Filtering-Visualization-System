from __future__ import annotations

import io
from typing import Tuple

import pandas as pd


class DataLoadingError(Exception):
    pass


def load_dataframe(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Load an uploaded file (CSV or Excel) into a pandas DataFrame named `df`.
    Returns (df, file_name).
    """
    if uploaded_file is None:
        raise DataLoadingError("No file provided")

    file_name = getattr(uploaded_file, "name", "uploaded_file")
    suffix = str(file_name).lower().split(".")[-1]

    # UploadedFile gives bytes; pandas can read from BytesIO.
    data = uploaded_file.read()
    if not data:
        raise DataLoadingError("Uploaded file is empty")

    bio = io.BytesIO(data)

    if suffix in ("csv", "txt"):
        # Try a few common encodings to be resilient.
        last_err: Exception | None = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                bio.seek(0)
                df = pd.read_csv(bio, encoding=enc)
                return df, file_name
            except Exception as e:
                last_err = e
        raise DataLoadingError(f"Failed to read CSV: {last_err}")

    if suffix in ("xlsx", "xls"):
        try:
            df = pd.read_excel(bio, engine="openpyxl")
            return df, file_name
        except Exception as e:
            raise DataLoadingError(f"Failed to read Excel: {e}") from e

    raise DataLoadingError("Unsupported file type. Please upload CSV or Excel.")

