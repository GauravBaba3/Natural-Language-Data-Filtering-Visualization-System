from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


NUMERIC_LIKE_THRESHOLD = 0.8
SAMPLE_SIZE = 200


class PreprocessingError(Exception):
    pass


@dataclass(frozen=True)
class ColumnProfile:
    detected_type: str
    numeric_ratio: float
    non_null_count: int
    converted: bool


def detect_column_types(df: pd.DataFrame) -> Dict[str, ColumnProfile]:
    """
    Detect whether each column should be treated as numeric, datetime, or categorical.
    For object-like columns, inspect a sample and treat the column as numeric-like when
    most non-null values can be converted with `pd.to_numeric(..., errors="coerce")`.
    """
    profiles: Dict[str, ColumnProfile] = {}

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])

        if pd.api.types.is_numeric_dtype(series):
            profiles[col] = ColumnProfile(
                detected_type="numeric",
                numeric_ratio=1.0,
                non_null_count=non_null_count,
                converted=False,
            )
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            profiles[col] = ColumnProfile(
                detected_type="datetime",
                numeric_ratio=0.0,
                non_null_count=non_null_count,
                converted=False,
            )
            continue

        if non_null.empty:
            profiles[col] = ColumnProfile(
                detected_type="categorical",
                numeric_ratio=0.0,
                non_null_count=0,
                converted=False,
            )
            continue

        sample = non_null.astype(str).head(SAMPLE_SIZE)
        numeric_attempt = pd.to_numeric(sample, errors="coerce")
        numeric_ratio = float(numeric_attempt.notna().mean())

        detected_type = (
            "numeric_like_string"
            if numeric_ratio >= NUMERIC_LIKE_THRESHOLD
            else "categorical"
        )
        profiles[col] = ColumnProfile(
            detected_type=detected_type,
            numeric_ratio=numeric_ratio,
            non_null_count=non_null_count,
            converted=False,
        )

    return profiles


def convert_numeric_columns(
    df: pd.DataFrame, column_profiles: Dict[str, ColumnProfile] | None = None
) -> tuple[pd.DataFrame, Dict[str, ColumnProfile]]:
    """
    Convert numeric-like string columns to numeric.
    - Uses `pd.to_numeric(errors="coerce")`
    - Leaves fully non-numeric / categorical columns untouched
    - Raises a user-friendly error only if a column was classified as numeric-like but
      conversion unexpectedly produced zero valid numeric values from non-null inputs
    """
    profiles = column_profiles or detect_column_types(df)
    cleaned_df = df.copy()
    updated_profiles: Dict[str, ColumnProfile] = {}

    for col, profile in profiles.items():
        if profile.detected_type != "numeric_like_string":
            updated_profiles[col] = profile
            continue

        original_non_null = cleaned_df[col].dropna()
        converted = pd.to_numeric(cleaned_df[col], errors="coerce")
        converted_non_null_count = int(converted.notna().sum())

        if profile.non_null_count > 0 and converted_non_null_count == 0:
            raise PreprocessingError(
                f"Could not safely convert numeric-like column `{col}` to numeric."
            )

        cleaned_df[col] = converted
        updated_profiles[col] = ColumnProfile(
            detected_type="numeric",
            numeric_ratio=profile.numeric_ratio,
            non_null_count=len(original_non_null),
            converted=True,
        )

    return cleaned_df, updated_profiles


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, ColumnProfile]]:
    """
    End-to-end cleaning step for the filtering pipeline.
    """
    profiles = detect_column_types(df)
    return convert_numeric_columns(df, profiles)

