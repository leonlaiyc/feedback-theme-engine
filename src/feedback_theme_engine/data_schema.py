"""Review data schema validation utilities."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_REVIEW_COLUMNS: tuple[str, ...] = ("review_id", "product_id", "text", "rating")


def validate_review_dataframe(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_REVIEW_COLUMNS,
) -> pd.DataFrame:
    """Validate that a review dataframe has the expected Phase 0 structure.

    The function intentionally performs only lightweight checks. Later phases can
    add stricter validation once ingestion requirements are better defined.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required review column(s): {missing}")

    if "rating" in df.columns and not pd.api.types.is_numeric_dtype(df["rating"]):
        raise TypeError("rating column must be numeric")

    return df

