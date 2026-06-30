"""Validation utilities for normalized review data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from feedback_theme_engine.text_cleaning import basic_clean_text

NORMALIZED_REVIEW_COLUMNS: tuple[str, ...] = (
    "review_id",
    "product_id",
    "parent_product_id",
    "user_id",
    "review_text",
    "review_title",
    "rating",
    "helpful_vote",
    "verified_purchase",
    "timestamp",
)


@dataclass(frozen=True)
class ValidationSummary:
    """Small, serializable summary of Phase 1 review data quality checks."""

    row_count: int
    missing_text_count: int
    missing_rating_count: int
    rating_distribution: dict[str, int]
    duplicate_review_id_count: int | None
    verified_purchase_distribution: dict[str, int] | None

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as plain Python data for logging or JSON output."""
        return asdict(self)


def validate_reviews(
    df: pd.DataFrame,
    *,
    required_columns: tuple[str, ...] = NORMALIZED_REVIEW_COLUMNS,
) -> ValidationSummary:
    """Validate a normalized reviews dataframe and return quality metrics."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required review column(s): {missing}")

    cleaned_text = df["review_text"].map(_clean_text_for_validation)
    missing_text_count = int((cleaned_text == "").sum())
    if missing_text_count:
        raise ValueError("review_text must not be empty after basic cleaning")

    rating = pd.to_numeric(df["rating"], errors="coerce")
    invalid_rating = rating.notna() & ~rating.between(1, 5)
    if invalid_rating.any():
        raise ValueError("rating values must be between 1 and 5 when present")

    return ValidationSummary(
        row_count=int(len(df)),
        missing_text_count=missing_text_count,
        missing_rating_count=int(rating.isna().sum()),
        rating_distribution=_value_counts_as_strings(rating.dropna().sort_values()),
        duplicate_review_id_count=_duplicate_count(df, "review_id"),
        verified_purchase_distribution=_optional_value_counts(df, "verified_purchase"),
    )


def _clean_text_for_validation(value: Any) -> str:
    if pd.isna(value):
        return ""
    return basic_clean_text(str(value))


def _value_counts_as_strings(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _optional_value_counts(df: pd.DataFrame, column: str) -> dict[str, int] | None:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return {}
    return _value_counts_as_strings(series)


def _duplicate_count(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    return int(series.duplicated().sum())
