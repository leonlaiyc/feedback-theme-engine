"""Simple deterministic sampling helpers for review data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from random import Random
from typing import Any

import pandas as pd

from feedback_theme_engine.text_cleaning import basic_clean_text


def sample_reviews(
    df: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
    min_review_text_length: int | None = None,
    text_column: str = "review_text",
) -> pd.DataFrame:
    """Return a deterministic dataframe sample after optional text filtering."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    if text_column not in df.columns:
        raise ValueError(f"Missing text column for sampling: {text_column}")

    filtered = filter_by_min_review_text_length(
        df,
        min_review_text_length=min_review_text_length,
        text_column=text_column,
    )
    if len(filtered) <= sample_size:
        return filtered.reset_index(drop=True)
    return filtered.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def filter_by_min_review_text_length(
    df: pd.DataFrame,
    *,
    min_review_text_length: int | None,
    text_column: str = "review_text",
) -> pd.DataFrame:
    """Filter reviews whose cleaned text is shorter than the requested length."""
    if min_review_text_length is None:
        return df.copy()
    if min_review_text_length < 0:
        raise ValueError("min_review_text_length cannot be negative")

    cleaned_lengths = df[text_column].map(_cleaned_text_length)
    return df.loc[cleaned_lengths >= min_review_text_length].copy()


def reservoir_sample_records(
    records: Iterable[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    min_review_text_length: int | None = None,
) -> list[Mapping[str, Any]]:
    """Deterministically sample from an iterable without loading all rows."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    rng = Random(seed)
    reservoir: list[Mapping[str, Any]] = []
    seen = 0

    for record in records:
        if min_review_text_length is not None and _record_text_length(record) < min_review_text_length:
            continue

        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(dict(record))
            continue

        replacement_index = rng.randrange(seen)
        if replacement_index < sample_size:
            reservoir[replacement_index] = dict(record)

    return reservoir


def _record_text_length(record: Mapping[str, Any]) -> int:
    value = record.get("review_text", record.get("text", ""))
    return _cleaned_text_length(value)


def _cleaned_text_length(value: Any) -> int:
    if pd.isna(value):
        return 0
    return len(basic_clean_text(str(value)))
