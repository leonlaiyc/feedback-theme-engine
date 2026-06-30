"""Local ingestion helpers for Amazon Reviews 2023 review subsets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd

from feedback_theme_engine.data_validation import (
    NORMALIZED_REVIEW_COLUMNS,
    ValidationSummary,
    validate_reviews,
)
from feedback_theme_engine.sampling import reservoir_sample_records
from feedback_theme_engine.text_cleaning import basic_clean_text

AMAZON_REVIEWS_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORY = "raw_review_All_Beauty"
DEFAULT_OUTPUT_PATH = Path("data/processed/reviews_sample.parquet")

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "review_id": ("review_id", "id"),
    "product_id": ("product_id", "asin"),
    "parent_product_id": ("parent_product_id", "parent_asin"),
    "user_id": ("user_id",),
    "review_text": ("review_text", "text"),
    "review_title": ("review_title", "title"),
    "rating": ("rating",),
    "helpful_vote": ("helpful_vote",),
    "verified_purchase": ("verified_purchase",),
    "timestamp": ("timestamp",),
}


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for a local Phase 1 review ingestion run."""

    category: str = DEFAULT_CATEGORY
    sample_size: int = 5_000
    seed: int = 42
    output_path: Path = DEFAULT_OUTPUT_PATH
    split: str = "full"
    min_review_text_length: int | None = 1
    local_files_only: bool = False


def iter_amazon_reviews_category(
    category: str = DEFAULT_CATEGORY,
    *,
    split: str = "full",
    streaming: bool = True,
    local_files_only: bool = False,
) -> Iterable[Mapping[str, Any]]:
    """Return an iterable of raw Amazon Reviews 2023 records.

    The loader uses Hugging Face ``datasets`` in streaming mode by default so a
    full category does not need to be materialized in memory. Set
    ``local_files_only=True`` to require an already-populated local dataset cache.
    """
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by users locally
        raise ImportError(
            "Install the optional 'datasets' dependency to load Amazon Reviews "
            "2023 categories: py -3 -m pip install -r requirements.txt"
        ) from exc

    download_config = DownloadConfig(local_files_only=local_files_only)
    dataset = load_dataset(
        AMAZON_REVIEWS_DATASET,
        category,
        split=split,
        streaming=streaming,
        download_config=download_config,
    )
    return dataset


def normalize_review_schema(
    reviews: pd.DataFrame | Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Normalize raw review rows to the project Phase 1 review schema."""
    raw_df = reviews.copy() if isinstance(reviews, pd.DataFrame) else pd.DataFrame.from_records(reviews)
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("reviews must be a pandas DataFrame or iterable of mappings")

    normalized = pd.DataFrame(index=raw_df.index)
    for target_column, aliases in COLUMN_ALIASES.items():
        normalized[target_column] = _first_available_series(raw_df, aliases)

    normalized["review_text"] = normalized["review_text"].map(_clean_optional_text)
    normalized["review_title"] = normalized["review_title"].map(_clean_optional_text)
    normalized["rating"] = pd.to_numeric(normalized["rating"], errors="coerce")
    normalized["helpful_vote"] = pd.to_numeric(normalized["helpful_vote"], errors="coerce")
    normalized["verified_purchase"] = normalized["verified_purchase"].map(_normalize_bool).astype("boolean")
    normalized["timestamp"] = _normalize_timestamp(normalized["timestamp"])

    missing_review_id = normalized["review_id"].isna() | (normalized["review_id"].astype("string").str.strip() == "")
    if missing_review_id.any():
        normalized.loc[missing_review_id, "review_id"] = normalized.loc[missing_review_id].apply(
            _generate_review_id,
            axis=1,
        )

    for column in ("review_id", "product_id", "parent_product_id", "user_id"):
        normalized[column] = normalized[column].astype("string")

    return normalized.loc[:, list(NORMALIZED_REVIEW_COLUMNS)]


def prepare_reviews_dataset(config: IngestionConfig) -> tuple[pd.DataFrame, ValidationSummary]:
    """Load, sample, normalize, validate, and save one review category subset."""
    if config.sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    raw_records = iter_amazon_reviews_category(
        config.category,
        split=config.split,
        streaming=True,
        local_files_only=config.local_files_only,
    )
    sampled_records = reservoir_sample_records(
        raw_records,
        sample_size=config.sample_size,
        seed=config.seed,
        min_review_text_length=config.min_review_text_length,
    )
    normalized_reviews = normalize_review_schema(sampled_records)
    summary = validate_reviews(normalized_reviews)
    save_processed_reviews(normalized_reviews, config.output_path)
    return normalized_reviews, summary


def save_processed_reviews(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save processed reviews under ``data/processed`` in a local-only format."""
    resolved_output = _resolve_processed_output_path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    suffix = resolved_output.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(resolved_output, index=False)
    elif suffix in {".jsonl", ".json"}:
        df.to_json(resolved_output, orient="records", lines=True, force_ascii=False)
    elif suffix == ".csv":
        df.to_csv(resolved_output, index=False)
    else:
        raise ValueError("output path must end in .parquet, .jsonl, .json, or .csv")

    return resolved_output


def _first_available_series(raw_df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    for alias in aliases:
        if alias in raw_df.columns:
            return raw_df[alias]
    return pd.Series(pd.NA, index=raw_df.index)


def _clean_optional_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return basic_clean_text(str(value))


def _normalize_bool(value: Any) -> bool | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return pd.NA


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().any():
        return pd.to_datetime(numeric_values, unit="ms", errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def _generate_review_id(row: pd.Series) -> str:
    stable_fields = [
        row.get("product_id", ""),
        row.get("parent_product_id", ""),
        row.get("user_id", ""),
        row.get("timestamp", ""),
        row.get("review_title", ""),
        row.get("review_text", ""),
        row.get("rating", ""),
    ]
    fingerprint = "|".join("" if pd.isna(value) else str(value) for value in stable_fields)
    return f"generated_{sha1(fingerprint.encode('utf-8')).hexdigest()[:16]}"


def _resolve_processed_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    resolved_path = path.resolve(strict=False)
    processed_dir = (Path.cwd() / "data" / "processed").resolve(strict=False)
    try:
        resolved_path.relative_to(processed_dir)
    except ValueError as exc:
        raise ValueError("processed review output must be written under data/processed/") from exc
    return resolved_path
