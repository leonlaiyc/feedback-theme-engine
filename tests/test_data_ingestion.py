import sys
from types import ModuleType

import pandas as pd
import pytest

from feedback_theme_engine.data_ingestion import (
    IngestionConfig,
    iter_amazon_reviews_category,
    normalize_review_schema,
)


def test_normalize_review_schema_maps_amazon_review_columns() -> None:
    raw = pd.DataFrame(
        {
            "asin": ["B001"],
            "parent_asin": ["P001"],
            "user_id": ["U001"],
            "text": ["  Works\nwell.  "],
            "title": ["  Great item  "],
            "rating": ["5.0"],
            "helpful_vote": ["2"],
            "verified_purchase": ["true"],
            "timestamp": [1_700_000_000_000],
        }
    )

    normalized = normalize_review_schema(raw)

    assert list(normalized.columns) == [
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
    ]
    assert normalized.loc[0, "review_id"].startswith("generated_")
    assert normalized.loc[0, "product_id"] == "B001"
    assert normalized.loc[0, "parent_product_id"] == "P001"
    assert normalized.loc[0, "review_text"] == "Works well."
    assert normalized.loc[0, "review_title"] == "Great item"
    assert normalized.loc[0, "rating"] == 5.0
    assert normalized.loc[0, "helpful_vote"] == 2
    assert normalized.loc[0, "verified_purchase"] == True


def test_normalize_review_schema_preserves_existing_review_id() -> None:
    raw = pd.DataFrame(
        {
            "review_id": ["r1"],
            "asin": ["B001"],
            "text": ["Works well."],
            "rating": [4],
        }
    )

    normalized = normalize_review_schema(raw)

    assert normalized.loc[0, "review_id"] == "r1"
    assert normalized.loc[0, "review_text"] == "Works well."


def test_ingestion_config_does_not_trust_remote_code_by_default() -> None:
    assert IngestionConfig().trust_remote_code is False


def test_iter_amazon_reviews_category_passes_trust_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}
    fake_datasets = ModuleType("datasets")

    class FakeDownloadConfig:
        def __init__(self, *, local_files_only: bool) -> None:
            self.local_files_only = local_files_only

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return [{"text": "Synthetic review.", "rating": 5}]

    fake_datasets.DownloadConfig = FakeDownloadConfig
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    records = iter_amazon_reviews_category(
        "raw_review_All_Beauty",
        local_files_only=True,
        trust_remote_code=True,
    )

    assert list(records) == [{"text": "Synthetic review.", "rating": 5}]
    assert calls["kwargs"]["trust_remote_code"] is True
    assert calls["kwargs"]["download_config"].local_files_only is True


def test_iter_amazon_reviews_category_explains_remote_code_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_datasets = ModuleType("datasets")

    class FakeDownloadConfig:
        def __init__(self, *, local_files_only: bool) -> None:
            self.local_files_only = local_files_only

    def fake_load_dataset(*args, **kwargs):
        raise ValueError("Use trust_remote_code=True to allow remote code")

    fake_datasets.DownloadConfig = FakeDownloadConfig
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="--trust-remote-code"):
        iter_amazon_reviews_category("raw_review_All_Beauty")
