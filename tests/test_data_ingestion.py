import pandas as pd

from feedback_theme_engine.data_ingestion import normalize_review_schema


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
