import pandas as pd
import pytest

from feedback_theme_engine.data_validation import NORMALIZED_REVIEW_COLUMNS, validate_reviews


def _valid_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r2"],
            "product_id": ["p1", "p2", "p2"],
            "parent_product_id": ["pp1", "pp2", "pp2"],
            "user_id": ["u1", "u2", "u3"],
            "review_text": ["Works well.", "Would buy again.", "Fine for travel."],
            "review_title": ["Good", "Great", "Fine"],
            "rating": [5, 4, None],
            "helpful_vote": [1, 0, 3],
            "verified_purchase": [True, False, True],
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"],
                utc=True,
            ),
        }
    )


def test_validate_reviews_returns_summary_metrics() -> None:
    summary = validate_reviews(_valid_reviews())

    assert summary.row_count == 3
    assert summary.missing_text_count == 0
    assert summary.missing_rating_count == 1
    assert summary.rating_distribution == {"4.0": 1, "5.0": 1}
    assert summary.duplicate_review_id_count == 1
    assert summary.verified_purchase_distribution == {"False": 1, "True": 2}


def test_validate_reviews_raises_for_missing_required_column() -> None:
    df = _valid_reviews().drop(columns=["review_text"])

    with pytest.raises(ValueError, match="review_text"):
        validate_reviews(df, required_columns=NORMALIZED_REVIEW_COLUMNS)


def test_validate_reviews_raises_for_rating_outside_range() -> None:
    df = _valid_reviews()
    df.loc[0, "rating"] = 6

    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_reviews(df)


def test_validate_reviews_raises_for_empty_review_text_after_cleaning() -> None:
    df = _valid_reviews()
    df.loc[0, "review_text"] = "   \n\t  "

    with pytest.raises(ValueError, match="review_text"):
        validate_reviews(df)
