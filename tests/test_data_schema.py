import pandas as pd
import pytest

from feedback_theme_engine.data_schema import validate_review_dataframe


def test_validate_review_dataframe_accepts_required_columns() -> None:
    df = pd.DataFrame(
        {
            "review_id": ["r1"],
            "product_id": ["p1"],
            "text": ["Works well for my use case."],
            "rating": [5],
        }
    )

    assert validate_review_dataframe(df) is df


def test_validate_review_dataframe_raises_for_missing_column() -> None:
    df = pd.DataFrame(
        {
            "review_id": ["r1"],
            "product_id": ["p1"],
            "text": ["Missing a rating."],
        }
    )

    with pytest.raises(ValueError, match="rating"):
        validate_review_dataframe(df)


def test_validate_review_dataframe_raises_for_non_numeric_rating() -> None:
    df = pd.DataFrame(
        {
            "review_id": ["r1"],
            "product_id": ["p1"],
            "text": ["Rating is not numeric."],
            "rating": ["five"],
        }
    )

    with pytest.raises(TypeError, match="rating"):
        validate_review_dataframe(df)

