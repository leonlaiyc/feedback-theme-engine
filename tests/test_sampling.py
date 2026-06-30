import pandas as pd

from feedback_theme_engine.sampling import reservoir_sample_records, sample_reviews


def test_sample_reviews_is_deterministic_with_seed() -> None:
    df = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(20)],
            "review_text": [f"Review text {i}" for i in range(20)],
        }
    )

    first = sample_reviews(df, sample_size=5, seed=123)
    second = sample_reviews(df, sample_size=5, seed=123)

    assert first["review_id"].tolist() == second["review_id"].tolist()


def test_sample_reviews_filters_by_minimum_text_length() -> None:
    df = pd.DataFrame(
        {
            "review_id": ["short", "long"],
            "review_text": ["ok", "This review has enough detail."],
        }
    )

    sampled = sample_reviews(df, sample_size=5, seed=123, min_review_text_length=10)

    assert sampled["review_id"].tolist() == ["long"]


def test_reservoir_sample_records_is_deterministic() -> None:
    records = [{"text": f"Review text {i}", "id": f"r{i}"} for i in range(30)]

    first = reservoir_sample_records(records, sample_size=5, seed=7)
    second = reservoir_sample_records(records, sample_size=5, seed=7)

    assert [record["id"] for record in first] == [record["id"] for record in second]
