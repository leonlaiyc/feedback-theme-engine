import pandas as pd
import pytest

from feedback_theme_engine.theme_representation import (
    build_cluster_frame,
    extract_tfidf_keywords_by_cluster,
    get_representative_reviews,
)


def test_build_cluster_frame_preserves_noise_rows() -> None:
    frame = build_cluster_frame(
        ["r1", "r2", "r3"],
        ["battery failed", "great scent", "late delivery"],
        [0, 1, -1],
    )

    assert frame.to_dict(orient="records") == [
        {"review_id": "r1", "review_text": "battery failed", "cluster_id": 0},
        {"review_id": "r2", "review_text": "great scent", "cluster_id": 1},
        {"review_id": "r3", "review_text": "late delivery", "cluster_id": -1},
    ]


def test_build_cluster_frame_raises_for_length_mismatch() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        build_cluster_frame(["r1"], ["text one", "text two"], [0])


def test_get_representative_reviews_sorts_deterministically_and_skips_noise() -> None:
    frame = pd.DataFrame(
        {
            "review_id": ["r3", "r1", "r2", "r4"],
            "review_text": ["third", "first", "second", "noise"],
            "cluster_id": [0, 0, 0, -1],
        }
    )

    representatives = get_representative_reviews(frame, max_per_cluster=2)

    assert representatives[["review_id", "example_rank"]].to_dict(orient="records") == [
        {"review_id": "r1", "example_rank": 1},
        {"review_id": "r2", "example_rank": 2},
    ]
    assert -1 not in representatives["cluster_id"].tolist()


def test_get_representative_reviews_handles_all_noise() -> None:
    frame = build_cluster_frame(["r1"], ["noise text"], [-1])

    representatives = get_representative_reviews(frame)

    assert representatives.empty
    assert list(representatives.columns) == [
        "cluster_id",
        "example_rank",
        "review_id",
        "review_text",
    ]


def test_extract_tfidf_keywords_by_cluster_on_toy_texts() -> None:
    frame = build_cluster_frame(
        ["r1", "r2", "r3", "r4"],
        [
            "battery battery charge lasts",
            "battery charge failed",
            "fragrance scent fresh",
            "fragrance scent bottle",
        ],
        [0, 0, 1, 1],
    )

    keywords = extract_tfidf_keywords_by_cluster(frame, top_n=2)

    cluster_zero_keywords = keywords.loc[keywords["cluster_id"] == 0, "keyword"].tolist()
    cluster_one_keywords = keywords.loc[keywords["cluster_id"] == 1, "keyword"].tolist()
    assert "battery" in cluster_zero_keywords
    assert "fragrance" in cluster_one_keywords


def test_extract_tfidf_keywords_returns_empty_for_all_noise() -> None:
    frame = build_cluster_frame(["r1"], ["only noise"], [-1])

    keywords = extract_tfidf_keywords_by_cluster(frame)

    assert keywords.empty
    assert list(keywords.columns) == ["cluster_id", "keyword", "score", "rank"]
