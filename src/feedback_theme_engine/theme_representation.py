"""Representation helpers for candidate feedback theme clusters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


CLUSTER_FRAME_COLUMNS: tuple[str, ...] = ("review_id", "review_text", "cluster_id")
REPRESENTATIVE_COLUMNS: tuple[str, ...] = (
    "cluster_id",
    "example_rank",
    "review_id",
    "review_text",
)
KEYWORD_COLUMNS: tuple[str, ...] = ("cluster_id", "keyword", "score", "rank")


def build_cluster_frame(
    review_ids: Sequence[str],
    review_texts: Sequence[str],
    labels: Sequence[int] | np.ndarray,
) -> pd.DataFrame:
    """Build a traceable dataframe of review IDs, text, and cluster labels."""
    ids = list(review_ids)
    texts = list(review_texts)
    label_array = np.asarray(labels)

    if label_array.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if len(ids) != len(texts) or len(ids) != len(label_array):
        raise ValueError("review_ids, review_texts, and labels must have matching lengths")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("review_texts must contain only strings")

    return pd.DataFrame(
        {
            "review_id": [str(review_id) for review_id in ids],
            "review_text": texts,
            "cluster_id": label_array.astype(int),
        },
        columns=list(CLUSTER_FRAME_COLUMNS),
    )


def get_representative_reviews(
    cluster_frame: pd.DataFrame,
    max_per_cluster: int = 5,
) -> pd.DataFrame:
    """Select deterministic representative review examples per non-noise cluster."""
    _validate_cluster_frame(cluster_frame)
    if max_per_cluster <= 0:
        raise ValueError("max_per_cluster must be greater than zero")

    non_noise = cluster_frame.loc[cluster_frame["cluster_id"] != -1].copy()
    if non_noise.empty:
        return pd.DataFrame(columns=list(REPRESENTATIVE_COLUMNS))

    non_noise["review_id_sort"] = non_noise["review_id"].astype(str)
    sorted_frame = non_noise.sort_values(["cluster_id", "review_id_sort"], kind="mergesort")
    representatives = sorted_frame.groupby("cluster_id", sort=True).head(max_per_cluster).copy()
    representatives["example_rank"] = representatives.groupby("cluster_id").cumcount() + 1
    return representatives.loc[:, list(REPRESENTATIVE_COLUMNS)].reset_index(drop=True)


def extract_tfidf_keywords_by_cluster(
    cluster_frame: pd.DataFrame,
    max_features: int = 5_000,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extract TF-IDF keywords for non-noise clusters as explainability metadata."""
    _validate_cluster_frame(cluster_frame)
    if max_features <= 0:
        raise ValueError("max_features must be greater than zero")
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    non_noise = cluster_frame.loc[cluster_frame["cluster_id"] != -1]
    if non_noise.empty:
        return pd.DataFrame(columns=list(KEYWORD_COLUMNS))

    cluster_docs = (
        non_noise.sort_values(["cluster_id", "review_id"], kind="mergesort")
        .groupby("cluster_id", sort=True)["review_text"]
        .apply(lambda values: " ".join(values))
    )

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - exercised by users locally
        raise ImportError(
            "Install scikit-learn to extract cluster keywords: "
            "py -3 -m pip install -r requirements.txt"
        ) from exc

    try:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        matrix = vectorizer.fit_transform(cluster_docs.tolist())
    except ValueError:
        return pd.DataFrame(columns=list(KEYWORD_COLUMNS))

    feature_names = vectorizer.get_feature_names_out()
    rows: list[dict[str, object]] = []
    for row_index, cluster_id in enumerate(cluster_docs.index.tolist()):
        scores = matrix[row_index].toarray().ravel()
        ranked_indices = sorted(
            np.flatnonzero(scores),
            key=lambda index: (-scores[index], feature_names[index]),
        )[:top_n]
        for rank, feature_index in enumerate(ranked_indices, start=1):
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "keyword": str(feature_names[feature_index]),
                    "score": float(scores[feature_index]),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows, columns=list(KEYWORD_COLUMNS))


def _validate_cluster_frame(cluster_frame: pd.DataFrame) -> None:
    if not isinstance(cluster_frame, pd.DataFrame):
        raise TypeError("cluster_frame must be a pandas DataFrame")
    missing_columns = [column for column in CLUSTER_FRAME_COLUMNS if column not in cluster_frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing cluster frame column(s): {missing}")
