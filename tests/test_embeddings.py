import json

import numpy as np
import pandas as pd
import pytest

from feedback_theme_engine.embeddings import (
    EmbeddingConfig,
    compute_similarity_examples,
    embed_reviews_dataframe,
    embed_texts,
    save_embeddings,
)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences,
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        self.calls.append(list(sentences))
        vectors = np.array([self._vector_for(sentence) for sentence in sentences], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors

    @staticmethod
    def _vector_for(sentence: str) -> list[float]:
        lowered = sentence.lower()
        if "damaged" in lowered or "broken" in lowered:
            return [1.0, 0.0, 0.0]
        if "fragrance" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_embed_texts_handles_empty_input_without_model_call() -> None:
    model = FakeEmbeddingModel()

    embeddings = embed_texts([], model, EmbeddingConfig())

    assert embeddings.shape == (0, 0)
    assert model.calls == []


def test_embed_texts_raises_for_non_string_text() -> None:
    model = FakeEmbeddingModel()

    with pytest.raises(TypeError, match="all texts must be strings"):
        embed_texts(["valid", None], model, EmbeddingConfig())  # type: ignore[list-item]


def test_embed_reviews_dataframe_returns_ids_and_embeddings_without_mutation() -> None:
    df = pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "review_text": ["the product arrived damaged", "I love the fragrance"],
            "extra": ["kept", "kept"],
        }
    )
    original = df.copy(deep=True)

    ids, embeddings = embed_reviews_dataframe(df, FakeEmbeddingModel(), EmbeddingConfig())

    assert ids == ["r1", "r2"]
    assert embeddings.shape == (2, 3)
    pd.testing.assert_frame_equal(df, original)


def test_embed_reviews_dataframe_raises_for_missing_required_column() -> None:
    df = pd.DataFrame({"review_id": ["r1"], "text": ["missing configured text column"]})

    with pytest.raises(ValueError, match="review_text"):
        embed_reviews_dataframe(df, FakeEmbeddingModel(), EmbeddingConfig())


def test_save_embeddings_creates_files_under_processed_dir(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "data" / "processed" / "embeddings"
    ids = ["r1", "r2"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    ids_path, embeddings_path = save_embeddings(ids, embeddings, output_dir, "toy")

    assert ids_path.exists()
    assert embeddings_path.exists()
    assert json.loads(ids_path.read_text(encoding="utf-8")) == ids
    np.testing.assert_array_equal(np.load(embeddings_path), embeddings)


def test_save_embeddings_rejects_output_outside_processed_dir(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="data/processed"):
        save_embeddings(["r1"], np.array([[1.0]], dtype=np.float32), tmp_path / "outside", "toy")


def test_similarity_examples_rank_related_texts_higher() -> None:
    texts = [
        "the product arrived damaged",
        "the item was broken when delivered",
        "I love the fragrance",
    ]

    similarity = compute_similarity_examples(texts, FakeEmbeddingModel(), EmbeddingConfig())

    assert similarity.shape == (3, 3)
    assert similarity[0, 1] > similarity[0, 2]
