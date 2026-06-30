"""Sentence embedding helpers for local semantic representation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_OUTPUT_DIR = Path("data/processed/embeddings")


class EmbeddingModel(Protocol):
    """Minimal interface needed from a sentence embedding model."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        """Encode text into vectors."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for local sentence embedding generation."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 32
    normalize_embeddings: bool = True
    text_column: str = "review_text"
    id_column: str = "review_id"
    output_dir: Path = DEFAULT_EMBEDDING_OUTPUT_DIR


def load_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    """Load a SentenceTransformer model.

    Model loading is isolated so tests can use fake models without downloading
    or importing model weights.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised by users locally
        raise ImportError(
            "Install sentence-transformers to generate local embeddings: "
            "py -3 -m pip install -r requirements.txt"
        ) from exc

    return SentenceTransformer(config.model_name)


def embed_texts(
    texts: Sequence[str],
    model: EmbeddingModel,
    config: EmbeddingConfig,
) -> np.ndarray:
    """Embed a sequence of text values and return a NumPy array."""
    text_list = _validate_texts(texts)
    if not text_list:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = model.encode(
        text_list,
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return _as_2d_float_array(embeddings)


def embed_reviews_dataframe(
    df: pd.DataFrame,
    model: EmbeddingModel,
    config: EmbeddingConfig,
) -> tuple[list[str], np.ndarray]:
    """Embed review text from a dataframe without mutating the input."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing_columns = [
        column
        for column in (config.id_column, config.text_column)
        if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required embedding column(s): {missing}")

    ids = [str(value) for value in df[config.id_column].tolist()]
    texts = df[config.text_column].tolist()
    embeddings = embed_texts(texts, model, config)
    return ids, embeddings


def save_embeddings(
    ids: Sequence[str],
    embeddings: np.ndarray,
    output_dir: str | Path,
    prefix: str,
) -> tuple[Path, Path]:
    """Save local embedding artifacts under ``data/processed``.

    The generated ``.json`` and ``.npy`` files are local artifacts and must not
    be tracked by git.
    """
    if not prefix or any(separator in prefix for separator in ("/", "\\")):
        raise ValueError("prefix must be a non-empty filename prefix")

    embedding_array = _as_2d_float_array(embeddings)
    if len(ids) != embedding_array.shape[0]:
        raise ValueError("ids length must match number of embedding rows")

    resolved_output_dir = _resolve_processed_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    ids_path = resolved_output_dir / f"{prefix}_ids.json"
    embeddings_path = resolved_output_dir / f"{prefix}_embeddings.npy"

    ids_path.write_text(
        json.dumps([str(identifier) for identifier in ids], indent=2),
        encoding="utf-8",
    )
    np.save(embeddings_path, embedding_array)
    return ids_path, embeddings_path


def compute_similarity_examples(
    texts: Sequence[str],
    model: EmbeddingModel,
    config: EmbeddingConfig,
) -> np.ndarray:
    """Return a pairwise cosine similarity matrix for a small text list."""
    embeddings = embed_texts(texts, model, config)
    if embeddings.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return _cosine_similarity_matrix(embeddings)


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")

    text_list = list(texts)
    invalid_types = [type(value).__name__ for value in text_list if not isinstance(value, str)]
    if invalid_types:
        raise TypeError("all texts must be strings")
    return text_list


def _as_2d_float_array(values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    return array


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / safe_norms
    return normalized @ normalized.T


def _resolve_processed_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    resolved_path = path.resolve(strict=False)
    processed_dir = (Path.cwd() / "data" / "processed").resolve(strict=False)
    try:
        resolved_path.relative_to(processed_dir)
    except ValueError as exc:
        raise ValueError("embedding output must be written under data/processed/") from exc
    return resolved_path
