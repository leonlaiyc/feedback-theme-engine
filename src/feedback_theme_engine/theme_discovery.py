"""UMAP and HDBSCAN workflow for candidate theme discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThemeDiscoveryConfig:
    """Configuration for local UMAP + HDBSCAN theme discovery."""

    n_components: int = 5
    n_neighbors: int = 15
    min_dist: float = 0.0
    metric: str = "cosine"
    random_state: int = 42
    min_cluster_size: int = 15
    min_samples: int | None = None
    cluster_selection_method: str = "eom"


def reduce_embeddings_umap(
    embeddings: np.ndarray,
    config: ThemeDiscoveryConfig,
) -> np.ndarray:
    """Reduce high-dimensional embeddings with UMAP."""
    embedding_array = _validate_2d_array(embeddings, "embeddings")
    _validate_umap_config(config)

    try:
        from umap import UMAP
    except ImportError as exc:  # pragma: no cover - exercised by users locally
        raise ImportError(
            "Install umap-learn to reduce embeddings: "
            "py -3 -m pip install -r requirements.txt"
        ) from exc

    reducer = UMAP(
        n_components=config.n_components,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.random_state,
    )
    return np.asarray(reducer.fit_transform(embedding_array), dtype=np.float32)


def cluster_reduced_embeddings(
    reduced_embeddings: np.ndarray,
    config: ThemeDiscoveryConfig,
) -> np.ndarray:
    """Cluster reduced embeddings with HDBSCAN, preserving noise label ``-1``."""
    reduced_array = _validate_2d_array(reduced_embeddings, "reduced_embeddings")
    _validate_hdbscan_config(config)

    try:
        import hdbscan
    except ImportError as exc:  # pragma: no cover - exercised by users locally
        raise ImportError(
            "Install hdbscan to cluster embeddings: "
            "py -3 -m pip install -r requirements.txt"
        ) from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method=config.cluster_selection_method,
    )
    return np.asarray(clusterer.fit_predict(reduced_array), dtype=int)


def discover_themes_from_embeddings(
    embeddings: np.ndarray,
    config: ThemeDiscoveryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run UMAP reduction followed by HDBSCAN clustering."""
    reduced_embeddings = reduce_embeddings_umap(embeddings, config)
    labels = cluster_reduced_embeddings(reduced_embeddings, config)
    return reduced_embeddings, labels


def _validate_2d_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return array


def _validate_umap_config(config: ThemeDiscoveryConfig) -> None:
    if config.n_components <= 0:
        raise ValueError("n_components must be greater than zero")
    if config.n_neighbors <= 1:
        raise ValueError("n_neighbors must be greater than one")
    if config.min_dist < 0:
        raise ValueError("min_dist cannot be negative")


def _validate_hdbscan_config(config: ThemeDiscoveryConfig) -> None:
    if config.min_cluster_size <= 1:
        raise ValueError("min_cluster_size must be greater than one")
    if config.min_samples is not None and config.min_samples <= 0:
        raise ValueError("min_samples must be greater than zero when provided")
