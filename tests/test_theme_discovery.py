import sys
from types import ModuleType

import numpy as np
import pytest

from feedback_theme_engine.theme_discovery import (
    ThemeDiscoveryConfig,
    cluster_reduced_embeddings,
    discover_themes_from_embeddings,
    reduce_embeddings_umap,
)


def test_reduce_embeddings_umap_uses_config_with_fake_umap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}
    fake_umap = ModuleType("umap")

    class FakeUMAP:
        def __init__(self, **kwargs) -> None:
            calls["kwargs"] = kwargs

        def fit_transform(self, embeddings) -> np.ndarray:
            calls["embeddings_shape"] = embeddings.shape
            return embeddings[:, :2]

    fake_umap.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)
    embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    config = ThemeDiscoveryConfig(n_components=2, n_neighbors=2, min_dist=0.1)

    reduced = reduce_embeddings_umap(embeddings, config)

    assert reduced.shape == (3, 2)
    assert calls["kwargs"]["n_components"] == 2
    assert calls["kwargs"]["n_neighbors"] == 2
    assert calls["kwargs"]["min_dist"] == 0.1
    assert calls["embeddings_shape"] == (3, 3)


def test_cluster_reduced_embeddings_uses_fake_hdbscan(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}
    fake_hdbscan = ModuleType("hdbscan")

    class FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            calls["kwargs"] = kwargs

        def fit_predict(self, reduced_embeddings) -> np.ndarray:
            calls["shape"] = reduced_embeddings.shape
            return np.array([-1, 0, 0], dtype=int)

    fake_hdbscan.HDBSCAN = FakeHDBSCAN
    monkeypatch.setitem(sys.modules, "hdbscan", fake_hdbscan)
    config = ThemeDiscoveryConfig(min_cluster_size=2, min_samples=1)

    labels = cluster_reduced_embeddings(np.ones((3, 2), dtype=np.float32), config)

    assert labels.tolist() == [-1, 0, 0]
    assert calls["kwargs"]["min_cluster_size"] == 2
    assert calls["kwargs"]["min_samples"] == 1
    assert calls["shape"] == (3, 2)


def test_discover_themes_from_embeddings_combines_fake_umap_and_hdbscan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_umap = ModuleType("umap")
    fake_hdbscan = ModuleType("hdbscan")

    class FakeUMAP:
        def __init__(self, **kwargs) -> None:
            pass

        def fit_transform(self, embeddings) -> np.ndarray:
            return embeddings[:, :2]

    class FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            pass

        def fit_predict(self, reduced_embeddings) -> np.ndarray:
            return np.array([0, 0, 1], dtype=int)

    fake_umap.UMAP = FakeUMAP
    fake_hdbscan.HDBSCAN = FakeHDBSCAN
    monkeypatch.setitem(sys.modules, "umap", fake_umap)
    monkeypatch.setitem(sys.modules, "hdbscan", fake_hdbscan)

    reduced, labels = discover_themes_from_embeddings(
        np.ones((3, 4), dtype=np.float32),
        ThemeDiscoveryConfig(n_neighbors=2, min_cluster_size=2),
    )

    assert reduced.shape == (3, 2)
    assert labels.tolist() == [0, 0, 1]


def test_reduce_embeddings_umap_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        reduce_embeddings_umap(np.array([1, 2, 3]), ThemeDiscoveryConfig())


def test_cluster_reduced_embeddings_rejects_invalid_min_cluster_size() -> None:
    with pytest.raises(ValueError, match="min_cluster_size"):
        cluster_reduced_embeddings(
            np.ones((3, 2), dtype=np.float32),
            ThemeDiscoveryConfig(min_cluster_size=1),
        )
