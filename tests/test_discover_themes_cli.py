import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "discover_themes.py"
SPEC = importlib.util.spec_from_file_location("discover_themes", SCRIPT_PATH)
assert SPEC is not None
discover_themes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(discover_themes)


def test_discover_themes_cli_defaults_to_ignored_paths() -> None:
    args = discover_themes.parse_args([])

    assert args.reviews_input == Path("data/processed/reviews_sample.parquet")
    assert args.ids_input == Path("data/processed/embeddings/reviews_sample_ids.json")
    assert args.embeddings_input == Path("data/processed/embeddings/reviews_sample_embeddings.npy")
    assert args.output_dir == Path("data/processed/themes")


def test_discover_themes_cli_accepts_modeling_parameters() -> None:
    args = discover_themes.parse_args(
        [
            "--n-components",
            "3",
            "--n-neighbors",
            "10",
            "--min-dist",
            "0.2",
            "--metric",
            "euclidean",
            "--min-cluster-size",
            "4",
            "--min-samples",
            "2",
            "--top-keywords",
            "7",
            "--representatives-per-cluster",
            "3",
        ]
    )

    assert args.n_components == 3
    assert args.n_neighbors == 10
    assert args.min_dist == 0.2
    assert args.metric == "euclidean"
    assert args.min_cluster_size == 4
    assert args.min_samples == 2
    assert args.top_keywords == 7
    assert args.representatives_per_cluster == 3


def test_lookup_review_texts_preserves_embedding_id_order() -> None:
    reviews = pd.DataFrame(
        {
            "review_id": ["r2", "r1"],
            "review_text": ["second", "first"],
        }
    )

    texts = discover_themes._lookup_review_texts(reviews, ["r1", "r2"])

    assert texts == ["first", "second"]


def test_lookup_review_texts_raises_for_missing_id() -> None:
    reviews = pd.DataFrame({"review_id": ["r1"], "review_text": ["first"]})

    with pytest.raises(ValueError, match="missing 1 embedded review"):
        discover_themes._lookup_review_texts(reviews, ["r1", "r2"])


def test_theme_output_dir_must_be_under_processed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="data/processed"):
        discover_themes._resolve_theme_output_dir(tmp_path / "outside")
