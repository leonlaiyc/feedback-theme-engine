"""Run local UMAP + HDBSCAN candidate theme discovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feedback_theme_engine.cluster_diagnostics import summarize_clusters  # noqa: E402
from feedback_theme_engine.theme_discovery import (  # noqa: E402
    ThemeDiscoveryConfig,
    discover_themes_from_embeddings,
)
from feedback_theme_engine.theme_representation import (  # noqa: E402
    build_cluster_frame,
    extract_tfidf_keywords_by_cluster,
    get_representative_reviews,
)

DEFAULT_REVIEWS_INPUT = Path("data/processed/reviews_sample.parquet")
DEFAULT_IDS_INPUT = Path("data/processed/embeddings/reviews_sample_ids.json")
DEFAULT_EMBEDDINGS_INPUT = Path("data/processed/embeddings/reviews_sample_embeddings.npy")
DEFAULT_THEME_OUTPUT_DIR = Path("data/processed/themes")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover candidate feedback theme clusters from local embeddings.",
    )
    parser.add_argument("--reviews-input", type=Path, default=DEFAULT_REVIEWS_INPUT)
    parser.add_argument("--ids-input", type=Path, default=DEFAULT_IDS_INPUT)
    parser.add_argument("--embeddings-input", type=Path, default=DEFAULT_EMBEDDINGS_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THEME_OUTPUT_DIR)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.0)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--top-keywords", type=int, default=10)
    parser.add_argument("--representatives-per-cluster", type=int, default=5)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = ThemeDiscoveryConfig(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    ids = _read_ids(args.ids_input)
    embeddings = _read_embeddings(args.embeddings_input)
    if len(ids) != embeddings.shape[0]:
        raise ValueError("number of IDs must match number of embedding rows")

    reviews = _read_reviews(args.reviews_input)
    review_texts = _lookup_review_texts(reviews, ids)

    _reduced_embeddings, labels = discover_themes_from_embeddings(embeddings, config)
    cluster_frame = build_cluster_frame(ids, review_texts, labels)
    diagnostics = summarize_clusters(labels)
    representatives = get_representative_reviews(
        cluster_frame,
        max_per_cluster=args.representatives_per_cluster,
    )
    keywords = extract_tfidf_keywords_by_cluster(
        cluster_frame,
        top_n=args.top_keywords,
    )

    output_paths = _write_theme_outputs(
        cluster_frame,
        diagnostics,
        representatives,
        keywords,
        args.output_dir,
    )

    print(f"Embeddings loaded: {embeddings.shape[0]}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of clusters: {diagnostics['number_of_clusters']}")
    print(f"Noise count: {diagnostics['noise_count']}")
    print(f"Noise share: {diagnostics['noise_share']:.3f}")
    for label, path in output_paths.items():
        print(f"{label}: {path}")
    print("Generated theme outputs are ignored local artifacts and should not be committed.")


def _read_ids(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("IDs input must contain a JSON list")
    return [str(value) for value in values]


def _read_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError("embeddings input must be a 2D array")
    return embeddings.astype(np.float32, copy=False)


def _read_reviews(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("reviews input must end in .parquet, .jsonl, .json, or .csv")


def _lookup_review_texts(reviews: pd.DataFrame, ids: list[str]) -> list[str]:
    required_columns = {"review_id", "review_text"}
    missing_columns = sorted(required_columns.difference(reviews.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"reviews input is missing required column(s): {missing}")

    lookup = reviews.assign(review_id=reviews["review_id"].astype(str)).drop_duplicates("review_id")
    text_by_id = lookup.set_index("review_id")["review_text"].to_dict()
    missing_ids = [review_id for review_id in ids if review_id not in text_by_id]
    if missing_ids:
        raise ValueError(f"reviews input is missing {len(missing_ids)} embedded review ID(s)")
    return [str(text_by_id[review_id]) for review_id in ids]


def _write_theme_outputs(
    cluster_frame: pd.DataFrame,
    diagnostics: dict[str, object],
    representatives: pd.DataFrame,
    keywords: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    resolved_output_dir = _resolve_theme_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = resolved_output_dir / "cluster_assignments.parquet"
    diagnostics_path = resolved_output_dir / "cluster_diagnostics.json"
    representatives_path = resolved_output_dir / "representative_reviews.csv"
    keywords_path = resolved_output_dir / "cluster_keywords.csv"

    cluster_frame.to_parquet(assignments_path, index=False)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    representatives.to_csv(representatives_path, index=False)
    keywords.to_csv(keywords_path, index=False)

    return {
        "Cluster assignments": assignments_path,
        "Cluster diagnostics": diagnostics_path,
        "Representative reviews": representatives_path,
        "Cluster keywords": keywords_path,
    }


def _resolve_theme_output_dir(output_dir: Path) -> Path:
    path = output_dir.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    resolved_path = path.resolve(strict=False)
    processed_dir = (Path.cwd() / "data" / "processed").resolve(strict=False)
    try:
        resolved_path.relative_to(processed_dir)
    except ValueError as exc:
        raise ValueError("theme output must be written under data/processed/") from exc
    return resolved_path


if __name__ == "__main__":
    main()
