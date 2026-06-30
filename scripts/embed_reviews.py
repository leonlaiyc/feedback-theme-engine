"""Generate local sentence embeddings for a prepared review sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feedback_theme_engine.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_OUTPUT_DIR,
    EmbeddingConfig,
    embed_reviews_dataframe,
    load_embedding_model,
    save_embeddings,
)

DEFAULT_INPUT_PATH = Path("data/processed/reviews_sample.parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local sentence embeddings for prepared reviews.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--text-column", default="review_text")
    parser.add_argument("--id-column", default="review_id")
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EMBEDDING_OUTPUT_DIR)
    parser.add_argument("--prefix", default="reviews_sample")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a small local smoke test.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = EmbeddingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        text_column=args.text_column,
        id_column=args.id_column,
        output_dir=args.output_dir,
    )

    reviews = _read_reviews(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero when provided")
        reviews = reviews.head(args.limit).copy()

    model = load_embedding_model(config)
    ids, embeddings = embed_reviews_dataframe(reviews, model, config)
    ids_path, embeddings_path = save_embeddings(
        ids,
        embeddings,
        config.output_dir,
        args.prefix,
    )

    print(f"Input path: {args.input}")
    print(f"Model name: {config.model_name}")
    print(f"Rows embedded: {len(ids)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"ID output: {ids_path}")
    print(f"Embedding output: {embeddings_path}")
    print("Generated embeddings are ignored local artifacts and should not be committed.")


def _read_reviews(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(input_path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError("input path must end in .parquet, .jsonl, .json, or .csv")


if __name__ == "__main__":
    main()
