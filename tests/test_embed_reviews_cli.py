import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "embed_reviews.py"
SPEC = importlib.util.spec_from_file_location("embed_reviews", SCRIPT_PATH)
assert SPEC is not None
embed_reviews = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(embed_reviews)


def test_embed_reviews_cli_defaults_to_ignored_paths() -> None:
    args = embed_reviews.parse_args([])

    assert args.input == Path("data/processed/reviews_sample.parquet")
    assert args.output_dir == Path("data/processed/embeddings")
    assert args.prefix == "reviews_sample"


def test_embed_reviews_cli_accepts_limit_for_smoke_test() -> None:
    args = embed_reviews.parse_args(
        [
            "--input",
            "data/processed/reviews_sample.parquet",
            "--text-column",
            "review_text",
            "--id-column",
            "review_id",
            "--model-name",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--batch-size",
            "8",
            "--output-dir",
            "data/processed/embeddings",
            "--prefix",
            "smoke",
            "--limit",
            "3",
        ]
    )

    assert args.batch_size == 8
    assert args.limit == 3
    assert args.prefix == "smoke"


def test_embed_reviews_script_rejects_unknown_arguments() -> None:
    with pytest.raises(SystemExit):
        embed_reviews.parse_args(["--unknown"])
