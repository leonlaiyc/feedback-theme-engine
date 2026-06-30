"""Run the Phase 1 review ingestion workflow locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feedback_theme_engine.data_ingestion import (
    DEFAULT_CATEGORY,
    DEFAULT_OUTPUT_PATH,
    IngestionConfig,
    TRUST_REMOTE_CODE_NOTE,
    prepare_reviews_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local Amazon Reviews 2023 sample.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--split", default="full")
    parser.add_argument("--min-review-text-length", type=int, default=1)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the dataset to already exist in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow Hugging Face to execute dataset loading code. "
            "Use only after reviewing and trusting the dataset source."
        ),
    )
    parser.epilog = TRUST_REMOTE_CODE_NOTE
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = IngestionConfig(
        category=args.category,
        sample_size=args.sample_size,
        seed=args.seed,
        output_path=args.output,
        split=args.split,
        min_review_text_length=args.min_review_text_length,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    _reviews, summary = prepare_reviews_dataset(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print(f"Saved processed sample to {config.output_path}")


if __name__ == "__main__":
    main()
