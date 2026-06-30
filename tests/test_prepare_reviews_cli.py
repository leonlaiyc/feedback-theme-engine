import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_reviews.py"
SPEC = importlib.util.spec_from_file_location("prepare_reviews", SCRIPT_PATH)
assert SPEC is not None
prepare_reviews = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_reviews)


def test_prepare_reviews_cli_does_not_trust_remote_code_by_default() -> None:
    args = prepare_reviews.parse_args([])

    assert args.trust_remote_code is False


def test_prepare_reviews_cli_accepts_trust_remote_code_flag() -> None:
    args = prepare_reviews.parse_args(
        [
            "--category",
            "raw_review_All_Beauty",
            "--sample-size",
            "10",
            "--seed",
            "7",
            "--output",
            "data/processed/toy.parquet",
            "--trust-remote-code",
        ]
    )

    assert args.category == "raw_review_All_Beauty"
    assert args.sample_size == 10
    assert args.seed == 7
    assert args.output == Path("data/processed/toy.parquet")
    assert args.trust_remote_code is True
