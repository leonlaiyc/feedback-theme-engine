"""Create evidence-grounded theme insight drafts with a mock LLM provider."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feedback_theme_engine.insight_reporting import (  # noqa: E402
    build_insight_report,
    records_to_json,
    render_markdown_insight_report,
)
from feedback_theme_engine.llm_labeling import ThemeLabelingConfig  # noqa: E402

DEFAULT_SIGNALS_INPUT = Path("data/processed/themes/theme_signals.csv")
DEFAULT_REPRESENTATIVES_INPUT = Path("data/processed/themes/representative_reviews.csv")
DEFAULT_KEYWORDS_INPUT = Path("data/processed/themes/cluster_keywords.csv")
DEFAULT_INSIGHT_OUTPUT_DIR = Path("data/processed/themes")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draft evidence-grounded theme insights from local artifacts.",
    )
    parser.add_argument("--signals-input", type=Path, default=DEFAULT_SIGNALS_INPUT)
    parser.add_argument("--representatives-input", type=Path, default=DEFAULT_REPRESENTATIVES_INPUT)
    parser.add_argument("--keywords-input", type=Path, default=DEFAULT_KEYWORDS_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INSIGHT_OUTPUT_DIR)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-representative-reviews", type=int, default=5)
    parser.add_argument("--max-keywords", type=int, default=10)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = ThemeLabelingConfig(
        provider=args.provider,
        model_name=args.model_name,
        max_representative_reviews=args.max_representative_reviews,
        max_keywords=args.max_keywords,
    )

    signal_frame = _read_table(args.signals_input)
    representatives = _read_table(args.representatives_input)
    keywords = _read_table(args.keywords_input)
    insight_records = build_insight_report(signal_frame, representatives, keywords, config)
    output_paths = _write_insight_outputs(insight_records, args.output_dir)

    print(f"Themes processed: {len(insight_records)}")
    print(f"Provider used: {config.provider}")
    for label, path in output_paths.items():
        print(f"{label}: {path}")
    print("Generated insight outputs are ignored local artifacts and should not be committed.")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("input tables must end in .csv, .json, .jsonl, or .parquet")


def _write_insight_outputs(
    insight_records: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    resolved_output_dir = _resolve_insight_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    json_path = resolved_output_dir / "theme_insights.json"
    markdown_path = resolved_output_dir / "theme_insights.md"
    json_path.write_text(records_to_json(insight_records), encoding="utf-8")
    markdown_path.write_text(render_markdown_insight_report(insight_records), encoding="utf-8")
    return {"Theme insights JSON": json_path, "Theme insights Markdown": markdown_path}


def _resolve_insight_output_dir(output_dir: Path) -> Path:
    path = output_dir.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    resolved_path = path.resolve(strict=False)
    processed_dir = (Path.cwd() / "data" / "processed").resolve(strict=False)
    try:
        resolved_path.relative_to(processed_dir)
    except ValueError as exc:
        raise ValueError("insight output must be written under data/processed/") from exc
    return resolved_path


if __name__ == "__main__":
    main()
