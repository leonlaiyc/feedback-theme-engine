"""Analyze exploratory statistical signals for discovered themes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feedback_theme_engine.statistical_signals import (  # noqa: E402
    StatisticalSignalConfig,
    build_statistical_signal_table,
)

DEFAULT_CLUSTER_INPUT = Path("data/processed/themes/cluster_assignments.parquet")
DEFAULT_SIGNAL_OUTPUT_DIR = Path("data/processed/themes")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze exploratory rating signals for discovered themes.",
    )
    parser.add_argument("--cluster-input", type=Path, default=DEFAULT_CLUSTER_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SIGNAL_OUTPUT_DIR)
    parser.add_argument("--rating-column", default="rating")
    parser.add_argument("--cluster-column", default="cluster_id")
    parser.add_argument("--min-theme-size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--p-adjust-method", default="fdr_bh")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = StatisticalSignalConfig(
        alpha=args.alpha,
        min_theme_size=args.min_theme_size,
        p_adjust_method=args.p_adjust_method,
        rating_column=args.rating_column,
        cluster_column=args.cluster_column,
    )

    cluster_frame = _read_cluster_frame(args.cluster_input)
    _validate_rating_present(cluster_frame, config.rating_column)
    signal_table = build_statistical_signal_table(cluster_frame, config)
    output_paths = _write_signal_outputs(signal_table, args.output_dir)

    insufficient_count = int(signal_table["insufficient_sample"].sum()) if not signal_table.empty else 0
    print(f"Rows analyzed: {len(cluster_frame)}")
    print(f"Non-noise themes tested: {len(signal_table)}")
    print(f"Insufficient-sample themes: {insufficient_count}")
    print(f"P-value adjustment method: {config.p_adjust_method}")
    for label, path in output_paths.items():
        print(f"{label}: {path}")
    print("Generated signal outputs are ignored local artifacts and should not be committed.")


def _read_cluster_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("cluster input must end in .parquet, .jsonl, .json, or .csv")


def _validate_rating_present(cluster_frame: pd.DataFrame, rating_column: str) -> None:
    if rating_column not in cluster_frame.columns:
        raise ValueError(f"cluster input must include rating column: {rating_column}")


def _write_signal_outputs(signal_table: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    resolved_output_dir = _resolve_signal_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolved_output_dir / "theme_signals.csv"
    json_path = resolved_output_dir / "theme_signals.json"
    signal_table.to_csv(csv_path, index=False)
    signal_table.to_json(json_path, orient="records", indent=2)
    return {"Theme signals CSV": csv_path, "Theme signals JSON": json_path}


def _resolve_signal_output_dir(output_dir: Path) -> Path:
    path = output_dir.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    resolved_path = path.resolve(strict=False)
    processed_dir = (Path.cwd() / "data" / "processed").resolve(strict=False)
    try:
        resolved_path.relative_to(processed_dir)
    except ValueError as exc:
        raise ValueError("signal output must be written under data/processed/") from exc
    return resolved_path


if __name__ == "__main__":
    main()
