import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_theme_signals.py"
SPEC = importlib.util.spec_from_file_location("analyze_theme_signals", SCRIPT_PATH)
assert SPEC is not None
analyze_theme_signals = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyze_theme_signals)


def test_analyze_theme_signals_cli_defaults_to_ignored_paths() -> None:
    args = analyze_theme_signals.parse_args([])

    assert args.cluster_input == Path("data/processed/themes/cluster_assignments.parquet")
    assert args.output_dir == Path("data/processed/themes")
    assert args.rating_column == "rating"
    assert args.cluster_column == "cluster_id"


def test_analyze_theme_signals_cli_accepts_signal_parameters() -> None:
    args = analyze_theme_signals.parse_args(
        [
            "--cluster-input",
            "data/processed/themes/cluster_assignments.parquet",
            "--output-dir",
            "data/processed/themes",
            "--rating-column",
            "stars",
            "--cluster-column",
            "theme",
            "--min-theme-size",
            "5",
            "--alpha",
            "0.1",
            "--p-adjust-method",
            "fdr_bh",
        ]
    )

    assert args.rating_column == "stars"
    assert args.cluster_column == "theme"
    assert args.min_theme_size == 5
    assert args.alpha == 0.1


def test_validate_rating_present_raises_for_missing_rating() -> None:
    with pytest.raises(ValueError, match="rating"):
        analyze_theme_signals._validate_rating_present(pd.DataFrame({"cluster_id": [0]}), "rating")


def test_write_signal_outputs_creates_temp_files_under_processed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    signal_table = pd.DataFrame(
        {
            "cluster_id": [0],
            "prevalence": [0.5],
            "interpretation": ["no_clear_rating_signal"],
        }
    )

    paths = analyze_theme_signals._write_signal_outputs(
        signal_table,
        Path("data/processed/themes"),
    )

    assert paths["Theme signals CSV"].exists()
    assert paths["Theme signals JSON"].exists()


def test_signal_output_dir_must_be_under_processed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="data/processed"):
        analyze_theme_signals._resolve_signal_output_dir(tmp_path / "outside")
