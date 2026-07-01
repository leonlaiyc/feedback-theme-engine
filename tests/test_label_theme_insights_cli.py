import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "label_theme_insights.py"
SPEC = importlib.util.spec_from_file_location("label_theme_insights", SCRIPT_PATH)
assert SPEC is not None
label_theme_insights = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(label_theme_insights)


def test_label_theme_insights_cli_defaults_to_ignored_paths() -> None:
    args = label_theme_insights.parse_args([])

    assert args.signals_input == Path("data/processed/themes/theme_signals.csv")
    assert args.representatives_input == Path("data/processed/themes/representative_reviews.csv")
    assert args.keywords_input == Path("data/processed/themes/cluster_keywords.csv")
    assert args.output_dir == Path("data/processed/themes")
    assert args.provider == "mock"


def test_label_theme_insights_cli_accepts_provider_settings() -> None:
    args = label_theme_insights.parse_args(
        [
            "--provider",
            "mock",
            "--model-name",
            "offline-test",
            "--max-representative-reviews",
            "3",
            "--max-keywords",
            "4",
        ]
    )

    assert args.model_name == "offline-test"
    assert args.max_representative_reviews == 3
    assert args.max_keywords == 4


def test_write_insight_outputs_creates_temp_json_and_markdown(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    records = [
        {
            "cluster_id": 1,
            "theme_label": "Pump Failure",
            "theme_type": "customer_pain_point",
            "summary": "Customers mention pump issues.",
            "evidence": ["The pump stopped working."],
            "suggested_action": "Review examples.",
            "confidence": "medium",
            "caution": "Exploratory.",
            "keywords": ["pump"],
            "representative_reviews": [],
            "statistics": {"prevalence": 0.2},
        }
    ]

    paths = label_theme_insights._write_insight_outputs(
        records,
        Path("data/processed/themes"),
    )

    assert paths["Theme insights JSON"].exists()
    assert paths["Theme insights Markdown"].exists()
    assert json.loads(paths["Theme insights JSON"].read_text(encoding="utf-8"))[0]["cluster_id"] == 1


def test_insight_output_dir_must_be_under_processed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="data/processed"):
        label_theme_insights._resolve_insight_output_dir(tmp_path / "outside")


def test_read_table_reads_csv(tmp_path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"cluster_id": [1]}).to_csv(path, index=False)

    table = label_theme_insights._read_table(path)

    assert table["cluster_id"].tolist() == [1]
